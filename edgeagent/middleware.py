"""
EdgeAgent Middleware

LangChain MCP Adapters를 확장하여 location-aware tool routing을 제공

핵심 설계:
- LLM은 "무엇을 할지"만 결정 (tool 이름에 location suffix 없음)
- Middleware/Scheduler가 "어디서 실행할지" 결정
- ProxyTool 패턴으로 호출 시점에 적절한 backend tool로 routing
- 통합 메트릭 수집 지원

실행 모드:
- agent 모드: get_tools() -> ProxyTool 반환, 스케줄러가 런타임에 location 결정
- with_metrics 모드: get_backend_tools(placement_map) -> MetricsWrappedTool 반환,
                     placement_map 기반으로 필요한 서버만 연결
"""

import asyncio
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager, AsyncExitStack

from pydantic import Field, ConfigDict
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .types import Location, ChainSchedulingResult, SchedulingResult
from .registry import ToolRegistry
from .scheduler import BaseScheduler, BruteForceChainScheduler
from .proxy_tool import LocationAwareProxyTool
from .metrics import MetricsCollector, MetricsConfig


class MetricsWrappedTool(BaseTool):
    """
    Backend tool을 감싸서 메트릭만 수집하는 가벼운 래퍼

    with_metrics 모드에서 사용:
    - schedule_chain()으로 placement가 이미 결정됨
    - ProxyTool의 스케줄링 로직 불필요
    - 메트릭 수집만 수행
    - Lazy 초기화: 도구 호출 시점에 해당 서버 세션만 생성
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Tool name")
    description: str = Field(default="", description="Tool description")

    # Lazy 초기화용 필드
    backend_tool: Optional[BaseTool] = Field(default=None, description="Actual backend tool (lazily loaded)")
    location: str = Field(description="Location where this tool is placed")
    parent_tool_name: str = Field(default="", description="Parent MCP server name")
    metrics_collector: Optional[Any] = Field(default=None)

    # Lazy 초기화를 위한 추가 필드
    client: Optional[Any] = Field(default=None, description="EdgeAgentMCPClient reference")
    server_location: str = Field(default="", description="Server-location identifier (e.g., 'summarize_EDGE')")
    _initialized: bool = False

    # Chain Scheduling에서 계산된 SchedulingResult (score, exec_cost 등 포함)
    scheduling_result: Optional[Any] = Field(default=None, description="SchedulingResult for this tool")

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._arun(*args, run_manager=run_manager, **kwargs)
        )

    async def _ensure_initialized(self):
        """Lazy 초기화: 첫 호출 시 해당 서버 세션만 생성"""
        if self._initialized:
            return

        if self.backend_tool is not None:
            # 이미 backend_tool이 설정됨 (기존 방식)
            self._initialized = True
            return

        if self.client is None or not self.server_location:
            raise RuntimeError(
                f"MetricsWrappedTool({self.name}) requires client and server_location for lazy init"
            )

        # 클라이언트를 통해 세션 생성 및 backend tool 로드
        self.backend_tool = await self.client._lazy_load_backend_tool(
            tool_name=self.name,
            server_location=self.server_location,
        )
        self._initialized = True

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """메트릭 수집과 함께 backend tool 호출 (스케줄링 없음, lazy 초기화)"""
        # Lazy 초기화
        await self._ensure_initialized()

        if self.metrics_collector is not None:
            async with self.metrics_collector.start_call(
                tool_name=self.name,
                parent_tool_name=self.parent_tool_name,
                location=self.location,
                args=kwargs,
            ) as ctx:
                # SchedulingResult가 있으면 원래 계산된 정보 사용
                if self.scheduling_result is not None:
                    sr = self.scheduling_result
                    # constraints 변환
                    constraints_list = []
                    if hasattr(sr, 'constraints') and sr.constraints:
                        if sr.constraints.requires_gpu:
                            constraints_list.append("requires_gpu")
                        if sr.constraints.privacy_sensitive:
                            constraints_list.append("privacy_sensitive")
                    ctx.add_scheduling_info(
                        reason=sr.reason,
                        constraints=constraints_list,
                        available=sr.available_locations or [self.location],
                        decision_time_ns=sr.decision_time_ns or 0,
                        score=sr.score or 0.0,
                        exec_cost=sr.exec_cost or 0.0,
                        trans_cost=sr.trans_cost or 0.0,
                        fixed=sr.fixed if sr.fixed is not None else True,
                    )
                else:
                    # fallback: 기본값 사용
                    ctx.add_scheduling_info(
                        reason="placement_map_static",
                        constraints=[],
                        available=[self.location],
                        decision_time_ns=0,
                        score=0.0,
                        exec_cost=0.0,
                        trans_cost=0.0,
                        fixed=True,
                    )
                try:
                    result = await self.backend_tool.ainvoke(kwargs)
                    ctx.set_result(result)
                    ctx.set_actual_location(self.location, fallback=False)
                    return result
                except Exception as e:
                    ctx.set_error(e)
                    raise
        else:
            return await self.backend_tool.ainvoke(kwargs)

    def __repr__(self) -> str:
        return f"MetricsWrappedTool({self.name}, location={self.location}, initialized={self._initialized})"


class EdgeAgentMCPClient:
    """
    Location-aware MCP Client

    langchain-mcp-adapters의 MultiServerMCPClient를 확장하여:
    1. Tool별로 여러 endpoint (DEVICE/EDGE/CLOUD) 관리
    2. Scheduler를 통해 최적 location 선택
    3. 선택된 location의 endpoint로 자동 라우팅

    사용 패턴:
        async with EdgeAgentMCPClient(config_path) as client:
            tools = await client.get_tools()
            # use tools...
    """

    def __init__(
        self,
        config_path: str | Path,
        metrics_config: Optional[MetricsConfig] = None,
        collect_metrics: bool = True,
        filter_tools: Optional[list[str]] = None,
        exclude_tools: Optional[set[str]] = None,
        filter_location: Optional[Location] = None,
        scheduler: Optional[BaseScheduler] = None,
        system_config_path: Optional[str | Path] = None,
        scenario_name: str = "unknown",
    ):
        """
        Args:
            config_path: YAML 설정 파일 경로
            metrics_config: 메트릭 수집 설정 (None이면 기본값 사용)
            collect_metrics: 메트릭 수집 활성화 여부 (기본: True)
            filter_tools: 연결할 tool 목록 (None이면 모든 tool 연결)
            exclude_tools: 제외할 tool 목록 (예: {"directory_tree"})
            filter_location: 특정 location의 endpoint만 사용 (None이면 모든 location)
            scheduler: 외부에서 생성한 Scheduler (None이면 BruteForceChainScheduler 사용)
            system_config_path: System 설정 YAML 경로 (BruteForceChainScheduler 사용 시)
            scenario_name: 시나리오 이름 (metrics에 기록)
        """
        self.config_path = Path(config_path)
        self.filter_tools = filter_tools
        self.exclude_tools = exclude_tools or set()
        self.filter_location = filter_location
        self.scenario_name = scenario_name

        # Registry 및 Scheduler 초기화
        self.registry = ToolRegistry.from_yaml(config_path)
        if scheduler:
            self.scheduler = scheduler
        else:
            sys_config = system_config_path or (Path(config_path).parent / "system.yaml")
            self.scheduler = BruteForceChainScheduler(
                config_path, sys_config, self.registry
            )

        # Scheduler type 추출
        self.scheduler_type = self._get_scheduler_type()

        # Location별 MCP client 저장
        self.clients: dict[Location, MultiServerMCPClient] = {}

        # Tool → Location 매핑 저장 (스케줄링 결과)
        self.tool_placement: dict[str, Location] = {}

        # 실행 trace 저장 (backward compatibility)
        self.execution_trace: list[dict] = []

        # 통합 메트릭 수집기 초기화
        self._collect_metrics = collect_metrics
        if collect_metrics:
            config = metrics_config or MetricsConfig()
            self.metrics_collector = MetricsCollector(
                config,
                scenario_name=scenario_name,
                scheduler_type=self.scheduler_type,
            )
        else:
            self.metrics_collector = None

        # Session 관리를 위한 AsyncExitStack
        self._exit_stack: Optional[AsyncExitStack] = None
        self._active_sessions: dict[str, any] = {}

        # Lazy loading을 위한 connection queue (메인 task에서 세션 생성)
        self._connection_queue: Optional[asyncio.Queue] = None
        self._processor_task: Optional[asyncio.Task] = None

        # Chain Scheduling 결과 저장 (스크립트/SubAgent 모드에서 사용)
        self.chain_scheduling_result: Optional[ChainSchedulingResult] = None

    def _get_scheduler_type(self) -> str:
        """Scheduler 타입 문자열 추출"""
        scheduler_name = type(self.scheduler).__name__
        type_map = {
            "BruteForceChainScheduler": "brute_force",
            "StaticScheduler": "static",
            "HeuristicScheduler": "heuristic",
            "AllDeviceScheduler": "all_device",
            "AllEdgeScheduler": "all_edge",
            "AllCloudScheduler": "all_cloud",
        }
        return type_map.get(scheduler_name, scheduler_name.lower())

    async def __aenter__(self):
        """Context manager entry - DEVICE 초기화, 나머지는 스케줄러 선택 시 연결"""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Lazy loading을 위한 connection processor 시작 (메인 task에서 실행)
        self._connection_queue = asyncio.Queue()
        self._processor_task = asyncio.create_task(self._connection_processor())

        # get_tools()에서 DEVICE 연결, 이후 스케줄러 선택에 따라 EDGE/CLOUD 연결
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - 모든 session 정리"""
        # Connection processor 종료
        if self._connection_queue and self._processor_task:
            await self._connection_queue.put(None)  # 종료 신호
            await self._processor_task  # processor 완료 대기
            self._processor_task = None
            self._connection_queue = None

        # 모든 세션이 메인 task에서 열렸으므로 정상 cleanup
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
        self._active_sessions.clear()

    async def _connection_processor(self):
        """
        Lazy loading 연결 요청을 처리하는 background processor

        Processor task가 자신만의 exit_stack을 소유하여
        자신이 열은 세션을 자신이 정리 (cancel scope 문제 해결).
        """
        # Processor task 전용 exit_stack (이 task에서 열고 이 task에서 닫음)
        lazy_exit_stack = AsyncExitStack()
        await lazy_exit_stack.__aenter__()

        try:
            while True:
                request = await self._connection_queue.get()

                # 종료 신호
                if request is None:
                    break

                server_session_name, tool_config, location, response_queue = request

                try:
                    # 이미 연결되어 있으면 기존 세션 사용
                    if server_session_name in self._active_sessions:
                        session = self._active_sessions[server_session_name]
                        mcp_tools = await load_mcp_tools(session)
                        await response_queue.put({
                            "success": True,
                            "tools": {t.name: t for t in mcp_tools}
                        })
                        continue

                    # 새 연결 생성
                    endpoint = tool_config.get_endpoint(location)
                    if not endpoint:
                        await response_queue.put({
                            "success": False,
                            "error": ValueError(f"No endpoint for location '{location}'")
                        })
                        continue

                    # 클라이언트 생성 또는 재사용
                    if server_session_name not in self.clients:
                        mcp_config = endpoint.to_mcp_config()
                        self.clients[server_session_name] = MultiServerMCPClient(
                            {server_session_name: mcp_config}
                        )

                    # Processor task의 exit_stack에 등록 (같은 task에서 열고 닫음)
                    client = self.clients[server_session_name]
                    session = await lazy_exit_stack.enter_async_context(
                        client.session(server_session_name)
                    )
                    self._active_sessions[server_session_name] = session

                    # tool 로드하여 반환
                    mcp_tools = await load_mcp_tools(session)
                    await response_queue.put({
                        "success": True,
                        "tools": {t.name: t for t in mcp_tools}
                    })

                except Exception as e:
                    await response_queue.put({
                        "success": False,
                        "error": e
                    })
        finally:
            # Processor task가 자신이 열은 세션을 직접 정리
            await lazy_exit_stack.__aexit__(None, None, None)

    async def initialize(self):
        """
        MCP client 초기화

        filter_tools와 filter_location이 설정된 경우 해당 tool/location만 연결합니다.

        클라이언트 생성 전략:
        - filter_location이 설정된 경우 (EDGE/CLOUD SubAgent):
          Tool별로 개별 MultiServerMCPClient 생성 (동적 연결 지원)
        - filter_location이 없는 경우 (DEVICE 또는 Orchestrator):
          Location별로 MultiServerMCPClient 그룹화 (기존 방식)
        """
        # 연결할 tool 목록 결정
        tools_to_connect = self.filter_tools or self.registry.list_tools()

        if self.filter_location:
            # EDGE/CLOUD SubAgent: Tool별 개별 클라이언트 생성
            # 각 tool이 독립적인 MCP 서버에 연결되므로 개별 클라이언트 필요
            await self._initialize_per_tool_clients(tools_to_connect)
        else:
            # DEVICE/Orchestrator: Location별 클라이언트 그룹화
            # 여러 tool이 같은 location에서 실행되므로 그룹화 효율적
            await self._initialize_per_location_clients(tools_to_connect)

    async def _initialize_per_tool_clients(self, tools_to_connect: list[str]):
        """
        Tool별 개별 MultiServerMCPClient 생성

        EDGE/CLOUD SubAgent에서 사용. 각 tool이 독립적인 Knative 서비스로
        배포되어 있으므로 개별 클라이언트가 필요합니다.

        주의: 개별 tool 이름이 전달되어도 부모 MCP 서버 단위로 클라이언트 생성.
        get_tools()에서 서버 이름으로 클라이언트를 조회하므로 일관성 유지.
        """
        # 이미 생성된 서버 추적 (중복 방지)
        created_servers: set[str] = set()

        for tool_name in tools_to_connect:
            if self.filter_tools and tool_name not in self.filter_tools:
                continue

            # 개별 tool이면 부모 MCP 서버 이름으로 변환
            parent_server = self.registry.get_server_for_tool(tool_name)
            mcp_server_name = parent_server if parent_server else tool_name

            # 이미 생성된 서버면 스킵
            server_key = f"{mcp_server_name}_{self.filter_location}"
            if server_key in created_servers:
                continue

            tool_config = self.registry.get_tool(mcp_server_name)
            if not tool_config:
                continue

            # filter_location에 해당하는 endpoint만 확인
            if self.filter_location not in tool_config.available_locations():
                continue

            endpoint = tool_config.get_endpoint(self.filter_location)
            if not endpoint:
                continue

            # Server 이름: mcp_server_LOCATION (get_tools()와 일치)
            server_name = f"{mcp_server_name}_{self.filter_location}"

            # 서버별 클라이언트 생성
            mcp_config = endpoint.to_mcp_config()
            self.clients[server_name] = MultiServerMCPClient({
                server_name: mcp_config
            })
            created_servers.add(server_key)

    async def _initialize_per_location_clients(self, tools_to_connect: list[str]):
        """
        Location별 MultiServerMCPClient 그룹화

        DEVICE 환경 또는 Orchestrator에서 사용. 같은 location의 tool들을
        하나의 클라이언트로 그룹화하여 효율성 향상.

        주의: MCP 서버 단위로 클라이언트를 생성합니다.
        개별 tool이 아닌 서버만 연결하여 중복 세션 방지.
        스케줄러의 get_required_locations()에 포함된 location만 초기화.
        """
        # 스케줄러가 사용하는 location만 초기화
        required_locations = self.scheduler.get_required_locations()

        # Location별 server 설정 그룹화
        servers_by_location: dict[Location, dict] = {
            loc: {} for loc in required_locations
        }

        # MCP 서버만 순회 (개별 tool은 같은 서버 공유)
        for server_name in self.registry.list_servers():
            # filter_tools가 설정된 경우, 해당 서버의 tool이 포함되어 있는지 확인
            if self.filter_tools:
                # 서버에 속한 tool 중 하나라도 filter_tools에 있으면 연결
                server_tools = [
                    t for t in self.registry.list_individual_tools()
                    if self.registry.get_server_for_tool(t) == server_name
                ]
                if not any(t in self.filter_tools for t in server_tools):
                    # 서버 자체가 filter_tools에 있는지도 확인
                    if server_name not in self.filter_tools:
                        continue

            tool_config = self.registry.get_tool(server_name)
            if not tool_config:
                continue

            for location in tool_config.available_locations():
                # 스케줄러가 사용하지 않는 location은 스킵
                if location not in required_locations:
                    continue

                endpoint = tool_config.get_endpoint(location)
                if not endpoint:
                    continue

                # Server 이름: server_name_LOCATION
                mcp_server_name = f"{server_name}_{location}"

                # MCP config 생성
                mcp_config = endpoint.to_mcp_config()
                servers_by_location[location][mcp_server_name] = mcp_config

        # Location별 MultiServerMCPClient 생성
        for location in required_locations:
            if servers_by_location[location]:
                self.clients[location] = MultiServerMCPClient(
                    servers_by_location[location]
                )

    async def get_tools(self) -> list:
        """
        LangChain-compatible tools 로드

        DEVICE만 초기 연결하여 schema 획득, 이후 스케줄러 선택에 따라 EDGE/CLOUD 연결

        Returns:
            LangChain tool 목록 (ProxyTool 인스턴스들)
        """
        if self._exit_stack is None:
            raise RuntimeError(
                "EdgeAgentMCPClient must be used as async context manager. "
                "Use: async with EdgeAgentMCPClient(config) as client:"
            )

        # MCP tool name → {location → backend tool}
        backend_tools_map: dict[str, dict[str, any]] = {}

        # MCP 서버 단위로 세션 생성
        for mcp_server in self.registry.list_servers():
            # filter_tools 체크
            if self.filter_tools:
                server_tools = [
                    t for t in self.registry.list_individual_tools()
                    if self.registry.get_server_for_tool(t) == mcp_server
                ]
                if not any(t in self.filter_tools for t in server_tools):
                    if mcp_server not in self.filter_tools:
                        continue

            tool_config = self.registry.get_tool(mcp_server)
            if not tool_config:
                continue

            available_locations = tool_config.available_locations()

            # 초기 연결할 location 결정
            if self.filter_location:
                # SubAgent: filter_location 사용
                if self.filter_location in available_locations:
                    initial_location = self.filter_location
                else:
                    continue
            else:
                # Orchestrator: required_locations 중 사용 가능한 첫 번째 location
                required_locations = self.scheduler.get_required_locations()
                initial_location = None
                for loc in required_locations:
                    if loc in available_locations:
                        initial_location = loc
                        break

            if not initial_location:
                continue

            # 초기 location만 연결 (schema용)
            await self._connect_and_load_tools(
                mcp_server, initial_location, tool_config, backend_tools_map
            )

            # Tool placement 저장
            if self.filter_location:
                # SubAgent: filter_location으로 고정
                default_location = self.filter_location
            else:
                # Orchestrator: scheduler에서 결정
                default_location = self.scheduler.get_location(mcp_server)
            self.tool_placement[mcp_server] = default_location

        # ProxyTool 생성
        proxy_tools = []
        for mcp_tool_name, tool_data in backend_tools_map.items():
            # exclude_tools 체크
            if mcp_tool_name in self.exclude_tools:
                continue

            first_tool = tool_data.pop("_first_tool")
            parent_tool = tool_data.pop("_parent_tool")
            available_locs = tool_data.pop("_available_locations")

            # location → backend tool 매핑
            location_map = tool_data

            # SubAgent (filter_location 설정됨): scheduler 없이 고정 location 사용
            # Orchestrator: scheduler로 동적 location 결정
            effective_scheduler = None if self.filter_location else self.scheduler

            proxy_tool = LocationAwareProxyTool(
                name=mcp_tool_name,
                description=first_tool.description,
                backend_tools=location_map,
                scheduler=effective_scheduler,
                parent_tool_name=parent_tool,
                execution_trace=[],
                metrics_collector=self.metrics_collector,
                client=self,  # lazy loading용
                config_locations=available_locs,
            )
            object.__setattr__(proxy_tool, 'execution_trace', self.execution_trace)
            if hasattr(first_tool, 'args_schema') and first_tool.args_schema:
                proxy_tool.args_schema = first_tool.args_schema
            proxy_tools.append(proxy_tool)

        return proxy_tools

    async def get_backend_tools(
        self,
        placement_map: dict[str, str],
    ) -> dict[str, BaseTool]:
        """
        placement_map 기반으로 필요한 서버만 연결하고 MetricsWrappedTool 반환

        with_metrics 모드용:
        - schedule_chain() 결과인 placement_map을 받아서
        - 필요한 server-location 쌍만 연결
        - ProxyTool 대신 MetricsWrappedTool 반환 (스케줄링 없음)

        Args:
            placement_map: {tool_name -> location} 매핑

        Returns:
            {tool_name -> MetricsWrappedTool} 매핑
        """
        if self._exit_stack is None:
            raise RuntimeError(
                "EdgeAgentMCPClient must be used as async context manager"
            )

        # 1. placement_map에서 필요한 server-location 쌍 도출
        server_location_tools: dict[str, list[str]] = {}
        for tool_name, location in placement_map.items():
            parent_server = self.registry.get_server_for_tool(tool_name)
            if not parent_server:
                parent_server = tool_name
            server_location = f"{parent_server}_{location}"
            if server_location not in server_location_tools:
                server_location_tools[server_location] = []
            server_location_tools[server_location].append(tool_name)

        # 2. 필요한 server-location 쌍만 연결
        backend_tools: dict[str, BaseTool] = {}
        tool_metadata: dict[str, tuple[str, str]] = {}  # tool_name -> (parent_server, location)

        for server_location, tool_names in server_location_tools.items():
            parts = server_location.rsplit("_", 1)
            if len(parts) != 2:
                continue
            server_name, location = parts

            tool_config = self.registry.get_tool(server_name)
            if not tool_config:
                continue

            endpoint = tool_config.get_endpoint(location)
            if not endpoint:
                continue

            # 세션이 없으면 연결 - _connection_processor를 통해 세션 생성
            # (같은 task에서 세션 생성/정리하여 anyio cancel scope 문제 해결)
            if server_location not in self._active_sessions:
                # Queue를 통해 processor task에 연결 요청
                response_queue = asyncio.Queue()
                await self._connection_queue.put(
                    (server_location, tool_config, location, response_queue)
                )

                # Processor task의 응답 대기
                result = await response_queue.get()

                if not result["success"]:
                    raise result["error"]

                # tool 로드 완료 - 바로 backend_tools에 추가
                for mcp_tool in result["tools"].values():
                    if mcp_tool.name in placement_map and placement_map[mcp_tool.name] == location:
                        backend_tools[mcp_tool.name] = mcp_tool
                        tool_metadata[mcp_tool.name] = (server_name, location)
                continue

            # 이미 세션이 있으면 tool 로드
            session = self._active_sessions[server_location]
            mcp_tools = await load_mcp_tools(session)

            for mcp_tool in mcp_tools:
                # placement_map의 location과 현재 location이 일치하는 경우만 추가
                if mcp_tool.name in placement_map and placement_map[mcp_tool.name] == location:
                    backend_tools[mcp_tool.name] = mcp_tool
                    tool_metadata[mcp_tool.name] = (server_name, location)

        # 3. MetricsWrappedTool로 감싸서 반환
        wrapped_tools: dict[str, BaseTool] = {}
        for tool_name, backend_tool in backend_tools.items():
            parent_server, location = tool_metadata[tool_name]
            # chain_scheduling_result에서 해당 tool의 SchedulingResult 조회
            scheduling_result = self.get_scheduling_result_for_tool(tool_name)
            wrapped_tool = MetricsWrappedTool(
                name=tool_name,
                description=backend_tool.description,
                backend_tool=backend_tool,
                location=location,
                parent_tool_name=parent_server,
                metrics_collector=self.metrics_collector,
                scheduling_result=scheduling_result,
            )
            if hasattr(backend_tool, 'args_schema') and backend_tool.args_schema:
                wrapped_tool.args_schema = backend_tool.args_schema
            wrapped_tools[tool_name] = wrapped_tool

        return wrapped_tools

    async def _connect_and_load_tools(
        self,
        mcp_server: str,
        location: str,
        tool_config,
        backend_tools_map: dict,
    ):
        """특정 location의 MCP 서버 연결 및 tool 로드"""
        server_session_name = f"{mcp_server}_{location}"

        # 이미 세션이 있으면 스킵
        if server_session_name in self._active_sessions:
            return

        # MultiServerMCPClient 생성 - 키를 server_session_name으로 사용 (클라이언트 교체 방지)
        if server_session_name not in self.clients:
            endpoint = tool_config.get_endpoint(location)
            if not endpoint:
                return
            mcp_config = endpoint.to_mcp_config()
            self.clients[server_session_name] = MultiServerMCPClient({
                server_session_name: mcp_config
            })

        client = self.clients.get(server_session_name)
        if not client:
            return

        try:
            session = await self._exit_stack.enter_async_context(
                client.session(server_session_name)
            )
            self._active_sessions[server_session_name] = session

            mcp_tools = await load_mcp_tools(session)

            # backend_tools_map에 저장
            available_locs = tool_config.available_locations()
            for mcp_tool in mcp_tools:
                if mcp_tool.name not in backend_tools_map:
                    backend_tools_map[mcp_tool.name] = {
                        "_first_tool": mcp_tool,
                        "_parent_tool": mcp_server,
                        "_available_locations": available_locs,
                    }
                backend_tools_map[mcp_tool.name][location] = mcp_tool

        except Exception as e:
            print(f"Error loading tools from '{server_session_name}': {e}")

    async def ensure_location_connected(self, tool_name: str, location: str):
        """
        스케줄러 선택에 따라 해당 location의 서버 lazy 연결

        Queue를 통해 메인 task에서 연결을 처리하여 cancel scope 문제 해결.

        Returns:
            dict[str, BaseTool]: tool_name -> MCP tool 매핑
        """
        # parent server 찾기
        parent_server = self.registry.get_server_for_tool(tool_name)
        if not parent_server:
            parent_server = tool_name

        server_session_name = f"{parent_server}_{location}"

        # 이미 연결되어 있으면 세션에서 tool 로드하여 반환
        if server_session_name in self._active_sessions:
            session = self._active_sessions[server_session_name]
            mcp_tools = await load_mcp_tools(session)
            return {t.name: t for t in mcp_tools}

        tool_config = self.registry.get_tool(parent_server)
        if not tool_config:
            raise ValueError(f"Tool config not found for '{parent_server}'")

        if location not in tool_config.available_locations():
            raise ValueError(f"Location '{location}' not available for '{parent_server}'")

        # Queue를 통해 메인 task에 연결 요청
        response_queue = asyncio.Queue()
        await self._connection_queue.put(
            (server_session_name, tool_config, location, response_queue)
        )

        # 메인 task의 응답 대기
        result = await response_queue.get()

        if result["success"]:
            return result["tools"]
        else:
            raise result["error"]

    @asynccontextmanager
    async def session(self, tool_name: str):
        """
        특정 tool의 MCP session 제공

        스케줄러가 결정한 location의 session을 반환합니다.

        Args:
            tool_name: Tool 이름

        Yields:
            MCP ClientSession
        """
        # Location 결정
        location = self.scheduler.get_location(tool_name)

        # Parent server 찾기 (tool이 MCP 서버의 개별 tool일 수 있음)
        parent_server = self.registry.get_server_for_tool(tool_name)
        if not parent_server:
            parent_server = tool_name

        # Server session 이름
        server_session_name = f"{parent_server}_{location}"

        # Client 가져오기 - server_session_name 키로 조회
        client = self.clients.get(server_session_name)
        if not client:
            raise ValueError(f"No client available for session: {server_session_name}")

        # Session 제공
        async with client.session(server_session_name) as session:
            # Trace 기록
            self.execution_trace.append({
                "tool": tool_name,
                "location": location,
                "server": server_session_name,
            })

            yield session

    def get_tool_location(self, tool_name: str) -> Optional[Location]:
        """Tool이 배치된 location 조회"""
        return self.tool_placement.get(tool_name)

    def get_execution_trace(self) -> list[dict]:
        """실행 trace 조회 (backward compatibility)"""
        return self.execution_trace

    def get_metrics(self) -> Optional[MetricsCollector]:
        """통합 메트릭 수집기 반환"""
        return self.metrics_collector

    def set_chain_scheduling_result(self, result: ChainSchedulingResult):
        """
        Chain Scheduling 결과 설정

        스크립트 모드에서 schedule_chain() 호출 후 결과를 저장합니다.
        get_backend_tools()에서 각 tool의 SchedulingResult를 참조합니다.

        Args:
            result: ChainSchedulingResult (각 tool의 SchedulingResult 포함)
        """
        self.chain_scheduling_result = result

    def get_scheduling_result_for_tool(self, tool_name: str) -> Optional[SchedulingResult]:
        """
        특정 tool의 SchedulingResult 조회

        chain_scheduling_result에서 해당 tool의 스케줄링 결과를 찾습니다.

        Args:
            tool_name: Tool 이름

        Returns:
            SchedulingResult 또는 None
        """
        if self.chain_scheduling_result is None:
            return None
        for p in self.chain_scheduling_result.placements:
            if p.tool_name == tool_name:
                return p
        return None

    def reset_metrics(self):
        """메트릭 수집기 초기화 (새 세션 시작 시)"""
        if self.metrics_collector is not None:
            self.metrics_collector.reset()
        self.execution_trace.clear()

    def print_placement_summary(self):
        """Tool placement 요약 출력"""
        print("\n" + "=" * 80)
        print("Tool Placement Summary (Default Location)")
        print("=" * 80)
        print()
        print("┌─────────────────────────────┬──────────┬──────────┬────────────────────┐")
        print("│ Tool                        │ Default  │ Runtime  │ Available          │")
        print("├─────────────────────────────┼──────────┼──────────┼────────────────────┤")

        for tool_name in self.registry.list_tools():
            location = self.tool_placement.get(tool_name, "N/A")
            runtime = self.scheduler.select_runtime(tool_name, location) if location != "N/A" else "N/A"
            tool_config = self.registry.get_tool(tool_name)
            available = ",".join(tool_config.available_locations()) if tool_config else "N/A"
            print(f"│ {tool_name:27s} │ {location:8s} │ {runtime:8s} │ {available:18s} │")

        print("└─────────────────────────────┴──────────┴──────────┴────────────────────┘")
        print()

    def print_execution_trace(self):
        """실행 trace 출력"""
        if not self.execution_trace:
            print("No execution trace available.")
            return

        print("\n" + "=" * 80)
        print("Execution Trace")
        print("=" * 80)
        print()

        for i, trace in enumerate(self.execution_trace, 1):
            print(f"{i}. Tool: {trace['tool']}")
            print(f"   Location: {trace['location']}")
            print(f"   Server: {trace['server']}")
            print()

    def print_metrics_summary(self):
        """통합 메트릭 요약 출력"""
        if self.metrics_collector is not None:
            self.metrics_collector.print_summary()
        else:
            print("Metrics collection is disabled.")

    def __repr__(self) -> str:
        return (
            f"EdgeAgentMCPClient("
            f"tools={len(self.registry.list_tools())}, "
            f"clients={len(self.clients)})"
        )
