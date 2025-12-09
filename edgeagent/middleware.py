"""
EdgeAgent Middleware

LangChain MCP Adapters를 확장하여 location-aware tool routing을 제공

핵심 설계:
- LLM은 "무엇을 할지"만 결정 (tool 이름에 location suffix 없음)
- Middleware/Scheduler가 "어디서 실행할지" 결정
- ProxyTool 패턴으로 호출 시점에 적절한 backend tool로 routing
- 통합 메트릭 수집 지원
"""

from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager, AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .types import Location
from .registry import ToolRegistry
from .scheduler import StaticScheduler
from .proxy_tool import LocationAwareProxyTool
from .metrics import MetricsCollector, MetricsConfig


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
    ):
        """
        Args:
            config_path: YAML 설정 파일 경로
            metrics_config: 메트릭 수집 설정 (None이면 기본값 사용)
            collect_metrics: 메트릭 수집 활성화 여부 (기본: True)
        """
        self.config_path = Path(config_path)

        # Registry 및 Scheduler 초기화
        self.registry = ToolRegistry.from_yaml(config_path)
        self.scheduler = StaticScheduler(config_path, self.registry)

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
            self.metrics_collector = MetricsCollector(config)
        else:
            self.metrics_collector = None

        # Session 관리를 위한 AsyncExitStack
        self._exit_stack: Optional[AsyncExitStack] = None
        self._active_sessions: dict[str, any] = {}

    async def __aenter__(self):
        """Context manager entry - session lifecycle 관리"""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - 모든 session 정리"""
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
        self._active_sessions.clear()

    async def initialize(self):
        """
        모든 location의 MCP client 초기화

        각 location별로 MultiServerMCPClient를 생성하고 연결합니다.
        """
        # Location별 server 설정 그룹화
        servers_by_location: dict[Location, dict] = {
            "DEVICE": {},
            "EDGE": {},
            "CLOUD": {},
        }

        for tool_name in self.registry.list_tools():
            tool_config = self.registry.get_tool(tool_name)
            if not tool_config:
                continue

            for location in tool_config.available_locations():
                endpoint = tool_config.get_endpoint(location)
                if not endpoint:
                    continue

                # Server 이름: tool_name_LOCATION
                server_name = f"{tool_name}_{location}"

                # MCP config 생성
                mcp_config = endpoint.to_mcp_config()
                servers_by_location[location][server_name] = mcp_config

        # Location별 MultiServerMCPClient 생성
        for location in ["DEVICE", "EDGE", "CLOUD"]:
            if servers_by_location[location]:
                self.clients[location] = MultiServerMCPClient(
                    servers_by_location[location]
                )

    async def get_tools(self) -> list:
        """
        LangChain-compatible tools 로드

        ProxyTool 패턴을 사용하여:
        - LLM에는 location suffix 없는 tool 노출 (예: read_file)
        - 각 proxy tool이 내부적으로 여러 location의 backend tool 보유
        - 호출 시점에 Scheduler가 location 결정

        Context manager 내에서 호출해야 합니다.

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

        for tool_name in self.registry.list_tools():
            tool_config = self.registry.get_tool(tool_name)
            if not tool_config:
                continue

            # 각 location별로 backend tool 로드
            for location in tool_config.available_locations():
                # 해당 location의 client 가져오기
                client = self.clients.get(location)
                if not client:
                    print(
                        f"Warning: No client for location '{location}' "
                        f"when loading tool '{tool_name}'"
                    )
                    continue

                # Server 이름: tool_name_LOCATION
                server_name = f"{tool_name}_{location}"

                # MCP session을 통해 tools 로드
                try:
                    session = await self._exit_stack.enter_async_context(
                        client.session(server_name)
                    )
                    self._active_sessions[server_name] = session

                    mcp_tools = await load_mcp_tools(session)

                    # 각 MCP tool을 backend_tools_map에 저장
                    for mcp_tool in mcp_tools:
                        if mcp_tool.name not in backend_tools_map:
                            backend_tools_map[mcp_tool.name] = {
                                "_first_tool": mcp_tool,
                                "_parent_tool": tool_name,
                            }
                        backend_tools_map[mcp_tool.name][location] = mcp_tool

                except Exception as e:
                    print(
                        f"Error loading tools from '{server_name}' "
                        f"at {location}: {e}"
                    )
                    continue

            # Tool placement 저장 (기본 location)
            default_location = self.scheduler.get_location(tool_name)
            self.tool_placement[tool_name] = default_location

        # ProxyTool 생성
        proxy_tools = []
        for mcp_tool_name, tool_data in backend_tools_map.items():
            first_tool = tool_data.pop("_first_tool")
            parent_tool = tool_data.pop("_parent_tool")

            # location → backend tool 매핑만 남김
            location_map = tool_data

            proxy_tool = LocationAwareProxyTool(
                name=mcp_tool_name,  # "read_file" (suffix 없음!)
                description=first_tool.description,
                backend_tools=location_map,
                scheduler=self.scheduler,
                parent_tool_name=parent_tool,
                execution_trace=[],  # 임시 빈 리스트로 초기화
                metrics_collector=self.metrics_collector,  # 메트릭 수집기 전달
            )
            # execution_trace 리스트 참조 공유 (Pydantic model 생성 후)
            object.__setattr__(proxy_tool, 'execution_trace', self.execution_trace)
            # args_schema는 BaseTool에서 별도로 처리 필요
            # first_tool에서 직접 복사
            if hasattr(first_tool, 'args_schema') and first_tool.args_schema:
                proxy_tool.args_schema = first_tool.args_schema
            proxy_tools.append(proxy_tool)

        return proxy_tools

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

        # Client 가져오기
        client = self.clients.get(location)
        if not client:
            raise ValueError(f"No client available for location: {location}")

        # Server 이름
        server_name = f"{tool_name}_{location}"

        # Session 제공
        async with client.session(server_name) as session:
            # Trace 기록
            self.execution_trace.append({
                "tool": tool_name,
                "location": location,
                "server": server_name,
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
