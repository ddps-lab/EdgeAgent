"""
Location-Aware Proxy Tool

LLM에 노출되는 Proxy tool - 호출 시 Scheduler가 location을 결정하고
적절한 backend tool로 routing

메트릭 수집 기능 통합:
- 자동 latency 측정
- 입출력 크기 계산
- 리소스 사용량 추적
"""

from typing import Any, Optional, TYPE_CHECKING

from pydantic import Field, ConfigDict

from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from .types import Location

if TYPE_CHECKING:
    from .scheduler import StaticScheduler
    from .metrics import MetricsCollector


class LocationAwareProxyTool(BaseTool):
    """
    LLM에 노출되는 Proxy tool

    LLM은 이 tool만 보고 "무엇을 할지" 결정합니다.
    실제 호출 시 Scheduler가 "어디서 실행할지" 결정하고
    적절한 backend tool로 routing합니다.

    예시:
        LLM은 "read_file" tool만 보고 파일 읽기 결정
        → Scheduler가 DEVICE/EDGE/CLOUD 중 선택
        → 해당 location의 실제 MCP tool 호출
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Tool 메타데이터
    name: str = Field(description="Tool name (without location suffix)")
    description: str = Field(default="", description="Tool description")

    # Backend tools (location -> actual MCP tool)
    backend_tools: dict[str, BaseTool] = Field(
        default_factory=dict,
        description="Map of location to actual backend tool",
    )

    # Scheduler reference
    scheduler: Any = Field(
        default=None,
        description="Scheduler for location decision",
    )

    # Parent tool name (for multi-tool MCP servers)
    parent_tool_name: str = Field(
        default="",
        description="Parent tool name in registry (e.g., 'filesystem')",
    )

    # Execution trace reference (shared with middleware)
    execution_trace: list = Field(
        default_factory=list,
        description="Shared execution trace list",
    )

    # Metrics collector (optional, for unified metrics collection)
    metrics_collector: Optional[Any] = Field(
        default=None,
        description="MetricsCollector instance for unified metrics",
    )

    # Client reference for lazy loading (EdgeAgentMCPClient)
    client: Optional[Any] = Field(
        default=None,
        description="EdgeAgentMCPClient for lazy location connection",
    )

    # All available locations from config (not just connected ones)
    # Note: property 'available_locations'와 이름 충돌을 피하기 위해 config_locations 사용
    config_locations: list[str] = Field(
        default_factory=list,
        description="All available locations from config",
    )

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Sync execution - delegates to async"""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._arun(*args, run_manager=run_manager, **kwargs)
        )

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Async execution with location-aware routing

        1. filter_location이 설정되어 있으면 scheduler 호출 스킵 (SubAgent 모드)
        2. 그렇지 않으면 Scheduler가 location 결정 (Agent 모드)
        3. 해당 location의 backend tool 선택
        4. Trace 기록 및 메트릭 수집
        5. 실제 tool 호출
        """
        fallback_occurred = False

        # 1. filter_location이 설정되어 있으면 scheduler 호출 스킵
        #    (SubAgent 모드: schedule_chain()으로 이미 결정됨)
        if self.client and self.client.filter_location:
            location = self.client.filter_location
            # chain_scheduling_result에서 해당 tool의 SchedulingResult 조회
            scheduling_result = self.client.get_scheduling_result_for_tool(self.name)
            if scheduling_result is None:
                # fallback: chain_scheduling_result가 없으면 기본값 생성
                scheduling_result = self._create_fixed_scheduling_result(location)
        else:
            # Agent 모드: Scheduler가 location 결정
            scheduling_result = self._get_location_with_reason(kwargs)
            location = scheduling_result.location

        # 2. 해당 location의 backend tool 선택 (lazy loading)
        backend_tool = self.backend_tools.get(location)

        if not backend_tool:
            # Lazy loading: client가 있고 해당 location이 config에 정의되어 있으면 연결
            if self.client and location in self.config_locations:
                try:
                    new_tools = await self.client.ensure_location_connected(
                        self.name, location
                    )
                    if new_tools and self.name in new_tools:
                        backend_tool = new_tools[self.name]
                        self.backend_tools[location] = backend_tool
                except Exception as e:
                    raise ValueError(
                        f"Lazy loading failed for '{self.name}' at {location}: {e}"
                    )

            if not backend_tool:
                raise ValueError(
                    f"No backend tool available for '{self.name}'. "
                    f"Scheduled location: {location}, "
                    f"Available: {list(self.backend_tools.keys())}"
                )

        # 3. 메트릭 수집과 함께 tool 호출
        if self.metrics_collector is not None:
            async with self.metrics_collector.start_call(
                tool_name=self.name,
                parent_tool_name=self.parent_tool_name,
                location=location,
                args=kwargs,
            ) as ctx:
                # Scheduling 정보 추가 (cost 포함)
                # SchedulingConstraints → list[str] 변환
                constraints_list = []
                if scheduling_result.constraints.requires_cloud_api:
                    constraints_list.append("requires_cloud_api")
                if scheduling_result.constraints.privacy_sensitive:
                    constraints_list.append("privacy_sensitive")
                ctx.add_scheduling_info(
                    reason=scheduling_result.reason,
                    constraints=constraints_list,
                    available=scheduling_result.available_locations,
                    decision_time_ns=scheduling_result.decision_time_ns,
                    score=scheduling_result.score,
                    exec_cost=scheduling_result.exec_cost,
                    trans_cost=scheduling_result.trans_cost,
                    fixed=scheduling_result.fixed,
                )
                try:
                    result = await backend_tool.ainvoke(kwargs)
                    ctx.set_result(result)
                    ctx.set_actual_location(location, fallback=fallback_occurred)
                    return result
                except Exception as e:
                    ctx.set_error(e)
                    raise
        else:
            # 기존 동작 유지 (메트릭 수집 없음)
            return await backend_tool.ainvoke(kwargs)

    def _get_location(self, args: dict[str, Any]) -> Location:
        """
        Location 결정

        Phase 1: Scheduler의 static mapping 사용
        Phase 2: args 기반 동적 결정 (향후)
        """
        if self.scheduler is None:
            # Scheduler 없으면 첫 번째 available location
            return list(self.backend_tools.keys())[0]

        # Scheduler에게 location 결정 요청
        # self.name (개별 tool 이름)을 사용하여 정확한 profile 조회
        return self.scheduler.get_location_for_call(
            tool_name=self.name,
            args=args,
        )

    def _get_location_with_reason(self, args: dict[str, Any]):
        """
        Location 결정 (상세 정보 포함)

        Returns:
            SchedulingResult: location과 결정 메타데이터
        """
        from .types import SchedulingResult

        if self.scheduler is None:
            # Scheduler 없으면 첫 번째 available location
            location = list(self.backend_tools.keys())[0]
            return SchedulingResult(
                tool_name=self.name,
                location=location,
                reason="no_scheduler_default",
                available_locations=list(self.backend_tools.keys()),
                decision_time_ns=0,
            )

        # Scheduler에게 location 결정 요청 (상세 정보 포함)
        # self.name (개별 tool 이름)을 사용하여 정확한 profile 조회
        return self.scheduler.get_location_for_call_with_reason(
            tool_name=self.name,
            args=args,
        )

    def _create_fixed_scheduling_result(self, location: str):
        """
        filter_location으로 고정된 경우의 SchedulingResult 생성

        SubAgent 모드에서 schedule_chain()으로 이미 결정된 location을 사용하므로
        scheduler 호출 없이 SchedulingResult를 생성합니다.

        Args:
            location: 고정된 location (filter_location)

        Returns:
            SchedulingResult: 고정 location 정보
        """
        from .scheduler import SchedulingResult

        return SchedulingResult(
            tool_name=self.name,
            location=location,
            reason="filter_location_fixed",
            available_locations=[location],
            decision_time_ns=0,
            score=0.0,
            exec_cost=0.0,
            trans_cost=0.0,
            fixed=True,
        )

    @property
    def available_locations(self) -> list[str]:
        """사용 가능한 location 목록"""
        return list(self.backend_tools.keys())

    def __repr__(self) -> str:
        locations = ", ".join(self.available_locations)
        return f"ProxyTool({self.name}, locations=[{locations}])"
