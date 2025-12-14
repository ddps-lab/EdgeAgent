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

        1. Scheduler가 location 결정 (with reason)
        2. 해당 location의 backend tool 선택 (fallback 지원)
        3. Trace 기록 및 메트릭 수집
        4. 실제 tool 호출
        """
        # 1. Scheduler가 location 결정 (상세 정보 포함)
        scheduling_result = self._get_location_with_reason(kwargs)
        location = scheduling_result.location
        fallback_occurred = False

        # 2. 해당 location의 backend tool 선택 (fallback 지원)
        backend_tool = self.backend_tools.get(location)
        if not backend_tool:
            # Scheduler가 결정한 location에 backend이 없으면 fallback
            # SubAgent가 특정 location만 지원하는 경우 발생
            available = list(self.backend_tools.keys())
            if available:
                location = available[0]  # 첫 번째 available location 사용
                backend_tool = self.backend_tools[location]
                fallback_occurred = True
            else:
                raise ValueError(
                    f"No backend tool available for '{self.name}'. "
                    f"Scheduled location: {scheduling_result.location}"
                )

        # 3. Trace 기록 (backward compatibility)
        self.execution_trace.append({
            "tool": self.name,
            "parent_tool": self.parent_tool_name,
            "location": location,
            "fallback": fallback_occurred,
            "args_keys": list(kwargs.keys()),
        })

        # 4. 메트릭 수집과 함께 tool 호출
        if self.metrics_collector is not None:
            # 통합 메트릭 수집 모드
            async with self.metrics_collector.start_call(
                tool_name=self.name,
                parent_tool_name=self.parent_tool_name,
                location=location,
                args=kwargs,
            ) as ctx:
                # Scheduling 정보 추가
                ctx.add_scheduling_info(
                    reason=scheduling_result.reason,
                    constraints=scheduling_result.constraints_checked,
                    available=scheduling_result.available_locations,
                    decision_time_ns=scheduling_result.decision_time_ns,
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
        # parent_tool_name을 사용 (registry에 등록된 이름)
        return self.scheduler.get_location_for_call(
            tool_name=self.parent_tool_name,
            args=args,
        )

    def _get_location_with_reason(self, args: dict[str, Any]):
        """
        Location 결정 (상세 정보 포함)

        Returns:
            SchedulingResult: location과 결정 메타데이터
        """
        from .scheduler import SchedulingResult

        if self.scheduler is None:
            # Scheduler 없으면 첫 번째 available location
            location = list(self.backend_tools.keys())[0]
            return SchedulingResult(
                location=location,
                reason="no_scheduler_default",
                constraints_checked=[],
                available_locations=list(self.backend_tools.keys()),
                decision_time_ns=0,
            )

        # Scheduler에게 location 결정 요청 (상세 정보 포함)
        # parent_tool_name을 사용 (registry에 등록된 이름)
        return self.scheduler.get_location_for_call_with_reason(
            tool_name=self.parent_tool_name,
            args=args,
        )

    @property
    def available_locations(self) -> list[str]:
        """사용 가능한 location 목록"""
        return list(self.backend_tools.keys())

    def __repr__(self) -> str:
        locations = ", ".join(self.available_locations)
        return f"ProxyTool({self.name}, locations=[{locations}])"
