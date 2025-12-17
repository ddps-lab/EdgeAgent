"""
Location-Aware Proxy Tool

LLM에 노출되는 Proxy tool - 호출 시 Scheduler가 location을 결정하고
적절한 backend tool로 routing

메트릭 수집 기능 통합:
- 자동 latency 측정
- 입출력 크기 계산
- 리소스 사용량 추적
"""

import re
from typing import Any, Optional, TYPE_CHECKING


# EDGE 서버 중 camelCase 파라미터를 사용하는 서버 목록
EDGE_CAMELCASE_SERVERS = {"log_parser", "data_aggregate"}

# snake_case → camelCase 변환이 필요한 파라미터 매핑
SNAKE_TO_CAMEL_MAP = {
    # log_parser
    "log_content": "logContent",
    "format_type": "formatType",
    "max_entries": "maxEntries",
    "min_level": "minLevel",
    "include_levels": "includeLevels",
    "case_sensitive": "caseSensitive",
    # data_aggregate
    "group_by": "groupBy",
    "count_field": "countField",
    "sum_fields": "sumFields",
    "title_field": "titleField",
    "summary_field": "summaryField",
    "score_field": "scoreField",
    "key_fields": "keyFields",
    "time_series": "timeSeries",
    "time_field": "timeField",
    "value_field": "valueField",
    "bucket_count": "bucketCount",
}


def convert_args_to_camelcase(args: dict[str, Any]) -> dict[str, Any]:
    """
    snake_case 파라미터를 camelCase로 변환

    EDGE 서버 중 일부(log_parser, data_aggregate)가 camelCase를 사용하므로
    호출 전 파라미터명 변환이 필요
    """
    converted = {}
    for key, value in args.items():
        new_key = SNAKE_TO_CAMEL_MAP.get(key, key)
        converted[new_key] = value
    return converted


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

        1. Scheduler가 location 결정 (with reason)
        2. 해당 location의 backend tool 선택 (fallback 지원)
        3. Trace 기록 및 메트릭 수집
        4. 실제 tool 호출
        """
        # 1. Scheduler가 location 결정 (상세 정보 포함)
        scheduling_result = self._get_location_with_reason(kwargs)
        location = scheduling_result.location
        fallback_occurred = False

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

        # 3. EDGE camelCase 서버용 파라미터 변환
        invoke_kwargs = kwargs
        if location == "EDGE" and self.parent_tool_name in EDGE_CAMELCASE_SERVERS:
            invoke_kwargs = convert_args_to_camelcase(kwargs)

        # 4. 메트릭 수집과 함께 tool 호출
        if self.metrics_collector is not None:
            async with self.metrics_collector.start_call(
                tool_name=self.name,
                parent_tool_name=self.parent_tool_name,
                location=location,
                args=kwargs,  # 원본 args 기록 (snake_case)
            ) as ctx:
                # Scheduling 정보 추가 (cost 포함)
                ctx.add_scheduling_info(
                    reason=scheduling_result.reason,
                    constraints=scheduling_result.constraints_checked,
                    available=scheduling_result.available_locations,
                    decision_time_ns=scheduling_result.decision_time_ns,
                    score=scheduling_result.score,
                    exec_cost=scheduling_result.exec_cost,
                    trans_cost=scheduling_result.trans_cost,
                    fixed=scheduling_result.fixed,
                )
                try:
                    result = await backend_tool.ainvoke(invoke_kwargs)
                    ctx.set_result(result)
                    ctx.set_actual_location(location, fallback=fallback_occurred)
                    return result
                except Exception as e:
                    ctx.set_error(e)
                    raise
        else:
            # 기존 동작 유지 (메트릭 수집 없음)
            return await backend_tool.ainvoke(invoke_kwargs)

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
        # self.name (개별 tool 이름)을 사용하여 정확한 profile 조회
        return self.scheduler.get_location_for_call_with_reason(
            tool_name=self.name,
            args=args,
        )

    @property
    def available_locations(self) -> list[str]:
        """사용 가능한 location 목록"""
        return list(self.backend_tools.keys())

    def __repr__(self) -> str:
        locations = ", ".join(self.available_locations)
        return f"ProxyTool({self.name}, locations=[{locations}])"
