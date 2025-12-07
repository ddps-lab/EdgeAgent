"""
Location-Aware Proxy Tool

LLM에 노출되는 Proxy tool - 호출 시 Scheduler가 location을 결정하고
적절한 backend tool로 routing
"""

from typing import Any, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from .types import Location

if TYPE_CHECKING:
    from .scheduler import StaticScheduler


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

        1. Scheduler가 location 결정
        2. 해당 location의 backend tool 선택
        3. Trace 기록
        4. 실제 tool 호출
        """
        # 1. Scheduler가 location 결정
        location = self._get_location(kwargs)

        # 2. 해당 location의 backend tool 선택
        backend_tool = self.backend_tools.get(location)
        if not backend_tool:
            available = list(self.backend_tools.keys())
            raise ValueError(
                f"No backend tool for location '{location}'. "
                f"Available locations: {available}"
            )

        # 3. Trace 기록
        self.execution_trace.append({
            "tool": self.name,
            "parent_tool": self.parent_tool_name,
            "location": location,
            "args_keys": list(kwargs.keys()),
        })

        # 4. 실제 tool 호출
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

    @property
    def available_locations(self) -> list[str]:
        """사용 가능한 location 목록"""
        return list(self.backend_tools.keys())

    def __repr__(self) -> str:
        locations = ", ".join(self.available_locations)
        return f"ProxyTool({self.name}, locations=[{locations}])"
