"""
Tool Sequence Planner

Tool들을 location별로 grouping하여 Sub-Agent 실행 계획을 생성합니다.

기능:
- Tool list를 location별로 grouping
- 순차적 실행을 위한 partition 생성
- Static (YAML) 및 Dynamic (LLM) planning 지원

Usage:
    planner = ToolSequencePlanner(scheduler, registry)
    plan = planner.plan_by_location(["filesystem", "log_parser", "summarize"])
    # Returns: {"DEVICE": ["filesystem"], "EDGE": ["log_parser", "summarize"]}
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
from pathlib import Path

from .types import Location, LOCATIONS
from .scheduler import BaseScheduler
from .registry import ToolRegistry

if TYPE_CHECKING:
    from .types import ChainSchedulingResult


@dataclass
class Partition:
    """
    Location별 tool 그룹 (Sub-Agent 실행 단위)

    예: EDGE에서 실행할 [log_parser, summarize, data_aggregate]
    """
    location: Location
    tools: list[str] = field(default_factory=list)
    tool_configs: dict[str, dict] = field(default_factory=dict)

    def add_tool(self, tool_name: str, config: dict | None = None):
        """Tool을 partition에 추가"""
        self.tools.append(tool_name)
        if config:
            self.tool_configs[tool_name] = config


@dataclass
class ExecutionPlan:
    """
    전체 실행 계획

    여러 Partition을 순서대로 실행하기 위한 계획.
    각 Partition은 특정 location의 Sub-Agent에서 실행됩니다.
    """
    partitions: list[Partition] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    total_tools: int = 0
    chain_scheduling_result: Optional["ChainSchedulingResult"] = None

    def add_partition(self, partition: Partition):
        """Partition 추가"""
        self.partitions.append(partition)
        self.total_tools += len(partition.tools)

    def get_location_groups(self) -> dict[Location, list[str]]:
        """Location별 tool 목록 반환"""
        result: dict[Location, list[str]] = {}
        for partition in self.partitions:
            if partition.location not in result:
                result[partition.location] = []
            result[partition.location].extend(partition.tools)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Dictionary 형태로 변환"""
        return {
            "partitions": [
                {
                    "location": p.location,
                    "tools": p.tools,
                    "tool_configs": p.tool_configs,
                }
                for p in self.partitions
            ],
            "tool_sequence": self.tool_sequence,
            "total_tools": self.total_tools,
        }


class ToolSequencePlanner:
    """
    Tool sequence를 분석하고 location별로 grouping

    Sub-Agent 실행을 위해 tool들을 location별로 묶습니다.
    연속된 같은 location의 tool들은 하나의 partition으로 묶입니다.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        registry: ToolRegistry,
    ):
        """
        Args:
            scheduler: 위치 결정을 위한 스케줄러
            registry: Tool 정보가 저장된 레지스트리
        """
        self.scheduler = scheduler
        self.registry = registry

    def plan_by_location(
        self,
        tool_names: list[str],
        merge_adjacent: bool = True,
    ) -> dict[Location, list[str]]:
        """
        Tool list를 location별로 grouping (단순 버전)

        Args:
            tool_names: Tool 이름 목록
            merge_adjacent: 인접한 같은 location을 병합할지 여부

        Returns:
            Location별 tool 목록 dict
            예: {"DEVICE": ["filesystem"], "EDGE": ["log_parser", "summarize"]}
        """
        result: dict[Location, list[str]] = {}

        for tool_name in tool_names:
            location = self.scheduler.get_location(tool_name)

            if location not in result:
                result[location] = []
            result[location].append(tool_name)

        return result

    def create_execution_plan(
        self,
        tool_names: list[str],
        preserve_order: bool = True,
    ) -> ExecutionPlan:
        """
        실행 계획 생성

        Tool sequence를 location별 partition으로 분할합니다.
        preserve_order=True이면 tool sequence 순서를 유지하면서
        연속된 같은 location tool들을 하나의 partition으로 묶습니다.

        Args:
            tool_names: Tool 이름 목록 (실행 순서대로)
            preserve_order: 순서 유지 여부

        Returns:
            ExecutionPlan: 실행 계획

        Example:
            tool_names = ["filesystem", "log_parser", "summarize", "filesystem"]
            locations  = [DEVICE,       EDGE,         EDGE,        DEVICE      ]

            Result (preserve_order=True):
                Partition 1: DEVICE [filesystem]
                Partition 2: EDGE   [log_parser, summarize]
                Partition 3: DEVICE [filesystem]

            Result (preserve_order=False):
                Group by location only (순서 무관)
        """
        plan = ExecutionPlan(tool_sequence=tool_names)

        if not tool_names:
            return plan

        if preserve_order:
            # 순서 유지하면서 연속된 같은 location 병합
            current_partition: Optional[Partition] = None

            for tool_name in tool_names:
                location = self.scheduler.get_location(tool_name)
                endpoint = self.registry.get_endpoint(tool_name, location)

                # Tool config 생성
                tool_config = {}
                if endpoint:
                    tool_config = {
                        "transport": endpoint.transport,
                        "command": endpoint.command,
                        "args": endpoint.args,
                    }

                if current_partition is None:
                    # 첫 번째 partition 시작
                    current_partition = Partition(location=location)
                    current_partition.add_tool(tool_name, tool_config)
                elif current_partition.location == location:
                    # 같은 location이면 현재 partition에 추가
                    current_partition.add_tool(tool_name, tool_config)
                else:
                    # 다른 location이면 새 partition 시작
                    plan.add_partition(current_partition)
                    current_partition = Partition(location=location)
                    current_partition.add_tool(tool_name, tool_config)

            # 마지막 partition 추가
            if current_partition:
                plan.add_partition(current_partition)
        else:
            # 순서 무관하게 location별로 grouping
            location_groups = self.plan_by_location(tool_names)

            for location in LOCATIONS:
                if location in location_groups:
                    partition = Partition(location=location)
                    for tool_name in location_groups[location]:
                        endpoint = self.registry.get_endpoint(tool_name, location)
                        tool_config = {}
                        if endpoint:
                            tool_config = {
                                "transport": endpoint.transport,
                                "command": endpoint.command,
                                "args": endpoint.args,
                            }
                        partition.add_tool(tool_name, tool_config)
                    plan.add_partition(partition)

        return plan

    def create_execution_plan_with_chain_scheduler(
        self,
        tool_names: list[str],
        chain_scheduler: BaseScheduler,
        tool_args: Optional[list[dict]] = None,
    ) -> ExecutionPlan:
        """
        schedule_chain을 사용한 실행 계획 생성

        전체 tool chain을 한 번에 최적화하여 각 tool의 실행 위치를 결정합니다.
        모든 스케줄러가 schedule_chain 메서드를 구현하므로 어떤 스케줄러든 사용 가능.

        Args:
            tool_names: Tool 이름 목록 (실행 순서대로)
            chain_scheduler: schedule_chain을 갖는 스케줄러 인스턴스
            tool_args: 각 Tool의 인자 리스트 (optional, local_data 처리용)

        Returns:
            ExecutionPlan: 최적화된 실행 계획

        Example:
            tool_names = ["filesystem", "image_resize", "data_aggregate", "filesystem"]

            BruteForceChainScheduler가 전체 chain 비용을 최소화하는 배치 결정:
                filesystem → DEVICE (local_data로 고정)
                image_resize → EDGE (Score 기반 최적화)
                data_aggregate → EDGE (Score 기반 최적화)
                filesystem → DEVICE (local_data로 고정)
        """
        plan = ExecutionPlan(tool_sequence=tool_names)

        if not tool_names:
            return plan

        # 전체 chain 최적화
        chain_result = chain_scheduler.schedule_chain(tool_names, tool_args)

        # ChainSchedulingResult → Partition 변환
        current_partition: Optional[Partition] = None

        for placement in chain_result.placements:
            tool_name = placement.tool_name
            location = placement.location

            endpoint = self.registry.get_endpoint(tool_name, location)
            tool_config = {}
            if endpoint:
                tool_config = {
                    "transport": endpoint.transport,
                    "command": endpoint.command,
                    "args": endpoint.args,
                }

            if current_partition is None:
                # 첫 번째 partition 시작
                current_partition = Partition(location=location)
                current_partition.add_tool(tool_name, tool_config)
            elif current_partition.location == location:
                # 같은 location이면 현재 partition에 추가
                current_partition.add_tool(tool_name, tool_config)
            else:
                # 다른 location이면 새 partition 시작
                plan.add_partition(current_partition)
                current_partition = Partition(location=location)
                current_partition.add_tool(tool_name, tool_config)

        # 마지막 partition 추가
        if current_partition:
            plan.add_partition(current_partition)

        # Chain scheduling 결과 저장
        plan.chain_scheduling_result = chain_result

        return plan

    def get_tools_for_location(
        self,
        tool_names: list[str],
        target_location: Location,
    ) -> list[str]:
        """
        특정 location에서 실행될 tool 목록 반환

        Args:
            tool_names: 전체 tool 목록
            target_location: 대상 location

        Returns:
            해당 location에서 실행될 tool 목록
        """
        result = []
        for tool_name in tool_names:
            location = self.scheduler.get_location(tool_name)
            if location == target_location:
                result.append(tool_name)
        return result

    def estimate_partition_sizes(
        self,
        tool_names: list[str],
    ) -> dict[Location, int]:
        """
        Location별 partition 크기 (tool 개수) 추정

        Args:
            tool_names: Tool 목록

        Returns:
            Location별 tool 개수 dict
        """
        location_groups = self.plan_by_location(tool_names)
        return {loc: len(tools) for loc, tools in location_groups.items()}

    def __repr__(self) -> str:
        return f"ToolSequencePlanner(scheduler={self.scheduler})"
