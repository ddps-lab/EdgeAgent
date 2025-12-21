"""
Scheduler Module

Tool 실행 location 결정을 위한 스케줄러 구현

- BaseScheduler: 추상 인터페이스 (향후 Heuristic/DP Scheduler로 교체 가능)
- StaticScheduler: YAML static_mapping + args 기반 동적 routing
- Baseline Schedulers: All-DEVICE, All-EDGE, All-CLOUD (비교 실험용)
- HeuristicScheduler: Profile 기반 휴리스틱 스케줄러
"""

import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Any

from .types import Location, Runtime, LOCATIONS, ChainSchedulingResult, SchedulingConstraints, SchedulingResult
from .registry import ToolRegistry
from .profiles import ToolProfile
from .scoring import ScoringEngine


# ============================================================================
# Scheduling Context
# ============================================================================

@dataclass
class SchedulingContext:
    """
    Scheduling 결정에 필요한 컨텍스트

    향후 확장:
    - latency_requirements: 지연시간 요구사항
    - cost_budget: 비용 예산
    - current_load: 각 location의 현재 부하
    """
    tool_profile: Optional[ToolProfile] = None
    available_locations: list[Location] = field(default_factory=list)


# ============================================================================
# Base Scheduler Interface
# ============================================================================

class BaseScheduler(ABC):
    """
    Scheduler 추상 인터페이스

    향후 Heuristic/DP Scheduler로 교체 가능하도록 인터페이스 정의
    """

    @abstractmethod
    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        """
        호출 시점 location 결정

        Args:
            tool_name: Tool 이름
            args: Tool 호출 인자
            context: Scheduling 컨텍스트

        Returns:
            결정된 location (DEVICE/EDGE/CLOUD)
        """
        pass

    @abstractmethod
    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        """
        호출 시점 location 결정 (상세 정보 포함)

        Args:
            tool_name: Tool 이름
            args: Tool 호출 인자
            context: Scheduling 컨텍스트

        Returns:
            SchedulingResult: location과 결정 메타데이터
        """
        pass

    @abstractmethod
    def get_location(self, tool_name: str) -> Location:
        """
        기본 location 조회 (static)

        Args:
            tool_name: Tool 이름

        Returns:
            기본 location
        """
        pass

    @abstractmethod
    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        """
        Runtime 선택

        Args:
            tool_name: Tool 이름
            location: 실행 location

        Returns:
            선택된 runtime (WASI/CONTAINER)
        """
        pass

    def get_required_locations(self) -> list[Location]:
        """
        스케줄러가 사용하는 location 목록 (서버 초기화용)

        All* 스케줄러에서 override하여 단일 location만 반환.
        기본값: 모든 location

        Returns:
            필요한 location 목록
        """
        return list(LOCATIONS)

    @abstractmethod
    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        """
        Tool Chain 전체의 스케줄링 결정

        Args:
            tool_names: Tool 이름 리스트
            tool_args: 각 Tool의 인자 리스트 (optional)

        Returns:
            ChainSchedulingResult: Chain 전체의 스케줄링 결과
        """
        pass


# ============================================================================
# Static Scheduler Implementation
# ============================================================================

class StaticScheduler(BaseScheduler):
    """
    Static mapping 기반 스케줄러

    결정 규칙:
    1. tool_name → server_name 매핑 (tools 섹션에서 빌드)
    2. server_name → location 매핑 (static_mapping 섹션)
    3. 기본값: DEVICE
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry
        self.static_mapping: dict[str, Location] = {}  # server_name → Location
        self.tool_to_server: dict[str, str] = {}  # tool_name → server_name
        self._load_config()

    def _load_config(self):
        """YAML 파일에서 static_mapping과 tool_to_server 매핑 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # 1. tools 섹션에서 tool_name → server_name 매핑 빌드
        tools_config = config.get("tools", {})
        for server_name, server_config in tools_config.items():
            tool_profiles = server_config.get("tool_profiles", {})
            for tool_name in tool_profiles:
                self.tool_to_server[tool_name] = server_name

        # 2. static_mapping 섹션 로드 (server_name → Location)
        raw_mapping = config.get("static_mapping", {})
        for server_name, location in raw_mapping.items():
            self.static_mapping[server_name] = location

    def get_location(self, tool_name: str) -> Location:
        """tool_name → server_name → location"""
        server_name = self.tool_to_server.get(tool_name)
        if server_name:
            return self.static_mapping.get(server_name, "DEVICE")
        return "DEVICE"

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        return self.get_location(tool_name)

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        start_time_ns = time.perf_counter_ns()
        location = self.get_location(tool_name)
        decision_time_ns = time.perf_counter_ns() - start_time_ns

        server_name = self.tool_to_server.get(tool_name)
        reason = "static_mapping" if server_name and server_name in self.static_mapping else "default_device"

        return SchedulingResult(
            tool_name=tool_name,
            location=location,
            reason=reason,
            decision_time_ns=decision_time_ns,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        if location == "DEVICE":
            return "WASI"
        profile = self.registry.get_profile(tool_name)
        if profile and not profile.wasi_compatible:
            return "CONTAINER"
        return "WASI"

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        start_time_ns = time.perf_counter_ns()
        placements = []

        for tool_name in tool_names:
            result = self.get_location_for_call_with_reason(tool_name)
            placements.append(result)

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=0.0,
            search_space_size=1,
            valid_combinations=1,
            optimization_method="static_mapping",
            decision_time_ns=decision_time_ns,
        )

    def __repr__(self) -> str:
        return f"StaticScheduler(mappings={len(self.static_mapping)})"


# ============================================================================
# Heuristic Scheduler (Profile + Server Constraints 기반)
# ============================================================================

class HeuristicScheduler(BaseScheduler):
    """
    Profile 기반 휴리스틱 스케줄러

    SchedulingConstraints와 DataLocality를 기반으로 location을 결정합니다.

    결정 규칙:
    1. Type C (local_data) → path 기반 노드 고정
    2. requires_gpu → CLOUD (GPU는 클라우드에만 존재)
    3. privacy_sensitive → DEVICE
    4. 기본값 → DEVICE
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry
        self.tool_constraints: dict[str, SchedulingConstraints] = {}
        self._load_config()

    def _load_config(self):
        """YAML 파일에서 tool_profiles의 constraints 로드"""
        if not self.config_path.exists():
            return

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # tools 섹션에서 tool별 constraints 빌드
        tools_config = config.get("tools", {})
        for server_name, server_config in tools_config.items():
            tool_profiles = server_config.get("tool_profiles", {})
            for tool_name, profile in tool_profiles.items():
                self.tool_constraints[tool_name] = SchedulingConstraints(
                    requires_gpu=profile.get("requires_gpu", False),
                    privacy_sensitive=profile.get("privacy_sensitive", False),
                    wasi_compatible=profile.get("wasi_compatible", True),
                )

    def _get_constraints(self, tool_name: str) -> SchedulingConstraints:
        """Tool의 SchedulingConstraints 반환"""
        return self.tool_constraints.get(tool_name, SchedulingConstraints())

    def _extract_fixed_location(
        self,
        tool_name: str,
        args: Optional[dict] = None,
    ) -> Optional[Location]:
        """
        Type C (local_data) Tool의 고정 노드 추출

        Path prefix 규칙:
        - edge://... 또는 /edgeagent_edge/ → EDGE
        - cloud://... 또는 /edgeagent_cloud/ → CLOUD
        - 그 외 → DEVICE (기본)

        Args:
            tool_name: Tool 이름
            args: Tool 호출 인자

        Returns:
            고정 노드 또는 None (고정 불필요)
        """
        profile = self.registry.get_profile(tool_name)
        data_locality = getattr(profile, 'data_locality', 'args_only') if profile else 'args_only'

        if data_locality != "local_data":
            return None

        # args가 없으면 DEVICE 기본값
        if not args:
            return "DEVICE"

        # args에서 path 추출
        path = args.get("path") or args.get("directory") or args.get("image_path") or args.get("repo_path")
        if not path:
            return "DEVICE"

        path_str = str(path).lower()
        if path_str.startswith("edge://") or "edgeagent_edge" in path_str or "/edge/" in path_str:
            return "EDGE"
        elif path_str.startswith("cloud://") or "edgeagent_cloud" in path_str or "/cloud/" in path_str:
            return "CLOUD"
        else:
            return "DEVICE"

    def get_location(self, tool_name: str) -> Location:
        """
        Tool의 실행 location 결정 (args 없이)

        우선순위:
        1. requires_gpu → CLOUD (GPU는 클라우드에만 존재)
        2. wasi_compatible=False → EDGE 제외 (DEVICE 반환)
        3. privacy_sensitive → DEVICE
        4. 기본값 → DEVICE
        """
        constraints = self._get_constraints(tool_name)

        if constraints.requires_gpu:
            return "CLOUD"

        if not constraints.wasi_compatible:
            return "DEVICE"  # EDGE 제외

        if constraints.privacy_sensitive:
            return "DEVICE"

        return "DEVICE"  # 기본값

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        """
        호출 시점 location 결정

        우선순위:
        1. Type C (local_data) → path 기반 노드 고정
        2. requires_gpu → CLOUD (GPU는 클라우드에만 존재)
        3. wasi_compatible=False → EDGE 제외
        4. privacy_sensitive → DEVICE
        5. 기본값 → DEVICE
        """
        constraints = self._get_constraints(tool_name)

        # 1. Type C (local_data) 체크 → 노드 고정
        fixed_location = self._extract_fixed_location(tool_name, args)
        if fixed_location:
            # wasi_compatible=False이고 EDGE로 고정되면 DEVICE로 변경
            if fixed_location == "EDGE" and not constraints.wasi_compatible:
                return "DEVICE"
            return fixed_location

        # 2. requires_gpu → CLOUD
        if constraints.requires_gpu:
            return "CLOUD"

        # 3. wasi_compatible=False → EDGE 제외
        if not constraints.wasi_compatible:
            return "DEVICE"

        # 4. privacy_sensitive → DEVICE
        if constraints.privacy_sensitive:
            return "DEVICE"

        # 5. 기본값 DEVICE
        return "DEVICE"

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        """
        호출 시점 location 결정 (상세 정보 포함)
        """
        start_time_ns = time.perf_counter_ns()
        constraints = self._get_constraints(tool_name)

        # 1. Type C (local_data) 체크 → 노드 고정
        fixed_location = self._extract_fixed_location(tool_name, args)
        if fixed_location:
            # wasi_compatible=False이고 EDGE로 고정되면 DEVICE로 변경
            if fixed_location == "EDGE" and not constraints.wasi_compatible:
                fixed_location = "DEVICE"
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                tool_name=tool_name,
                location=fixed_location,
                reason="local_data_fixed",
                available_locations=[fixed_location],
                decision_time_ns=decision_time_ns,
                constraints=constraints,
                fixed=True,
            )

        # 2. requires_gpu → CLOUD
        if constraints.requires_gpu:
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                tool_name=tool_name,
                location="CLOUD",
                reason="requires_gpu",
                available_locations=["CLOUD"],
                decision_time_ns=decision_time_ns,
                constraints=constraints,
            )

        # 3. wasi_compatible=False → EDGE 제외 (DEVICE 또는 CLOUD)
        if not constraints.wasi_compatible:
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                tool_name=tool_name,
                location="DEVICE",
                reason="wasi_incompatible_no_edge",
                available_locations=["DEVICE", "CLOUD"],
                decision_time_ns=decision_time_ns,
                constraints=constraints,
            )

        # 4. privacy_sensitive → DEVICE
        if constraints.privacy_sensitive:
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                tool_name=tool_name,
                location="DEVICE",
                reason="privacy_constraint",
                available_locations=["DEVICE"],
                decision_time_ns=decision_time_ns,
                constraints=constraints,
            )

        # 4. 기본값 → DEVICE
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            tool_name=tool_name,
            location="DEVICE",
            reason="default_device",
            available_locations=list(LOCATIONS),
            decision_time_ns=decision_time_ns,
            constraints=constraints,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)

        if location == "DEVICE":
            return "WASI"

        if profile and not profile.wasi_compatible:
            return "CONTAINER"

        return "WASI"

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        """
        Tool Chain의 노드 배치 결정 (Heuristic)

        각 Tool에 대해 휴리스틱 규칙을 적용하여 location을 결정합니다:
        1. Type C (local_data) → path 기반 노드 고정
        2. requires_gpu → CLOUD (GPU는 클라우드에만 존재)
        3. privacy_sensitive → DEVICE
        4. static_mapping 사용
        """
        start_time_ns = time.perf_counter_ns()
        placements = []

        for i, tool_name in enumerate(tool_names):
            args = tool_args[i] if tool_args and i < len(tool_args) else None
            result = self.get_location_for_call_with_reason(tool_name, args)
            placements.append(result)

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=0.0,
            search_space_size=len(tool_names),
            valid_combinations=1,
            optimization_method="heuristic",
            decision_time_ns=decision_time_ns,
        )

    def __repr__(self) -> str:
        return "HeuristicScheduler()"


# ============================================================================
# Baseline Schedulers (비교 실험용)
# ============================================================================

class AllDeviceScheduler(BaseScheduler):
    """
    Baseline: 모든 Tool → DEVICE

    비교 실험을 위한 baseline 스케줄러.
    모든 tool을 DEVICE에서 실행합니다.
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry

    def get_location(self, tool_name: str) -> Location:
        return "DEVICE"

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        return "DEVICE"

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        start_time_ns = time.perf_counter_ns()
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            tool_name=tool_name,
            location="DEVICE",
            reason="all_device_policy",
            available_locations=["DEVICE"],
            decision_time_ns=decision_time_ns,
            score=0.0,
            exec_cost=0.0,
            trans_cost=0.0,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["DEVICE"]

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        start_time_ns = time.perf_counter_ns()
        placements = []

        for tool_name in tool_names:
            placements.append(SchedulingResult(
                tool_name=tool_name,
                location="DEVICE",
                reason="all_device_policy",
                decision_time_ns=0,
                score=0.0,
                exec_cost=0.0,
                trans_cost=0.0,
            ))

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=0.0,
            search_space_size=1,
            valid_combinations=1,
            optimization_method="all_device",
            decision_time_ns=decision_time_ns,
        )

    def __repr__(self) -> str:
        return "AllDeviceScheduler()"


class AllEdgeScheduler(BaseScheduler):
    """
    Baseline: 모든 Tool → EDGE

    비교 실험을 위한 baseline 스케줄러.
    모든 tool을 EDGE에서 실행합니다.
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry

    def get_location(self, tool_name: str) -> Location:
        return "EDGE"

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        return "EDGE"

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        start_time_ns = time.perf_counter_ns()
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            tool_name=tool_name,
            location="EDGE",
            reason="all_edge_policy",
            available_locations=["EDGE"],
            decision_time_ns=decision_time_ns,
            score=0.0,
            exec_cost=0.0,
            trans_cost=0.0,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)
        if profile and not profile.wasi_compatible:
            return "CONTAINER"
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["EDGE"]

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        start_time_ns = time.perf_counter_ns()
        placements = []

        for tool_name in tool_names:
            placements.append(SchedulingResult(
                tool_name=tool_name,
                location="EDGE",
                reason="all_edge_policy",
                decision_time_ns=0,
                score=0.0,
                exec_cost=0.0,
                trans_cost=0.0,
            ))

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=0.0,
            search_space_size=1,
            valid_combinations=1,
            optimization_method="all_edge",
            decision_time_ns=decision_time_ns,
        )

    def __repr__(self) -> str:
        return "AllEdgeScheduler()"


class AllCloudScheduler(BaseScheduler):
    """
    Baseline: 모든 Tool → CLOUD

    비교 실험을 위한 baseline 스케줄러.
    모든 tool을 CLOUD에서 실행합니다.
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry

    def get_location(self, tool_name: str) -> Location:
        return "CLOUD"

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        return "CLOUD"

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        start_time_ns = time.perf_counter_ns()
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            tool_name=tool_name,
            location="CLOUD",
            reason="all_cloud_policy",
            available_locations=["CLOUD"],
            decision_time_ns=decision_time_ns,
            score=0.0,
            exec_cost=0.0,
            trans_cost=0.0,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)
        if profile and not profile.wasi_compatible:
            return "CONTAINER"
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["CLOUD"]

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        start_time_ns = time.perf_counter_ns()
        placements = []

        for tool_name in tool_names:
            placements.append(SchedulingResult(
                tool_name=tool_name,
                location="CLOUD",
                reason="all_cloud_policy",
                decision_time_ns=0,
                score=0.0,
                exec_cost=0.0,
                trans_cost=0.0,
            ))

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=0.0,
            search_space_size=1,
            valid_combinations=1,
            optimization_method="all_cloud",
            decision_time_ns=decision_time_ns,
        )

    def __repr__(self) -> str:
        return "AllCloudScheduler()"


# ============================================================================
# BruteForce Chain Scheduler
# ============================================================================

class BruteForceChainScheduler(BaseScheduler):
    """
    Tool Chain 전체를 Brute-Force로 최적화하는 스케줄러

    Cost 공식:
      Cost_i(u, v) = α * (P_comp[i][u] + β * P_net[u]) + (1-α) * P_comm[(v,u)]

    Data Locality 처리:
      - Type A (args_only): Score 기반 최적화
      - Type B (external_data): beta=1 자동 적용, Score 기반 최적화
      - Type C (local_data): args의 path prefix로 노드 고정

    BaseScheduler 인터페이스:
      - get_location_for_call: proxy_tool에서 호출, 내부 상태(prev_location) 추적
      - get_location_for_call_with_reason: 상세 정보 포함 버전
    """

    def __init__(
        self,
        config_path: str | Path,
        system_config_path: str | Path,
        registry: ToolRegistry,
    ):
        """
        Args:
            config_path: tools_scenario*.yaml 경로
            system_config_path: system.yaml 경로
            registry: ToolRegistry 인스턴스
        """
        self.config_path = Path(config_path)
        self.system_config_path = Path(system_config_path)
        self.registry = registry
        self.scoring_engine = ScoringEngine(system_config_path, registry)

        # Agent 모드용 상태 (prev_location 추적)
        self._prev_location: Optional[Location] = None
        self._call_count: int = 0

    def _extract_fixed_location_with_reason(
        self,
        tool_name: str,
        args: Optional[dict] = None,
    ) -> tuple[Optional[Location], Optional[str]]:
        """
        Tool의 고정 노드와 이유 추출

        고정 조건 (우선순위):
        1. requires_gpu=True → CLOUD (GPU는 클라우드에만 존재)
        2. privacy_sensitive=True → DEVICE
        3. data_locality="local_data" → path prefix 기반 노드 결정

        Args:
            tool_name: Tool 이름
            args: Tool 호출 인자

        Returns:
            (고정 노드, 고정 이유) 또는 (None, None)
        """
        profile = self.registry.get_profile(tool_name)
        if not profile:
            return None, None

        # 1. requires_gpu 체크 (GPU는 클라우드에만 존재)
        if getattr(profile, 'requires_gpu', False):
            return "CLOUD", "requires_gpu"

        # 2. privacy_sensitive 체크
        if getattr(profile, 'privacy_sensitive', False):
            return "DEVICE", "privacy_sensitive"

        # 3. data_locality="local_data" 체크
        data_locality = getattr(profile, 'data_locality', 'args_only')
        if data_locality == "local_data":
            location = self._get_local_data_location(args)
            # wasi_compatible=False이고 EDGE로 고정되면 DEVICE로 변경
            wasi_compatible = getattr(profile, 'wasi_compatible', True)
            if location == "EDGE" and not wasi_compatible:
                location = "DEVICE"
            return location, f"local_data_{location.lower()}"

        return None, None

    def _get_local_data_location(self, args: Optional[dict] = None) -> Location:
        """local_data Tool의 path prefix 기반 노드 결정"""
        if not args:
            return "DEVICE"

        path = args.get("path") or args.get("directory") or args.get("image_path") or args.get("repo_path")
        if not path:
            return "DEVICE"

        path_str = str(path).lower()
        if path_str.startswith("edge://") or "edgeagent_edge" in path_str or "/edge/" in path_str:
            return "EDGE"
        elif path_str.startswith("cloud://") or "edgeagent_cloud" in path_str or "/cloud/" in path_str:
            return "CLOUD"
        else:
            return "DEVICE"

    def _extract_fixed_location(
        self,
        tool_name: str,
        args: Optional[dict] = None,
    ) -> Optional[Location]:
        """
        Type C (local_data) Tool의 고정 노드 추출 (하위 호환성 유지)

        Returns:
            고정 노드 또는 None (고정 불필요)
        """
        location, _ = self._extract_fixed_location_with_reason(tool_name, args)
        return location

    def get_optimal_location_for_tool(
        self,
        tool_name: str,
        prev_location: Optional[Location] = None,
        is_first: bool = False,
        is_last: bool = False,
        args: Optional[dict] = None,
    ) -> SchedulingResult:
        """
        Agent 모드용: 단일 Tool의 최적 Location 결정 (Greedy)

        ScoringEngine의 cost 공식을 그대로 사용하며,
        이전 tool의 location을 v로 적용합니다.

        TransCost 계산 (미들웨어 경유):
            - Job 시작: P^{in}(u)
            - 노드 변경 (v≠u): P^{out}(v) + P^{in}(u)
            - 노드 유지 (v==u): 0
            - Job 종료: + P^{out}(u)

        Args:
            tool_name: Tool 이름
            prev_location: 이전 tool의 location (v), None이면 is_first로 처리
            is_first: 첫 번째 tool 여부 (Job 시작 비용 추가)
            is_last: 마지막 tool 여부 (Job 종료 비용 추가)
            args: Tool 호출 인자 (local_data 처리용)

        Returns:
            SchedulingResult: 최적 location과 cost 정보
        """
        # Type C (local_data) 체크 → 노드 고정
        fixed_location = self._extract_fixed_location(tool_name, args)

        if fixed_location:
            # 노드 고정인 경우: 해당 location으로 cost 계산
            cost, comp_cost, comm_cost = self.scoring_engine.compute_cost(
                tool_name,
                u=fixed_location,
                v=prev_location,
                is_first=is_first,
                is_last=is_last,
            )
            return SchedulingResult(
                tool_name=tool_name,
                location=fixed_location,
                reason="local_data_fixed",
                score=cost,
                exec_cost=comp_cost,
                trans_cost=comm_cost,
                fixed=True,
            )

        # wasi_compatible 체크 - False면 EDGE 제외
        profile = self.registry.get_profile(tool_name)
        wasi_compatible = getattr(profile, 'wasi_compatible', True) if profile else True
        candidate_locations = [loc for loc in LOCATIONS if wasi_compatible or loc != "EDGE"]

        # 모든 location 후보에 대해 cost 계산하여 최적 선택
        best_placement = None
        best_cost = float('inf')

        for location in candidate_locations:
            cost, comp_cost, comm_cost = self.scoring_engine.compute_cost(
                tool_name,
                u=location,
                v=prev_location,
                is_first=is_first,
                is_last=is_last,
            )

            if cost < best_cost:
                best_cost = cost
                best_placement = SchedulingResult(
                    tool_name=tool_name,
                    location=location,
                    reason="brute_force_greedy",
                    score=cost,
                    exec_cost=comp_cost,
                    trans_cost=comm_cost,
                    fixed=False,
                )

        return best_placement

    def schedule_chain(
        self,
        tool_names: list[str],
        tool_args: Optional[list[dict]] = None,
    ) -> ChainSchedulingResult:
        """
        Tool Chain의 최적 노드 배치 결정 (Brute-Force)

        Args:
            tool_names: Tool 이름 리스트
            tool_args: 각 Tool의 인자 리스트 (optional, local_data 처리용)

        Returns:
            ChainSchedulingResult: 최적 배치 결과
        """
        start_time_ns = time.perf_counter_ns()
        n = len(tool_names)

        if n == 0:
            return ChainSchedulingResult(
                placements=[],
                total_score=0.0,
                search_space_size=0,
                valid_combinations=0,
                optimization_method="brute_force",
                decision_time_ns=0,
            )

        # 1. 각 Tool의 후보 노드 및 고정 이유 결정
        candidate_nodes: list[list[Location]] = []
        fixed_info: list[tuple[Optional[Location], Optional[str]]] = []

        for i, tool_name in enumerate(tool_names):
            args = tool_args[i] if tool_args and i < len(tool_args) else None

            # 고정 노드와 이유 추출
            fixed_location, fixed_reason = self._extract_fixed_location_with_reason(tool_name, args)
            fixed_info.append((fixed_location, fixed_reason))

            if fixed_location:
                candidate_nodes.append([fixed_location])
            else:
                # wasi_compatible 체크 - False면 EDGE 제외
                profile = self.registry.get_profile(tool_name)
                wasi_compatible = getattr(profile, 'wasi_compatible', True) if profile else True
                if wasi_compatible:
                    candidate_nodes.append(list(LOCATIONS))
                else:
                    candidate_nodes.append(["DEVICE", "CLOUD"])  # EDGE 제외

        # 2. 전체 탐색 공간 계산
        search_space_size = 1
        for candidates in candidate_nodes:
            search_space_size *= len(candidates)

        # 3. Brute-Force 탐색
        best_cost = float('inf')
        best_assignment: list[Location] = []
        valid_combinations = 0

        for assignment in product(*candidate_nodes):
            assignment_list = list(assignment)
            valid_combinations += 1

            cost, _ = self.scoring_engine.calculate_chain_cost(
                tool_names, assignment_list
            )

            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment_list

        # 4. 최적 배치로 상세 결과 생성
        _, placements = self.scoring_engine.calculate_chain_cost(
            tool_names, best_assignment
        )

        # fixed 플래그 및 reason 설정
        for i, placement in enumerate(placements):
            fixed_location, fixed_reason = fixed_info[i]
            if fixed_location is not None:
                placement.fixed = True
                placement.reason = fixed_reason  # "requires_gpu", "privacy_sensitive", "local_data_device" 등
            else:
                placement.fixed = False
                # 기본 reason은 "brute_force_optimal" 유지

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        return ChainSchedulingResult(
            placements=placements,
            total_score=best_cost,
            search_space_size=search_space_size,
            valid_combinations=valid_combinations,
            optimization_method="brute_force",
            decision_time_ns=decision_time_ns,
        )

    # ========================================================================
    # BaseScheduler 인터페이스 구현 (Agent 모드용)
    # ========================================================================

    def reset_state(self):
        """Agent 모드 상태 초기화 (새 세션 시작 시 호출)"""
        self._prev_location = None
        self._call_count = 0

    def get_location(self, tool_name: str) -> Location:
        """기본 location 조회 (static) - 첫 번째 tool 호출 가정"""
        placement = self.get_optimal_location_for_tool(
            tool_name=tool_name,
            prev_location=None,
            is_first=True,
            is_last=False,
        )
        return placement.location

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        """Runtime 선택 - 기본값 CONTAINER"""
        return Runtime.CONTAINER

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        """
        호출 시점 location 결정 (상태 추적 포함)

        내부적으로 prev_location을 추적하여 communication cost 계산에 반영합니다.
        """
        result = self.get_location_for_call_with_reason(tool_name, args, context)
        return result.location

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        """
        호출 시점 location 결정 (상세 정보 포함, 상태 추적)

        BruteForce Greedy 방식으로 단일 tool의 최적 location을 결정합니다.
        이전 tool의 location(prev_location)을 내부 상태로 추적하여
        communication cost를 계산합니다.
        """
        start_time_ns = time.perf_counter_ns()

        is_first = (self._call_count == 0)

        # get_optimal_location_for_tool 호출
        placement = self.get_optimal_location_for_tool(
            tool_name=tool_name,
            prev_location=self._prev_location,
            is_first=is_first,
            is_last=False,  # Agent 모드에서는 마지막인지 알 수 없음
            args=args,
        )

        # 상태 업데이트
        self._prev_location = placement.location
        self._call_count += 1

        decision_time_ns = time.perf_counter_ns() - start_time_ns

        # 결정 이유 생성
        if placement.fixed:
            reason = "local_data_fixed"
        else:
            reason = "brute_force_greedy"

        return SchedulingResult(
            tool_name=tool_name,
            location=placement.location,
            reason=reason,
            available_locations=list(LOCATIONS),
            decision_time_ns=decision_time_ns,
            score=placement.score,
            exec_cost=placement.exec_cost,
            trans_cost=placement.trans_cost,
            fixed=placement.fixed,
        )

    def __repr__(self) -> str:
        return "BruteForceChainScheduler()"


# ============================================================================
# Scheduler Registry
# ============================================================================

SCHEDULER_REGISTRY: dict[str, type[BaseScheduler]] = {
    "static": StaticScheduler,
    "all_device": AllDeviceScheduler,
    "all_edge": AllEdgeScheduler,
    "all_cloud": AllCloudScheduler,
    "heuristic": HeuristicScheduler,
    "brute_force": BruteForceChainScheduler,
}


def create_scheduler(
    name: str,
    config_path: str | Path,
    registry: ToolRegistry,
    system_config_path: Optional[str | Path] = None,
) -> BaseScheduler:
    """
    이름으로 Scheduler 생성

    Args:
        name: Scheduler 이름 (static, all_device, all_edge, all_cloud, heuristic, brute_force)
        config_path: YAML 설정 파일 경로
        registry: ToolRegistry 인스턴스
        system_config_path: System 설정 경로 (brute_force용, 없으면 config_path.parent/system.yaml)

    Returns:
        생성된 Scheduler 인스턴스

    Raises:
        ValueError: 알 수 없는 Scheduler 이름
    """
    if name not in SCHEDULER_REGISTRY:
        available = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {available}")

    scheduler_class = SCHEDULER_REGISTRY[name]

    # BruteForceChainScheduler는 system_config_path 필요
    if name == "brute_force":
        if system_config_path is None:
            system_config_path = Path(config_path).parent / "system.yaml"
        return scheduler_class(config_path, system_config_path, registry)

    return scheduler_class(config_path, registry)
