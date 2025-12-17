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

from .types import Location, Runtime, LOCATIONS, ToolPlacement, ChainSchedulingResult
from .registry import ToolRegistry
from .profiles import ToolProfile
from .scoring import ScoringEngine


# ============================================================================
# Scheduling Result
# ============================================================================

@dataclass
class SchedulingResult:
    """
    Scheduler 결정 결과 (location + metadata)

    스케줄링 결정의 추적을 위해 location뿐 아니라
    결정 이유, 검사된 제약조건, 사용 가능한 위치 등을 함께 반환합니다.
    """
    location: Location                          # 결정된 실행 위치
    reason: str                                  # 결정 이유 (static_mapping, data_affinity, etc.)
    constraints_checked: list[str] = field(default_factory=list)  # 검사된 제약조건
    available_locations: list[str] = field(default_factory=list)  # 사용 가능한 위치
    decision_time_ns: int = 0                    # 결정 소요 시간 (나노초)
    # Cost 정보 (BruteForceChainScheduler)
    score: float = 0.0                           # 총 비용
    exec_cost: float = 0.0                       # 연산 비용
    trans_cost: float = 0.0                      # 통신 비용
    fixed: bool = False                          # 노드 고정 여부


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


# ============================================================================
# Static Scheduler Implementation
# ============================================================================

class StaticScheduler(BaseScheduler):
    """
    Static mapping 기반 스케줄러

    YAML 설정 파일의 static_mapping 섹션을 읽어서
    tool → location 매핑을 제공합니다.

    static_mapping 포맷:
        tool_name:
            location: DEVICE|EDGE|CLOUD (기본값: DEVICE)
            requires_cloud_api: bool
            privacy_sensitive: bool
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        """
        Args:
            config_path: YAML 설정 파일 경로
            registry: ToolRegistry 인스턴스
        """
        self.config_path = Path(config_path)
        self.registry = registry
        # {tool_name: {location, requires_cloud_api, privacy_sensitive}}
        self.static_mapping: dict[str, dict] = {}

        self._load_static_mapping()

    def _load_static_mapping(self):
        """YAML 파일에서 static_mapping 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # static_mapping 섹션 파싱
        raw_mapping = config.get("static_mapping", {})

        for tool_name, value in raw_mapping.items():
            if isinstance(value, dict):
                # 새 포맷: {location, requires_cloud_api, privacy_sensitive}
                self.static_mapping[tool_name] = {
                    "location": value.get("location", "DEVICE"),
                    "requires_cloud_api": value.get("requires_cloud_api", False),
                    "privacy_sensitive": value.get("privacy_sensitive", False),
                }
            else:
                # 이전 포맷 (backward compatibility): tool_name: LOCATION
                self.static_mapping[tool_name] = {
                    "location": value,
                    "requires_cloud_api": False,
                    "privacy_sensitive": False,
                }

    def _get_mapping(self, tool_name: str) -> Optional[dict]:
        """tool_name에 대한 static_mapping 엔트리 반환"""
        return self.static_mapping.get(tool_name)

    def get_location(self, tool_name: str) -> Location:
        """
        Tool의 실행 location 결정

        우선순위:
        1. Static mapping에 명시된 경우 → 해당 location 사용
           (requires_cloud_api, privacy_sensitive 제약 적용)
        2. 기본값: DEVICE

        Args:
            tool_name: Tool 이름

        Returns:
            결정된 location (DEVICE/EDGE/CLOUD)
        """
        mapping = self._get_mapping(tool_name)
        if mapping:
            location = mapping["location"]
            requires_cloud_api = mapping["requires_cloud_api"]
            privacy_sensitive = mapping["privacy_sensitive"]

            # Constraint 적용
            # requires_cloud_api=True → CLOUD 강제
            if requires_cloud_api and location != "CLOUD":
                return "CLOUD"
            # privacy_sensitive=True → CLOUD 불가 (DEVICE로 강제)
            if privacy_sensitive and location == "CLOUD":
                return "DEVICE"

            return location

        # Static mapping에 없으면 DEVICE 기본값
        return "DEVICE"

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        """
        Tool의 runtime 선택

        규칙:
        1. DEVICE → 항상 WASI (경량 런타임 선호)
        2. wasi_compatible = False → CONTAINER
        3. 그 외 → WASI 선호 (빠른 Cold Start)

        Args:
            tool_name: Tool 이름
            location: 실행 location

        Returns:
            선택된 runtime (WASI 또는 CONTAINER)
        """
        profile = self.registry.get_profile(tool_name)

        # DEVICE는 항상 WASI
        if location == "DEVICE":
            return "WASI"

        # Profile이 없으면 WASI 기본값
        if not profile:
            return "WASI"

        # wasi_compatible = False면 CONTAINER
        if not profile.wasi_compatible:
            return "CONTAINER"

        # 기본값: WASI (빠른 Cold Start)
        return "WASI"

    def get_location_for_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> Location:
        """
        호출 시점 location 결정

        ProxyTool에서 호출되어 실제 실행할 location을 결정합니다.

        우선순위:
        1. Static mapping에 명시된 경우 → 해당 location 사용
           (requires_cloud_api, privacy_sensitive 제약 적용)
        2. Args에서 location hint 추출 (path 기반)
        3. 기본값: DEVICE

        Args:
            tool_name: Tool 이름 (registry에 등록된 이름)
            args: Tool 호출 인자
            context: Scheduling 컨텍스트 (향후 확장용)

        Returns:
            결정된 location (DEVICE/EDGE/CLOUD)
        """
        # 1. Static mapping 최우선 확인
        mapping = self._get_mapping(tool_name)
        if mapping:
            location = mapping["location"]
            requires_cloud_api = mapping["requires_cloud_api"]
            privacy_sensitive = mapping["privacy_sensitive"]

            # Constraint 적용
            if requires_cloud_api and location != "CLOUD":
                return "CLOUD"
            if privacy_sensitive and location == "CLOUD":
                return "DEVICE"

            return location

        # 2. Args 기반 location 추론
        if args:
            location_hint = self._extract_location_from_args(tool_name, args)
            if location_hint:
                return location_hint

        # 3. 기본값: DEVICE
        return "DEVICE"

    def get_location_for_call_with_reason(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        context: SchedulingContext | None = None,
    ) -> SchedulingResult:
        """
        호출 시점 location 결정 (상세 정보 포함)

        get_location_for_call()과 동일한 로직이지만,
        결정 이유와 제약조건 정보를 함께 반환합니다.

        우선순위:
        1. Static mapping에 명시된 경우 → 해당 location 사용
           (requires_cloud_api, privacy_sensitive 제약 적용)
        2. Args에서 location hint 추출 (path 기반)
        3. 기본값: DEVICE

        Args:
            tool_name: Tool 이름 (registry에 등록된 이름)
            args: Tool 호출 인자
            context: Scheduling 컨텍스트 (향후 확장용)

        Returns:
            SchedulingResult: location과 결정 메타데이터
        """
        start_time_ns = time.perf_counter_ns()
        constraints_checked: list[str] = []
        available_locations: list[str] = list(LOCATIONS)

        # 1. Static mapping 최우선 확인
        mapping = self._get_mapping(tool_name)
        if mapping:
            location = mapping["location"]
            requires_cloud_api = mapping["requires_cloud_api"]
            privacy_sensitive = mapping["privacy_sensitive"]

            # Constraint 적용
            if requires_cloud_api:
                constraints_checked.append("requires_cloud_api")
                if location != "CLOUD":
                    decision_time_ns = time.perf_counter_ns() - start_time_ns
                    return SchedulingResult(
                        location="CLOUD",
                        reason="static_mapping_cloud_api_required",
                        constraints_checked=constraints_checked,
                        available_locations=["CLOUD"],
                        decision_time_ns=decision_time_ns,
                    )

            if privacy_sensitive:
                constraints_checked.append("privacy_sensitive")
                available_locations = ["DEVICE", "EDGE"]
                if location == "CLOUD":
                    decision_time_ns = time.perf_counter_ns() - start_time_ns
                    return SchedulingResult(
                        location="DEVICE",
                        reason="static_mapping_privacy_override",
                        constraints_checked=constraints_checked,
                        available_locations=available_locations,
                        decision_time_ns=decision_time_ns,
                    )

            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                location=location,
                reason="static_mapping",
                constraints_checked=constraints_checked,
                available_locations=available_locations,
                decision_time_ns=decision_time_ns,
            )

        # 2. Args 기반 location 추론
        if args:
            location_hint = self._extract_location_from_args(tool_name, args)
            if location_hint:
                decision_time_ns = time.perf_counter_ns() - start_time_ns
                return SchedulingResult(
                    location=location_hint,
                    reason="args_based_hint",
                    constraints_checked=constraints_checked,
                    available_locations=available_locations,
                    decision_time_ns=decision_time_ns,
                )

        # 3. 기본값: DEVICE
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            location="DEVICE",
            reason="default_device",
            constraints_checked=constraints_checked,
            available_locations=available_locations,
            decision_time_ns=decision_time_ns,
        )

    def _extract_location_from_args(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> Optional[Location]:
        """
        Tool args에서 location hint 추출

        Args:
            tool_name: Tool 이름
            args: Tool 호출 인자

        Returns:
            추출된 location hint 또는 None

        예시:
        - filesystem: path 인자에서 /device/, /edge/, /cloud/ 패턴
        - database: connection_string에서 host 분석 (향후)
        """
        # 1. 명시적 preferred_location 인자 (최우선)
        if "preferred_location" in args:
            loc = str(args["preferred_location"]).upper()
            if loc in LOCATIONS:
                return loc

        # 2. Path 기반 location 추론 (filesystem 계열)
        if "path" in args:
            path = str(args["path"])

            # 패턴 매칭: /edgeagent_device/ 또는 /device/
            if "/edgeagent_device/" in path or "/device/" in path.lower():
                return "DEVICE"
            elif "/edgeagent_edge/" in path or "/edge/" in path.lower():
                return "EDGE"
            elif "/edgeagent_cloud/" in path or "/cloud/" in path.lower():
                return "CLOUD"

        # 3. Key 기반 location 추론 (credentials 등)
        if "key" in args:
            key = str(args["key"]).lower()

            # key에 location hint가 포함된 경우
            if key.startswith("device/") or "/device/" in key:
                return "DEVICE"
            elif key.startswith("edge/") or "/edge/" in key:
                return "EDGE"
            elif key.startswith("cloud/") or "/cloud/" in key:
                return "CLOUD"

        return None

    def schedule(
        self, tool_names: list[str]
    ) -> list[tuple[str, Location, Runtime]]:
        """
        여러 tool의 placement를 일괄 결정

        Args:
            tool_names: Tool 이름 목록

        Returns:
            (tool_name, location, runtime) 튜플 리스트
        """
        placements = []

        for tool_name in tool_names:
            location = self.get_location(tool_name)
            runtime = self.select_runtime(tool_name, location)
            placements.append((tool_name, location, runtime))

        return placements

    def __repr__(self) -> str:
        return f"StaticScheduler(mappings={len(self.static_mapping)})"


# ============================================================================
# Heuristic Scheduler (Profile + Server Constraints 기반)
# ============================================================================

class HeuristicScheduler(BaseScheduler):
    """
    Profile 기반 휴리스틱 스케줄러

    Tool의 Profile 정보와 Server의 static_mapping constraints를 활용하여
    최적의 location을 결정합니다.

    결정 규칙:
    1. Type C (local_data) → path 기반 노드 고정
    2. requires_cloud_api (서버 static_mapping) → CLOUD
    3. privacy_sensitive (서버 static_mapping) → DEVICE
    4. 기본값: DEVICE
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        self.config_path = Path(config_path)
        self.registry = registry
        # {server_name: {location, requires_cloud_api, privacy_sensitive}}
        self.static_mapping: dict[str, dict] = {}
        self._load_static_mapping()

    def _load_static_mapping(self):
        """YAML 파일에서 static_mapping 로드"""
        if not self.config_path.exists():
            return

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        raw_mapping = config.get("static_mapping", {})

        for name, value in raw_mapping.items():
            if isinstance(value, dict):
                self.static_mapping[name] = {
                    "location": value.get("location", "DEVICE"),
                    "requires_cloud_api": value.get("requires_cloud_api", False),
                    "privacy_sensitive": value.get("privacy_sensitive", False),
                }
            else:
                self.static_mapping[name] = {
                    "location": value,
                    "requires_cloud_api": False,
                    "privacy_sensitive": False,
                }

    def _get_server_constraints(self, tool_name: str) -> dict:
        """
        Tool이 속한 Server의 static_mapping constraints 반환

        Args:
            tool_name: Tool 이름

        Returns:
            {requires_cloud_api, privacy_sensitive, location} dict
        """
        server_name = self.registry.get_server_for_tool(tool_name)
        if not server_name:
            server_name = tool_name

        mapping = self.static_mapping.get(server_name)
        if mapping:
            return mapping

        return {
            "requires_cloud_api": False,
            "privacy_sensitive": False,
            "location": "DEVICE",
        }

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
        1. requires_cloud_api (서버 static_mapping) → CLOUD
        2. privacy_sensitive (서버 static_mapping) → DEVICE
        3. 서버의 static_mapping location 사용
        """
        constraints = self._get_server_constraints(tool_name)

        if constraints["requires_cloud_api"]:
            return "CLOUD"

        if constraints["privacy_sensitive"]:
            return "DEVICE"

        return constraints["location"]

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
        2. requires_cloud_api (서버 static_mapping) → CLOUD
        3. privacy_sensitive (서버 static_mapping) → DEVICE
        4. 서버의 static_mapping location 사용
        """
        # 1. Type C (local_data) 체크 → 노드 고정
        fixed_location = self._extract_fixed_location(tool_name, args)
        if fixed_location:
            return fixed_location

        # 2, 3. Server constraints
        constraints = self._get_server_constraints(tool_name)

        if constraints["requires_cloud_api"]:
            return "CLOUD"

        if constraints["privacy_sensitive"]:
            return "DEVICE"

        # 4. 서버의 static_mapping location 사용
        return constraints["location"]

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
        constraints_checked: list[str] = []

        # 1. Type C (local_data) 체크 → 노드 고정
        fixed_location = self._extract_fixed_location(tool_name, args)
        if fixed_location:
            constraints_checked.append("local_data")
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                location=fixed_location,
                reason="local_data_fixed",
                constraints_checked=constraints_checked,
                available_locations=[fixed_location],
                decision_time_ns=decision_time_ns,
                fixed=True,
            )

        # 2, 3. Server constraints
        constraints = self._get_server_constraints(tool_name)

        if constraints["requires_cloud_api"]:
            constraints_checked.append("requires_cloud_api")
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                location="CLOUD",
                reason="cloud_api_required",
                constraints_checked=constraints_checked,
                available_locations=["CLOUD"],
                decision_time_ns=decision_time_ns,
            )

        if constraints["privacy_sensitive"]:
            constraints_checked.append("privacy_sensitive")
            decision_time_ns = time.perf_counter_ns() - start_time_ns
            return SchedulingResult(
                location="DEVICE",
                reason="privacy_constraint",
                constraints_checked=constraints_checked,
                available_locations=["DEVICE", "EDGE"],
                decision_time_ns=decision_time_ns,
            )

        # 4. 서버의 static_mapping location 사용
        decision_time_ns = time.perf_counter_ns() - start_time_ns
        return SchedulingResult(
            location=constraints["location"],
            reason="server_mapping",
            constraints_checked=constraints_checked,
            available_locations=list(LOCATIONS),
            decision_time_ns=decision_time_ns,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)

        if location == "DEVICE":
            return "WASI"

        if profile and not profile.wasi_compatible:
            return "CONTAINER"

        return "WASI"

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
            location="DEVICE",
            reason="all_device_policy",
            constraints_checked=[],
            available_locations=["DEVICE"],
            decision_time_ns=decision_time_ns,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["DEVICE"]

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
            location="EDGE",
            reason="all_edge_policy",
            constraints_checked=[],
            available_locations=["EDGE"],
            decision_time_ns=decision_time_ns,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)
        if profile and not profile.wasi_compatible:
            return "CONTAINER"
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["EDGE"]

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
            location="CLOUD",
            reason="all_cloud_policy",
            constraints_checked=[],
            available_locations=["CLOUD"],
            decision_time_ns=decision_time_ns,
        )

    def select_runtime(self, tool_name: str, location: Location) -> Runtime:
        profile = self.registry.get_profile(tool_name)
        if profile and not profile.wasi_compatible:
            return "CONTAINER"
        return "WASI"

    def get_required_locations(self) -> list[Location]:
        return ["CLOUD"]

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
        subagent_mode: bool = True,
    ):
        """
        Args:
            config_path: tools_scenario*.yaml 경로
            system_config_path: system.yaml 경로
            registry: ToolRegistry 인스턴스
            subagent_mode: True면 SubAgent 직접 통신, False면 middleware 경유 모델
        """
        self.config_path = Path(config_path)
        self.system_config_path = Path(system_config_path)
        self.registry = registry
        self.subagent_mode = subagent_mode
        self.scoring_engine = ScoringEngine(system_config_path, registry, subagent_mode)

        # Agent 모드용 상태 (prev_location 추적)
        self._prev_location: Optional[Location] = None
        self._call_count: int = 0

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
            args: Tool 호출 인자 (없으면 DEVICE 기본값)

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

    def get_optimal_location_for_tool(
        self,
        tool_name: str,
        prev_location: Optional[Location] = None,
        is_first: bool = False,
        is_last: bool = False,
        args: Optional[dict] = None,
    ) -> ToolPlacement:
        """
        Agent 모드용: 단일 Tool의 최적 Location 결정 (Greedy)

        ScoringEngine의 cost 공식을 그대로 사용하며,
        이전 tool의 location을 v로 적용합니다.

        Cost 계산 (subagent_mode에 따라):
          subagent_mode=True (SubAgent 직접 통신):
            - Job 시작: P_comm[(D, u)]
            - 노드 변경: P_comm[(v, u)]
            - Job 종료: + P_comm[(u, D)]
          subagent_mode=False (middleware 경유):
            - Job 시작: P^{in}(u)
            - 노드 변경: P^{out}(v) + P^{in}(u)
            - Job 종료: + P^{out}(u)

        Args:
            tool_name: Tool 이름
            prev_location: 이전 tool의 location (v), None이면 is_first로 처리
            is_first: 첫 번째 tool 여부 (Job 시작 비용 추가)
            is_last: 마지막 tool 여부 (Job 종료 비용 추가)
            args: Tool 호출 인자 (local_data 처리용)

        Returns:
            ToolPlacement: 최적 location과 cost 정보
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
            return ToolPlacement(
                tool_name=tool_name,
                location=fixed_location,
                score=cost,
                exec_cost=comp_cost,
                trans_cost=comm_cost,
                fixed=True,
            )

        # 모든 location 후보에 대해 cost 계산하여 최적 선택
        best_placement = None
        best_cost = float('inf')

        for location in LOCATIONS:
            cost, comp_cost, comm_cost = self.scoring_engine.compute_cost(
                tool_name,
                u=location,
                v=prev_location,
                is_first=is_first,
                is_last=is_last,
            )

            if cost < best_cost:
                best_cost = cost
                best_placement = ToolPlacement(
                    tool_name=tool_name,
                    location=location,
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

        # 1. 각 Tool의 후보 노드 결정
        candidate_nodes: list[list[Location]] = []
        fixed_nodes: list[Optional[Location]] = []

        for i, tool_name in enumerate(tool_names):
            args = tool_args[i] if tool_args and i < len(tool_args) else None

            # Type C (local_data) 체크 → 노드 고정
            fixed = self._extract_fixed_location(tool_name, args)
            fixed_nodes.append(fixed)

            if fixed:
                candidate_nodes.append([fixed])
            else:
                candidate_nodes.append(list(LOCATIONS))

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

        # fixed 플래그 설정
        for i, placement in enumerate(placements):
            placement.fixed = (fixed_nodes[i] is not None)

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
            location=placement.location,
            reason=reason,
            constraints_checked=["data_locality", "compute_cost", "comm_cost"],
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
}


def create_scheduler(
    name: str,
    config_path: str | Path,
    registry: ToolRegistry,
) -> BaseScheduler:
    """
    이름으로 Scheduler 생성

    Args:
        name: Scheduler 이름 (static, all_device, all_edge, all_cloud, heuristic)
        config_path: YAML 설정 파일 경로
        registry: ToolRegistry 인스턴스

    Returns:
        생성된 Scheduler 인스턴스

    Raises:
        ValueError: 알 수 없는 Scheduler 이름
    """
    if name not in SCHEDULER_REGISTRY:
        available = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {available}")

    scheduler_class = SCHEDULER_REGISTRY[name]
    return scheduler_class(config_path, registry)
