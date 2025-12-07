"""
Scheduler Module

Tool 실행 location 결정을 위한 스케줄러 구현

- BaseScheduler: 추상 인터페이스 (향후 Heuristic/DP Scheduler로 교체 가능)
- StaticScheduler: YAML static_mapping + args 기반 동적 routing
"""

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from .types import Location, Runtime, LOCATIONS
from .registry import ToolRegistry
from .profiles import ToolProfile


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


# ============================================================================
# Static Scheduler Implementation
# ============================================================================

class StaticScheduler(BaseScheduler):
    """
    Static mapping 기반 스케줄러

    YAML 설정 파일의 static_mapping 섹션을 읽어서
    tool → location 매핑을 제공합니다.

    향후 Heuristic, DP 알고리즘으로 확장 예정.
    """

    def __init__(self, config_path: str | Path, registry: ToolRegistry):
        """
        Args:
            config_path: YAML 설정 파일 경로
            registry: ToolRegistry 인스턴스
        """
        self.config_path = Path(config_path)
        self.registry = registry
        self.static_mapping: dict[str, Location] = {}

        self._load_static_mapping()

    def _load_static_mapping(self):
        """YAML 파일에서 static_mapping 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # static_mapping 섹션 파싱
        self.static_mapping = config.get("static_mapping", {})

    def get_location(self, tool_name: str) -> Location:
        """
        Tool의 실행 location 결정

        우선순위:
        1. Static mapping에 명시된 경우 → 해당 location 사용
        2. Constraint 검증 후 유효한 location 선택
        3. Profile의 data_affinity를 기본값으로 사용

        Args:
            tool_name: Tool 이름

        Returns:
            결정된 location (DEVICE/EDGE/CLOUD)
        """
        # 1. Static mapping 확인
        if tool_name in self.static_mapping:
            mapped_location = self.static_mapping[tool_name]

            # Constraint 검증
            profile = self.registry.get_profile(tool_name)
            if profile and self._is_valid_location(profile, mapped_location):
                return mapped_location
            else:
                print(
                    f"Warning: Static mapping for '{tool_name}' → '{mapped_location}' "
                    f"violates constraints. Falling back to auto-selection."
                )

        # 2. Profile 기반 location 선택
        profile = self.registry.get_profile(tool_name)
        if not profile:
            # Profile이 없으면 EDGE 기본값
            return "EDGE"

        # Constraint 기반 유효한 location 찾기
        valid_locations = self._get_valid_locations(profile)

        if not valid_locations:
            # 유효한 location이 없으면 EDGE 기본값
            print(
                f"Warning: No valid location for '{tool_name}'. Using EDGE as fallback."
            )
            return "EDGE"

        # 우선순위: data_affinity > 첫 번째 유효 location
        if profile.data_affinity in valid_locations:
            return profile.data_affinity
        else:
            return valid_locations[0]

    def _get_valid_locations(self, profile: ToolProfile) -> list[Location]:
        """
        Constraint를 만족하는 유효한 location 목록

        제약사항:
        1. requires_cloud_api = True → CLOUD만 가능
        2. privacy_sensitive = True → CLOUD 불가
        3. data_affinity = DEVICE → DEVICE 또는 EDGE 선호
        """
        valid = set(LOCATIONS)

        # Constraint 1: Cloud API 필요
        if profile.requires_cloud_api:
            valid = {"CLOUD"}

        # Constraint 2: Privacy 민감
        if profile.privacy_sensitive:
            valid.discard("CLOUD")

        # Constraint 3: Data affinity
        # (선호도이므로 강제하지 않음, get_location에서 우선순위 적용)

        return list(valid) if valid else ["EDGE"]

    def _is_valid_location(self, profile: ToolProfile, location: Location) -> bool:
        """특정 location이 constraint를 만족하는지 검증"""
        valid_locations = self._get_valid_locations(profile)
        return location in valid_locations

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
        1. Hard Constraints (requires_cloud_api, privacy_sensitive)
        2. Args에서 location hint 추출 (path 기반)
        3. Static mapping
        4. Profile data_affinity

        Args:
            tool_name: Tool 이름 (registry에 등록된 이름)
            args: Tool 호출 인자
            context: Scheduling 컨텍스트 (향후 확장용)

        Returns:
            결정된 location (DEVICE/EDGE/CLOUD)
        """
        profile = self.registry.get_profile(tool_name)

        # 1. Hard Constraints 체크
        if profile:
            # Cloud API 필수 → 무조건 CLOUD
            if profile.requires_cloud_api:
                return "CLOUD"

            # Privacy 민감 → CLOUD 제외 (args에서 DEVICE/EDGE 결정)
            # (아래 args 기반 routing에서 처리)

        # 2. Args 기반 location 추론
        if args:
            location_hint = self._extract_location_from_args(tool_name, args)
            if location_hint:
                # Privacy 민감한 경우 CLOUD hint는 무시
                if profile and profile.privacy_sensitive and location_hint == "CLOUD":
                    pass  # fallback to static mapping
                else:
                    return location_hint

        # 3. Static mapping / Profile 기반 fallback
        return self.get_location(tool_name)

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
