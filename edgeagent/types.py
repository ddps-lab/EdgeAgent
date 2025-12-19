"""
EdgeAgent Type Definitions

기본 타입 및 상수 정의
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# ============================================================================
# Location Types
# ============================================================================

Location = Literal["DEVICE", "EDGE", "CLOUD"]
"""
Tool이 실행될 수 있는 위치

- DEVICE: 로컬 디바이스 (로컬 파일시스템, 센서 접근)
- EDGE: Edge 서버 (중간 처리, 데이터 전처리)
- CLOUD: Cloud 서버 (외부 API, 무거운 연산)
"""

LOCATIONS: list[Location] = ["DEVICE", "EDGE", "CLOUD"]


# ============================================================================
# Runtime Types
# ============================================================================

Runtime = Literal["WASI", "CONTAINER"]
"""
Tool 실행 런타임

- WASI: WebAssembly System Interface (빠른 Cold Start, 경량)
- CONTAINER: Docker/Kubernetes Container (호환성 높음, Cold Start 느림)
"""

RUNTIMES: list[Runtime] = ["WASI", "CONTAINER"]


# ============================================================================
# Scheduling Constraints
# ============================================================================

@dataclass
class SchedulingConstraints:
    """
    스케줄링 제약조건 결과

    각 Tool 호출에 대해 검사된 제약조건 결과를 저장합니다.
    """
    requires_cloud_api: bool = False    # Cloud API 필요 여부
    privacy_sensitive: bool = False     # 프라이버시 민감 여부
    tool_available: bool = True         # 해당 위치에서 툴 실행 가능 여부 (TODO: 스케줄러에서 로직 구현)


# ============================================================================
# Transport Types
# ============================================================================

TransportType = Literal["stdio", "streamable_http"]
"""
MCP 서버 연결 방식

- stdio: 로컬 프로세스 (stdin/stdout)
- streamable_http: HTTP/SSE 기반 원격 연결
"""


# ============================================================================
# Data Locality Types (Score-based Scheduling)
# ============================================================================

DataLocality = Literal["args_only", "external_data", "local_data"]
"""
Tool의 데이터 접근 유형

- args_only (Type A): 데이터가 args로 전달됨 → Score 기반 노드 선택
- external_data (Type B): 외부 API 접근 → Score 기반 + β 패널티
- local_data (Type C): 로컬 파일 접근 → 경로 기반 노드 결정 (Score 무시)
"""


# ============================================================================
# Scheduling Result
# ============================================================================

@dataclass
class SchedulingResult:
    """
    Scheduler 결정 결과 (location + metadata)

    스케줄링 결정의 추적을 위해 location뿐 아니라
    결정 이유, 검사된 제약조건, 사용 가능한 위치 등을 함께 반환합니다.

    - agent 모드: 개별 tool 호출마다 생성
    - script/subagent 모드: schedule_chain()에서 ChainSchedulingResult.placements로 반환
    """
    tool_name: str                               # Tool 이름
    location: Location                           # 결정된 실행 위치
    reason: str                                  # 결정 이유 (static_mapping, brute_force_optimal, etc.)
    decision_time_ns: int = 0                    # 결정 소요 시간 (나노초)
    available_locations: list[str] = field(default_factory=list)  # 사용 가능한 위치
    # 제약조건 결과
    constraints: SchedulingConstraints = field(default_factory=SchedulingConstraints)
    # Cost 정보 (brute_force만 값 설정, 나머지 None)
    score: Optional[float] = None                # 총 비용
    exec_cost: Optional[float] = None            # 연산 비용
    trans_cost: Optional[float] = None           # 통신 비용
    fixed: Optional[bool] = None                 # 노드 고정 여부


# ============================================================================
# Chain Scheduling Types
# ============================================================================

@dataclass
class ChainSchedulingResult:
    """
    Tool Chain 전체의 스케줄링 결과

    schedule_chain() 메서드의 반환 타입.
    brute_force만 total_score, search_space_size, valid_combinations 값 설정.
    """
    placements: list[SchedulingResult]              # 각 Tool의 스케줄링 결과
    optimization_method: str                        # 스케줄러 이름
    decision_time_ns: int                           # 전체 chain 스케줄링 소요 시간
    total_score: Optional[float] = None             # 전체 비용 (brute_force만)
    search_space_size: Optional[int] = None         # 탐색 공간 크기 (brute_force만)
    valid_combinations: Optional[int] = None        # 유효 조합 수 (brute_force만)

    def get_location(self, tool_name: str) -> Location:
        """특정 Tool의 배치된 노드 반환"""
        for p in self.placements:
            if p.tool_name == tool_name:
                return p.location
        raise KeyError(f"Tool not found: {tool_name}")

    def print_chain(self):
        """Chain 배치 출력 (* = 고정 노드)"""
        parts = []
        for p in self.placements:
            prefix = "*" if p.fixed else ""
            parts.append(f"{prefix}{p.location[0]}")
        print(" → ".join(parts))

    def print_details(self):
        """상세 배치 출력"""
        for p in self.placements:
            fixed_mark = " [fixed]" if p.fixed else ""
            score = p.score if p.score is not None else 0.0
            exec_cost = p.exec_cost if p.exec_cost is not None else 0.0
            trans_cost = p.trans_cost if p.trans_cost is not None else 0.0
            print(f"  {p.tool_name} @ {p.location} "
                  f"(score: {score:.2f}, exec: {exec_cost:.2f}, trans: {trans_cost:.2f}){fixed_mark}")
