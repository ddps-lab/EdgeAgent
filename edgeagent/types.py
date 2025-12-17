"""
EdgeAgent Type Definitions

기본 타입 및 상수 정의
"""

from dataclasses import dataclass, field
from typing import Literal

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
# 4D Taxonomy Types
# ============================================================================

DataAffinity = Literal["DEVICE", "EDGE", "CLOUD"]
"""
Dimension 1: Data Affinity

Tool이 어떤 데이터에 접근하는지에 따른 분류

- DEVICE: 로컬 파일시스템, 센서, 로컬 DB 접근 필수
- EDGE: 지역 데이터 처리, 전처리/변환 작업
- CLOUD: 외부 API 의존, 글로벌 데이터 접근
"""

ComputeIntensity = Literal["LOW", "MEDIUM", "HIGH"]
"""
Dimension 2: Compute Intensity

Tool의 연산 복잡도

- LOW: 단순 I/O, CRUD 작업 (예: read_file, get_time)
- MEDIUM: 파싱, 변환, 집계 (예: parse_json, aggregate_data)
- HIGH: ML 추론, Browser 렌더링 (예: embedding, screenshot)
"""

DataFlow = Literal["REDUCTION", "TRANSFORM", "EXPANSION"]
"""
Dimension 3: Data Flow

Tool의 입출력 데이터 크기 비율

- REDUCTION: Output << Input (예: search, filter, summarize)
  → Edge에서 실행하면 네트워크 전송량 감소
- TRANSFORM: Output ≈ Input (예: convert, format, translate)
  → 위치 선택이 덜 중요
- EXPANSION: Output >> Input (예: read_file, fetch, download)
  → Data source 가까운 곳에서 실행
"""


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
# Chain Scheduling Types
# ============================================================================

@dataclass
class ToolPlacement:
    """
    단일 Tool의 배치 정보

    Score(i, u, v) = α * { ExecCost(i, u) + β } + (1-α) * TransCost(v → u)
    """
    tool_name: str
    location: Location
    score: float
    exec_cost: float
    trans_cost: float
    fixed: bool = False  # Type C (local_data)로 고정된 경우


@dataclass
class ChainSchedulingResult:
    """
    Tool Chain 전체의 스케줄링 결과

    Brute-force 완전 탐색으로 최적 노드 배치 조합을 찾은 결과
    """
    placements: list[ToolPlacement]
    total_score: float
    search_space_size: int
    valid_combinations: int
    optimization_method: str  # "brute_force"
    decision_time_ns: int

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
            print(f"  {p.tool_name} @ {p.location} "
                  f"(score: {p.score:.2f}, exec: {p.exec_cost:.2f}, trans: {p.trans_cost:.2f}){fixed_mark}")
