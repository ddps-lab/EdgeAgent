"""
EdgeAgent Type Definitions

기본 타입 및 상수 정의
"""

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
