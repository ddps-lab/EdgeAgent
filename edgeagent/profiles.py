"""
Tool Profile 정의

4D Taxonomy 기반 Tool 특성 프로파일 및 Endpoint 설정
"""

from dataclasses import dataclass, field
from typing import Optional

from .types import (
    Location,
    Runtime,
    TransportType,
    DataLocality,
)


# ============================================================================
# Network & Tool Measurements (실측 기반 동적 계산)
# ============================================================================

@dataclass
class NetworkMeasurements:
    """
    네트워크 측정값 (system.yaml에서 로드, 티어별)

    T_comm = (datasize_mb * 8 / bandwidth_mbps * 1000) + latency_ms
    """
    bandwidth_mbps: dict[str, float]  # {"D_E": 944.44, "D_C": 325.66}
    latency_ms: dict[str, float]      # {"D_E": 0.500, "D_C": 4.329}

    def get_t_comm(self, path: str, datasize_mb: float) -> float:
        """
        T_comm 계산 (ms)

        Args:
            path: "D_E", "D_C"
            datasize_mb: 데이터 크기 (MB, 십진)
        """
        bw = self.bandwidth_mbps.get(path, 100.0)
        lat = self.latency_ms.get(path, 1.0)
        return (datasize_mb * 8 / bw * 1000) + lat

    @classmethod
    def from_config(cls, config: dict) -> "NetworkMeasurements":
        """system.yaml의 network_measurements에서 로드"""
        nm = config.get("network_measurements", {})
        return cls(
            bandwidth_mbps=nm.get("bandwidth_mbps", {}),
            latency_ms=nm.get("latency_ms", {}),
        )


@dataclass
class ToolMeasurements:
    """
    Tool별 실측값 (tools_scenario*.yaml에서 로드)

    α = T_exec / (T_exec + T_comm_in + T_comm_out)
    """
    input_datasize_mb: float    # 입력 데이터 크기 (MB)
    output_datasize_mb: float   # 출력 데이터 크기 (MB)
    t_exec_ms: dict[str, float] # {"device-rpi": 2.1, "edge-nuc": 0.57, ...}

    def get_tier_t_exec(self, hardware_mapping: dict) -> dict[str, float]:
        """
        하드웨어 값을 tier별로 가져오기 (대표 하드웨어)

        Args:
            hardware_mapping: {"DEVICE": ["device-rpi"], "EDGE": ["edge-orin"], "CLOUD": ["cloud-c7i48xlarge"]}

        Returns:
            {"DEVICE": 44516.523, "EDGE": 22237.747, "CLOUD": 12170.633}
        """
        result = {}
        for tier, hw_list in hardware_mapping.items():
            # 대표 하드웨어 (첫 번째)의 값 사용
            hw = hw_list[0] if hw_list else None
            result[tier] = self.t_exec_ms.get(hw, 0.0) if hw else 0.0
        return result

    def calculate_alpha(self, network: NetworkMeasurements,
                        hardware_mapping: dict) -> float:
        """
        α = T_exec / (T_exec + T_comm_in + T_comm_out)

        대표 엣지의 T_exec와 D_E 경로 기준으로 계산

        Args:
            network: 네트워크 측정값
            hardware_mapping: 티어 → 하드웨어 매핑
        """
        # EDGE 티어의 대표 하드웨어
        edge_hw = hardware_mapping.get("EDGE", ["edge-orin"])[0]
        t_exec = self.t_exec_ms.get(edge_hw, 0.0)
        t_comm_in = network.get_t_comm("D_E", self.input_datasize_mb)
        t_comm_out = network.get_t_comm("D_E", self.output_datasize_mb)
        return t_exec / (t_exec + t_comm_in + t_comm_out) if (t_exec + t_comm_in + t_comm_out) > 0 else 0.5

    def calculate_p_exec(self, hardware_mapping: dict) -> list[float]:
        """
        정규화된 P_exec [DEVICE, EDGE, CLOUD]
        높은 T_exec = 높은 비용 (느림)

        Args:
            hardware_mapping: 티어 → 하드웨어 매핑
        """
        tier_t_exec = self.get_tier_t_exec(hardware_mapping)
        max_t = max(tier_t_exec.values()) if tier_t_exec.values() else 1.0
        if max_t == 0:
            return [1.0, 1.0, 1.0]  # fallback
        return [tier_t_exec.get(loc, max_t) / max_t for loc in ["DEVICE", "EDGE", "CLOUD"]]


# ============================================================================
# Tool Profile (4D Taxonomy)
# ============================================================================

@dataclass
class ToolProfile:
    """
    MCP Tool의 4D 특성 프로파일

    연구 계획서의 4D Taxonomy에 기반:
    1. Data Affinity: Tool이 접근하는 데이터 위치
    2. Compute Intensity: 연산 복잡도
    3. Data Flow: 입출력 데이터 크기 비율
    4. Constraints: 실행 제약사항
    """

    # 기본 정보
    name: str
    """Tool 이름"""

    description: str = ""
    """Tool 설명"""

    # ========================================================================
    # Dimension 1: Data Affinity
    # ========================================================================
    data_affinity: str = "EDGE"
    """
    Tool이 주로 접근하는 데이터의 위치

    예시:
    - DEVICE: filesystem, local_db
    - EDGE: log_parser, image_resize
    - CLOUD: slack_send, google_search
    """

    # ========================================================================
    # Dimension 2: Compute Intensity
    # ========================================================================
    compute_intensity: str = "LOW"
    """
    Tool의 연산 복잡도

    예시:
    - LOW: read_file, list_directory
    - MEDIUM: parse_json, filter_logs
    - HIGH: ml_inference, browser_screenshot
    """

    # ========================================================================
    # Dimension 3: Data Flow
    # ========================================================================
    data_flow: str = "TRANSFORM"
    """
    Tool의 입출력 데이터 크기 비율

    예시:
    - REDUCTION: search (1GB → 10KB), filter (100MB → 1MB)
    - TRANSFORM: convert_format (1MB → 1MB)
    - EXPANSION: read_file (path → content), fetch_url (URL → HTML)
    """

    reduction_ratio: float = 1.0
    """
    Output size / Input size 비율

    예시:
    - 0.01: REDUCTION (1GB → 10MB)
    - 1.0: TRANSFORM (동일 크기)
    - 100.0: EXPANSION (10KB → 1MB)
    """

    # ========================================================================
    # Dimension 4: Constraints
    # ========================================================================
    privacy_sensitive: bool = False
    """
    Privacy 민감 여부

    True인 경우:
    - CLOUD로 데이터 전송 불가
    - DEVICE 또는 EDGE에서만 실행
    - 예: filesystem, credential_manager
    """

    wasi_compatible: bool = True
    """
    WASI 런타임 호환 여부

    False인 경우:
    - CONTAINER 런타임만 사용 가능
    - 예: puppeteer, postgres, gpu_inference
    """

    requires_gpu: bool = False
    """
    GPU 필요 여부

    True인 경우:
    - GPU가 있는 위치에서만 실행
    - 예: ml_inference, image_generation
    """


# ============================================================================
# Endpoint Configuration
# ============================================================================

@dataclass
class EndpointConfig:
    """
    특정 location의 MCP 서버 endpoint 설정
    """

    location: Location
    """Endpoint의 위치 (DEVICE/EDGE/CLOUD)"""

    transport: TransportType
    """연결 방식 (stdio 또는 streamable_http)"""

    # stdio 설정
    command: Optional[str] = None
    """stdio: 실행할 명령어 (예: npx, python)"""

    args: list[str] = field(default_factory=list)
    """stdio: 명령어 인자"""

    env: dict[str, str] = field(default_factory=dict)
    """환경 변수"""

    # streamable_http 설정
    url: Optional[str] = None
    """streamable_http: MCP 서버 URL"""

    def __post_init__(self):
        """유효성 검증"""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"stdio transport requires 'command' for {self.location}"
                )
        elif self.transport == "streamable_http":
            if not self.url:
                raise ValueError(
                    f"streamable_http transport requires 'url' for {self.location}"
                )

    def to_mcp_config(self) -> dict:
        """
        langchain-mcp-adapters의 MultiServerMCPClient 설정 형식으로 변환
        """
        if self.transport == "stdio":
            return {
                "transport": "stdio",  # transport 키 추가
                "command": self.command,
                "args": self.args,
                "env": self.env,
            }
        else:  # streamable_http
            # Note: langchain-mcp-adapters doesn't support 'env' for streamable_http
            return {
                "transport": "streamable_http",
                "url": self.url,
            }


# ============================================================================
# Multi-Endpoint Tool Configuration
# ============================================================================

@dataclass
class ToolConfig:
    """
    Tool의 전체 설정 (Profile + Multi-endpoint)
    """

    name: str
    """Tool 이름"""

    profile: ToolProfile
    """4D 특성 프로파일"""

    endpoints: dict[Location, EndpointConfig] = field(default_factory=dict)
    """
    Location별 endpoint 설정

    예시:
    {
        "DEVICE": EndpointConfig(location="DEVICE", transport="stdio", ...),
        "EDGE": EndpointConfig(location="EDGE", transport="streamable_http", ...),
        "CLOUD": EndpointConfig(location="CLOUD", transport="streamable_http", ...)
    }
    """

    def get_endpoint(self, location: Location) -> Optional[EndpointConfig]:
        """특정 location의 endpoint 가져오기"""
        return self.endpoints.get(location)

    def available_locations(self) -> list[Location]:
        """사용 가능한 location 목록"""
        return list(self.endpoints.keys())

    def has_location(self, location: Location) -> bool:
        """특정 location에 endpoint가 있는지 확인"""
        return location in self.endpoints
