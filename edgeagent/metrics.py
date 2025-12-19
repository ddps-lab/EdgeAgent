"""
EdgeAgent Metrics Collection Framework

통합 메트릭 수집 시스템 - 논문 작성을 위한 상세 메트릭 수집

Metrics Categories:
1. Latency - 도구 실행 시간, 파이프라인 시간
2. Data Flow - 입출력 크기, 감소율
3. Location Routing - 스케줄링 결정, 위치 분포
4. MCP Protocol - 직렬화/역직렬화 오버헤드
5. Resource - 메모리, CPU 사용량
6. Pipeline - 단계, 병렬 처리, 재시도
"""

import json
import os
import resource
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ChainSchedulingResult:
    """
    Chain Scheduling 결과 메트릭

    BruteForceChainScheduler의 결과를 저장합니다.
    """
    total_cost: float = 0.0                    # 전체 Tool Chain의 최적 비용
    search_space_size: int = 0                 # 탐색 공간 크기 (조합 수)
    valid_combinations: int = 0                # 유효한 조합 수
    decision_time_ms: float = 0.0              # 스케줄링 결정 시간 (ms)
    optimization_method: str = "brute_force"   # 최적화 방법
    tool_chain: list[str] = field(default_factory=list)  # Tool Chain 목록
    placements: list[dict] = field(default_factory=list)  # 각 Tool의 배치 결과

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_cost": self.total_cost,
            "search_space_size": self.search_space_size,
            "valid_combinations": self.valid_combinations,
            "decision_time_ms": self.decision_time_ms,
            "optimization_method": self.optimization_method,
            "tool_chain": self.tool_chain,
            "placements": self.placements,
        }


@dataclass
class ScenarioMetrics:
    """
    시나리오별 커스텀 메트릭

    각 시나리오에서 수집하는 도메인 특화 메트릭
    """
    # 공통
    data_source: str = ""                      # 데이터 소스 (예: "LogHub", "COCO 2017")
    input_items_count: int = 0                 # 입력 항목 수
    output_items_count: int = 0                # 출력 항목 수

    # Scenario별 특화 메트릭 (optional)
    # Log Analysis
    entries_parsed: int = 0
    entries_filtered: int = 0

    # Research Assistant
    articles_fetched: int = 0
    summaries_generated: int = 0

    # Image Processing
    images_found: int = 0
    unique_images: int = 0
    duplicate_groups: int = 0
    thumbnails_created: int = 0

    # Code Review
    commit_count: int = 0
    diff_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if value:  # 0이 아닌 값만 포함
                result[key] = value
        return result


@dataclass
class MetricEntry:
    """
    단일 도구 호출의 메트릭 (Entry 수준)

    모든 메트릭을 수집하여 나중에 분석에 사용
    """

    # === Identity ===
    tool_name: str  # MCP tool name (e.g., "read_file")
    parent_tool_name: str  # Registry tool name (e.g., "filesystem")
    call_id: str  # Unique identifier
    pipeline_step: int  # Step number in pipeline (1, 2, 3...)

    # === Timing ===
    timestamp: float  # Unix timestamp
    latency_ms: float  # Tool execution time
    inter_tool_latency_ms: float = 0.0  # Time since previous tool ended

    # === LLM Latency (Agent 모드 전용) ===
    llm_latency_ms: float = 0.0  # LLM 추론 시간 (이 tool 호출 결정까지)
    llm_input_tokens: int = 0  # LLM 입력 토큰 수
    llm_output_tokens: int = 0  # LLM 출력 토큰 수

    # === Location & Routing ===
    scheduled_location: str = ""  # What scheduler decided
    actual_location: str = ""  # Where it actually ran
    fallback_occurred: bool = False  # Did fallback happen?

    # === Scheduling Decision ===
    scheduling_decision_time_ns: int = 0  # Time to make scheduling decision
    scheduling_reason: str = ""  # Why this location
    constraints_checked: list[str] = field(default_factory=list)
    available_locations: list[str] = field(default_factory=list)

    # === Scheduling Cost (BruteForceChainScheduler) ===
    scheduling_score: float = 0.0  # Total cost
    exec_cost: float = 0.0  # Computation cost
    trans_cost: float = 0.0  # Communication cost
    fixed_location: bool = False  # Node fixed (data_locality)

    # === Data Flow ===
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    reduction_ratio: float = 0.0  # output/input
    data_flow_type: str = ""  # EXPANSION, REDUCTION, TRANSFORM
    expected_reduction_ratio: float = 0.0  # From tool profile

    # === MCP Protocol ===
    mcp_serialization_time_ms: float = 0.0
    mcp_deserialization_time_ms: float = 0.0
    mcp_request_size_bytes: int = 0
    mcp_response_size_bytes: int = 0

    # === Resource ===
    memory_before_bytes: int = 0
    memory_after_bytes: int = 0
    memory_delta_bytes: int = 0
    cpu_time_user_ms: float = 0.0
    cpu_time_system_ms: float = 0.0

    # === Status ===
    success: bool = True
    retry_count: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None

    # === Network Traffic ===
    network_type: str = ""  # "local", "lan", "wan"
    request_bytes: int = 0  # client → tool
    response_bytes: int = 0  # tool → client

    # === Extensible ===
    args_keys: list[str] = field(default_factory=list)
    custom_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "tool_name": self.tool_name,
            "parent_tool_name": self.parent_tool_name,
            "call_id": self.call_id,
            "pipeline_step": self.pipeline_step,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "timing": {
                "latency_ms": self.latency_ms,
                "inter_tool_latency_ms": self.inter_tool_latency_ms,
                "llm_latency_ms": self.llm_latency_ms,
                "total_latency_ms": self.llm_latency_ms + self.latency_ms,  # LLM + Tool 실행 시간
                "mcp_serialization_time_ms": self.mcp_serialization_time_ms,
                "mcp_deserialization_time_ms": self.mcp_deserialization_time_ms,
            },
            "llm": {
                "input_tokens": self.llm_input_tokens,
                "output_tokens": self.llm_output_tokens,
            },
            "location": {
                "scheduled_location": self.scheduled_location,
                "actual_location": self.actual_location,
                "fallback_occurred": self.fallback_occurred,
            },
            "scheduling": {
                "decision_time_ns": self.scheduling_decision_time_ns,
                "reason": self.scheduling_reason,
                "constraints_checked": self.constraints_checked,
                "available_locations": self.available_locations,
                "score": self.scheduling_score,
                "exec_cost": self.exec_cost,
                "trans_cost": self.trans_cost,
                "fixed": self.fixed_location,
            },
            "data_flow": {
                "input_size_bytes": self.input_size_bytes,
                "output_size_bytes": self.output_size_bytes,
                "reduction_ratio": self.reduction_ratio,
                "data_flow_type": self.data_flow_type,
                "expected_reduction_ratio": self.expected_reduction_ratio,
                "mcp_request_size_bytes": self.mcp_request_size_bytes,
                "mcp_response_size_bytes": self.mcp_response_size_bytes,
            },
            "resource": {
                "memory_before_bytes": self.memory_before_bytes,
                "memory_after_bytes": self.memory_after_bytes,
                "memory_delta_bytes": self.memory_delta_bytes,
                "cpu_time_user_ms": self.cpu_time_user_ms,
                "cpu_time_system_ms": self.cpu_time_system_ms,
            },
            "status": {
                "success": self.success,
                "retry_count": self.retry_count,
                "error": self.error,
                "error_type": self.error_type,
            },
            "network": {
                "network_type": self.network_type,
                "request_bytes": self.request_bytes,
                "response_bytes": self.response_bytes,
            },
            "args_keys": self.args_keys,
            "custom_metadata": self.custom_metadata,
        }


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""

    enabled: bool = True
    collect_sizes: bool = True
    collect_timing: bool = True
    collect_resource: bool = True  # Memory/CPU - may have overhead
    max_arg_sample_size: int = 1000  # Truncate large args for logging


class MetricsCollector:
    """
    통합 메트릭 수집기

    Design Goals:
    1. Single point of collection (in proxy_tool._arun)
    2. Zero-config basic usage
    3. Extensible for custom metrics per scenario
    4. Multiple export formats (JSON, console, DataFrame)
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        scenario_name: str = "unknown",
        client_location: str = "CLIENT",
        scheduler_type: str = "unknown",
    ):
        self.config = config or MetricsConfig()
        self.scenario_name = scenario_name
        self.client_location = client_location
        self.scheduler_type = scheduler_type
        self._entries: list[MetricEntry] = []
        self._custom_metrics: dict[str, Any] = {}
        self._session_id: str = str(uuid.uuid4())[:8]
        self._session_start: float = time.time()
        self._session_end: Optional[float] = None
        self._server_startup_time_ms: float = 0.0
        self._last_tool_end_time: Optional[float] = None
        self._pipeline_step: int = 0

        # Chain Scheduling 결과
        self._chain_scheduling: Optional[ChainSchedulingResult] = None
        # 시나리오별 메트릭
        self._scenario_metrics: ScenarioMetrics = ScenarioMetrics()
        # LLM Latency (Agent 모드용)
        self._pending_llm_latency: Optional[dict] = None
        self._pending_llm_consumed: bool = False

    # =========================================================================
    # Core Collection API
    # =========================================================================

    def start_call(
        self,
        tool_name: str,
        parent_tool_name: str,
        location: str,
        args: dict[str, Any],
    ) -> "CallContext":
        """
        Start tracking a tool call. Returns a context manager.

        Usage:
            async with collector.start_call("read_file", "filesystem", "DEVICE", args) as ctx:
                result = await backend_tool.ainvoke(args)
                ctx.set_result(result)
        """
        self._pipeline_step += 1
        return CallContext(
            collector=self,
            tool_name=tool_name,
            parent_tool_name=parent_tool_name,
            scheduled_location=location,
            args=args,
            config=self.config,
            pipeline_step=self._pipeline_step,
            last_tool_end_time=self._last_tool_end_time,
        )

    def record_entry(self, entry: MetricEntry):
        """Record a completed metric entry"""
        self._entries.append(entry)
        self._last_tool_end_time = time.time()

    def add_custom_metric(self, key: str, value: Any):
        """Add a custom session-level metric"""
        self._custom_metrics[key] = value

    def set_server_startup_time(self, time_ms: float):
        """Record server startup time (cold start)"""
        self._server_startup_time_ms = time_ms

    def set_chain_scheduling_result(
        self,
        total_cost: float,
        search_space_size: int,
        valid_combinations: int,
        decision_time_ms: float,
        tool_chain: list[str],
        placements: list[dict],
        optimization_method: str = "brute_force",
    ):
        """
        Chain Scheduling 결과 기록

        Args:
            total_cost: 전체 Tool Chain의 최적 비용
            search_space_size: 탐색 공간 크기 (조합 수)
            valid_combinations: 유효한 조합 수
            decision_time_ms: 스케줄링 결정 시간 (ms)
            tool_chain: Tool Chain 목록
            placements: 각 Tool의 배치 결과 (dict list)
            optimization_method: 최적화 방법
        """
        self._chain_scheduling = ChainSchedulingResult(
            total_cost=total_cost,
            search_space_size=search_space_size,
            valid_combinations=valid_combinations,
            decision_time_ms=decision_time_ms,
            optimization_method=optimization_method,
            tool_chain=tool_chain,
            placements=placements,
        )

    def set_scenario_metrics(self, **kwargs):
        """
        시나리오별 메트릭 설정

        사용 예:
            collector.set_scenario_metrics(
                data_source="LogHub",
                entries_parsed=1000,
                entries_filtered=50,
            )
        """
        for key, value in kwargs.items():
            if hasattr(self._scenario_metrics, key):
                setattr(self._scenario_metrics, key, value)

    def set_pending_llm_latency(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """
        다음 tool call(들)에 연결할 LLM latency 설정 (Agent 모드용)

        Args:
            latency_ms: LLM 추론 시간 (ms)
            input_tokens: LLM 입력 토큰 수
            output_tokens: LLM 출력 토큰 수
        """
        self._pending_llm_latency = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self._pending_llm_consumed = False  # 아직 사용되지 않음

    def get_pending_llm_latency(self) -> Optional[dict]:
        """
        pending LLM latency 반환 (병렬 tool call 지원)

        첫 번째 tool이 가져간 후에도 같은 LLM 호출에서 나온
        병렬 tool들이 같은 latency를 공유할 수 있도록 함.
        새로운 LLM 호출이 있을 때만 초기화됨.

        Returns:
            LLM latency 정보 dict 또는 None
        """
        if self._pending_llm_latency is None:
            return None

        # 첫 번째 tool에만 full latency 기록, 나머지는 0 (중복 방지)
        if not self._pending_llm_consumed:
            self._pending_llm_consumed = True
            return self._pending_llm_latency
        else:
            # 병렬 tool들은 같은 LLM 호출에서 나왔지만 latency는 0으로 기록
            # (total 합산 시 중복 방지)
            return {
                "latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def get_and_clear_pending_llm_latency(self) -> Optional[dict]:
        """
        pending LLM latency 반환 후 초기화 (하위 호환성)

        Returns:
            LLM latency 정보 dict 또는 None
        """
        return self.get_pending_llm_latency()

    @property
    def chain_scheduling(self) -> Optional[ChainSchedulingResult]:
        """Get chain scheduling result"""
        return self._chain_scheduling

    @property
    def scenario_metrics(self) -> ScenarioMetrics:
        """Get scenario-specific metrics"""
        return self._scenario_metrics

    def finalize(self):
        """Mark session as complete"""
        self._session_end = time.time()

    # =========================================================================
    # Query API
    # =========================================================================

    @property
    def entries(self) -> list[MetricEntry]:
        """Get all recorded metric entries"""
        return self._entries.copy()

    @property
    def custom_metrics(self) -> dict[str, Any]:
        """Get custom session-level metrics"""
        return self._custom_metrics

    def get_entries_by_tool(self, tool_name: str) -> list[MetricEntry]:
        """Filter entries by tool name"""
        return [e for e in self._entries if e.tool_name == tool_name]

    def get_entries_by_location(self, location: str) -> list[MetricEntry]:
        """Filter entries by location"""
        return [e for e in self._entries if e.actual_location == location]

    # =========================================================================
    # Computed Aggregates
    # =========================================================================

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all calls"""
        return sum(e.latency_ms for e in self._entries)

    @property
    def total_input_bytes(self) -> int:
        """Total input data size"""
        return sum(e.input_size_bytes for e in self._entries)

    @property
    def total_output_bytes(self) -> int:
        """Total output data size"""
        return sum(e.output_size_bytes for e in self._entries)

    @property
    def overall_reduction_ratio(self) -> float:
        """Overall data reduction ratio"""
        if self.total_input_bytes == 0:
            return 0.0
        return self.total_output_bytes / self.total_input_bytes

    @property
    def success_rate(self) -> float:
        """Success rate across all calls"""
        if not self._entries:
            return 0.0
        return sum(1 for e in self._entries if e.success) / len(self._entries)

    @property
    def pipeline_depth(self) -> int:
        """Number of tools in pipeline"""
        return len(self._entries)

    def latency_by_location(self) -> dict[str, float]:
        """Total latency breakdown by location"""
        result: dict[str, float] = {"DEVICE": 0.0, "EDGE": 0.0, "CLOUD": 0.0}
        for entry in self._entries:
            if entry.actual_location in result:
                result[entry.actual_location] += entry.latency_ms
        return result

    def call_count_by_location(self) -> dict[str, int]:
        """Call count breakdown by location"""
        result: dict[str, int] = {"DEVICE": 0, "EDGE": 0, "CLOUD": 0}
        for entry in self._entries:
            if entry.actual_location in result:
                result[entry.actual_location] += 1
        return result

    def data_by_location(self) -> dict[str, int]:
        """Data transfer by location (input + output)"""
        result: dict[str, int] = {"DEVICE": 0, "EDGE": 0, "CLOUD": 0}
        for entry in self._entries:
            if entry.actual_location in result:
                result[entry.actual_location] += (
                    entry.input_size_bytes + entry.output_size_bytes
                )
        return result

    def total_mcp_overhead_ms(self) -> float:
        """Total MCP serialization/deserialization overhead"""
        return sum(
            e.mcp_serialization_time_ms + e.mcp_deserialization_time_ms
            for e in self._entries
        )

    def total_resource_usage(self) -> dict[str, float]:
        """Total resource usage"""
        return {
            "peak_memory_delta_bytes": max(
                (e.memory_delta_bytes for e in self._entries), default=0
            ),
            "total_cpu_user_ms": sum(e.cpu_time_user_ms for e in self._entries),
            "total_cpu_system_ms": sum(e.cpu_time_system_ms for e in self._entries),
        }

    def _classify_network(self, tool_location: str) -> str:
        """
        네트워크 타입 분류

        Client 위치에 따라 네트워크 타입을 분류합니다:
        - local: Client와 Tool이 같은 위치
        - lan: DEVICE ↔ EDGE (Local Area Network)
        - wan: ↔ CLOUD (Wide Area Network)

        Args:
            tool_location: Tool 실행 위치 (DEVICE/EDGE/CLOUD)

        Returns:
            네트워크 타입 ("local", "lan", "wan")

        Note:
            CLIENT는 DEVICE와 같은 위치에 있다고 가정합니다.
            즉, CLIENT에서 DEVICE로의 호출은 local,
            CLIENT에서 EDGE로의 호출은 lan,
            CLIENT에서 CLOUD로의 호출은 wan입니다.
        """
        # CLIENT는 DEVICE와 동일하게 취급
        effective_client = self.client_location
        if effective_client == "CLIENT":
            effective_client = "DEVICE"

        # Client가 해당 location에 있는 경우 local
        if effective_client == tool_location:
            return "local"

        # CLOUD는 항상 WAN
        if tool_location == "CLOUD":
            return "wan"

        # DEVICE ↔ EDGE는 LAN
        if effective_client in ("DEVICE", "EDGE") and tool_location in ("DEVICE", "EDGE"):
            return "lan"

        # 그 외는 WAN
        return "wan"

    def network_summary(self) -> dict[str, Any]:
        """
        네트워크 트래픽 요약

        Returns:
            네트워크 트래픽 요약 딕셔너리
        """
        routes = {
            "local": {"calls": 0, "bytes": 0},
            "lan": {"calls": 0, "bytes": 0},
            "wan": {"calls": 0, "bytes": 0},
        }
        by_target = {
            "DEVICE": {"calls": 0, "bytes": 0},
            "EDGE": {"calls": 0, "bytes": 0},
            "CLOUD": {"calls": 0, "bytes": 0},
        }

        for entry in self._entries:
            location = entry.actual_location
            network_type = self._classify_network(location)
            total_bytes = entry.input_size_bytes + entry.output_size_bytes

            # 네트워크 타입별 집계
            if network_type in routes:
                routes[network_type]["calls"] += 1
                routes[network_type]["bytes"] += total_bytes

            # 대상 위치별 집계
            if location in by_target:
                by_target[location]["calls"] += 1
                by_target[location]["bytes"] += total_bytes

        return {
            "client_location": self.client_location,
            "routes": routes,
            "by_target": by_target,
            "total_wan_bytes": routes["wan"]["bytes"],
            "total_lan_bytes": routes["lan"]["bytes"],
            "total_local_bytes": routes["local"]["bytes"],
        }

    # =========================================================================
    # Export API
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as dictionary"""
        end_time = self._session_end or time.time()

        # chain_scheduling이 없으면 entries에서 합산
        chain_scheduling_dict = None
        if self._chain_scheduling:
            chain_scheduling_dict = self._chain_scheduling.to_dict()
        elif self._entries:
            # Agent 모드: entries에서 scheduling 정보 합산
            chain_scheduling_dict = self._aggregate_scheduling_from_entries()

        result = {
            "session_id": self._session_id,
            "scenario_name": self.scenario_name,
            "scheduler_type": self.scheduler_type,
            "start_time": self._session_start,
            "start_time_iso": datetime.fromtimestamp(self._session_start).isoformat(),
            "end_time": end_time,
            "end_time_iso": datetime.fromtimestamp(end_time).isoformat(),
            "summary": {
                "total_calls": len(self._entries),
                "total_latency_ms": self.total_latency_ms,
                "pipeline_depth": self.pipeline_depth,
                "parallel_calls_count": 0,  # TODO: detect parallel calls
                "success_rate": self.success_rate,
                "data_flow": {
                    "cumulative_input_bytes": self.total_input_bytes,
                    "cumulative_output_bytes": self.total_output_bytes,
                    "overall_reduction_ratio": self.overall_reduction_ratio,
                    "data_by_location": self.data_by_location(),
                },
                "location_distribution": {
                    "call_count_by_location": self.call_count_by_location(),
                    "latency_by_location": self.latency_by_location(),
                },
                "mcp_overhead": {
                    "total_serialization_ms": sum(
                        e.mcp_serialization_time_ms for e in self._entries
                    ),
                    "total_deserialization_ms": sum(
                        e.mcp_deserialization_time_ms for e in self._entries
                    ),
                    "server_startup_time_ms": self._server_startup_time_ms,
                },
                "resource_usage": self.total_resource_usage(),
                "network": self.network_summary(),
                "llm_usage": self._aggregate_llm_usage(),
            },
            "chain_scheduling": chain_scheduling_dict,
            "scenario_metrics": self._scenario_metrics.to_dict(),
            "custom_metrics": self._custom_metrics,
            "entries": [e.to_dict() for e in self._entries],
        }

        return result

    def _aggregate_scheduling_from_entries(self) -> dict[str, Any]:
        """Agent 모드: entries에서 scheduling 정보 합산"""
        if not self._entries:
            return None

        total_score = 0.0
        total_exec_cost = 0.0
        total_trans_cost = 0.0
        total_decision_time_ns = 0
        placements = []

        for entry in self._entries:
            total_score += entry.scheduling_score
            total_exec_cost += entry.exec_cost
            total_trans_cost += entry.trans_cost
            total_decision_time_ns += entry.scheduling_decision_time_ns

            placements.append({
                "tool_name": entry.tool_name,
                "location": entry.actual_location,
                "reason": entry.scheduling_reason or "unknown",
                "score": entry.scheduling_score,
                "exec_cost": entry.exec_cost,
                "trans_cost": entry.trans_cost,
                "fixed": entry.fixed_location,
            })

        return {
            "total_cost": total_score,
            "search_space_size": len(self._entries),
            "valid_combinations": 1,  # Agent 모드는 동적 결정
            "decision_time_ns": total_decision_time_ns,
            "decision_time_ms": total_decision_time_ns / 1e6,
            "optimization_method": self.scheduler_type,
            "placements": placements,
        }

    def _aggregate_llm_usage(self) -> dict[str, Any]:
        """LLM 사용량 합산 (Agent 모드용)"""
        if not self._entries:
            return {
                "total_llm_latency_ms": 0.0,
                "total_tool_latency_ms": 0.0,
                "total_combined_latency_ms": 0.0,
                "total_llm_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "avg_llm_latency_ms": 0.0,
            }

        total_llm_latency_ms = 0.0
        total_tool_latency_ms = 0.0
        total_llm_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for entry in self._entries:
            total_tool_latency_ms += entry.latency_ms
            if entry.llm_latency_ms > 0:
                total_llm_latency_ms += entry.llm_latency_ms
                total_llm_calls += 1
                total_input_tokens += entry.llm_input_tokens
                total_output_tokens += entry.llm_output_tokens

        avg_llm_latency_ms = (
            total_llm_latency_ms / total_llm_calls if total_llm_calls > 0 else 0.0
        )

        return {
            "total_llm_latency_ms": total_llm_latency_ms,
            "total_tool_latency_ms": total_tool_latency_ms,
            "total_combined_latency_ms": total_llm_latency_ms + total_tool_latency_ms,
            "total_llm_calls": total_llm_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_llm_latency_ms": avg_llm_latency_ms,
        }

    def save_json(self, output_path: str | Path, pretty: bool = True) -> Path:
        """Save metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2 if pretty else None)

        return output_path

    def save_csv(self, output_path: str | Path) -> Path:
        """Save entries to CSV file for pandas analysis"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten entries for CSV
        rows = []
        for e in self._entries:
            rows.append(
                {
                    "session_id": self._session_id,
                    "scenario_name": self.scenario_name,
                    "tool_name": e.tool_name,
                    "parent_tool_name": e.parent_tool_name,
                    "pipeline_step": e.pipeline_step,
                    "timestamp": e.timestamp,
                    "latency_ms": e.latency_ms,
                    "inter_tool_latency_ms": e.inter_tool_latency_ms,
                    "scheduled_location": e.scheduled_location,
                    "actual_location": e.actual_location,
                    "fallback_occurred": e.fallback_occurred,
                    "scheduling_decision_time_ns": e.scheduling_decision_time_ns,
                    "scheduling_reason": e.scheduling_reason,
                    "scheduling_score": e.scheduling_score,
                    "exec_cost": e.exec_cost,
                    "trans_cost": e.trans_cost,
                    "fixed_location": e.fixed_location,
                    "input_size_bytes": e.input_size_bytes,
                    "output_size_bytes": e.output_size_bytes,
                    "reduction_ratio": e.reduction_ratio,
                    "data_flow_type": e.data_flow_type,
                    "mcp_serialization_time_ms": e.mcp_serialization_time_ms,
                    "mcp_deserialization_time_ms": e.mcp_deserialization_time_ms,
                    "memory_delta_bytes": e.memory_delta_bytes,
                    "cpu_time_user_ms": e.cpu_time_user_ms,
                    "cpu_time_system_ms": e.cpu_time_system_ms,
                    "success": e.success,
                    "retry_count": e.retry_count,
                    "error": e.error,
                }
            )

        # Write CSV
        if rows:
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return output_path

    def print_summary(self):
        """Print human-readable summary to console"""
        print("\n" + "=" * 70)
        print("Metrics Summary")
        print("=" * 70)
        print(f"Session ID: {self._session_id}")
        print(f"Scenario: {self.scenario_name}")
        print(f"Total Calls: {len(self._entries)}")
        print(f"Success Rate: {self.success_rate * 100:.1f}%")
        print(f"Total Latency: {self.total_latency_ms:.2f} ms")
        print(
            f"Data: {self.total_input_bytes:,} -> {self.total_output_bytes:,} bytes"
        )
        print(
            f"Reduction: {self.overall_reduction_ratio:.4f} "
            f"({self.overall_reduction_ratio * 100:.2f}%)"
        )

        # Chain Scheduling 결과 출력
        if self._chain_scheduling:
            print("\nChain Scheduling:")
            print(f"  Total Cost: {self._chain_scheduling.total_cost:.4f}")
            print(f"  Search Space: {self._chain_scheduling.search_space_size}")
            print(f"  Valid Combinations: {self._chain_scheduling.valid_combinations}")
            print(f"  Decision Time: {self._chain_scheduling.decision_time_ms:.2f} ms")
            print(f"  Method: {self._chain_scheduling.optimization_method}")
            if self._chain_scheduling.placements:
                print("  Placements:")
                for p in self._chain_scheduling.placements:
                    fixed = "[FIXED]" if p.get("fixed") else ""
                    print(f"    {p.get('tool_name', 'N/A'):20} -> {p.get('location', 'N/A'):6} "
                          f"(cost={p.get('score', 0):.3f}) {fixed}")

        print("\nLatency by Location:")
        for loc, latency in self.latency_by_location().items():
            count = self.call_count_by_location()[loc]
            if count > 0:
                print(f"  {loc}: {latency:.2f} ms ({count} calls)")

        if self._server_startup_time_ms > 0:
            print(f"\nServer Startup: {self._server_startup_time_ms:.2f} ms")

        print(f"\nMCP Overhead: {self.total_mcp_overhead_ms():.2f} ms")

        print("\nTool Calls:")
        for i, entry in enumerate(self._entries, 1):
            status = "[OK]" if entry.success else "[FAIL]"
            print(
                f"  {i}. [{entry.actual_location}] {entry.tool_name}: "
                f"{entry.input_size_bytes:,} -> {entry.output_size_bytes:,} bytes "
                f"({entry.latency_ms:.2f}ms) {status}"
            )

        # Scenario Metrics 출력
        scenario_dict = self._scenario_metrics.to_dict()
        if scenario_dict:
            print("\nScenario Metrics:")
            for key, value in scenario_dict.items():
                print(f"  {key}: {value}")

        if self._custom_metrics:
            print("\nCustom Metrics:")
            for key, value in self._custom_metrics.items():
                print(f"  {key}: {value}")

        print("=" * 70 + "\n")

    def to_dataframe(self):
        """Convert to pandas DataFrame (for analysis)"""
        try:
            import pandas as pd

            return pd.DataFrame([e.to_dict() for e in self._entries])
        except ImportError:
            raise ImportError("pandas required for DataFrame export")

    # =========================================================================
    # Backward Compatibility
    # =========================================================================

    def to_execution_trace(self) -> list[dict]:
        """
        Convert to execution_trace format with scheduling cost info.

        Includes:
        {"tool": ..., "location": ..., "cost": ..., "comp": ..., "comm": ..., "fixed": ...}
        """
        return [
            {
                "tool": e.tool_name,
                "parent_tool": e.parent_tool_name,
                "location": e.actual_location,
                "args_keys": e.args_keys,
                # Scheduling cost info
                "cost": e.scheduling_score,
                "comp": e.exec_cost,
                "comm": e.trans_cost,
                "fixed": e.fixed_location,
                "reason": e.scheduling_reason,
                # Enhanced fields
                "latency_ms": e.latency_ms,
                "input_size": e.input_size_bytes,
                "output_size": e.output_size_bytes,
                "success": e.success,
            }
            for e in self._entries
        ]


class CallContext:
    """
    Context manager for tracking a single tool call.

    Handles timing, size calculation, and error capture automatically.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        tool_name: str,
        parent_tool_name: str,
        scheduled_location: str,
        args: dict[str, Any],
        config: MetricsConfig,
        pipeline_step: int,
        last_tool_end_time: Optional[float],
    ):
        self.collector = collector
        self.tool_name = tool_name
        self.parent_tool_name = parent_tool_name
        self.scheduled_location = scheduled_location
        self.actual_location = scheduled_location  # May be updated on fallback
        self.args = args
        self.config = config
        self.pipeline_step = pipeline_step
        self.last_tool_end_time = last_tool_end_time

        self.call_id = str(uuid.uuid4())[:12]
        self.start_time_ns: int = 0
        self.end_time_ns: int = 0
        self.timestamp: float = 0
        self.result: Any = None
        self.error: Optional[Exception] = None

        # Scheduling info
        self.scheduling_decision_time_ns: int = 0
        self.scheduling_reason: str = ""
        self.constraints_checked: list[str] = []
        self.available_locations: list[str] = []
        self.fallback_occurred: bool = False

        # Scheduling cost (BruteForceChainScheduler)
        self.scheduling_score: float = 0.0
        self.exec_cost: float = 0.0
        self.trans_cost: float = 0.0
        self.fixed_location: bool = False

        # Data flow info
        self.data_flow_type: str = ""
        self.expected_reduction_ratio: float = 0.0

        # Resource tracking
        self.memory_before: int = 0
        self.memory_after: int = 0
        self.cpu_before: tuple[float, float] = (0.0, 0.0)
        self.cpu_after: tuple[float, float] = (0.0, 0.0)

        # MCP timing
        self.mcp_serialization_time_ms: float = 0.0
        self.mcp_deserialization_time_ms: float = 0.0

        # LLM latency (Agent 모드용)
        self.llm_latency_ms: float = 0.0
        self.llm_input_tokens: int = 0
        self.llm_output_tokens: int = 0

    async def __aenter__(self) -> "CallContext":
        """Start timing and resource tracking"""
        self.timestamp = time.time()
        self.start_time_ns = time.perf_counter_ns()

        if self.config.collect_resource:
            self.memory_before = self._get_memory_usage()
            self.cpu_before = self._get_cpu_times()

        # Get pending LLM latency (Agent 모드용)
        pending_llm = self.collector.get_and_clear_pending_llm_latency()
        if pending_llm:
            self.llm_latency_ms = pending_llm.get("latency_ms", 0.0)
            self.llm_input_tokens = pending_llm.get("input_tokens", 0)
            self.llm_output_tokens = pending_llm.get("output_tokens", 0)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record entry"""
        self.end_time_ns = time.perf_counter_ns()

        if self.config.collect_resource:
            self.memory_after = self._get_memory_usage()
            self.cpu_after = self._get_cpu_times()

        if exc_val is not None:
            self.error = exc_val

        # Calculate sizes
        input_size = self._calculate_size(self.args) if self.config.collect_sizes else 0
        output_size = (
            self._calculate_size(self.result) if self.config.collect_sizes else 0
        )

        # Calculate inter-tool latency
        inter_tool_latency_ms = 0.0
        if self.last_tool_end_time is not None:
            inter_tool_latency_ms = (self.timestamp - self.last_tool_end_time) * 1000

        # Calculate reduction ratio
        reduction_ratio = output_size / input_size if input_size > 0 else 0.0

        # Calculate network type
        network_type = self.collector._classify_network(self.actual_location)

        # Create and record entry
        entry = MetricEntry(
            tool_name=self.tool_name,
            parent_tool_name=self.parent_tool_name,
            call_id=self.call_id,
            pipeline_step=self.pipeline_step,
            timestamp=self.timestamp,
            latency_ms=(self.end_time_ns - self.start_time_ns) / 1_000_000,
            inter_tool_latency_ms=inter_tool_latency_ms,
            # LLM latency (Agent 모드용)
            llm_latency_ms=self.llm_latency_ms,
            llm_input_tokens=self.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens,
            scheduled_location=self.scheduled_location,
            actual_location=self.actual_location,
            fallback_occurred=self.actual_location != self.scheduled_location,
            scheduling_decision_time_ns=self.scheduling_decision_time_ns,
            scheduling_reason=self.scheduling_reason,
            constraints_checked=self.constraints_checked,
            available_locations=self.available_locations,
            # Scheduling cost
            scheduling_score=self.scheduling_score,
            exec_cost=self.exec_cost,
            trans_cost=self.trans_cost,
            fixed_location=self.fixed_location,
            # Data flow
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            reduction_ratio=reduction_ratio,
            data_flow_type=self.data_flow_type,
            expected_reduction_ratio=self.expected_reduction_ratio,
            mcp_serialization_time_ms=self.mcp_serialization_time_ms,
            mcp_deserialization_time_ms=self.mcp_deserialization_time_ms,
            mcp_request_size_bytes=input_size,  # Approximate
            mcp_response_size_bytes=output_size,  # Approximate
            memory_before_bytes=self.memory_before,
            memory_after_bytes=self.memory_after,
            memory_delta_bytes=self.memory_after - self.memory_before,
            cpu_time_user_ms=(self.cpu_after[0] - self.cpu_before[0]) * 1000,
            cpu_time_system_ms=(self.cpu_after[1] - self.cpu_before[1]) * 1000,
            success=self.error is None,
            error=str(self.error) if self.error else None,
            error_type=type(self.error).__name__ if self.error else None,
            network_type=network_type,
            request_bytes=input_size,
            response_bytes=output_size,
            args_keys=list(self.args.keys()),
        )

        self.collector.record_entry(entry)

        # Don't suppress exceptions
        return False

    def set_result(self, result: Any):
        """Set the result for size calculation"""
        self.result = result

    def set_actual_location(self, location: str, fallback: bool = False):
        """Update actual location if different from scheduled"""
        self.actual_location = location
        self.fallback_occurred = fallback

    def add_scheduling_info(
        self,
        reason: str,
        constraints: list[str],
        available: list[str],
        decision_time_ns: int = 0,
        score: float = 0.0,
        exec_cost: float = 0.0,
        trans_cost: float = 0.0,
        fixed: bool = False,
    ):
        """Add scheduling decision information"""
        self.scheduling_reason = reason
        self.constraints_checked = constraints
        self.available_locations = available
        self.scheduling_decision_time_ns = decision_time_ns
        self.scheduling_score = score
        self.exec_cost = exec_cost
        self.trans_cost = trans_cost
        self.fixed_location = fixed

    def add_data_flow_info(self, data_flow_type: str, expected_ratio: float = 0.0):
        """Add data flow information from tool profile"""
        self.data_flow_type = data_flow_type
        self.expected_reduction_ratio = expected_ratio

    def add_mcp_timing(self, serialization_ms: float, deserialization_ms: float):
        """Add MCP protocol timing"""
        self.mcp_serialization_time_ms = serialization_ms
        self.mcp_deserialization_time_ms = deserialization_ms

    def set_error(self, error: Exception):
        """Set error information for failed tool calls"""
        self.error = error

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        if data is None:
            return 0

        try:
            return len(json.dumps(data, default=str))
        except (TypeError, ValueError):
            return len(str(data))

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return 0

    def _get_cpu_times(self) -> tuple[float, float]:
        """Get current CPU times (user, system) in seconds"""
        try:
            r = resource.getrusage(resource.RUSAGE_SELF)
            return (r.ru_utime, r.ru_stime)
        except Exception:
            return (0.0, 0.0)
