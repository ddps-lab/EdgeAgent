#!/usr/bin/env python3
"""
Scheduler Comparison Script

다양한 스케줄러를 비교하여 각 스케줄러의 성능을 평가합니다.

비교 대상:
- all_device: 모든 Tool → DEVICE
- all_edge: 모든 Tool → EDGE
- all_cloud: 모든 Tool → CLOUD
- static: Static mapping 기반
- heuristic: Profile 기반 휴리스틱

비교 메트릭:
- Location Distribution (DEVICE/EDGE/CLOUD 호출 수)
- Network Traffic (WAN/LAN/Local bytes)
- Latency (총 지연시간, location별 지연시간)
- Compliance (Privacy 위반 여부, 실행 가능성)

Usage:
    python scripts/compare_schedulers.py
    python scripts/compare_schedulers.py --scenario s2
    python scripts/compare_schedulers.py --schedulers static,heuristic
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import EdgeAgentMCPClient, ScenarioRunner
from edgeagent.scheduler import SCHEDULER_REGISTRY, create_scheduler


@dataclass
class SchedulerMetrics:
    """단일 스케줄러의 실행 결과 메트릭"""
    scheduler_name: str
    success: bool = True
    error: Optional[str] = None

    # Location Distribution
    device_calls: int = 0
    edge_calls: int = 0
    cloud_calls: int = 0

    # Network Traffic
    wan_bytes: int = 0
    lan_bytes: int = 0
    local_bytes: int = 0

    # Latency
    total_latency_ms: float = 0.0
    device_latency_ms: float = 0.0
    edge_latency_ms: float = 0.0
    cloud_latency_ms: float = 0.0

    # Compliance
    privacy_violations: int = 0
    fallback_count: int = 0

    # Scheduling Decisions
    scheduling_reasons: dict[str, int] = field(default_factory=dict)


def extract_metrics(result, scheduler_name: str) -> SchedulerMetrics:
    """ScenarioResult에서 비교 메트릭 추출"""
    metrics = SchedulerMetrics(scheduler_name=scheduler_name)

    if not result.success:
        metrics.success = False
        metrics.error = result.error
        return metrics

    collector = result.metrics
    if collector is None:
        return metrics

    # Location Distribution
    call_count = collector.call_count_by_location()
    metrics.device_calls = call_count.get("DEVICE", 0)
    metrics.edge_calls = call_count.get("EDGE", 0)
    metrics.cloud_calls = call_count.get("CLOUD", 0)

    # Network Traffic
    network = collector.network_summary()
    metrics.wan_bytes = network.get("total_wan_bytes", 0)
    metrics.lan_bytes = network.get("total_lan_bytes", 0)
    metrics.local_bytes = network.get("total_local_bytes", 0)

    # Latency
    metrics.total_latency_ms = collector.total_latency_ms
    latency_by_loc = collector.latency_by_location()
    metrics.device_latency_ms = latency_by_loc.get("DEVICE", 0.0)
    metrics.edge_latency_ms = latency_by_loc.get("EDGE", 0.0)
    metrics.cloud_latency_ms = latency_by_loc.get("CLOUD", 0.0)

    # Compliance & Scheduling Decisions
    for entry in collector.entries:
        if entry.fallback_occurred:
            metrics.fallback_count += 1

        # Privacy 위반 체크: privacy_sensitive인데 CLOUD에서 실행
        if "privacy_sensitive" in entry.constraints_checked and entry.actual_location == "CLOUD":
            metrics.privacy_violations += 1

        # Scheduling reason 집계
        reason = entry.scheduling_reason or "unknown"
        metrics.scheduling_reasons[reason] = metrics.scheduling_reasons.get(reason, 0) + 1

    return metrics


def print_comparison_table(results: dict[str, SchedulerMetrics]):
    """비교 테이블 출력"""
    schedulers = list(results.keys())

    print()
    print("=" * 100)
    print("SCHEDULER COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Metric':<30}"
    for name in schedulers:
        header += f"{name:>14}"
    print(header)
    print("-" * 100)

    # Location Distribution
    print("Location Distribution:")
    row = f"  {'DEVICE calls':<28}"
    for name in schedulers:
        row += f"{results[name].device_calls:>14}"
    print(row)

    row = f"  {'EDGE calls':<28}"
    for name in schedulers:
        row += f"{results[name].edge_calls:>14}"
    print(row)

    row = f"  {'CLOUD calls':<28}"
    for name in schedulers:
        row += f"{results[name].cloud_calls:>14}"
    print(row)

    print()

    # Network Traffic
    print("Network Traffic:")
    row = f"  {'WAN bytes':<28}"
    for name in schedulers:
        row += f"{results[name].wan_bytes:>14,}"
    print(row)

    row = f"  {'LAN bytes':<28}"
    for name in schedulers:
        row += f"{results[name].lan_bytes:>14,}"
    print(row)

    row = f"  {'Local bytes':<28}"
    for name in schedulers:
        row += f"{results[name].local_bytes:>14,}"
    print(row)

    print()

    # Latency
    print("Latency:")
    row = f"  {'Total (ms)':<28}"
    for name in schedulers:
        row += f"{results[name].total_latency_ms:>14.2f}"
    print(row)

    row = f"  {'DEVICE (ms)':<28}"
    for name in schedulers:
        row += f"{results[name].device_latency_ms:>14.2f}"
    print(row)

    row = f"  {'EDGE (ms)':<28}"
    for name in schedulers:
        row += f"{results[name].edge_latency_ms:>14.2f}"
    print(row)

    row = f"  {'CLOUD (ms)':<28}"
    for name in schedulers:
        row += f"{results[name].cloud_latency_ms:>14.2f}"
    print(row)

    print()

    # Compliance
    print("Compliance:")
    row = f"  {'Privacy violations':<28}"
    for name in schedulers:
        row += f"{results[name].privacy_violations:>14}"
    print(row)

    row = f"  {'Fallback count':<28}"
    for name in schedulers:
        row += f"{results[name].fallback_count:>14}"
    print(row)

    row = f"  {'Success':<28}"
    for name in schedulers:
        status = "✓" if results[name].success else "✗"
        row += f"{status:>14}"
    print(row)

    print("=" * 100)


def save_comparison_json(results: dict[str, SchedulerMetrics], output_path: Path):
    """비교 결과를 JSON으로 저장"""
    data = {}
    for name, metrics in results.items():
        data[name] = {
            "success": metrics.success,
            "error": metrics.error,
            "location_distribution": {
                "device_calls": metrics.device_calls,
                "edge_calls": metrics.edge_calls,
                "cloud_calls": metrics.cloud_calls,
            },
            "network": {
                "wan_bytes": metrics.wan_bytes,
                "lan_bytes": metrics.lan_bytes,
                "local_bytes": metrics.local_bytes,
            },
            "latency": {
                "total_ms": metrics.total_latency_ms,
                "device_ms": metrics.device_latency_ms,
                "edge_ms": metrics.edge_latency_ms,
                "cloud_ms": metrics.cloud_latency_ms,
            },
            "compliance": {
                "privacy_violations": metrics.privacy_violations,
                "fallback_count": metrics.fallback_count,
            },
            "scheduling_reasons": metrics.scheduling_reasons,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


async def run_with_scheduler(
    scenario_class: type[ScenarioRunner],
    config_path: Path,
    scheduler_name: str,
    output_dir: Path,
) -> Any:
    """특정 스케줄러로 시나리오 실행"""
    print(f"\n{'='*60}")
    print(f"Running with scheduler: {scheduler_name}")
    print(f"{'='*60}")

    scenario = scenario_class(
        config_path=config_path,
        output_dir=output_dir / scheduler_name,
    )

    # Scheduler 오버라이드를 위해 custom client 설정 필요
    # 현재는 기본 scheduler로 실행하고, 향후 middleware에 scheduler 선택 옵션 추가 예정
    result = await scenario.run(
        save_results=True,
        print_summary=False,  # 개별 summary는 생략
    )

    return result


async def compare_schedulers(
    scenario_class: type[ScenarioRunner],
    config_path: Path,
    output_dir: Path,
    scheduler_names: list[str] | None = None,
):
    """
    여러 스케줄러를 비교

    Args:
        scenario_class: 실행할 시나리오 클래스
        config_path: YAML 설정 파일 경로
        output_dir: 결과 저장 디렉토리
        scheduler_names: 비교할 스케줄러 목록 (None이면 전체)
    """
    if scheduler_names is None:
        scheduler_names = list(SCHEDULER_REGISTRY.keys())

    results: dict[str, SchedulerMetrics] = {}

    for scheduler_name in scheduler_names:
        try:
            result = await run_with_scheduler(
                scenario_class,
                config_path,
                scheduler_name,
                output_dir,
            )
            results[scheduler_name] = extract_metrics(result, scheduler_name)
        except Exception as e:
            print(f"[ERROR] Failed to run with {scheduler_name}: {e}")
            results[scheduler_name] = SchedulerMetrics(
                scheduler_name=scheduler_name,
                success=False,
                error=str(e),
            )

    # 결과 출력
    print_comparison_table(results)

    # JSON 저장
    save_comparison_json(results, output_dir / "comparison.json")

    return results


# ============================================================================
# Main
# ============================================================================

async def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare different schedulers")
    parser.add_argument(
        "--scenario",
        type=str,
        default="s2",
        choices=["s2", "s3", "s4"],
        help="Scenario to run (default: s2)",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default=None,
        help="Comma-separated list of schedulers (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scheduler_comparison",
        help="Output directory (default: results/scheduler_comparison)",
    )

    args = parser.parse_args()

    # 스케줄러 목록
    scheduler_names = None
    if args.schedulers:
        scheduler_names = [s.strip() for s in args.schedulers.split(",")]

    # 시나리오 선택
    # 현재는 S2 (Log Analysis)만 지원
    # TODO: S3, S4 시나리오 추가

    from edgeagent.scenario_runner import ScenarioRunner

    # LogAnalysisScenario import (동적으로)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from run_scenario2_with_metrics import LogAnalysisScenario
        scenario_class = LogAnalysisScenario
    except ImportError:
        print("[ERROR] Could not import LogAnalysisScenario")
        print("Please ensure run_scenario2_with_metrics.py exists")
        return

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    output_dir = Path(args.output)

    print("=" * 60)
    print("Scheduler Comparison")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Schedulers: {scheduler_names or 'all'}")
    print(f"Output: {output_dir}")
    print()

    # 현재 버전에서는 Static scheduler만 테스트 가능
    # Scheduler 선택 기능은 middleware 수정이 필요함
    print("[NOTE] Current version only supports static scheduler.")
    print("       Full scheduler comparison requires middleware modification.")
    print()

    # 단일 스케줄러로 실행 (데모용)
    results = await compare_schedulers(
        scenario_class,
        config_path,
        output_dir,
        ["static"],  # 현재는 static만
    )

    print("\n[INFO] Comparison complete!")
    print("[TODO] To enable full scheduler comparison:")
    print("       1. Add 'scheduler' parameter to EdgeAgentMCPClient")
    print("       2. Use create_scheduler() in middleware")


if __name__ == "__main__":
    asyncio.run(main())
