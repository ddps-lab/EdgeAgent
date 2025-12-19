#!/usr/bin/env python3
"""
Scenario 2: Log Analysis Pipeline - With Chain Scheduling

Tool Chain:
    filesystem(read) -> log_parser -> data_aggregate -> filesystem(write)

Mode: script
- Tool sequence is STATIC (predefined)
- Scheduler runs FIRST via schedule_chain() to determine optimal placement
- get_backend_tools(placement_map) 사용: 필요한 서버만 연결, MetricsWrappedTool 반환
- ProxyTool 없음 (스케줄링 오버헤드 제거)
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import EdgeAgentMCPClient
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import create_scheduler
from edgeagent.metrics import MetricsCollector, MetricsConfig, print_chain_scheduling_result


# Tool Chain 정의 (orchestrated와 동일)
TOOL_CHAIN = [
    "read_text_file",
    "parse_logs",
    "compute_log_statistics",
    "write_file",
]


def parse_tool_result(result):
    """Parse tool result - handle MCP response format, dict, and JSON string."""
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and 'text' in first_item:
            text_content = first_item['text']
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"raw": text_content}
        if isinstance(first_item, dict):
            return first_item
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return {"raw": str(result)}


async def run_log_analysis(
    config_path: Path,
    system_config_path: Path,
    scheduler_type: str = "brute_force",
    output_dir: str = "results/scenario2",
) -> dict:
    """
    Log Analysis Pipeline 실행 (Chain Scheduling 적용)

    핵심 설계:
    1. 스케줄러가 먼저 실행 (MCP 연결 없이) -> 최적 배치 결정
    2. get_backend_tools(placement_map)으로 필요한 서버만 연결
    3. MetricsWrappedTool 사용 (ProxyTool 오버헤드 없음)
    """
    start_time = time.time()

    # 경로 설정 (모든 location에서 동일한 구조)
    log_path = "/edgeagent/data/scenario2/server.log"
    report_path = "/edgeagent/results/scenario2_log_report.md"

    # 디렉토리 생성
    Path("/edgeagent/data/scenario2").mkdir(parents=True, exist_ok=True)
    Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: 스케줄러 먼저 실행 (MCP 연결 없이)
    # ================================================================
    registry = ToolRegistry.from_yaml(config_path)

    # 모든 scheduler_type에서 create_scheduler()와 schedule_chain() 사용
    chain_scheduler = create_scheduler(
        scheduler_type,
        config_path,
        registry,
        system_config_path=system_config_path,
    )
    scheduling_result = chain_scheduler.schedule_chain(TOOL_CHAIN)

    # Chain Scheduling 결과 출력 (metrics.py 유틸리티 사용)
    print_chain_scheduling_result(
        scheduling_result,
        title=f"Step 1: Chain Scheduling ({scheduler_type})",
    )
    print()

    placement_map = {p.tool_name: p.location for p in scheduling_result.placements}

    # ================================================================
    # Step 2: get_backend_tools()로 필요한 서버만 연결
    # ================================================================
    print("=" * 70)
    print("Step 2: Connect required servers only")
    print("=" * 70)

    async with EdgeAgentMCPClient(
        config_path,
        scheduler=chain_scheduler,
        system_config_path=system_config_path,
        collect_metrics=True,
    ) as client:
        # Chain Scheduling 결과 설정 (개별 tool 메트릭에 score, exec_cost 등 기록)
        client.set_chain_scheduling_result(scheduling_result)
        # placement_map 기반으로 필요한 서버만 연결
        tool_by_name = await client.get_backend_tools(placement_map)
        print(f"  Loaded {len(tool_by_name)} tools (MetricsWrappedTool)")
        for name, tool in tool_by_name.items():
            print(f"    {name} -> {tool.location}")
        print()

        def get_tool(tool_name: str):
            return tool_by_name.get(tool_name)

        # ================================================================
        # Step 3: Prepare data
        # ================================================================
        print("=" * 70)
        print("Step 3: Prepare data")
        print("=" * 70)

        data_dir = Path(__file__).parent.parent / "data" / "scenario2"
        loghub_dir = data_dir / "loghub_samples"
        sample_log = data_dir / "server.log"

        log_file = None
        data_source = None

        if loghub_dir.exists():
            for log_name in ["medium_python.log", "small_python.log"]:
                candidate = loghub_dir / log_name
                if candidate.exists():
                    log_file = candidate
                    data_source = f"LogHub ({log_name})"
                    break

        if log_file is None and sample_log.exists():
            log_file = sample_log
            data_source = "Sample server.log"

        if log_file is None:
            raise FileNotFoundError(f"No log file found in {data_dir}")

        print(f"  Data Source: {data_source}")
        print(f"  Log file size: {log_file.stat().st_size:,} bytes")

        target_log = Path(log_path)
        target_log.write_text(log_file.read_text())
        print()

        # ================================================================
        # Step 4: Execute pipeline (orchestrated와 동일한 TOOL_CHAIN)
        # read_text_file -> parse_logs -> compute_log_statistics -> write_file
        # ================================================================
        print("=" * 70)
        print("Step 4: Execute pipeline")
        print("=" * 70)

        # 4.1: read_text_file
        tool_name = "read_text_file"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        read_tool = get_tool("read_text_file")
        log_content = await read_tool.ainvoke({"path": str(target_log)})
        print(f"    Read {len(str(log_content))} chars")

        # 4.2: parse_logs
        tool_name = "parse_logs"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        parse_tool = get_tool("parse_logs")
        raw_parsed = await parse_tool.ainvoke({
            "log_content": str(log_content),
            "format_type": "python"
        })
        parsed = parse_tool_result(raw_parsed)
        print(f"    Parsed {parsed.get('parsed_count', 0)} entries")

        # 4.3: compute_log_statistics
        tool_name = "compute_log_statistics"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        stats_tool = get_tool("compute_log_statistics")
        raw_stats = await stats_tool.ainvoke({"entries": parsed.get("entries", [])})
        stats = parse_tool_result(raw_stats)
        print(f"    By level: {stats.get('by_level', {})}")

        # 4.4: write_file
        tool_name = "write_file"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        report = f"""# Log Analysis Report

## Summary
- Total entries: {parsed.get('parsed_count', 0)}
- Format: {parsed.get('format_detected', 'N/A')}

## By Level
{chr(10).join(f"- {k}: {v}" for k, v in stats.get('by_level', {}).items())}
"""

        write_tool = get_tool("write_file")
        await write_tool.ainvoke({"path": report_path, "content": report})
        print(f"    Written to {report_path}")

        # Extract metrics before context closes
        metrics_entries = []
        metrics_collector = client.get_metrics()
        if metrics_collector:
            metrics_entries = [e.to_dict() for e in metrics_collector.entries]
            # Save CSV
            csv_path = Path(output_dir) / "metrics.csv"
            metrics_collector.save_csv(str(csv_path))
            print(f"  Metrics CSV saved to: {csv_path}")

    # ================================================================
    # Step 5: Summary
    # ================================================================
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    used_locations = set(placement_map.values())

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Scheduler: {scheduler_type}")
    print(f"  Success: True")
    print(f"  Total Time: {total_time_ms:.2f} ms")
    print(f"  Locations used: {used_locations}")
    print(f"  Tool calls: {len(metrics_entries)}")
    print("=" * 70)

    # Save results
    result = {
        "scenario_name": "log_analysis",
        "scheduler_type": scheduler_type,
        "success": True,
        "total_time_ms": total_time_ms,
        "used_locations": list(used_locations),
        "placement_map": placement_map,
        "chain_scheduling": {
            "total_cost": scheduling_result.total_score,
            "search_space_size": scheduling_result.search_space_size,
            "decision_time_ns": scheduling_result.decision_time_ns,
            "decision_time_ms": scheduling_result.decision_time_ns / 1e6,
        },
        "metrics_entries": metrics_entries,
        "tool_call_count": len(metrics_entries),
    }

    output_path = Path(output_dir) / f"log_analysis_{int(start_time)}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default="brute_force")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    print("=" * 70)
    print("Scenario 2: Log Analysis Pipeline")
    print("=" * 70)
    print(f"Scheduler: {args.scheduler}")
    print()

    result = await run_log_analysis(
        config_path=config_path,
        system_config_path=system_config_path,
        scheduler_type=args.scheduler,
    )

    return result.get("success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
