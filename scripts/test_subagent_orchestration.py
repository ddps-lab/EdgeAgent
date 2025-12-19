#!/usr/bin/env python3
"""
Sub-Agent Orchestration Test Script

Sub-Agent 모드와 Legacy 모드를 비교 테스트합니다.
Scenario 2 (Log Analysis)를 사용합니다.

Usage:
    # 로컬 테스트 (HTTP 서버 없이)
    python scripts/test_subagent_orchestration.py --mode local

    # HTTP 서버 테스트 (별도 터미널에서 Sub-Agent 서버 실행 필요)
    python scripts/test_subagent_orchestration.py --mode http
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

from edgeagent import (
    SubAgentOrchestrator,
    OrchestrationConfig,
    ToolSequencePlanner,
    ToolRegistry,
    StaticScheduler,
)


async def test_planner():
    """Planner 기능 테스트"""
    print("=" * 70)
    print("Test 1: ToolSequencePlanner")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    registry = ToolRegistry.from_yaml(config_path)
    scheduler = StaticScheduler(config_path, registry)
    planner = ToolSequencePlanner(scheduler, registry)

    # Tool sequence
    tool_sequence = ["filesystem", "log_parser", "data_aggregate", "filesystem"]

    print(f"\nTool sequence: {tool_sequence}")
    print()

    # Location별 grouping (단순)
    location_groups = planner.plan_by_location(tool_sequence)
    print("Location groups (simple):")
    for loc, tools in location_groups.items():
        print(f"  {loc}: {tools}")
    print()

    # Execution plan (순서 유지)
    plan = planner.create_execution_plan(tool_sequence, preserve_order=True)
    print("Execution plan (preserve order):")
    for i, partition in enumerate(plan.partitions):
        print(f"  Partition {i + 1}: {partition.location} -> {partition.tools}")
    print()

    # Execution plan (순서 무시)
    plan_unordered = planner.create_execution_plan(tool_sequence, preserve_order=False)
    print("Execution plan (location only):")
    for i, partition in enumerate(plan_unordered.partitions):
        print(f"  Partition {i + 1}: {partition.location} -> {partition.tools}")
    print()

    print("✓ Planner test passed")
    return True


async def test_orchestrator_local():
    """Orchestrator 로컬 테스트 (HTTP 서버 없이)"""
    import sys

    print("=" * 70, flush=True)
    print("Test 2: SubAgentOrchestrator (Local Mode)", flush=True)
    print("=" * 70, flush=True)

    tools_config = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    print(f"Config: {tools_config}", flush=True)

    # Orchestration config (endpoint 없음 = 로컬 실행)
    config = OrchestrationConfig(
        mode="subagent",
        subagent_endpoints={},  # 빈 dict = 로컬 실행
        model="gpt-4o-mini",
        temperature=0,
        max_iterations=10,
    )

    print("Creating orchestrator...", flush=True)
    orchestrator = SubAgentOrchestrator(tools_config, config)
    print("Orchestrator created", flush=True)

    # Log 파일 준비
    device_log = Path("/tmp/edgeagent_device_hy/server.log")
    device_log.parent.mkdir(parents=True, exist_ok=True)

    # 항상 작은 테스트 로그 사용 (토큰 초과 방지)
    test_log = """2024-01-01 10:00:00,000 - test - INFO - Application started
2024-01-01 10:00:01,000 - test - WARNING - High memory usage
2024-01-01 10:00:02,000 - test - ERROR - Connection timeout
2024-01-01 10:00:03,000 - test - INFO - Retry attempt 1
2024-01-01 10:00:04,000 - test - ERROR - Connection failed
2024-01-01 10:00:05,000 - test - INFO - Shutdown
"""
    device_log.write_text(test_log)
    print(f"Created test log file: {device_log} ({len(test_log)} bytes)", flush=True)

    # 실행 계획 미리보기
    tool_sequence = ["filesystem", "log_parser", "data_aggregate"]
    print("\n--- Execution Plan Preview ---", flush=True)
    orchestrator.print_execution_plan(tool_sequence)

    # 실제 실행 (로컬 모드)
    print("\n--- Running Orchestrator (Local Sub-Agent) ---", flush=True)
    print("This will call LLM for each partition, may take 1-2 minutes...", flush=True)
    sys.stdout.flush()

    user_request = """
    Analyze the server log file at /tmp/edgeagent_device_hy/server.log.
    1. Read the log file using read_text_file
    2. Parse the logs using parse_logs with format_type='python'
    3. Compute statistics using compute_log_statistics
    Return a summary of the analysis.
    """

    print("Starting orchestrator context...", flush=True)
    async with orchestrator:
        print("Running orchestrator.run()...", flush=True)
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
            mode="subagent",
        )
    print("Orchestrator finished", flush=True)

    print("\n--- Result ---")
    print(f"Success: {result.success}")
    print(f"Mode: {result.mode}")
    print(f"Partitions executed: {result.partitions_executed}")
    print(f"Total tool calls: {result.total_tool_calls}")
    print(f"Execution time: {result.execution_time_ms:.2f} ms")

    if result.error:
        print(f"Error: {result.error}")

    if result.final_result:
        final_str = str(result.final_result)
        print(f"\nFinal result preview:\n{final_str[:500]}...")

    print()
    print("✓ Orchestrator local test passed" if result.success else "✗ Orchestrator local test failed")
    return result.success


async def test_compare_modes():
    """Legacy vs Sub-Agent 모드 비교"""
    print("=" * 70)
    print("Test 3: Compare Legacy vs Sub-Agent Mode")
    print("=" * 70)

    tools_config = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"

    # Log 파일 준비
    device_log = Path("/tmp/edgeagent_device_hy/server.log")
    if not device_log.exists():
        test_log = """2024-01-01 10:00:00,000 - test - INFO - Application started
2024-01-01 10:00:01,000 - test - WARNING - High memory usage
2024-01-01 10:00:02,000 - test - ERROR - Connection timeout
"""
        device_log.parent.mkdir(parents=True, exist_ok=True)
        device_log.write_text(test_log)

    user_request = """
    Read the log file at /tmp/edgeagent_device_hy/server.log,
    parse it with parse_logs (format_type='python'),
    and compute statistics with compute_log_statistics.
    """

    tool_sequence = ["filesystem", "log_parser"]

    results = {}

    for mode in ["legacy", "subagent"]:
        print(f"\n--- Running {mode.upper()} mode ---")

        config = OrchestrationConfig(
            mode=mode,
            subagent_endpoints={},
            model="gpt-4o-mini",
            temperature=0,
            max_iterations=10,
        )

        orchestrator = SubAgentOrchestrator(tools_config, config)

        async with orchestrator:
            result = await orchestrator.run(
                user_request=user_request,
                tool_sequence=tool_sequence,
                mode=mode,
            )

        results[mode] = result
        print(f"  Success: {result.success}")
        print(f"  Tool calls: {result.total_tool_calls}")
        print(f"  Time: {result.execution_time_ms:.2f} ms")

    print("\n--- Comparison ---")
    print(f"{'Mode':<12} {'Success':<10} {'Tool Calls':<12} {'Time (ms)':<12}")
    print("-" * 46)
    for mode, result in results.items():
        print(f"{mode:<12} {str(result.success):<10} {result.total_tool_calls:<12} {result.execution_time_ms:<12.2f}")

    print()
    all_success = all(r.success for r in results.values())
    print("✓ Mode comparison test passed" if all_success else "✗ Mode comparison test failed")
    return all_success


async def main():
    """메인 테스트 실행"""
    import sys

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        print("Create a .env file with OPENAI_API_KEY=your-key")
        return False

    print("\n" + "=" * 70, flush=True)
    print("EdgeAgent Sub-Agent Orchestration Test", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    results = []

    # Test 1: Planner (빠른 테스트)
    try:
        results.append(await test_planner())
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Planner test failed: {e}", flush=True)
        results.append(False)

    print(flush=True)

    # Test 2: Orchestrator (로컬) - LLM 호출 포함, 시간이 걸림
    print("Starting Orchestrator test (this may take 1-2 minutes)...", flush=True)
    try:
        result = await test_orchestrator_local()
        results.append(result)
    except Exception as e:
        print(f"✗ Orchestrator local test failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results.append(False)

    print(flush=True)

    # Test 3: 모드 비교 (시간이 오래 걸릴 수 있음)
    # try:
    #     results.append(await test_compare_modes())
    # except Exception as e:
    #     print(f"✗ Mode comparison test failed: {e}")
    #     results.append(False)

    print(flush=True)
    print("=" * 70, flush=True)
    print("Test Summary", flush=True)
    print("=" * 70, flush=True)
    print(f"Passed: {sum(results)}/{len(results)}", flush=True)

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
