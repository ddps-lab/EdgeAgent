#!/usr/bin/env python3
"""
Scenario 2: Log Analysis Pipeline - Orchestration Mode Comparison

This script supports both execution modes:
1. legacy: Single Agent with all tools (current approach)
2. subagent: Sub-Agent Orchestration (location-aware partitioning)

Usage:
    python scripts/run_scenario2_orchestrated.py --mode legacy
    python scripts/run_scenario2_orchestrated.py --mode subagent
    python scripts/run_scenario2_orchestrated.py --compare  # Run both and compare

Tool Chain:
    filesystem(read) -> log_parser -> data_aggregate -> filesystem(write)
    DEVICE            EDGE         EDGE             DEVICE
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from edgeagent import (
    EdgeAgentMCPClient,
    SubAgentOrchestrator,
    OrchestrationConfig,
)


@dataclass
class ExecutionResult:
    """Execution result for comparison"""
    mode: str
    success: bool
    total_time_ms: float
    tool_calls: int
    partitions: int = 0
    partition_times: list[float] = field(default_factory=list)
    final_result: str = ""
    error: str = ""


def load_log_source() -> tuple[Path, str]:
    """Load log file source"""
    data_dir = Path(__file__).parent.parent / "data" / "scenario2"
    loghub_dir = data_dir / "loghub_samples"
    sample_log = data_dir / "server.log"

    if loghub_dir.exists():
        for log_name in ["small_python.log", "medium_python.log"]:
            candidate = loghub_dir / log_name
            if candidate.exists():
                return candidate, f"LogHub ({log_name})"

    if sample_log.exists():
        return sample_log, "Sample server.log"

    # Create minimal test log if nothing exists
    test_log = Path("/tmp/edgeagent_device/server.log")
    test_log.parent.mkdir(parents=True, exist_ok=True)
    test_log.write_text("""2024-01-01 10:00:00,000 - root - INFO - Application started
2024-01-01 10:00:01,000 - root - WARNING - High memory usage detected
2024-01-01 10:00:02,000 - root - ERROR - Connection timeout to database
2024-01-01 10:00:03,000 - root - INFO - Retry attempt 1
2024-01-01 10:00:04,000 - root - ERROR - Connection failed after retry
2024-01-01 10:00:05,000 - root - INFO - Graceful shutdown initiated
""")
    return test_log, "Generated test log"


LOG_SOURCE, DATA_SOURCE = load_log_source()


USER_REQUEST = """
Analyze the server log file at /tmp/edgeagent_device/server.log.
1. Read the log file using read_text_file
2. Parse the logs using parse_logs with format_type='python' to get entries
3. Compute statistics using compute_log_statistics with the entries
4. Write a summary report to /tmp/edgeagent_device/log_report.md

Return the analysis summary.
"""

# Tool sequence for Sub-Agent mode (order matters)
TOOL_SEQUENCE = ["filesystem", "log_parser", "data_aggregate", "filesystem"]


async def run_legacy_mode(config_path: Path, model: str) -> ExecutionResult:
    """Run with legacy single-agent mode"""
    print("\n" + "=" * 70, flush=True)
    print("LEGACY MODE: Single Agent with All Tools", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    tool_calls = 0

    try:
        async with EdgeAgentMCPClient(config_path) as client:
            tools = await client.get_tools()
            print(f"Loaded {len(tools)} tools", flush=True)

            llm = ChatOpenAI(model=model, temperature=0)
            agent = create_agent(llm, tools)

            print("Running agent...", flush=True)
            result_content = ""
            seen_tool_ids = set()  # 중복 방지용

            async for chunk in agent.astream(
                {"messages": [("user", USER_REQUEST)]},
                stream_mode="values",
            ):
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tc_id = tc.get("id", "")
                                if tc_id and tc_id not in seen_tool_ids:
                                    seen_tool_ids.add(tc_id)
                                    tool_calls += 1
                                    print(f"  -> Tool: {tc.get('name', 'unknown')}", flush=True)

                    final_msg = chunk["messages"][-1]
                    if hasattr(final_msg, "content"):
                        result_content = final_msg.content

            elapsed = (time.time() - start_time) * 1000

            return ExecutionResult(
                mode="legacy",
                success=True,
                total_time_ms=elapsed,
                tool_calls=tool_calls,
                final_result=result_content[:500] if result_content else "",
            )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return ExecutionResult(
            mode="legacy",
            success=False,
            total_time_ms=elapsed,
            tool_calls=tool_calls,
            error=str(e),
        )


async def run_subagent_mode(config_path: Path, model: str) -> ExecutionResult:
    """Run with Sub-Agent Orchestration mode"""
    print("\n" + "=" * 70)
    print("SUBAGENT MODE: Location-Aware Orchestration")
    print("=" * 70)

    start_time = time.time()

    try:
        config = OrchestrationConfig(
            mode="subagent",
            subagent_endpoints={},  # Local execution
            model=model,
            temperature=0,
            max_iterations=10,
        )

        orchestrator = SubAgentOrchestrator(config_path, config)

        # Show execution plan
        print("\nExecution Plan:")
        orchestrator.print_execution_plan(TOOL_SEQUENCE)

        print("\nRunning orchestration...")

        async with orchestrator:
            result = await orchestrator.run(
                user_request=USER_REQUEST,
                tool_sequence=TOOL_SEQUENCE,
                mode="subagent",
            )

        elapsed = (time.time() - start_time) * 1000

        # Extract partition times
        partition_times = []
        if result.partition_results:
            for pr in result.partition_results:
                if "execution_time_ms" in pr:
                    partition_times.append(pr["execution_time_ms"])

        return ExecutionResult(
            mode="subagent",
            success=result.success,
            total_time_ms=elapsed,
            tool_calls=result.total_tool_calls,
            partitions=result.partitions_executed,
            partition_times=partition_times,
            final_result=str(result.final_result)[:500] if result.final_result else "",
            error=result.error or "",
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return ExecutionResult(
            mode="subagent",
            success=False,
            total_time_ms=elapsed,
            tool_calls=0,
            error=f"{e}\n{traceback.format_exc()}",
        )


def print_result(result: ExecutionResult):
    """Print execution result"""
    print(f"\n--- {result.mode.upper()} Result ---")
    print(f"Success: {result.success}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Tool calls: {result.tool_calls}")

    if result.partitions > 0:
        print(f"Partitions: {result.partitions}")
        if result.partition_times:
            print(f"Partition times: {[f'{t:.0f}ms' for t in result.partition_times]}")

    if result.error:
        print(f"Error: {result.error[:300]}")

    if result.final_result:
        print(f"\nResult preview:\n{result.final_result[:300]}...")


def compare_results(legacy: ExecutionResult, subagent: ExecutionResult):
    """Compare legacy vs subagent results"""
    print("\n" + "=" * 70)
    print("COMPARISON: Legacy vs Sub-Agent")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Legacy':<20} {'Sub-Agent':<20}")
    print("-" * 65)
    print(f"{'Success':<25} {str(legacy.success):<20} {str(subagent.success):<20}")
    print(f"{'Total time (ms)':<25} {legacy.total_time_ms:<20.0f} {subagent.total_time_ms:<20.0f}")
    print(f"{'Tool calls':<25} {legacy.tool_calls:<20} {subagent.tool_calls:<20}")
    print(f"{'Partitions':<25} {'1 (all)':<20} {subagent.partitions:<20}")

    if legacy.success and subagent.success:
        speedup = legacy.total_time_ms / subagent.total_time_ms if subagent.total_time_ms > 0 else 0
        print(f"\n{'Speedup':<25} {speedup:.2f}x")

        if speedup > 1:
            print(f"Sub-Agent mode is {speedup:.1f}x faster")
        else:
            print(f"Legacy mode is {1/speedup:.1f}x faster")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Scenario 2 with different orchestration modes"
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "subagent"],
        default="subagent",
        help="Execution mode (default: subagent)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and compare",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        return False

    print("=" * 70)
    print("Scenario 2: Log Analysis Pipeline")
    print("=" * 70)
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Log file: {LOG_SOURCE}")
    print(f"Model: {args.model}")

    # Prepare log file
    device_log = Path("/tmp/edgeagent_device/server.log")
    device_log.parent.mkdir(parents=True, exist_ok=True)
    device_log.write_text(LOG_SOURCE.read_text())
    print(f"Prepared: {device_log} ({device_log.stat().st_size} bytes)")

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"

    if args.compare:
        # Run both modes
        legacy_result = await run_legacy_mode(config_path, args.model)
        print_result(legacy_result)

        subagent_result = await run_subagent_mode(config_path, args.model)
        print_result(subagent_result)

        compare_results(legacy_result, subagent_result)

        return legacy_result.success and subagent_result.success

    elif args.mode == "legacy":
        result = await run_legacy_mode(config_path, args.model)
        print_result(result)
        return result.success

    else:  # subagent
        result = await run_subagent_mode(config_path, args.model)
        print_result(result)
        return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
