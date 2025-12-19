#!/usr/bin/env python3
"""
Scenario 2: Log Analysis Pipeline - SubAgent Mode

This script supports both execution modes:
1. legacy: Single Agent with all tools (current approach)
2. subagent: Sub-Agent Orchestration (location-aware partitioning)

Usage:
    python scripts/run_scenario2_subagent.py --mode legacy
    python scripts/run_scenario2_subagent.py --mode subagent
    python scripts/run_scenario2_subagent.py --compare  # Run both and compare

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
from edgeagent.metrics import (
    print_orchestration_summary,
    save_orchestration_result,
    save_orchestration_metrics_csv,
)


def load_log_source() -> tuple[Path, str]:
    """Load log file source.

    Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
    """
    # Use unified path that works across all locations
    loghub_dir = Path("/edgeagent/data/scenario2/loghub_samples")

    if loghub_dir.exists():
        for log_name in ["apache_small.log", "apache_medium.log"]:
            candidate = loghub_dir / log_name
            if candidate.exists():
                return candidate, f"LogHub ({log_name})"

    raise FileNotFoundError(
        f"No log file found in {loghub_dir}\n"
        "Run 'python scripts/setup_test_data.py -s 2' for test data"
    )


LOG_SOURCE, DATA_SOURCE = load_log_source()


USER_REQUEST = f"""
Analyze the server log file at {LOG_SOURCE}.
1. Read the log file using read_text_file
2. Parse the logs using parse_logs with format_type='auto' to get entries
3. Compute statistics using compute_log_statistics with the entries
4. Write a summary report to /edgeagent/results/scenario2_log_report.md

Return the analysis summary.
"""

# Tool sequence for Sub-Agent mode (order matters)
TOOL_SEQUENCE = ["read_text_file", "parse_logs", "compute_log_statistics", "write_file"]


async def run_legacy_mode(config_path: Path, model: str, scenario_name: str = "log_analysis") -> dict:
    """Run with legacy single-agent mode

    Returns:
        dict 형태의 결과 (OrchestrationResult.to_dict()와 호환)
    """
    print("\n" + "=" * 70, flush=True)
    print("LEGACY MODE: Single Agent with All Tools", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    tool_calls = 0

    try:
        async with EdgeAgentMCPClient(config_path) as client:
            tools = await client.get_tools()
            print(f"Loaded {len(tools)} tools", flush=True)

            # gpt-4o-mini doesn't support temperature=0
            llm_kwargs = {"model": model}
            if "gpt-5" not in model:
                llm_kwargs["temperature"] = 0
            llm = ChatOpenAI(**llm_kwargs)
            agent = create_agent(llm, tools)

            print("Running agent...", flush=True)
            result_content = ""
            seen_tool_ids = set()

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

            return {
                "scenario_name": scenario_name,
                "mode": "legacy",
                "scheduler_type": "",
                "success": True,
                "total_time_ms": elapsed,
                "tool_calls": tool_calls,
                "partitions": 1,
                "partition_times": [elapsed],
                "partition_details": [],
                "placement_map": {},
                "chain_scheduling": {},
                "metrics_entries": [],
                "tool_call_count": tool_calls,
                "error": None,
                "final_result_preview": result_content[:500] if result_content else "",
            }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return {
            "scenario_name": scenario_name,
            "mode": "legacy",
            "scheduler_type": "",
            "success": False,
            "total_time_ms": elapsed,
            "tool_calls": tool_calls,
            "partitions": 0,
            "partition_times": [],
            "partition_details": [],
            "placement_map": {},
            "chain_scheduling": {},
            "metrics_entries": [],
            "tool_call_count": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


async def run_subagent_mode(config_path: Path, model: str, scheduler: str = "brute_force", scenario_name: str = "log_analysis") -> dict:
    """Run with Sub-Agent Orchestration mode

    Returns:
        OrchestrationResult.to_dict() 형태의 dict
    """
    print("\n" + "=" * 70)
    print(f"SUBAGENT MODE: Location-Aware Orchestration (scheduler={scheduler})")
    print("=" * 70)

    start_time = time.time()

    try:
        # Load subagent endpoints from config file
        config = OrchestrationConfig.from_yaml(config_path)
        config.mode = "subagent"
        config.model = model
        config.temperature = 0
        config.max_iterations = 10

        # Print loaded endpoints for verification
        if config.subagent_endpoints:
            print(f"\nLoaded SubAgent Endpoints:")
            for loc, ep in config.subagent_endpoints.items():
                print(f"  {loc}: {ep.host}:{ep.port}")

        system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"
        orchestrator = SubAgentOrchestrator(
            config_path,
            config,
            system_config_path=system_config_path,
            scheduler_type=scheduler,
        )

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

        # OrchestrationResult에 scenario_name 설정
        result.scenario_name = scenario_name
        result.execution_time_ms = elapsed

        return result.to_dict()

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return {
            "scenario_name": scenario_name,
            "mode": "subagent",
            "scheduler_type": scheduler,
            "success": False,
            "total_time_ms": elapsed,
            "tool_calls": 0,
            "partitions": 0,
            "partition_times": [],
            "partition_details": [],
            "placement_map": {},
            "chain_scheduling": {},
            "metrics_entries": [],
            "tool_call_count": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def save_result(result_dict: dict, output_dir: str = "results/scenario2_subagent"):
    """Save execution result to JSON and CSV files using metrics.py utilities"""
    scenario_name = result_dict.get("scenario_name", "log_analysis")

    # Save JSON
    save_orchestration_result(result_dict, output_dir, scenario_name)

    # Save CSV
    csv_path = Path(output_dir) / "metrics.csv"
    save_orchestration_metrics_csv(
        result_dict.get("metrics_entries", []),
        str(csv_path),
        scenario_name=scenario_name,
    )


def compare_results(legacy: dict, subagent: dict):
    """Compare legacy vs subagent results"""
    print("\n" + "=" * 70)
    print("COMPARISON: Legacy vs Sub-Agent")
    print("=" * 70)

    legacy_time = legacy.get("total_time_ms", legacy.get("execution_time_ms", 0))
    subagent_time = subagent.get("total_time_ms", subagent.get("execution_time_ms", 0))

    print(f"\n{'Metric':<25} {'Legacy':<20} {'Sub-Agent':<20}")
    print("-" * 65)
    print(f"{'Success':<25} {str(legacy.get('success')):<20} {str(subagent.get('success')):<20}")
    print(f"{'Total time (ms)':<25} {legacy_time:<20.0f} {subagent_time:<20.0f}")
    print(f"{'Tool calls':<25} {legacy.get('tool_calls', 0):<20} {subagent.get('tool_calls', 0):<20}")
    print(f"{'Partitions':<25} {'1 (all)':<20} {subagent.get('partitions', 0):<20}")

    if legacy.get("success") and subagent.get("success"):
        speedup = legacy_time / subagent_time if subagent_time > 0 else 0
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
    parser.add_argument(
        "--scheduler",
        choices=["brute_force", "static", "all_device", "all_edge", "all_cloud", "heuristic"],
        default="brute_force",
        help="Scheduler type (default: brute_force)",
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
    print(f"Scheduler: {args.scheduler}")

    # Prepare log file
    log_path = Path("/edgeagent/data/scenario2/server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(LOG_SOURCE.read_text())
    print(f"Prepared: {log_path} ({log_path.stat().st_size} bytes)")

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"

    if args.compare:
        # Run both modes
        legacy_result = await run_legacy_mode(config_path, args.model)
        print_orchestration_summary(legacy_result)
        save_result(legacy_result)

        subagent_result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_orchestration_summary(subagent_result)
        save_result(subagent_result)

        compare_results(legacy_result, subagent_result)

        return legacy_result.get("success", False) and subagent_result.get("success", False)

    elif args.mode == "legacy":
        result = await run_legacy_mode(config_path, args.model)
        print_orchestration_summary(result)
        save_result(result)
        return result.get("success", False)

    else:  # subagent
        result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_orchestration_summary(result)
        save_result(result)
        return result.get("success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
