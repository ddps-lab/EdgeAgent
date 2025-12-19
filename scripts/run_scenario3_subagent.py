#!/usr/bin/env python3
"""
Scenario 3: Research Assistant Pipeline - SubAgent Mode

This script supports both execution modes:
1. legacy: Single Agent with all tools (current approach)
2. subagent: Sub-Agent Orchestration (location-aware partitioning)

Usage:
    python scripts/run_scenario3_subagent.py --mode legacy
    python scripts/run_scenario3_subagent.py --mode subagent
    python scripts/run_scenario3_subagent.py --compare  # Run both and compare

Tool Chain:
    fetch -> summarize -> data_aggregate -> filesystem(write)
    EDGE    EDGE        EDGE              DEVICE
"""

import argparse
import asyncio
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
    scheduler_type: str = ""
    metrics_entries: list[dict] = field(default_factory=list)
    placement_map: dict = field(default_factory=dict)


def load_research_urls() -> tuple[list[str], str]:
    """Load research URLs from S2ORC data or use defaults.

    Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
    """
    # Use unified path that works across all locations
    data_dir = Path("/edgeagent/data/scenario3")
    s2orc_file = data_dir / "s2orc_papers.json"

    # Check for S2ORC papers
    if s2orc_file.exists():
        import json
        try:
            with open(s2orc_file) as f:
                papers = json.load(f)
            urls = []
            for paper in papers[:3]:  # Use first 3 papers
                if paper.get("url"):
                    urls.append(paper["url"])
            if urls:
                return urls, "S2ORC papers"
        except Exception:
            pass

    # Default Wikipedia URLs (reliable and fast)
    default_urls = [
        "https://en.wikipedia.org/wiki/Intelligent_agent",
        "https://en.wikipedia.org/wiki/Large_language_model",
    ]
    return default_urls, "Default Wikipedia URLs"


RESEARCH_URLS, DATA_SOURCE = load_research_urls()


USER_REQUEST = f"""
Research the topic of AI agents using these URLs:
{chr(10).join(f'- {url}' for url in RESEARCH_URLS)}

Please:
1. Fetch content from each URL using the fetch tool
2. Summarize the content from each page using summarize_text
3. Aggregate the summaries into a cohesive research summary
4. Write a research report to /edgeagent/results/scenario3_research_report.md

Return a summary of the research findings.
"""

# Tool sequence for Sub-Agent mode (order matters)
# 개별 tool 이름 사용 - Scheduler가 정확한 profile 참조 가능
TOOL_SEQUENCE = ["fetch", "summarize_text", "combine_research_results", "write_file"]


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

            return ExecutionResult(
                mode="legacy",
                success=True,
                total_time_ms=elapsed,
                tool_calls=tool_calls,
                final_result=result_content[:500] if result_content else "",
            )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return ExecutionResult(
            mode="legacy",
            success=False,
            total_time_ms=elapsed,
            tool_calls=tool_calls,
            error=f"{e}\n{traceback.format_exc()}",
        )


async def run_subagent_mode(config_path: Path, model: str, scheduler: str = "brute_force") -> ExecutionResult:
    """Run with Sub-Agent Orchestration mode"""
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
        config.max_iterations = 15  # More iterations for multiple URLs

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

        # Extract partition times and metrics
        partition_times = []
        metrics_entries = []
        if result.partition_results:
            for pr in result.partition_results:
                if "execution_time_ms" in pr:
                    partition_times.append(pr["execution_time_ms"])
                if "metrics_entries" in pr:
                    metrics_entries.extend(pr["metrics_entries"])

        # Get placement map from execution plan
        placement_map = {}
        plan = orchestrator.get_execution_plan(TOOL_SEQUENCE)
        if plan.chain_scheduling_result:
            placement_map = {p.tool_name: p.location for p in plan.chain_scheduling_result.placements}
        else:
            for partition in plan.partitions:
                for tool in partition.tools:
                    placement_map[tool] = partition.location

        return ExecutionResult(
            mode="subagent",
            success=result.success,
            total_time_ms=elapsed,
            tool_calls=result.total_tool_calls,
            partitions=result.partitions_executed,
            partition_times=partition_times,
            final_result=str(result.final_result)[:500] if result.final_result else "",
            error=result.error or "",
            scheduler_type=scheduler,
            metrics_entries=metrics_entries,
            placement_map=placement_map,
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
    if result.scheduler_type:
        print(f"Scheduler: {result.scheduler_type}")
    print(f"Success: {result.success}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Metrics entries: {len(result.metrics_entries)}")

    if result.partitions > 0:
        print(f"Partitions: {result.partitions}")
        if result.partition_times:
            print(f"Partition times: {[f'{t:.0f}ms' for t in result.partition_times]}")

    if result.placement_map:
        print(f"\nPlacement Map:")
        for tool, location in result.placement_map.items():
            print(f"  {tool:25} -> {location}")

    if result.error:
        print(f"\nError: {result.error[:300]}")

    if result.final_result:
        print(f"\nResult preview:\n{result.final_result[:300]}...")


def save_metrics_csv(metrics_entries: list[dict], output_path: Path, scenario_name: str = ""):
    """Save metrics entries to CSV file (flattened format)"""
    import csv
    import uuid

    if not metrics_entries:
        return

    session_id = str(uuid.uuid4())[:8]
    rows = []

    for e in metrics_entries:
        timing = e.get("timing", {})
        location = e.get("location", {})
        scheduling = e.get("scheduling", {})
        data_flow = e.get("data_flow", {})
        resource = e.get("resource", {})
        status = e.get("status", {})

        rows.append({
            "session_id": session_id,
            "scenario_name": scenario_name,
            "tool_name": e.get("tool_name", ""),
            "parent_tool_name": e.get("parent_tool_name", ""),
            "pipeline_step": e.get("pipeline_step", 0),
            "timestamp": e.get("timestamp", 0),
            "latency_ms": timing.get("latency_ms", 0),
            "inter_tool_latency_ms": timing.get("inter_tool_latency_ms", 0),
            "scheduled_location": location.get("scheduled_location", ""),
            "actual_location": location.get("actual_location", ""),
            "fallback_occurred": location.get("fallback_occurred", False),
            "scheduling_decision_time_ns": scheduling.get("decision_time_ns", 0),
            "scheduling_reason": scheduling.get("reason", ""),
            "scheduling_score": scheduling.get("score", 0),
            "exec_cost": scheduling.get("exec_cost", 0),
            "trans_cost": scheduling.get("trans_cost", 0),
            "fixed_location": scheduling.get("fixed", ""),
            "input_size_bytes": data_flow.get("input_size_bytes", 0),
            "output_size_bytes": data_flow.get("output_size_bytes", 0),
            "reduction_ratio": data_flow.get("reduction_ratio", 0),
            "data_flow_type": data_flow.get("data_flow_type", ""),
            "mcp_serialization_time_ms": timing.get("mcp_serialization_time_ms", 0),
            "mcp_deserialization_time_ms": timing.get("mcp_deserialization_time_ms", 0),
            "memory_delta_bytes": resource.get("memory_delta_bytes", 0),
            "cpu_time_user_ms": resource.get("cpu_time_user_ms", 0),
            "cpu_time_system_ms": resource.get("cpu_time_system_ms", 0),
            "success": status.get("success", True),
            "retry_count": status.get("retry_count", 0),
            "error": status.get("error", ""),
        })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def save_result(result: ExecutionResult, output_dir: str = "results/scenario3_subagent"):
    """Save execution result to JSON and CSV files"""
    import time as time_module
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_dict = {
        "scenario_name": "research_assistant",
        "mode": result.mode,
        "scheduler_type": result.scheduler_type,
        "success": result.success,
        "total_time_ms": result.total_time_ms,
        "tool_calls": result.tool_calls,
        "partitions": result.partitions,
        "partition_times": result.partition_times,
        "placement_map": result.placement_map,
        "metrics_entries": result.metrics_entries,
        "tool_call_count": len(result.metrics_entries),
        "error": result.error if result.error else None,
    }

    output_path = Path(output_dir) / f"research_assistant_{result.mode}_{int(time_module.time())}.json"
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Save CSV
    csv_path = Path(output_dir) / "metrics.csv"
    save_metrics_csv(result.metrics_entries, csv_path, scenario_name="research_assistant")
    print(f"Metrics CSV saved to: {csv_path}")


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
        description="Run Scenario 3 with different orchestration modes"
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

    # Ensure output directory exists
    Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Scenario 3: Research Assistant Pipeline")
    print("=" * 70)
    print(f"Data Source: {DATA_SOURCE}")
    print(f"URLs to research: {len(RESEARCH_URLS)}")
    for url in RESEARCH_URLS:
        print(f"  - {url}")
    print(f"Model: {args.model}")
    print(f"Scheduler: {args.scheduler}")

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"

    if args.compare:
        # Run both modes
        legacy_result = await run_legacy_mode(config_path, args.model)
        print_result(legacy_result)
        save_result(legacy_result)

        subagent_result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_result(subagent_result)
        save_result(subagent_result)

        compare_results(legacy_result, subagent_result)

        return legacy_result.success and subagent_result.success

    elif args.mode == "legacy":
        result = await run_legacy_mode(config_path, args.model)
        print_result(result)
        save_result(result)
        return result.success

    else:  # subagent
        result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_result(result)
        save_result(result)
        return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
