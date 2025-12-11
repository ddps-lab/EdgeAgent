#!/usr/bin/env python3
"""
Run All EdgeAgent Scenarios

Unified experiment runner that executes all scenarios across different execution modes
and generates a comprehensive comparison report.

Execution Modes:
- script: Hardcoded sequential tool calls (baseline orchestration)
- agent: LLM Agent autonomously selects tools (single agent with all tools)
- subagent: Location-aware partitioned execution (Sub-Agent orchestration)
- subagent_legacy: Single agent baseline for SubAgent comparison

Scenarios:
1. Code Review Pipeline (S1)
2. Log Analysis Pipeline (S2)
3. Research Assistant Pipeline (S3)
4. Image Processing Pipeline (S4)

Usage:
    python scripts/run_all_scenarios.py                    # Run all (script only)
    python scripts/run_all_scenarios.py --include-agent    # Run script + agent versions
    python scripts/run_all_scenarios.py --include-subagent # Run script + subagent versions
    python scripts/run_all_scenarios.py --scenarios 1,2    # Run specific scenarios
    python scripts/run_all_scenarios.py --agent-only       # Run agent versions only
    python scripts/run_all_scenarios.py --subagent-only    # Run subagent versions only
    python scripts/run_all_scenarios.py --all-modes        # Run all modes (script, agent, subagent)
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ScenarioSummary:
    """Summary of a single scenario run"""
    scenario_name: str
    version: str  # "script" or "agent"
    description: str
    success: bool
    total_latency_ms: float
    tool_calls: int
    input_bytes: int
    output_bytes: int
    reduction_ratio: float
    locations: dict[str, int]
    data_source: str = ""
    error: Optional[str] = None


# =============================================================================
# Script Version Runners
# =============================================================================

async def run_scenario1_script() -> ScenarioSummary:
    """Run Scenario 1: Code Review Pipeline (Script Version)"""
    from run_scenario1_with_metrics import CodeReviewScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
    scenario = CodeReviewScenario(config_path=config_path, output_dir="results/scenario1")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S1: Code Review",
        version="script",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario2_script() -> ScenarioSummary:
    """Run Scenario 2: Log Analysis Pipeline (Script Version)"""
    from run_scenario2_with_metrics import LogAnalysisScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    scenario = LogAnalysisScenario(config_path=config_path, output_dir="results/scenario2")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S2: Log Analysis",
        version="script",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario3_script() -> ScenarioSummary:
    """Run Scenario 3: Research Assistant Pipeline (Script Version)"""
    from run_scenario3_with_metrics import ResearchAssistantScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
    scenario = ResearchAssistantScenario(
        config_path=config_path,
        output_dir="results/scenario3",
        max_urls=3,
    )

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S3: Research Assistant",
        version="script",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario4_script() -> ScenarioSummary:
    """Run Scenario 4: Image Processing Pipeline (Script Version)"""
    from run_scenario4_with_metrics import ImageProcessingScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"
    scenario = ImageProcessingScenario(config_path=config_path, output_dir="results/scenario4")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S4: Image Processing",
        version="script",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


# =============================================================================
# Agent Version Runners
# =============================================================================

async def run_scenario1_agent() -> ScenarioSummary:
    """Run Scenario 1: Code Review Pipeline (Agent Version)"""
    from run_scenario1_agent import AgentCodeReviewScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
    scenario = AgentCodeReviewScenario(config_path=config_path, output_dir="results/scenario1_agent")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S1: Code Review",
        version="agent",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario2_agent() -> ScenarioSummary:
    """Run Scenario 2: Log Analysis Pipeline (Agent Version)"""
    from run_scenario2_agent import AgentLogAnalysisScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    scenario = AgentLogAnalysisScenario(config_path=config_path, output_dir="results/scenario2_agent")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S2: Log Analysis",
        version="agent",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario3_agent() -> ScenarioSummary:
    """Run Scenario 3: Research Assistant Pipeline (Agent Version)"""
    from run_scenario3_agent import AgentResearchAssistantScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
    scenario = AgentResearchAssistantScenario(config_path=config_path, output_dir="results/scenario3_agent")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S3: Research Assistant",
        version="agent",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


async def run_scenario4_agent() -> ScenarioSummary:
    """Run Scenario 4: Image Processing Pipeline (Agent Version)"""
    from run_scenario4_agent import AgentImageProcessingScenario

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"
    scenario = AgentImageProcessingScenario(config_path=config_path, output_dir="results/scenario4_agent")

    result = await scenario.run(save_results=True, print_summary=False)

    data_source = ""
    if result.metrics and result.metrics.custom_metrics:
        data_source = result.metrics.custom_metrics.get("data_source", "")

    return ScenarioSummary(
        scenario_name="S4: Image Processing",
        version="agent",
        description=scenario.description,
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_call_count,
        input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
        output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
        reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
        locations=result.metrics.call_count_by_location() if result.metrics else {},
        data_source=data_source,
        error=result.error,
    )


# =============================================================================
# SubAgent Version Runners
# =============================================================================

async def run_scenario1_subagent() -> ScenarioSummary:
    """Run Scenario 1: Code Review Pipeline (SubAgent Version)"""
    from run_scenario_subagent import run_scenario_subagent, SubAgentScenarioResult

    result = await run_scenario_subagent(
        scenario_num=1,
        mode="subagent",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S1: Code Review", "subagent")


async def run_scenario1_subagent_legacy() -> ScenarioSummary:
    """Run Scenario 1: Code Review Pipeline (SubAgent Legacy Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=1,
        mode="subagent_legacy",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S1: Code Review", "subagent_legacy")


async def run_scenario2_subagent() -> ScenarioSummary:
    """Run Scenario 2: Log Analysis Pipeline (SubAgent Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=2,
        mode="subagent",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S2: Log Analysis", "subagent")


async def run_scenario2_subagent_legacy() -> ScenarioSummary:
    """Run Scenario 2: Log Analysis Pipeline (SubAgent Legacy Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=2,
        mode="subagent_legacy",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S2: Log Analysis", "subagent_legacy")


async def run_scenario3_subagent() -> ScenarioSummary:
    """Run Scenario 3: Research Assistant Pipeline (SubAgent Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=3,
        mode="subagent",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S3: Research Assistant", "subagent")


async def run_scenario3_subagent_legacy() -> ScenarioSummary:
    """Run Scenario 3: Research Assistant Pipeline (SubAgent Legacy Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=3,
        mode="subagent_legacy",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S3: Research Assistant", "subagent_legacy")


async def run_scenario4_subagent() -> ScenarioSummary:
    """Run Scenario 4: Image Processing Pipeline (SubAgent Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=4,
        mode="subagent",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S4: Image Processing", "subagent")


async def run_scenario4_subagent_legacy() -> ScenarioSummary:
    """Run Scenario 4: Image Processing Pipeline (SubAgent Legacy Version)"""
    from run_scenario_subagent import run_scenario_subagent

    result = await run_scenario_subagent(
        scenario_num=4,
        mode="subagent_legacy",
        save_results=True,
        print_summary=False,
    )

    return _subagent_result_to_summary(result, "S4: Image Processing", "subagent_legacy")


def _subagent_result_to_summary(result, scenario_name: str, version: str) -> ScenarioSummary:
    """Convert SubAgentScenarioResult to ScenarioSummary"""
    # Get detailed metrics - 새로운 통일된 형식
    detailed = result.get_detailed_metrics()
    summary = detailed.get("summary", {})
    data_flow = summary.get("data_flow", {})

    # 새 형식: cumulative_input_bytes (Script와 동일)
    input_bytes = data_flow.get("cumulative_input_bytes", 0)
    output_bytes = data_flow.get("cumulative_output_bytes", 0)
    reduction_ratio = data_flow.get("overall_reduction_ratio", 0)

    # Location 분포
    location_dist = summary.get("location_distribution", {}).get("call_count_by_location", {})

    return ScenarioSummary(
        scenario_name=scenario_name,
        version=version,
        description=f"{scenario_name} with SubAgent Orchestration",
        success=result.success,
        total_latency_ms=result.total_latency_ms,
        tool_calls=result.tool_calls,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        reduction_ratio=reduction_ratio,
        locations={**location_dist, "partitions": result.partitions_executed},
        data_source=result.data_source,
        error=result.error,
    )


# =============================================================================
# Comparison and Reporting
# =============================================================================

def print_comparison_table(summaries: list[ScenarioSummary]):
    """Print comparison table of all scenarios"""
    print()
    print("=" * 120)
    print("All Scenarios Comparison")
    print("=" * 120)
    print()

    # Header
    print(f"{'Scenario':<25} {'Version':<8} {'Status':<8} {'Latency':>12} {'Calls':>8} {'Input':>12} {'Output':>10} {'Data Source':<25}")
    print("-" * 120)

    for s in summaries:
        status = "PASS" if s.success else "FAIL"
        data_src = s.data_source[:24] if s.data_source else ""
        print(f"{s.scenario_name:<25} {s.version:<8} {status:<8} {s.total_latency_ms:>10.1f}ms {s.tool_calls:>8} "
              f"{s.input_bytes:>10,}B {s.output_bytes:>8,}B {data_src:<25}")

    print("-" * 120)

    # Totals by version
    script_summaries = [s for s in summaries if s.version == "script"]
    agent_summaries = [s for s in summaries if s.version == "agent"]
    subagent_summaries = [s for s in summaries if s.version == "subagent"]
    subagent_legacy_summaries = [s for s in summaries if s.version == "subagent_legacy"]

    if script_summaries:
        script_latency = sum(s.total_latency_ms for s in script_summaries)
        script_calls = sum(s.tool_calls for s in script_summaries)
        script_success = sum(1 for s in script_summaries if s.success)
        print(f"{'SCRIPT TOTAL':<25} {'':<8} {script_success}/{len(script_summaries):<6} {script_latency:>10.1f}ms {script_calls:>8}")

    if agent_summaries:
        agent_latency = sum(s.total_latency_ms for s in agent_summaries)
        agent_calls = sum(s.tool_calls for s in agent_summaries)
        agent_success = sum(1 for s in agent_summaries if s.success)
        print(f"{'AGENT TOTAL':<25} {'':<8} {agent_success}/{len(agent_summaries):<6} {agent_latency:>10.1f}ms {agent_calls:>8}")

    if subagent_summaries:
        subagent_latency = sum(s.total_latency_ms for s in subagent_summaries)
        subagent_calls = sum(s.tool_calls for s in subagent_summaries)
        subagent_success = sum(1 for s in subagent_summaries if s.success)
        print(f"{'SUBAGENT TOTAL':<25} {'':<8} {subagent_success}/{len(subagent_summaries):<6} {subagent_latency:>10.1f}ms {subagent_calls:>8}")

    if subagent_legacy_summaries:
        legacy_latency = sum(s.total_latency_ms for s in subagent_legacy_summaries)
        legacy_calls = sum(s.tool_calls for s in subagent_legacy_summaries)
        legacy_success = sum(1 for s in subagent_legacy_summaries if s.success)
        print(f"{'SUBAGENT_LEGACY TOTAL':<25} {'':<8} {legacy_success}/{len(subagent_legacy_summaries):<6} {legacy_latency:>10.1f}ms {legacy_calls:>8}")

    # SubAgent speedup comparison
    if subagent_summaries and subagent_legacy_summaries:
        subagent_total = sum(s.total_latency_ms for s in subagent_summaries)
        legacy_total = sum(s.total_latency_ms for s in subagent_legacy_summaries)
        if subagent_total > 0:
            speedup = legacy_total / subagent_total
            print(f"\nSubAgent Speedup: {speedup:.2f}x")

    print()

    # Location distribution
    print("Location Distribution:")
    all_locations = {}
    for s in summaries:
        for loc, count in s.locations.items():
            all_locations[loc] = all_locations.get(loc, 0) + count

    for loc, count in sorted(all_locations.items()):
        print(f"  {loc}: {count} calls")
    print()


def save_results(summaries: list[ScenarioSummary], output_dir: Path):
    """Save results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": [
            {
                "name": s.scenario_name,
                "version": s.version,
                "description": s.description,
                "success": s.success,
                "total_latency_ms": s.total_latency_ms,
                "tool_calls": s.tool_calls,
                "input_bytes": s.input_bytes,
                "output_bytes": s.output_bytes,
                "reduction_ratio": s.reduction_ratio,
                "locations": s.locations,
                "data_source": s.data_source,
                "error": s.error,
            }
            for s in summaries
        ],
        "summary": {
            "total_scenarios": len(summaries),
            "successful": sum(1 for s in summaries if s.success),
            "script_runs": len([s for s in summaries if s.version == "script"]),
            "agent_runs": len([s for s in summaries if s.version == "agent"]),
            "subagent_runs": len([s for s in summaries if s.version == "subagent"]),
            "subagent_legacy_runs": len([s for s in summaries if s.version == "subagent_legacy"]),
            "total_latency_ms": sum(s.total_latency_ms for s in summaries),
            "total_tool_calls": sum(s.tool_calls for s in summaries),
        }
    }

    output_path = output_dir / "all_scenarios_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Run all EdgeAgent scenarios")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of scenarios to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--include-agent",
        action="store_true",
        help="Include LLM agent versions (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run only LLM agent versions (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--include-subagent",
        action="store_true",
        help="Include SubAgent versions (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--subagent-only",
        action="store_true",
        help="Run only SubAgent versions (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--subagent-compare",
        action="store_true",
        help="Run SubAgent and SubAgent-Legacy for comparison (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run all modes: script, agent, subagent, subagent_legacy",
    )
    parser.add_argument(
        "--skip-failing",
        action="store_true",
        help="Continue even if a scenario fails",
    )
    args = parser.parse_args()

    scenarios_to_run = [int(s.strip()) for s in args.scenarios.split(",")]

    # Determine which modes to run
    run_script = not (args.agent_only or args.subagent_only or args.subagent_compare)
    run_agent = args.include_agent or args.agent_only or args.all_modes
    run_subagent = args.include_subagent or args.subagent_only or args.subagent_compare or args.all_modes
    run_subagent_legacy = args.subagent_compare or args.all_modes

    print("=" * 120)
    print("EdgeAgent: Run All Scenarios")
    print("=" * 120)
    print()
    print(f"Scenarios to run: {scenarios_to_run}")
    print(f"Modes: script={run_script}, agent={run_agent}, subagent={run_subagent}, subagent_legacy={run_subagent_legacy}")
    print()

    # Define scenario runners
    script_runners = {
        1: ("S1: Code Review (Script)", run_scenario1_script),
        2: ("S2: Log Analysis (Script)", run_scenario2_script),
        3: ("S3: Research Assistant (Script)", run_scenario3_script),
        4: ("S4: Image Processing (Script)", run_scenario4_script),
    }

    agent_runners = {
        1: ("S1: Code Review (Agent)", run_scenario1_agent),
        2: ("S2: Log Analysis (Agent)", run_scenario2_agent),
        3: ("S3: Research Assistant (Agent)", run_scenario3_agent),
        4: ("S4: Image Processing (Agent)", run_scenario4_agent),
    }

    subagent_runners = {
        1: ("S1: Code Review (SubAgent)", run_scenario1_subagent),
        2: ("S2: Log Analysis (SubAgent)", run_scenario2_subagent),
        3: ("S3: Research Assistant (SubAgent)", run_scenario3_subagent),
        4: ("S4: Image Processing (SubAgent)", run_scenario4_subagent),
    }

    subagent_legacy_runners = {
        1: ("S1: Code Review (SubAgent-Legacy)", run_scenario1_subagent_legacy),
        2: ("S2: Log Analysis (SubAgent-Legacy)", run_scenario2_subagent_legacy),
        3: ("S3: Research Assistant (SubAgent-Legacy)", run_scenario3_subagent_legacy),
        4: ("S4: Image Processing (SubAgent-Legacy)", run_scenario4_subagent_legacy),
    }

    summaries = []

    # Run script versions
    if run_script:
        for num in scenarios_to_run:
            if num not in script_runners:
                print(f"[WARN] Unknown scenario: {num}")
                continue

            name, runner = script_runners[num]
            print("=" * 120)
            print(f"Running {name}")
            print("=" * 120)

            try:
                summary = await runner()
                summaries.append(summary)

                if summary.success:
                    print(f"  [PASS] {summary.total_latency_ms:.1f}ms, {summary.tool_calls} calls")
                else:
                    print(f"  [FAIL] {summary.error}")
                    if not args.skip_failing:
                        print("  Stopping (use --skip-failing to continue)")
                        break

            except Exception as e:
                print(f"  [ERROR] {e}")
                summaries.append(ScenarioSummary(
                    scenario_name=name.split(" (")[0],
                    version="script",
                    description="",
                    success=False,
                    total_latency_ms=0,
                    tool_calls=0,
                    input_bytes=0,
                    output_bytes=0,
                    reduction_ratio=0,
                    locations={},
                    error=str(e),
                ))
                if not args.skip_failing:
                    break

    # Run agent versions (if requested)
    if run_agent:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[ERROR] OPENAI_API_KEY not set. Agent versions require OpenAI API key.")
            print("Set OPENAI_API_KEY environment variable or add to .env file.")
        else:
            for num in scenarios_to_run:
                if num not in agent_runners:
                    continue

                name, runner = agent_runners[num]
                print("=" * 120)
                print(f"Running {name}")
                print("=" * 120)

                try:
                    summary = await runner()
                    summaries.append(summary)

                    if summary.success:
                        print(f"  [PASS] {summary.total_latency_ms:.1f}ms, {summary.tool_calls} calls")
                    else:
                        print(f"  [FAIL] {summary.error}")
                        if not args.skip_failing:
                            break

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    summaries.append(ScenarioSummary(
                        scenario_name=name.split(" (")[0],
                        version="agent",
                        description="",
                        success=False,
                        total_latency_ms=0,
                        tool_calls=0,
                        input_bytes=0,
                        output_bytes=0,
                        reduction_ratio=0,
                        locations={},
                        error=str(e),
                    ))
                    if not args.skip_failing:
                        break

    # Run subagent_legacy versions (if requested - run before subagent for fair comparison)
    if run_subagent_legacy:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[ERROR] OPENAI_API_KEY not set. SubAgent versions require OpenAI API key.")
            print("Set OPENAI_API_KEY environment variable or add to .env file.")
        else:
            for num in scenarios_to_run:
                if num not in subagent_legacy_runners:
                    continue

                name, runner = subagent_legacy_runners[num]
                print("=" * 120)
                print(f"Running {name}")
                print("=" * 120)

                try:
                    summary = await runner()
                    summaries.append(summary)

                    if summary.success:
                        print(f"  [PASS] {summary.total_latency_ms:.1f}ms, {summary.tool_calls} calls")
                    else:
                        print(f"  [FAIL] {summary.error}")
                        if not args.skip_failing:
                            break

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    summaries.append(ScenarioSummary(
                        scenario_name=name.split(" (")[0],
                        version="subagent_legacy",
                        description="",
                        success=False,
                        total_latency_ms=0,
                        tool_calls=0,
                        input_bytes=0,
                        output_bytes=0,
                        reduction_ratio=0,
                        locations={},
                        error=str(e),
                    ))
                    if not args.skip_failing:
                        break

    # Run subagent versions (if requested)
    if run_subagent:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[ERROR] OPENAI_API_KEY not set. SubAgent versions require OpenAI API key.")
            print("Set OPENAI_API_KEY environment variable or add to .env file.")
        else:
            for num in scenarios_to_run:
                if num not in subagent_runners:
                    continue

                name, runner = subagent_runners[num]
                print("=" * 120)
                print(f"Running {name}")
                print("=" * 120)

                try:
                    summary = await runner()
                    summaries.append(summary)

                    if summary.success:
                        print(f"  [PASS] {summary.total_latency_ms:.1f}ms, {summary.tool_calls} calls, {summary.locations.get('partitions', 0)} partitions")
                    else:
                        print(f"  [FAIL] {summary.error}")
                        if not args.skip_failing:
                            break

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    summaries.append(ScenarioSummary(
                        scenario_name=name.split(" (")[0],
                        version="subagent",
                        description="",
                        success=False,
                        total_latency_ms=0,
                        tool_calls=0,
                        input_bytes=0,
                        output_bytes=0,
                        reduction_ratio=0,
                        locations={},
                        error=str(e),
                    ))
                    if not args.skip_failing:
                        break

    # Print comparison
    if summaries:
        print_comparison_table(summaries)

        # Save results
        output_dir = Path("results/comparison")
        save_results(summaries, output_dir)

    # Exit code
    all_success = all(s.success for s in summaries) if summaries else False
    return all_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
