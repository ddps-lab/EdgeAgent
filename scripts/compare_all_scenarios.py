#!/usr/bin/env python3
"""
Compare Agent vs Script Execution for All Scenarios

This script runs both Script-based and Agent-based versions of each scenario
and compares their metrics. This is useful for:

1. Paper: Demonstrating both "Agent autonomy" and "Middleware efficiency"
2. Understanding trade-offs between deterministic and autonomous execution
3. Analyzing tool selection patterns by LLM agents

Comparison Metrics:
- Total latency
- Number of tool calls
- Tool call sequence
- Data reduction ratio
- Location distribution
- Success rate

Usage:
    python scripts/compare_all_scenarios.py
    python scripts/compare_all_scenarios.py --scenarios 2,4
    python scripts/compare_all_scenarios.py --skip-agent  # Only run script versions
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


@dataclass
class ExecutionResult:
    """Result of a single execution"""
    version: str  # "script" or "agent"
    scenario_name: str
    success: bool
    total_latency_ms: float
    tool_calls: int
    tool_sequence: list[str]
    input_bytes: int
    output_bytes: int
    reduction_ratio: float
    locations: dict[str, int]
    error: Optional[str] = None
    custom_metrics: dict = field(default_factory=dict)


@dataclass
class ScenarioComparison:
    """Comparison between script and agent versions"""
    scenario_name: str
    script_result: Optional[ExecutionResult] = None
    agent_result: Optional[ExecutionResult] = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "script": self._result_to_dict(self.script_result) if self.script_result else None,
            "agent": self._result_to_dict(self.agent_result) if self.agent_result else None,
            "comparison": self._compute_comparison(),
        }

    def _result_to_dict(self, result: ExecutionResult) -> dict:
        return {
            "success": result.success,
            "total_latency_ms": result.total_latency_ms,
            "tool_calls": result.tool_calls,
            "tool_sequence": result.tool_sequence,
            "input_bytes": result.input_bytes,
            "output_bytes": result.output_bytes,
            "reduction_ratio": result.reduction_ratio,
            "locations": result.locations,
            "error": result.error,
            "custom_metrics": result.custom_metrics,
        }

    def _compute_comparison(self) -> dict:
        if not self.script_result or not self.agent_result:
            return {}

        s = self.script_result
        a = self.agent_result

        return {
            "latency_diff_ms": a.total_latency_ms - s.total_latency_ms,
            "latency_ratio": a.total_latency_ms / s.total_latency_ms if s.total_latency_ms > 0 else 0,
            "tool_call_diff": a.tool_calls - s.tool_calls,
            "sequence_match": s.tool_sequence == a.tool_sequence,
            "both_success": s.success and a.success,
        }


async def run_script_scenario(scenario_num: int) -> Optional[ExecutionResult]:
    """Run the script-based version of a scenario"""
    try:
        if scenario_num == 1:
            from run_scenario1_with_metrics import CodeReviewScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
            scenario = CodeReviewScenario(config_path=config_path, output_dir="results/comparison/s1_script")

        elif scenario_num == 2:
            from run_scenario2_with_metrics import LogAnalysisScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
            scenario = LogAnalysisScenario(config_path=config_path, output_dir="results/comparison/s2_script")

        elif scenario_num == 3:
            from run_scenario3_with_metrics import ResearchAssistantScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
            scenario = ResearchAssistantScenario(config_path=config_path, output_dir="results/comparison/s3_script", max_urls=2)

        elif scenario_num == 4:
            from run_scenario4_with_metrics import ImageProcessingScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"
            scenario = ImageProcessingScenario(config_path=config_path, output_dir="results/comparison/s4_script")

        else:
            print(f"  [ERROR] Unknown scenario: {scenario_num}")
            return None

        result = await scenario.run(save_results=True, print_summary=False)

        return ExecutionResult(
            version="script",
            scenario_name=scenario.name,
            success=result.success,
            total_latency_ms=result.total_latency_ms,
            tool_calls=result.tool_call_count,
            tool_sequence=[e.tool_name for e in result.metrics.entries] if result.metrics else [],
            input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
            output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
            reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
            locations=result.metrics.call_count_by_location() if result.metrics else {},
            error=result.error,
            custom_metrics=result.metrics.custom_metrics if result.metrics else {},
        )

    except Exception as e:
        print(f"  [ERROR] Script version failed: {e}")
        return ExecutionResult(
            version="script",
            scenario_name=f"scenario_{scenario_num}",
            success=False,
            total_latency_ms=0,
            tool_calls=0,
            tool_sequence=[],
            input_bytes=0,
            output_bytes=0,
            reduction_ratio=0,
            locations={},
            error=str(e),
        )


async def run_agent_scenario(scenario_num: int) -> Optional[ExecutionResult]:
    """Run the agent-based version of a scenario"""
    try:
        if scenario_num == 1:
            from run_scenario1_agent import AgentCodeReviewScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
            scenario = AgentCodeReviewScenario(config_path=config_path, output_dir="results/comparison/s1_agent")

        elif scenario_num == 2:
            from run_scenario2_agent import AgentLogAnalysisScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
            scenario = AgentLogAnalysisScenario(config_path=config_path, output_dir="results/comparison/s2_agent")

        elif scenario_num == 3:
            from run_scenario3_agent import AgentResearchAssistantScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
            scenario = AgentResearchAssistantScenario(config_path=config_path, output_dir="results/comparison/s3_agent")

        elif scenario_num == 4:
            from run_scenario4_agent import AgentImageProcessingScenario
            config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"
            scenario = AgentImageProcessingScenario(config_path=config_path, output_dir="results/comparison/s4_agent")

        else:
            print(f"  [ERROR] Unknown scenario: {scenario_num}")
            return None

        result = await scenario.run(save_results=True, print_summary=False)

        return ExecutionResult(
            version="agent",
            scenario_name=scenario.name,
            success=result.success,
            total_latency_ms=result.total_latency_ms,
            tool_calls=result.tool_call_count,
            tool_sequence=[e.tool_name for e in result.metrics.entries] if result.metrics else [],
            input_bytes=result.metrics.total_input_bytes if result.metrics else 0,
            output_bytes=result.metrics.total_output_bytes if result.metrics else 0,
            reduction_ratio=result.metrics.overall_reduction_ratio if result.metrics else 0,
            locations=result.metrics.call_count_by_location() if result.metrics else {},
            error=result.error,
            custom_metrics=result.metrics.custom_metrics if result.metrics else {},
        )

    except Exception as e:
        print(f"  [ERROR] Agent version failed: {e}")
        return ExecutionResult(
            version="agent",
            scenario_name=f"scenario_{scenario_num}_agent",
            success=False,
            total_latency_ms=0,
            tool_calls=0,
            tool_sequence=[],
            input_bytes=0,
            output_bytes=0,
            reduction_ratio=0,
            locations={},
            error=str(e),
        )


def print_comparison_report(comparisons: list[ScenarioComparison]):
    """Print comprehensive comparison report"""
    print()
    print("=" * 100)
    print("Agent vs Script Comparison Report")
    print("=" * 100)
    print()

    # Summary table
    print("Summary Table:")
    print("-" * 100)
    print(f"{'Scenario':<20} {'Version':<10} {'Status':<8} {'Latency':>12} {'Calls':>8} {'Reduction':>12}")
    print("-" * 100)

    for comp in comparisons:
        if comp.script_result:
            r = comp.script_result
            status = "PASS" if r.success else "FAIL"
            print(f"{comp.scenario_name:<20} {'Script':<10} {status:<8} {r.total_latency_ms:>10.1f}ms {r.tool_calls:>8} {r.reduction_ratio:>12.4f}")

        if comp.agent_result:
            r = comp.agent_result
            status = "PASS" if r.success else "FAIL"
            print(f"{'':<20} {'Agent':<10} {status:<8} {r.total_latency_ms:>10.1f}ms {r.tool_calls:>8} {r.reduction_ratio:>12.4f}")

        print("-" * 100)

    print()

    # Detailed comparison
    print("Detailed Comparison:")
    print("-" * 100)

    for comp in comparisons:
        if comp.script_result and comp.agent_result:
            s = comp.script_result
            a = comp.agent_result

            latency_diff = a.total_latency_ms - s.total_latency_ms
            latency_ratio = a.total_latency_ms / s.total_latency_ms if s.total_latency_ms > 0 else 0
            call_diff = a.tool_calls - s.tool_calls

            print(f"\n{comp.scenario_name}:")
            print(f"  Latency: Script={s.total_latency_ms:.1f}ms, Agent={a.total_latency_ms:.1f}ms")
            print(f"           Diff={latency_diff:+.1f}ms ({latency_ratio:.2f}x)")
            print(f"  Tool Calls: Script={s.tool_calls}, Agent={a.tool_calls}, Diff={call_diff:+d}")
            print(f"  Sequence Match: {'Yes' if s.tool_sequence == a.tool_sequence else 'No (Agent chose different path)'}")

            # Tool sequence comparison
            print(f"  Script Sequence: {' -> '.join(s.tool_sequence[:5])}{'...' if len(s.tool_sequence) > 5 else ''}")
            print(f"  Agent Sequence:  {' -> '.join(a.tool_sequence[:5])}{'...' if len(a.tool_sequence) > 5 else ''}")

            # Location distribution
            print(f"  Script Locations: {s.locations}")
            print(f"  Agent Locations:  {a.locations}")

    print()
    print("=" * 100)


def save_comparison_results(comparisons: list[ScenarioComparison], output_dir: Path):
    """Save comparison results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "comparisons": [comp.to_dict() for comp in comparisons],
        "summary": {
            "total_scenarios": len(comparisons),
            "script_successes": sum(1 for c in comparisons if c.script_result and c.script_result.success),
            "agent_successes": sum(1 for c in comparisons if c.agent_result and c.agent_result.success),
        }
    }

    output_path = output_dir / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Comparison results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Compare Agent vs Script for all scenarios")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of scenarios to compare (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip agent versions (only run script versions)",
    )
    parser.add_argument(
        "--skip-script",
        action="store_true",
        help="Skip script versions (only run agent versions)",
    )
    args = parser.parse_args()

    load_dotenv()

    # Check for API key if running agent versions
    if not args.skip_agent and not os.getenv("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY not set - agent versions will be skipped")
        args.skip_agent = True

    scenarios = [int(s.strip()) for s in args.scenarios.split(",")]

    print("=" * 100)
    print("EdgeAgent: Agent vs Script Comparison")
    print("=" * 100)
    print()
    print(f"Scenarios: {scenarios}")
    print(f"Run Script: {not args.skip_script}")
    print(f"Run Agent: {not args.skip_agent}")
    print()

    scenario_names = {
        1: "S1: Code Review",
        2: "S2: Log Analysis",
        3: "S3: Research Assistant",
        4: "S4: Image Processing",
    }

    comparisons = []

    for num in scenarios:
        name = scenario_names.get(num, f"Scenario {num}")
        print("=" * 100)
        print(f"Comparing: {name}")
        print("=" * 100)

        comparison = ScenarioComparison(scenario_name=name)

        # Run script version
        if not args.skip_script:
            print(f"\n  Running Script version...")
            comparison.script_result = await run_script_scenario(num)
            if comparison.script_result:
                status = "PASS" if comparison.script_result.success else "FAIL"
                print(f"  Script: [{status}] {comparison.script_result.total_latency_ms:.1f}ms, {comparison.script_result.tool_calls} calls")

        # Run agent version
        if not args.skip_agent:
            print(f"\n  Running Agent version...")
            comparison.agent_result = await run_agent_scenario(num)
            if comparison.agent_result:
                status = "PASS" if comparison.agent_result.success else "FAIL"
                print(f"  Agent: [{status}] {comparison.agent_result.total_latency_ms:.1f}ms, {comparison.agent_result.tool_calls} calls")

        comparisons.append(comparison)

    # Print report
    print_comparison_report(comparisons)

    # Save results
    output_dir = Path("results/comparison")
    save_comparison_results(comparisons, output_dir)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
