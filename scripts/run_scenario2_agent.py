#!/usr/bin/env python3
"""
Scenario 02: Log Analysis Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the log analysis task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Tool Chain (expected, but LLM decides):
    filesystem(read) -> log_parser -> data_aggregate -> filesystem(write)
    DEVICE            EDGE         EDGE             DEVICE
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent

from edgeagent import ScenarioRunner, EdgeAgentMCPClient
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import create_scheduler, BruteForceChainScheduler
from scripts.agent_utils import run_agent_with_logging, create_llm_with_latency_tracking


def load_log_source() -> tuple[Path, str]:
    """Load log file source from LogHub.

    Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).

    Returns:
        Tuple of (log_file_path, data_source_description)
    """
    # Use unified path that works across all locations
    data_dir = Path("/edgeagent/data/scenario2")
    loghub_dir = data_dir / "loghub_samples"

    if loghub_dir.exists():
        # Priority: apache_small for agent
        for log_name in ["apache_small.log", "apache_medium.log"]:
            candidate = loghub_dir / log_name
            if candidate.exists():
                return candidate, f"LogHub ({log_name})"

        # Sort for deterministic fallback
        log_files = sorted(loghub_dir.glob("*.log"))
        if log_files:
            return log_files[0], f"LogHub ({log_files[0].name})"

    raise FileNotFoundError(
        f"No log file found in {loghub_dir}\n"
        "Run 'python scripts/setup_test_data.py -s 2' for test data"
    )


# Load log source at module level
LOG_SOURCE, DATA_SOURCE = load_log_source()


# System prompt for the Log Analysis Agent (dynamically includes log path)
LOG_ANALYSIS_SYSTEM_PROMPT = f"""You are a log analysis assistant. Your task is to analyze server logs and generate a comprehensive error report.

You have access to the following tools:
- read_text_file: Read file contents
- parse_logs: Parse log file content into structured entries (returns "entries" array)
- filter_entries: Filter log entries by severity level (requires "entries" array)
- compute_log_statistics: Compute statistics on log entries (requires "entries" array)
- write_file: Write files to the filesystem

The log file is located at {LOG_SOURCE}

IMPORTANT - Data Flow:
- parse_logs returns {{"entries": [...]}} - you MUST pass this "entries" array to other tools
- filter_entries(entries=<parsed_result>["entries"], min_level="warning")
- compute_log_statistics(entries=<parsed_result>["entries"])

CRITICAL: Always use max_lines parameter to ensure consistent performance!

Example workflow:
1. log_content = read_text_file("{LOG_SOURCE}", max_lines=200)  # Read EXACTLY 200 lines
2. parsed = parse_logs(log_content=log_content, format_type="auto")
3. stats = compute_log_statistics(entries=parsed["entries"])
4. filtered = filter_entries(entries=parsed["entries"], min_level="warning")
5. write_file(path="/edgeagent/results/scenario2_agent_log_report.md", content=report_markdown)

Include in your report:
- Total entries analyzed
- Breakdown by severity level
- Error details and recommendations
"""


class AgentLogAnalysisScenario(ScenarioRunner):
    """
    LLM Agent-based Log Analysis Scenario.

    Unlike the script-based version, this uses an LLM to autonomously
    decide which tools to call and in what order.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results/scenario2_agent",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        scheduler=None,
    ):
        super().__init__(config_path, output_dir, scheduler=scheduler)
        self.model = model
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "log_analysis_agent"

    @property
    def description(self) -> str:
        return "LLM Agent autonomously analyzes logs and generates a report"

    @property
    def user_request(self) -> str:
        return (
            f"Analyze the server log file at {LOG_SOURCE}. "
            "Follow these steps: "
            "1) Read EXACTLY the first 200 lines with read_text_file(max_lines=200), "
            "2) Parse using parse_logs with format_type='auto' to get entries array, "
            "3) Compute statistics using compute_log_statistics with the entries array, "
            "4) Write a comprehensive analysis report to /edgeagent/results/scenario2_agent_log_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute log analysis using LLM Agent"""

        # Filter out problematic tools (directory_tree causes issues with gpt-4o-mini)
        excluded_tools = {"directory_tree"}
        tools = [t for t in tools if t.name not in excluded_tools]

        print("-" * 70)
        print("LLM Agent Log Analysis")
        print("-" * 70)
        print(f"Data Source: {DATA_SOURCE}")
        print(f"Log file size: {LOG_SOURCE.stat().st_size:,} bytes")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Available tools: {len(tools)}")
        print()

        # Initialize LLM with latency tracking
        llm = create_llm_with_latency_tracking(
            model=self.model,
            temperature=self.temperature,
            metrics_collector=client.metrics_collector,
        )

        # Create agent with system prompt
        agent = create_agent(llm, tools, system_prompt=LOG_ANALYSIS_SYSTEM_PROMPT)

        print("Agent created. Sending user request...")
        print()
        print(f"User Request: {self.user_request[:200]}...")
        print()
        print("-" * 70)
        print("Agent Execution (tool calls will be shown)")
        print("-" * 70)

        # Execute agent with logging (metrics_collector로 LLM latency 추적)
        result = await run_agent_with_logging(
            agent,
            self.user_request,
            verbose=True,
            metrics_collector=client.metrics_collector,
        )

        # Extract final response
        final_message = result["messages"][-1].content

        print()
        print("-" * 70)
        print("Agent Final Response")
        print("-" * 70)
        print(final_message[:1000] + "..." if len(final_message) > 1000 else final_message)
        print()

        # Add custom metrics
        if client.metrics_collector:
            tool_counts = {}
            for entry in client.metrics_collector.entries:
                tool = entry.tool_name
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            client.metrics_collector.add_custom_metric("agent_model", self.model)
            client.metrics_collector.add_custom_metric("data_source", DATA_SOURCE)
            client.metrics_collector.add_custom_metric("tool_call_counts", tool_counts)
            client.metrics_collector.add_custom_metric("total_tool_calls", len(client.metrics_collector.entries))
            client.metrics_collector.add_custom_metric("log_file_size", LOG_SOURCE.stat().st_size)

        # Check if report was written
        report_path = Path("/edgeagent/results/scenario2_agent_log_report.md")
        if report_path.exists():
            report_content = report_path.read_text()
            print(f"Report written to: {report_path}")
            print(f"Report size: {len(report_content)} bytes")
            return report_content
        else:
            print("[WARN] Agent did not write a report file")
            return final_message


async def main():
    """Run the LLM Agent-based Log Analysis scenario"""

    parser = argparse.ArgumentParser(description="Run Scenario 02 with LLM Agent")
    parser.add_argument(
        "--scheduler",
        choices=["brute_force", "static", "all_device", "all_edge", "all_cloud", "heuristic"],
        default="brute_force",
        help="Scheduler type (default: brute_force)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        print("Please create a .env file with OPENAI_API_KEY=your-key")
        return False

    print("=" * 70)
    print("Scenario 02: Log Analysis Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print(f"Scheduler: {args.scheduler}")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    # Create scheduler
    registry = ToolRegistry.from_yaml(config_path)
    if args.scheduler == "brute_force":
        scheduler = BruteForceChainScheduler(config_path, system_config_path, registry)
    else:
        scheduler = create_scheduler(args.scheduler, config_path, registry)

    scenario = AgentLogAnalysisScenario(
        config_path=config_path,
        output_dir="results/scenario2_agent",
        model="gpt-4o-mini",
        temperature=0,
        scheduler=scheduler,
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional analysis
    if result.execution_trace:
        print()
        print("=" * 70)
        print("Agent Execution Trace (Scheduler Results)")
        print("=" * 70)
        for i, trace in enumerate(result.execution_trace, 1):
            fixed_mark = "[FIXED]" if trace.get('fixed') else ""
            if args.scheduler == "brute_force":
                cost = trace.get('cost', 0)
                comp = trace.get('comp', 0)
                comm = trace.get('comm', 0)
                print(f"  {i}. {trace['tool']:25} -> {trace['location']:6} (cost={cost:.3f}, comp={comp:.3f}, comm={comm:.3f}) {fixed_mark}")
            else:
                print(f"  {i}. {trace['tool']:25} -> {trace['location']:6} {fixed_mark}")

    if result.metrics:
        # Export metrics
        csv_path = result.metrics.save_csv("results/scenario2_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
