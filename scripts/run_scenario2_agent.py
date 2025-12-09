#!/usr/bin/env python3
"""
Scenario 2: Log Analysis Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the log analysis task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Comparison with run_scenario2_with_metrics.py:
- Script version: Hardcoded sequential tool calls (orchestration)
- Agent version: LLM autonomously selects tools (autonomous agent)

Tool Chain (expected, but LLM decides):
    filesystem(read) -> log_parser -> data_aggregate -> filesystem(write)
    DEVICE            EDGE         EDGE             DEVICE
"""

import asyncio
import os
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from edgeagent import ScenarioRunner, EdgeAgentMCPClient


def load_log_source() -> tuple[Path, str]:
    """Load log file source from LogHub or sample log.

    Returns:
        Tuple of (log_file_path, data_source_description)
    """
    data_dir = Path(__file__).parent.parent / "data" / "scenario2"
    loghub_dir = data_dir / "loghub_samples"
    sample_log = data_dir / "server.log"

    if loghub_dir.exists():
        # Priority: medium_python > small_python > any .log file
        for log_name in ["medium_python.log", "small_python.log"]:
            candidate = loghub_dir / log_name
            if candidate.exists():
                return candidate, f"LogHub ({log_name})"

        log_files = list(loghub_dir.glob("*.log"))
        if log_files:
            return log_files[0], f"LogHub ({log_files[0].name})"

    if sample_log.exists():
        return sample_log, "Sample server.log"

    raise FileNotFoundError(
        f"No log file found in {data_dir}\n"
        "Run 'python scripts/download_public_datasets.py -s 2' for LogHub data"
    )


# Load log source at module level
LOG_SOURCE, DATA_SOURCE = load_log_source()


# System prompt for the Log Analysis Agent
LOG_ANALYSIS_SYSTEM_PROMPT = f"""You are a log analysis assistant. Your task is to analyze server logs and generate a comprehensive error report.

You have access to the following tools:
- read_file: Read file contents
- parse_logs: Parse log file content into structured entries
- filter_entries: Filter log entries by severity level
- compute_log_statistics: Compute statistics on log entries
- aggregate_list: Group and aggregate data
- write_file: Write files to the filesystem

The log file is located at /tmp/edgeagent_device/server.log

When conducting log analysis, follow this workflow:
1. Read the log file using read_file
2. Parse the logs using parse_logs (format_type="python")
3. Filter for warnings and errors using filter_entries (min_level="warning")
4. Compute statistics using compute_log_statistics
5. Aggregate by severity level using aggregate_list (group_by="_level")
6. Write a comprehensive log analysis report to /tmp/edgeagent_device/agent_log_report.md

Include in your report:
- Total entries analyzed
- Breakdown by severity level
- Critical issues that need attention
- Recommendations for addressing errors
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
    ):
        super().__init__(config_path, output_dir)
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
            "Analyze the server log file at /tmp/edgeagent_device/server.log. "
            "Parse the logs, filter for warnings and errors, compute statistics, "
            "and generate a comprehensive log analysis report to "
            "/tmp/edgeagent_device/agent_log_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute log analysis using LLM Agent"""

        # Prepare log file
        device_log = Path("/tmp/edgeagent_device/server.log")
        device_log.parent.mkdir(parents=True, exist_ok=True)
        device_log.write_text(LOG_SOURCE.read_text())

        print("-" * 70)
        print("LLM Agent Log Analysis")
        print("-" * 70)
        print(f"Data Source: {DATA_SOURCE}")
        print(f"Log file size: {LOG_SOURCE.stat().st_size:,} bytes")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Available tools: {len(tools)}")
        print()

        # Initialize LLM
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
        )

        # Create agent
        agent = create_agent(llm, tools)

        print("Agent created. Sending user request...")
        print()
        print(f"User Request: {self.user_request[:200]}...")
        print()
        print("-" * 70)
        print("Agent Execution (tool calls will be shown)")
        print("-" * 70)

        # Execute agent
        result = await agent.ainvoke({
            "messages": [("user", self.user_request)]
        })

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
        report_path = Path("/tmp/edgeagent_device/agent_log_report.md")
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

    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        print("Please create a .env file with OPENAI_API_KEY=your-key")
        return False

    print("=" * 70)
    print("Scenario 2: Log Analysis Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"

    scenario = AgentLogAnalysisScenario(
        config_path=config_path,
        output_dir="results/scenario2_agent",
        model="gpt-4o-mini",
        temperature=0,
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional analysis
    if result.metrics:
        print()
        print("=" * 70)
        print("Agent Execution Trace")
        print("=" * 70)
        for i, trace in enumerate(result.metrics.to_execution_trace(), 1):
            print(f"  {i}. {trace['tool']} -> {trace['location']}")

        # Export metrics
        csv_path = result.metrics.save_csv("results/scenario2_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
