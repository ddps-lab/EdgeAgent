#!/usr/bin/env python3
"""
Scenario 1: Code Review Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the code review task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Comparison with run_scenario1_with_metrics.py:
- Script version: Hardcoded sequential tool calls (orchestration)
- Agent version: LLM autonomously selects tools (autonomous agent)

Tool Chain (expected, but LLM decides):
    filesystem -> git -> summarize -> data_aggregate -> filesystem(write)
    DEVICE       DEVICE  EDGE        EDGE             DEVICE
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from edgeagent import ScenarioRunner, EdgeAgentMCPClient


def load_repo_source() -> tuple[Path, str]:
    """Load Git repository source from Defects4J or sample repo.

    Returns:
        Tuple of (repo_path, data_source_description)
    """
    data_dir = Path(__file__).parent.parent / "data" / "scenario1"
    defects4j_dir = data_dir / "defects4j"
    sample_repo = data_dir / "sample_repo"

    if defects4j_dir.exists():
        for subdir in defects4j_dir.iterdir():
            if subdir.is_dir() and (subdir / ".git").exists():
                return subdir, f"Defects4J ({subdir.name})"

    if sample_repo.exists() and (sample_repo / ".git").exists():
        return sample_repo, "Generated sample repository"

    raise FileNotFoundError(
        f"No Git repository found in {data_dir}\n"
        "Run 'python scripts/download_public_datasets.py -s 1' for Defects4J, or\n"
        "Run 'python scripts/generate_test_repo.py' for sample repository"
    )


# System prompt for the Code Review Agent
CODE_REVIEW_SYSTEM_PROMPT = """You are a code review assistant. Your task is to analyze a Git repository and generate a comprehensive code review report.

You have access to the following tools:
- list_directory: List files in a directory
- git_status: Get Git repository status
- git_log: Get commit history
- git_diff: Get code differences
- summarize_text: Summarize text content
- aggregate_list: Group and aggregate data
- write_file: Write files to the filesystem

The repository is located at /tmp/edgeagent_device/repo

When conducting code review, follow this workflow:
1. List the repository files to understand the structure
2. Get git status to see current state
3. Get git log to see recent commits (use max_count=10)
4. Get git diff to see code changes (use target="HEAD~3")
5. Summarize the changes
6. Write a comprehensive code review report to /tmp/edgeagent_device/agent_code_review_report.md

Include in your report:
- Repository overview
- Recent commits summary
- Code changes analysis
- Recommendations for improvement
"""


class AgentCodeReviewScenario(ScenarioRunner):
    """
    LLM Agent-based Code Review Scenario.

    Unlike the script-based version, this uses an LLM to autonomously
    decide which tools to call and in what order.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results/scenario1_agent",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
    ):
        super().__init__(config_path, output_dir)
        self.model = model
        self.temperature = temperature
        self._repo_source = None
        self._data_source = None

    @property
    def name(self) -> str:
        return "code_review_agent"

    @property
    def description(self) -> str:
        return "LLM Agent autonomously reviews code and generates a report"

    @property
    def user_request(self) -> str:
        return (
            "Review the Git repository at /tmp/edgeagent_device/repo. "
            "Analyze the commit history, code changes, and generate a comprehensive "
            "code review report to /tmp/edgeagent_device/agent_code_review_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute code review using LLM Agent"""

        # Prepare repository
        repo_source, data_source = load_repo_source()
        self._repo_source = repo_source
        self._data_source = data_source

        device_repo = Path("/tmp/edgeagent_device/repo")
        if device_repo.exists():
            shutil.rmtree(device_repo)
        shutil.copytree(repo_source, device_repo)

        # Ensure output directory exists
        Path("/tmp/edgeagent_device").mkdir(parents=True, exist_ok=True)

        print("-" * 70)
        print("LLM Agent Code Review")
        print("-" * 70)
        print(f"Data Source: {data_source}")
        print(f"Repository: {device_repo}")
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
            client.metrics_collector.add_custom_metric("data_source", data_source)
            client.metrics_collector.add_custom_metric("tool_call_counts", tool_counts)
            client.metrics_collector.add_custom_metric("total_tool_calls", len(client.metrics_collector.entries))

        # Check if report was written
        report_path = Path("/tmp/edgeagent_device/agent_code_review_report.md")
        if report_path.exists():
            report_content = report_path.read_text()
            print(f"Report written to: {report_path}")
            print(f"Report size: {len(report_content)} bytes")
            return report_content
        else:
            print("[WARN] Agent did not write a report file")
            return final_message


async def main():
    """Run the LLM Agent-based Code Review scenario"""

    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        print("Please create a .env file with OPENAI_API_KEY=your-key")
        return False

    print("=" * 70)
    print("Scenario 1: Code Review Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"

    scenario = AgentCodeReviewScenario(
        config_path=config_path,
        output_dir="results/scenario1_agent",
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
        csv_path = result.metrics.save_csv("results/scenario1_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
