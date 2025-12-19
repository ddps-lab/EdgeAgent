#!/usr/bin/env python3
"""
Scenario 01: Code Review Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the code review task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Tool Chain (expected, but LLM decides):
    filesystem -> git -> summarize -> data_aggregate -> filesystem(write)
    DEVICE       DEVICE  EDGE        EDGE             DEVICE
"""

import argparse
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
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import create_scheduler, BruteForceChainScheduler
from scripts.agent_utils import run_agent_with_logging


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
- list_directory: List files in a directory (use this, NOT directory_tree)
- read_file: Read file contents
- git_status: Get Git repository status
- git_log: Get commit history
- git_diff: Get code differences
- summarize_text: Summarize text content
- aggregate_list: Group and aggregate data
- write_file: Write files to the filesystem

IMPORTANT: Do NOT use directory_tree tool. Use list_directory instead.

The repository is located at /edgeagent/data/scenario1/defects4j/lang

When conducting code review, follow this workflow:
1. List the repository files to understand the structure
2. Get git status to see current state
3. Get git log to see recent commits (use max_count=5)
4. Get git diff to see recent code changes
5. Summarize the changes
6. Write a comprehensive code review report to /edgeagent/results/scenario1_agent_code_review_report.md

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
        scheduler=None,
    ):
        super().__init__(config_path, output_dir, scheduler=scheduler)
        self.model = model
        self.temperature = temperature
        self._repo_source = None
        self._data_source = None

        # Pre-initialize repo directory for MCP git server
        # (git server validates repo path on startup)
        repo_source, data_source = load_repo_source()
        self._repo_source = repo_source
        self._data_source = data_source

        repo_path = Path("/edgeagent/data/scenario1/defects4j/lang")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.exists():
            shutil.rmtree(repo_path)
        shutil.copytree(repo_source, repo_path)

    @property
    def name(self) -> str:
        return "code_review_agent"

    @property
    def description(self) -> str:
        return "LLM Agent autonomously reviews code and generates a report"

    @property
    def user_request(self) -> str:
        return (
            "Review the Git repository at /edgeagent/data/scenario1/defects4j/lang."
            "Analyze the commit history, code changes, and generate a comprehensive "
            "code review report to /edgeagent/results/scenario1_agent_code_review_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute code review using LLM Agent"""

        repo_path = Path("/edgeagent/data/scenario1/defects4j/lang")

        # Filter out problematic tools (directory_tree causes issues with gpt-4o-mini)
        excluded_tools = {"directory_tree"}
        tools = [t for t in tools if t.name not in excluded_tools]

        print("-" * 70)
        print("LLM Agent Code Review")
        print("-" * 70)
        print(f"Data Source: {self._data_source}")
        print(f"Repository: {repo_path}")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Available tools: {len(tools)}")
        print()

        # Initialize LLM (gpt-4o-mini doesn't support temperature=0)
        llm_kwargs = {"model": self.model}
        if "gpt-5" not in self.model:
            llm_kwargs["temperature"] = self.temperature
        llm = ChatOpenAI(**llm_kwargs)

        # Create agent
        agent = create_agent(llm, tools)

        print("Agent created. Sending user request...")
        print()
        print(f"User Request: {self.user_request[:200]}...")
        print()
        print("-" * 70)
        print("Agent Execution (tool calls will be shown)")
        print("-" * 70)

        # Execute agent with logging
        result = await run_agent_with_logging(agent, self.user_request, verbose=True)

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
            client.metrics_collector.add_custom_metric("data_source", self._data_source)
            client.metrics_collector.add_custom_metric("tool_call_counts", tool_counts)
            client.metrics_collector.add_custom_metric("total_tool_calls", len(client.metrics_collector.entries))

        # Check if report was written
        report_file = Path("/edgeagent/results/scenario1_agent_code_review_report.md")
        if report_file.exists():
            report_content = report_file.read_text()
            print(f"Report written to: {report_file}")
            print(f"Report size: {len(report_content)} bytes")
            return report_content
        else:
            print("[WARN] Agent did not write a report file")
            return final_message


async def main():
    """Run the LLM Agent-based Code Review scenario"""

    parser = argparse.ArgumentParser(description="Run Scenario 01 with LLM Agent")
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
    print("Scenario 01: Code Review Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print(f"Scheduler: {args.scheduler}")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    # Create scheduler
    registry = ToolRegistry.from_yaml(config_path)
    if args.scheduler == "brute_force":
        scheduler = BruteForceChainScheduler(config_path, system_config_path, registry)
    else:
        scheduler = create_scheduler(args.scheduler, config_path, registry)

    scenario = AgentCodeReviewScenario(
        config_path=config_path,
        output_dir="results/scenario1_agent",
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
        csv_path = result.metrics.save_csv("results/scenario1_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
