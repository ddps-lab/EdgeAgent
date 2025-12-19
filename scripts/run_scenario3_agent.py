#!/usr/bin/env python3
"""
Scenario 03: Research Assistant Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the research task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Tool Chain (expected, but LLM decides):
    fetch -> summarize -> aggregate -> filesystem(write)
    EDGE    EDGE        EDGE        DEVICE
"""

import argparse
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
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import create_scheduler, BruteForceChainScheduler
from scripts.agent_utils import run_agent_with_logging


# Default URLs (fallback if S2ORC not available)
DEFAULT_RESEARCH_URLS = [
    "https://en.wikipedia.org/wiki/Intelligent_agent",
    "https://en.wikipedia.org/wiki/Large_language_model",
]


def load_s2orc_urls(max_urls: int = 3) -> tuple[list[str], str]:
    """Load research URLs from S2ORC dataset.

    Note: Semantic Scholar paper URLs are automatically converted to API calls
    by the fetch server, avoiding the HTTP 202 issue.

    Returns:
        Tuple of (urls, data_source_description)
    """
    s2orc_dir = Path(__file__).parent.parent / "data" / "scenario3" / "s2orc"
    paper_urls_file = s2orc_dir / "paper_urls.txt"

    if paper_urls_file.exists():
        urls = paper_urls_file.read_text().strip().split('\n')
        urls = [u.strip() for u in urls if u.strip()][:max_urls]
        if urls:
            return urls, f"S2ORC ({len(urls)} papers)"

    return DEFAULT_RESEARCH_URLS, "Wikipedia (fallback)"


# Load URLs at module level
RESEARCH_URLS, DATA_SOURCE = load_s2orc_urls(max_urls=3)

# System prompt for the Research Assistant Agent
RESEARCH_ASSISTANT_SYSTEM_PROMPT = f"""You are a research assistant. Your task is to research topics by fetching web content, summarizing it, and generating research reports.

You have access to the following tools:
- fetch: Fetch content from a URL (converts to markdown)
- summarize_text: Summarize text content
- summarize_documents: Summarize multiple documents at once
- aggregate_list: Group and aggregate data
- combine_research_results: Combine multiple research results
- write_file: Write files to the filesystem

For this research task, use these URLs:
{chr(10).join(f"- {url}" for url in RESEARCH_URLS)}

When conducting research, follow this workflow:
1. Fetch content from each URL using the fetch tool
2. Summarize each fetched content using summarize_text
3. Aggregate the summaries using aggregate_list or combine_research_results
4. Write a comprehensive research report using write_file

Write the final report to /edgeagent/results/scenario3_agent_research_report.md

Include in your report:
- Overview of the research topic
- Key findings from each source
- Synthesized conclusions
- References to the sources
"""


class AgentResearchAssistantScenario(ScenarioRunner):
    """
    LLM Agent-based Research Assistant Scenario.

    Unlike the script-based version, this uses an LLM to autonomously
    decide which tools to call and in what order.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results/scenario3_agent",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        scheduler=None,
    ):
        super().__init__(config_path, output_dir, scheduler=scheduler)
        self.model = model
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "research_assistant_agent"

    @property
    def description(self) -> str:
        return "LLM Agent autonomously researches a topic and generates a report"

    @property
    def user_request(self) -> str:
        urls_str = ", ".join(RESEARCH_URLS)
        return (
            f"Research the topic of AI agents by fetching and analyzing these URLs: {urls_str}. "
            "Summarize each source, synthesize the findings, and write a comprehensive "
            "research report to /edgeagent/results/scenario3_agent_research_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute research using LLM Agent"""

        # Filter out problematic tools (directory_tree causes issues with gpt-4o-mini)
        excluded_tools = {"directory_tree"}
        tools = [t for t in tools if t.name not in excluded_tools]

        # Ensure output directory exists
        Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)

        print("-" * 70)
        print("LLM Agent Research Assistant")
        print("-" * 70)
        print(f"Data Source: {DATA_SOURCE}")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Research URLs: {len(RESEARCH_URLS)}")
        for url in RESEARCH_URLS:
            print(f"  - {url}")
        print(f"Available tools: {len(tools)}")
        print()

        # Initialize LLM (gpt-4o-mini doesn't support temperature=0)
        llm_kwargs = {"model": self.model}
        if "gpt-5" not in self.model:
            llm_kwargs["temperature"] = self.temperature
        llm = ChatOpenAI(**llm_kwargs)

        # Create agent using langchain.agents.create_agent
        agent = create_agent(llm, tools)

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
            client.metrics_collector.add_custom_metric("urls_researched", len(RESEARCH_URLS))

        # Check if report was written
        report_path = Path("/edgeagent/results/scenario3_agent_research_report.md")
        if report_path.exists():
            report_content = report_path.read_text()
            print(f"Report written to: {report_path}")
            print(f"Report size: {len(report_content)} bytes")
            return report_content
        else:
            print("[WARN] Agent did not write a report file")
            return final_message


async def main():
    """Run the LLM Agent-based Research Assistant scenario"""

    parser = argparse.ArgumentParser(description="Run Scenario 03 with LLM Agent")
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
    print("Scenario 03: Research Assistant Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print(f"Scheduler: {args.scheduler}")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    # Create scheduler
    registry = ToolRegistry.from_yaml(config_path)
    if args.scheduler == "brute_force":
        scheduler = BruteForceChainScheduler(config_path, system_config_path, registry)
    else:
        scheduler = create_scheduler(args.scheduler, config_path, registry)

    scenario = AgentResearchAssistantScenario(
        config_path=config_path,
        output_dir="results/scenario3_agent",
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
        csv_path = result.metrics.save_csv("results/scenario3_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
