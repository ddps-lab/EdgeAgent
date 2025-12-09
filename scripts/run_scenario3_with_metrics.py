#!/usr/bin/env python3
"""
Scenario 3: Research Assistant Pipeline - With Unified Metrics Collection

Tool Chain:
    (S2ORC papers) -> fetch(×N) -> summarize(×N) -> data_aggregate -> filesystem(write)
    N/A               EDGE          EDGE            EDGE             DEVICE

This scenario demonstrates:
- Processing academic papers from S2ORC (Semantic Scholar Open Research Corpus)
- Fetching paper pages from Semantic Scholar
- Summarizing each article
- Aggregating summaries into a research report
- Data reduction from raw web pages to concise report

Data Sources:
- Primary: S2ORC dataset (download via scripts/download_public_datasets.py -s 3)
- Fallback: Wikipedia URLs for quick testing
"""

import asyncio
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import ScenarioRunner, EdgeAgentMCPClient


def parse_tool_result(result):
    """Parse tool result - handle both dict and JSON string."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return {"raw": str(result)}


def load_s2orc_papers(data_dir: Path, max_papers: int = 5) -> list[dict]:
    """Load papers from S2ORC dataset."""
    s2orc_dir = data_dir / "scenario3" / "s2orc"
    papers_file = s2orc_dir / "papers.json"

    if papers_file.exists():
        with open(papers_file) as f:
            papers = json.load(f)
        return [
            {
                "title": p["title"],
                "url": f"https://www.semanticscholar.org/paper/{p['id']}",
                "topic": "AI/ML",
                "abstract": p.get("abstract", ""),
            }
            for p in papers[:max_papers]
        ]
    return []


# Fallback URLs (Wikipedia) for quick testing without S2ORC download
FALLBACK_URLS = [
    {
        "title": "Wikipedia: Intelligent Agent",
        "url": "https://en.wikipedia.org/wiki/Intelligent_agent",
        "topic": "AI Agents",
    },
    {
        "title": "Wikipedia: Large Language Model",
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
        "topic": "LLM",
    },
    {
        "title": "Wikipedia: Machine Learning",
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "topic": "ML",
    },
]


class ResearchAssistantScenario(ScenarioRunner):
    """Research Assistant Pipeline Scenario"""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results/scenario3",
        max_urls: int = 3,
        use_s2orc: bool = True,
    ):
        super().__init__(config_path, output_dir)
        self.max_urls = max_urls
        self.use_s2orc = use_s2orc
        self._report_path = "/tmp/edgeagent_device/research_report.md"
        # Ensure device directory exists
        Path("/tmp/edgeagent_device").mkdir(parents=True, exist_ok=True)

        # Try to load S2ORC papers first
        data_dir = Path(__file__).parent.parent / "data"
        if use_s2orc:
            s2orc_papers = load_s2orc_papers(data_dir, max_urls)
            if s2orc_papers:
                self._research_items = s2orc_papers
                self._data_source = "S2ORC"
            else:
                self._research_items = FALLBACK_URLS[:max_urls]
                self._data_source = "Wikipedia (S2ORC not found)"
        else:
            self._research_items = FALLBACK_URLS[:max_urls]
            self._data_source = "Wikipedia"

        self._urls = [item["url"] for item in self._research_items]

    @property
    def name(self) -> str:
        return "research_assistant"

    @property
    def description(self) -> str:
        return "Fetch web articles, summarize each, and generate a research report"

    @property
    def user_request(self) -> str:
        return "Research AI agents topic: fetch articles, summarize, and create a report"

    def get_validation_context(self) -> dict:
        """Provide context for validation"""
        return {
            "report_path": self._report_path,
            "urls": self._urls,
        }

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute the research assistant pipeline"""

        # Find tools by name
        tool_by_name = {t.name: t for t in tools}

        print("Available tools:")
        for name in sorted(tool_by_name.keys()):
            print(f"  - {name}")
        print()

        # Research items to process (S2ORC papers or fallback URLs)
        items = self._research_items
        print(f"Data Source: {self._data_source}")
        print(f"Research Items ({len(items)}):")
        for item in items:
            print(f"  - {item['title'][:60]}...")
        print()

        total_input_size = 0
        articles = []
        summaries = []

        # Step 1: Fetch articles (fetch -> EDGE)
        print("-" * 70)
        print(f"Step 1: Fetch {len(items)} web articles (fetch -> EDGE)")
        print("-" * 70)

        fetch_tool = tool_by_name.get("fetch")

        for i, item in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] Fetching: {item['title'][:50]}...")

            if fetch_tool:
                try:
                    raw_content = await fetch_tool.ainvoke({
                        "url": item["url"],
                    })
                    content = str(raw_content) if raw_content else ""
                    content_size = len(content)
                    total_input_size += content_size
                    articles.append({
                        "title": item["title"],
                        "url": item["url"],
                        "topic": item["topic"],
                        "content": content[:10000],  # Limit content size
                        "size_bytes": content_size,
                    })
                    print(f"       Fetched {content_size:,} bytes")
                except Exception as e:
                    print(f"       [ERROR] {e}")
                    articles.append({
                        "title": item["title"],
                        "url": item["url"],
                        "topic": item["topic"],
                        "content": f"Error fetching: {e}",
                        "size_bytes": 0,
                    })
            else:
                print("       [SKIP] fetch tool not available")
                articles.append({
                    "title": item["title"],
                    "url": item["url"],
                    "topic": item["topic"],
                    "content": "Mock content for testing",
                    "size_bytes": 100,
                })

        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("articles_fetched", len(articles))
            client.metrics_collector.add_custom_metric("total_fetch_bytes", total_input_size)
        print()

        # Step 2: Summarize each article (summarize -> EDGE)
        print("-" * 70)
        print(f"Step 2: Summarize {len(articles)} articles (summarize -> EDGE)")
        print("-" * 70)

        summarize_tool = tool_by_name.get("summarize_text")

        for i, article in enumerate(articles, 1):
            print(f"  [{i}/{len(articles)}] Summarizing: {article['title']}")

            if summarize_tool and len(article["content"]) > 100:
                try:
                    raw_summary = await summarize_tool.ainvoke({
                        "text": article["content"],
                        "max_length": 100,
                        "style": "concise",
                    })
                    summary = str(raw_summary)
                    summaries.append({
                        "title": article["title"],
                        "topic": article["topic"],
                        "summary": summary,
                    })
                    print(f"       Generated {len(summary)} chars")
                except Exception as e:
                    print(f"       [ERROR] {e}")
                    summaries.append({
                        "title": article["title"],
                        "topic": article["topic"],
                        "summary": f"Summary error: {e}",
                    })
            else:
                # Fallback or short content
                summaries.append({
                    "title": article["title"],
                    "topic": article["topic"],
                    "summary": article["content"][:200] if len(article["content"]) < 100 else "Content too short to summarize.",
                })
                print("       [SKIP] Content too short or no tool")

        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("summaries_generated", len(summaries))
        print()

        # Step 3: Aggregate summaries (data_aggregate -> EDGE)
        print("-" * 70)
        print("Step 3: Aggregate research data (data_aggregate -> EDGE)")
        print("-" * 70)

        aggregate_tool = tool_by_name.get("aggregate_list")

        if aggregate_tool:
            raw_agg = await aggregate_tool.ainvoke({
                "items": summaries,
                "group_by": "topic",
            })
            agg_result = parse_tool_result(raw_agg)
            print(f"  Grouped by topic: {list(agg_result.get('groups', {}).keys())}")
        else:
            agg_result = {"groups": {"all": len(summaries)}}
            print("  Using basic aggregation")

        # Also use combine_research_results if available
        combine_tool = tool_by_name.get("combine_research_results")
        if combine_tool:
            raw_combined = await combine_tool.ainvoke({
                "results": [{"title": s["title"], "summary": s["summary"]} for s in summaries],
            })
            combined = parse_tool_result(raw_combined)
            # Server returns 'combined_text', not 'combined_summary'
            combined_text = combined.get('combined_text', combined.get('combined_summary', ''))
            print(f"  Combined: {len(combined_text)} chars")
            combined = {"combined_summary": combined_text}
        else:
            combined = {"combined_summary": "\n\n".join(s["summary"] for s in summaries)}
        print()

        # Step 4: Generate and write report (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 4: Write research report (filesystem -> DEVICE)")
        print("-" * 70)

        # Generate report
        report = f"""# Research Report: AI Agents

## Overview
- Topic: AI Agents and Related Technologies
- Sources analyzed: {len(articles)}
- Total data processed: {total_input_size:,} bytes

## Sources

"""
        for article in articles:
            report += f"### {article['title']}\n"
            report += f"- URL: {article['url']}\n"
            report += f"- Topic: {article['topic']}\n"
            report += f"- Size: {article['size_bytes']:,} bytes\n\n"

        report += "## Summaries\n\n"
        for summary in summaries:
            report += f"### {summary['title']}\n"
            report += f"{summary['summary']}\n\n"

        report += f"""## Key Findings

Based on the analyzed sources, the following key themes emerge:

1. **Intelligent Agents**: Software entities that perceive and act on their environment
2. **Large Language Models**: Foundation for modern AI agents with natural language understanding
3. **Machine Learning**: Underlying technology enabling adaptive and learning agents

## Aggregated Analysis

{combined.get('combined_summary', 'No combined summary available.')[:500]}

## Conclusion

This research provides an overview of AI agents and related technologies.
The field is rapidly evolving with advances in language models and machine learning.

---
*Generated by EdgeAgent Research Assistant Pipeline*
"""

        write_tool = tool_by_name.get("write_file")
        output_path = "/tmp/edgeagent_device/research_report.md"
        if write_tool:
            await write_tool.ainvoke({
                "path": output_path,
                "content": report
            })
        else:
            Path(output_path).write_text(report)

        print(f"  Report written to: {output_path}")
        print(f"  Report size: {len(report)} bytes")
        print()

        # Data flow summary
        output_size = len(report)
        reduction = (1 - output_size / total_input_size) * 100 if total_input_size > 0 else 0

        print("=" * 70)
        print("Data Flow Summary")
        print("=" * 70)
        print(f"  Input size:  {total_input_size:,} bytes (fetched articles)")
        print(f"  Output size: {output_size:,} bytes (report)")
        print(f"  Reduction:   {reduction:.1f}%")
        print()

        return report


async def main():
    """Run the Research Assistant scenario with metrics collection"""
    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"

    scenario = ResearchAssistantScenario(
        config_path=config_path,
        output_dir="results/scenario3",
        max_urls=3,  # Limit to 3 URLs for faster testing
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional: Export metrics to CSV for pandas analysis
    if result.metrics:
        csv_path = result.metrics.save_csv("results/scenario3/metrics.csv")
        print(f"Metrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
