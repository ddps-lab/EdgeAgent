#!/usr/bin/env python3
"""
Scenario 3: Research Assistant Pipeline - With Chain Scheduling

Tool Chain:
    fetch(×N) -> summarize_text(×N) -> aggregate_list -> combine_research_results -> write_file

Mode: script
- Tool sequence is STATIC (predefined)
- Scheduler runs FIRST via schedule_chain() to determine optimal placement
- get_backend_tools(placement_map) 사용: 필요한 서버만 연결, MetricsWrappedTool 반환
- ProxyTool 없음 (스케줄링 오버헤드 제거)
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import EdgeAgentMCPClient
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import create_scheduler
from edgeagent.metrics import print_chain_scheduling_result


# Tool Chain 정의 (순서 고정)
TOOL_CHAIN = [
    "fetch",
    "summarize_text",
    "aggregate_list",
    "combine_research_results",
    "write_file",
]


def parse_tool_result(result):
    """Parse tool result - handle MCP response format, dict, and JSON string."""
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and 'text' in first_item:
            text_content = first_item['text']
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"raw": text_content}
        if isinstance(first_item, dict):
            return first_item
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


# Fallback URLs for quick testing
FALLBACK_URLS = [
    {"title": "Wikipedia: Intelligent Agent", "url": "https://en.wikipedia.org/wiki/Intelligent_agent", "topic": "AI Agents"},
    {"title": "Wikipedia: Large Language Model", "url": "https://en.wikipedia.org/wiki/Large_language_model", "topic": "LLM"},
    {"title": "Wikipedia: Machine Learning", "url": "https://en.wikipedia.org/wiki/Machine_learning", "topic": "ML"},
]


async def run_research_assistant(
    config_path: Path,
    system_config_path: Path,
    scheduler_type: str = "brute_force",
    output_dir: str = "results/scenario3",
    max_urls: int = 3,
) -> dict:
    """
    Run Research Assistant Pipeline with Chain Scheduling.

    Scheduler runs FIRST (no MCP connection), then only required locations connect.
    """
    start_time = time.time()

    # 경로 설정 (모든 location에서 동일한 구조)
    report_path = "/edgeagent/results/scenario3_research_report.md"

    # Ensure directories exist
    Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load research items
    data_dir = Path(__file__).parent.parent / "data"
    s2orc_papers = load_s2orc_papers(data_dir, max_urls)
    if s2orc_papers:
        research_items = s2orc_papers
        data_source = "S2ORC"
    else:
        research_items = FALLBACK_URLS[:max_urls]
        data_source = "Wikipedia"

    # ================================================================
    # Step 1: Run Scheduler FIRST (NO MCP connection!)
    # ================================================================
    registry = ToolRegistry.from_yaml(config_path)

    # 모든 scheduler_type에서 create_scheduler()와 schedule_chain() 사용
    chain_scheduler = create_scheduler(
        scheduler_type,
        config_path,
        registry,
        system_config_path=system_config_path,
    )
    scheduling_result = chain_scheduler.schedule_chain(TOOL_CHAIN)

    # Chain Scheduling 결과 출력 (metrics.py 유틸리티 사용)
    print_chain_scheduling_result(
        scheduling_result,
        title=f"Step 1: Chain Scheduling ({scheduler_type})",
    )
    print()

    placement_map = {p.tool_name: p.location for p in scheduling_result.placements}

    # ================================================================
    # Step 2: get_backend_tools()로 필요한 서버만 연결
    # ================================================================
    print("=" * 70)
    print("Step 2: Connect required servers only")
    print("=" * 70)

    async with EdgeAgentMCPClient(
        config_path,
        scheduler=chain_scheduler,
        system_config_path=system_config_path,
        collect_metrics=True,
    ) as client:
        # Chain Scheduling 결과 설정 (개별 tool 메트릭에 score, exec_cost 등 기록)
        client.set_chain_scheduling_result(scheduling_result)
        # placement_map 기반으로 필요한 서버만 연결
        tool_by_name = await client.get_backend_tools(placement_map)
        print(f"  Loaded {len(tool_by_name)} tools (MetricsWrappedTool)")
        for name, tool in tool_by_name.items():
            print(f"    {name} -> {tool.location}")
        print()

        def get_tool(tool_name: str):
            return tool_by_name.get(tool_name)

        # ================================================================
        # Step 3: Prepare data
        # ================================================================
        print("=" * 70)
        print("Step 3: Prepare data")
        print("=" * 70)
        print(f"  Data Source: {data_source}")
        print(f"  Research Items ({len(research_items)}):")
        for item in research_items:
            print(f"    - {item['title'][:50]}...")
        print()

        # ================================================================
        # Step 4: Execute pipeline
        # ================================================================
        print("=" * 70)
        print("Step 4: Execute pipeline")
        print("=" * 70)

        total_input_size = 0
        articles = []
        summaries = []

        # 4.1: fetch (×N)
        tool_name = "fetch"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location} (×{len(research_items)})")

        fetch_tool = get_tool("fetch")
        for i, item in enumerate(research_items, 1):
            print(f"    [{i}/{len(research_items)}] Fetching: {item['title'][:40]}...")
            if fetch_tool:
                try:
                    raw_content = await fetch_tool.ainvoke({"url": item["url"]})
                    content = str(raw_content) if raw_content else ""
                    content_size = len(content)
                    total_input_size += content_size
                    articles.append({
                        "title": item["title"],
                        "url": item["url"],
                        "topic": item["topic"],
                        "content": content[:10000],
                        "size_bytes": content_size,
                    })
                    print(f"      Fetched {content_size:,} bytes")
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    articles.append({"title": item["title"], "url": item["url"], "topic": item["topic"], "content": f"Error: {e}", "size_bytes": 0})
            else:
                articles.append({"title": item["title"], "url": item["url"], "topic": item["topic"], "content": "Mock content", "size_bytes": 100})
                print(f"      [SKIP] No tool available")

        # 4.2: summarize_text (×N)
        tool_name = "summarize_text"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location} (×{len(articles)})")

        summarize_tool = get_tool("summarize_text")
        for i, article in enumerate(articles, 1):
            print(f"    [{i}/{len(articles)}] Summarizing: {article['title'][:40]}...")
            if summarize_tool and len(article["content"]) > 100:
                try:
                    raw_summary = await summarize_tool.ainvoke({
                        "text": article["content"],
                        "max_length": 100,
                        "style": "concise",
                    })
                    summary = str(raw_summary)
                    summaries.append({"title": article["title"], "topic": article["topic"], "summary": summary})
                    print(f"      Generated {len(summary)} chars")
                except Exception as e:
                    summaries.append({"title": article["title"], "topic": article["topic"], "summary": f"Error: {e}"})
                    print(f"      [ERROR] {e}")
            else:
                summaries.append({"title": article["title"], "topic": article["topic"], "summary": article["content"][:200]})
                print(f"      [SKIP] Content too short or no tool")

        # 4.3: aggregate_list
        tool_name = "aggregate_list"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location}")

        aggregate_tool = get_tool("aggregate_list")
        if aggregate_tool:
            raw_agg = await aggregate_tool.ainvoke({"items": summaries, "group_by": "topic"})
            agg_result = parse_tool_result(raw_agg)
            print(f"    Grouped by topic: {list(agg_result.get('groups', {}).keys())}")
        else:
            agg_result = {"groups": {"all": len(summaries)}}
            print(f"    Using basic aggregation")

        # 4.4: combine_research_results
        tool_name = "combine_research_results"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location}")

        combine_tool = get_tool("combine_research_results")
        if combine_tool:
            raw_combined = await combine_tool.ainvoke({
                "results": [{"title": s["title"], "summary": s["summary"]} for s in summaries],
            })
            combined = parse_tool_result(raw_combined)
            combined_text = combined.get('combined_text', combined.get('combined_summary', ''))
            print(f"    Combined: {len(combined_text)} chars")
        else:
            combined_text = "\n\n".join(s["summary"] for s in summaries)
            print(f"    Using basic combination")

        # 4.5: write_file
        tool_name = "write_file"
        location = placement_map.get(tool_name, "DEVICE")
        print(f"\n  [{tool_name}] -> {location}")

        report = f"""# Research Report: AI Agents

## Overview
- Topic: AI Agents and Related Technologies
- Sources analyzed: {len(articles)}
- Total data processed: {total_input_size:,} bytes

## Sources
"""
        for article in articles:
            report += f"### {article['title']}\n- URL: {article['url']}\n- Size: {article['size_bytes']:,} bytes\n\n"

        report += "## Summaries\n\n"
        for summary in summaries:
            report += f"### {summary['title']}\n{summary['summary']}\n\n"

        report += f"""## Combined Analysis
{combined_text[:500]}

---
*Generated by EdgeAgent Research Assistant Pipeline*
"""

        write_tool = get_tool("write_file")
        if write_tool:
            await write_tool.ainvoke({"path": report_path, "content": report})
        else:
            Path(report_path).write_text(report)
        print(f"    Written to {report_path}")

        # Extract metrics before context closes
        metrics_entries = []
        metrics_collector = client.get_metrics()
        if metrics_collector:
            metrics_entries = [e.to_dict() for e in metrics_collector.entries]
            # Save CSV
            csv_path = Path(output_dir) / "metrics.csv"
            metrics_collector.save_csv(str(csv_path))
            print(f"  Metrics CSV saved to: {csv_path}")

    # ================================================================
    # Step 5: Summary
    # ================================================================
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    used_locations = set(placement_map.values())

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Scheduler: {scheduler_type}")
    print(f"  Success: True")
    print(f"  Total Time: {total_time_ms:.2f} ms")
    print(f"  Locations used: {used_locations}")
    print(f"  Tool calls: {len(metrics_entries)}")
    print("=" * 70)

    result = {
        "scenario_name": "research_assistant",
        "scheduler_type": scheduler_type,
        "success": True,
        "total_time_ms": total_time_ms,
        "used_locations": list(used_locations),
        "placement_map": placement_map,
        "chain_scheduling": {
            "total_cost": scheduling_result.total_score,
            "search_space_size": scheduling_result.search_space_size,
            "decision_time_ns": scheduling_result.decision_time_ns,
            "decision_time_ms": scheduling_result.decision_time_ns / 1e6,
        },
        "metrics_entries": metrics_entries,
        "tool_call_count": len(metrics_entries),
    }

    output_path = Path(output_dir) / f"research_assistant_{int(start_time)}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default="brute_force")
    parser.add_argument("--max-urls", type=int, default=3)
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario3.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    print("=" * 70)
    print("Scenario 3: Research Assistant Pipeline")
    print("=" * 70)
    print(f"Scheduler: {args.scheduler}")
    print()

    result = await run_research_assistant(
        config_path=config_path,
        system_config_path=system_config_path,
        scheduler_type=args.scheduler,
        max_urls=args.max_urls,
    )

    return result.get("success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
