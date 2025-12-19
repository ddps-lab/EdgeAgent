#!/usr/bin/env python3
"""
Scenario 1: Code Review Pipeline - With Chain Scheduling

Tool Chain:
    read_file -> git_diff -> summarize_text -> merge_summaries -> write_file

Mode: script
- Tool sequence is STATIC (predefined)
- Scheduler runs FIRST via schedule_chain() to determine optimal placement
- get_backend_tools(placement_map) 사용: 필요한 서버만 연결, MetricsWrappedTool 반환
- ProxyTool 없음 (스케줄링 오버헤드 제거)
"""

import asyncio
import json
import shutil
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import EdgeAgentMCPClient
from edgeagent.registry import ToolRegistry
from edgeagent.scheduler import BruteForceChainScheduler, create_scheduler


# Tool Chain 정의 (실제 서버에 존재하는 툴 사용)
TOOL_CHAIN = [
    "read_file",
    "git_diff",
    "summarize_text",
    "merge_summaries",
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


def prepare_repo(repo_path: str) -> tuple[Path, str]:
    """Prepare Git repository - returns (repo_path, data_source)."""
    data_dir = Path(__file__).parent.parent / "data" / "scenario1"
    defects4j_dir = data_dir / "defects4j"
    sample_repo = data_dir / "sample_repo"

    repo_source = None
    data_source = None

    if defects4j_dir.exists():
        for subdir in defects4j_dir.iterdir():
            if subdir.is_dir() and (subdir / ".git").exists():
                repo_source = subdir
                data_source = f"Defects4J ({subdir.name})"
                break

    if repo_source is None and sample_repo.exists() and (sample_repo / ".git").exists():
        repo_source = sample_repo
        data_source = "Generated sample repository"

    if repo_source is None:
        raise FileNotFoundError(
            f"No Git repository found in {data_dir}\n"
            "Run 'python scripts/setup_test_data.py -s 1' for test data"
        )

    device_repo = Path(repo_path)
    device_repo.parent.mkdir(parents=True, exist_ok=True)
    if device_repo.exists():
        shutil.rmtree(device_repo)
    shutil.copytree(repo_source, device_repo)

    return device_repo, data_source


async def run_code_review(
    config_path: Path,
    system_config_path: Path,
    scheduler_type: str = "brute_force",
    subagent_mode: bool = False,
    output_dir: str = "results/scenario1",
) -> dict:
    """
    Code Review Pipeline 실행 (Chain Scheduling 적용)

    핵심 설계:
    1. 스케줄러가 먼저 실행 (MCP 연결 없이) -> 최적 배치 결정
    2. get_backend_tools(placement_map)으로 필요한 서버만 연결
    3. MetricsWrappedTool 사용 (ProxyTool 오버헤드 없음)
    """
    start_time = time.time()

    # 경로 설정 (모든 location에서 동일한 구조)
    repo_path = "/edgeagent/repos/scenario1"
    report_path = "/edgeagent/results/scenario1_code_review_report.md"

    # 디렉토리 생성
    Path("/edgeagent/repos").mkdir(parents=True, exist_ok=True)
    Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare repository first
    device_repo, data_source = prepare_repo(repo_path)

    # ================================================================
    # Step 1: 스케줄러 먼저 실행 (MCP 연결 없이)
    # ================================================================
    print("=" * 70)
    print(f"Step 1: Chain Scheduling ({scheduler_type})")
    print("=" * 70)

    registry = ToolRegistry.from_yaml(config_path)

    if scheduler_type == "brute_force":
        chain_scheduler = BruteForceChainScheduler(
            config_path=config_path,
            system_config_path=system_config_path,
            registry=registry,
            subagent_mode=subagent_mode,
        )
        scheduling_result = chain_scheduler.schedule_chain(TOOL_CHAIN)

        print(f"Total Cost: {scheduling_result.total_score:.4f}")
        print(f"Search Space: {scheduling_result.search_space_size}")
        print(f"Decision Time: {scheduling_result.decision_time_ns / 1e6:.2f} ms")
        print()
        print("Optimal Placement:")
        for p in scheduling_result.placements:
            fixed_mark = "[FIXED]" if p.fixed else ""
            print(f"  {p.tool_name:25} -> {p.location:6} (cost={p.score:.3f}, comp={p.exec_cost:.3f}, comm={p.trans_cost:.3f}) {fixed_mark}")
        print()

        placement_map = {p.tool_name: p.location for p in scheduling_result.placements}
    elif scheduler_type == "all_device":
        chain_scheduler = None
        placement_map = {tool: "DEVICE" for tool in TOOL_CHAIN}
        print("Placement: All tools -> DEVICE")
    elif scheduler_type == "all_edge":
        chain_scheduler = None
        placement_map = {tool: "EDGE" for tool in TOOL_CHAIN}
        print("Placement: All tools -> EDGE")
    elif scheduler_type == "all_cloud":
        chain_scheduler = None
        placement_map = {tool: "CLOUD" for tool in TOOL_CHAIN}
        print("Placement: All tools -> CLOUD")
    elif scheduler_type in ("static", "heuristic"):
        chain_scheduler = create_scheduler(scheduler_type, config_path, registry)
        placement_map = {}
        print("Optimal Placement:")
        for tool in TOOL_CHAIN:
            result = chain_scheduler.get_location_for_call_with_reason(tool, {})
            placement_map[tool] = result.location
            print(f"  {tool:25} -> {result.location:6} (reason={result.reason})")
        print()
    else:
        # 기본값: all_device
        chain_scheduler = None
        placement_map = {tool: "DEVICE" for tool in TOOL_CHAIN}
        print(f"Unknown scheduler '{scheduler_type}', using all_device")

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
        print(f"  Repository: {device_repo}")
        print()

        # ================================================================
        # Step 4: Execute pipeline
        # read_file -> git_diff -> summarize_text -> merge_summaries -> write_file
        # ================================================================
        print("=" * 70)
        print("Step 4: Execute pipeline")
        print("=" * 70)

        # 4.1: read_file
        tool_name = "read_file"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        read_tool = get_tool("read_file")
        # 리포지토리의 주요 파일 읽기
        main_file = device_repo / "src" / "main.py"
        if not main_file.exists():
            # 다른 파일 찾기
            py_files = list(device_repo.rglob("*.py"))
            main_file = py_files[0] if py_files else device_repo / "README.md"

        file_content = await read_tool.ainvoke({"path": str(main_file)})
        print(f"    Read {len(str(file_content))} chars from {main_file.name}")

        # 4.2: git_diff
        tool_name = "git_diff"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        diff_tool = get_tool("git_diff")
        raw_diff = await diff_tool.ainvoke({"repo_path": str(device_repo), "target": "HEAD~1"})
        diff_result = parse_tool_result(raw_diff)
        diff_content = diff_result.get("diff", str(raw_diff)) if isinstance(diff_result, dict) else str(raw_diff)
        print(f"    Retrieved diff ({len(str(diff_content))} chars)")

        # 4.3: summarize_text
        tool_name = "summarize_text"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        summarize_tool = get_tool("summarize_text")
        # Combine file content and diff for summarization
        text_to_summarize = f"Code:\n{str(file_content)[:3000]}\n\nRecent Changes:\n{str(diff_content)[:2000]}"
        raw_summary = await summarize_tool.ainvoke({
            "text": text_to_summarize,
            "max_length": 200,
            "style": "detailed"
        })
        code_summary = parse_tool_result(raw_summary)
        print(f"    Generated code summary")

        # 4.4: merge_summaries
        tool_name = "merge_summaries"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        merge_tool = get_tool("merge_summaries")
        summaries_to_merge = [
            {"type": "code", "content": str(code_summary)},
            {"type": "diff", "content": str(diff_content)[:1000]}
        ]
        raw_merged = await merge_tool.ainvoke({"summaries": summaries_to_merge})
        merged_summary = parse_tool_result(raw_merged)
        print(f"    Merged summaries")

        # 4.5: write_file
        tool_name = "write_file"
        location = placement_map[tool_name]
        print(f"\n  [{tool_name}] -> {location}")

        report = f"""# Code Review Report

## Repository
- Path: {device_repo}

## Code Summary
{code_summary.get('summary', str(code_summary)) if isinstance(code_summary, dict) else code_summary}

## Recent Changes (Diff)
```
{str(diff_content)[:2000]}
```

## Merged Summary
{merged_summary.get('merged', str(merged_summary)) if isinstance(merged_summary, dict) else merged_summary}
"""

        write_tool = get_tool("write_file")
        await write_tool.ainvoke({"path": report_path, "content": report})
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
        "scenario_name": "code_review",
        "scheduler_type": scheduler_type,
        "success": True,
        "total_time_ms": total_time_ms,
        "used_locations": list(used_locations),
        "placement_map": placement_map,
        "metrics_entries": metrics_entries,
        "tool_call_count": len(metrics_entries),
    }

    output_path = Path(output_dir) / f"code_review_{int(start_time)}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default="brute_force")
    parser.add_argument("--subagent-mode", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    print("=" * 70)
    print("Scenario 1: Code Review Pipeline")
    print("=" * 70)
    print(f"Scheduler: {args.scheduler}")
    print(f"SubAgent Mode: {args.subagent_mode}")
    print()

    result = await run_code_review(
        config_path=config_path,
        system_config_path=system_config_path,
        scheduler_type=args.scheduler,
        subagent_mode=args.subagent_mode,
    )

    return result.get("success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
