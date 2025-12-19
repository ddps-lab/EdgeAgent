#!/usr/bin/env python3
"""
Scenario 4: Image Processing Pipeline - With Chain Scheduling

Tool Chain:
    scan_directory -> compute_image_hash(×N) -> compare_hashes -> batch_resize -> aggregate_list -> write_file

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


# Tool Chain 정의 (순서 고정)
TOOL_CHAIN = [
    "scan_directory",
    "compute_image_hash",
    "compare_hashes",
    "batch_resize",
    "aggregate_list",
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


async def run_image_processing(
    config_path: Path,
    system_config_path: Path,
    scheduler_type: str = "brute_force",
    subagent_mode: bool = False,
    output_dir: str = "results/scenario4",
) -> dict:
    """
    Run Image Processing Pipeline with Chain Scheduling.

    Scheduler runs FIRST (no MCP connection), then only required locations connect.
    """
    start_time = time.time()

    # 경로 설정 (모든 location에서 동일한 구조)
    input_dir = "/edgeagent/data/scenario4/sample_images"
    report_path = "/edgeagent/results/scenario4_image_report.md"

    # Ensure directories exist
    Path("/edgeagent/data/scenario4").mkdir(parents=True, exist_ok=True)
    Path("/edgeagent/results").mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Run Scheduler FIRST (NO MCP connection!)
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

        # Use unified path that works across all locations (DEVICE/EDGE/CLOUD)
        data_dir = Path("/edgeagent/data/scenario4")
        coco_images = data_dir / "coco" / "images"
        sample_images = data_dir / "sample_images"

        if coco_images.exists() and len(list(coco_images.glob("*.jpg"))) > 0:
            image_source = coco_images
            data_source = "COCO 2017"
        elif sample_images.exists():
            image_source = sample_images
            data_source = "Generated test images"
        else:
            raise FileNotFoundError(
                f"No image directory found in {data_dir}\n"
                "Run 'python scripts/setup_test_data.py -s 4' for test data"
            )

        print(f"  Data Source: {data_source}")

        device_images = Path(input_dir)
        device_images.mkdir(parents=True, exist_ok=True)

        # Copy images to target location if different from source
        # Skip if target already has files (from previous run)
        if image_source != device_images:
            existing_files = set(f.name for f in device_images.glob("*") if f.is_file())
            for img in image_source.glob("*"):
                if img.is_file() and img.name not in existing_files:
                    try:
                        shutil.copyfile(img, device_images / img.name)
                    except PermissionError:
                        # Skip if permission denied (already exists from another user)
                        pass

        total_input_size = sum(f.stat().st_size for f in device_images.glob("*") if f.is_file())
        print(f"  Prepared {len(list(device_images.glob('*')))} images ({total_input_size:,} bytes)")
        print()

        # ================================================================
        # Step 4: Execute pipeline
        # ================================================================
        print("=" * 70)
        print("Step 4: Execute pipeline")
        print("=" * 70)

        # 4.1: scan_directory
        tool_name = "scan_directory"
        location = placement_map.get(tool_name, "DEVICE")
        print(f"\n  [{tool_name}] -> {location}")

        scan_tool = get_tool("scan_directory")
        if scan_tool:
            raw_scan = await scan_tool.ainvoke({
                "directory": str(device_images),
                "recursive": False,
                "include_info": True,
            })
            scan_result = parse_tool_result(raw_scan)
        else:
            scan_result = {"image_count": 0, "image_paths": []}
            print(f"    [SKIP] No tool available")

        image_count = scan_result.get("image_count", 0)
        image_paths = scan_result.get("image_paths", [])
        print(f"    Found {image_count} images")

        # 4.2: compute_image_hash (×N)
        tool_name = "compute_image_hash"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location} (×{len(image_paths)})")

        hash_tool = get_tool("compute_image_hash")
        hashes = []
        if hash_tool:
            for img_path in image_paths:
                raw_hash = await hash_tool.ainvoke({
                    "image_path": img_path,
                    "hash_type": "phash",
                })
                hashes.append(parse_tool_result(raw_hash))
            print(f"    Computed {len([h for h in hashes if 'hash' in h])} hashes")
        else:
            print(f"    [SKIP] No tool available")

        # 4.3: compare_hashes
        tool_name = "compare_hashes"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location}")

        compare_tool = get_tool("compare_hashes")
        if compare_tool and hashes:
            raw_compare = await compare_tool.ainvoke({
                "hashes": hashes,
                "threshold": 5,
            })
            compare_result = parse_tool_result(raw_compare)
        else:
            compare_result = {"duplicate_groups": [], "unique_count": len(image_paths)}
            print(f"    [SKIP] No tool available")

        duplicate_groups = compare_result.get("duplicate_groups", [])
        unique_count = compare_result.get("unique_count", 0)
        print(f"    Duplicate groups: {len(duplicate_groups)}, Unique: {unique_count}")

        # 4.4: batch_resize
        tool_name = "batch_resize"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location}")

        unique_paths = compare_result.get("unique_paths", image_paths[:unique_count])
        batch_tool = get_tool("batch_resize")
        if batch_tool and unique_paths:
            raw_batch = await batch_tool.ainvoke({
                "image_paths": unique_paths,
                "max_size": 150,
                "quality": 75,
                "output_format": "JPEG",
            })
            batch_result = parse_tool_result(raw_batch)
        else:
            batch_result = {"successful": 0, "overall_reduction": 0}
            if not batch_tool:
                print(f"    [SKIP] No tool available")
            elif not unique_paths:
                print(f"    [SKIP] No unique images to resize")

        thumbnails_created = batch_result.get("successful", 0)
        thumbnail_reduction = batch_result.get("overall_reduction", 0)
        print(f"    Thumbnails: {thumbnails_created}, Reduction: {thumbnail_reduction:.2%}")

        # 4.5: aggregate_list
        tool_name = "aggregate_list"
        location = placement_map.get(tool_name, "EDGE")
        print(f"\n  [{tool_name}] -> {location}")

        images_info = scan_result.get("images", [])
        if not images_info:
            images_info = [{"path": p, "format": Path(p).suffix.upper()[1:]} for p in image_paths]

        aggregate_tool = get_tool("aggregate_list")
        if aggregate_tool:
            raw_agg = await aggregate_tool.ainvoke({
                "items": images_info,
                "group_by": "format",
            })
            agg_result = parse_tool_result(raw_agg)
            print(f"    Format groups: {list(agg_result.get('groups', {}).keys())}")
        else:
            agg_result = {"groups": {}}
            print(f"    [SKIP] No tool available")

        # 4.6: write_file
        tool_name = "write_file"
        location = placement_map.get(tool_name, "DEVICE")
        print(f"\n  [{tool_name}] -> {location}")

        report = f"""# Image Processing Report

## Summary
- Images scanned: {image_count}
- Total input size: {total_input_size:,} bytes
- Unique images: {unique_count}
- Duplicate groups: {len(duplicate_groups)}

## Thumbnail Generation
- Thumbnails created: {thumbnails_created}
- Reduction ratio: {thumbnail_reduction:.2%}

## Images by Format
"""
        for fmt, count in agg_result.get("groups", {}).items():
            report += f"- {fmt}: {count} images\n"

        report += """
---
*Generated by EdgeAgent Image Processing Pipeline*
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
        "scenario_name": "image_processing",
        "scheduler_type": scheduler_type,
        "success": True,
        "total_time_ms": total_time_ms,
        "used_locations": list(used_locations),
        "placement_map": placement_map,
        "metrics_entries": metrics_entries,
        "tool_call_count": len(metrics_entries),
    }

    output_path = Path(output_dir) / f"image_processing_{int(start_time)}.json"
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

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"
    system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"

    print("=" * 70)
    print("Scenario 4: Image Processing Pipeline")
    print("=" * 70)
    print(f"Scheduler: {args.scheduler}")
    print(f"SubAgent Mode: {args.subagent_mode}")
    print()

    result = await run_image_processing(
        config_path=config_path,
        system_config_path=system_config_path,
        scheduler_type=args.scheduler,
        subagent_mode=args.subagent_mode,
    )

    return result.get("success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
