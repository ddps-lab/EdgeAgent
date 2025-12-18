#!/usr/bin/env python3
"""
Scenario 4: Image Processing Pipeline - SubAgent Mode

This script supports both execution modes:
1. legacy: Single Agent with all tools (current approach)
2. subagent: Sub-Agent Orchestration (location-aware partitioning)

Usage:
    python scripts/run_scenario4_subagent.py --mode legacy
    python scripts/run_scenario4_subagent.py --mode subagent
    python scripts/run_scenario4_subagent.py --compare  # Run both and compare

Tool Chain:
    filesystem(scan) -> image_resize(hash,batch) -> data_aggregate -> filesystem(write)
    DEVICE            EDGE                         EDGE              DEVICE
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from edgeagent import (
    EdgeAgentMCPClient,
    SubAgentOrchestrator,
    OrchestrationConfig,
)


@dataclass
class ExecutionResult:
    """Execution result for comparison"""
    mode: str
    success: bool
    total_time_ms: float
    tool_calls: int
    partitions: int = 0
    partition_times: list[float] = field(default_factory=list)
    final_result: str = ""
    error: str = ""
    scheduler_type: str = ""
    metrics_entries: list[dict] = field(default_factory=list)
    placement_map: dict = field(default_factory=dict)


def load_image_source() -> tuple[Path, str]:
    """Load image source directory"""
    data_dir = Path(__file__).parent.parent / "data" / "scenario4"
    coco_images = data_dir / "coco" / "images"
    sample_images = data_dir / "sample_images"

    if coco_images.exists() and len(list(coco_images.glob("*.jpg"))) > 0:
        return coco_images, "COCO 2017"

    if sample_images.exists() and len(list(sample_images.glob("*"))) > 0:
        return sample_images, "Generated test images"

    raise FileNotFoundError(
        f"No image directory found in {data_dir}\n"
        "Run 'python scripts/setup_test_data.py -s 4' for test data"
    )


def prepare_images() -> tuple[Path, str, int]:
    """Prepare images at device location"""
    image_source, data_source = load_image_source()

    device_images = Path("/edgeagent/data/scenario4/sample_images")
    device_images.mkdir(parents=True, exist_ok=True)

    # Clear existing images
    for f in device_images.glob("*"):
        f.unlink()

    # Copy images
    count = 0
    for img in image_source.glob("*"):
        if img.is_file():
            shutil.copyfile(img, device_images / img.name)
            count += 1

    return device_images, data_source, count


# Prepare images at module load time
IMAGE_PATH, DATA_SOURCE, IMAGE_COUNT = prepare_images()


USER_REQUEST = """
Process images at /edgeagent/data/scenario4/sample_images.

Please:
1. List the directory contents using list_directory to find image files
2. For each image, compute a perceptual hash using compute_image_hash (hash_type="phash")
3. Find duplicate images using compare_hashes with threshold=5
4. Create thumbnails for unique images using batch_resize (max_size=150, quality=75)
5. Write an image processing report to /edgeagent/results/scenario4_image_report.md

The report should include:
- Number of images found
- Number of duplicates detected
- Number of thumbnails created
- Size reduction achieved

Return a summary of the image processing results.
"""

# Tool sequence for Sub-Agent mode (order matters)
# 개별 tool 이름 사용 (서버 이름 아님) - Scheduler가 정확한 profile 참조 가능
TOOL_SEQUENCE = ["list_directory", "scan_directory", "batch_resize", "aggregate_list", "write_file"]


async def run_legacy_mode(config_path: Path, model: str) -> ExecutionResult:
    """Run with legacy single-agent mode"""
    print("\n" + "=" * 70, flush=True)
    print("LEGACY MODE: Single Agent with All Tools", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    tool_calls = 0

    try:
        async with EdgeAgentMCPClient(config_path) as client:
            tools = await client.get_tools()
            print(f"Loaded {len(tools)} tools", flush=True)

            # gpt-4o-mini doesn't support temperature=0
            llm_kwargs = {"model": model}
            if "gpt-5" not in model:
                llm_kwargs["temperature"] = 0
            llm = ChatOpenAI(**llm_kwargs)
            agent = create_agent(llm, tools)

            print("Running agent...", flush=True)
            result_content = ""
            seen_tool_ids = set()

            async for chunk in agent.astream(
                {"messages": [("user", USER_REQUEST)]},
                stream_mode="values",
            ):
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tc_id = tc.get("id", "")
                                if tc_id and tc_id not in seen_tool_ids:
                                    seen_tool_ids.add(tc_id)
                                    tool_calls += 1
                                    print(f"  -> Tool: {tc.get('name', 'unknown')}", flush=True)

                    final_msg = chunk["messages"][-1]
                    if hasattr(final_msg, "content"):
                        result_content = final_msg.content

            elapsed = (time.time() - start_time) * 1000

            return ExecutionResult(
                mode="legacy",
                success=True,
                total_time_ms=elapsed,
                tool_calls=tool_calls,
                final_result=result_content[:500] if result_content else "",
            )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return ExecutionResult(
            mode="legacy",
            success=False,
            total_time_ms=elapsed,
            tool_calls=tool_calls,
            error=f"{e}\n{traceback.format_exc()}",
        )


async def run_subagent_mode(config_path: Path, model: str, scheduler: str = "brute_force") -> ExecutionResult:
    """Run with Sub-Agent Orchestration mode"""
    print("\n" + "=" * 70)
    print(f"SUBAGENT MODE: Location-Aware Orchestration (scheduler={scheduler})")
    print("=" * 70)

    start_time = time.time()

    try:
        config = OrchestrationConfig(
            mode="subagent",
            subagent_endpoints={},  # Local execution
            model=model,
            temperature=0,
            max_iterations=15,  # More iterations for multiple images
        )

        system_config_path = Path(__file__).parent.parent / "config" / "system.yaml"
        orchestrator = SubAgentOrchestrator(
            config_path,
            config,
            system_config_path=system_config_path,
            scheduler_type=scheduler,
        )

        # Show execution plan
        print("\nExecution Plan:")
        orchestrator.print_execution_plan(TOOL_SEQUENCE)

        print("\nRunning orchestration...")

        async with orchestrator:
            result = await orchestrator.run(
                user_request=USER_REQUEST,
                tool_sequence=TOOL_SEQUENCE,
                mode="subagent",
            )

        elapsed = (time.time() - start_time) * 1000

        # Extract partition times and metrics
        partition_times = []
        metrics_entries = []
        if result.partition_results:
            for pr in result.partition_results:
                if "execution_time_ms" in pr:
                    partition_times.append(pr["execution_time_ms"])
                if "metrics_entries" in pr:
                    metrics_entries.extend(pr["metrics_entries"])

        # Get placement map from execution plan
        placement_map = {}
        plan = orchestrator.get_execution_plan(TOOL_SEQUENCE)
        if plan.chain_scheduling_result:
            placement_map = {p.tool_name: p.location for p in plan.chain_scheduling_result.placements}
        else:
            for partition in plan.partitions:
                for tool in partition.tools:
                    placement_map[tool] = partition.location

        return ExecutionResult(
            mode="subagent",
            success=result.success,
            total_time_ms=elapsed,
            tool_calls=result.total_tool_calls,
            partitions=result.partitions_executed,
            partition_times=partition_times,
            final_result=str(result.final_result)[:500] if result.final_result else "",
            error=result.error or "",
            scheduler_type=scheduler,
            metrics_entries=metrics_entries,
            placement_map=placement_map,
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        import traceback
        return ExecutionResult(
            mode="subagent",
            success=False,
            total_time_ms=elapsed,
            tool_calls=0,
            error=f"{e}\n{traceback.format_exc()}",
        )


def print_result(result: ExecutionResult):
    """Print execution result"""
    print(f"\n--- {result.mode.upper()} Result ---")
    if result.scheduler_type:
        print(f"Scheduler: {result.scheduler_type}")
    print(f"Success: {result.success}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Metrics entries: {len(result.metrics_entries)}")

    if result.partitions > 0:
        print(f"Partitions: {result.partitions}")
        if result.partition_times:
            print(f"Partition times: {[f'{t:.0f}ms' for t in result.partition_times]}")

    if result.placement_map:
        print(f"\nPlacement Map:")
        for tool, location in result.placement_map.items():
            print(f"  {tool:25} -> {location}")

    if result.error:
        print(f"\nError: {result.error[:300]}")

    if result.final_result:
        print(f"\nResult preview:\n{result.final_result[:300]}...")


def save_result(result: ExecutionResult, output_dir: str = "results/scenario4_subagent"):
    """Save execution result to JSON file"""
    import time as time_module
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_dict = {
        "scenario_name": "image_processing",
        "mode": result.mode,
        "scheduler_type": result.scheduler_type,
        "success": result.success,
        "total_time_ms": result.total_time_ms,
        "tool_calls": result.tool_calls,
        "partitions": result.partitions,
        "partition_times": result.partition_times,
        "placement_map": result.placement_map,
        "metrics_entries": result.metrics_entries,
        "tool_call_count": len(result.metrics_entries),
        "error": result.error if result.error else None,
    }

    output_path = Path(output_dir) / f"image_processing_{result.mode}_{int(time_module.time())}.json"
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to: {output_path}")


def compare_results(legacy: ExecutionResult, subagent: ExecutionResult):
    """Compare legacy vs subagent results"""
    print("\n" + "=" * 70)
    print("COMPARISON: Legacy vs Sub-Agent")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Legacy':<20} {'Sub-Agent':<20}")
    print("-" * 65)
    print(f"{'Success':<25} {str(legacy.success):<20} {str(subagent.success):<20}")
    print(f"{'Total time (ms)':<25} {legacy.total_time_ms:<20.0f} {subagent.total_time_ms:<20.0f}")
    print(f"{'Tool calls':<25} {legacy.tool_calls:<20} {subagent.tool_calls:<20}")
    print(f"{'Partitions':<25} {'1 (all)':<20} {subagent.partitions:<20}")

    if legacy.success and subagent.success:
        speedup = legacy.total_time_ms / subagent.total_time_ms if subagent.total_time_ms > 0 else 0
        print(f"\n{'Speedup':<25} {speedup:.2f}x")

        if speedup > 1:
            print(f"Sub-Agent mode is {speedup:.1f}x faster")
        else:
            print(f"Legacy mode is {1/speedup:.1f}x faster")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Scenario 4 with different orchestration modes"
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "subagent"],
        default="subagent",
        help="Execution mode (default: subagent)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and compare",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["brute_force", "static", "all_device", "all_edge", "all_cloud", "heuristic"],
        default="brute_force",
        help="Scheduler type (default: brute_force)",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        return False

    print("=" * 70)
    print("Scenario 4: Image Processing Pipeline")
    print("=" * 70)
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Images: {IMAGE_COUNT} at {IMAGE_PATH}")
    print(f"Model: {args.model}")
    print(f"Scheduler: {args.scheduler}")

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"

    if args.compare:
        # Run both modes
        legacy_result = await run_legacy_mode(config_path, args.model)
        print_result(legacy_result)
        save_result(legacy_result)

        # Re-prepare images for subagent mode
        prepare_images()

        subagent_result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_result(subagent_result)
        save_result(subagent_result)

        compare_results(legacy_result, subagent_result)

        return legacy_result.success and subagent_result.success

    elif args.mode == "legacy":
        result = await run_legacy_mode(config_path, args.model)
        print_result(result)
        save_result(result)
        return result.success

    else:  # subagent
        result = await run_subagent_mode(config_path, args.model, args.scheduler)
        print_result(result)
        save_result(result)
        return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
