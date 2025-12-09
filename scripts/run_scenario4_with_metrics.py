#!/usr/bin/env python3
"""
Scenario 4: Image Processing Pipeline - With Unified Metrics Collection

Tool Chain:
    filesystem(search) -> image_resize(scan) -> image_resize(hash) ->
    image_resize(compare) -> image_resize(batch) -> data_aggregate -> filesystem(write)
    DEVICE              EDGE                   EDGE
    EDGE                     EDGE                  EDGE             DEVICE

This scenario demonstrates:
- Image directory scanning and metadata extraction
- Duplicate detection using perceptual hashing
- Batch thumbnail generation
- Data reduction from raw images to summary report

Data Sources:
- Primary: COCO 2017 dataset (download via scripts/download_public_datasets.py -s 4)
- Fallback: Generated test images (scripts/generate_test_images.py)
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


class ImageProcessingScenario(ScenarioRunner):
    """Image Processing Pipeline Scenario"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_dir = "/tmp/edgeagent_device/images"
        self._report_path = "/tmp/edgeagent_device/image_report.md"
        # Ensure device directory exists
        Path("/tmp/edgeagent_device").mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "image_processing"

    @property
    def description(self) -> str:
        return "Process images: scan, detect duplicates, create thumbnails, generate report"

    @property
    def user_request(self) -> str:
        return "Scan images, find duplicates, create thumbnails, and generate a summary report"

    def get_validation_context(self) -> dict:
        """Provide context for validation"""
        return {
            "report_path": self._report_path,
            "input_dir": self._input_dir,
        }

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute the image processing pipeline"""

        # Find tools by name
        tool_by_name = {t.name: t for t in tools}

        print("Available tools:")
        for name in sorted(tool_by_name.keys()):
            print(f"  - {name}")
        print()

        # Prepare image directory - try COCO first, then sample_images
        data_dir = Path(__file__).parent.parent / "data" / "scenario4"
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
                f"No image directory found.\n"
                "Run 'python scripts/download_public_datasets.py -s 4' for COCO 2017, or\n"
                "Run 'python scripts/generate_test_images.py' for test images"
            )

        print(f"Data Source: {data_source}")

        device_images = Path(self._input_dir)
        device_images.mkdir(parents=True, exist_ok=True)

        # Copy images to device directory
        import shutil
        for img in image_source.glob("*"):
            if img.is_file():
                shutil.copy(img, device_images / img.name)

        total_input_size = sum(f.stat().st_size for f in device_images.glob("*") if f.is_file())
        print(f"Prepared {len(list(device_images.glob('*')))} images ({total_input_size:,} bytes)")
        print()

        # Step 1: Scan directory (image_resize -> EDGE)
        print("-" * 70)
        print("Step 1: Scan image directory (image_resize -> EDGE)")
        print("-" * 70)

        scan_tool = tool_by_name.get("scan_directory")
        if scan_tool:
            raw_scan = await scan_tool.ainvoke({
                "directory": str(device_images),
                "recursive": False,
                "include_info": True,
            })
            scan_result = parse_tool_result(raw_scan)
        else:
            from servers.image_resize_server import scan_directory
            scan_result = scan_directory.fn(str(device_images), recursive=False, include_info=True)

        image_count = scan_result.get("image_count", 0)
        image_paths = scan_result.get("image_paths", [])
        print(f"  Found {image_count} images")
        print(f"  Total size: {scan_result.get('total_size_mb', 0)} MB")

        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("images_found", image_count)
        print()

        # Step 2: Compute hashes for duplicate detection (image_resize -> EDGE)
        print("-" * 70)
        print("Step 2: Compute perceptual hashes (image_resize -> EDGE)")
        print("-" * 70)

        hash_tool = tool_by_name.get("compute_image_hash")
        hashes = []

        if hash_tool:
            for img_path in image_paths:
                raw_hash = await hash_tool.ainvoke({
                    "image_path": img_path,
                    "hash_type": "phash",
                })
                hashes.append(parse_tool_result(raw_hash))
        else:
            from servers.image_resize_server import compute_image_hash
            for img_path in image_paths:
                hashes.append(compute_image_hash.fn(img_path, hash_type="phash"))

        valid_hashes = len([h for h in hashes if "hash" in h])
        print(f"  Computed {valid_hashes} hashes")
        print()

        # Step 3: Find duplicates (image_resize -> EDGE)
        print("-" * 70)
        print("Step 3: Find duplicate images (image_resize -> EDGE)")
        print("-" * 70)

        compare_tool = tool_by_name.get("compare_hashes")
        if compare_tool:
            raw_compare = await compare_tool.ainvoke({
                "hashes": hashes,
                "threshold": 5,
            })
            compare_result = parse_tool_result(raw_compare)
        else:
            from servers.image_resize_server import compare_hashes
            compare_result = compare_hashes.fn(hashes, threshold=5)

        duplicate_groups = compare_result.get("duplicate_groups", [])
        unique_count = compare_result.get("unique_count", 0)
        print(f"  Duplicate groups: {len(duplicate_groups)}")
        print(f"  Unique images: {unique_count}")

        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("duplicate_groups", len(duplicate_groups))
            client.metrics_collector.add_custom_metric("unique_images", unique_count)
        print()

        # Step 4: Create thumbnails (image_resize -> EDGE)
        print("-" * 70)
        print("Step 4: Create thumbnails (image_resize -> EDGE)")
        print("-" * 70)

        # Only process unique images (skip duplicates)
        unique_paths = compare_result.get("unique_paths", image_paths[:unique_count])

        batch_tool = tool_by_name.get("batch_resize")
        if batch_tool:
            raw_batch = await batch_tool.ainvoke({
                "image_paths": unique_paths,
                "max_size": 150,
                "quality": 75,
                "output_format": "JPEG",
            })
            batch_result = parse_tool_result(raw_batch)
        else:
            from servers.image_resize_server import batch_resize
            batch_result = batch_resize.fn(unique_paths, max_size=150, quality=75)

        thumbnails_created = batch_result.get("successful", 0)
        thumbnail_reduction = batch_result.get("overall_reduction", 0)
        print(f"  Thumbnails created: {thumbnails_created}")
        print(f"  Size reduction: {thumbnail_reduction:.2%}")

        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("thumbnails_created", thumbnails_created)
            client.metrics_collector.add_custom_metric("thumbnail_reduction_ratio", thumbnail_reduction)
        print()

        # Step 5: Aggregate statistics (data_aggregate -> EDGE)
        print("-" * 70)
        print("Step 5: Aggregate image statistics (data_aggregate -> EDGE)")
        print("-" * 70)

        # Prepare image info for aggregation
        images_info = scan_result.get("images", [])
        if not images_info:
            # Fallback: create basic info from paths
            images_info = [{"path": p, "format": Path(p).suffix.upper()[1:]} for p in image_paths]

        aggregate_tool = tool_by_name.get("aggregate_list")
        if aggregate_tool:
            raw_agg = await aggregate_tool.ainvoke({
                "items": images_info,
                "group_by": "format",
            })
            agg_result = parse_tool_result(raw_agg)
        else:
            from servers.data_aggregate_server import aggregate_list
            agg_result = aggregate_list.fn(images_info, group_by="format")

        print(f"  Format groups: {list(agg_result.get('groups', {}).keys())}")
        print()

        # Step 6: Write report (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 6: Write report (filesystem -> DEVICE)")
        print("-" * 70)

        # Generate report
        report = f"""# Image Processing Report

## Summary
- Images scanned: {image_count}
- Total input size: {total_input_size:,} bytes ({total_input_size / 1024 / 1024:.2f} MB)
- Unique images: {unique_count}
- Duplicate groups: {len(duplicate_groups)}

## Duplicate Groups
"""
        for i, group in enumerate(duplicate_groups, 1):
            report += f"\n### Group {i} ({len(group)} images)\n"
            for path in group:
                report += f"- {Path(path).name}\n"

        report += f"""
## Thumbnail Generation
- Thumbnails created: {thumbnails_created}
- Original size: {batch_result.get('total_input_bytes', 0):,} bytes
- Thumbnail size: {batch_result.get('total_output_bytes', 0):,} bytes
- Reduction ratio: {thumbnail_reduction:.2%}

## Images by Format
"""
        for fmt, count in agg_result.get("groups", {}).items():
            report += f"- {fmt}: {count} images\n"

        report += f"""
## Processing Metrics
- Hash algorithm: perceptual hash (phash)
- Duplicate threshold: 5 (lower = more strict)
- Thumbnail size: 150px max dimension
- Thumbnail quality: 75%
"""

        write_tool = tool_by_name.get("write_file")
        output_path = "/tmp/edgeagent_device/image_report.md"
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

        # Data reduction stats
        output_size = len(report)
        reduction = (1 - output_size / total_input_size) * 100 if total_input_size > 0 else 0

        print("=" * 70)
        print("Data Flow Summary")
        print("=" * 70)
        print(f"  Input size:  {total_input_size:,} bytes")
        print(f"  Output size: {output_size:,} bytes")
        print(f"  Reduction:   {reduction:.1f}%")
        print()

        return report


async def main():
    """Run the Image Processing scenario with metrics collection"""
    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"

    scenario = ImageProcessingScenario(
        config_path=config_path,
        output_dir="results/scenario4",
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional: Export metrics to CSV for pandas analysis
    if result.metrics:
        csv_path = result.metrics.save_csv("results/scenario4/metrics.csv")
        print(f"Metrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
