#!/usr/bin/env python3
"""
Scenario 4: Image Processing Pipeline - LLM Agent Version

This script uses an LLM Agent (ReAct pattern) that autonomously selects tools
to complete the image processing task. This demonstrates "true AI Agent" behavior
where the LLM decides the tool execution flow at runtime.

Comparison with run_scenario4_with_metrics.py:
- Script version: Hardcoded sequential tool calls (orchestration)
- Agent version: LLM autonomously selects tools (autonomous agent)

Tool Chain (expected, but LLM decides):
    scan_directory -> compute_image_hash -> compare_hashes -> batch_resize -> aggregate_list -> write_file
    EDGE             EDGE                  EDGE              EDGE           EDGE             DEVICE
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
from scripts.agent_utils import run_agent_with_logging


def load_image_source() -> tuple[Path, str]:
    """Load image source from COCO or sample images.

    Returns:
        Tuple of (image_dir_path, data_source_description)
    """
    data_dir = Path(__file__).parent.parent / "data" / "scenario4"
    coco_images = data_dir / "coco" / "images"
    sample_images = data_dir / "sample_images"

    if coco_images.exists() and len(list(coco_images.glob("*.jpg"))) > 0:
        return coco_images, "COCO 2017"

    if sample_images.exists():
        return sample_images, "Generated test images"

    raise FileNotFoundError(
        f"No image directory found.\n"
        "Run 'python scripts/download_public_datasets.py -s 4' for COCO 2017, or\n"
        "Run 'python scripts/generate_test_images.py' for test images"
    )


# System prompt for the Image Processing Agent
IMAGE_PROCESSING_SYSTEM_PROMPT = """You are an image processing assistant. Your task is to analyze a directory of images, find duplicates, create thumbnails, and generate a report.

You have access to the following tools:
- scan_directory: Scan a directory for images and get metadata
- compute_image_hash: Compute perceptual hash for an image
- compare_hashes: Compare hashes to find duplicate images
- batch_resize: Create thumbnails for multiple images
- aggregate_list: Group and aggregate data
- write_file: Write files to the filesystem

The images are located at /tmp/edgeagent_device/images

When processing images, follow this workflow:
1. Scan the image directory using scan_directory (recursive=False, include_info=True)
2. Compute perceptual hashes for each image using compute_image_hash (hash_type="phash")
3. Compare hashes to find duplicates using compare_hashes (threshold=5)
4. Create thumbnails for unique images using batch_resize (max_size=150, quality=75)
5. Aggregate statistics using aggregate_list (group_by="format")
6. Write a comprehensive report to /tmp/edgeagent_device/agent_image_report.md

Include in your report:
- Total images scanned
- Number of unique images vs duplicates
- Thumbnail generation results
- Image format distribution
"""


class AgentImageProcessingScenario(ScenarioRunner):
    """
    LLM Agent-based Image Processing Scenario.

    Unlike the script-based version, this uses an LLM to autonomously
    decide which tools to call and in what order.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results/scenario4_agent",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
    ):
        super().__init__(config_path, output_dir)
        self.model = model
        self.temperature = temperature
        self._image_source = None
        self._data_source = None

    @property
    def name(self) -> str:
        return "image_processing_agent"

    @property
    def description(self) -> str:
        return "LLM Agent autonomously processes images and generates a report"

    @property
    def user_request(self) -> str:
        return (
            "Process the images at /tmp/edgeagent_device/images. "
            "Scan the directory, compute perceptual hashes, find duplicate images, "
            "create thumbnails for unique images, and generate a comprehensive "
            "image processing report to /tmp/edgeagent_device/agent_image_report.md"
        )

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute image processing using LLM Agent"""

        # Filter out problematic tools (directory_tree causes issues with gpt-4o-mini)
        excluded_tools = {"directory_tree"}
        tools = [t for t in tools if t.name not in excluded_tools]

        # Prepare image directory
        image_source, data_source = load_image_source()
        self._image_source = image_source
        self._data_source = data_source

        device_images = Path("/tmp/edgeagent_device/images")
        device_images.mkdir(parents=True, exist_ok=True)

        # Clear and copy images
        for f in device_images.glob("*"):
            f.unlink()
        for img in image_source.glob("*"):
            if img.is_file():
                shutil.copy(img, device_images / img.name)

        total_input_size = sum(f.stat().st_size for f in device_images.glob("*") if f.is_file())
        image_count = len(list(device_images.glob("*")))

        print("-" * 70)
        print("LLM Agent Image Processing")
        print("-" * 70)
        print(f"Data Source: {data_source}")
        print(f"Images: {image_count} ({total_input_size:,} bytes)")
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
            client.metrics_collector.add_custom_metric("data_source", data_source)
            client.metrics_collector.add_custom_metric("tool_call_counts", tool_counts)
            client.metrics_collector.add_custom_metric("total_tool_calls", len(client.metrics_collector.entries))
            client.metrics_collector.add_custom_metric("images_count", image_count)
            client.metrics_collector.add_custom_metric("total_input_size", total_input_size)

        # Check if report was written
        report_path = Path("/tmp/edgeagent_device/agent_image_report.md")
        if report_path.exists():
            report_content = report_path.read_text()
            print(f"Report written to: {report_path}")
            print(f"Report size: {len(report_content)} bytes")
            return report_content
        else:
            print("[WARN] Agent did not write a report file")
            return final_message


async def main():
    """Run the LLM Agent-based Image Processing scenario"""

    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        print("Please create a .env file with OPENAI_API_KEY=your-key")
        return False

    print("=" * 70)
    print("Scenario 4: Image Processing Pipeline (LLM Agent Version)")
    print("=" * 70)
    print()
    print("This version uses an LLM Agent that autonomously decides")
    print("which tools to call and in what order.")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools_scenario4.yaml"

    scenario = AgentImageProcessingScenario(
        config_path=config_path,
        output_dir="results/scenario4_agent",
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
        csv_path = result.metrics.save_csv("results/scenario4_agent/metrics.csv")
        print(f"\nMetrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
