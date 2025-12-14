#!/usr/bin/env python3
"""
CLOUD + EDGE E2E Scenario Test

Tests scenarios using both CLOUD and EDGE SubAgents via Knative.

Usage:
    python scripts/test_cloud_edge_e2e.py
    python scripts/test_cloud_edge_e2e.py --scenario 2
    python scripts/test_cloud_edge_e2e.py --location cloud
    python scripts/test_cloud_edge_e2e.py --location edge
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

# SubAgent endpoints
SUBAGENT_ENDPOINTS = {
    "CLOUD": "http://cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud",
    "EDGE": "http://edge-subagent.edgeagent.edge.edgeagent.ddps.cloud",
}

# Data paths (EFS for CLOUD, NFS for EDGE)
DATA_PATHS = {
    "CLOUD": "/edgeagent/data",
    "EDGE": "/edgeagent/data",
}


@dataclass
class ScenarioResult:
    """Result of a scenario test"""
    scenario: str
    location: str
    success: bool
    execution_time_ms: float
    tool_calls: int
    result: Optional[str] = None
    error: Optional[str] = None


async def call_subagent(
    location: str,
    task: str,
    tools: list[str],
    timeout: float = 300.0,
) -> dict:
    """Call a SubAgent and return the response"""
    endpoint = SUBAGENT_ENDPOINTS[location]

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{endpoint}/execute",
            json={"task": task, "tools": tools},
        )
        return response.json()


async def test_s2_log_analysis(location: str) -> ScenarioResult:
    """
    Scenario 2: Log Analysis

    Tools: filesystem -> log_parser -> summarize -> filesystem
    """
    data_path = DATA_PATHS[location]
    log_file = f"{data_path}/scenario2/server.log"
    report_file = f"{data_path}/scenario2/log_report_{location.lower()}.md"

    task = f"""
Analyze the server log file at {log_file}.

1. Read the log file using read_file
2. Parse the logs using parse_logs with format_type='python'
3. Summarize the key findings using summarize_text
4. Write a summary report to {report_file}

Return the analysis summary.
"""

    tools = ["filesystem", "log_parser", "summarize"]

    start_time = time.time()
    try:
        result = await call_subagent(location, task, tools)
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="S2_log_analysis",
            location=location,
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:500] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="S2_log_analysis",
            location=location,
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


async def test_s3_research(location: str) -> ScenarioResult:
    """
    Scenario 3: Research Assistant

    Tools: fetch -> summarize -> data_aggregate
    """
    data_path = DATA_PATHS[location]

    task = """
Research AI agents by:

1. Fetch content from https://en.wikipedia.org/wiki/Intelligent_agent
2. Summarize the fetched content using summarize_text (max_length=200)
3. Return the summary

Keep the response concise.
"""

    tools = ["fetch", "summarize"]

    start_time = time.time()
    try:
        result = await call_subagent(location, task, tools, timeout=180.0)
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="S3_research",
            location=location,
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:500] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="S3_research",
            location=location,
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


async def test_s4_image_processing(location: str) -> ScenarioResult:
    """
    Scenario 4: Image Processing

    Tools: filesystem -> image_resize
    """
    data_path = DATA_PATHS[location]
    image_dir = f"{data_path}/scenario4/sample_images"

    task = f"""
Process images at {image_dir}:

1. List the directory to find image files using list_directory
2. Find duplicate images using find_duplicates with threshold=10
3. Return the list of duplicate pairs found

Keep the response concise.
"""

    tools = ["filesystem", "image_resize"]

    start_time = time.time()
    try:
        result = await call_subagent(location, task, tools, timeout=180.0)
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="S4_image_processing",
            location=location,
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:500] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="S4_image_processing",
            location=location,
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


async def test_time_simple(location: str) -> ScenarioResult:
    """Simple time test to verify SubAgent connectivity"""
    task = "What time is it now in Seoul? Use the time tool."
    tools = ["time"]

    start_time = time.time()
    try:
        result = await call_subagent(location, task, tools, timeout=60.0)
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="time_simple",
            location=location,
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:200] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="time_simple",
            location=location,
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


def print_result(result: ScenarioResult):
    """Print a single result"""
    status = "✅" if result.success else "❌"
    print(f"\n{status} {result.scenario} @ {result.location}")
    print(f"   Time: {result.execution_time_ms:.0f}ms, Tool calls: {result.tool_calls}")
    if result.error:
        print(f"   Error: {result.error[:100]}")
    elif result.result:
        print(f"   Result: {result.result[:150]}...")


def print_summary(results: list[ScenarioResult]):
    """Print summary of all results"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Group by location
    for location in ["CLOUD", "EDGE"]:
        loc_results = [r for r in results if r.location == location]
        if not loc_results:
            continue

        success_count = sum(1 for r in loc_results if r.success)
        total_time = sum(r.execution_time_ms for r in loc_results)

        print(f"\n{location}:")
        print(f"  Passed: {success_count}/{len(loc_results)}")
        print(f"  Total time: {total_time:.0f}ms")

        for r in loc_results:
            status = "✅" if r.success else "❌"
            print(f"    {status} {r.scenario}: {r.execution_time_ms:.0f}ms")

    # Overall
    total_success = sum(1 for r in results if r.success)
    print(f"\nOverall: {total_success}/{len(results)} passed")


async def run_all_scenarios(locations: list[str] = None, scenario: int = None):
    """Run all scenarios on specified locations"""
    if locations is None:
        locations = ["CLOUD", "EDGE"]

    results = []

    # Define scenario tests
    scenario_tests = {
        0: ("time_simple", test_time_simple),
        2: ("S2_log_analysis", test_s2_log_analysis),
        3: ("S3_research", test_s3_research),
        4: ("S4_image_processing", test_s4_image_processing),
    }

    # Filter scenarios if specified
    if scenario is not None:
        if scenario not in scenario_tests:
            print(f"Unknown scenario: {scenario}")
            return results
        scenario_tests = {scenario: scenario_tests[scenario]}

    print("=" * 70)
    print("CLOUD + EDGE E2E Scenario Tests")
    print("=" * 70)
    print(f"Locations: {', '.join(locations)}")
    print(f"Scenarios: {', '.join(name for name, _ in scenario_tests.values())}")

    for location in locations:
        print(f"\n{'='*70}")
        print(f"Testing {location} SubAgent")
        print("=" * 70)

        # Health check first
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{SUBAGENT_ENDPOINTS[location]}/health")
                health = resp.json()
                print(f"Health: {health.get('status', 'unknown')}")
                print(f"Available tools: {', '.join(health.get('available_tools', []))}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            continue

        # Run scenarios
        for scenario_num, (name, test_func) in scenario_tests.items():
            print(f"\n--- {name} ---")
            result = await test_func(location)
            results.append(result)
            print_result(result)

    print_summary(results)
    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLOUD + EDGE E2E Tests")
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[0, 2, 3, 4],
        help="Run specific scenario (0=time, 2=log, 3=research, 4=image)",
    )
    parser.add_argument(
        "--location", "-l",
        choices=["cloud", "edge", "both"],
        default="both",
        help="Run on specific location",
    )

    args = parser.parse_args()

    if args.location == "both":
        locations = ["CLOUD", "EDGE"]
    else:
        locations = [args.location.upper()]

    results = await run_all_scenarios(locations, args.scenario)

    # Return exit code based on results
    success = all(r.success for r in results) if results else False
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
