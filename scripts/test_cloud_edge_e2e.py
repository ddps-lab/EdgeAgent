#!/usr/bin/env python3
"""
DEVICE + EDGE + CLOUD E2E Scenario Test

Tests S1-S4 scenarios using SubAgentOrchestrator across all three tiers.

Scenarios:
- S1: Code Review (DEVICE: git)
- S2: Log Analysis (DEVICE: filesystem -> EDGE: log_parser -> CLOUD: summarize)
- S3: Research Assistant (EDGE: fetch -> CLOUD: summarize)
- S4: Image Processing (DEVICE: filesystem -> EDGE: image_resize)

Additional tests:
- Edge filesystem read test
- Direct SubAgent calls

Usage:
    # Run all scenarios with orchestrator (3-tier)
    python scripts/test_cloud_edge_e2e.py

    # Run specific scenario
    python scripts/test_cloud_edge_e2e.py --scenario 2

    # Run on specific location only (direct SubAgent call)
    python scripts/test_cloud_edge_e2e.py --direct --location edge

    # Test Edge filesystem access
    python scripts/test_cloud_edge_e2e.py --test-edge-fs
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from edgeagent import (
    SubAgentOrchestrator,
    OrchestrationConfig,
    SubAgentEndpoint,
)

# SubAgent endpoints
SUBAGENT_ENDPOINTS = {
    "CLOUD": "http://cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud",
    "EDGE": "http://edge-subagent.edgeagent.edge.edgeagent.ddps.cloud",
}

EDGE_SUBAGENT = SubAgentEndpoint(
    host="edge-subagent.edgeagent.edge.edgeagent.ddps.cloud",
    port=80,
    timeout=300.0,
)
CLOUD_SUBAGENT = SubAgentEndpoint(
    host="cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud",
    port=80,
    timeout=300.0,
)

# Data paths
DATA_PATHS = {
    "CLOUD": "/edgeagent/data",
    "EDGE": "/edgeagent/data",
    "DEVICE": "/tmp/edgeagent_device",
}

CONFIG_PATH = Path(__file__).parent.parent / "config" / "tools_3tier.yaml"


@dataclass
class ScenarioResult:
    """Result of a scenario test"""
    scenario: str
    location: str  # e.g., "DEVICE", "EDGE", "DEVICE->EDGE->CLOUD"
    success: bool
    execution_time_ms: float
    tool_calls: int
    partitions: int = 1
    result: Optional[str] = None
    error: Optional[str] = None


def create_orchestrator(use_edge: bool = True, use_cloud: bool = True) -> SubAgentOrchestrator:
    """Create orchestrator with specified tiers"""
    endpoints = {}
    if use_edge:
        endpoints["EDGE"] = EDGE_SUBAGENT
    if use_cloud:
        endpoints["CLOUD"] = CLOUD_SUBAGENT

    config = OrchestrationConfig(
        mode="subagent",
        subagent_endpoints=endpoints,
        model="gpt-4o-mini",
        temperature=0,
        max_iterations=10,
    )

    return SubAgentOrchestrator(CONFIG_PATH, config)


# =============================================================================
# S1: Code Review (DEVICE only - git operations)
# =============================================================================

async def test_s1_code_review() -> ScenarioResult:
    """
    S1: Code Review
    Tools: git (DEVICE only)

    Reviews recent commits in a git repository.
    """
    orchestrator = create_orchestrator(use_edge=False, use_cloud=False)

    # Use current edgeagent repo
    repo_path = Path(__file__).parent.parent

    user_request = f"""Review the git repository at {repo_path}:
1. Get the git status
2. Show the last 3 commit messages using git log
3. Summarize what's been recently changed
"""
    tool_sequence = ["git"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return ScenarioResult(
        scenario="S1_code_review",
        location="DEVICE",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


# =============================================================================
# S2: Log Analysis (DEVICE -> EDGE -> CLOUD)
# =============================================================================

async def test_s2_log_analysis() -> ScenarioResult:
    """
    S2: Log Analysis (Full 3-Tier)
    Tools: filesystem (DEVICE) -> log_parser (EDGE) -> summarize (CLOUD)

    Reads log file from DEVICE, parses on EDGE, summarizes on CLOUD.
    """
    # Create test log file
    test_dir = Path("/tmp/edgeagent_device")
    test_dir.mkdir(parents=True, exist_ok=True)
    log_file = test_dir / "server.log"
    log_file.write_text("""2025-01-01 10:00:00 INFO Application started successfully
2025-01-01 10:00:01 INFO Connected to database
2025-01-01 10:00:02 WARNING High memory usage detected: 85%
2025-01-01 10:00:03 ERROR Database query timeout after 30s
2025-01-01 10:00:04 INFO Retrying database connection
2025-01-01 10:00:05 ERROR Connection refused by database server
2025-01-01 10:00:06 WARNING Rate limit exceeded for API endpoint /api/users
2025-01-01 10:00:07 INFO Cache hit ratio: 95%
2025-01-01 10:00:08 ERROR Out of memory error in worker process
2025-01-01 10:00:09 INFO Graceful shutdown initiated
2025-01-01 10:00:10 INFO Application shutdown complete
""")

    orchestrator = create_orchestrator(use_edge=True, use_cloud=True)

    user_request = f"""Analyze the server log file at {log_file}:
1. Read the log file using filesystem
2. Parse the logs to extract structured entries using log_parser
3. Summarize the issues found and recommend actions using summarize
"""
    tool_sequence = ["filesystem", "log_parser", "summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return ScenarioResult(
        scenario="S2_log_analysis",
        location="DEVICE->EDGE->CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


# =============================================================================
# S3: Research Assistant (EDGE -> CLOUD)
# =============================================================================

async def test_s3_research() -> ScenarioResult:
    """
    S3: Research Assistant
    Tools: fetch (EDGE) -> summarize (CLOUD)

    Fetches web content on EDGE, summarizes on CLOUD.
    """
    orchestrator = create_orchestrator(use_edge=True, use_cloud=True)

    user_request = """Research a topic:
1. Fetch content from https://httpbin.org/html using fetch
2. Summarize the fetched content (max 200 words) using summarize
"""
    tool_sequence = ["fetch", "summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return ScenarioResult(
        scenario="S3_research",
        location="EDGE->CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


# =============================================================================
# S4: Image Processing (DEVICE -> EDGE)
# =============================================================================

async def test_s4_image_processing() -> ScenarioResult:
    """
    S4: Image Processing
    Tools: filesystem (DEVICE) -> image_resize (EDGE)

    Lists images from DEVICE, processes on EDGE.
    Note: For actual image processing, images need to be on shared storage.
    This test verifies the tool chain works.
    """
    # Create test directory with placeholder files
    test_dir = Path("/tmp/edgeagent_device/images")
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        img_file = test_dir / f"test_image_{i}.txt"
        img_file.write_text(f"Test image {i} placeholder - size: {100*(i+1)}x{100*(i+1)}")

    orchestrator = create_orchestrator(use_edge=True, use_cloud=False)

    user_request = f"""Process images at {test_dir}:
1. List the directory contents using filesystem to find all files
2. Report what files were found
"""
    tool_sequence = ["filesystem"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return ScenarioResult(
        scenario="S4_image_processing",
        location="DEVICE",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


# =============================================================================
# Additional: Edge Filesystem Test
# =============================================================================

async def test_edge_filesystem() -> ScenarioResult:
    """
    Test Edge filesystem access

    Tests if Edge SubAgent can read files from shared storage (/edgeagent/data).
    This verifies NFS mount is working correctly on Edge.
    """
    task = """Read the file at /edgeagent/data/scenario2/server.log using read_file.
If the file doesn't exist, list the contents of /edgeagent/data directory.
Report what you found."""
    tools = ["filesystem"]

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{SUBAGENT_ENDPOINTS['EDGE']}/execute",
                json={"task": task, "tools": tools},
            )
            result = response.json()
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="edge_filesystem_test",
            location="EDGE",
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:500] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="edge_filesystem_test",
            location="EDGE",
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


async def test_cloud_filesystem() -> ScenarioResult:
    """
    Test Cloud filesystem access

    Tests if Cloud SubAgent can read files from shared storage (/edgeagent/data).
    This verifies EFS mount is working correctly on Cloud.
    """
    task = """Read the file at /edgeagent/data/scenario2/server.log using read_file.
If the file doesn't exist, list the contents of /edgeagent/data directory.
Report what you found."""
    tools = ["filesystem"]

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{SUBAGENT_ENDPOINTS['CLOUD']}/execute",
                json={"task": task, "tools": tools},
            )
            result = response.json()
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="cloud_filesystem_test",
            location="CLOUD",
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:500] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="cloud_filesystem_test",
            location="CLOUD",
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


# =============================================================================
# Direct SubAgent Tests
# =============================================================================

async def test_time_direct(location: str) -> ScenarioResult:
    """Simple time test via direct SubAgent call"""
    task = "What time is it now in Seoul? Use the time tool."
    tools = ["time"]

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SUBAGENT_ENDPOINTS[location]}/execute",
                json={"task": task, "tools": tools},
            )
            result = response.json()
        execution_time = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario="time_direct",
            location=location,
            success=result.get("success", False),
            execution_time_ms=execution_time,
            tool_calls=len(result.get("tool_calls", [])),
            result=result.get("result", "")[:200] if result.get("result") else None,
            error=result.get("error"),
        )
    except Exception as e:
        return ScenarioResult(
            scenario="time_direct",
            location=location,
            success=False,
            execution_time_ms=(time.time() - start_time) * 1000,
            tool_calls=0,
            error=str(e),
        )


# =============================================================================
# Result Printing
# =============================================================================

def print_result(result: ScenarioResult):
    """Print a single result"""
    status = "PASS" if result.success else "FAIL"
    print(f"\n--- Result: {status} ---")
    print(f"Scenario: {result.scenario}")
    print(f"Tiers: {result.location}")
    print(f"Execution Time: {result.execution_time_ms:.0f}ms")
    print(f"Partitions: {result.partitions}")
    print(f"Tool Calls: {result.tool_calls}")
    if result.error:
        print(f"Error: {result.error[:300]}")
    if result.result:
        print(f"Result: {result.result[:300]}...")


def print_summary(results: list[ScenarioResult]):
    """Print summary of all results"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Tiers':<25} {'Status':<8} {'Time':<12} {'Calls':<6}")
    print("-" * 80)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"{r.scenario:<25} {r.location:<25} {status:<8} {r.execution_time_ms:>8.0f}ms {r.tool_calls:>4}")

    passed = sum(1 for r in results if r.success)
    total_time = sum(r.execution_time_ms for r in results)
    print("-" * 80)
    print(f"Total: {passed}/{len(results)} passed, {total_time:.0f}ms total")


# =============================================================================
# Main
# =============================================================================

async def run_all_scenarios(scenario: int = None) -> list[ScenarioResult]:
    """Run S1-S4 scenarios using SubAgentOrchestrator"""
    results = []

    scenarios = {
        1: ("S1_code_review", test_s1_code_review),
        2: ("S2_log_analysis", test_s2_log_analysis),
        3: ("S3_research", test_s3_research),
        4: ("S4_image_processing", test_s4_image_processing),
    }

    if scenario is not None:
        if scenario not in scenarios:
            print(f"Unknown scenario: {scenario}. Available: 1-4")
            return results
        scenarios = {scenario: scenarios[scenario]}

    print("\n" + "=" * 80)
    print("DEVICE + EDGE + CLOUD E2E Tests")
    print("=" * 80)
    print("Scenarios:")
    print("  S1: Code Review (DEVICE: git)")
    print("  S2: Log Analysis (DEVICE->EDGE->CLOUD: filesystem->log_parser->summarize)")
    print("  S3: Research (EDGE->CLOUD: fetch->summarize)")
    print("  S4: Image Processing (DEVICE: filesystem)")

    for scenario_num, (name, test_func) in scenarios.items():
        print(f"\n>>> Scenario {scenario_num}: {name}")
        print("=" * 60)
        try:
            result = await test_func()
            print_result(result)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(ScenarioResult(
                scenario=name,
                location="UNKNOWN",
                success=False,
                execution_time_ms=0,
                tool_calls=0,
                error=str(e),
            ))

    print_summary(results)
    return results


async def run_filesystem_tests() -> list[ScenarioResult]:
    """Test filesystem access on Edge and Cloud"""
    results = []

    print("\n" + "=" * 80)
    print("Filesystem Access Tests (Edge & Cloud)")
    print("=" * 80)

    print("\n>>> Edge Filesystem Test")
    result = await test_edge_filesystem()
    print_result(result)
    results.append(result)

    print("\n>>> Cloud Filesystem Test")
    result = await test_cloud_filesystem()
    print_result(result)
    results.append(result)

    print_summary(results)
    return results


async def run_direct_tests(locations: list[str]) -> list[ScenarioResult]:
    """Run direct SubAgent tests"""
    results = []

    print("\n" + "=" * 80)
    print("Direct SubAgent Tests")
    print("=" * 80)
    print(f"Locations: {', '.join(locations)}")

    for location in locations:
        print(f"\n>>> Testing {location} SubAgent")

        # Health check
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{SUBAGENT_ENDPOINTS[location]}/health")
                health = resp.json()
                print(f"Health: {health.get('status', 'unknown')}")
                tools = health.get('available_tools', [])
                print(f"Available tools ({len(tools)}): {', '.join(tools[:5])}...")
        except Exception as e:
            print(f"Health check failed: {e}")
            continue

        # Time test
        result = await test_time_direct(location)
        print_result(result)
        results.append(result)

    print_summary(results)
    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="DEVICE + EDGE + CLOUD E2E Tests")
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific scenario (1=code_review, 2=log_analysis, 3=research, 4=image)",
    )
    parser.add_argument(
        "--location", "-l",
        choices=["cloud", "edge", "both"],
        default=None,
        help="Run on specific location (direct mode only)",
    )
    parser.add_argument(
        "--direct", "-d",
        action="store_true",
        help="Use direct SubAgent calls instead of orchestrator",
    )
    parser.add_argument(
        "--test-edge-fs",
        action="store_true",
        help="Test Edge filesystem access",
    )
    parser.add_argument(
        "--test-fs",
        action="store_true",
        help="Test filesystem access on both Edge and Cloud",
    )

    args = parser.parse_args()

    if args.test_fs or args.test_edge_fs:
        results = await run_filesystem_tests()
    elif args.direct or args.location:
        if args.location == "both" or args.location is None:
            locations = ["EDGE", "CLOUD"]
        else:
            locations = [args.location.upper()]
        results = await run_direct_tests(locations)
    else:
        results = await run_all_scenarios(args.scenario)

    success = all(r.success for r in results) if results else False
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
