#!/usr/bin/env python3
"""
DEVICE + EDGE + CLOUD 3-Tier E2E Test

SubAgentOrchestrator를 사용하여 3-Tier 아키텍처를 테스트합니다.

테스트 시나리오:
1. DEVICE only: filesystem (로컬 파일 읽기)
2. EDGE only: time (WASM)
3. CLOUD only: summarize (Container)
4. DEVICE -> EDGE: filesystem -> log_parser
5. DEVICE -> CLOUD: filesystem -> summarize
6. EDGE -> CLOUD: time -> summarize (결과를 텍스트로 요약)
7. DEVICE -> EDGE -> CLOUD: filesystem -> log_parser -> summarize

사용법:
    python scripts/test_3tier_e2e.py
    python scripts/test_3tier_e2e.py --scenario 1
    python scripts/test_3tier_e2e.py --scenario 7
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from edgeagent import (
    SubAgentOrchestrator,
    OrchestrationConfig,
    SubAgentEndpoint,
)


# SubAgent endpoints
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

CONFIG_PATH = Path(__file__).parent.parent / "config" / "tools_3tier.yaml"


@dataclass
class TestResult:
    """Test result"""
    scenario: str
    tiers: str
    success: bool
    execution_time_ms: float
    partitions: int
    tool_calls: int
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


async def test_device_only() -> TestResult:
    """
    Scenario 1: DEVICE only
    Tool: filesystem (로컬 파일 읽기)
    """
    print("\n" + "=" * 70)
    print("Test 1: DEVICE only (filesystem)")
    print("=" * 70)

    # 테스트 파일 생성
    test_dir = Path("/tmp/edgeagent_device")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_device.txt"
    test_file.write_text("Hello from DEVICE tier!\nThis is a test file.")

    orchestrator = create_orchestrator(use_edge=False, use_cloud=False)

    user_request = f"Read the file at {test_file}"
    tool_sequence = ["filesystem"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="1_device_only",
        tiers="DEVICE",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_edge_only() -> TestResult:
    """
    Scenario 2: EDGE only
    Tool: time (WASM)
    """
    print("\n" + "=" * 70)
    print("Test 2: EDGE only (time)")
    print("=" * 70)

    orchestrator = create_orchestrator(use_edge=True, use_cloud=False)

    user_request = "Get the current time in Asia/Seoul timezone"
    tool_sequence = ["time"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="2_edge_only",
        tiers="EDGE",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_cloud_only() -> TestResult:
    """
    Scenario 3: CLOUD only
    Tool: summarize (Container)
    """
    print("\n" + "=" * 70)
    print("Test 3: CLOUD only (summarize)")
    print("=" * 70)

    orchestrator = create_orchestrator(use_edge=False, use_cloud=True)

    user_request = """Summarize the following text:

    EdgeAgent is a locality-aware serverless execution framework for MCP tools.
    It dynamically places tools across Device, Edge, and Cloud tiers based on
    data affinity and compute requirements. WASM-based tools run on Edge with
    fast cold starts, while container-based tools run on Cloud for full functionality.
    """
    tool_sequence = ["summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="3_cloud_only",
        tiers="CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_device_to_edge() -> TestResult:
    """
    Scenario 4: DEVICE -> EDGE
    Tools: filesystem -> log_parser
    """
    print("\n" + "=" * 70)
    print("Test 4: DEVICE -> EDGE (filesystem -> log_parser)")
    print("=" * 70)

    # 테스트 로그 파일 생성
    test_dir = Path("/tmp/edgeagent_device")
    test_dir.mkdir(parents=True, exist_ok=True)
    log_file = test_dir / "server.log"
    log_file.write_text("""2025-01-01 10:00:00 INFO Application started
2025-01-01 10:00:01 WARNING High memory usage: 85%
2025-01-01 10:00:02 ERROR Database connection timeout
2025-01-01 10:00:03 INFO Request processed in 150ms
2025-01-01 10:00:04 ERROR Authentication failed for user admin
""")

    orchestrator = create_orchestrator(use_edge=True, use_cloud=False)

    user_request = f"Read the log file at {log_file} and parse it to extract structured log entries"
    tool_sequence = ["filesystem", "log_parser"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="4_device_to_edge",
        tiers="DEVICE -> EDGE",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_device_to_cloud() -> TestResult:
    """
    Scenario 5: DEVICE -> CLOUD
    Tools: filesystem -> summarize
    """
    print("\n" + "=" * 70)
    print("Test 5: DEVICE -> CLOUD (filesystem -> summarize)")
    print("=" * 70)

    # 테스트 문서 파일 생성
    test_dir = Path("/tmp/edgeagent_device")
    test_dir.mkdir(parents=True, exist_ok=True)
    doc_file = test_dir / "document.txt"
    doc_file.write_text("""EdgeAgent Research Summary

EdgeAgent is a novel framework for executing MCP (Model Context Protocol) tools
in an Edge-Cloud Continuum. The key innovation is locality-aware scheduling that
places tools based on their data affinity and compute requirements.

Key Features:
1. Dual-runtime support: WASM for Edge, Container for Cloud
2. Dynamic placement: Tools are scheduled to optimal locations
3. Serverless execution: Scale-to-zero with fast cold starts
4. Tool chaining: Results flow between tiers without LLM intervention

The framework achieves 60-70% latency reduction compared to LLM-centric approaches.
""")

    orchestrator = create_orchestrator(use_edge=False, use_cloud=True)

    user_request = f"Read the document at {doc_file} and summarize it briefly"
    tool_sequence = ["filesystem", "summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="5_device_to_cloud",
        tiers="DEVICE -> CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_edge_to_cloud() -> TestResult:
    """
    Scenario 6: EDGE -> CLOUD
    Tools: time -> summarize
    """
    print("\n" + "=" * 70)
    print("Test 6: EDGE -> CLOUD (time -> summarize)")
    print("=" * 70)

    orchestrator = create_orchestrator(use_edge=True, use_cloud=True)

    user_request = "Get the current time in multiple timezones (Asia/Seoul, America/New_York, Europe/London) and summarize the time differences"
    tool_sequence = ["time", "summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="6_edge_to_cloud",
        tiers="EDGE -> CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


async def test_device_edge_cloud() -> TestResult:
    """
    Scenario 7: DEVICE -> EDGE -> CLOUD (Full 3-Tier)
    Tools: filesystem -> log_parser -> summarize
    """
    print("\n" + "=" * 70)
    print("Test 7: DEVICE -> EDGE -> CLOUD (filesystem -> log_parser -> summarize)")
    print("=" * 70)

    # 테스트 로그 파일 생성
    test_dir = Path("/tmp/edgeagent_device")
    test_dir.mkdir(parents=True, exist_ok=True)
    log_file = test_dir / "app.log"
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

    user_request = f"""Analyze the application log file at {log_file}:
1. Read the log file
2. Parse the logs to extract structured entries
3. Summarize the issues found and recommend actions
"""
    tool_sequence = ["filesystem", "log_parser", "summarize"]

    start_time = time.time()
    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
        )
    execution_time = (time.time() - start_time) * 1000

    return TestResult(
        scenario="7_device_edge_cloud",
        tiers="DEVICE -> EDGE -> CLOUD",
        success=result.success,
        execution_time_ms=execution_time,
        partitions=result.partitions_executed or 0,
        tool_calls=result.total_tool_calls or 0,
        result=str(result.final_result)[:500] if result.final_result else None,
        error=result.error,
    )


def print_result(result: TestResult):
    """Print test result"""
    status = "PASS" if result.success else "FAIL"
    print(f"\n--- Result: {status} ---")
    print(f"Scenario: {result.scenario}")
    print(f"Tiers: {result.tiers}")
    print(f"Execution Time: {result.execution_time_ms:.0f}ms")
    print(f"Partitions: {result.partitions}")
    print(f"Tool Calls: {result.tool_calls}")
    if result.error:
        print(f"Error: {result.error}")
    if result.result:
        print(f"Result: {result.result[:300]}...")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="3-Tier E2E Test")
    parser.add_argument("--scenario", type=int, help="Run specific scenario (1-7)")
    args = parser.parse_args()

    scenarios = {
        1: ("DEVICE only", test_device_only),
        2: ("EDGE only", test_edge_only),
        3: ("CLOUD only", test_cloud_only),
        4: ("DEVICE -> EDGE", test_device_to_edge),
        5: ("DEVICE -> CLOUD", test_device_to_cloud),
        6: ("EDGE -> CLOUD", test_edge_to_cloud),
        7: ("DEVICE -> EDGE -> CLOUD", test_device_edge_cloud),
    }

    results = []

    if args.scenario:
        if args.scenario in scenarios:
            name, test_func = scenarios[args.scenario]
            print(f"\nRunning Scenario {args.scenario}: {name}")
            result = await test_func()
            print_result(result)
            results.append(result)
        else:
            print(f"Invalid scenario: {args.scenario}. Valid: 1-7")
            return
    else:
        # Run all scenarios
        print("\n" + "=" * 70)
        print("Running ALL 3-Tier E2E Tests")
        print("=" * 70)

        for scenario_num, (name, test_func) in scenarios.items():
            print(f"\n>>> Scenario {scenario_num}: {name}")
            try:
                result = await test_func()
                print_result(result)
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append(TestResult(
                    scenario=f"{scenario_num}_{name}",
                    tiers=name,
                    success=False,
                    execution_time_ms=0,
                    partitions=0,
                    tool_calls=0,
                    error=str(e),
                ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<30} {'Tiers':<25} {'Status':<8} {'Time':<10}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"{r.scenario:<30} {r.tiers:<25} {status:<8} {r.execution_time_ms:>6.0f}ms")

    passed = sum(1 for r in results if r.success)
    print("-" * 70)
    print(f"Total: {passed}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
