#!/usr/bin/env python3
"""
DEVICE + CLOUD Hybrid E2E Test (SubAgentOrchestrator 사용)

로컬 DEVICE에서 Orchestrator가 실행되며:
- DEVICE partition: 로컬 MCP 서버로 직접 실행
- CLOUD partition: Knative SubAgent HTTP 호출

테스트 시나리오:
1. S2 Log Analysis: DEVICE(filesystem) → CLOUD(summarize)
2. S3 Research: CLOUD(fetch) → CLOUD(summarize)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from edgeagent import (
    SubAgentOrchestrator,
    OrchestrationConfig,
    SubAgentEndpoint,
)


# Knative SubAgent URLs
CLOUD_SUBAGENT_URL = "http://cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud"


async def test_s2_log_analysis():
    """
    시나리오 S2: Log Analysis

    Flow:
    1. DEVICE: filesystem → read log file
    2. CLOUD: summarize → summarize errors

    Expected partitions:
    - Partition 0: DEVICE (filesystem)
    - Partition 1: CLOUD (summarize)
    """
    print("\n" + "=" * 70)
    print("Test S2: Log Analysis (DEVICE → CLOUD)")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "config" / "tools_hybrid.yaml"

    # Orchestration 설정 - CLOUD SubAgent 사용
    orch_config = OrchestrationConfig(
        mode="subagent",
        subagent_endpoints={
            "CLOUD": SubAgentEndpoint(
                host="cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud",
                port=80,
                timeout=300.0,
            ),
        },
        model="gpt-4o-mini",
        temperature=0,
        max_iterations=10,
    )

    orchestrator = SubAgentOrchestrator(config_path, orch_config)

    # 테스트용 로그 파일 생성
    log_dir = Path("/tmp/edgeagent_device_hy")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"
    log_file.write_text("""2025-01-01 10:00:00 INFO Application started
2025-01-01 10:00:01 WARNING High memory usage: 85%
2025-01-01 10:00:02 ERROR Database connection timeout
2025-01-01 10:00:03 INFO Request processed in 150ms
2025-01-01 10:00:04 ERROR Authentication failed for user admin
2025-01-01 10:00:05 WARNING Rate limit exceeded
2025-01-01 10:00:06 INFO Cache hit ratio: 95%
2025-01-01 10:00:07 ERROR Out of memory error
2025-01-01 10:00:08 INFO Graceful shutdown initiated
""")

    user_request = """
    Analyze the server log file at /tmp/edgeagent_device_hy/server.log.
    1. Read the log file
    2. Summarize the errors and warnings found

    Return a brief summary of the issues.
    """

    tool_sequence = ["filesystem", "summarize"]

    print(f"\nUser Request: {user_request.strip()}")
    print(f"Tool Sequence: {tool_sequence}")
    print()

    async with orchestrator:
        # 실행 계획 출력
        print("Execution Plan:")
        orchestrator.print_execution_plan(tool_sequence)

        # 실행
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
            mode="subagent",
        )

    print(f"\nSuccess: {result.success}")
    print(f"Partitions Executed: {result.partitions_executed}")
    print(f"Total Tool Calls: {result.total_tool_calls}")
    print(f"Execution Time: {result.execution_time_ms:.2f} ms")

    if result.error:
        print(f"\nError: {result.error}")

    if result.partition_results:
        print("\nPartition Results:")
        for pr in result.partition_results:
            loc = pr.get("location", "?")
            tools = pr.get("tools", [])
            success = pr.get("success", False)
            calls = len(pr.get("tool_calls", []))
            print(f"  [{pr['partition_index']}] {loc}: {tools} - success={success}, calls={calls}")

    if result.final_result:
        print(f"\nFinal Result (preview):")
        print(str(result.final_result)[:500])

    # 검증
    if result.success and result.partitions_executed >= 2:
        # DEVICE와 CLOUD 모두 실행되었는지 확인
        locations = [pr.get("location") for pr in result.partition_results]
        if "DEVICE" in locations and "CLOUD" in locations:
            print("\n[PASS] DEVICE + CLOUD hybrid execution successful")
            return True
        else:
            print(f"\n[FAIL] Expected DEVICE+CLOUD, got {locations}")
            return False
    else:
        print("\n[FAIL] Execution failed or incomplete")
        return False


async def test_cloud_only():
    """
    시나리오: CLOUD Only (time tool)

    Flow:
    1. CLOUD: time → get current time
    """
    print("\n" + "=" * 70)
    print("Test: CLOUD Only (Time)")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "config" / "tools_hybrid.yaml"

    orch_config = OrchestrationConfig(
        mode="subagent",
        subagent_endpoints={
            "CLOUD": SubAgentEndpoint(
                host="cloud-subagent.edgeagent.cloud.edgeagent.ddps.cloud",
                port=80,
                timeout=300.0,
            ),
        },
        model="gpt-4o-mini",
        temperature=0,
    )

    orchestrator = SubAgentOrchestrator(config_path, orch_config)

    user_request = "What is the current time in Seoul, Korea?"
    tool_sequence = ["time"]

    print(f"\nUser Request: {user_request}")
    print(f"Tool Sequence: {tool_sequence}")
    print()

    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
            mode="subagent",
        )

    print(f"\nSuccess: {result.success}")
    print(f"Execution Time: {result.execution_time_ms:.2f} ms")

    if result.final_result:
        print(f"Result: {result.final_result}")

    if result.error:
        print(f"Error: {result.error}")

    if result.success:
        print("\n[PASS] CLOUD time tool executed successfully")
        return True
    else:
        print("\n[FAIL] CLOUD execution failed")
        return False


async def test_device_only():
    """
    시나리오: DEVICE Only (filesystem)

    Flow:
    1. DEVICE: filesystem → list directory
    """
    print("\n" + "=" * 70)
    print("Test: DEVICE Only (Filesystem)")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "config" / "tools_hybrid.yaml"

    # DEVICE는 로컬 실행이므로 endpoint 필요 없음
    orch_config = OrchestrationConfig(
        mode="subagent",
        subagent_endpoints={},  # 로컬 실행
        model="gpt-4o-mini",
        temperature=0,
    )

    orchestrator = SubAgentOrchestrator(config_path, orch_config)

    user_request = "List the files in /tmp/edgeagent_device_hy/ directory"
    tool_sequence = ["filesystem"]

    print(f"\nUser Request: {user_request}")
    print(f"Tool Sequence: {tool_sequence}")
    print()

    async with orchestrator:
        result = await orchestrator.run(
            user_request=user_request,
            tool_sequence=tool_sequence,
            mode="subagent",
        )

    print(f"\nSuccess: {result.success}")
    print(f"Execution Time: {result.execution_time_ms:.2f} ms")

    if result.final_result:
        print(f"Result: {str(result.final_result)[:300]}")

    if result.error:
        print(f"Error: {result.error}")

    if result.success:
        print("\n[PASS] DEVICE filesystem executed successfully")
        return True
    else:
        print("\n[FAIL] DEVICE execution failed")
        return False


async def main():
    print("=" * 70)
    print("DEVICE + CLOUD Hybrid E2E Test (SubAgentOrchestrator)")
    print("=" * 70)
    print()
    print("This test validates EdgeAgent hybrid execution:")
    print("  - DEVICE: Local MCP servers (filesystem, git)")
    print("  - CLOUD: Knative SubAgent (summarize, time, fetch)")
    print()
    print(f"CLOUD SubAgent URL: {CLOUD_SUBAGENT_URL}")
    print()

    results = []

    # Test 1: DEVICE only
    try:
        results.append(("DEVICE Only (filesystem)", await test_device_only()))
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DEVICE Only (filesystem)", False))

    # Test 2: CLOUD only
    try:
        results.append(("CLOUD Only (time)", await test_cloud_only()))
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CLOUD Only (time)", False))

    # Test 3: DEVICE + CLOUD hybrid
    try:
        results.append(("DEVICE → CLOUD (S2 Log Analysis)", await test_s2_log_analysis()))
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DEVICE → CLOUD (S2 Log Analysis)", False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1

    print()
    print(f"Total: {len(results)} tests, Passed: {passed}, Failed: {len(results) - passed}")

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
