"""
Scheduler Constraint 검증 테스트

다양한 tool profile constraint에 따른 routing 검증:

1. requires_cloud_api=true (slack)
   → 무조건 CLOUD로 routing

2. privacy_sensitive=true (credentials)
   → CLOUD 제외, DEVICE/EDGE로만 routing

3. requires_gpu=true (compute)
   → GPU 있는 location (EDGE/CLOUD)으로 routing

4. filesystem (일반)
   → args.path 기반 동적 routing
"""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient


async def main():
    print("=" * 80)
    print("Scheduler Constraint 검증 테스트")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 테스트 환경 준비
    # ========================================================================
    print("[1] 테스트 환경 준비...")
    print()

    # filesystem 테스트용 디렉토리 생성
    device_dir = Path("/tmp/edgeagent_device_hy")
    edge_dir = Path("/tmp/edgeagent_edge_hy")
    cloud_dir = Path("/tmp/edgeagent_cloud_hy")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)
        (dir_path / "test.txt").write_text(f"Content from {dir_path.name}")

    print("   Filesystem directories prepared")
    print()

    # ========================================================================
    # 2. Middleware 초기화
    # ========================================================================
    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    async with EdgeAgentMCPClient(config_path) as client:
        print("[2] EdgeAgent Middleware 초기화...")
        print(f"   Config: {config_path.name}")
        print(f"   Location Clients: {list(client.clients.keys())}")
        print()

        tools = await client.get_tools()

        # Tool 목록 출력
        print("[3] 로드된 Tools...")
        print()

        tool_by_name = {}
        for tool in tools:
            tool_by_name[tool.name] = tool
            parent = getattr(tool, "parent_tool_name", "unknown")
            backends = list(tool.backend_tools.keys()) if hasattr(tool, "backend_tools") else []
            print(f"   {tool.name} ({parent})")
            print(f"     Backends: {backends}")

        print()

        # Placement Summary 출력
        client.print_placement_summary()

        # ================================================================
        # 4. Constraint 테스트
        # ================================================================
        print("\n[4] Constraint 테스트...")
        print()

        results = []

        # ------------------------------------------------------------
        # Test 1: requires_cloud_api=true (slack)
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 1: requires_cloud_api=true (slack)")
        print("  Expected: 무조건 CLOUD로 routing")
        print()

        if "send_message" in tool_by_name:
            tool = tool_by_name["send_message"]
            try:
                result = await tool.ainvoke({
                    "channel": "general",
                    "message": "Hello from EdgeAgent!"
                })

                trace = client.execution_trace[-1]
                actual = trace["location"]

                if actual == "CLOUD":
                    print(f"  [PASS] Routed to {actual}")
                    print(f"  Result: {result}")
                    results.append(("slack/send_message", "PASS"))
                else:
                    print(f"  [FAIL] Expected CLOUD, got {actual}")
                    results.append(("slack/send_message", "FAIL"))

            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("slack/send_message", "ERROR"))
        else:
            print("  [SKIP] send_message tool not found")
            results.append(("slack/send_message", "SKIP"))

        print()

        # ------------------------------------------------------------
        # Test 2: privacy_sensitive=true (credentials)
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 2: privacy_sensitive=true (credentials)")
        print("  Expected: CLOUD 제외, DEVICE 또는 EDGE로 routing")
        print()

        if "get_secret" in tool_by_name:
            tool = tool_by_name["get_secret"]
            try:
                result = await tool.ainvoke({"key": "api_key"})

                trace = client.execution_trace[-1]
                actual = trace["location"]

                if actual in ["DEVICE", "EDGE"]:
                    print(f"  [PASS] Routed to {actual} (CLOUD excluded)")
                    print(f"  Result: {result}")
                    results.append(("credentials/get_secret", "PASS"))
                else:
                    print(f"  [FAIL] Should not route to {actual}")
                    results.append(("credentials/get_secret", "FAIL"))

            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("credentials/get_secret", "ERROR"))
        else:
            print("  [SKIP] get_secret tool not found")
            results.append(("credentials/get_secret", "SKIP"))

        print()

        # ------------------------------------------------------------
        # Test 3: credentials with EDGE key hint
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 3: privacy_sensitive + EDGE key hint")
        print("  Expected: key에 'edge/' prefix → EDGE로 routing")
        print()

        if "get_secret" in tool_by_name:
            tool = tool_by_name["get_secret"]
            try:
                # key에 edge/ prefix → EDGE로 routing
                result = await tool.ainvoke({
                    "key": "edge/server_token"
                })

                trace = client.execution_trace[-1]
                actual = trace["location"]

                if actual == "EDGE":
                    print(f"  [PASS] Routed to {actual} via key hint")
                    print(f"  Result: {result}")
                    results.append(("credentials/edge_key", "PASS"))
                else:
                    print(f"  [FAIL] Expected EDGE, got {actual}")
                    results.append(("credentials/edge_key", "FAIL"))

            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("credentials/edge_key", "ERROR"))
        else:
            print("  [SKIP] get_secret tool not found")
            results.append(("credentials/edge_key", "SKIP"))

        print()

        # ------------------------------------------------------------
        # Test 4: credentials with CLOUD key hint (should ignore)
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 4: privacy_sensitive + CLOUD key hint")
        print("  Expected: CLOUD hint 무시, DEVICE/EDGE로 routing")
        print()

        if "store_secret" in tool_by_name:
            tool = tool_by_name["store_secret"]
            try:
                # key에 cloud/ prefix가 있어도 privacy_sensitive이므로 CLOUD 선택 안함
                result = await tool.ainvoke({
                    "key": "cloud/should_not_go_there",
                    "value": "sensitive_data"
                })

                trace = client.execution_trace[-1]
                actual = trace["location"]

                if actual in ["DEVICE", "EDGE"]:
                    print(f"  [PASS] CLOUD hint ignored, routed to {actual}")
                    print(f"  Result: {result}")
                    results.append(("credentials/cloud_blocked", "PASS"))
                else:
                    print(f"  [FAIL] Should not route to {actual}")
                    results.append(("credentials/cloud_blocked", "FAIL"))

            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("credentials/cloud_blocked", "ERROR"))
        else:
            print("  [SKIP] store_secret tool not found")
            results.append(("credentials/cloud_blocked", "SKIP"))

        print()

        # ------------------------------------------------------------
        # Test 5: requires_gpu=true (compute)
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 5: requires_gpu=true (compute)")
        print("  Expected: EDGE 또는 CLOUD (GPU 지원 location)")
        print()

        if "matrix_multiply" in tool_by_name:
            tool = tool_by_name["matrix_multiply"]
            try:
                result = await tool.ainvoke({"size": 1000})

                trace = client.execution_trace[-1]
                actual = trace["location"]

                if actual in ["EDGE", "CLOUD"]:
                    print(f"  [PASS] Routed to {actual} (GPU available)")
                    print(f"  Result: {result}")
                    results.append(("compute/matrix_multiply", "PASS"))
                else:
                    print(f"  [FAIL] DEVICE has no GPU, should not route there")
                    results.append(("compute/matrix_multiply", "FAIL"))

            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("compute/matrix_multiply", "ERROR"))
        else:
            print("  [SKIP] matrix_multiply tool not found")
            results.append(("compute/matrix_multiply", "SKIP"))

        print()

        # ------------------------------------------------------------
        # Test 6: filesystem args 기반 routing
        # ------------------------------------------------------------
        print("-" * 60)
        print("Test 6: filesystem args 기반 routing")
        print("  Expected: path에 따라 DEVICE/EDGE/CLOUD 동적 선택")
        print()

        if "read_file" in tool_by_name:
            tool = tool_by_name["read_file"]

            for location, path in [
                ("DEVICE", str(device_dir / "test.txt")),
                ("EDGE", str(edge_dir / "test.txt")),
                ("CLOUD", str(cloud_dir / "test.txt")),
            ]:
                try:
                    result = await tool.ainvoke({"path": path})

                    trace = client.execution_trace[-1]
                    actual = trace["location"]

                    if actual == location:
                        print(f"  [PASS] {path} → {actual}")
                        results.append((f"filesystem/{location}", "PASS"))
                    else:
                        print(f"  [FAIL] {path} → {actual} (expected {location})")
                        results.append((f"filesystem/{location}", "FAIL"))

                except Exception as e:
                    print(f"  [ERROR] {path}: {e}")
                    results.append((f"filesystem/{location}", "ERROR"))
        else:
            print("  [SKIP] read_file tool not found")

        print()

        # ================================================================
        # 5. Execution Trace
        # ================================================================
        print("=" * 80)
        print("[5] Execution Trace")
        print("=" * 80)
        print()

        for i, trace in enumerate(client.execution_trace, 1):
            parent = trace.get("parent_tool", "?")
            print(f"   {i}. {trace['tool']} ({parent}) → {trace['location']}")

        print()

        # ================================================================
        # 6. 결과 요약
        # ================================================================
        print("=" * 80)
        print("[6] 결과 요약")
        print("=" * 80)
        print()

        passed = sum(1 for _, status in results if status == "PASS")
        failed = sum(1 for _, status in results if status == "FAIL")
        errors = sum(1 for _, status in results if status == "ERROR")
        skipped = sum(1 for _, status in results if status == "SKIP")

        for name, status in results:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "ERROR": "[ERR]", "SKIP": "[SKIP]"}
            print(f"   {icon.get(status, '[?]')} {name}")

        print()
        print(f"   Total: {len(results)} tests")
        print(f"   Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
        print()

        success = failed == 0 and errors == 0

        if success:
            print("=" * 80)
            print("[PASS] Scheduler Constraint 검증 성공!")
            print("=" * 80)
            print()
            print("검증된 Constraint:")
            print("  1. requires_cloud_api=true → 무조건 CLOUD")
            print("  2. privacy_sensitive=true → CLOUD 제외")
            print("  3. requires_gpu=true → GPU location (EDGE/CLOUD)")
            print("  4. args.path 기반 동적 routing")
            print()
        else:
            print("=" * 80)
            print("[FAIL] 일부 검증 실패")
            print("=" * 80)

    print("[7] Cleanup 완료")
    print()

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
