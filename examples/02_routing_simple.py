"""
Middleware Routing 검증 (단순화 버전)

LLM 없이 직접 middleware의 routing 로직을 검증합니다.
이 테스트는 ProxyTool을 통한 args 기반 동적 routing을 확인합니다.

현재 구조:
- 14개 ProxyTool (location suffix 없음)
- Scheduler가 args.path에서 location hint 추출
- 각 location의 MCP server로 routing
"""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient


async def main():
    print("=" * 80)
    print("Middleware Routing 검증 (단순화 버전)")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 테스트 환경 준비
    # ========================================================================
    print("[1] 테스트 환경 준비...")

    device_dir = Path("/tmp/edgeagent_device")
    edge_dir = Path("/tmp/edgeagent_edge")
    cloud_dir = Path("/tmp/edgeagent_cloud")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)

    # 각 location에 marker 파일 생성
    (device_dir / "device_marker.txt").write_text("This is DEVICE\n")
    (edge_dir / "edge_marker.txt").write_text("This is EDGE\n")
    (cloud_dir / "cloud_marker.txt").write_text("This is CLOUD\n")

    print("   Done")
    print()

    # ========================================================================
    # 2-7. Context manager 내에서 실행
    # ========================================================================
    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    # async with를 사용하여 session lifecycle 자동 관리
    async with EdgeAgentMCPClient(config_path) as client:
        # ================================================================
        # 2. Middleware 초기화 확인
        # ================================================================
        print("[2] Middleware 초기화...")
        print(f"   Config: {config_path.name}")
        print(f"   Location Clients: {list(client.clients.keys())}")
        print()

        # ================================================================
        # 3. Tool 로드 및 구조 검증
        # ================================================================
        print("[3] Tool 로드 및 구조 검증...")
        print()

        tools = await client.get_tools()
        print(f"   Loaded {len(tools)} proxy tools (no location suffix)")
        print()

        # Tool 이름 확인 - suffix가 없어야 함
        print("   Tool names (verifying no location suffix):")
        for tool in sorted(tools, key=lambda t: t.name)[:5]:
            has_suffix = tool.name.endswith(("_device", "_edge", "_cloud"))
            status = "[FAIL]" if has_suffix else "[PASS]"
            backends = list(tool.backend_tools.keys()) if hasattr(tool, "backend_tools") else []
            print(f"     {status} {tool.name} → backends: {backends}")
        print("     ...")
        print()

        # suffix 검사
        tools_with_suffix = [t for t in tools if t.name.endswith(("_device", "_edge", "_cloud"))]
        if tools_with_suffix:
            print(f"   [FAIL] Found {len(tools_with_suffix)} tools with location suffix!")
            return False

        print(f"   [PASS] All {len(tools)} tools have no location suffix")
        print()

        # ================================================================
        # 4. Placement Summary
        # ================================================================
        client.print_placement_summary()
        print()

        # ================================================================
        # 5. Args 기반 Routing 검증
        # ================================================================
        print("[5] Args 기반 Routing 검증...")
        print()

        # read_file tool 찾기
        read_file = next((t for t in tools if t.name == "read_file"), None)
        if not read_file:
            print("   [ERROR] read_file tool not found")
            return False

        test_results = []

        # 각 location의 파일 읽기 테스트
        test_cases = [
            {
                "path": str(device_dir / "device_marker.txt"),
                "expected_location": "DEVICE",
                "expected_content": "This is DEVICE",
            },
            {
                "path": str(edge_dir / "edge_marker.txt"),
                "expected_location": "EDGE",
                "expected_content": "This is EDGE",
            },
            {
                "path": str(cloud_dir / "cloud_marker.txt"),
                "expected_location": "CLOUD",
                "expected_content": "This is CLOUD",
            },
        ]

        for test in test_cases:
            print(f"   Testing path: {test['path']}")
            print(f"   Expected location: {test['expected_location']}")

            try:
                result = await read_file.ainvoke({"path": test["path"]})

                # Trace에서 실제 routing 확인
                trace = client.execution_trace[-1]
                actual_location = trace["location"]

                # 검증
                location_ok = actual_location == test["expected_location"]
                content_ok = test["expected_content"] in str(result)

                if location_ok and content_ok:
                    print(f"   [PASS] Routed to {actual_location}, content verified")
                    test_results.append(("PASS", test["expected_location"]))
                else:
                    if not location_ok:
                        print(f"   [FAIL] Wrong location: {actual_location} (expected {test['expected_location']})")
                    if not content_ok:
                        print(f"   [FAIL] Content mismatch: {str(result)[:50]}...")
                    test_results.append(("FAIL", test["expected_location"]))

            except Exception as e:
                print(f"   [ERROR] {e}")
                test_results.append(("ERROR", test["expected_location"]))

            print()

        # ================================================================
        # 6. Execution Trace 출력
        # ================================================================
        print("[6] Execution Trace...")
        print()
        for i, trace in enumerate(client.execution_trace, 1):
            print(f"   {i}. Tool: {trace['tool']}, Location: {trace['location']}")
        print()

        # ================================================================
        # 7. 결과 요약
        # ================================================================
        print("=" * 80)
        print("[7] 결과 요약")
        print("=" * 80)
        print()

        passed = sum(1 for status, _ in test_results if status == "PASS")
        total = len(test_results)

        print(f"   Direct Tool Call 검증: {passed}/{total} passed")
        for status, location in test_results:
            print(f"     [{status}] {location} routing")
        print()

        # 최종 판정
        success = passed == total

        if success:
            print("=" * 80)
            print("[PASS] Middleware Routing 검증 성공!")
            print("=" * 80)
            print()
            print("검증 완료:")
            print("  1. ProxyTool 구조 (suffix 없음)")
            print("  2. Args 기반 location routing")
            print("  3. Scheduler가 path 패턴에서 location 추출")
            print("  4. Execution trace에 routing 기록")
            print()
        else:
            print("=" * 80)
            print("[FAIL] 일부 검증 실패")
            print("=" * 80)

    # Context manager가 여기서 자동으로 모든 session 정리
    print("[8] Cleanup 완료 (async with 자동 처리)")
    print()

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
