"""
Multi-Location Routing 검증 테스트

EdgeAgentMCPClient middleware의 Args 기반 동적 routing 검증:
1. 동일한 tool (read_file)이 path에 따라 다른 location으로 routing
2. Scheduler가 args에서 location hint를 추출
3. Execution trace로 실제 routing 확인

현재 구조:
- LLM은 14개의 ProxyTool만 봄 (suffix 없음)
- Scheduler가 args.path에서 location 결정
- 각 location에 동일한 MCP server가 다른 디렉토리로 배포됨
"""

import asyncio
from pathlib import Path

# EdgeAgent middleware import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient


async def main():
    print("=" * 80)
    print("Multi-Location Routing 검증 테스트 (Args 기반)")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 테스트 환경 준비 - 3개 location에 각각 다른 파일 생성
    # ========================================================================
    print("[1] 테스트 환경 준비...")
    print()

    # 각 location별 디렉토리
    device_dir = Path("/tmp/edgeagent_device_hy")
    edge_dir = Path("/tmp/edgeagent_edge_hy")
    cloud_dir = Path("/tmp/edgeagent_cloud_hy")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)
        # 기존 파일 정리
        for f in dir_path.glob("*"):
            if f.is_file():
                f.unlink()

    # DEVICE: 민감한 로컬 데이터
    (device_dir / "location_marker.txt").write_text(
        "=== DEVICE LOCATION ===\n"
        "This file is stored on DEVICE.\n"
        "Contains: sensitive local data\n"
    )
    (device_dir / "credentials.txt").write_text(
        "Username: admin\n"
        "Password: secret123\n"
    )

    # EDGE: 서버 로그
    (edge_dir / "location_marker.txt").write_text(
        "=== EDGE LOCATION ===\n"
        "This file is stored on EDGE.\n"
        "Contains: server logs\n"
    )
    (edge_dir / "server.log").write_text(
        "[2024-01-01 10:00:00] INFO: Server started\n"
        "[2024-01-01 10:05:00] ERROR: Connection timeout\n"
    )

    # CLOUD: 분석 결과
    (cloud_dir / "location_marker.txt").write_text(
        "=== CLOUD LOCATION ===\n"
        "This file is stored on CLOUD.\n"
        "Contains: analysis results\n"
    )
    (cloud_dir / "analysis.txt").write_text(
        "Analysis Report\n"
        "Total requests: 1000\n"
        "Error rate: 5%\n"
    )

    print(f"   DEVICE ({device_dir}): location_marker.txt, credentials.txt")
    print(f"   EDGE   ({edge_dir}): location_marker.txt, server.log")
    print(f"   CLOUD  ({cloud_dir}): location_marker.txt, analysis.txt")
    print()

    # ========================================================================
    # 2. Middleware 초기화
    # ========================================================================
    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    async with EdgeAgentMCPClient(config_path) as client:
        print("[2] EdgeAgent Middleware 초기화...")
        print(f"   Config: {config_path.name}")
        print(f"   Initialized {len(client.clients)} location clients")
        print()

        # ================================================================
        # 3. Tools 로드 - ProxyTool 구조 확인
        # ================================================================
        print("[3] Tools 로드...")
        tools = await client.get_tools()
        print(f"   Loaded {len(tools)} proxy tools (no location suffix)")
        print()

        # Tool 이름 출력
        print("   Tool names:")
        for tool in sorted(tools, key=lambda t: t.name)[:5]:
            locations = list(tool.backend_tools.keys())
            print(f"     - {tool.name} → backends: {locations}")
        print("     ...")
        print()

        # Tool placement 출력
        client.print_placement_summary()

        # ================================================================
        # 4. Args 기반 Routing 검증
        # ================================================================
        print("\n[4] Args 기반 Routing 검증...")
        print()
        print("동일한 read_file tool을 다른 path로 호출하여")
        print("Scheduler가 올바른 location으로 routing하는지 확인합니다.")
        print()

        # read_file tool 찾기
        read_file = next((t for t in tools if t.name == "read_file"), None)
        if not read_file:
            print("   [ERROR] read_file tool not found")
            return False

        # 테스트 케이스
        test_cases = [
            {
                "name": "DEVICE routing",
                "path": f"{device_dir}/location_marker.txt",
                "expected_location": "DEVICE",
                "expected_content": "DEVICE LOCATION",
            },
            {
                "name": "EDGE routing",
                "path": f"{edge_dir}/location_marker.txt",
                "expected_location": "EDGE",
                "expected_content": "EDGE LOCATION",
            },
            {
                "name": "CLOUD routing",
                "path": f"{cloud_dir}/location_marker.txt",
                "expected_location": "CLOUD",
                "expected_content": "CLOUD LOCATION",
            },
        ]

        results = []

        for test in test_cases:
            print("-" * 60)
            print(f"Test: {test['name']}")
            print(f"Path: {test['path']}")
            print(f"Expected location: {test['expected_location']}")

            try:
                # Tool 호출
                result = await read_file.ainvoke({"path": test["path"]})

                # Trace에서 실제 routing 확인
                trace = client.execution_trace[-1]
                actual_location = trace["location"]

                # 검증
                location_ok = actual_location == test["expected_location"]
                content_ok = test["expected_content"] in str(result)

                if location_ok and content_ok:
                    print(f"   [PASS] Routed to {actual_location}, content verified")
                    results.append({"test": test["name"], "status": "PASS"})
                else:
                    print(f"   [FAIL] Location: {actual_location} (expected {test['expected_location']})")
                    if not content_ok:
                        print(f"          Content mismatch: {str(result)[:100]}...")
                    results.append({"test": test["name"], "status": "FAIL"})

            except Exception as e:
                print(f"   [ERROR] {e}")
                results.append({"test": test["name"], "status": "ERROR", "error": str(e)})

            print()

        # ================================================================
        # 5. Execution Trace 출력
        # ================================================================
        print("=" * 80)
        print("[5] Execution Trace")
        print("=" * 80)
        print()

        for i, trace in enumerate(client.execution_trace, 1):
            print(f"   {i}. Tool: {trace['tool']}, Location: {trace['location']}")

        print()

        # ================================================================
        # 6. 결과 요약
        # ================================================================
        print("=" * 80)
        print("[6] 테스트 결과 요약")
        print("=" * 80)
        print()

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        errors = sum(1 for r in results if r["status"] == "ERROR")

        for result in results:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "ERROR": "[ERROR]"}[result["status"]]
            print(f"   {icon} {result['test']}")

        print()
        print(f"   Total: {len(results)} tests")
        print(f"   Passed: {passed}, Failed: {failed}, Errors: {errors}")
        print()

        success = passed == len(results)

        if success:
            print("=" * 80)
            print("[PASS] Multi-Location Routing 검증 성공!")
            print("=" * 80)
            print()
            print("검증된 항목:")
            print("  1. ProxyTool이 args.path에서 location hint 추출")
            print("  2. Scheduler가 /edgeagent_device/, /edgeagent_edge/, /edgeagent_cloud/ 패턴 인식")
            print("  3. 각 호출이 올바른 backend tool로 routing")
            print("  4. Execution trace에 routing 기록")
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
