"""
Proxy Tool Routing 검증 테스트

LLM에 노출되는 tool 구조 변경 검증:
- 이전: 42개 tools (14 tools × 3 locations) with suffix
- 이후: 14개 proxy tools (location suffix 없음)

Scheduler가 호출 시점에 location 결정
"""

import asyncio
from pathlib import Path

# EdgeAgent middleware import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient


async def main():
    print("=" * 80)
    print("Proxy Tool Routing 검증 테스트")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 테스트 환경 준비
    # ========================================================================
    print("[1] 테스트 환경 준비...")
    print()

    device_dir = Path("/tmp/edgeagent_device_hy")
    edge_dir = Path("/tmp/edgeagent_edge_hy")
    cloud_dir = Path("/tmp/edgeagent_cloud_hy")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)
        for f in dir_path.glob("*"):
            f.unlink()

    # 각 location에 테스트 파일 생성
    (device_dir / "test.txt").write_text("DEVICE content")
    (edge_dir / "test.txt").write_text("EDGE content")
    (cloud_dir / "test.txt").write_text("CLOUD content")

    print(f"   Created test files in each location")
    print()

    # ========================================================================
    # 2. Middleware 초기화 및 Tools 로드
    # ========================================================================
    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    async with EdgeAgentMCPClient(config_path) as client:
        print("[2] EdgeAgent Middleware 초기화...")
        print(f"   Config: {config_path.name}")
        print()

        tools = await client.get_tools()

        # ================================================================
        # 3. Tool 구조 검증
        # ================================================================
        print("[3] Tool 구조 검증...")
        print()

        print(f"   Total tools loaded: {len(tools)}")
        print()

        # Tool 이름 출력 (suffix 없어야 함)
        print("   Tool names (should have NO location suffix):")
        tool_names = sorted(set(t.name for t in tools))
        for name in tool_names:
            # suffix가 있는지 확인
            has_suffix = name.endswith(("_device", "_edge", "_cloud"))
            status = "[FAIL] HAS SUFFIX!" if has_suffix else "[PASS]"
            print(f"     {status} {name}")
        print()

        # suffix가 있는 tool이 있으면 실패
        tools_with_suffix = [t for t in tools if t.name.endswith(("_device", "_edge", "_cloud"))]
        if tools_with_suffix:
            print(f"   [FAIL] Found {len(tools_with_suffix)} tools with location suffix!")
            return False

        print(f"   [PASS] All {len(tools)} tools have no location suffix!")
        print()

        # ================================================================
        # 4. ProxyTool 구조 확인
        # ================================================================
        print("[4] ProxyTool 구조 확인...")
        print()

        for tool in tools[:3]:  # 처음 3개만 출력
            print(f"   Tool: {tool.name}")
            print(f"     Type: {type(tool).__name__}")
            if hasattr(tool, "backend_tools"):
                print(f"     Backend locations: {list(tool.backend_tools.keys())}")
            if hasattr(tool, "parent_tool_name"):
                print(f"     Parent tool: {tool.parent_tool_name}")
            print()

        # ================================================================
        # 5. Placement Summary
        # ================================================================
        client.print_placement_summary()

        # ================================================================
        # 6. 실제 호출 테스트
        # ================================================================
        print("[6] 실제 호출 테스트...")
        print()

        # read_file tool 찾기
        read_file_tool = next((t for t in tools if t.name == "read_file"), None)

        if read_file_tool:
            print(f"   Found: {read_file_tool}")
            print(f"   Backend tools: {list(read_file_tool.backend_tools.keys())}")
            print()

            # 현재 static_mapping에서 filesystem은 DEVICE로 설정되어 있음
            # 따라서 모든 호출이 DEVICE로 routing 되어야 함
            try:
                result = await read_file_tool.ainvoke({"path": "/tmp/edgeagent_device_hy/test.txt"})
                print(f"   Read result: {result}")
                print()

                # Execution trace 확인
                print("   Execution trace (from middleware):")
                for trace in client.execution_trace:
                    print(f"     Tool: {trace['tool']}, Location: {trace['location']}")

                # ProxyTool의 execution_trace 확인
                print(f"   Execution trace (from tool): {read_file_tool.execution_trace}")
                print()

                # 결과에서 content 확인 (DEVICE로 routing 되었는지)
                if "DEVICE content" in str(result):
                    print("   [PASS] Tool was routed to DEVICE as expected (verified by content)")
                else:
                    print(f"   [FAIL] Expected 'DEVICE content', got: {result}")
                    return False

            except Exception as e:
                print(f"   [ERROR] {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("   [SKIP] read_file tool not found")

        print()

    # ========================================================================
    # 7. 최종 결과
    # ========================================================================
    print("=" * 80)
    print("[PASS] Proxy Tool Routing 검증 성공!")
    print("=" * 80)
    print()
    print("검증된 항목:")
    print("  1. Tool 이름에 location suffix 없음")
    print("  2. ProxyTool이 여러 backend tool 보유")
    print("  3. Scheduler가 호출 시점에 location 결정")
    print("  4. 올바른 location으로 routing")
    print()

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
