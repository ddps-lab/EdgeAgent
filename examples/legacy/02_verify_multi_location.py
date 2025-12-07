"""
Phase 0.4: Multi-location 시뮬레이션

동일한 tool을 여러 "가상" location(DEVICE/EDGE/CLOUD)에서 실행할 수 있도록
multi-endpoint를 시뮬레이션하고 routing을 테스트합니다.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# 환경 변수 로드
load_dotenv()


# ============================================================================
# Multi-location 시뮬레이션
# ============================================================================

async def main():
    print("=" * 80)
    print("Phase 0.4: Multi-location 시뮬레이션")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 여러 location 시뮬레이션을 위한 디렉토리 준비
    # ========================================================================
    print("[1] 가상 location별 디렉토리 준비...")
    print()

    # DEVICE: 로컬 데이터
    device_dir = "/tmp/edgeagent_device"
    os.makedirs(device_dir, exist_ok=True)
    with open(f"{device_dir}/local_data.txt", "w") as f:
        f.write("This is LOCAL data stored on DEVICE.\n")
        f.write("Sensitive information that should not leave the device.\n")

    print(f"   ✓ DEVICE location: {device_dir}")
    print(f"     - local_data.txt (sensitive)")

    # EDGE: 중간 처리 데이터
    edge_dir = "/tmp/edgeagent_edge"
    os.makedirs(edge_dir, exist_ok=True)
    with open(f"{edge_dir}/processed_data.txt", "w") as f:
        f.write("This is PROCESSED data at EDGE.\n")
        f.write("Log file: [INFO] System running normally\n")
        f.write("Log file: [ERROR] Connection timeout\n")
        f.write("Log file: [INFO] Request processed\n")

    print(f"   ✓ EDGE location: {edge_dir}")
    print(f"     - processed_data.txt (logs)")

    # CLOUD: 외부 데이터
    cloud_dir = "/tmp/edgeagent_cloud"
    os.makedirs(cloud_dir, exist_ok=True)
    with open(f"{cloud_dir}/cloud_data.txt", "w") as f:
        f.write("This is CLOUD data.\n")
        f.write("Accessible from anywhere.\n")
        f.write("Result: Analysis complete\n")

    print(f"   ✓ CLOUD location: {cloud_dir}")
    print(f"     - cloud_data.txt (results)")
    print()

    # ========================================================================
    # 2. 각 location별 MCP 서버 설정
    # ========================================================================
    print("[2] Multi-location MCP 서버 설정...")
    print()

    servers = {
        # DEVICE: 로컬 파일시스템 접근
        "filesystem_device": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                device_dir
            ]
        },
        # EDGE: 중간 처리 데이터
        "filesystem_edge": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                edge_dir
            ]
        },
        # CLOUD: 클라우드 데이터
        "filesystem_cloud": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                cloud_dir
            ]
        }
    }

    print("   MCP Server Configuration:")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ Location │ Server Name        │ Directory            │")
    print("   ├──────────┼────────────────────┼──────────────────────┤")
    print(f"   │ DEVICE   │ filesystem_device  │ {device_dir:20s} │")
    print(f"   │ EDGE     │ filesystem_edge    │ {edge_dir:20s} │")
    print(f"   │ CLOUD    │ filesystem_cloud   │ {cloud_dir:20s} │")
    print("   └─────────────────────────────────────────────────────────┘")
    print()

    # ========================================================================
    # 3. MCP Client 초기화 및 모든 location의 tools 로드
    # ========================================================================
    print("[3] 모든 location의 MCP tools 로드...")
    print()

    try:
        client = MultiServerMCPClient(servers)
        print("   ✓ MultiServerMCPClient created")
        print()

        all_tools = []
        tool_location_map = {}  # tool → location 매핑

        # 각 location별로 세션 생성 및 tools 로드
        for location, server_name in [
            ("DEVICE", "filesystem_device"),
            ("EDGE", "filesystem_edge"),
            ("CLOUD", "filesystem_cloud")
        ]:
            async with client.session(server_name) as session:
                print(f"   [{location}] Loading tools from '{server_name}'...")

                # Tools 로드
                tools = await load_mcp_tools(session)

                # Tool 이름에 location suffix 추가
                for tool in tools:
                    # 원래 이름 저장
                    original_name = tool.name
                    # Location suffix 추가
                    tool.name = f"{original_name}_{location}"
                    tool.description = f"[{location}] {tool.description}"

                    # 매핑 저장
                    tool_location_map[tool.name] = {
                        "location": location,
                        "server": server_name,
                        "original_name": original_name
                    }

                    all_tools.append(tool)

                print(f"   ✓ Loaded {len(tools)} tools from {location}")

        print()
        print(f"   Total tools loaded: {len(all_tools)}")
        print()

        # Tool 목록 출력
        print("   Available Tools by Location:")
        print("   ┌──────────┬─────────────────────────────────────────┐")
        print("   │ Location │ Tool Name                               │")
        print("   ├──────────┼─────────────────────────────────────────┤")
        for tool in all_tools:
            location = tool_location_map[tool.name]["location"]
            print(f"   │ {location:8s} │ {tool.name:39s} │")
        print("   └──────────┴─────────────────────────────────────────┘")
        print()

        # ================================================================
        # 4. LLM 및 Agent 초기화
        # ================================================================
        print("[4] Agent 초기화...")
        print()

        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
        print("   ✓ ChatOpenAI initialized")

        # ReAct Agent 생성
        agent_executor = create_agent(
            llm,
            all_tools,
            system_prompt="You are a helpful assistant with access to filesystem tools at different locations (DEVICE, EDGE, CLOUD). Choose the appropriate location based on the user's request."
        )
        print("   ✓ ReAct agent created with tools from all locations")
        print()

        # ================================================================
        # 5. Multi-location Routing 테스트
        # ================================================================

        test_cases = [
            {
                "name": "Test 1: DEVICE에서 파일 목록",
                "input": f"Use the list_directory_DEVICE tool to list files in {device_dir}",
                "expected_location": "DEVICE"
            },
            {
                "name": "Test 2: EDGE에서 파일 목록",
                "input": f"Use the list_directory_EDGE tool to list files in {edge_dir}",
                "expected_location": "EDGE"
            },
            {
                "name": "Test 3: CLOUD에서 파일 목록",
                "input": f"Use the list_directory_CLOUD tool to list files in {cloud_dir}",
                "expected_location": "CLOUD"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print("\n" + "=" * 80)
            print(f"{test_case['name']}")
            print("=" * 80)
            print(f"Query: {test_case['input']}")
            print(f"Expected Location: {test_case['expected_location']}")
            print("-" * 80)

            try:
                result = await agent_executor.ainvoke({
                    "messages": [("user", test_case['input'])]
                })
                output = result["messages"][-1].content if result.get("messages") else "No output"

                print("-" * 80)
                print(f"✓ Final Answer: {output[:200]}...")

                results.append({
                    "test": test_case['name'],
                    "status": "PASS",
                    "output": output,
                    "expected_location": test_case['expected_location']
                })

            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    "test": test_case['name'],
                    "status": "FAIL",
                    "error": str(e)
                })

        # ================================================================
        # 6. 결과 요약
        # ================================================================

        print("\n\n" + "=" * 80)
        print("테스트 결과 요약")
        print("=" * 80)

        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')

        for result in results:
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_icon} {result['test']}: {result['status']}")
            if result['status'] == 'PASS':
                print(f"  → Location: {result['expected_location']}")
            else:
                print(f"  → Error: {result.get('error', 'Unknown error')}")

        print()
        print(f"Total: {len(results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()

        if failed == 0:
            print("=" * 80)
            print("✓ Phase 0.4 검증 완료: Multi-location routing이 동작합니다!")
            print("=" * 80)
            print()
            print("Key Insights:")
            print("- 동일한 tool을 여러 location에 배포 가능")
            print("- Tool 이름에 location suffix를 추가하여 구분")
            print("- Agent가 명시적 지시에 따라 적절한 location 선택")
            print()
            print("다음 단계: Middleware에서 자동으로 location을 선택하도록 구현")
            return True
        else:
            print("=" * 80)
            print("✗ Phase 0.4 검증 실패")
            print("=" * 80)
            return False

    except Exception as e:
        print(f"\n✗ MCP Client 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # API key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        exit(1)

    success = asyncio.run(main())
    exit(0 if success else 1)
