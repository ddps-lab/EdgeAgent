"""
Phase 0.3: MCP Adapter 통합 확인

langchain-mcp-adapters를 사용하여 실제 MCP server와 연결하고
LangChain agent가 MCP tools를 호출할 수 있는지 검증합니다.
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
# MCP 서버 설정
# ============================================================================

async def main():
    print("=" * 80)
    print("Phase 0.3: MCP Adapter 통합 확인")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. MCP 서버 설정
    # ========================================================================
    print("[1] MCP 서버 설정...")
    print()

    # filesystem MCP 서버 설정
    # /tmp/edgeagent_test 디렉토리에 테스트 파일 생성
    test_dir = "/tmp/edgeagent_test"
    os.makedirs(test_dir, exist_ok=True)

    # 테스트 파일 생성
    with open(f"{test_dir}/test1.txt", "w") as f:
        f.write("This is test file 1.\nIt contains multiple lines.\n")

    with open(f"{test_dir}/test2.txt", "w") as f:
        f.write("This is test file 2.\nAnother test file.\n")

    with open(f"{test_dir}/data.json", "w") as f:
        f.write('{"name": "EdgeAgent", "version": "0.1.0"}\n')

    print(f"   ✓ 테스트 디렉토리 생성: {test_dir}")
    print(f"   ✓ 테스트 파일 생성: test1.txt, test2.txt, data.json")
    print()

    # MCP 서버 설정
    servers = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                test_dir
            ]
        }
    }

    print("   MCP Server Config:")
    print(f"   - Name: filesystem")
    print(f"   - Command: npx -y @modelcontextprotocol/server-filesystem")
    print(f"   - Directory: {test_dir}")
    print()

    # ========================================================================
    # 2. MCP Client 초기화 및 Tools 로드
    # ========================================================================
    print("[2] MCP Client 초기화 및 Tools 로드...")
    print()

    try:
        # MultiServerMCPClient 생성
        client = MultiServerMCPClient(servers)
        print("   ✓ MultiServerMCPClient created")

        # filesystem 서버 세션 시작 및 tools 로드
        async with client.session("filesystem") as session:
            print("   ✓ MCP session started for 'filesystem'")

            # Tools 로드
            tools = await load_mcp_tools(session)
            print(f"   ✓ Loaded {len(tools)} tools from MCP server")
            print()

            # Tool 목록 출력
            print("   Available MCP Tools:")
            for tool in tools:
                print(f"   - {tool.name}: {tool.description[:60]}...")
            print()

            # ================================================================
            # 3. LLM 및 Agent 초기화
            # ================================================================
            print("[3] LLM 및 Agent 초기화...")
            print()

            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0
            )
            print("   ✓ ChatOpenAI initialized")

            # ReAct Agent 생성
            agent_executor = create_agent(
                llm,
                tools,
                system_prompt="You are a helpful assistant with access to filesystem tools."
            )
            print("   ✓ ReAct agent created with MCP tools")
            print()

            # ================================================================
            # 4. 테스트 케이스 실행
            # ================================================================

            test_cases = [
                {
                    "name": "Test 1: 파일 목록 조회",
                    "input": f"List all files in the directory {test_dir}"
                },
                {
                    "name": "Test 2: 파일 읽기",
                    "input": f"Read the contents of {test_dir}/test1.txt"
                },
                {
                    "name": "Test 3: 파일 검색",
                    "input": f"Search for '.txt' files in {test_dir}"
                }
            ]

            results = []

            for i, test_case in enumerate(test_cases, 1):
                print("\n" + "=" * 80)
                print(f"{test_case['name']}")
                print("=" * 80)
                print(f"Query: {test_case['input']}")
                print("-" * 80)

                try:
                    result = await agent_executor.ainvoke({
                        "messages": [("user", test_case['input'])]
                    })
                    output = result["messages"][-1].content if result.get("messages") else "No output"

                    print("-" * 80)
                    print(f"✓ Final Answer: {output}")

                    results.append({
                        "test": test_case['name'],
                        "status": "PASS",
                        "output": output
                    })

                except Exception as e:
                    print(f"✗ Error: {e}")
                    results.append({
                        "test": test_case['name'],
                        "status": "FAIL",
                        "error": str(e)
                    })

            # ================================================================
            # 5. 결과 요약
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
                    # 긴 출력은 잘라서 표시
                    output = result['output']
                    if len(output) > 100:
                        output = output[:100] + "..."
                    print(f"  → {output}")
                else:
                    print(f"  → Error: {result.get('error', 'Unknown error')}")

            print()
            print(f"Total: {len(results)} tests")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print()

            if failed == 0:
                print("=" * 80)
                print("✓ Phase 0.3 검증 완료: MCP adapter가 정상 동작합니다!")
                print("=" * 80)
                return True
            else:
                print("=" * 80)
                print("✗ Phase 0.3 검증 실패: 일부 테스트가 실패했습니다.")
                print("=" * 80)
                return False

    except Exception as e:
        print(f"\n✗ MCP Client 초기화 실패: {e}")
        print()
        print("문제 해결:")
        print("1. Node.js와 npm이 설치되어 있는지 확인")
        print("2. MCP filesystem server 설치:")
        print("   npm install -g @modelcontextprotocol/server-filesystem")
        return False


if __name__ == "__main__":
    # API key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        exit(1)

    # asyncio로 실행
    success = asyncio.run(main())
    exit(0 if success else 1)
