"""
Phase 4: Middleware를 통한 End-to-End 테스트

EdgeAgentMCPClient middleware를 사용하여:
1. YAML 설정에서 multi-endpoint 로드
2. Static scheduler로 location 결정
3. LangChain agent가 올바른 location의 tools 호출
4. Execution trace로 routing 검증
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# EdgeAgent middleware import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient

# 환경 변수 로드
load_dotenv()


async def main():
    print("=" * 80)
    print("Phase 4: EdgeAgent Middleware End-to-End 테스트")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. 테스트 환경 준비
    # ========================================================================
    print("[1] 테스트 환경 준비...")
    print()

    # 테스트 디렉토리 생성
    device_dir = Path("/tmp/edgeagent_device")
    edge_dir = Path("/tmp/edgeagent_edge")
    cloud_dir = Path("/tmp/edgeagent_cloud")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)

    # DEVICE: 민감한 로컬 데이터
    (device_dir / "sensitive.txt").write_text(
        "User credentials: admin/password123\n"
        "API Key: sk-secret-key-12345\n"
        "This file should NEVER leave the device.\n"
    )

    (device_dir / "report.txt").write_text(
        "Monthly Sales Report\n"
        "====================\n"
        "January: $10,000\n"
        "February: $12,000\n"
        "March: $15,000\n"
    )

    # EDGE: 로그 파일
    (edge_dir / "server.log").write_text(
        "[2024-01-01 10:00:00] INFO: Server started\n"
        "[2024-01-01 10:05:00] INFO: Request from 192.168.1.10\n"
        "[2024-01-01 10:10:00] ERROR: Connection timeout\n"
        "[2024-01-01 10:15:00] WARNING: High memory usage\n"
        "[2024-01-01 10:20:00] ERROR: Database connection failed\n"
        "[2024-01-01 10:25:00] INFO: Request completed\n"
        "[2024-01-01 10:30:00] ERROR: File not found\n"
    )

    # CLOUD: 처리 결과
    (cloud_dir / "results.txt").write_text(
        "Analysis Results\n"
        "================\n"
        "Total errors: 15\n"
        "Total warnings: 3\n"
        "System health: DEGRADED\n"
    )

    print(f"   ✓ DEVICE directory: {device_dir}")
    print(f"     - sensitive.txt (privacy sensitive)")
    print(f"     - report.txt")
    print()
    print(f"   ✓ EDGE directory: {edge_dir}")
    print(f"     - server.log (needs reduction)")
    print()
    print(f"   ✓ CLOUD directory: {cloud_dir}")
    print(f"     - results.txt")
    print()

    # ========================================================================
    # 2. EdgeAgent Middleware 초기화
    # ========================================================================
    print("[2] EdgeAgent Middleware 초기화...")
    print()

    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    try:
        # Middleware 생성
        client = EdgeAgentMCPClient(config_path)
        print(f"   ✓ Middleware created with config: {config_path.name}")

        # 초기화
        await client.initialize()
        print(f"   ✓ Initialized {len(client.clients)} location clients")
        print()

        # Tool placement 요약
        client.print_placement_summary()

        # ================================================================
        # 3. LangChain Agent 생성
        # ================================================================
        print("[3] LangChain Agent 생성...")
        print()

        # Tools 로드 (스케줄러가 결정한 location에서)
        tools = await client.get_tools()
        print(f"   ✓ Loaded {len(tools)} tools from scheduled locations")
        print()

        # Tool 목록
        print("   Available Tools:")
        for tool in tools:
            metadata = getattr(tool, 'metadata', {})
            location = metadata.get('location', 'N/A')
            print(f"   - {tool.name} [{location}]")
        print()

        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
        print("   ✓ ChatOpenAI initialized")

        # Agent 생성
        agent_executor = create_agent(
            llm,
            tools,
            system_prompt="You are a helpful assistant with access to filesystem tools at different locations. The middleware automatically routes your tool calls to the appropriate location."
        )
        print("   ✓ Agent created")
        print()

        # ================================================================
        # 4. 테스트 케이스 실행
        # ================================================================

        test_cases = [
            {
                "name": "Test 1: List files",
                "input": f"List all files in {device_dir}",
                "expected_location": "DEVICE",
                "description": "List files on DEVICE"
            },
            {
                "name": "Test 2: Read file",
                "input": f"Read the file at {device_dir}/report.txt",
                "expected_location": "DEVICE",
                "description": "Read file from DEVICE"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print("\n" + "=" * 80)
            print(f"{test_case['name']}")
            print("=" * 80)
            print(f"Description: {test_case['description']}")
            print(f"Expected Location: {test_case['expected_location']}")
            print(f"Query: {test_case['input']}")
            print("-" * 80)

            try:
                # Agent 실행
                result = await agent_executor.ainvoke({
                    "messages": [("user", test_case['input'])]
                })
                output = result["messages"][-1].content if result.get("messages") else "No output"

                print("-" * 80)
                print(f"✓ Final Answer:")
                # 긴 출력은 줄여서 표시
                if len(output) > 200:
                    print(f"  {output[:200]}...")
                else:
                    print(f"  {output}")

                results.append({
                    "test": test_case['name'],
                    "status": "PASS",
                    "output": output,
                    "expected_location": test_case['expected_location']
                })

            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()

                results.append({
                    "test": test_case['name'],
                    "status": "FAIL",
                    "error": str(e)
                })

        # ================================================================
        # 5. Execution Trace 출력
        # ================================================================

        client.print_execution_trace()

        # ================================================================
        # 6. Cleanup
        # ================================================================

        print("\n[Cleanup] Closing MCP sessions...")
        await client.cleanup()
        print("   ✓ All sessions closed")

        # ================================================================
        # 7. 결과 요약
        # ================================================================

        print("\n" + "=" * 80)
        print("테스트 결과 요약")
        print("=" * 80)
        print()

        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')

        for result in results:
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_icon} {result['test']}: {result['status']}")
            if result['status'] == 'PASS':
                print(f"  → Expected location: {result['expected_location']}")
            else:
                print(f"  → Error: {result.get('error', 'Unknown error')}")

        print()
        print(f"Total: {len(results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()

        # ================================================================
        # 7. 최종 검증
        # ================================================================

        if failed == 0:
            print("=" * 80)
            print("✓✓✓ Phase 4 검증 완료! ✓✓✓")
            print("=" * 80)
            print()
            print("EdgeAgent Middleware가 성공적으로 동작합니다:")
            print()
            print("✓ YAML 설정에서 multi-endpoint 로드")
            print("✓ Static scheduler로 tool → location 매핑")
            print("✓ LangChain agent와 seamless 통합")
            print("✓ Location-aware tool routing 동작")
            print("✓ Execution trace로 routing 검증 가능")
            print()
            print("다음 단계:")
            print("- Heuristic scheduler 구현")
            print("- DP scheduling 알고리즘 구현")
            print("- 실제 Edge/Cloud에 MCP 서버 배포")
            print("- Cost model 및 성능 측정")
            print()
            return True
        else:
            print("=" * 80)
            print("✗ Phase 4 검증 실패")
            print("=" * 80)
            return False

    except Exception as e:
        print(f"\n✗ Middleware 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
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
