"""
LLM Agent Integration 테스트

LLM Agent가 자연어 prompt를 받아 tool을 선택하고,
Scheduler가 location을 결정하는 End-to-End 테스트.

테스트 시나리오:
1. 파일 읽기 요청 → LLM이 read_file 선택 → Scheduler가 path 기반 routing
2. 여러 location의 파일 비교 요청 → 다중 tool 호출
3. Constraint 기반 routing 확인 (slack → CLOUD)
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# EdgeAgent middleware import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from edgeagent.middleware import EdgeAgentMCPClient


async def setup_test_environment():
    """테스트용 파일 생성"""
    device_dir = Path("/tmp/edgeagent_device_hy")
    edge_dir = Path("/tmp/edgeagent_edge_hy")
    cloud_dir = Path("/tmp/edgeagent_cloud_hy")

    for dir_path in [device_dir, edge_dir, cloud_dir]:
        dir_path.mkdir(exist_ok=True)

    # DEVICE: 사용자 설정
    (device_dir / "user_config.txt").write_text(
        "User: Alice\n"
        "Theme: dark\n"
        "Language: ko\n"
    )

    # EDGE: 서버 상태
    (edge_dir / "server_status.txt").write_text(
        "Server: edge-server-01\n"
        "Status: running\n"
        "Uptime: 72 hours\n"
        "Load: 45%\n"
    )

    # CLOUD: 분석 리포트
    (cloud_dir / "report.txt").write_text(
        "Monthly Report - December 2024\n"
        "Total Users: 10,000\n"
        "Active Users: 7,500\n"
        "Revenue: $50,000\n"
    )

    print("테스트 파일 생성 완료:")
    print(f"  - {device_dir}/user_config.txt")
    print(f"  - {edge_dir}/server_status.txt")
    print(f"  - {cloud_dir}/report.txt")

    return device_dir, edge_dir, cloud_dir


async def test_single_file_read(agent, client):
    """테스트 1: 단일 파일 읽기"""
    print("\n" + "=" * 70)
    print("Test 1: 단일 파일 읽기 (DEVICE routing)")
    print("=" * 70)

    prompt = "Read the file at /tmp/edgeagent_device_hy/user_config.txt and tell me what's in it."
    print(f"\nPrompt: {prompt}\n")

    # 이전 trace 개수 기록
    trace_before = len(client.execution_trace)

    # Agent 실행
    result = await agent.ainvoke({"messages": [("user", prompt)]})

    # 결과 출력
    final_message = result["messages"][-1].content
    print(f"Agent Response:\n{final_message}\n")

    # Routing 확인
    if len(client.execution_trace) > trace_before:
        trace = client.execution_trace[-1]
        print(f"Routing: tool={trace['tool']}, location={trace['location']}")

        if trace["location"] == "DEVICE":
            print("[PASS] DEVICE로 올바르게 routing됨")
            return True
        else:
            print(f"[FAIL] Expected DEVICE, got {trace['location']}")
            return False
    else:
        print("[WARN] No tool call detected")
        return False


async def test_multi_location_comparison(agent, client):
    """테스트 2: 여러 location 파일 비교"""
    print("\n" + "=" * 70)
    print("Test 2: 여러 location 파일 비교 (DEVICE + EDGE)")
    print("=" * 70)

    prompt = """I need to check two files:
1. /tmp/edgeagent_device_hy/user_config.txt
2. /tmp/edgeagent_edge_hy/server_status.txt

Please read both files and summarize what you found."""

    print(f"\nPrompt: {prompt}\n")

    trace_before = len(client.execution_trace)

    result = await agent.ainvoke({"messages": [("user", prompt)]})

    final_message = result["messages"][-1].content
    print(f"Agent Response:\n{final_message}\n")

    # 새로운 trace들 확인
    new_traces = client.execution_trace[trace_before:]
    print(f"Tool calls: {len(new_traces)}")

    locations_used = set()
    for trace in new_traces:
        print(f"  - tool={trace['tool']}, location={trace['location']}")
        locations_used.add(trace["location"])

    # DEVICE와 EDGE 모두 사용되었는지 확인
    if "DEVICE" in locations_used and "EDGE" in locations_used:
        print("[PASS] DEVICE와 EDGE 모두 routing됨")
        return True
    else:
        print(f"[FAIL] Expected DEVICE+EDGE, got {locations_used}")
        return False


async def test_cloud_routing(agent, client):
    """테스트 3: CLOUD routing"""
    print("\n" + "=" * 70)
    print("Test 3: CLOUD 파일 읽기")
    print("=" * 70)

    prompt = "Read /tmp/edgeagent_cloud_hy/report.txt and give me the revenue figure."
    print(f"\nPrompt: {prompt}\n")

    trace_before = len(client.execution_trace)

    result = await agent.ainvoke({"messages": [("user", prompt)]})

    final_message = result["messages"][-1].content
    print(f"Agent Response:\n{final_message}\n")

    if len(client.execution_trace) > trace_before:
        trace = client.execution_trace[-1]
        print(f"Routing: tool={trace['tool']}, location={trace['location']}")

        if trace["location"] == "CLOUD":
            print("[PASS] CLOUD로 올바르게 routing됨")
            return True
        else:
            print(f"[FAIL] Expected CLOUD, got {trace['location']}")
            return False
    else:
        print("[WARN] No tool call detected")
        return False


async def test_file_write(agent, client):
    """테스트 4: 파일 쓰기"""
    print("\n" + "=" * 70)
    print("Test 4: 파일 쓰기 (EDGE routing)")
    print("=" * 70)

    prompt = "Create a file at /tmp/edgeagent_edge_hy/test_output.txt with the content 'Hello from LLM Agent!'"
    print(f"\nPrompt: {prompt}\n")

    trace_before = len(client.execution_trace)

    result = await agent.ainvoke({"messages": [("user", prompt)]})

    final_message = result["messages"][-1].content
    print(f"Agent Response:\n{final_message}\n")

    if len(client.execution_trace) > trace_before:
        trace = client.execution_trace[-1]
        print(f"Routing: tool={trace['tool']}, location={trace['location']}")

        # 파일이 실제로 생성되었는지 확인
        test_file = Path("/tmp/edgeagent_edge_hy/test_output.txt")
        if test_file.exists() and trace["location"] == "EDGE":
            print(f"File content: {test_file.read_text()}")
            print("[PASS] EDGE에 파일 생성됨")
            return True
        else:
            print(f"[FAIL] File exists: {test_file.exists()}, Location: {trace['location']}")
            return False
    else:
        print("[WARN] No tool call detected")
        return False


async def main():
    # 환경 변수 로드
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        return False

    print("=" * 70)
    print("LLM Agent Integration Test")
    print("=" * 70)
    print()
    print("이 테스트는 LLM Agent가 자연어 prompt를 받아")
    print("적절한 tool을 선택하고, Scheduler가 location을 결정하는")
    print("End-to-End 흐름을 검증합니다.")
    print()

    # 테스트 환경 준비
    await setup_test_environment()

    # Middleware 초기화
    config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

    async with EdgeAgentMCPClient(config_path) as client:
        print(f"\nMiddleware 초기화 완료 (clients: {len(client.clients)})")

        # Tools 로드
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} proxy tools")

        # Tool 목록 (일부)
        print("\nAvailable tools:")
        for tool in sorted(tools, key=lambda t: t.name)[:5]:
            print(f"  - {tool.name}")
        print("  ...")

        # LLM 초기화
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print(f"\nLLM: {llm.model_name}")

        # Agent 생성
        agent = create_agent(llm, tools)
        print("Agent created with create_agent()")

        # 테스트 실행
        results = []

        results.append(("Single file read (DEVICE)", await test_single_file_read(agent, client)))
        results.append(("Multi-location comparison", await test_multi_location_comparison(agent, client)))
        results.append(("Cloud routing", await test_cloud_routing(agent, client)))
        results.append(("File write (EDGE)", await test_file_write(agent, client)))

        # 결과 요약
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

        # Execution Trace 전체 출력
        print("\n" + "=" * 70)
        print("Full Execution Trace")
        print("=" * 70)
        for i, trace in enumerate(client.execution_trace, 1):
            print(f"  {i}. {trace['tool']} → {trace['location']}")

        return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
