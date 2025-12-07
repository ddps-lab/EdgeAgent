"""
Phase 0.2: 순수 LangChain Agent 동작 확인

MCP 없이 기본 LangChain tool과 agent가 정상 동작하는지 검증합니다.
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 환경 변수 로드
load_dotenv()

# ============================================================================
# Custom Tools 정의
# ============================================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    print(f"[TOOL CALL] add({a}, {b})")
    result = a + b
    print(f"[TOOL RESULT] {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    print(f"[TOOL CALL] multiply({a}, {b})")
    result = a * b
    print(f"[TOOL RESULT] {result}")
    return result


@tool
def divide(a: float, b: float) -> float:
    """Divide first number by second number.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of a / b
    """
    print(f"[TOOL CALL] divide({a}, {b})")
    if b == 0:
        return "Error: Cannot divide by zero"
    result = a / b
    print(f"[TOOL RESULT] {result}")
    return result


# ============================================================================
# Agent 설정 및 실행
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 0.2: 순수 LangChain Agent 동작 확인")
    print("=" * 80)
    print()

    # LLM 초기화
    print("[1] LLM 초기화...")
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        verbose=True
    )
    print("   ✓ ChatOpenAI initialized (model: gpt-4)")
    print()

    # Tools 준비
    print("[2] Tools 준비...")
    tools = [add, multiply, divide]
    for tool in tools:
        print(f"   - {tool.name}: {tool.description.split('.')[0]}")
    print()

    # ReAct Agent 생성
    print("[3] ReAct Agent 생성...")
    # LangChain의 create_agent 사용
    agent_executor = create_agent(
        llm,
        tools,
        system_prompt="You are a helpful assistant that can perform mathematical calculations."
    )
    print("   ✓ ReAct agent created")
    print()

    # ========================================================================
    # 테스트 케이스 실행
    # ========================================================================

    test_cases = [
        {
            "name": "Test 1: 단순 덧셈",
            "input": "What is 15 plus 27?"
        },
        {
            "name": "Test 2: 연속 연산",
            "input": "Calculate (10 + 5) multiplied by 3"
        },
        {
            "name": "Test 3: 복잡한 연산",
            "input": "What is (100 + 50) divided by 3?"
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
            result = agent_executor.invoke(
                {"messages": [("user", test_case['input'])]}
            )
            # LangGraph agent returns messages
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

    # ========================================================================
    # 결과 요약
    # ========================================================================

    print("\n\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')

    for result in results:
        status_icon = "✓" if result['status'] == 'PASS' else "✗"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if result['status'] == 'PASS':
            print(f"  → {result['output']}")
        else:
            print(f"  → Error: {result.get('error', 'Unknown error')}")

    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("=" * 80)
        print("✓ Phase 0.2 검증 완료: LangChain agent가 정상 동작합니다!")
        print("=" * 80)
        return True
    else:
        print("=" * 80)
        print("✗ Phase 0.2 검증 실패: 일부 테스트가 실패했습니다.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    # API key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        print("Example: OPENAI_API_KEY=sk-...")
        exit(1)

    success = main()
    exit(0 if success else 1)
