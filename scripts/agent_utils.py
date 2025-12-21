"""
Agent Utilities for LLM Agent scenarios

Provides common utilities for agent execution including:
- Tool call logging callback
- LLM latency tracking callback
- Streaming agent execution with progress
"""

import time
from typing import Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult


class LLMLatencyTracker(BaseCallbackHandler):
    """
    LLM 호출 시간 및 토큰 사용량을 추적하는 콜백 핸들러.

    Agent 모드에서 LLM 추론 시간을 측정하여 MetricsCollector에 전달합니다.
    """

    def __init__(self, metrics_collector=None):
        """
        Args:
            metrics_collector: MetricsCollector 인스턴스 (optional)
        """
        self.metrics_collector = metrics_collector
        self._llm_start_time: Optional[float] = None
        self._llm_latencies: list[dict] = []
        self._current_llm_latency: Optional[dict] = None

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """LLM 호출 시작"""
        self._llm_start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 호출 종료 - latency 및 토큰 사용량 기록"""
        if self._llm_start_time is None:
            return

        end_time = time.perf_counter()
        latency_ms = (end_time - self._llm_start_time) * 1000

        # 토큰 정보 추출
        input_tokens = 0
        output_tokens = 0

        # LLMResult에서 토큰 정보 추출 시도
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        self._current_llm_latency = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self._llm_latencies.append(self._current_llm_latency)

        # MetricsCollector에 pending latency 설정
        if self.metrics_collector:
            self.metrics_collector.set_pending_llm_latency(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        self._llm_start_time = None

    def get_total_llm_latency_ms(self) -> float:
        """총 LLM latency 반환"""
        return sum(l.get("latency_ms", 0) for l in self._llm_latencies)

    def get_total_tokens(self) -> tuple[int, int]:
        """총 토큰 사용량 반환 (input, output)"""
        input_tokens = sum(l.get("input_tokens", 0) for l in self._llm_latencies)
        output_tokens = sum(l.get("output_tokens", 0) for l in self._llm_latencies)
        return input_tokens, output_tokens

    def get_llm_call_count(self) -> int:
        """LLM 호출 횟수 반환"""
        return len(self._llm_latencies)


class ToolCallLogger(BaseCallbackHandler):
    """Callback handler that logs tool calls during agent execution."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tool_calls = []

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent decides to use a tool."""
        if self.verbose:
            print(f"  [TOOL] {action.tool}")
            if action.tool_input:
                # Truncate long inputs
                input_str = str(action.tool_input)
                if len(input_str) > 200:
                    input_str = input_str[:200] + "..."
                print(f"         Input: {input_str}")
        self.tool_calls.append({
            "tool": action.tool,
            "input": action.tool_input,
        })

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        if self.verbose:
            print(f"  [START] {tool_name}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        if self.verbose:
            output_preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
            print(f"  [DONE] Output: {output_preview}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool errors."""
        if self.verbose:
            print(f"  [ERROR] {error}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes."""
        if self.verbose:
            print(f"  [FINISH] Agent completed with {len(self.tool_calls)} tool calls")


def create_llm_with_latency_tracking(
    model: str,
    temperature: float = 0,
    metrics_collector=None,
):
    """
    LLM latency tracking이 설정된 ChatOpenAI 인스턴스 생성

    LangGraph에서 LLM latency를 추적하려면 LLM 생성 시 callback을 등록해야 합니다.

    Args:
        model: OpenAI 모델 이름
        temperature: Temperature 설정
        metrics_collector: MetricsCollector 인스턴스

    Returns:
        ChatOpenAI 인스턴스 (callback 등록됨)
    """
    from langchain_openai import ChatOpenAI

    llm_tracker = LLMLatencyTracker(metrics_collector=metrics_collector)

    llm_kwargs = {"model": model}
    if "gpt-5" not in model:
        llm_kwargs["temperature"] = temperature
    llm_kwargs["seed"] = 42  # Ensure deterministic behavior

    # LLM 생성 시 callback 직접 등록 (LangGraph에서 제대로 전파됨)
    llm_kwargs["callbacks"] = [llm_tracker]

    return ChatOpenAI(**llm_kwargs)


async def run_agent_with_logging(
    agent,
    user_request: str,
    verbose: bool = True,
    metrics_collector=None,
) -> dict:
    """
    Run agent with tool call logging and LLM latency tracking.

    Args:
        agent: LangGraph agent (CompiledStateGraph)
        user_request: The user's request string
        verbose: Whether to print tool calls
        metrics_collector: MetricsCollector 인스턴스 (optional, 하위 호환성용)

    Returns:
        Agent result dictionary with messages

    Note:
        LLM latency 추적을 위해서는 create_llm_with_latency_tracking()으로
        LLM을 생성해야 합니다. 이 함수의 metrics_collector 파라미터는
        하위 호환성을 위해 유지되지만, 실제 LLM latency 추적에는 사용되지 않습니다.
    """
    if verbose:
        print()

    # For LangGraph agents, we use astream with "values" mode to get full state
    # This avoids duplicate execution that happens with ainvoke after astream
    tool_calls = []
    all_messages = []

    async for chunk in agent.astream(
        {"messages": [("user", user_request)]},
        stream_mode="values",
    ):
        # In "values" mode, chunk contains the full state
        if "messages" in chunk:
            messages = chunk["messages"]
            all_messages = messages  # Keep updating with latest state

            # Log new tool-related messages
            for msg in messages:
                # Check for AI message with tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_id = tc.get("id", "")
                        # Only log if we haven't seen this tool call
                        if tc_id and not any(t.get("id") == tc_id for t in tool_calls):
                            tool_name = tc.get("name", "unknown")
                            tool_input = tc.get("args", {})
                            if verbose:
                                input_str = str(tool_input)
                                if len(input_str) > 150:
                                    input_str = input_str[:150] + "..."
                                print(f"  [CALL] {tool_name}")
                                print(f"         Input: {input_str}")
                            tool_calls.append({"id": tc_id, "name": tool_name, "input": tool_input})

                # Check for tool result messages
                if hasattr(msg, "name") and hasattr(msg, "tool_call_id"):
                    tool_call_id = getattr(msg, "tool_call_id", "")
                    # Check if we already logged this result
                    logged_ids = [t.get("result_logged") for t in tool_calls if t.get("result_logged")]
                    if tool_call_id and tool_call_id not in logged_ids:
                        tool_name = msg.name
                        tool_output = str(msg.content) if hasattr(msg, "content") else ""
                        if verbose:
                            output_preview = tool_output[:150] + "..." if len(tool_output) > 150 else tool_output
                            print(f"  [DONE] {tool_name}")
                            print(f"         Output: {output_preview}")
                        # Mark as logged
                        for t in tool_calls:
                            if t.get("id") == tool_call_id:
                                t["result_logged"] = tool_call_id
                                t["output"] = tool_output
                                break

    if verbose:
        print()
        print(f"  Total tool calls: {len(tool_calls)}")

    return {"messages": all_messages}
