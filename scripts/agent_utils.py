"""
Agent Utilities for LLM Agent scenarios

Provides common utilities for agent execution including:
- Tool call logging callback
- Streaming agent execution with progress
"""

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish


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


async def run_agent_with_logging(agent, user_request: str, verbose: bool = True) -> dict:
    """
    Run agent with tool call logging.

    Args:
        agent: LangGraph agent (CompiledStateGraph)
        user_request: The user's request string
        verbose: Whether to print tool calls

    Returns:
        Agent result dictionary with messages
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
