"""
Sub-Agent Module

특정 location에서 실행되는 Sub-Agent 정의.

Sub-Agent는 Main Agent로부터 task를 받아서
해당 location의 tool들만 사용하여 작업을 수행합니다.

기능:
- HTTP 서버로 동작 (FastAPI 기반)
- 특정 location의 MCP tools만 로드
- LangGraph 기반 ReAct agent 실행
- 결과를 Main Agent에게 반환

Usage (서버 실행):
    python -m edgeagent.subagent --location EDGE --port 8002

Usage (클라이언트):
    response = await client.post("http://localhost:8002/execute", json={
        "task": "Summarize the following data...",
        "context": {"input_data": "..."},
        "tools": ["summarize", "data_aggregate"]
    })
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from typing import TYPE_CHECKING
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .types import Location, LOCATIONS, ChainSchedulingResult
from .registry import ToolRegistry
from .planner import Partition
from .metrics import aggregate_metrics_entries

if TYPE_CHECKING:
    from .scheduler import BaseScheduler


class LLMLatencyTracker(BaseCallbackHandler):
    """
    SubAgent용 LLM 호출 시간 및 토큰 사용량 추적 콜백 핸들러.

    LLM 추론 시간을 측정하여 MetricsCollector에 전달합니다.
    """

    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        self._llm_start_time: Optional[float] = None
        self._llm_latencies: list[dict] = []

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

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        llm_latency = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self._llm_latencies.append(llm_latency)

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


@dataclass
class SubAgentRequest:
    """
    Sub-Agent 실행 요청

    Main Agent가 Sub-Agent에게 보내는 요청 형식
    """
    task: str                                    # 수행할 작업 설명
    context: dict[str, Any] = field(default_factory=dict)   # 이전 결과, 입력 데이터
    tools: list[str] = field(default_factory=list)          # 사용할 tool 목록
    tool_configs: dict[str, dict] = field(default_factory=dict)  # Tool 설정
    max_iterations: int = 10                     # 최대 반복 횟수

    @classmethod
    def from_dict(cls, data: dict) -> "SubAgentRequest":
        return cls(
            task=data.get("task", ""),
            context=data.get("context", {}),
            tools=data.get("tools", []),
            tool_configs=data.get("tool_configs", {}),
            max_iterations=data.get("max_iterations", 10),
        )

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "context": self.context,
            "tools": self.tools,
            "tool_configs": self.tool_configs,
            "max_iterations": self.max_iterations,
        }


@dataclass
class SubAgentResponse:
    """
    Sub-Agent 실행 결과

    Sub-Agent가 Main Agent에게 반환하는 결과 형식
    """
    success: bool
    result: Any = None                           # 최종 결과
    error: Optional[str] = None                  # 에러 메시지
    tool_calls: list[dict] = field(default_factory=list)    # 실행된 tool 호출 목록
    execution_time_ms: float = 0.0               # 실행 시간 (ms)
    metrics_entries: list[dict] = field(default_factory=list)  # 상세 메트릭 (MetricEntry.to_dict())
    # Aggregated metrics (이 partition의 합산 정보)
    aggregated: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "execution_time_ms": self.execution_time_ms,
            "metrics_entries": self.metrics_entries,
            "aggregated": self.aggregated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubAgentResponse":
        return cls(
            success=data.get("success", False),
            result=data.get("result"),
            error=data.get("error"),
            tool_calls=data.get("tool_calls", []),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            metrics_entries=data.get("metrics_entries", []),
            aggregated=data.get("aggregated", {}),
        )


class SubAgent:
    """
    Location별 Sub-Agent

    특정 location에서 실행되며, 해당 location의 tool들만 사용합니다.
    """

    def __init__(
        self,
        location: Location,
        config_path: str | Path,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        collect_metrics: bool = True,
        scheduler: "BaseScheduler | None" = None,
        chain_scheduling_result: Optional[ChainSchedulingResult] = None,
    ):
        """
        Args:
            location: 이 Sub-Agent가 담당하는 location
            config_path: YAML 설정 파일 경로
            model: LLM 모델 이름
            temperature: LLM temperature
            collect_metrics: 상세 메트릭 수집 여부
            scheduler: Orchestrator에서 전달받은 scheduler (location 결정에 사용)
            chain_scheduling_result: schedule_chain() 결과 (각 tool의 SchedulingResult 포함)
        """
        self.location = location
        self.config_path = Path(config_path)
        self.model = model
        self.temperature = temperature
        self.collect_metrics = collect_metrics
        self.scheduler = scheduler
        self.chain_scheduling_result = chain_scheduling_result

        # Registry 로드
        self.registry = ToolRegistry.from_yaml(config_path)

        # 이 location에서 사용 가능한 tool 목록
        self.available_tools = self._get_location_tools()

    def _get_location_tools(self) -> list[str]:
        """
        이 location에서 사용 가능한 개별 tool 목록 반환

        개별 tool 이름만 포함합니다 (서버 이름 제외).
        예: ["read_file", "write_file", "parse_logs", ...]
        """
        tools = []

        # 개별 tool 이름만 추가 (부모 서버가 이 location에서 사용 가능한 경우)
        for tool_name in self.registry.list_individual_tools():
            parent_server = self.registry.get_server_for_tool(tool_name)
            if parent_server:
                tool_config = self.registry.get_tool(parent_server)
                if tool_config and self.location in tool_config.available_locations():
                    tools.append(tool_name)

        return tools

    async def execute(
        self,
        request: SubAgentRequest,
        direct_call: bool = False,
        direct_call_args: dict | None = None,
    ) -> SubAgentResponse:
        """
        Task 실행

        LangGraph agent를 사용하여 주어진 task를 수행합니다.
        direct_call=True이면 LLM 없이 직접 tool을 호출합니다.

        Args:
            request: SubAgentRequest 객체
            direct_call: True이면 LLM 우회하고 직접 tool 호출
            direct_call_args: 직접 호출 시 사용할 tool arguments

        Returns:
            SubAgentResponse 객체
        """
        import time
        start_time = time.time()
        tool_calls = []
        metrics_entries = []

        try:
            # 요청된 tool이 이 location에서 사용 가능한지 확인
            requested_tools = request.tools if request.tools else self.available_tools
            valid_tools = [t for t in requested_tools if t in self.available_tools]

            if not valid_tools:
                return SubAgentResponse(
                    success=False,
                    error=f"No valid tools available at {self.location}. "
                          f"Requested: {requested_tools}, Available: {self.available_tools}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # 직접 tool call 모드 (단일 tool, LLM 우회)
            if direct_call and len(valid_tools) == 1:
                result, collected_metrics = await self._run_direct_tool_call(
                    request, valid_tools[0], direct_call_args or {}, tool_calls
                )
                metrics_entries.extend(collected_metrics)
            else:
                # LangChain agent 생성 및 실행
                result, collected_metrics = await self._run_agent(request, valid_tools, tool_calls)
                metrics_entries.extend(collected_metrics)

            # Aggregate metrics before returning
            aggregated = aggregate_metrics_entries(metrics_entries)

            return SubAgentResponse(
                success=True,
                result=result,
                tool_calls=tool_calls,
                execution_time_ms=(time.time() - start_time) * 1000,
                metrics_entries=metrics_entries,
                aggregated=aggregated,
            )

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            return SubAgentResponse(
                success=False,
                error=error_detail,
                tool_calls=tool_calls,
                execution_time_ms=(time.time() - start_time) * 1000,
                aggregated={},  # 실패 시 빈 dict
            )

    async def _run_direct_tool_call(
        self,
        request: SubAgentRequest,
        tool_name: str,
        args: dict,
        tool_calls: list[dict],
    ) -> tuple[Any, list[dict]]:
        """
        LLM 없이 직접 tool 호출 (단일 tool 최적화)

        Args:
            request: 실행 요청
            tool_name: 호출할 tool 이름
            args: Tool arguments (비어있으면 context에서 추출)
            tool_calls: Tool 호출 기록 (output)

        Returns:
            (Tool 실행 결과, 메트릭 entries 목록)
        """
        from .middleware import EdgeAgentMCPClient
        from contextlib import AsyncExitStack

        result_str = None
        client = None
        exit_stack = AsyncExitStack()
        metrics_entries = []

        try:
            await exit_stack.__aenter__()
            # 요청된 tools만 필터링하여 연결 (동적 연결)
            client = EdgeAgentMCPClient(
                self.config_path,
                collect_metrics=self.collect_metrics,
                filter_tools=[tool_name],  # 단일 tool만 연결
                filter_location=self.location,  # 현재 location의 endpoint만 사용
                scheduler=self.scheduler,  # Orchestrator의 scheduler 사용
            )
            # Chain Scheduling 결과 설정 (원래 계산된 score, reason 등 사용)
            if self.chain_scheduling_result:
                client.set_chain_scheduling_result(self.chain_scheduling_result)
            await exit_stack.enter_async_context(client)

            all_tools = await client.get_tools()

            # 전달된 tool_name으로 직접 tool 찾기
            # (기존: MCP 서버 이름 → _select_best_tool로 선택)
            # (변경: 구체적 tool 이름이 전달됨 → 직접 매칭)
            target_tool = None
            for t in all_tools:
                if t.name == tool_name:
                    target_tool = t
                    break

            if not target_tool:
                raise ValueError(f"Tool '{tool_name}' not found. Available: {[t.name for t in all_tools]}")

            # Arguments 결정 (tool_name이 이제 구체적 tool 이름으로 전달됨)
            if not args:
                args = self._extract_args_from_context(tool_name, request)

            # 직접 호출 (_select_best_tool() 삭제됨)
            result = await target_tool.ainvoke(args)
            result_str = str(result)

            tool_calls.append({
                "tool": target_tool.name,
                "input": args,
                "output": result_str[:500] if len(result_str) > 500 else result_str,
            })

            # 메트릭 수집
            if self.collect_metrics and client.get_metrics():
                metrics = client.get_metrics()
                metrics_entries = [e.to_dict() for e in metrics.entries]

        finally:
            # MCP STDIO 클라이언트 종료 시 TaskGroup 에러 무시
            try:
                await exit_stack.__aexit__(None, None, None)
            except (BaseExceptionGroup, ExceptionGroup):
                pass  # MCP STDIO cleanup errors are expected

        return result_str, metrics_entries

    def _extract_args_from_context(self, tool_name: str, request: SubAgentRequest) -> dict:
        """Context에서 tool arguments 추출

        Args:
            tool_name: 구체적 tool 이름 (예: "read_text_file", "write_file", "parse_logs")
            request: SubAgentRequest 객체
        """
        import re
        args = {}

        # 이전 결과 가져오기
        prev_result = request.context.get("previous_result", "")
        task = request.task

        # tool_name에서 server_name 매핑 (기존 분기 로직 재사용)
        server_name = self.registry.get_server_for_tool(tool_name) or tool_name

        if server_name == "filesystem":
            # Path 추출 - /edgeagent/{data,repos,results} 경로만 허용
            path_pattern = r'/edgeagent/[\w\._/-]+(?:\.\w+)?'
            match = re.search(path_pattern, task)
            if match:
                path = match.group(0).rstrip('.')
                args["path"] = path
            else:
                args["path"] = "/edgeagent/data"

            # Write 체크 (tool_name에서 직접 판단)
            is_write_tool = "write" in tool_name.lower()
            if prev_result and (is_write_tool or "report" in task.lower()):
                write_match = re.search(r'(?:to|write)\s+([/\w\._-]+\.(?:md|txt|json))', task, re.IGNORECASE)
                if write_match:
                    args["path"] = write_match.group(1)
                    if not args["path"].startswith("/edgeagent"):
                        args["path"] = "/edgeagent/results/" + args["path"].lstrip("/")
                elif is_write_tool:
                    args["path"] = "/edgeagent/results/output.txt"
                args["content"] = str(prev_result)

        elif server_name == "log_parser":
            if prev_result:
                args["text"] = str(prev_result)[:10000]

        elif server_name == "git":
            repo_match = re.search(r'/edgeagent/repos/[\w\._/-]+', task)
            args["repo_path"] = repo_match.group(0) if repo_match else "/edgeagent/repos/scenario1"

        elif server_name == "summarize":
            if prev_result:
                args["text"] = str(prev_result)[:5000]
                args["max_length"] = 200

        elif server_name == "data_aggregate":
            if prev_result:
                # combine_research_results는 'results' 파라미터 사용
                if tool_name == "combine_research_results":
                    if isinstance(prev_result, list):
                        args["results"] = prev_result
                    else:
                        args["results"] = [{"title": "Result", "summary": str(prev_result)[:1000]}]
                else:
                    # 다른 data_aggregate tools는 'items' 사용
                    if isinstance(prev_result, list):
                        args["items"] = prev_result
                    else:
                        args["items"] = [{"data": str(prev_result)[:1000]}]
                    args["group_by"] = "type"

        elif server_name == "fetch":
            url_match = re.search(r'https?://[^\s\"\'\`]+', task)
            if url_match:
                args["url"] = url_match.group(0)

        elif server_name == "image":
            path_match = re.search(r'/edgeagent/[\w\._/-]+', task)
            if path_match:
                args["path"] = path_match.group(0)

        return args

    async def _run_agent(
        self,
        request: SubAgentRequest,
        valid_tools: list[str],
        tool_calls: list[dict],
    ) -> tuple[Any, list[dict]]:
        """
        LangGraph agent 실행

        EdgeAgentMCPClient를 사용하여 location-specific tool들만 로드합니다.

        Args:
            request: 실행 요청
            valid_tools: 사용 가능한 tool 목록
            tool_calls: Tool 호출 기록 (output)

        Returns:
            (Agent 실행 결과, 메트릭 entries 목록)
        """
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent
        from .middleware import EdgeAgentMCPClient
        from contextlib import AsyncExitStack

        metrics_entries = []
        result = ({"messages": []}, [])  # 기본값
        exit_stack = AsyncExitStack()

        try:
            await exit_stack.__aenter__()

            # EdgeAgentMCPClient를 사용하여 tools 로드 (요청된 tools만 동적 연결)
            client = EdgeAgentMCPClient(
                self.config_path,
                collect_metrics=self.collect_metrics,
                filter_tools=valid_tools,  # 요청된 tool만 연결
                filter_location=self.location,  # 현재 location의 endpoint만 사용
                scheduler=self.scheduler,  # Orchestrator의 scheduler 사용
            )
            # Chain Scheduling 결과 설정 (원래 계산된 score, reason 등 사용)
            if self.chain_scheduling_result:
                client.set_chain_scheduling_result(self.chain_scheduling_result)
            await exit_stack.enter_async_context(client)

            # LLM latency tracker 생성 (MetricsCollector 연결)
            llm_tracker = LLMLatencyTracker(
                metrics_collector=client.get_metrics() if self.collect_metrics else None
            )

            # LLM 초기화 (gpt-5-mini는 temperature=0 지원 안 함)
            llm_kwargs = {"model": self.model, "callbacks": [llm_tracker]}
            if "gpt-5" not in self.model:
                llm_kwargs["temperature"] = self.temperature
            llm = ChatOpenAI(**llm_kwargs)

            # 필터링된 tools 가져오기
            all_tools = await client.get_tools()

            # 요청된 tool만 필터링 (엄격하게)
            # valid_tools는 논리적 tool 이름 (예: "filesystem", "log_parser")
            # MCP tool 이름과 매핑 필요
            tools = []
            tool_name_mapping = {}  # MCP tool name -> parent tool name

            # MCP 응답 형식 문제가 있는 툴 제외
            TOOL_BLACKLIST = {
                "list_directory_with_sizes",  # returns array instead of string
                "directory_tree",  # causes issues with gpt-5-mini (array vs string)
            }

            for t in all_tools:
                # 블랙리스트 체크
                if t.name in TOOL_BLACKLIST:
                    continue
                parent_name = getattr(t, 'parent_tool_name', None)
                mcp_tool_name = t.name

                # 1. MCP tool 이름이 valid_tools에 있으면 포함 (개별 tool 이름 매칭)
                if mcp_tool_name in valid_tools:
                    tools.append(t)
                    tool_name_mapping[mcp_tool_name] = parent_name or mcp_tool_name
                # 2. parent_tool_name이 valid_tools에 있으면 포함 (서버 이름 매칭)
                elif parent_name and parent_name in valid_tools:
                    tools.append(t)
                    tool_name_mapping[mcp_tool_name] = parent_name
                # 3. parent_tool_name이 없으면 tool 이름 자체로 매칭 시도
                elif parent_name is None:
                    # MCP tool 이름에서 parent를 추론 (예: read_text_file -> filesystem)
                    for vt in valid_tools:
                        if vt in mcp_tool_name or mcp_tool_name.startswith(vt):
                            tools.append(t)
                            tool_name_mapping[mcp_tool_name] = vt
                            break

            if not tools:
                raise ValueError(f"No tools loaded for: {valid_tools}. Available: {[t.name for t in all_tools]}")

            # Context를 포함한 system prompt 생성
            context_str = ""
            if request.context:
                # Context가 너무 크면 요약
                ctx_str = json.dumps(request.context, indent=2)
                if len(ctx_str) > 2000:
                    ctx_str = ctx_str[:2000] + "... (truncated)"
                context_str = f"\n\nData from previous steps (use this as input):\n{ctx_str}"

            # 실제 사용 가능한 tool 이름 목록 (MCP tool names)
            available_tool_names = [t.name for t in tools]

            # Tool descriptions for better selection
            tool_descriptions = []
            for t in tools:
                desc = getattr(t, 'description', '')[:100] if hasattr(t, 'description') else ''
                tool_descriptions.append(f"- {t.name}: {desc}")

            system_prompt = (
                f"You are a specialized assistant at {self.location} location.\n\n"
                f"AVAILABLE TOOLS:\n"
                f"{chr(10).join(tool_descriptions)}\n\n"
                f"CRITICAL RULES:\n"
                f"1. ONLY use tools from the list above.\n"
                f"2. ALWAYS provide ALL required parameters when calling a tool.\n"
                f"3. IMPORTANT: Pass the output from one tool as input to the next tool.\n"
                f"   - For log analysis: parse_logs returns 'entries', pass these entries to compute_log_statistics(entries=...)\n"
                f"   - For image processing: batch_resize returns results, pass these to aggregate_list(items=...)\n"
                f"   - aggregate_list REQUIRES 'items' parameter (a list of objects to aggregate)\n"
                f"   - Example: aggregate_list(items=[...results from previous tool...], group_by='format')\n"
                f"4. Do NOT use merge_summaries or combine_research_results unless explicitly asked.\n"
                f"5. Execute ALL steps mentioned in the task. Do NOT stop early.\n"
                f"6. If processing multiple items, process ALL of them.\n"
                f"7. NEVER call a tool with missing required parameters.\n"
                f"{context_str}"
            )

            # LangGraph Agent 생성 (recursion_limit 증가)
            agent = create_agent(llm, tools, system_prompt=system_prompt)

            # Agent 실행 (astream으로 실행하여 tool calls 추적)
            # recursion_limit을 높여서 많은 tool call 허용 (기본값 25)
            all_messages = []
            async for chunk in agent.astream(
                {"messages": [("user", request.task)]},
                stream_mode="values",
                config={"recursion_limit": 100},  # 많은 반복 작업 지원
            ):
                if "messages" in chunk:
                    all_messages = chunk["messages"]

                    # Tool calls 추적
                    for msg in all_messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tc_id = tc.get("id", "")
                                if tc_id and not any(t.get("id") == tc_id for t in tool_calls):
                                    tool_calls.append({
                                        "id": tc_id,
                                        "tool": tc.get("name", "unknown"),
                                        "input": tc.get("args", {}),
                                    })

                        # Tool 결과 추적
                        if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
                            for t in tool_calls:
                                if t.get("id") == msg.tool_call_id and "output" not in t:
                                    t["output"] = str(msg.content)[:500]

            # 메트릭 수집
            if self.collect_metrics and client.get_metrics():
                metrics = client.get_metrics()
                metrics_entries = [e.to_dict() for e in metrics.entries]

            # 최종 메시지 추출
            if all_messages:
                final_msg = all_messages[-1]
                if hasattr(final_msg, "content"):
                    result = (final_msg.content, metrics_entries)
                else:
                    result = ({"messages": all_messages}, metrics_entries)
            else:
                result = ({"messages": all_messages}, metrics_entries)

        finally:
            # MCP STDIO 클라이언트 종료 시 TaskGroup 에러 무시
            try:
                await exit_stack.__aexit__(None, None, None)
            except (BaseExceptionGroup, ExceptionGroup):
                pass  # MCP STDIO cleanup errors

        return result


def create_subagent_server(
    location: Location,
    config_path: str | Path,
    model: str = "gpt-4o-mini",
):
    """
    Sub-Agent HTTP 서버 생성 (FastAPI)

    Args:
        location: Sub-Agent location
        config_path: YAML 설정 파일 경로
        model: LLM 모델

    Returns:
        FastAPI 앱 인스턴스
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title=f"EdgeAgent Sub-Agent ({location})",
        description=f"Sub-Agent server for {location} location",
    )

    subagent = SubAgent(
        location=location,
        config_path=config_path,
        model=model,
    )

    class ExecuteRequest(BaseModel):
        task: str
        context: dict = {}
        tools: list[str] = []
        tool_configs: dict = {}
        max_iterations: int = 10

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "location": location,
            "available_tools": subagent.available_tools,
        }

    @app.get("/tools")
    async def list_tools():
        return {
            "location": location,
            "tools": subagent.available_tools,
        }

    @app.post("/execute")
    async def execute(request: ExecuteRequest):
        subagent_request = SubAgentRequest(
            task=request.task,
            context=request.context,
            tools=request.tools,
            tool_configs=request.tool_configs,
            max_iterations=request.max_iterations,
        )

        response = await subagent.execute(subagent_request)

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return response.to_dict()

    return app


async def run_subagent_server(
    location: Location,
    config_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str = "gpt-4o-mini",
):
    """
    Sub-Agent 서버 실행

    Args:
        location: Sub-Agent location
        config_path: YAML 설정 파일 경로
        host: 서버 호스트
        port: 서버 포트
        model: LLM 모델
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Uvicorn is required. Install with: pip install uvicorn")

    app = create_subagent_server(location, config_path, model)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeAgent Sub-Agent Server")
    parser.add_argument(
        "--location",
        type=str,
        choices=LOCATIONS,
        required=True,
        help="Sub-Agent location (DEVICE, EDGE, CLOUD)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tools.yaml",
        help="Path to tools YAML config file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name",
    )

    args = parser.parse_args()

    asyncio.run(run_subagent_server(
        location=args.location,
        config_path=args.config,
        host=args.host,
        port=args.port,
        model=args.model,
    ))
