"""
Sub-Agent Orchestrator

Main Agent에서 Sub-Agent들을 조율하여 분산 실행을 관리합니다.

기능:
- Tool sequence를 location별로 분할 (Planner 사용)
- 각 location의 Sub-Agent에게 작업 위임
- 결과를 수집하여 다음 Sub-Agent에게 전달
- Legacy 모드와 Sub-Agent 모드 선택 지원

Usage:
    orchestrator = SubAgentOrchestrator(config_path)
    result = await orchestrator.run(
        user_request="Analyze the log file...",
        tool_sequence=["filesystem", "log_parser", "summarize"],
        mode="subagent"  # or "legacy"
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import yaml

from .types import Location, LOCATIONS
from .registry import ToolRegistry
from .scheduler import StaticScheduler
from .planner import ToolSequencePlanner, ExecutionPlan, Partition
from .subagent import SubAgentRequest, SubAgentResponse


OrchestrationMode = Literal["legacy", "subagent"]


@dataclass
class SubAgentEndpoint:
    """Sub-Agent HTTP endpoint 설정"""
    host: str = "localhost"
    port: int = 8000
    timeout: float = 300.0  # 5분

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class OrchestrationConfig:
    """
    Orchestration 설정

    YAML 파일에서 로드하거나 직접 생성 가능
    """
    mode: OrchestrationMode = "legacy"
    subagent_endpoints: dict[Location, SubAgentEndpoint] = field(default_factory=dict)
    model: str = "gpt-4o-mini"
    temperature: float = 0
    max_iterations: int = 10

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "OrchestrationConfig":
        """YAML 파일에서 설정 로드"""
        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        endpoints = {}
        for loc, ep_data in data.get("subagent_endpoints", {}).items():
            if loc in LOCATIONS:
                endpoints[loc] = SubAgentEndpoint(
                    host=ep_data.get("host", "localhost"),
                    port=ep_data.get("port", 8000),
                    timeout=ep_data.get("timeout", 300.0),
                )

        return cls(
            mode=data.get("orchestration_mode", "legacy"),
            subagent_endpoints=endpoints,
            model=data.get("model", "gpt-4o-mini"),
            temperature=data.get("temperature", 0),
            max_iterations=data.get("max_iterations", 10),
        )


@dataclass
class OrchestrationResult:
    """
    Orchestration 실행 결과

    전체 실행 결과와 각 partition의 세부 결과를 포함
    """
    success: bool
    final_result: Any = None
    error: Optional[str] = None
    mode: OrchestrationMode = "legacy"
    partitions_executed: int = 0
    total_tool_calls: int = 0
    execution_time_ms: float = 0.0
    partition_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_result": self.final_result,
            "error": self.error,
            "mode": self.mode,
            "partitions_executed": self.partitions_executed,
            "total_tool_calls": self.total_tool_calls,
            "execution_time_ms": self.execution_time_ms,
            "partition_results": self.partition_results,
        }


class SubAgentOrchestrator:
    """
    Sub-Agent 조율자

    Main Agent에서 Sub-Agent들을 조율하여 분산 실행을 관리합니다.
    """

    def __init__(
        self,
        tools_config_path: str | Path,
        orchestration_config: Optional[OrchestrationConfig] = None,
    ):
        """
        Args:
            tools_config_path: Tool 설정 YAML 경로
            orchestration_config: Orchestration 설정 (None이면 기본값)
        """
        self.tools_config_path = Path(tools_config_path)

        # Tool 관련 초기화
        self.registry = ToolRegistry.from_yaml(tools_config_path)
        self.scheduler = StaticScheduler(tools_config_path, self.registry)
        self.planner = ToolSequencePlanner(self.scheduler, self.registry)

        # Orchestration 설정
        self.config = orchestration_config or OrchestrationConfig()

        # HTTP client for Sub-Agent 호출
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Context manager entry"""
        self._http_client = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def run(
        self,
        user_request: str,
        tool_sequence: Optional[list[str]] = None,
        mode: Optional[OrchestrationMode] = None,
        initial_context: Optional[dict] = None,
    ) -> OrchestrationResult:
        """
        Orchestration 실행

        Args:
            user_request: 사용자 요청 문자열
            tool_sequence: 사용할 tool 목록 (None이면 모든 tool)
            mode: 실행 모드 (None이면 config 기본값)
            initial_context: 초기 context

        Returns:
            OrchestrationResult
        """
        start_time = time.time()
        mode = mode or self.config.mode

        # Tool sequence가 없으면 모든 tool 사용
        if tool_sequence is None:
            tool_sequence = self.registry.list_tools()

        try:
            if mode == "subagent":
                result = await self._run_subagent_mode(
                    user_request, tool_sequence, initial_context or {}
                )
            else:
                result = await self._run_legacy_mode(
                    user_request, tool_sequence, initial_context or {}
                )

            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            return OrchestrationResult(
                success=False,
                error=str(e),
                mode=mode,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _run_subagent_mode(
        self,
        user_request: str,
        tool_sequence: list[str],
        context: dict,
    ) -> OrchestrationResult:
        """
        Sub-Agent 모드 실행

        Tool sequence를 location별로 분할하고,
        각 Sub-Agent에게 순차적으로 작업을 위임합니다.

        최적화: 단일 tool partition은 LLM 없이 직접 tool call
        """
        # 실행 계획 생성
        plan = self.planner.create_execution_plan(tool_sequence, preserve_order=True)

        if not plan.partitions:
            return OrchestrationResult(
                success=False,
                error="No partitions created from tool sequence",
                mode="subagent",
            )

        partition_results = []
        current_context = context.copy()
        total_tool_calls = 0

        # Partition별 sub-task 생성
        partition_tasks = self._generate_partition_tasks(
            user_request=user_request,
            partitions=plan.partitions,
            tool_sequence=tool_sequence,
        )

        # 각 partition 순차 실행
        for i, partition in enumerate(plan.partitions):
            # Partition용 task 생성 (구체적인 sub-task 사용)
            partition_sub_task = partition_tasks[i] if i < len(partition_tasks) else ""

            if i == 0:
                task = (
                    f"YOUR TASK: {partition_sub_task}\n\n"
                    f"CONTEXT (for reference only): {user_request}\n\n"
                    f"AVAILABLE TOOLS: {', '.join(partition.tools)}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Focus ONLY on your specific task above.\n"
                    f"2. Use ALL available tools as needed to complete your task.\n"
                    f"3. If the task mentions processing multiple items, process ALL of them.\n"
                    f"4. Return all results in a structured format."
                )
            else:
                # 이전 결과를 참조하도록 구체적인 지시
                prev_result = current_context.get("previous_result", "")
                prev_result_str = str(prev_result)

                task = (
                    f"YOUR TASK: {partition_sub_task}\n\n"
                    f"INPUT DATA FROM PREVIOUS STEP:\n"
                    f"```\n{prev_result_str[:5000]}\n```\n\n"
                    f"AVAILABLE TOOLS: {', '.join(partition.tools)}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Use the INPUT DATA above to complete your task.\n"
                    f"2. If processing multiple items from INPUT DATA, process ALL of them.\n"
                    f"3. Use ALL available tools as needed.\n"
                    f"4. Return all results in a structured format."
                )

            # 단일 tool partition인 경우 직접 tool call (LLM 우회)
            use_direct_call = len(partition.tools) == 1

            # Sub-Agent 호출 (direct_call=True면 LLM 우회)
            response = await self._call_subagent(
                location=partition.location,
                task=task,
                context=current_context,
                tools=partition.tools,
                direct_call=use_direct_call,
            )

            partition_result = {
                "partition_index": i,
                "location": partition.location,
                "tools": partition.tools,
                "success": response.success,
                "tool_calls": response.tool_calls,
                "execution_time_ms": response.execution_time_ms,
                "metrics_entries": response.metrics_entries,  # 상세 tool 메트릭
            }

            if not response.success:
                partition_result["error"] = response.error
                partition_results.append(partition_result)
                return OrchestrationResult(
                    success=False,
                    error=f"Partition {i} ({partition.location}) failed: {response.error}",
                    mode="subagent",
                    partitions_executed=i + 1,
                    total_tool_calls=total_tool_calls,
                    partition_results=partition_results,
                )

            partition_result["result"] = response.result
            partition_results.append(partition_result)

            # Context 업데이트 (다음 partition으로 전달)
            # Tool call 결과에서 실제 output 데이터 추출
            last_tool_output = None
            if response.tool_calls:
                for tc in reversed(response.tool_calls):
                    if "output" in tc and tc["output"]:
                        last_tool_output = tc["output"]
                        break

            current_context["previous_result"] = last_tool_output or response.result
            current_context[f"partition_{i}_result"] = response.result
            current_context[f"partition_{i}_tool_calls"] = response.tool_calls
            total_tool_calls += len(response.tool_calls)

        # 최종 결과
        return OrchestrationResult(
            success=True,
            final_result=current_context.get("previous_result"),
            mode="subagent",
            partitions_executed=len(plan.partitions),
            total_tool_calls=total_tool_calls,
            partition_results=partition_results,
        )

    async def _run_legacy_mode(
        self,
        user_request: str,
        tool_sequence: list[str],
        context: dict,
    ) -> OrchestrationResult:
        """
        Legacy 모드 실행

        기존 방식처럼 단일 Agent가 모든 tool을 직접 호출합니다.
        EdgeAgentMCPClient를 사용합니다.
        """
        from .middleware import EdgeAgentMCPClient
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent

        async with EdgeAgentMCPClient(self.tools_config_path) as client:
            tools = await client.get_tools()

            # 요청된 tool만 필터링
            if tool_sequence:
                tool_names = set()
                for t in tool_sequence:
                    tool_config = self.registry.get_tool(t)
                    if tool_config:
                        tool_names.add(t)

                tools = [t for t in tools if getattr(t, 'parent_tool_name', t.name) in tool_names
                        or t.name in tool_names]

            if not tools:
                return OrchestrationResult(
                    success=False,
                    error="No tools available",
                    mode="legacy",
                )

            # LLM 및 Agent 생성
            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
            )

            context_str = ""
            if context:
                context_str = f"\n\nContext:\n{context}"

            system_prompt = f"You are a helpful assistant.{context_str}"

            # LangGraph Agent 생성
            agent = create_agent(llm, tools, system_prompt=system_prompt)

            # Agent 실행 (astream으로 실행)
            tool_calls = []
            all_messages = []

            async for chunk in agent.astream(
                {"messages": [("user", user_request)]},
                stream_mode="values",
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

            # 최종 결과 추출
            final_result = None
            if all_messages:
                final_msg = all_messages[-1]
                if hasattr(final_msg, "content"):
                    final_result = final_msg.content

            return OrchestrationResult(
                success=True,
                final_result=final_result,
                mode="legacy",
                partitions_executed=1,
                total_tool_calls=len(tool_calls),
                partition_results=[{
                    "partition_index": 0,
                    "location": "ALL",
                    "tools": [t.name for t in tools],
                    "success": True,
                    "tool_calls": tool_calls,
                }],
            )

    async def _call_subagent(
        self,
        location: Location,
        task: str,
        context: dict,
        tools: list[str],
        direct_call: bool = False,
    ) -> SubAgentResponse:
        """
        Sub-Agent HTTP 호출

        Args:
            location: 대상 location
            task: 수행할 작업
            context: Context 데이터
            tools: 사용할 tool 목록
            direct_call: True이면 LLM 우회하고 직접 tool 호출

        Returns:
            SubAgentResponse
        """
        endpoint = self.config.subagent_endpoints.get(location)
        if not endpoint:
            # Endpoint가 설정되지 않은 경우 로컬 실행
            return await self._execute_locally(location, task, context, tools, direct_call)

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=endpoint.timeout)

        request = SubAgentRequest(
            task=task,
            context=context,
            tools=tools,
            max_iterations=self.config.max_iterations,
        )

        try:
            response = await self._http_client.post(
                f"{endpoint.base_url}/execute",
                json=request.to_dict(),
                timeout=endpoint.timeout,
            )
            response.raise_for_status()
            return SubAgentResponse.from_dict(response.json())

        except httpx.HTTPStatusError as e:
            return SubAgentResponse(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except httpx.RequestError as e:
            return SubAgentResponse(
                success=False,
                error=f"Request failed: {str(e)}",
            )

    async def _execute_locally(
        self,
        location: Location,
        task: str,
        context: dict,
        tools: list[str],
        direct_call: bool = False,
    ) -> SubAgentResponse:
        """
        로컬에서 Sub-Agent 실행 (HTTP 서버 없이)

        테스트 또는 단일 머신 실행 시 사용

        Args:
            direct_call: True이면 LLM 우회하고 직접 tool 호출 (단일 tool 최적화)
        """
        from .subagent import SubAgent

        subagent = SubAgent(
            location=location,
            config_path=self.tools_config_path,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        request = SubAgentRequest(
            task=task,
            context=context,
            tools=tools,
            max_iterations=self.config.max_iterations,
        )

        return await subagent.execute(request, direct_call=direct_call)

    def _generate_partition_tasks(
        self,
        user_request: str,
        partitions: list[Partition],
        tool_sequence: list[str],
    ) -> list[str]:
        """
        각 partition에 대한 구체적인 sub-task 생성

        user_request를 분석하여 각 partition의 tools에 해당하는
        작업만 추출하여 구체적인 지시사항을 생성합니다.

        Args:
            user_request: 전체 사용자 요청
            partitions: 실행 partition 목록
            tool_sequence: 원본 tool sequence

        Returns:
            각 partition에 대한 sub-task 문자열 리스트
        """
        import re

        # user_request에서 번호 매겨진 단계들 추출 (1. xxx 2. xxx 등)
        step_pattern = r'(\d+)\.\s*([^\n\d]+(?:\n(?!\d+\.).*)*)'
        steps = re.findall(step_pattern, user_request)

        # 단계가 없으면 전체를 하나의 작업으로 취급
        if not steps:
            return [user_request.strip() for _ in partitions]

        # Tool 이름과 단계 매핑을 위한 키워드 맵 (read/write 분리)
        tool_keywords = {
            # filesystem - read 관련 (read_text_file, read_file, list_directory)
            "filesystem_read": ["read_text", "read_file", "read the", "list_directory", "list the directory", "list directory"],
            # filesystem - write 관련 (write_file)
            "filesystem_write": ["write_file", "write a", "write the", "write to", "create", "save to", "save a"],
            # git 관련
            "git": ["git log", "git diff", "git commit", "git_log", "git_diff", "repository"],
            # summarize 관련
            "summarize": ["summarize", "summary"],
            # data_aggregate 관련
            "data_aggregate": ["aggregate", "statistics", "compute_log_statistics", "compute statistics", "compare_hashes", "compare hashes", "find duplicate"],
            # fetch 관련
            "fetch": ["fetch", "url", "http://", "https://", "download"],
            # log_parser 관련
            "log_parser": ["parse_logs", "parse the logs", "parse logs"],
            # image_resize 관련
            "image_resize": ["compute_image_hash", "compute a perceptual hash", "image hash", "batch_resize", "resize", "thumbnail"],
        }

        partition_tasks = []
        assigned_steps = set()  # 이미 할당된 step 추적

        # filesystem partition 수 계산 (첫번째는 read, 마지막은 write)
        filesystem_partitions = [i for i, p in enumerate(partitions) if "filesystem" in p.tools]
        first_filesystem = filesystem_partitions[0] if filesystem_partitions else -1
        last_filesystem = filesystem_partitions[-1] if filesystem_partitions else -1

        for p_idx, partition in enumerate(partitions):
            # 이 partition의 tools에 해당하는 단계들 수집
            partition_steps = []

            for step_num, step_text in steps:
                step_key = step_num
                if step_key in assigned_steps:
                    continue  # 이미 다른 partition에 할당됨

                step_lower = step_text.lower()
                matched = False

                # filesystem write 관련 step은 filesystem partition에서만 처리
                is_filesystem_write_step = any(kw in step_lower for kw in tool_keywords.get("filesystem_write", []))
                is_filesystem_read_step = any(kw in step_lower for kw in tool_keywords.get("filesystem_read", []))

                # 이 partition의 각 tool에 대해 키워드 매칭
                for tool in partition.tools:
                    if tool == "filesystem":
                        # filesystem의 경우 read/write 구분
                        if p_idx == first_filesystem and first_filesystem != last_filesystem:
                            # 첫 번째 filesystem partition: read 관련만
                            keywords = tool_keywords.get("filesystem_read", [])
                            # write 관련 키워드가 있으면 건너뜀
                            if is_filesystem_write_step:
                                continue
                        elif p_idx == last_filesystem:
                            # 마지막 filesystem partition: write 관련
                            keywords = tool_keywords.get("filesystem_write", [])
                            # 단일 filesystem이면 read도 허용
                            if first_filesystem == last_filesystem:
                                keywords = keywords + tool_keywords.get("filesystem_read", [])
                        else:
                            # 중간 filesystem partition: 모두 허용
                            keywords = tool_keywords.get("filesystem_read", []) + tool_keywords.get("filesystem_write", [])
                    else:
                        # filesystem이 아닌 tool: filesystem 관련 step이면 건너뜀
                        if is_filesystem_write_step or is_filesystem_read_step:
                            continue
                        keywords = tool_keywords.get(tool, [tool])

                    if any(kw in step_lower for kw in keywords):
                        partition_steps.append(f"{step_num}. {step_text.strip()}")
                        assigned_steps.add(step_key)
                        matched = True
                        break

            if partition_steps:
                # 매칭된 단계들을 sub-task로 조합
                sub_task = (
                    f"Complete the following steps using tools [{', '.join(partition.tools)}]:\n"
                    + "\n".join(partition_steps)
                )
            else:
                # 매칭된 단계가 없으면 도구 기반으로 일반적인 지시 생성
                if "filesystem" in partition.tools:
                    if p_idx == first_filesystem and first_filesystem != last_filesystem:
                        sub_task = f"Read the required input data using: {', '.join(partition.tools)}"
                    elif p_idx == last_filesystem:
                        sub_task = f"Write the final results/report using: {', '.join(partition.tools)}"
                    else:
                        sub_task = f"Process file operations using: {', '.join(partition.tools)}"
                elif "fetch" in partition.tools:
                    sub_task = f"Fetch all required data using: {', '.join(partition.tools)}"
                elif "summarize" in partition.tools:
                    sub_task = f"Summarize the input data using: {', '.join(partition.tools)}"
                elif "data_aggregate" in partition.tools:
                    sub_task = f"Aggregate and analyze the data using: {', '.join(partition.tools)}"
                else:
                    sub_task = f"Process using tools: {', '.join(partition.tools)}"

            partition_tasks.append(sub_task)

        return partition_tasks

    def get_execution_plan(
        self,
        tool_sequence: list[str],
        preserve_order: bool = True,
    ) -> ExecutionPlan:
        """
        실행 계획 미리보기

        Args:
            tool_sequence: Tool 목록
            preserve_order: 순서 유지 여부

        Returns:
            ExecutionPlan
        """
        return self.planner.create_execution_plan(tool_sequence, preserve_order)

    def print_execution_plan(self, tool_sequence: list[str]):
        """실행 계획 출력"""
        plan = self.get_execution_plan(tool_sequence)

        print("\n" + "=" * 70)
        print("Execution Plan")
        print("=" * 70)
        print(f"Total tools: {plan.total_tools}")
        print(f"Partitions: {len(plan.partitions)}")
        print()

        for i, partition in enumerate(plan.partitions):
            print(f"Partition {i + 1}: {partition.location}")
            print(f"  Tools: {', '.join(partition.tools)}")
            print()
