#!/usr/bin/env python3
"""
SubAgent Scenario Runner - With Unified Metrics Collection

This module provides SubAgent-based scenario execution with metrics collection,
compatible with the existing run_all_scenarios.py infrastructure.

Modes:
- subagent: Location-aware partitioned execution
- subagent_legacy: Single agent with all tools (baseline for comparison)

Usage:
    # Run single scenario
    python scripts/run_scenario_subagent.py --scenario 1 --mode subagent
    python scripts/run_scenario_subagent.py --scenario 2 --mode subagent_legacy

    # Run with comparison
    python scripts/run_scenario_subagent.py --scenario 1 --compare
"""

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Literal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from edgeagent import (
    SubAgentOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    MetricsCollector,
    MetricEntry,
)


SubAgentMode = Literal["subagent", "subagent_legacy"]


@dataclass
class SubAgentScenarioResult:
    """Result from SubAgent scenario execution - 통일된 형식"""
    scenario_name: str
    description: str
    user_request: str
    mode: SubAgentMode
    success: bool
    start_time: float
    end_time: float
    tool_calls: int
    partitions_executed: int
    partition_times: list[float] = field(default_factory=list)
    partition_results: list[dict] = field(default_factory=list)  # 상세 partition 결과
    final_result: Any = None
    error: Optional[str] = None
    data_source: str = ""
    custom_metrics: dict = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        """Total end-to-end latency (Script 모드와 동일)"""
        return (self.end_time - self.start_time) * 1000

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls (Script 모드와 동일한 이름)"""
        return self.tool_calls

    def to_dict(self) -> dict:
        """Script 모드와 동일한 형식으로 변환"""
        return {
            # === Script 모드와 공통 필드 ===
            "scenario_name": self.scenario_name,
            "description": self.description,
            "user_request": self.user_request[:500] if self.user_request else "",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_latency_ms": self.total_latency_ms,
            "tool_call_count": self.tool_call_count,
            "success": self.success,
            "error": self.error,
            # === SubAgent 고유 필드는 custom_data에 포함 ===
            "custom_data": {
                "mode": self.mode,
                "data_source": self.data_source,
                "partitions_executed": self.partitions_executed,
                "partition_times": self.partition_times,
                **self.custom_metrics,
            },
            # final_output_preview는 별도로 추가
        }

    def get_detailed_metrics(self) -> dict:
        """
        상세 메트릭 반환 - Script 모드의 MetricsCollector.to_dict()와 동일한 구조

        Returns:
            Script 모드와 호환되는 metrics dict
        """
        import uuid

        # Location 분포 계산
        call_count_by_location = {"DEVICE": 0, "EDGE": 0, "CLOUD": 0}
        latency_by_location = {"DEVICE": 0.0, "EDGE": 0.0, "CLOUD": 0.0}
        data_by_location = {"DEVICE": 0, "EDGE": 0, "CLOUD": 0}
        all_metrics_entries = []

        for pr in self.partition_results:
            loc = pr.get("location", "UNKNOWN")

            # MetricEntry 수집 (MetricsCollector가 수집한 상세 메트릭)
            metrics_entries = pr.get("metrics_entries", [])
            for entry in metrics_entries:
                # partition 정보 추가
                entry_with_partition = {**entry, "partition_index": pr.get("partition_index")}
                all_metrics_entries.append(entry_with_partition)

                # Location별 집계
                actual_loc = entry.get("location", {}).get("actual_location", loc)
                if actual_loc in call_count_by_location:
                    call_count_by_location[actual_loc] += 1
                    latency_by_location[actual_loc] += entry.get("timing", {}).get("latency_ms", 0)
                    data_by_location[actual_loc] += (
                        entry.get("data_flow", {}).get("input_size_bytes", 0) +
                        entry.get("data_flow", {}).get("output_size_bytes", 0)
                    )

        # 메트릭 요약 계산
        total_input_bytes = sum(e.get("data_flow", {}).get("input_size_bytes", 0) for e in all_metrics_entries)
        total_output_bytes = sum(e.get("data_flow", {}).get("output_size_bytes", 0) for e in all_metrics_entries)
        total_tool_latency_ms = sum(e.get("timing", {}).get("latency_ms", 0) for e in all_metrics_entries)

        # Script 모드와 동일한 구조로 반환
        return {
            "session_id": str(uuid.uuid4())[:8],
            "scenario_name": self.scenario_name,
            "start_time": self.start_time,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": self.end_time,
            "end_time_iso": datetime.fromtimestamp(self.end_time).isoformat(),
            "summary": {
                "total_calls": self.tool_calls,
                "total_latency_ms": total_tool_latency_ms,
                "pipeline_depth": self.tool_calls,
                "parallel_calls_count": 0,
                "success_rate": 1.0 if self.success else 0.0,
                "data_flow": {
                    "cumulative_input_bytes": total_input_bytes,
                    "cumulative_output_bytes": total_output_bytes,
                    "overall_reduction_ratio": total_output_bytes / total_input_bytes if total_input_bytes > 0 else 0,
                    "data_by_location": data_by_location,
                },
                "location_distribution": {
                    "call_count_by_location": call_count_by_location,
                    "latency_by_location": latency_by_location,
                },
                "mcp_overhead": {
                    "total_serialization_ms": sum(e.get("timing", {}).get("mcp_serialization_time_ms", 0) for e in all_metrics_entries),
                    "total_deserialization_ms": sum(e.get("timing", {}).get("mcp_deserialization_time_ms", 0) for e in all_metrics_entries),
                    "server_startup_time_ms": 0.0,
                },
                "resource_usage": {
                    "peak_memory_delta_bytes": max((e.get("resource", {}).get("memory_delta_bytes", 0) for e in all_metrics_entries), default=0),
                    "total_cpu_user_ms": sum(e.get("resource", {}).get("cpu_time_user_ms", 0) for e in all_metrics_entries),
                    "total_cpu_system_ms": sum(e.get("resource", {}).get("cpu_time_system_ms", 0) for e in all_metrics_entries),
                },
                # === SubAgent 고유 필드 ===
                "subagent": {
                    "partitions_executed": self.partitions_executed,
                    "partition_times_ms": self.partition_times,
                    "partition_results": [
                        {
                            "index": pr.get("partition_index"),
                            "location": pr.get("location"),
                            "tools": pr.get("tools"),
                            "execution_time_ms": pr.get("execution_time_ms"),
                            "tool_call_count": len(pr.get("tool_calls", [])),
                            "direct_call": len(pr.get("tools", [])) == 1,
                        }
                        for pr in self.partition_results
                    ],
                },
            },
            "custom_metrics": self.custom_metrics,
            "entries": all_metrics_entries,
        }


class SubAgentScenarioRunner:
    """
    Base class for SubAgent-based scenario execution.

    Provides unified metrics collection compatible with existing infrastructure.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.model = model
        self.temperature = temperature

    @property
    def name(self) -> str:
        """Scenario name for results files"""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Human-readable description"""
        raise NotImplementedError

    @property
    def user_request(self) -> str:
        """User request to send to the orchestrator"""
        raise NotImplementedError

    @property
    def tool_sequence(self) -> list[str]:
        """Tool sequence for SubAgent partitioning"""
        raise NotImplementedError

    def cleanup(self):
        """
        Clean up previous results before execution.
        Override in subclass to specify files/directories to remove.
        """
        pass

    def prepare(self):
        """Prepare data/environment before execution (override in subclass)"""
        pass

    @property
    def data_source(self) -> str:
        """Data source description (override in subclass)"""
        return ""

    def validate(self, result: SubAgentScenarioResult) -> dict:
        """
        Validate execution results (override in subclass)

        Returns:
            dict with keys:
                - valid: bool - overall validation result
                - checks: list of dict - individual check results
                - warnings: list of str - non-critical issues
        """
        return {
            "valid": True,
            "checks": [],
            "warnings": [],
        }

    def _count_tool_calls(self, result: SubAgentScenarioResult, tool_name: str) -> int:
        """Count how many times a specific tool was called"""
        count = 0
        if result.partition_results:
            for pr in result.partition_results:
                for tc in pr.get("tool_calls", []):
                    if tc.get("tool", "") == tool_name:
                        count += 1
        return count

    def _get_all_tool_names(self, result: SubAgentScenarioResult) -> list[str]:
        """Get list of all tool names that were called"""
        tools = []
        if result.partition_results:
            for pr in result.partition_results:
                for tc in pr.get("tool_calls", []):
                    tools.append(tc.get("tool", "unknown"))
        return tools

    async def run(
        self,
        mode: SubAgentMode = "subagent",
        save_results: bool = True,
        print_summary: bool = True,
    ) -> SubAgentScenarioResult:
        """
        Run the scenario with the specified mode.

        Args:
            mode: "subagent" for partitioned execution, "subagent_legacy" for single agent
            save_results: Whether to save results to JSON
            print_summary: Whether to print summary to console
        """
        # Clean up previous results
        self.cleanup()

        # Prepare environment
        self.prepare()

        start_time = time.time()

        try:
            config = OrchestrationConfig(
                mode="subagent" if mode == "subagent" else "legacy",
                subagent_endpoints={},  # Local execution
                model=self.model,
                temperature=self.temperature,
                max_iterations=15,
            )

            orchestrator = SubAgentOrchestrator(self.config_path, config)

            if print_summary:
                print()
                print("=" * 70)
                print(f"SubAgent Scenario: {self.name}")
                print("=" * 70)
                print(f"Mode: {mode}")
                print(f"Data Source: {self.data_source}")
                print(f"Model: {self.model}")
                print()

                if mode == "subagent":
                    print("Execution Plan:")
                    orchestrator.print_execution_plan(self.tool_sequence)

            async with orchestrator:
                orch_result = await orchestrator.run(
                    user_request=self.user_request,
                    tool_sequence=self.tool_sequence,
                    mode="subagent" if mode == "subagent" else "legacy",
                )

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract partition times
            partition_times = []
            if orch_result.partition_results:
                for pr in orch_result.partition_results:
                    if "execution_time_ms" in pr:
                        partition_times.append(pr["execution_time_ms"])

            end_time = time.time()

            result = SubAgentScenarioResult(
                scenario_name=self.name,
                description=self.description,
                user_request=self.user_request,
                mode=mode,
                success=orch_result.success,
                start_time=start_time,
                end_time=end_time,
                tool_calls=orch_result.total_tool_calls,
                partitions_executed=orch_result.partitions_executed,
                partition_times=partition_times,
                partition_results=orch_result.partition_results,
                final_result=orch_result.final_result,
                error=orch_result.error,
                data_source=self.data_source,
                custom_metrics={
                    "model": self.model,
                    "tool_sequence": self.tool_sequence,
                },
            )

            # Validation 수행
            validation = self.validate(result)
            result.custom_metrics["validation"] = validation

            if print_summary:
                self._print_summary(result, validation)

            if save_results:
                self._save_results(result)

            return result

        except Exception as e:
            import traceback
            end_time = time.time()
            return SubAgentScenarioResult(
                scenario_name=self.name,
                description=self.description,
                user_request=self.user_request,
                mode=mode,
                success=False,
                start_time=start_time,
                end_time=end_time,
                tool_calls=0,
                partitions_executed=0,
                error=f"{e}\n{traceback.format_exc()}",
                data_source=self.data_source,
            )

    def _print_summary(self, result: SubAgentScenarioResult, validation: dict = None):
        """Print execution summary with detailed tool calling info"""
        print()
        print("-" * 70)
        print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        print("-" * 70)
        print(f"Total time: {result.total_latency_ms:.0f}ms")
        print(f"Tool calls: {result.tool_calls}")
        print(f"Partitions: {result.partitions_executed}")

        if result.partition_times:
            print(f"Partition times: {[f'{t:.0f}ms' for t in result.partition_times]}")

        if result.error:
            print(f"Error: {result.error[:200]}")

        # Validation 결과 출력
        if validation:
            print()
            print("=" * 70)
            print("Validation Results:")
            print("=" * 70)

            valid = validation.get("valid", True)
            print(f"Overall: {'✓ VALID' if valid else '✗ INVALID'}")

            checks = validation.get("checks", [])
            if checks:
                print("\nChecks:")
                for check in checks:
                    status = "✓" if check.get("passed", False) else "✗"
                    name = check.get("name", "Unknown")
                    expected = check.get("expected", "")
                    actual = check.get("actual", "")
                    print(f"  {status} {name}: expected={expected}, actual={actual}")

            warnings = validation.get("warnings", [])
            if warnings:
                print("\nWarnings:")
                for w in warnings:
                    print(f"  ⚠ {w}")

        # Tool calling 상세 정보 출력
        if result.partition_results:
            print()
            print("=" * 70)
            print("Tool Calling Details:")
            print("=" * 70)

            for pr in result.partition_results:
                idx = pr.get("partition_index", 0)
                loc = pr.get("location", "?")
                tools = pr.get("tools", [])
                tool_calls = pr.get("tool_calls", [])
                exec_time = pr.get("execution_time_ms", 0)

                print(f"\nPartition {idx + 1} [{loc}] - {len(tool_calls)} calls in {exec_time:.0f}ms")
                print(f"  Available tools: {', '.join(tools)}")

                # Tool call별 상세 정보
                for i, tc in enumerate(tool_calls, 1):
                    tool_name = tc.get("tool", "unknown")
                    args = tc.get("input", tc.get("args", {}))

                    # args 요약 (너무 길면 자르기)
                    args_str = str(args)
                    if len(args_str) > 100:
                        args_str = args_str[:97] + "..."

                    print(f"    {i}. {tool_name}({args_str})")

                    # output 미리보기 (있는 경우)
                    output = tc.get("output", "")
                    if output:
                        output_str = str(output)
                        if len(output_str) > 80:
                            output_str = output_str[:77] + "..."
                        print(f"       → {output_str}")

        print()

    def _save_results(self, result: SubAgentScenarioResult):
        """
        Save results to JSON file - Script 모드와 동일한 형식

        Script 모드의 ScenarioResult.to_dict()와 동일한 구조로 저장합니다.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Script 모드와 동일한 파일명 형식: {scenario_name}_{timestamp}.json
        filename = f"{self.name}_{int(result.start_time)}.json"
        output_path = self.output_dir / filename

        # Script 모드와 동일한 형식으로 저장
        base_dict = result.to_dict()
        metrics = result.get_detailed_metrics()

        data = {
            **base_dict,
            "metrics": metrics,
        }

        # final_output_preview 추가
        if result.final_result:
            output_str = str(result.final_result)
            data["final_output_preview"] = output_str[:500] if len(output_str) > 500 else output_str

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")


# =============================================================================
# Scenario Implementations
# =============================================================================

class S1CodeReviewSubAgent(SubAgentScenarioRunner):
    """Scenario 1: Code Review Pipeline with SubAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source = ""
        self._repo_path = None
        self._prepare_repo()

    def cleanup(self):
        """Clean up previous S1 results"""
        report_path = Path("/edgeagent/results/scenario1_code_review_report.md")
        if report_path.exists():
            report_path.unlink()

    def _prepare_repo(self):
        """Prepare Git repository.

        Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
        """
        # Use unified path that works across all locations
        data_dir = Path("/edgeagent/data/scenario1")
        defects4j_dir = data_dir / "defects4j"

        repo_source = None

        if defects4j_dir.exists():
            for subdir in defects4j_dir.iterdir():
                if subdir.is_dir() and (subdir / ".git").exists():
                    repo_source = subdir
                    self._data_source = f"Defects4J ({subdir.name})"
                    self._repo_path = repo_source
                    break

        if repo_source is None:
            raise FileNotFoundError(
                f"No Git repository found in {defects4j_dir}\n"
                "Run 'python scripts/setup_test_data.py -s 1' for test data"
            )

    @property
    def name(self) -> str:
        return "code_review"

    @property
    def description(self) -> str:
        return "Code Review Pipeline with SubAgent Orchestration"

    @property
    def data_source(self) -> str:
        return self._data_source

    @property
    def user_request(self) -> str:
        repo_path = self._repo_path or "/edgeagent/data/scenario1/defects4j/lang"
        return f"""
Review the Git repository at {repo_path}.
1. List repository files using list_directory to understand the structure
2. Get git log to see recent commits (max_count=5)
3. Get git diff to see recent code changes
4. Summarize the key changes using summarize_text
5. Write a code review report to /edgeagent/results/scenario1_code_review_report.md

Return a summary of the code review.
"""

    @property
    def tool_sequence(self) -> list[str]:
        return ["filesystem", "git", "summarize", "data_aggregate", "filesystem"]

    def validate(self, result: SubAgentScenarioResult) -> dict:
        """Validate S1 Code Review scenario - 결과 기반 검증"""
        checks = []
        warnings = []
        valid = True

        # Check 1: 리포트 파일이 생성되었는지
        report_path = Path("/edgeagent/results/scenario1_code_review_report.md")
        report_exists = report_path.exists()
        checks.append({
            "name": "report_file_exists",
            "passed": report_exists,
            "expected": "True",
            "actual": str(report_exists),
        })
        if not report_exists:
            valid = False

        # Check 2: 리포트 내용이 비어있지 않은지
        report_size = 0
        if report_exists:
            report_size = report_path.stat().st_size
        check2_passed = report_size > 100  # 최소 100 bytes
        checks.append({
            "name": "report_content",
            "passed": check2_passed,
            "expected": ">100 bytes",
            "actual": f"{report_size} bytes",
        })
        if not check2_passed:
            valid = False

        # Check 3: 실행 성공 여부
        check3_passed = result.success
        checks.append({
            "name": "execution_success",
            "passed": check3_passed,
            "expected": "True",
            "actual": str(result.success),
        })
        if not check3_passed:
            valid = False

        # Check 4: tool call 횟수 (최소 요건)
        check4_passed = result.tool_calls >= 3
        checks.append({
            "name": "min_tool_calls",
            "passed": check4_passed,
            "expected": ">=3",
            "actual": result.tool_calls,
        })
        if not check4_passed:
            warnings.append(f"Few tool calls ({result.tool_calls}) - may be incomplete")

        return {
            "valid": valid,
            "checks": checks,
            "warnings": warnings,
        }


class S2LogAnalysisSubAgent(SubAgentScenarioRunner):
    """Scenario 2: Log Analysis Pipeline with SubAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source = ""
        self._log_path = ""

    def cleanup(self):
        """Clean up previous S2 results"""
        report_path = Path("/edgeagent/results/scenario2_log_report.md")
        if report_path.exists():
            report_path.unlink()

    def prepare(self):
        """Prepare log file.

        Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
        """
        # Use unified path that works across all locations
        data_dir = Path("/edgeagent/data/scenario2")
        loghub_dir = data_dir / "loghub_samples"

        log_source = None

        if loghub_dir.exists():
            for log_name in ["apache_small.log", "apache_medium.log"]:
                candidate = loghub_dir / log_name
                if candidate.exists():
                    log_source = candidate
                    self._data_source = f"LogHub ({log_name})"
                    self._log_path = str(candidate)
                    break

        if log_source is None:
            raise FileNotFoundError(
                f"No log file found in {loghub_dir}\n"
                "Run 'python scripts/setup_test_data.py -s 2' for test data"
            )

    @property
    def name(self) -> str:
        return "log_analysis"

    @property
    def description(self) -> str:
        return "Log Analysis Pipeline with SubAgent Orchestration"

    @property
    def data_source(self) -> str:
        return self._data_source

    @property
    def user_request(self) -> str:
        return f"""
Analyze the server log file at {self._log_path}.
1. Read the log file using read_text_file
2. Parse the logs using parse_logs with format_type='auto' to get entries
3. Compute statistics using compute_log_statistics with the entries
4. Write a summary report to /edgeagent/results/scenario2_log_report.md

Return the analysis summary.
"""

    @property
    def tool_sequence(self) -> list[str]:
        return ["filesystem", "log_parser", "data_aggregate", "filesystem"]

    def validate(self, result: SubAgentScenarioResult) -> dict:
        """Validate S2 Log Analysis scenario - 결과 기반 검증"""
        checks = []
        warnings = []
        valid = True

        # Check 1: 리포트 파일이 생성되었는지
        report_path = Path("/edgeagent/results/scenario2_log_report.md")
        report_exists = report_path.exists()
        checks.append({
            "name": "report_file_exists",
            "passed": report_exists,
            "expected": "True",
            "actual": str(report_exists),
        })
        if not report_exists:
            valid = False

        # Check 2: 리포트 내용이 비어있지 않은지
        report_size = 0
        if report_exists:
            report_size = report_path.stat().st_size
        check2_passed = report_size > 50  # 최소 50 bytes
        checks.append({
            "name": "report_content",
            "passed": check2_passed,
            "expected": ">50 bytes",
            "actual": f"{report_size} bytes",
        })
        if not check2_passed:
            valid = False

        # Check 3: 실행 성공 여부
        check3_passed = result.success
        checks.append({
            "name": "execution_success",
            "passed": check3_passed,
            "expected": "True",
            "actual": str(result.success),
        })
        if not check3_passed:
            valid = False

        # Check 4: tool call 횟수 (최소 요건)
        check4_passed = result.tool_calls >= 3
        checks.append({
            "name": "min_tool_calls",
            "passed": check4_passed,
            "expected": ">=3",
            "actual": result.tool_calls,
        })
        if not check4_passed:
            warnings.append(f"Few tool calls ({result.tool_calls}) - may be incomplete")

        return {
            "valid": valid,
            "checks": checks,
            "warnings": warnings,
        }


class S3ResearchSubAgent(SubAgentScenarioRunner):
    """Scenario 3: Research Assistant Pipeline with SubAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._research_urls, self._data_source_desc = self._load_research_urls()

    def _load_research_urls(self) -> tuple[list[str], str]:
        """Load research URLs from S2ORC data or use defaults.

        Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
        """
        # Use unified path that works across all locations
        data_dir = Path("/edgeagent/data/scenario3")
        s2orc_dir = data_dir / "s2orc"
        paper_urls_file = s2orc_dir / "paper_urls.txt"

        if paper_urls_file.exists():
            urls = paper_urls_file.read_text().strip().split('\n')
            urls = [u.strip() for u in urls if u.strip()][:3]
            if urls:
                return urls, f"S2ORC ({len(urls)} papers)"

        # Default Wikipedia URLs (reliable and fast)
        default_urls = [
            "https://en.wikipedia.org/wiki/Intelligent_agent",
            "https://en.wikipedia.org/wiki/Large_language_model",
        ]
        return default_urls, "Wikipedia URLs"

    def cleanup(self):
        """Clean up previous S3 results"""
        report_path = Path("/edgeagent/results/scenario3_research_report.md")
        if report_path.exists():
            report_path.unlink()

    @property
    def name(self) -> str:
        return "research_assistant"

    @property
    def description(self) -> str:
        return "Research Assistant Pipeline with SubAgent Orchestration"

    @property
    def data_source(self) -> str:
        return self._data_source_desc

    @property
    def user_request(self) -> str:
        urls_str = "\n".join(f"- {url}" for url in self._research_urls)
        return f"""
Research the topic of AI agents using these URLs:
{urls_str}

1. Fetch content from each URL using fetch tool
2. Summarize each fetched content
3. Aggregate the summaries
4. Write a research report to /edgeagent/results/scenario3_research_report.md

Return a summary of the research.
"""

    @property
    def tool_sequence(self) -> list[str]:
        return ["fetch", "summarize", "data_aggregate", "filesystem"]

    def validate(self, result: SubAgentScenarioResult) -> dict:
        """Validate S3 Research scenario - 결과 기반 검증"""
        checks = []
        warnings = []
        valid = True

        # Check 1: 리포트 파일이 생성되었는지
        report_path = Path("/edgeagent/results/scenario3_research_report.md")
        report_exists = report_path.exists()
        checks.append({
            "name": "report_file_exists",
            "passed": report_exists,
            "expected": "True",
            "actual": str(report_exists),
        })
        if not report_exists:
            valid = False

        # Check 2: 리포트 내용이 비어있지 않은지
        report_size = 0
        if report_exists:
            report_size = report_path.stat().st_size
        check2_passed = report_size > 50  # 최소 50 bytes
        checks.append({
            "name": "report_content",
            "passed": check2_passed,
            "expected": ">50 bytes",
            "actual": f"{report_size} bytes",
        })
        if not check2_passed:
            valid = False

        # Check 3: 실행 성공 여부
        check3_passed = result.success
        checks.append({
            "name": "execution_success",
            "passed": check3_passed,
            "expected": "True",
            "actual": str(result.success),
        })
        if not check3_passed:
            valid = False

        # Check 4: tool call 횟수 (최소 요건 - fetch 2회 + summarize + write)
        min_expected = len(self._research_urls) + 2  # fetch per URL + summarize + write
        check4_passed = result.tool_calls >= min_expected
        checks.append({
            "name": "min_tool_calls",
            "passed": check4_passed,
            "expected": f">={min_expected}",
            "actual": result.tool_calls,
        })
        if not check4_passed:
            warnings.append(f"Few tool calls ({result.tool_calls}) - may be incomplete")

        return {
            "valid": valid,
            "checks": checks,
            "warnings": warnings,
        }


class S4ImageProcessingSubAgent(SubAgentScenarioRunner):
    """Scenario 4: Image Processing Pipeline with SubAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_source = ""
        self._image_count = 0

    def cleanup(self):
        """Clean up previous S4 results"""
        # Remove report file
        report_path = Path("/edgeagent/results/scenario4_image_report.md")
        if report_path.exists():
            report_path.unlink()

        # Remove thumbnails directory
        thumbnail_dir = Path("/edgeagent/data/scenario4/sample_images/thumbnails")
        if thumbnail_dir.exists():
            shutil.rmtree(thumbnail_dir)

    def prepare(self):
        """Prepare images.

        Uses unified path /edgeagent/data that works across all locations (DEVICE/EDGE/CLOUD).
        """
        # Use unified path that works across all locations
        data_dir = Path("/edgeagent/data/scenario4")
        coco_images = data_dir / "coco" / "images"
        sample_images = data_dir / "sample_images"

        if coco_images.exists() and len(list(coco_images.glob("*.jpg"))) > 0:
            self._data_source = "COCO 2017"
            self._image_count = len(list(coco_images.glob("*.jpg")))
        elif sample_images.exists() and len(list(sample_images.glob("*"))) > 0:
            self._data_source = "Generated test images"
            self._image_count = len([f for f in sample_images.glob("*") if f.is_file()])
        else:
            raise FileNotFoundError(
                f"No image directory found in {data_dir}\n"
                "Run 'python scripts/setup_test_data.py -s 4' for test data"
            )

    @property
    def name(self) -> str:
        return "image_processing"

    @property
    def description(self) -> str:
        return "Image Processing Pipeline with SubAgent Orchestration"

    @property
    def data_source(self) -> str:
        return f"{self._data_source} ({self._image_count} images)"

    @property
    def user_request(self) -> str:
        return """
Process images at /edgeagent/data/scenario4/sample_images.

Please:
1. List the directory contents using list_directory to find image files
2. For each image, compute a perceptual hash using compute_image_hash (hash_type="phash")
3. Find duplicate images using compare_hashes with threshold=5
4. Create thumbnails for unique images using batch_resize (max_size=150, quality=75)
5. Write an image processing report to /edgeagent/results/scenario4_image_report.md

Return a summary of the image processing results.
"""

    @property
    def tool_sequence(self) -> list[str]:
        return ["filesystem", "image_resize", "data_aggregate", "filesystem"]

    def validate(self, result: SubAgentScenarioResult) -> dict:
        """Validate S4 Image Processing scenario - 결과 기반 검증"""
        checks = []
        warnings = []
        valid = True

        # Check 1: 리포트 파일이 생성되었는지
        report_path = Path("/edgeagent/results/scenario4_image_report.md")
        report_exists = report_path.exists()
        checks.append({
            "name": "report_file_exists",
            "passed": report_exists,
            "expected": "True",
            "actual": str(report_exists),
        })
        if not report_exists:
            valid = False

        # Check 2: 리포트 내용이 비어있지 않은지
        report_size = 0
        if report_exists:
            report_size = report_path.stat().st_size
        check2_passed = report_size > 50  # 최소 50 bytes
        checks.append({
            "name": "report_content",
            "passed": check2_passed,
            "expected": ">50 bytes",
            "actual": f"{report_size} bytes",
        })
        if not check2_passed:
            valid = False

        # Check 3: 실행 성공 여부
        check3_passed = result.success
        checks.append({
            "name": "execution_success",
            "passed": check3_passed,
            "expected": "True",
            "actual": str(result.success),
        })
        if not check3_passed:
            valid = False

        # Check 4: 썸네일 디렉토리가 생성되었는지 (batch_resize 결과)
        thumbnail_dir = Path("/edgeagent/data/scenario4/sample_images/thumbnails")
        # 또는 이미지 파일들이 처리되었는지 확인
        thumbnails_exist = thumbnail_dir.exists() and len(list(thumbnail_dir.glob("*"))) > 0
        checks.append({
            "name": "thumbnails_created",
            "passed": thumbnails_exist,
            "expected": "True",
            "actual": str(thumbnails_exist),
        })
        if not thumbnails_exist:
            warnings.append("No thumbnails directory found - batch_resize may have failed")

        # Check 5: tool call 횟수 (최소 요건)
        check5_passed = result.tool_calls >= self._image_count  # 최소 이미지 개수만큼
        checks.append({
            "name": "min_tool_calls",
            "passed": check5_passed,
            "expected": f">={self._image_count}",
            "actual": result.tool_calls,
        })
        if not check5_passed:
            warnings.append(f"Few tool calls ({result.tool_calls}) - may be incomplete")

        return {
            "valid": valid,
            "checks": checks,
            "warnings": warnings,
        }


# =============================================================================
# Factory and CLI
# =============================================================================

SCENARIO_CLASSES = {
    1: S1CodeReviewSubAgent,
    2: S2LogAnalysisSubAgent,
    3: S3ResearchSubAgent,
    4: S4ImageProcessingSubAgent,
}

SCENARIO_CONFIGS = {
    1: "config/tools_scenario1.yaml",
    2: "config/tools_scenario2.yaml",
    3: "config/tools_scenario3.yaml",
    4: "config/tools_scenario4.yaml",
}


def create_subagent_scenario(
    scenario_num: int,
    mode: SubAgentMode = "subagent",
    model: str = "gpt-4o-mini",
) -> SubAgentScenarioRunner:
    """Create a SubAgent scenario runner"""
    if scenario_num not in SCENARIO_CLASSES:
        raise ValueError(f"Unknown scenario: {scenario_num}")

    config_path = Path(__file__).parent.parent / SCENARIO_CONFIGS[scenario_num]
    output_suffix = "subagent" if mode == "subagent" else "subagent_legacy"
    output_dir = f"results/scenario{scenario_num}_{output_suffix}"

    return SCENARIO_CLASSES[scenario_num](
        config_path=config_path,
        output_dir=output_dir,
        model=model,
    )


async def run_scenario_subagent(
    scenario_num: int,
    mode: SubAgentMode = "subagent",
    model: str = "gpt-4o-mini",
    save_results: bool = True,
    print_summary: bool = True,
) -> SubAgentScenarioResult:
    """Run a single scenario with SubAgent"""
    scenario = create_subagent_scenario(scenario_num, mode, model)
    return await scenario.run(
        mode=mode,
        save_results=save_results,
        print_summary=print_summary,
    )


async def compare_modes(
    scenario_num: int,
    model: str = "gpt-4o-mini",
) -> tuple[SubAgentScenarioResult, SubAgentScenarioResult]:
    """Run both modes and compare results"""
    print("=" * 70)
    print(f"Scenario {scenario_num}: Comparing SubAgent vs SubAgent-Legacy")
    print("=" * 70)

    # Run subagent_legacy first (baseline)
    legacy_result = await run_scenario_subagent(
        scenario_num,
        mode="subagent_legacy",
        model=model,
    )

    # Run subagent
    subagent_result = await run_scenario_subagent(
        scenario_num,
        mode="subagent",
        model=model,
    )

    # Print comparison
    print()
    print("=" * 70)
    print("COMPARISON: SubAgent-Legacy vs SubAgent")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Legacy':<20} {'SubAgent':<20}")
    print("-" * 65)
    print(f"{'Success':<25} {str(legacy_result.success):<20} {str(subagent_result.success):<20}")
    print(f"{'Total time (ms)':<25} {legacy_result.total_latency_ms:<20.0f} {subagent_result.total_latency_ms:<20.0f}")
    print(f"{'Tool calls':<25} {legacy_result.tool_calls:<20} {subagent_result.tool_calls:<20}")
    print(f"{'Partitions':<25} {legacy_result.partitions_executed:<20} {subagent_result.partitions_executed:<20}")

    if legacy_result.success and subagent_result.success:
        speedup = legacy_result.total_latency_ms / subagent_result.total_latency_ms if subagent_result.total_latency_ms > 0 else 0
        print()
        print(f"{'Speedup':<25} {speedup:.2f}x")

        if speedup > 1:
            print(f"SubAgent mode is {speedup:.1f}x faster")
        else:
            print(f"Legacy mode is {1/speedup:.1f}x faster")

    print()

    return legacy_result, subagent_result


async def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run SubAgent scenarios")
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Scenario number (1-4)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["subagent", "subagent_legacy"],
        default="subagent",
        help="Execution mode (default: subagent)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and compare",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set")
        return False

    if args.compare:
        legacy_result, subagent_result = await compare_modes(args.scenario, args.model)
        return legacy_result.success and subagent_result.success
    else:
        result = await run_scenario_subagent(
            args.scenario,
            mode=args.mode,
            model=args.model,
        )
        return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
