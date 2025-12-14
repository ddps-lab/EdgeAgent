#!/usr/bin/env python3
"""
MCP Server Comparator Framework

재사용 가능한 MCP 서버 비교 테스트 프레임워크.
다양한 MCP 구현체(Node.js, Python, WasmMCP 등)를 동일한 기준으로 비교.

사용법:
    from mcp_comparator import MCPServerConfig, MCPComparator, ComparisonReport

    # 서버 설정
    nodejs = MCPServerConfig.nodejs_filesystem("/tmp")
    wasmmcp = MCPServerConfig.wasmmcp_stdio("/tmp", "path/to/wasm")

    # 비교 실행
    comparator = MCPComparator([nodejs, wasmmcp])
    report = await comparator.run_comparison()
    report.print_summary()
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


class TransportType(Enum):
    STDIO = "stdio"
    HTTP = "streamable_http"


@dataclass
class MCPServerConfig:
    """MCP 서버 설정"""
    name: str
    transport: TransportType
    config: Dict[str, Any]
    description: str = ""

    @classmethod
    def nodejs_filesystem(cls, allowed_dir: str) -> "MCPServerConfig":
        """Node.js @modelcontextprotocol/server-filesystem 설정"""
        return cls(
            name="nodejs",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
            },
            description="Node.js (@modelcontextprotocol/server-filesystem)"
        )

    @classmethod
    def wasmmcp_stdio(cls, allowed_dir: str, wasm_path: str,
                      wasmtime_path: str = None) -> "MCPServerConfig":
        """WasmMCP Stdio 버전 설정"""
        if wasmtime_path is None:
            wasmtime_path = shutil.which("wasmtime")
            if wasmtime_path is None:
                home_wasmtime = os.path.expanduser("~/.wasmtime/bin/wasmtime")
                if os.path.exists(home_wasmtime):
                    wasmtime_path = home_wasmtime
                else:
                    wasmtime_path = "wasmtime"

        return cls(
            name="wasm_stdio",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": wasmtime_path,
                "args": ["run", f"--dir={allowed_dir}", wasm_path],
            },
            description="WasmMCP (stdio)"
        )

    @classmethod
    def wasmmcp_http(cls, url: str = "http://localhost:8000") -> "MCPServerConfig":
        """WasmMCP HTTP 버전 설정"""
        return cls(
            name="wasm_http",
            transport=TransportType.HTTP,
            config={
                "transport": "streamable_http",
                "url": url,
            },
            description="WasmMCP (HTTP)"
        )

    @classmethod
    def nodejs_with_proxy(cls, allowed_dir: str, proxy_port: int = 8081) -> "MCPServerConfig":
        """Node.js 서버를 mcp-proxy로 HTTP 제공"""
        return cls(
            name="nodejs_proxy",
            transport=TransportType.HTTP,
            config={
                "transport": "streamable_http",
                "url": f"http://localhost:{proxy_port}/mcp",
            },
            description=f"Node.js + mcp-proxy (HTTP:{proxy_port})"
        )

    @classmethod
    def wasmmcp_stdio_with_proxy(cls, allowed_dir: str, wasm_path: str,
                                  proxy_port: int = 8082) -> "MCPServerConfig":
        """WasmMCP stdio 서버를 mcp-proxy로 HTTP 제공"""
        return cls(
            name="wasm_stdio_proxy",
            transport=TransportType.HTTP,
            config={
                "transport": "streamable_http",
                "url": f"http://localhost:{proxy_port}/mcp",
            },
            description=f"WasmMCP (stdio) + mcp-proxy (HTTP:{proxy_port})"
        )

    @classmethod
    def custom(cls, name: str, transport: TransportType,
               config: Dict[str, Any], description: str = "") -> "MCPServerConfig":
        """커스텀 MCP 서버 설정"""
        return cls(name=name, transport=transport, config=config, description=description)


@dataclass
class ToolResult:
    """도구 실행 결과"""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class TestCase:
    """테스트 케이스 정의"""
    name: str
    tool_name: str
    args: Dict[str, Any]
    expected_contains: Optional[List[str]] = None
    expected_not_contains: Optional[List[str]] = None
    expect_error: bool = False
    setup: Optional[callable] = None
    teardown: Optional[callable] = None


@dataclass
class ServerTestResult:
    """서버별 테스트 결과"""
    server_name: str
    server_description: str
    tools_available: List[str]
    test_results: Dict[str, ToolResult]
    startup_time_ms: float = 0.0  # 서버 시작 시간
    total_time_ms: float = 0.0
    connection_success: bool = True
    connection_error: Optional[str] = None


@dataclass
class ComparisonResult:
    """단일 테스트 케이스의 서버 간 비교 결과"""
    test_name: str
    tool_name: str
    results: Dict[str, ToolResult]
    all_passed: bool = False
    differences: List[str] = field(default_factory=list)


class ComparisonReport:
    """비교 결과 리포트"""

    def __init__(self):
        self.server_results: Dict[str, ServerTestResult] = {}
        self.comparisons: List[ComparisonResult] = []
        self.timestamp: datetime = datetime.now()
        self.test_dir: str = ""
        self.server_type: str = ""  # e.g., "filesystem"

    def add_server_result(self, result: ServerTestResult):
        self.server_results[result.server_name] = result

    def add_comparison(self, comparison: ComparisonResult):
        self.comparisons.append(comparison)

    def print_summary(self):
        """요약 출력"""
        print("\n" + "=" * 80)
        print(f"MCP Server Comparison Report - {self.server_type}")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Directory: {self.test_dir}")
        print()

        # 서버별 요약 테이블
        print("-" * 80)
        print("Server Summary")
        print("-" * 80)
        print(f"{'Server':<35} {'Tools':>6} {'Tests':>10} {'Startup':>12} {'Total':>12}")
        print("-" * 80)

        for name, result in self.server_results.items():
            if result.connection_success:
                passed = sum(1 for r in result.test_results.values() if r.success)
                total = len(result.test_results)
                tests_str = f"{passed}/{total}"
                startup_str = f"{result.startup_time_ms:.1f}ms"
                total_str = f"{result.total_time_ms:.1f}ms"
                print(f"{result.server_description:<35} {len(result.tools_available):>6} {tests_str:>10} {startup_str:>12} {total_str:>12}")
            else:
                print(f"{result.server_description:<35} {'CONNECTION FAILED':>42}")

        # 도구별 테스트 결과 비교
        print("\n" + "-" * 80)
        print("Test Results by Tool")
        print("-" * 80)

        # 도구별로 그룹화
        tools_tests: Dict[str, List[ComparisonResult]] = {}
        for comp in self.comparisons:
            if comp.tool_name not in tools_tests:
                tools_tests[comp.tool_name] = []
            tools_tests[comp.tool_name].append(comp)

        server_names = list(self.server_results.keys())

        for tool_name in sorted(tools_tests.keys()):
            tests = tools_tests[tool_name]
            print(f"\n[{tool_name}]")

            # 헤더
            header = f"  {'Test Case':<30}"
            for name in server_names:
                header += f" {name:>15}"
            print(header)

            for comp in tests:
                row = f"  {comp.test_name:<30}"
                for name in server_names:
                    if name in comp.results:
                        r = comp.results[name]
                        if r.success:
                            row += f" {'PASS':>12} {r.execution_time_ms:>4.1f}ms"
                        else:
                            row += f" {'FAIL':>12}      "
                    else:
                        row += f" {'-':>15}"
                print(row)

        # 성능 비교
        print("\n" + "-" * 80)
        print("Performance Comparison (Average execution time per server)")
        print("-" * 80)

        for name, result in self.server_results.items():
            if result.connection_success and result.test_results:
                times = [r.execution_time_ms for r in result.test_results.values()]
                avg_time = sum(times) / len(times)
                print(f"  {result.server_description:<35} avg: {avg_time:.2f}ms, startup: {result.startup_time_ms:.1f}ms")

        # 최종 요약
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        total_tests = len(self.comparisons)
        matching_tests = sum(1 for c in self.comparisons if c.all_passed)
        print(f"Total Tests: {total_tests}")
        print(f"All Passed: {matching_tests}")
        print(f"Differences: {total_tests - matching_tests}")

        if matching_tests == total_tests:
            print("\n[SUCCESS] All servers behave identically!")
        else:
            print(f"\n[WARNING] {total_tests - matching_tests} test(s) have differences.")
            # 차이점 상세
            for comp in self.comparisons:
                if not comp.all_passed:
                    print(f"  - {comp.test_name}: ", end="")
                    passed = [n for n, r in comp.results.items() if r.success]
                    failed = [n for n, r in comp.results.items() if not r.success]
                    if passed:
                        print(f"passed={passed}", end="")
                    if failed:
                        print(f" failed={failed}", end="")
                    print()

    def to_json(self) -> str:
        """JSON 형식으로 내보내기 (구조 개선)"""
        data = {
            "meta": {
                "timestamp": self.timestamp.isoformat(),
                "server_type": self.server_type,
                "test_dir": self.test_dir,
            },
            "servers": {},
            "tests_by_tool": {},
            "summary": {
                "total_tests": len(self.comparisons),
                "all_passed": sum(1 for c in self.comparisons if c.all_passed),
                "differences": sum(1 for c in self.comparisons if not c.all_passed),
            }
        }

        # 서버 정보
        for name, result in self.server_results.items():
            data["servers"][name] = {
                "description": result.server_description,
                "tools_count": len(result.tools_available),
                "tools": result.tools_available,
                "connection_success": result.connection_success,
                "startup_time_ms": result.startup_time_ms,
                "total_time_ms": result.total_time_ms,
                "tests_passed": sum(1 for r in result.test_results.values() if r.success),
                "tests_total": len(result.test_results),
            }

        # 도구별 테스트 결과
        for comp in self.comparisons:
            if comp.tool_name not in data["tests_by_tool"]:
                data["tests_by_tool"][comp.tool_name] = []

            test_data = {
                "name": comp.test_name,
                "all_passed": comp.all_passed,
                "results": {}
            }
            for server_name, result in comp.results.items():
                test_data["results"][server_name] = {
                    "success": result.success,
                    "time_ms": round(result.execution_time_ms, 2),
                    "error": result.error
                }
            data["tests_by_tool"][comp.tool_name].append(test_data)

        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(self, reports_dir: str, filename: Optional[str] = None) -> str:
        """리포트 저장"""
        os.makedirs(reports_dir, exist_ok=True)

        if filename is None:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.server_type}_comparison_{timestamp_str}.json"

        filepath = os.path.join(reports_dir, filename)
        with open(filepath, "w") as f:
            f.write(self.to_json())

        return filepath


class MCPComparator:
    """MCP 서버 비교기"""

    def __init__(self, servers: List[MCPServerConfig], test_dir: Optional[str] = None,
                 server_type: str = "filesystem"):
        self.servers = servers
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="mcp_compare_")
        self.server_type = server_type
        self.report = ComparisonReport()
        self.report.test_dir = self.test_dir
        self.report.server_type = server_type

    def get_default_test_cases(self) -> List[TestCase]:
        """기본 테스트 케이스 목록"""
        test_file = os.path.join(self.test_dir, "test.txt")
        test_content = "Hello, MCP!\nLine 2\nLine 3"

        os.makedirs(self.test_dir, exist_ok=True)
        with open(test_file, "w") as f:
            f.write(test_content)

        subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "nested.txt"), "w") as f:
            f.write("nested content")

        return [
            TestCase(
                name="basic_read",
                tool_name="read_text_file",
                args={"path": test_file},
                expected_contains=["Hello, MCP!", "Line 2", "Line 3"]
            ),
            TestCase(
                name="read_head",
                tool_name="read_text_file",
                args={"path": test_file, "head": 1},
                expected_contains=["Hello, MCP!"],
                expected_not_contains=["Line 3"]
            ),
            TestCase(
                name="read_tail",
                tool_name="read_text_file",
                args={"path": test_file, "tail": 1},
                expected_contains=["Line 3"],
                expected_not_contains=["Hello, MCP!"]
            ),
            TestCase(
                name="basic_list",
                tool_name="list_directory",
                args={"path": self.test_dir},
                expected_contains=["test.txt", "subdir"]
            ),
            TestCase(
                name="write_new",
                tool_name="write_file",
                args={
                    "path": os.path.join(self.test_dir, "written.txt"),
                    "content": "Written by test"
                },
                expected_contains=["Successfully", "written.txt"]
            ),
            TestCase(
                name="read_written",
                tool_name="read_text_file",
                args={"path": os.path.join(self.test_dir, "written.txt")},
                expected_contains=["Written by test"]
            ),
            TestCase(
                name="create_new",
                tool_name="create_directory",
                args={"path": os.path.join(self.test_dir, "new_dir")}
            ),
            TestCase(
                name="search_txt",
                tool_name="search_files",
                args={"path": self.test_dir, "pattern": "*.txt"},
                expected_contains=["test.txt"]
            ),
            TestCase(
                name="get_info",
                tool_name="get_file_info",
                args={"path": test_file},
                expected_contains=["size", "file"]
            ),
            TestCase(
                name="read_nonexistent",
                tool_name="read_text_file",
                args={"path": os.path.join(self.test_dir, "nonexistent.txt")},
                expect_error=True
            ),
            TestCase(
                name="tree",
                tool_name="directory_tree",
                args={"path": self.test_dir},
                expected_contains=["test.txt", "subdir"]
            ),
            TestCase(
                name="list_with_sizes",
                tool_name="list_directory_with_sizes",
                args={"path": self.test_dir},
                expected_contains=["test.txt"]
            ),
            TestCase(
                name="read_multiple",
                tool_name="read_multiple_files",
                args={"paths": [test_file, os.path.join(subdir, "nested.txt")]},
                expected_contains=["Hello, MCP!", "nested content"]
            ),
            TestCase(
                name="list_allowed",
                tool_name="list_allowed_directories",
                args={}
            ),
        ]

    async def _measure_startup_time(self, server: MCPServerConfig) -> float:
        """서버 시작 시간 측정 (초기 연결까지의 시간)"""
        mcp_config = {server.name: server.config}
        client = MultiServerMCPClient(mcp_config)

        start_time = time.time()
        try:
            async with client.session(server.name) as session:
                # 도구 로드까지 완료되면 시작 완료
                await load_mcp_tools(session)
            return (time.time() - start_time) * 1000
        except Exception:
            return -1  # 실패

    async def _test_server(self, server: MCPServerConfig,
                           test_cases: List[TestCase]) -> ServerTestResult:
        """단일 서버 테스트"""
        result = ServerTestResult(
            server_name=server.name,
            server_description=server.description,
            tools_available=[],
            test_results={}
        )

        mcp_config = {server.name: server.config}
        client = MultiServerMCPClient(mcp_config)

        try:
            # 시작 시간 측정
            startup_start = time.time()

            async with client.session(server.name) as session:
                tools = await load_mcp_tools(session)
                result.tools_available = sorted([t.name for t in tools])
                result.startup_time_ms = (time.time() - startup_start) * 1000

                total_start = time.time()

                for test_case in test_cases:
                    tool = next((t for t in tools if t.name == test_case.tool_name), None)

                    if tool is None:
                        result.test_results[test_case.name] = ToolResult(
                            tool_name=test_case.tool_name,
                            success=False,
                            output=None,
                            error=f"Tool '{test_case.tool_name}' not found"
                        )
                        continue

                    start_time = time.time()
                    try:
                        if test_case.setup:
                            test_case.setup()

                        output = await tool.ainvoke(test_case.args)
                        output_str = str(output)
                        execution_time = (time.time() - start_time) * 1000

                        success = True
                        error_msg = None

                        if test_case.expect_error:
                            if "error" not in output_str.lower() and "Error" not in output_str:
                                success = False
                                error_msg = "Expected error but got success"
                        else:
                            if test_case.expected_contains:
                                for expected in test_case.expected_contains:
                                    if expected not in output_str:
                                        success = False
                                        error_msg = f"Expected '{expected}' not found"
                                        break

                            if success and test_case.expected_not_contains:
                                for not_expected in test_case.expected_not_contains:
                                    if not_expected in output_str:
                                        success = False
                                        error_msg = f"Unexpected '{not_expected}' found"
                                        break

                        result.test_results[test_case.name] = ToolResult(
                            tool_name=test_case.tool_name,
                            success=success,
                            output=output,
                            error=error_msg,
                            execution_time_ms=execution_time
                        )

                    except Exception as e:
                        execution_time = (time.time() - start_time) * 1000
                        if test_case.expect_error:
                            result.test_results[test_case.name] = ToolResult(
                                tool_name=test_case.tool_name,
                                success=True,
                                output=str(e),
                                execution_time_ms=execution_time
                            )
                        else:
                            result.test_results[test_case.name] = ToolResult(
                                tool_name=test_case.tool_name,
                                success=False,
                                output=None,
                                error=str(e),
                                execution_time_ms=execution_time
                            )
                    finally:
                        if test_case.teardown:
                            test_case.teardown()

                result.total_time_ms = (time.time() - total_start) * 1000

        except Exception as e:
            result.connection_success = False
            result.connection_error = str(e)

        return result

    def _compare_results(self, test_case: TestCase,
                        results: Dict[str, ToolResult]) -> ComparisonResult:
        """서버 간 결과 비교"""
        comparison = ComparisonResult(
            test_name=test_case.name,
            tool_name=test_case.tool_name,
            results=results
        )

        success_values = [r.success for r in results.values()]
        if len(set(success_values)) == 1 and all(success_values):
            comparison.all_passed = True
        else:
            comparison.all_passed = False
            successful = [name for name, r in results.items() if r.success]
            failed = [name for name, r in results.items() if not r.success]
            if successful and failed:
                comparison.differences.append(f"passed: {successful}, failed: {failed}")

        return comparison

    async def run_comparison(self, test_cases: Optional[List[TestCase]] = None) -> ComparisonReport:
        """비교 테스트 실행"""
        if test_cases is None:
            test_cases = self.get_default_test_cases()

        for server in self.servers:
            print(f"\nTesting: {server.description}...")
            result = await self._test_server(server, test_cases)
            self.report.add_server_result(result)

        for test_case in test_cases:
            results = {}
            for server in self.servers:
                server_result = self.report.server_results.get(server.name)
                if server_result and test_case.name in server_result.test_results:
                    results[server.name] = server_result.test_results[test_case.name]

            if results:
                comparison = self._compare_results(test_case, results)
                self.report.add_comparison(comparison)

        return self.report

    def cleanup(self):
        """테스트 디렉토리 정리"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


async def main():
    """예제 실행"""
    import sys

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_filesystem.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem")
        sys.exit(1)

    test_dir = "/tmp/mcp_compare_test"
    os.makedirs(test_dir, exist_ok=True)

    servers = [
        MCPServerConfig.nodejs_filesystem(test_dir),
        MCPServerConfig.wasmmcp_stdio(test_dir, str(wasm_path)),
    ]

    comparator = MCPComparator(servers, test_dir, server_type="filesystem")

    try:
        report = await comparator.run_comparison()
        report.print_summary()

        reports_dir = Path(__file__).parent / "reports"
        report_path = report.save(str(reports_dir))
        print(f"\nReport saved: {report_path}")

    finally:
        comparator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
