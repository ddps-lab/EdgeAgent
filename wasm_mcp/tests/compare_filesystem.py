#!/usr/bin/env python3
"""
Filesystem MCP Server Comparison Test

Node.js, WasmMCP (stdio), WasmMCP (HTTP), mcp-proxy 버전을 비교합니다.

실행:
    # Node.js vs WasmMCP (stdio) 비교
    python tests/compare_filesystem.py

    # HTTP 포함 3-way 비교
    python tests/compare_filesystem.py --with-http --start-http

    # mcp-proxy를 통한 HTTP 비교 (Node.js + proxy, WasmMCP + proxy vs native HTTP)
    python tests/compare_filesystem.py --with-proxy --start-proxy

    # 전체 비교 (5-way: Node.js, WASM stdio, WASM HTTP, Node.js+proxy, WASM+proxy)
    python tests/compare_filesystem.py --with-http --with-proxy --start-http --start-proxy
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_comparator import (
    MCPServerConfig,
    MCPComparator,
)


def check_http_server(host: str = "localhost", port: int = 8000) -> bool:
    """HTTP 서버가 실행 중인지 확인"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port)) == 0
    sock.close()
    return result


def get_wasmtime_path() -> str:
    """wasmtime 경로 찾기"""
    import shutil
    wasmtime_path = shutil.which("wasmtime")
    if wasmtime_path is None:
        home_wasmtime = os.path.expanduser("~/.wasmtime/bin/wasmtime")
        if os.path.exists(home_wasmtime):
            return home_wasmtime
    return wasmtime_path or "wasmtime"


def check_mcp_proxy_installed() -> bool:
    """mcp-proxy가 설치되어 있는지 확인"""
    import shutil
    return shutil.which("mcp-proxy") is not None


def start_mcp_proxy(command_args: list, port: int, test_dir: str) -> subprocess.Popen:
    """mcp-proxy 시작 (stateless 모드로 streamable HTTP 지원)

    Args:
        command_args: 명령어와 인자 리스트 (예: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        port: 프록시 서버 포트
        test_dir: 작업 디렉토리
    """
    # mcp-proxy --port=PORT --stateless command -- args...
    # --stateless: Streamable HTTP 지원 (세션 상태 없이 매 요청 독립 처리)
    # "--" separator로 command와 args 구분
    proc = subprocess.Popen(
        ["mcp-proxy", f"--port={port}", "--stateless", command_args[0], "--"] + command_args[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=test_dir
    )
    return proc


async def run_comparison(include_http: bool = False, start_http_server: bool = False,
                         include_proxy: bool = False, start_proxy: bool = False):
    """파일시스템 MCP 서버 비교 실행"""

    print("=" * 80)
    print("Filesystem MCP Server Comparison")
    print("=" * 80)
    print()

    project_root = Path(__file__).parent.parent
    wasm_stdio = project_root / "target/wasm32-wasip2/release/mcp_server_filesystem_cli.wasm"
    wasm_http = project_root / "target/wasm32-wasip2/release/mcp_server_filesystem_http.wasm"

    # WASM 파일 확인
    if not wasm_stdio.exists():
        print(f"[ERROR] WASM file not found: {wasm_stdio}")
        print("\nBuild first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem")
        return False

    print(f"[INFO] WASM (stdio): {wasm_stdio}")
    print(f"       Size: {wasm_stdio.stat().st_size / 1024:.1f} KB")

    if include_http:
        if not wasm_http.exists():
            print(f"[ERROR] HTTP WASM file not found: {wasm_http}")
            print("\nBuild first:")
            print("  cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem-http")
            return False

        print(f"[INFO] WASM (HTTP): {wasm_http}")
        print(f"       Size: {wasm_http.stat().st_size / 1024:.1f} KB")

    print()

    # 테스트 디렉토리 설정
    test_dir = "/tmp/mcp_filesystem_compare"
    os.makedirs(test_dir, exist_ok=True)
    print(f"[INFO] Test directory: {test_dir}")
    print()

    # HTTP 서버 관리
    http_process = None
    if include_http:
        if not check_http_server():
            if start_http_server:
                print("[INFO] Starting HTTP server...")
                wasmtime_path = get_wasmtime_path()
                http_process = subprocess.Popen(
                    [wasmtime_path, "serve", "--addr", "127.0.0.1:8000",
                     "-S", "cli=y", f"--dir={test_dir}", str(wasm_http)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # 서버 시작 대기
                for _ in range(30):
                    if check_http_server():
                        print("[INFO] HTTP server started")
                        break
                    time.sleep(0.1)
                else:
                    print("[ERROR] Failed to start HTTP server")
                    if http_process:
                        http_process.terminate()
                    return False
            else:
                print("[WARNING] HTTP server not running.")
                print("Start it with:")
                print(f"  wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir={test_dir} {wasm_http}")
                print()
                print("Or use --start-http flag to auto-start")
                print()
                include_http = False

    # mcp-proxy 프로세스 관리
    proxy_processes = []
    NODEJS_PROXY_PORT = 8081
    WASM_PROXY_PORT = 8082

    if include_proxy:
        if not check_mcp_proxy_installed():
            print("[WARNING] mcp-proxy not installed.")
            print("Install with: pipx install mcp-proxy")
            print()
            include_proxy = False
        elif start_proxy:
            print("[INFO] Starting mcp-proxy servers...")
            wasmtime_path = get_wasmtime_path()

            # Node.js + mcp-proxy
            nodejs_proxy_args = ["npx", "-y", "@modelcontextprotocol/server-filesystem", test_dir]
            nodejs_proxy_proc = start_mcp_proxy(nodejs_proxy_args, NODEJS_PROXY_PORT, test_dir)
            proxy_processes.append(("Node.js proxy", nodejs_proxy_proc, NODEJS_PROXY_PORT))

            # WasmMCP (stdio) + mcp-proxy
            wasm_proxy_args = [wasmtime_path, "run", f"--dir={test_dir}", str(wasm_stdio)]
            wasm_proxy_proc = start_mcp_proxy(wasm_proxy_args, WASM_PROXY_PORT, test_dir)
            proxy_processes.append(("WASM proxy", wasm_proxy_proc, WASM_PROXY_PORT))

            # 프록시 서버 시작 대기
            for name, proc, port in proxy_processes:
                for _ in range(50):  # 5초 대기
                    if check_http_server(port=port):
                        print(f"[INFO] {name} started on port {port}")
                        break
                    time.sleep(0.1)
                else:
                    print(f"[WARNING] {name} failed to start on port {port}")
        else:
            # 프록시 서버가 이미 실행 중인지 확인
            nodejs_proxy_running = check_http_server(port=NODEJS_PROXY_PORT)
            wasm_proxy_running = check_http_server(port=WASM_PROXY_PORT)

            if not nodejs_proxy_running and not wasm_proxy_running:
                print("[WARNING] No proxy servers running.")
                print("Start them with --start-proxy flag or manually:")
                print(f"  mcp-proxy --port={NODEJS_PROXY_PORT} --stateless npx -- -y @modelcontextprotocol/server-filesystem {test_dir}")
                print(f"  mcp-proxy --port={WASM_PROXY_PORT} --stateless {get_wasmtime_path()} -- run --dir={test_dir} {wasm_stdio}")
                print()
                include_proxy = False

    # 서버 설정
    servers = [
        MCPServerConfig.nodejs_filesystem(test_dir),
        MCPServerConfig.wasmmcp_stdio(test_dir, str(wasm_stdio)),
    ]

    if include_http:
        servers.append(MCPServerConfig.wasmmcp_http("http://localhost:8000"))

    if include_proxy:
        if check_http_server(port=NODEJS_PROXY_PORT):
            servers.append(MCPServerConfig.nodejs_with_proxy(test_dir, NODEJS_PROXY_PORT))
        if check_http_server(port=WASM_PROXY_PORT):
            servers.append(MCPServerConfig.wasmmcp_stdio_with_proxy(test_dir, str(wasm_stdio), WASM_PROXY_PORT))

    print("[INFO] Servers to compare:")
    for server in servers:
        print(f"    - {server.description}")
    print()

    # 비교 실행
    comparator = MCPComparator(servers, test_dir, server_type="filesystem")

    try:
        report = await comparator.run_comparison()
        report.print_summary()

        # 리포트 저장
        reports_dir = project_root / "tests/reports"
        report_path = report.save(str(reports_dir))
        print(f"\n[INFO] Report saved: {report_path}")

        # 성공 여부
        all_passed = all(c.all_passed for c in report.comparisons)
        return all_passed

    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        comparator.cleanup()
        if http_process:
            http_process.terminate()
            print("[INFO] HTTP server stopped")
        for name, proc, port in proxy_processes:
            proc.terminate()
            print(f"[INFO] {name} stopped")


async def run_quick_test():
    """WasmMCP만 빠르게 테스트"""
    print("=" * 80)
    print("Quick WasmMCP Test (stdio only)")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    wasm_path = project_root / "target/wasm32-wasip2/release/mcp_server_filesystem_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        return False

    test_dir = "/tmp/mcp_quick_test"
    os.makedirs(test_dir, exist_ok=True)

    servers = [
        MCPServerConfig.wasmmcp_stdio(test_dir, str(wasm_path)),
    ]

    comparator = MCPComparator(servers, test_dir, server_type="filesystem")

    try:
        report = await comparator.run_comparison()
        report.print_summary()
        return all(c.all_passed for c in report.comparisons)
    finally:
        comparator.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filesystem MCP Server Comparison")
    parser.add_argument("--with-http", action="store_true",
                       help="Include native HTTP version (WasmMCP HTTP)")
    parser.add_argument("--start-http", action="store_true",
                       help="Auto-start native HTTP server if not running")
    parser.add_argument("--with-proxy", action="store_true",
                       help="Include mcp-proxy versions (Node.js+proxy, WASM+proxy)")
    parser.add_argument("--start-proxy", action="store_true",
                       help="Auto-start mcp-proxy servers if not running")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test WasmMCP only")
    args = parser.parse_args()

    if args.quick:
        success = asyncio.run(run_quick_test())
    else:
        success = asyncio.run(run_comparison(
            include_http=args.with_http,
            start_http_server=args.start_http,
            include_proxy=args.with_proxy,
            start_proxy=args.start_proxy
        ))

    sys.exit(0 if success else 1)
