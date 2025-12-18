#!/usr/bin/env python3
"""
Fetch MCP Server Comparison Test

Compares WasmMCP fetch server with Python mcp-server-fetch.

Usage:
    python tests/compare_fetch.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp_comparator import (
    MCPServerConfig,
    MCPComparator,
    TestCase,
    TransportType,
)


def get_fetch_test_cases() -> List[TestCase]:
    """Fetch server test cases - matches Python fetch_server.py format"""
    return [
        # Basic fetch tests - output is plain text markdown (matching Python fetch_server)
        TestCase(
            name="fetch_example",
            tool_name="fetch",
            args={"url": "https://example.com"},
            expected_contains=["Example Domain"]
        ),
        TestCase(
            name="fetch_with_max_length",
            tool_name="fetch",
            args={"url": "https://example.com", "max_length": 100},
            expected_contains=["Example"]
        ),

        # Error cases
        TestCase(
            name="invalid_url",
            tool_name="fetch",
            args={"url": "not-a-valid-url"},
            expect_error=True
        ),
        TestCase(
            name="unsupported_scheme",
            tool_name="fetch",
            args={"url": "ftp://example.com/file"},
            expect_error=True
        ),
    ]


async def main():
    """Run fetch server comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_fetch_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-fetch")
        sys.exit(1)

    wasmtime_path = os.path.expanduser("~/.wasmtime/bin/wasmtime")
    if not os.path.exists(wasmtime_path):
        wasmtime_path = "wasmtime"

    servers = [
        MCPServerConfig.custom(
            name="wasm_stdio",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": wasmtime_path,
                "args": ["run", "-S", "http", str(wasm_path)],
            },
            description="WasmMCP fetch (stdio)"
        ),
    ]

    # Use edgeagent's fetch_server.py
    python_server_path = Path(__file__).parent.parent.parent / "edgeagent/servers/fetch_server.py"
    if python_server_path.exists():
        servers.append(MCPServerConfig.custom(
            name="python",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": "python",
                "args": [str(python_server_path)],
            },
            description="Python fetch_server (FastMCP)"
        ))
    else:
        print(f"[INFO] Python server not found at {python_server_path}, testing WASM only")

    test_cases = get_fetch_test_cases()
    comparator = MCPComparator(servers, server_type="fetch")

    try:
        report = await comparator.run_comparison(test_cases)
        report.print_summary()

        # Save report
        reports_dir = Path(__file__).parent / "reports"
        report_path = report.save(str(reports_dir))
        print(f"\nReport saved: {report_path}")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
