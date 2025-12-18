#!/usr/bin/env python3
"""
Time MCP Server Comparison Test

Compares WasmMCP time server with Python mcp-server-time.

Usage:
    python tests/compare_time.py
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


def get_time_test_cases() -> List[TestCase]:
    """Time server test cases - matches Python mcp-server-time format"""
    return [
        # get_current_time tests (timezone is required in Python server)
        TestCase(
            name="get_utc_explicit",
            tool_name="get_current_time",
            args={"timezone": "Etc/UTC"},
            expected_contains=["timezone", "datetime", "day_of_week"]
        ),
        TestCase(
            name="get_seoul",
            tool_name="get_current_time",
            args={"timezone": "Asia/Seoul"},
            expected_contains=["Asia/Seoul", "datetime", "day_of_week"]
        ),
        TestCase(
            name="get_new_york",
            tool_name="get_current_time",
            args={"timezone": "America/New_York"},
            expected_contains=["America/New_York", "datetime"]
        ),
        TestCase(
            name="get_london",
            tool_name="get_current_time",
            args={"timezone": "Europe/London"},
            expected_contains=["Europe/London", "datetime"]
        ),
        TestCase(
            name="get_invalid_timezone",
            tool_name="get_current_time",
            args={"timezone": "Invalid/Timezone"},
            expect_error=True
        ),

        # convert_time tests - Python server uses HH:MM format
        TestCase(
            name="convert_utc_to_seoul",
            tool_name="convert_time",
            args={
                "time": "12:00",
                "source_timezone": "Etc/UTC",
                "target_timezone": "Asia/Seoul"
            },
            expected_contains=["source", "target", "Asia/Seoul"]
        ),
        TestCase(
            name="convert_seoul_to_new_york",
            tool_name="convert_time",
            args={
                "time": "12:00",
                "source_timezone": "Asia/Seoul",
                "target_timezone": "America/New_York"
            },
            expected_contains=["source", "target", "America/New_York"]
        ),
        TestCase(
            name="convert_with_dst",
            tool_name="convert_time",
            args={
                "time": "12:00",
                "source_timezone": "America/New_York",
                "target_timezone": "Europe/London"
            },
            expected_contains=["source", "target"]
        ),
        TestCase(
            name="convert_invalid_time",
            tool_name="convert_time",
            args={
                "time": "invalid-time-format",
                "source_timezone": "Etc/UTC",
                "target_timezone": "Asia/Seoul"
            },
            expect_error=True
        ),
        TestCase(
            name="convert_invalid_source_tz",
            tool_name="convert_time",
            args={
                "time": "12:00",
                "source_timezone": "Invalid/TZ",
                "target_timezone": "Etc/UTC"
            },
            expect_error=True
        ),
    ]


async def main():
    """Run time server comparison tests"""

    # Try CLI-specific build first, fallback to default
    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_time_cli.wasm"
    if not wasm_path.exists():
        wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_time_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found")
        print("Build first:")
        print("  ./scripts/build_all.sh")
        print("  or: cargo build --target wasm32-wasip2 --release -p mcp-server-time")
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
                "args": ["run", str(wasm_path)],
            },
            description="WasmMCP time (stdio)"
        ),
    ]

    # Try to find Python mcp-server-time
    try:
        import subprocess
        result = subprocess.run(["which", "mcp-server-time"], capture_output=True, text=True)
        if result.returncode == 0:
            servers.append(MCPServerConfig.custom(
                name="python",
                transport=TransportType.STDIO,
                config={
                    "transport": "stdio",
                    "command": "mcp-server-time",
                    "args": [],
                },
                description="Python mcp-server-time"
            ))
        else:
            print("[INFO] mcp-server-time not found, testing WASM only")
    except Exception:
        print("[INFO] Could not check for mcp-server-time, testing WASM only")

    test_cases = get_time_test_cases()
    comparator = MCPComparator(servers, server_type="time")

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
