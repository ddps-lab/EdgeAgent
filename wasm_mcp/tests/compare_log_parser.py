#!/usr/bin/env python3
"""
Log Parser MCP Server Comparison Test

Compares WasmMCP log-parser with Python FastMCP log_parser_server.

Usage:
    python tests/compare_log_parser.py
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

# Sample log data
APACHE_COMBINED_LOG = '''192.168.1.1 - - [10/Oct/2024:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326 "http://example.com/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
192.168.1.2 - - [10/Oct/2024:13:55:37 -0700] "POST /api/login HTTP/1.1" 401 512 "-" "curl/7.68.0"
192.168.1.1 - - [10/Oct/2024:13:55:38 -0700] "GET /css/style.css HTTP/1.1" 200 1024 "http://example.com/index.html" "Mozilla/5.0"
192.168.1.3 - - [10/Oct/2024:13:55:39 -0700] "GET /favicon.ico HTTP/1.1" 404 0 "-" "Mozilla/5.0"
192.168.1.2 - - [10/Oct/2024:13:55:40 -0700] "POST /api/login HTTP/1.1" 500 0 "-" "curl/7.68.0"
'''

PYTHON_LOG = '''2024-01-15 10:30:01,123 - myapp.main - INFO - Application started successfully
2024-01-15 10:30:02,456 - myapp.database - WARNING - Connection pool running low
2024-01-15 10:30:03,789 - myapp.api - ERROR - Failed to process request: timeout
2024-01-15 10:30:04,012 - myapp.main - DEBUG - Debug info for troubleshooting
2024-01-15 10:30:05,345 - myapp.database - CRITICAL - Database connection lost
'''

JSON_LOG = '''{"timestamp":"2024-01-15T10:30:01Z","level":"info","message":"Server started","service":"api"}
{"timestamp":"2024-01-15T10:30:02Z","level":"warning","message":"High latency detected","service":"api"}
{"timestamp":"2024-01-15T10:30:03Z","level":"error","message":"Request failed","service":"api","error_code":500}
'''


def get_log_parser_test_cases() -> List[TestCase]:
    """Log parser test cases - matching Python FastMCP log_parser_server.py (snake_case)"""
    return [
        # parse_logs tests - Python uses snake_case: log_content, format_type, max_entries
        TestCase(
            name="parse_apache_combined",
            tool_name="parse_logs",
            args={"log_content": APACHE_COMBINED_LOG, "format_type": "apache_combined"},
            expected_contains=["apache_combined", "192.168.1.1", "entries"]
        ),
        TestCase(
            name="parse_apache_auto",
            tool_name="parse_logs",
            args={"log_content": APACHE_COMBINED_LOG},
            expected_contains=["format_detected", "entries"]
        ),
        TestCase(
            name="parse_python_log",
            tool_name="parse_logs",
            args={"log_content": PYTHON_LOG, "format_type": "python"},
            expected_contains=["python", "myapp", "entries"]
        ),
        TestCase(
            name="parse_json_log",
            tool_name="parse_logs",
            args={"log_content": JSON_LOG, "format_type": "json"},
            expected_contains=["json", "Server started", "entries"]
        ),
        TestCase(
            name="parse_max_entries",
            tool_name="parse_logs",
            args={"log_content": APACHE_COMBINED_LOG, "max_entries": 2},
            expected_contains=["entries"]
        ),

        # filter_entries tests - Python uses snake_case: min_level, include_levels
        TestCase(
            name="filter_warning_level",
            tool_name="filter_entries",
            args={
                "entries": [
                    {"_level": "info", "message": "info msg"},
                    {"_level": "warning", "message": "warning msg"},
                    {"_level": "error", "message": "error msg"},
                ],
                "min_level": "warning"
            },
            expected_contains=["warning", "error"]
        ),
        TestCase(
            name="filter_include_levels",
            tool_name="filter_entries",
            args={
                "entries": [
                    {"_level": "info", "message": "info msg"},
                    {"_level": "warning", "message": "warning msg"},
                    {"_level": "error", "message": "error msg"},
                ],
                "include_levels": ["error"]
            },
            expected_contains=["error"]
        ),

        # compute_log_statistics tests
        TestCase(
            name="compute_statistics",
            tool_name="compute_log_statistics",
            args={
                "entries": [
                    {"_level": "info", "status": "200", "ip": "192.168.1.1"},
                    {"_level": "warning", "status": "404", "ip": "192.168.1.2"},
                    {"_level": "error", "status": "500", "ip": "192.168.1.1"},
                    {"_level": "info", "status": "200", "ip": "192.168.1.3"},
                ]
            },
            expected_contains=["by_level", "entry_count"]
        ),

        # search_entries tests
        TestCase(
            name="search_pattern",
            tool_name="search_entries",
            args={
                "entries": [
                    {"_level": "info", "message": "User logged in successfully"},
                    {"_level": "error", "message": "Failed to connect to database"},
                    {"_level": "info", "message": "Request processed"},
                ],
                "pattern": "database"
            },
            expected_contains=["database", "match"]
        ),
        TestCase(
            name="search_regex",
            tool_name="search_entries",
            args={
                "entries": [
                    {"_level": "error", "message": "Error code: 404"},
                    {"_level": "error", "message": "Error code: 500"},
                    {"_level": "info", "message": "Success"},
                ],
                "pattern": "Error code: \\d+"
            },
            expected_contains=["match"]
        ),

        # extract_time_range tests
        TestCase(
            name="extract_time_range",
            tool_name="extract_time_range",
            args={
                "entries": [
                    {"time": "2024-01-15 10:00:00", "_level": "info"},
                    {"time": "2024-01-15 10:30:00", "_level": "warning"},
                    {"time": "2024-01-15 11:00:00", "_level": "error"},
                ]
            },
            expected_contains=["first_timestamp", "last_timestamp"]
        ),
        TestCase(
            name="extract_time_range_no_timestamps",
            tool_name="extract_time_range",
            args={
                "entries": [
                    {"_level": "info", "message": "no time"},
                    {"_level": "warning", "message": "also no time"},
                ]
            },
            expected_contains=["has_timestamps"]
        ),
    ]


async def main():
    """Run log parser comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_log_parser.wasm"
    python_server_path = Path.home() / "edgeagent/servers/log_parser_server.py"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-log-parser")
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
            description="WasmMCP log-parser (stdio)"
        ),
    ]

    # Add Python server if it exists
    if python_server_path.exists():
        servers.append(MCPServerConfig.custom(
            name="python",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": "python3",
                "args": [str(python_server_path)],
            },
            description="Python FastMCP log_parser"
        ))
    else:
        print(f"[INFO] Python server not found at {python_server_path}, testing WASM only")

    test_cases = get_log_parser_test_cases()
    comparator = MCPComparator(servers, server_type="log_parser")

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
