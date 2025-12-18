#!/usr/bin/env python3
"""
Summarize MCP Server Comparison Test

Compares WasmMCP summarize server with Python FastMCP summarize_server.

Usage:
    python tests/compare_summarize.py

Note:
    Requires OPENAI_API_KEY or UPSTAGE_API_KEY environment variable for full tests.
    Tests without API key will only run get_provider_info.
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


def get_summarize_test_cases(has_api_key: bool = False) -> List[TestCase]:
    """Summarize server test cases"""
    test_cases = [
        # get_provider_info - doesn't require API key
        # Note: Python server uses "available_styles", WASM uses "supported_styles"
        TestCase(
            name="provider_info",
            tool_name="get_provider_info",
            args={},
            expected_contains=["provider"]
        ),
    ]

    if has_api_key:
        # Add tests that require API key
        test_cases.extend([
            # summarize_text tests
            TestCase(
                name="summarize_simple",
                tool_name="summarize_text",
                args={
                    "text": "The quick brown fox jumps over the lazy dog. This is a simple sentence that demonstrates all letters of the alphabet. It is often used for testing purposes.",
                    "max_length": 100,
                },
                expected_contains=["summary", "original_length", "summary_length"]
            ),
            TestCase(
                name="summarize_bullet_style",
                tool_name="summarize_text",
                args={
                    "text": "Python is a programming language. It was created by Guido van Rossum. Python is known for its simple syntax. It is widely used in data science and web development.",
                    "style": "bullet",
                    "max_length": 200,
                },
                expected_contains=["summary", "provider"]
            ),
            TestCase(
                name="summarize_technical_style",
                tool_name="summarize_text",
                args={
                    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It uses algorithms to identify patterns and make predictions.",
                    "style": "technical",
                },
                expected_contains=["summary"]
            ),

            # summarize_documents tests
            TestCase(
                name="summarize_multiple_docs",
                tool_name="summarize_documents",
                args={
                    "documents": [
                        {"title": "Doc1", "content": "First document about cats."},
                        {"title": "Doc2", "content": "Second document about dogs."},
                    ],
                    "max_length_per_doc": 50,
                },
                expected_contains=["summaries", "total_documents"]
            ),

            # Error cases
            TestCase(
                name="empty_text",
                tool_name="summarize_text",
                args={"text": "   "},
                expect_error=True
            ),
            TestCase(
                name="empty_documents",
                tool_name="summarize_documents",
                args={"documents": []},
                expect_error=True
            ),
        ])

    return test_cases


async def main():
    """Run summarize server comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_summarize_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-summarize")
        sys.exit(1)

    wasmtime_path = os.path.expanduser("~/.wasmtime/bin/wasmtime")
    if not os.path.exists(wasmtime_path):
        wasmtime_path = "wasmtime"

    # Check for API key
    has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("UPSTAGE_API_KEY"))
    if not has_api_key:
        print("[INFO] No API key found (OPENAI_API_KEY or UPSTAGE_API_KEY)")
        print("[INFO] Running limited tests (get_provider_info only)")
    else:
        print("[INFO] API key found, running full tests")

    # Build wasmtime args with environment variables
    wasmtime_args = ["run", "-S", "http"]
    if os.environ.get("OPENAI_API_KEY"):
        wasmtime_args.extend(["--env", f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}"])
    if os.environ.get("UPSTAGE_API_KEY"):
        wasmtime_args.extend(["--env", f"UPSTAGE_API_KEY={os.environ['UPSTAGE_API_KEY']}"])
    if os.environ.get("SUMMARIZE_PROVIDER"):
        wasmtime_args.extend(["--env", f"SUMMARIZE_PROVIDER={os.environ['SUMMARIZE_PROVIDER']}"])
    wasmtime_args.append(str(wasm_path))

    servers = [
        MCPServerConfig.custom(
            name="wasm_stdio",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": wasmtime_path,
                "args": wasmtime_args,
            },
            description="WasmMCP summarize (stdio)"
        ),
    ]

    # Try to find Python summarize_server
    python_server_path = Path(__file__).parent.parent.parent / "edgeagent/servers/summarize_server.py"
    if python_server_path.exists():
        servers.append(MCPServerConfig.custom(
            name="python",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": "python",
                "args": [str(python_server_path)],
            },
            description="Python summarize_server (FastMCP)"
        ))
    else:
        print(f"[INFO] Python server not found at {python_server_path}, testing WASM only")

    test_cases = get_summarize_test_cases(has_api_key)
    comparator = MCPComparator(servers, server_type="summarize")

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
