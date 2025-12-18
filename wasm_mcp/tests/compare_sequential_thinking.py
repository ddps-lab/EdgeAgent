#!/usr/bin/env python3
"""
Sequential Thinking MCP Server Comparison Test

Compares WasmMCP sequential-thinking server with TypeScript mcp-server-sequential-thinking.

Usage:
    python tests/compare_sequential_thinking.py
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


def get_sequential_thinking_test_cases() -> List[TestCase]:
    """Sequential thinking test cases - matches Node mcp-server-sequential-thinking format"""
    return [
        # Basic thought steps - tool name is "sequentialthinking" (no underscore)
        TestCase(
            name="first_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Let me analyze the problem step by step",
                "nextThoughtNeeded": True,
                "thoughtNumber": 1,
                "totalThoughts": 3
            },
            expected_contains=["thoughtNumber", "totalThoughts", "nextThoughtNeeded", "branches"]
        ),
        TestCase(
            name="middle_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Continuing analysis of the key factors",
                "nextThoughtNeeded": True,
                "thoughtNumber": 2,
                "totalThoughts": 3
            },
            expected_contains=["thoughtNumber", "branches", "thoughtHistoryLength"]
        ),
        TestCase(
            name="final_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "In conclusion, the solution is clear",
                "nextThoughtNeeded": False,
                "thoughtNumber": 3,
                "totalThoughts": 3
            },
            expected_contains=["thoughtNumber", "nextThoughtNeeded", "branches"]
        ),

        # Revision tests
        TestCase(
            name="revision_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Upon reconsideration, I need to revise my earlier thinking",
                "nextThoughtNeeded": True,
                "thoughtNumber": 4,
                "totalThoughts": 5,
                "isRevision": True,
                "revisesThought": 2
            },
            expected_contains=["thoughtNumber", "branches"]
        ),

        # Branch tests
        TestCase(
            name="branch_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Let me explore an alternative approach",
                "nextThoughtNeeded": True,
                "thoughtNumber": 3,
                "totalThoughts": 5,
                "branchFromThought": 2,
                "branchId": "alternative-path-A"
            },
            expected_contains=["thoughtNumber", "branches"]
        ),

        # Needs more thoughts
        TestCase(
            name="needs_more_thoughts",
            tool_name="sequentialthinking",
            args={
                "thought": "This is more complex than I initially thought",
                "nextThoughtNeeded": True,
                "thoughtNumber": 3,
                "totalThoughts": 5,
                "needsMoreThoughts": True
            },
            expected_contains=["thoughtNumber", "branches"]
        ),

        # Edge cases - TypeScript validates thoughtNumber >= 1 via zod schema
        TestCase(
            name="invalid_thought_number",
            tool_name="sequentialthinking",
            args={
                "thought": "Test",
                "nextThoughtNeeded": True,
                "thoughtNumber": 0,  # Invalid - TypeScript validates with zod (>= 1)
                "totalThoughts": 3
            },
            expect_error=True  # TypeScript validates thoughtNumber >= 1 via zod schema
        ),
        TestCase(
            name="empty_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "   ",  # Empty/whitespace - TypeScript processes it
                "nextThoughtNeeded": True,
                "thoughtNumber": 1,
                "totalThoughts": 3
            },
            expected_contains=["thoughtNumber", "branches"]  # TypeScript returns success
        ),
        TestCase(
            name="revision_without_revises_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Revising",
                "nextThoughtNeeded": True,
                "thoughtNumber": 2,
                "totalThoughts": 3,
                "isRevision": True
                # Missing revisesThought - TypeScript processes it anyway
            },
            expected_contains=["thoughtNumber", "branches"]  # TypeScript returns success
        ),
        TestCase(
            name="branch_without_from_thought",
            tool_name="sequentialthinking",
            args={
                "thought": "Branching",
                "nextThoughtNeeded": True,
                "thoughtNumber": 2,
                "totalThoughts": 3,
                "branchId": "test-branch"
                # Missing branchFromThought - TypeScript processes it anyway
            },
            expected_contains=["thoughtNumber", "branches"]  # TypeScript returns success
        ),
    ]


async def main():
    """Run sequential thinking comparison tests"""

    # Try CLI-specific build first, fallback to default
    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_sequential_thinking_cli.wasm"
    if not wasm_path.exists():
        wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_sequential_thinking_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found")
        print("Build first:")
        print("  ./scripts/build_all.sh")
        print("  or: cargo build --target wasm32-wasip2 --release -p mcp-server-sequential-thinking")
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
            description="WasmMCP sequential-thinking (stdio)"
        ),
    ]

    # Use npm-installed mcp-server-sequential-thinking
    try:
        import subprocess
        result = subprocess.run(["which", "mcp-server-sequential-thinking"], capture_output=True, text=True)
        if result.returncode == 0:
            servers.append(MCPServerConfig.custom(
                name="typescript",
                transport=TransportType.STDIO,
                config={
                    "transport": "stdio",
                    "command": "mcp-server-sequential-thinking",
                    "args": [],
                },
                description="TypeScript mcp-server-sequential-thinking (npm)"
            ))
        else:
            print("[INFO] mcp-server-sequential-thinking not found, testing WASM only")
    except Exception:
        print("[INFO] Could not check for TypeScript server, testing WASM only")

    test_cases = get_sequential_thinking_test_cases()
    comparator = MCPComparator(servers, server_type="sequential_thinking")

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
