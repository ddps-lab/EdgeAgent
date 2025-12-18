#!/usr/bin/env python3
"""
Data Aggregate MCP Server Comparison Test

Compares WasmMCP data-aggregate with Python FastMCP data_aggregate_server.

Usage:
    python tests/compare_data_aggregate.py
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


def get_data_aggregate_test_cases() -> List[TestCase]:
    """Data aggregate test cases - matching Python FastMCP data_aggregate_server.py (snake_case)"""
    return [
        # aggregate_list tests - Python uses snake_case: group_by, sum_fields, count_field
        TestCase(
            name="aggregate_group_by",
            tool_name="aggregate_list",
            args={
                "items": [
                    {"category": "A", "value": 10},
                    {"category": "B", "value": 20},
                    {"category": "A", "value": 15},
                    {"category": "B", "value": 25},
                    {"category": "C", "value": 30},
                ],
                "group_by": "category"
            },
            expected_contains=["groups", "group_count", "total_count"]
        ),
        TestCase(
            name="aggregate_sum_fields",
            tool_name="aggregate_list",
            args={
                "items": [
                    {"type": "sale", "amount": 100, "quantity": 5},
                    {"type": "sale", "amount": 200, "quantity": 10},
                    {"type": "return", "amount": 50, "quantity": 2},
                ],
                "sum_fields": ["amount", "quantity"]
            },
            expected_contains=["field_stats", "sum", "mean"]
        ),
        TestCase(
            name="aggregate_count_field",
            tool_name="aggregate_list",
            args={
                "items": [
                    {"status": "active", "name": "a"},
                    {"status": "inactive", "name": "b"},
                    {"status": "active", "name": "c"},
                    {"status": "pending", "name": "d"},
                ],
                "count_field": "status"
            },
            expected_contains=["counts", "active"]
        ),
        TestCase(
            name="aggregate_empty",
            tool_name="aggregate_list",
            args={"items": []},
            expected_contains=["total_count"]
        ),

        # merge_summaries tests
        TestCase(
            name="merge_basic",
            tool_name="merge_summaries",
            args={
                "summaries": [
                    {"count": 10, "total": 100},
                    {"count": 20, "total": 200},
                    {"count": 30, "total": 300},
                ]
            },
            expected_contains=["merged_count", "total"]
        ),
        TestCase(
            name="merge_weighted",
            tool_name="merge_summaries",
            args={
                "summaries": [
                    {"score": 80},
                    {"score": 90},
                ],
                "weights": [0.3, 0.7]
            },
            expected_contains=["weighted_avg"]
        ),
        TestCase(
            name="merge_empty",
            tool_name="merge_summaries",
            args={"summaries": []},
            expected_contains=["merged_count"]
        ),

        # combine_research_results tests
        TestCase(
            name="combine_research",
            tool_name="combine_research_results",
            args={
                "results": [
                    {"title": "Paper A", "summary": "Summary of paper A", "relevance_score": 0.9},
                    {"title": "Paper B", "summary": "Summary of paper B", "relevance_score": 0.7},
                    {"title": "Paper C", "summary": "Summary of paper C", "relevance_score": 0.8},
                ]
            },
            expected_contains=["result_count", "items", "combined_text"]
        ),
        TestCase(
            name="combine_custom_fields",
            tool_name="combine_research_results",
            args={
                "results": [
                    {"name": "Item 1", "description": "Desc 1", "score": 5},
                    {"name": "Item 2", "description": "Desc 2", "score": 3},
                ],
                "title_field": "name",
                "summary_field": "description",
                "score_field": "score"
            },
            expected_contains=["result_count", "combined_text"]
        ),

        # deduplicate tests
        TestCase(
            name="deduplicate_first",
            tool_name="deduplicate",
            args={
                "items": [
                    {"id": 1, "name": "a", "value": 10},
                    {"id": 2, "name": "b", "value": 20},
                    {"id": 1, "name": "a", "value": 15},  # duplicate
                    {"id": 3, "name": "c", "value": 30},
                ],
                "key_fields": ["id"]
            },
            expected_contains=["unique_count", "duplicates_removed"]
        ),
        TestCase(
            name="deduplicate_last",
            tool_name="deduplicate",
            args={
                "items": [
                    {"id": 1, "name": "a", "value": 10},
                    {"id": 1, "name": "a", "value": 15},  # duplicate - keep this
                ],
                "key_fields": ["id"],
                "keep": "last"
            },
            expected_contains=["unique_count"]
        ),
        TestCase(
            name="deduplicate_multiple_keys",
            tool_name="deduplicate",
            args={
                "items": [
                    {"type": "A", "region": "US", "count": 10},
                    {"type": "A", "region": "EU", "count": 20},
                    {"type": "A", "region": "US", "count": 15},  # duplicate
                    {"type": "B", "region": "US", "count": 30},
                ],
                "key_fields": ["type", "region"]
            },
            expected_contains=["unique_count", "duplicates_removed"]
        ),

        # compute_trends tests
        TestCase(
            name="trends_increasing",
            tool_name="compute_trends",
            args={
                "time_series": [
                    {"timestamp": "2024-01-01", "value": 10},
                    {"timestamp": "2024-01-02", "value": 12},
                    {"timestamp": "2024-01-03", "value": 15},
                    {"timestamp": "2024-01-04", "value": 18},
                    {"timestamp": "2024-01-05", "value": 25},
                ]
            },
            expected_contains=["trend", "increasing", "data_points"]
        ),
        TestCase(
            name="trends_decreasing",
            tool_name="compute_trends",
            args={
                "time_series": [
                    {"timestamp": "2024-01-01", "value": 100},
                    {"timestamp": "2024-01-02", "value": 80},
                    {"timestamp": "2024-01-03", "value": 60},
                    {"timestamp": "2024-01-04", "value": 40},
                    {"timestamp": "2024-01-05", "value": 20},
                ]
            },
            expected_contains=["trend", "decreasing", "data_points"]
        ),
        TestCase(
            name="trends_stable",
            tool_name="compute_trends",
            args={
                "time_series": [
                    {"timestamp": "2024-01-01", "value": 50},
                    {"timestamp": "2024-01-02", "value": 51},
                    {"timestamp": "2024-01-03", "value": 49},
                    {"timestamp": "2024-01-04", "value": 50},
                    {"timestamp": "2024-01-05", "value": 52},
                ]
            },
            expected_contains=["trend", "stable", "data_points"]
        ),
        TestCase(
            name="trends_custom_fields",
            tool_name="compute_trends",
            args={
                "time_series": [
                    {"date": "2024-01-01", "metric": 10},
                    {"date": "2024-01-02", "metric": 20},
                    {"date": "2024-01-03", "metric": 30},
                ],
                "time_field": "date",
                "value_field": "metric"
            },
            expected_contains=["trend", "data_points"]
        ),
        TestCase(
            name="trends_insufficient",
            tool_name="compute_trends",
            args={
                "time_series": [
                    {"timestamp": "2024-01-01", "value": 10},
                ]
            },
            expected_contains=["insufficient_data"]
        ),
    ]


async def main():
    """Run data aggregate comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_data_aggregate_cli.wasm"
    python_server_path = Path.home() / "edgeagent/servers/data_aggregate_server.py"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-data-aggregate")
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
            description="WasmMCP data-aggregate (stdio)"
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
            description="Python FastMCP data_aggregate"
        ))
    else:
        print(f"[INFO] Python server not found at {python_server_path}, testing WASM only")

    test_cases = get_data_aggregate_test_cases()
    comparator = MCPComparator(servers, server_type="data_aggregate")

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
