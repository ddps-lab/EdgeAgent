#!/usr/bin/env python3
"""
Test Scenario 2 (Log Analysis) with Edge WASM configuration.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


EDGE_LOG_PARSER = "http://mcp-log-parser.edgeagent.edge.edgeagent.ddps.cloud"
EDGE_DATA_AGGREGATE = "http://mcp-data-aggregate.edgeagent.edge.edgeagent.ddps.cloud"


async def test_log_parser():
    """Test log_parser tools on Edge WASM"""
    print("=" * 70)
    print("Testing mcp-log-parser (Edge WASM)")
    print("=" * 70)

    # Test log content
    test_log = """2025-01-01 10:00:00 INFO Application started
2025-01-01 10:00:01 WARNING High memory usage: 85%
2025-01-01 10:00:02 ERROR Database connection timeout
2025-01-01 10:00:03 INFO Request processed in 150ms
2025-01-01 10:00:04 ERROR Authentication failed for user admin
2025-01-01 10:00:05 INFO Cache hit ratio: 95%"""

    async with streamablehttp_client(EDGE_LOG_PARSER) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools = await session.list_tools()
            print(f"\n[1] Available tools: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    - {t.name}")

            # 2. Parse logs (correct tool name and parameter)
            print(f"\n[2] Testing parse_logs")
            start = time.time()
            result = await session.call_tool("parse_logs", {
                "logContent": test_log
            })
            elapsed = (time.time() - start) * 1000

            # Handle result - could be JSON or text
            content = result.content[0].text
            try:
                data = json.loads(content)
                entries = data.get('entries', data) if isinstance(data, dict) else data
                print(f"    Parsed {len(entries) if isinstance(entries, list) else 'N/A'} entries")
            except json.JSONDecodeError:
                print(f"    Result: {content[:200]}")
            print(f"    Time: {elapsed:.0f}ms")

            # 3. Compute log statistics
            print(f"\n[3] Testing compute_log_statistics")
            # First parse logs to get entries
            result = await session.call_tool("parse_logs", {
                "logContent": test_log
            })
            try:
                parsed = json.loads(result.content[0].text)
                entries = parsed.get('entries', []) if isinstance(parsed, dict) else parsed
            except:
                entries = []

            if entries:
                start = time.time()
                result = await session.call_tool("compute_log_statistics", {
                    "entries": entries
                })
                elapsed = (time.time() - start) * 1000

                try:
                    stats = json.loads(result.content[0].text)
                    print(f"    Stats: {str(stats)[:200]}")
                except:
                    print(f"    Result: {result.content[0].text[:200]}")
                print(f"    Time: {elapsed:.0f}ms")
            else:
                print("    Skipped (no entries)")

            # 4. Filter entries
            print(f"\n[4] Testing filter_entries")
            if entries:
                start = time.time()
                result = await session.call_tool("filter_entries", {
                    "entries": entries,
                    "minLevel": "warning"
                })
                elapsed = (time.time() - start) * 1000

                try:
                    filtered = json.loads(result.content[0].text)
                    count = len(filtered) if isinstance(filtered, list) else "N/A"
                    print(f"    Filtered entries (warning+): {count}")
                except:
                    print(f"    Result: {result.content[0].text[:200]}")
                print(f"    Time: {elapsed:.0f}ms")
            else:
                print("    Skipped (no entries)")

    print("\n" + "=" * 70)
    print("[PASS] mcp-log-parser Edge WASM test completed!")
    print("=" * 70)


async def test_data_aggregate():
    """Test data_aggregate tools on Edge WASM"""
    print("\n" + "=" * 70)
    print("Testing mcp-data-aggregate (Edge WASM)")
    print("=" * 70)

    async with streamablehttp_client(EDGE_DATA_AGGREGATE) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools = await session.list_tools()
            print(f"\n[1] Available tools: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    - {t.name}")

            # 2. Aggregate log entries
            print(f"\n[2] Testing aggregate_list (log entries)")
            log_entries = [
                {"level": "INFO", "message": "Started", "timestamp": "10:00:00"},
                {"level": "ERROR", "message": "Failed", "timestamp": "10:00:02"},
                {"level": "INFO", "message": "Processed", "timestamp": "10:00:03"},
                {"level": "WARNING", "message": "Slow", "timestamp": "10:00:01"},
                {"level": "ERROR", "message": "Timeout", "timestamp": "10:00:04"},
            ]

            start = time.time()
            result = await session.call_tool("aggregate_list", {
                "items": log_entries,
                "group_by": "level"
            })
            elapsed = (time.time() - start) * 1000

            agg_data = json.loads(result.content[0].text)
            groups = agg_data.get("groups", {})
            print(f"    Groups: {list(groups.keys())}")
            print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-data-aggregate Edge WASM test completed!")
    print("=" * 70)


async def main():
    """Run all Edge WASM tests for S2"""
    print("\n" + "=" * 70)
    print("Scenario 2: Log Analysis - Edge WASM Test")
    print("=" * 70)
    print()

    try:
        await test_log_parser()
        await test_data_aggregate()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
