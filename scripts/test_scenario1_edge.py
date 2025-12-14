#!/usr/bin/env python3
"""
Test Scenario 1 (Code Review) with Edge WASM configuration.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


EDGE_GIT = "http://mcp-git.edgeagent.edge.edgeagent.ddps.cloud"
EDGE_DATA_AGGREGATE = "http://mcp-data-aggregate.edgeagent.edge.edgeagent.ddps.cloud"


async def test_git():
    """Test git tools on Edge WASM"""
    print("=" * 70)
    print("Testing mcp-git (Edge WASM)")
    print("=" * 70)

    async with streamablehttp_client(EDGE_GIT) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools = await session.list_tools()
            print(f"\n[1] Available tools: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    - {t.name}")

            # 2. Test git_status (will fail without repo, but tests connectivity)
            print(f"\n[2] Testing git_status")
            start = time.time()
            try:
                result = await session.call_tool("git_status", {
                    "repo_path": "/edgeagent/data/scenario1/repo"
                })
                elapsed = (time.time() - start) * 1000
                content = result.content[0].text
                try:
                    data = json.loads(content)
                    print(f"    Result: {str(data)[:200]}")
                except:
                    print(f"    Result: {content[:200]}")
                print(f"    Time: {elapsed:.0f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"    Error (expected if no repo): {str(e)[:100]}")
                print(f"    Time: {elapsed:.0f}ms")

            # 3. Test git_branch (will also work as connectivity test)
            print(f"\n[3] Testing git_branch")
            start = time.time()
            try:
                result = await session.call_tool("git_branch", {
                    "repo_path": "/edgeagent/data/scenario1/repo"
                })
                elapsed = (time.time() - start) * 1000
                content = result.content[0].text
                try:
                    data = json.loads(content)
                    print(f"    Branches: {str(data)[:200]}")
                except:
                    print(f"    Result: {content[:200]}")
                print(f"    Time: {elapsed:.0f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"    Error (expected if no repo): {str(e)[:100]}")
                print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-git Edge WASM test completed!")
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

            # 2. Test aggregate_list with code review data
            print(f"\n[2] Testing aggregate_list (code changes)")
            code_changes = [
                {"file": "src/main.py", "type": "modified", "lines_added": 10, "lines_removed": 5},
                {"file": "src/utils.py", "type": "modified", "lines_added": 20, "lines_removed": 3},
                {"file": "tests/test_main.py", "type": "added", "lines_added": 50, "lines_removed": 0},
                {"file": "src/config.py", "type": "modified", "lines_added": 5, "lines_removed": 2},
                {"file": "README.md", "type": "modified", "lines_added": 15, "lines_removed": 8},
            ]

            start = time.time()
            result = await session.call_tool("aggregate_list", {
                "items": code_changes,
                "group_by": "type"
            })
            elapsed = (time.time() - start) * 1000

            content = result.content[0].text
            try:
                agg_data = json.loads(content)
                groups = agg_data.get("groups", {})
                print(f"    Groups: {list(groups.keys()) if groups else 'none'}")
                for k, v in groups.items():
                    print(f"      - {k}: {len(v)} files")
            except:
                print(f"    Result: {content[:200]}")
            print(f"    Time: {elapsed:.0f}ms")

            # 3. Test deduplicate
            print(f"\n[3] Testing deduplicate")
            items = [
                {"id": 1, "name": "file1"},
                {"id": 2, "name": "file2"},
                {"id": 1, "name": "file1"},  # duplicate
                {"id": 3, "name": "file3"},
            ]

            start = time.time()
            try:
                result = await session.call_tool("deduplicate", {
                    "items": items
                })
                elapsed = (time.time() - start) * 1000

                content = result.content[0].text
                try:
                    dedup = json.loads(content)
                    print(f"    Result: {str(dedup)[:200]}")
                except:
                    print(f"    Result: {content[:200]}")
                print(f"    Time: {elapsed:.0f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"    Error: {str(e)[:100]}")
                print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-data-aggregate Edge WASM test completed!")
    print("=" * 70)


async def main():
    """Run all Edge WASM tests for S1"""
    print("\n" + "=" * 70)
    print("Scenario 1: Code Review - Edge WASM Test")
    print("=" * 70)
    print()

    try:
        await test_git()
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
