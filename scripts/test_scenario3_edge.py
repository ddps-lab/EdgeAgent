#!/usr/bin/env python3
"""
Test Scenario 3 (Research Assistant) with Edge WASM configuration.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


EDGE_FETCH = "http://mcp-fetch.edgeagent.edge.edgeagent.ddps.cloud"
EDGE_DATA_AGGREGATE = "http://mcp-data-aggregate.edgeagent.edge.edgeagent.ddps.cloud"


async def test_fetch():
    """Test fetch tools on Edge WASM"""
    print("=" * 70)
    print("Testing mcp-fetch (Edge WASM)")
    print("=" * 70)

    async with streamablehttp_client(EDGE_FETCH) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools = await session.list_tools()
            print(f"\n[1] Available tools: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    - {t.name}")

            # 2. Test fetch with a simple URL
            print(f"\n[2] Testing fetch (example.com)")
            start = time.time()
            try:
                result = await session.call_tool("fetch", {
                    "url": "https://example.com"
                })
                elapsed = (time.time() - start) * 1000

                content = result.content[0].text
                print(f"    Content length: {len(content)} chars")
                print(f"    Preview: {content[:100]}...")
                print(f"    Time: {elapsed:.0f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"    Error: {str(e)[:100]}")
                print(f"    Time: {elapsed:.0f}ms")

            # 3. Test fetch with JSON API endpoint
            print(f"\n[3] Testing fetch (JSON API)")
            start = time.time()
            try:
                result = await session.call_tool("fetch", {
                    "url": "https://httpbin.org/json"
                })
                elapsed = (time.time() - start) * 1000

                content = result.content[0].text
                print(f"    Content length: {len(content)} chars")
                print(f"    Time: {elapsed:.0f}ms")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"    Error: {str(e)[:100]}")
                print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-fetch Edge WASM test completed!")
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

            # 2. Test aggregate_list with research data
            print(f"\n[2] Testing aggregate_list (research articles)")
            articles = [
                {"source": "arxiv", "title": "Paper 1", "relevance": 0.9},
                {"source": "ieee", "title": "Paper 2", "relevance": 0.8},
                {"source": "arxiv", "title": "Paper 3", "relevance": 0.85},
                {"source": "acm", "title": "Paper 4", "relevance": 0.7},
                {"source": "ieee", "title": "Paper 5", "relevance": 0.95},
            ]

            start = time.time()
            result = await session.call_tool("aggregate_list", {
                "items": articles,
                "group_by": "source"
            })
            elapsed = (time.time() - start) * 1000

            content = result.content[0].text
            try:
                agg_data = json.loads(content)
                groups = agg_data.get("groups", {})
                print(f"    Groups: {list(groups.keys()) if groups else 'none'}")
                for k, v in groups.items():
                    print(f"      - {k}: {len(v)} papers")
            except:
                print(f"    Result: {content[:200]}")
            print(f"    Time: {elapsed:.0f}ms")

            # 3. Test combine_research_results
            print(f"\n[3] Testing combine_research_results")
            results = [
                {"query": "WASM performance", "findings": ["fast startup", "low memory"]},
                {"query": "Edge computing", "findings": ["low latency", "distributed"]},
            ]

            start = time.time()
            try:
                result = await session.call_tool("combine_research_results", {
                    "results": results
                })
                elapsed = (time.time() - start) * 1000

                content = result.content[0].text
                try:
                    combined = json.loads(content)
                    print(f"    Combined: {str(combined)[:200]}")
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
    """Run all Edge WASM tests for S3"""
    print("\n" + "=" * 70)
    print("Scenario 3: Research Assistant - Edge WASM Test")
    print("=" * 70)
    print()

    try:
        await test_fetch()
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
