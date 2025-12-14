#!/usr/bin/env python3
"""
Test Scenario 4 (Image Processing) with Edge WASM configuration.

직접 MCP 클라이언트로 Edge에 배포된 WASM 서버를 테스트합니다.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# Edge URLs
EDGE_IMAGE_RESIZE = "http://mcp-image-resize.edgeagent.edge.edgeagent.ddps.cloud"
EDGE_DATA_AGGREGATE = "http://mcp-data-aggregate.edgeagent.edge.edgeagent.ddps.cloud"


async def test_image_resize():
    """Test image_resize tools on Edge WASM"""
    print("=" * 70)
    print("Testing mcp-image-resize (Edge WASM)")
    print("=" * 70)

    async with streamablehttp_client(EDGE_IMAGE_RESIZE) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools = await session.list_tools()
            print(f"\n[1] Available tools: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    - {t.name}")

            # 2. Scan directory
            print(f"\n[2] Testing scan_directory")
            start = time.time()
            result = await session.call_tool("scan_directory", {
                "directory": "/edgeagent/data/scenario4/coco/images"
            })
            elapsed = (time.time() - start) * 1000

            data = json.loads(result.content[0].text)
            print(f"    Found {data.get('image_count', 0)} images ({data.get('total_size_mb', 0):.2f} MB)")
            print(f"    Time: {elapsed:.0f}ms")

            image_paths = data.get("image_paths", [])[:5]  # 처음 5개만
            if not image_paths:
                print("    [WARN] No images found, skipping further tests")
                return

            # 3. Get image info
            print(f"\n[3] Testing get_image_info")
            for path in image_paths[:2]:
                start = time.time()
                result = await session.call_tool("get_image_info", {
                    "image_path": path
                })
                elapsed = (time.time() - start) * 1000
                info = json.loads(result.content[0].text)
                print(f"    {Path(path).name}: {info['width']}x{info['height']} ({info['size_bytes']} bytes) - {elapsed:.0f}ms")

            # 4. Compute hashes
            print(f"\n[4] Testing compute_image_hash")
            hashes = []
            for path in image_paths[:3]:
                start = time.time()
                result = await session.call_tool("compute_image_hash", {
                    "image_path": path,
                    "hash_type": "phash"
                })
                elapsed = (time.time() - start) * 1000
                hash_data = json.loads(result.content[0].text)
                hashes.append(hash_data)
                print(f"    {Path(path).name}: {hash_data['hash']} - {elapsed:.0f}ms")

            # 5. Compare hashes
            print(f"\n[5] Testing compare_hashes")
            start = time.time()
            result = await session.call_tool("compare_hashes", {
                "hashes": hashes,
                "threshold": 10
            })
            elapsed = (time.time() - start) * 1000
            compare_data = json.loads(result.content[0].text)
            print(f"    Unique: {compare_data['unique_count']}, Duplicates: {len(compare_data.get('duplicate_groups', []))}")
            print(f"    Time: {elapsed:.0f}ms")

            # 6. Batch resize (thumbnails)
            print(f"\n[6] Testing batch_resize")
            start = time.time()
            result = await session.call_tool("batch_resize", {
                "image_paths": image_paths[:3],
                "max_size": 100,
                "quality": 75
            })
            elapsed = (time.time() - start) * 1000
            batch_data = json.loads(result.content[0].text)
            print(f"    Successful: {batch_data.get('successful', 0)}/{batch_data.get('total', len(image_paths[:3]))}")
            print(f"    Reduction: {batch_data.get('overall_reduction', 0):.1%}")
            print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-image-resize Edge WASM test completed!")
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

            # 2. Aggregate list
            print(f"\n[2] Testing aggregate_list")
            test_data = [
                {"name": "img1.jpg", "format": "JPEG", "size": 1000},
                {"name": "img2.jpg", "format": "JPEG", "size": 2000},
                {"name": "img3.png", "format": "PNG", "size": 3000},
                {"name": "img4.jpg", "format": "JPEG", "size": 1500},
            ]

            start = time.time()
            result = await session.call_tool("aggregate_list", {
                "items": test_data,
                "group_by": "format"
            })
            elapsed = (time.time() - start) * 1000

            agg_data = json.loads(result.content[0].text)
            print(f"    Groups: {list(agg_data.get('groups', {}).keys())}")
            print(f"    Time: {elapsed:.0f}ms")

    print("\n" + "=" * 70)
    print("[PASS] mcp-data-aggregate Edge WASM test completed!")
    print("=" * 70)


async def main():
    """Run all Edge WASM tests"""
    print("\n" + "=" * 70)
    print("Scenario 4: Image Processing - Edge WASM Test")
    print("=" * 70)
    print()

    try:
        await test_image_resize()
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
