#!/usr/bin/env python3
"""
Image Resize MCP Server Comparison Test

Compares WasmMCP image-resize server with Python FastMCP image_resize_server.

Usage:
    python tests/compare_image_resize.py
"""

import asyncio
import json
import os
import sys
import tempfile
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


def create_test_images():
    """Create test images for comparison"""
    test_dir = tempfile.mkdtemp(prefix="wasm_mcp_image_test_")

    try:
        from PIL import Image

        # Create test images
        img1 = Image.new('RGB', (200, 150), color='red')
        img1.save(Path(test_dir) / "test1.jpg", "JPEG")

        img2 = Image.new('RGB', (300, 200), color='blue')
        img2.save(Path(test_dir) / "test2.png", "PNG")

        img3 = Image.new('RGB', (200, 150), color='red')  # Similar to img1
        img3.save(Path(test_dir) / "test3_similar.jpg", "JPEG")

        img4 = Image.new('RGB', (400, 300), color='green')
        img4.save(Path(test_dir) / "test4.jpg", "JPEG")

        print(f"[INFO] Test images created at: {test_dir}")
        return test_dir

    except ImportError:
        print("[ERROR] PIL not available. Install with: pip install pillow")
        sys.exit(1)


def get_image_resize_test_cases(test_dir: str) -> List[TestCase]:
    """Image resize server test cases"""
    return [
        # get_image_info tests
        TestCase(
            name="get_info_jpg",
            tool_name="get_image_info",
            args={"image_path": f"{test_dir}/test1.jpg"},
            expected_contains=["width", "height", "format", "size_bytes"]
        ),
        TestCase(
            name="get_info_png",
            tool_name="get_image_info",
            args={"image_path": f"{test_dir}/test2.png"},
            expected_contains=["width", "height", "format"]
        ),
        TestCase(
            name="get_info_not_found",
            tool_name="get_image_info",
            args={"image_path": f"{test_dir}/nonexistent.jpg"},
            expect_error=True
        ),

        # scan_directory tests
        TestCase(
            name="scan_basic",
            tool_name="scan_directory",
            args={"directory": test_dir},
            expected_contains=["image_count", "image_paths", "total_size_bytes"]
        ),
        TestCase(
            name="scan_with_info",
            tool_name="scan_directory",
            args={"directory": test_dir, "include_info": True},
            expected_contains=["image_count", "images"]
        ),
        TestCase(
            name="scan_not_found",
            tool_name="scan_directory",
            args={"directory": "/nonexistent/directory"},
            expect_error=True
        ),

        # resize_image tests
        TestCase(
            name="resize_max_size",
            tool_name="resize_image",
            args={
                "image_path": f"{test_dir}/test1.jpg",
                "max_size": 50,
                "quality": 80
            },
            expected_contains=["success", "new_size", "data_base64"]
        ),
        TestCase(
            name="resize_width_only",
            tool_name="resize_image",
            args={
                "image_path": f"{test_dir}/test1.jpg",
                "width": 100
            },
            expected_contains=["success", "new_size"]
        ),
        TestCase(
            name="resize_no_params",
            tool_name="resize_image",
            args={"image_path": f"{test_dir}/test1.jpg"},
            expect_error=True
        ),

        # compute_image_hash tests
        TestCase(
            name="hash_phash",
            tool_name="compute_image_hash",
            args={
                "image_path": f"{test_dir}/test1.jpg",
                "hash_type": "phash"
            },
            expected_contains=["path", "hash", "hash_type"]
        ),
        TestCase(
            name="hash_dhash",
            tool_name="compute_image_hash",
            args={
                "image_path": f"{test_dir}/test1.jpg",
                "hash_type": "dhash"
            },
            expected_contains=["hash"]
        ),
        TestCase(
            name="hash_ahash",
            tool_name="compute_image_hash",
            args={
                "image_path": f"{test_dir}/test1.jpg",
                "hash_type": "ahash"
            },
            expected_contains=["hash"]
        ),

        # batch_resize tests
        TestCase(
            name="batch_resize_basic",
            tool_name="batch_resize",
            args={
                "image_paths": [
                    f"{test_dir}/test1.jpg",
                    f"{test_dir}/test2.png"
                ],
                "max_size": 100,
                "quality": 75
            },
            expected_contains=["successful", "failed", "results"]
        ),
    ]


async def main():
    """Run image resize server comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_image_resize.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-image-resize")
        sys.exit(1)

    wasmtime_path = os.path.expanduser("~/.wasmtime/bin/wasmtime")
    if not os.path.exists(wasmtime_path):
        wasmtime_path = "wasmtime"

    # Create test images
    test_dir = create_test_images()

    servers = [
        MCPServerConfig.custom(
            name="wasm_stdio",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": wasmtime_path,
                "args": ["run", "--dir", test_dir, str(wasm_path)],
            },
            description="WasmMCP image-resize (stdio)"
        ),
    ]

    # Try to find Python image_resize_server
    python_server_path = Path(__file__).parent.parent.parent / "edgeagent/servers/image_resize_server.py"
    if python_server_path.exists():
        servers.append(MCPServerConfig.custom(
            name="python",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": "python",
                "args": [str(python_server_path)],
            },
            description="Python image_resize_server (FastMCP)"
        ))
    else:
        print(f"[INFO] Python server not found at {python_server_path}, testing WASM only")

    test_cases = get_image_resize_test_cases(test_dir)
    comparator = MCPComparator(servers, server_type="image_resize")

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
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"[INFO] Test images cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
