#!/usr/bin/env python3
"""
Test LangChain MCP Adapter with HTTP Transport

Tests connection to WasmMCP server via Streamable HTTP transport.
Requires http_proxy_server.py to be running on localhost:8000
"""

import asyncio
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


ALLOWED_DIR = "/tmp/wasm_http_test"


async def test_http_transport():
    """Test WasmMCP via HTTP transport"""

    print("=" * 70)
    print("LangChain MCP Adapter - HTTP Transport Test")
    print("=" * 70)
    print()

    # Setup test files
    os.makedirs(ALLOWED_DIR, exist_ok=True)
    test_file = Path(ALLOWED_DIR) / "http_test.txt"
    test_file.write_text("Hello from HTTP transport!\nThis is a remote test.\n")
    print(f"[INFO] Test file: {test_file}")
    print()

    # Configure HTTP connection (wasmtime serve runs on root path)
    mcp_config = {
        "wasmmcp_http": {
            "transport": "streamable_http",
            "url": "http://localhost:8000",
        }
    }

    print("[1] MCP Client Configuration:")
    print(f"    Transport: streamable_http")
    print(f"    URL: http://localhost:8000")
    print()

    print("[2] Connecting to WasmMCP HTTP server...")

    client = MultiServerMCPClient(mcp_config)

    try:
        async with client.session("wasmmcp_http") as session:
            # Load tools
            tools = await load_mcp_tools(session)

            print(f"    Connected! Loaded {len(tools)} tools")
            print()

            # List tools
            print("[3] Available tools:")
            for tool in sorted(tools, key=lambda t: t.name)[:5]:
                print(f"    - {tool.name}")
            print(f"    ... and {len(tools) - 5} more")
            print()

            # Test tool invocations
            print("[4] Testing tools via HTTP:")
            print()

            results = []

            # Test read_text_file
            print("    Test: read_text_file")
            try:
                read_tool = next((t for t in tools if t.name == "read_text_file"), None)
                if read_tool:
                    result = await read_tool.ainvoke({"path": str(test_file)})
                    ok = "Hello from HTTP transport!" in str(result)
                    print(f"    {'[PASS]' if ok else '[FAIL]'} read_text_file via HTTP")
                    results.append(("read_text_file", ok))
            except Exception as e:
                print(f"    [ERROR] {e}")
                results.append(("read_text_file", False))

            # Test list_directory
            print("    Test: list_directory")
            try:
                list_tool = next((t for t in tools if t.name == "list_directory"), None)
                if list_tool:
                    result = await list_tool.ainvoke({"path": ALLOWED_DIR})
                    ok = "http_test.txt" in str(result)
                    print(f"    {'[PASS]' if ok else '[FAIL]'} list_directory via HTTP")
                    results.append(("list_directory", ok))
            except Exception as e:
                print(f"    [ERROR] {e}")
                results.append(("list_directory", False))

            # Test write_file
            print("    Test: write_file")
            try:
                write_tool = next((t for t in tools if t.name == "write_file"), None)
                if write_tool:
                    new_file = str(Path(ALLOWED_DIR) / "http_created.txt")
                    result = await write_tool.ainvoke({
                        "path": new_file,
                        "content": "Created via HTTP!"
                    })
                    ok = Path(new_file).exists()
                    print(f"    {'[PASS]' if ok else '[FAIL]'} write_file via HTTP")
                    results.append(("write_file", ok))
            except Exception as e:
                print(f"    [ERROR] {e}")
                results.append(("write_file", False))

            print()

            # Summary
            print("=" * 70)
            print("[5] Results Summary")
            print("=" * 70)
            print()

            passed = sum(1 for _, ok in results if ok)
            total = len(results)

            for name, ok in results:
                print(f"    {'[PASS]' if ok else '[FAIL]'} {name}")

            print()
            print(f"    Total: {passed}/{total} tests passed")
            print()

            if passed == total:
                print("[SUCCESS] WasmMCP works via HTTP transport!")
                print()
                print("This means you can:")
                print("  - Deploy WasmMCP to serverless (Lambda, Workers)")
                print("  - Connect LangChain agents remotely")
                print("  - Use edgeagent with remote MCP servers")
                return True
            else:
                print(f"[FAIL] {total - passed} tests failed")
                return False

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print()
        print("Make sure http_proxy_server.py is running:")
        print("  python tests/http_proxy_server.py")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_http_transport())
    exit(0 if success else 1)
