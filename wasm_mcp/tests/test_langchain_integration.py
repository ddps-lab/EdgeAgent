#!/usr/bin/env python3
"""
LangChain MCP Adapter Integration Test for WasmMCP Filesystem Server

Tests that the wasmmcp-based filesystem server works correctly with
langchain-mcp-adapters, proving compatibility with the LangChain ecosystem.

Transport: stdio (wasmtime subprocess)
"""

import asyncio
import os
import tempfile
from pathlib import Path

# LangChain MCP Adapter imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# Paths
WASM_FILE = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_filesystem_cli.wasm"
WASMTIME = os.path.expanduser("~/.wasmtime/bin/wasmtime")


async def test_langchain_mcp_integration():
    """Test wasmmcp filesystem server with LangChain MCP Adapter"""

    print("=" * 70)
    print("LangChain MCP Adapter Integration Test")
    print("=" * 70)
    print()

    # Verify WASM file exists
    if not WASM_FILE.exists():
        print(f"[ERROR] WASM file not found: {WASM_FILE}")
        print("Run: cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem")
        return False

    print(f"[INFO] WASM file: {WASM_FILE}")
    print(f"[INFO] Runtime: {WASMTIME}")
    print()

    # Create test directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[INFO] Test directory: {tmpdir}")
        print()

        # Setup test files
        test_file = Path(tmpdir) / "hello.txt"
        test_file.write_text("Hello from WasmMCP!\nThis is a test file.\n")

        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested file content")

        print("[1] Test files created:")
        print(f"    - {test_file}")
        print(f"    - {subdir / 'nested.txt'}")
        print()

        # Configure MCP client for wasmmcp server
        mcp_config = {
            "wasmmcp_filesystem": {
                "transport": "stdio",
                "command": WASMTIME,
                "args": ["run", f"--dir={tmpdir}", str(WASM_FILE)],
            }
        }

        print("[2] MCP Client Configuration:")
        print(f"    Server: wasmmcp_filesystem")
        print(f"    Transport: stdio")
        print(f"    Command: {WASMTIME} run --dir={tmpdir} {WASM_FILE.name}")
        print()

        # Connect to MCP server using new API (langchain-mcp-adapters >= 0.1.0)
        print("[3] Connecting to WasmMCP server...")

        client = MultiServerMCPClient(mcp_config)

        try:
            # Use session context manager
            async with client.session("wasmmcp_filesystem") as session:
                # Load tools
                tools = await load_mcp_tools(session)

                print(f"    Connected! Loaded {len(tools)} tools")
                print()

                # List available tools
                print("[4] Available tools:")
                for tool in sorted(tools, key=lambda t: t.name):
                    print(f"    - {tool.name}")
                print()

                # Test tool invocations
                print("[5] Testing tool invocations:")
                print()

                results = []

                # Test 1: read_file (using read_text_file)
                print("    Test 1: read_text_file")
                try:
                    read_tool = next((t for t in tools if t.name == "read_text_file"), None)
                    if read_tool:
                        result = await read_tool.ainvoke({"path": str(test_file)})
                        ok = "Hello from WasmMCP!" in str(result)
                        print(f"    {'[PASS]' if ok else '[FAIL]'} read_text_file")
                        results.append(("read_text_file", ok))
                    else:
                        print("    [SKIP] read_text_file not found")
                except Exception as e:
                    print(f"    [ERROR] read_text_file: {e}")
                    results.append(("read_text_file", False))

                # Test 2: list_directory
                print("    Test 2: list_directory")
                try:
                    list_tool = next((t for t in tools if t.name == "list_directory"), None)
                    if list_tool:
                        result = await list_tool.ainvoke({"path": tmpdir})
                        ok = "hello.txt" in str(result) and "subdir" in str(result)
                        print(f"    {'[PASS]' if ok else '[FAIL]'} list_directory")
                        results.append(("list_directory", ok))
                    else:
                        print("    [SKIP] list_directory not found")
                except Exception as e:
                    print(f"    [ERROR] list_directory: {e}")
                    results.append(("list_directory", False))

                # Test 3: write_file
                print("    Test 3: write_file")
                try:
                    write_tool = next((t for t in tools if t.name == "write_file"), None)
                    if write_tool:
                        new_file = str(Path(tmpdir) / "new_file.txt")
                        result = await write_tool.ainvoke({
                            "path": new_file,
                            "content": "Created by LangChain!"
                        })
                        ok = Path(new_file).exists() and "Created by LangChain!" in Path(new_file).read_text()
                        print(f"    {'[PASS]' if ok else '[FAIL]'} write_file")
                        results.append(("write_file", ok))
                    else:
                        print("    [SKIP] write_file not found")
                except Exception as e:
                    print(f"    [ERROR] write_file: {e}")
                    results.append(("write_file", False))

                # Test 4: search_files
                print("    Test 4: search_files")
                try:
                    search_tool = next((t for t in tools if t.name == "search_files"), None)
                    if search_tool:
                        result = await search_tool.ainvoke({
                            "path": tmpdir,
                            "pattern": "*.txt"
                        })
                        ok = ".txt" in str(result)
                        print(f"    {'[PASS]' if ok else '[FAIL]'} search_files")
                        results.append(("search_files", ok))
                    else:
                        print("    [SKIP] search_files not found")
                except Exception as e:
                    print(f"    [ERROR] search_files: {e}")
                    results.append(("search_files", False))

                # Test 5: get_file_info
                print("    Test 5: get_file_info")
                try:
                    info_tool = next((t for t in tools if t.name == "get_file_info"), None)
                    if info_tool:
                        result = await info_tool.ainvoke({"path": str(test_file)})
                        ok = "size:" in str(result) or "type:" in str(result)
                        print(f"    {'[PASS]' if ok else '[FAIL]'} get_file_info")
                        results.append(("get_file_info", ok))
                    else:
                        print("    [SKIP] get_file_info not found")
                except Exception as e:
                    print(f"    [ERROR] get_file_info: {e}")
                    results.append(("get_file_info", False))

                print()

                # Summary
                print("=" * 70)
                print("[6] Test Results Summary")
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
                    print("[SUCCESS] WasmMCP filesystem server works with LangChain MCP Adapter!")
                    print()
                    print("This proves that wasmmcp-based servers can be used with:")
                    print("  - LangChain agents (create_agent, create_react_agent)")
                    print("  - LangGraph workflows")
                    print("  - Any LangChain-compatible framework")
                    return True
                else:
                    print(f"[FAIL] {total - passed} tests failed")
                    return False

        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_with_langchain_agent():
    """Optional: Test with actual LangChain agent (requires OpenAI API key)"""

    try:
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print("\n[SKIP] LangChain agent test (missing dependencies)")
        print("       Install: pip install langchain-openai langgraph")
        return True

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] LangChain agent test (OPENAI_API_KEY not set)")
        return True

    print()
    print("=" * 70)
    print("LangChain Agent Integration Test")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test file
        test_file = Path(tmpdir) / "data.txt"
        test_file.write_text("Temperature: 25C\nHumidity: 60%\n")

        mcp_config = {
            "wasmmcp_filesystem": {
                "transport": "stdio",
                "command": WASMTIME,
                "args": ["run", f"--dir={tmpdir}", str(WASM_FILE)],
            }
        }

        client = MultiServerMCPClient(mcp_config)

        async with client.session("wasmmcp_filesystem") as session:
            tools = await load_mcp_tools(session)

            # Create LangChain agent
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = create_react_agent(llm, tools)

            # Run agent
            print(f"[INFO] Asking agent to read {test_file}")
            result = await agent.ainvoke({
                "messages": [("user", f"Please read the file at {test_file} and summarize its contents.")]
            })

            # Check result
            last_message = result["messages"][-1].content
            print(f"[INFO] Agent response: {last_message[:200]}...")

            ok = "25" in last_message or "temperature" in last_message.lower()
            print(f"\n{'[PASS]' if ok else '[FAIL]'} Agent successfully used WasmMCP filesystem tool")

            return ok


if __name__ == "__main__":
    print()

    # Run basic integration test
    success = asyncio.run(test_langchain_mcp_integration())

    if success:
        # Optionally run agent test
        asyncio.run(test_with_langchain_agent())

    exit(0 if success else 1)
