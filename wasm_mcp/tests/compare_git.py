#!/usr/bin/env python3
"""
Git MCP Server Comparison Test

Compares WasmMCP git server with Python mcp-server-git.

Usage:
    python tests/compare_git.py
"""

import asyncio
import json
import os
import sys
import subprocess
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


def create_test_repo():
    """Create a test git repository"""
    repo_dir = tempfile.mkdtemp(prefix="wasm_mcp_git_test_")

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True)

    # Create initial commit
    test_file = Path(repo_dir) / "README.md"
    test_file.write_text("# Test Repository\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

    # Create a second commit
    test_file.write_text("# Test Repository\n\nSecond change.\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add description"], cwd=repo_dir, capture_output=True)

    # Create a branch
    subprocess.run(["git", "branch", "feature-branch"], cwd=repo_dir, capture_output=True)

    return repo_dir


def get_git_test_cases(repo_path: str) -> List[TestCase]:
    """Git server test cases - matches Python mcp-server-git format (plain text output)"""
    return [
        # git_status tests - output is plain text
        TestCase(
            name="status_basic",
            tool_name="git_status",
            args={"repo_path": repo_path},
            expected_contains=["Repository status", "On branch"]
        ),
        TestCase(
            name="status_invalid_repo",
            tool_name="git_status",
            args={"repo_path": "/nonexistent/repo"},
            expect_error=True
        ),

        # git_branch tests - branch_type is REQUIRED in Python mcp-server-git
        TestCase(
            name="branch_local",
            tool_name="git_branch",
            args={"repo_path": repo_path, "branch_type": "local"},
            expected_contains=["*"]
        ),
        TestCase(
            name="branch_all",
            tool_name="git_branch",
            args={"repo_path": repo_path, "branch_type": "all"},
            expected_contains=["master"]  # Default branch name
        ),

        # git_log tests - output is plain text
        TestCase(
            name="log_default",
            tool_name="git_log",
            args={"repo_path": repo_path},
            expected_contains=["Commit history", "Commit:"]
        ),
        TestCase(
            name="log_limited",
            tool_name="git_log",
            args={"repo_path": repo_path, "max_count": 1},
            expected_contains=["Commit"]
        ),

        # git_show tests - output is plain text
        TestCase(
            name="show_head",
            tool_name="git_show",
            args={"repo_path": repo_path, "revision": "HEAD"},
            expected_contains=["Commit:", "Author:", "Message:"]
        ),

        # Write operations - Python server returns success even with nothing to commit/reset
        TestCase(
            name="commit_test",
            tool_name="git_commit",
            args={"repo_path": repo_path, "message": "Test commit"},
            expected_contains=["committed", "hash"]  # Python: "Changes committed successfully with hash xxx"
        ),
        TestCase(
            name="add_test",
            tool_name="git_add",
            args={"repo_path": repo_path, "files": ["README.md"]},  # Use existing file
            expected_contains=["staged", "success"]  # Various success messages
        ),
        TestCase(
            name="reset_test",
            tool_name="git_reset",
            args={"repo_path": repo_path},
            expected_contains=["staged", "reset"]  # Python: "All staged changes reset"
        ),
    ]


async def main():
    """Run git server comparison tests"""

    wasm_path = Path(__file__).parent.parent / "target/wasm32-wasip2/release/mcp_server_git_cli.wasm"

    if not wasm_path.exists():
        print(f"[ERROR] WASM file not found: {wasm_path}")
        print("Build first:")
        print("  cargo build --target wasm32-wasip2 --release -p mcp-server-git")
        sys.exit(1)

    wasmtime_path = os.path.expanduser("~/.wasmtime/bin/wasmtime")
    if not os.path.exists(wasmtime_path):
        wasmtime_path = "wasmtime"

    # Create test repository
    repo_path = create_test_repo()
    print(f"[INFO] Test repository created at: {repo_path}")

    servers = [
        MCPServerConfig.custom(
            name="wasm_stdio",
            transport=TransportType.STDIO,
            config={
                "transport": "stdio",
                "command": wasmtime_path,
                "args": ["run", "--dir", repo_path, str(wasm_path)],
            },
            description="WasmMCP git (stdio)"
        ),
    ]

    # Use pip-installed mcp-server-git
    try:
        result = subprocess.run(["which", "mcp-server-git"], capture_output=True, text=True)
        if result.returncode == 0:
            servers.append(MCPServerConfig.custom(
                name="python",
                transport=TransportType.STDIO,
                config={
                    "transport": "stdio",
                    "command": "mcp-server-git",
                    "args": [],
                },
                description="Python mcp-server-git (pip)"
            ))
        else:
            print("[INFO] mcp-server-git not found, testing WASM only")
    except Exception:
        print("[INFO] Could not check for Python git server, testing WASM only")

    test_cases = get_git_test_cases(repo_path)
    comparator = MCPComparator(servers, server_type="git")

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
        shutil.rmtree(repo_path, ignore_errors=True)
        print(f"[INFO] Test repository cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
