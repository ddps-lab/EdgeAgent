#!/usr/bin/env python3
"""
Python ToolTimer for MCP Server Profiling

Thread-safe I/O timing instrumentation for Python MCP servers.
Measures tool execution time with I/O breakdown (disk/network vs compute).

Usage:
    from timing import ToolTimer, measure_io

    @mcp.tool()
    def my_tool(path: str) -> dict:
        timer = ToolTimer("my_tool")

        # I/O operations - wrap with measure_io
        content = measure_io(lambda: open(path).read())

        # Compute operations - no wrapping needed
        result = process(content)

        timer.finish()  # Outputs ---TIMING--- to stdout/stderr
        return result
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import TypeVar, Callable

T = TypeVar('T')

# Thread-local storage for I/O time accumulation
_timing_data = threading.local()

# Timing output file (for Docker volume mount)
TIMING_FILE = Path("/tmp/mcp_timing.json")


def _get_io_accumulator() -> float:
    """Get current I/O time accumulator for this thread"""
    if not hasattr(_timing_data, 'io_ms'):
        _timing_data.io_ms = 0.0
    return _timing_data.io_ms


def _reset_io_accumulator() -> None:
    """Reset I/O time accumulator for this thread"""
    _timing_data.io_ms = 0.0


def _add_io_time(ms: float) -> None:
    """Add I/O time to accumulator for this thread"""
    if not hasattr(_timing_data, 'io_ms'):
        _timing_data.io_ms = 0.0
    _timing_data.io_ms += ms


def measure_io(func: Callable[[], T]) -> T:
    """
    Wrap an I/O operation to measure its execution time.

    Args:
        func: A callable that performs I/O (disk read/write, network request, etc.)

    Returns:
        The result of the I/O operation

    Example:
        # Disk I/O
        content = measure_io(lambda: open(path).read())

        # Network I/O
        response = measure_io(lambda: requests.get(url))

        # PIL image loading
        img = measure_io(lambda: Image.open(path))
    """
    start = time.perf_counter()
    try:
        return func()
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _add_io_time(elapsed_ms)


async def measure_io_async(func: Callable[[], T]) -> T:
    """
    Async version of measure_io for async I/O operations.

    Args:
        func: An async callable that performs I/O

    Returns:
        The result of the I/O operation

    Example:
        content = await measure_io_async(lambda: aiofiles.open(path).read())
        response = await measure_io_async(lambda: aiohttp.get(url))
    """
    start = time.perf_counter()
    try:
        return await func()
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _add_io_time(elapsed_ms)


class ToolTimer:
    """
    Timer for measuring MCP tool execution with I/O breakdown.

    Usage:
        @mcp.tool()
        def read_and_process(path: str) -> dict:
            timer = ToolTimer("read_and_process")

            # I/O operation
            content = measure_io(lambda: open(path).read())

            # Compute operation
            result = process(content)

            timer.finish()
            return result

    Output (to stdout/stderr and /tmp/mcp_timing.json):
        ---TIMING---{"tool": "read_and_process", "fn_total_ms": 150.5, "io_ms": 120.3, "compute_ms": 30.2}
    """

    def __init__(self, tool_name: str):
        """
        Initialize timer and reset I/O accumulator.

        Args:
            tool_name: Name of the tool being timed
        """
        self.tool_name = tool_name
        self.start_time = time.perf_counter()
        _reset_io_accumulator()

    def finish(self) -> dict:
        """
        Finish timing and output results.

        Returns:
            Dictionary with timing data
        """
        elapsed = time.perf_counter() - self.start_time
        io_ms = _get_io_accumulator()

        fn_total_ms = elapsed * 1000
        compute_ms = max(0.0, fn_total_ms - io_ms)

        timing = {
            "tool": self.tool_name,
            "fn_total_ms": round(fn_total_ms, 3),
            "io_ms": round(io_ms, 3),
            "compute_ms": round(compute_ms, 3),
        }

        # Output to stdout and stderr (for Docker logs parsing)
        timing_line = f"---TIMING---{json.dumps(timing)}"
        print(timing_line, file=sys.stdout, flush=True)
        print(timing_line, file=sys.stderr, flush=True)

        # Write to file (for Docker volume mount)
        try:
            TIMING_FILE.write_text(json.dumps(timing))
        except Exception:
            pass  # Ignore file write errors

        return timing


# Convenience decorator for simple cases
def timed_tool(tool_name: str):
    """
    Decorator for timing a tool function.

    Note: This only measures total time, not I/O breakdown.
    For I/O breakdown, use ToolTimer and measure_io manually.

    Usage:
        @mcp.tool()
        @timed_tool("my_tool")
        def my_tool(arg: str) -> dict:
            return {"result": arg}
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            timer = ToolTimer(tool_name)
            try:
                return func(*args, **kwargs)
            finally:
                timer.finish()
        return wrapper
    return decorator
