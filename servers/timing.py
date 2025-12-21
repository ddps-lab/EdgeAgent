#!/usr/bin/env python3
"""
ToolTimer - MCP Tool Profiling Module

Measures execution time and peak memory for MCP tools.
Outputs timing data in parseable format: ---TIMING---{json}

Usage:
    from timing import ToolTimer, measure_io, measure_io_async

    @mcp.tool()
    def my_tool(arg: str) -> str:
        timer = ToolTimer("my_tool")

        # For I/O operations
        data = measure_io(lambda: read_file(path))

        result = process(data)
        timer.finish()
        return result

    @mcp.tool()
    async def my_async_tool(url: str) -> str:
        timer = ToolTimer("my_async_tool")

        # For async I/O
        data = await measure_io_async(fetch_data(url))

        timer.finish()
        return data
"""

import time
import json
import tracemalloc
from typing import Callable, TypeVar, Awaitable
from functools import wraps

T = TypeVar("T")


class ToolTimer:
    """Timer for measuring tool execution time and memory."""

    # Class-level I/O tracking
    _io_ms: float = 0.0

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.start_time = time.perf_counter()

        # Reset I/O tracking
        ToolTimer._io_ms = 0.0

        # Start memory tracking
        tracemalloc.start()

    def finish(self) -> dict:
        """Finish timing and output results."""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        io_ms = ToolTimer._io_ms
        compute_ms = max(0, elapsed_ms - io_ms)

        result = {
            "tool": self.tool_name,
            "fn_total_ms": round(elapsed_ms, 2),
            "io_ms": round(io_ms, 2),
            "compute_ms": round(compute_ms, 2),
            "peak_memory_bytes": peak,
        }

        # Output in parseable format
        print(f"---TIMING---{json.dumps(result)}", flush=True)
        return result


def measure_io(func: Callable[[], T]) -> T:
    """Measure synchronous I/O operation time.

    Usage:
        data = measure_io(lambda: open(path).read())
    """
    start = time.perf_counter()
    try:
        return func()
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        ToolTimer._io_ms += elapsed


async def measure_io_async(awaitable: Awaitable[T]) -> T:
    """Measure asynchronous I/O operation time.

    Usage:
        data = await measure_io_async(client.get(url))
    """
    start = time.perf_counter()
    try:
        return await awaitable
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        ToolTimer._io_ms += elapsed


def timed_tool(tool_name: str):
    """Decorator for timing entire tool function.

    Usage:
        @mcp.tool()
        @timed_tool("my_tool")
        def my_tool(arg: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            timer = ToolTimer(tool_name)
            try:
                return func(*args, **kwargs)
            finally:
                timer.finish()

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            timer = ToolTimer(tool_name)
            try:
                return await func(*args, **kwargs)
            finally:
                timer.finish()

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
