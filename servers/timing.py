#!/usr/bin/env python3
"""
Timing utilities for Native MCP servers profiling.

Provides timing markers compatible with WASM profiling format:
- ---NATIVE_TOTAL---{ms}  : Total time inside server (excludes process spawn)
- ---TOOL_EXEC---{ms}     : Tool execution time
- ---IO---{ms}            : Accumulated I/O time (disk + network)

Usage:
    from timing import TimingContext, io_timer

    @mcp.tool()
    def my_tool(arg: str) -> dict:
        with TimingContext("my_tool") as ctx:
            # I/O operation
            with ctx.io_timer():
                data = read_file(path)

            # Compute
            result = process(data)

            return result
"""

import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any
import threading

# Thread-local storage for timing data
_timing_data = threading.local()


def _get_timing():
    """Get thread-local timing data"""
    if not hasattr(_timing_data, 'io_time'):
        _timing_data.io_time = 0.0
        _timing_data.tool_start = 0.0
    return _timing_data


def reset_timing():
    """Reset timing counters"""
    data = _get_timing()
    data.io_time = 0.0
    data.tool_start = time.perf_counter()


def add_io_time(ms: float):
    """Add I/O time to accumulator"""
    _get_timing().io_time += ms


def get_io_time() -> float:
    """Get accumulated I/O time"""
    return _get_timing().io_time


class TimingContext:
    """Context manager for tool timing"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.start_time = 0.0
        self.io_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.io_time = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        tool_exec_ms = (end_time - self.start_time) * 1000

        # Output timing markers to stderr (same as WASM)
        print(f"---TOOL_EXEC---{tool_exec_ms:.3f}", file=sys.stderr)
        print(f"---IO---{self.io_time:.3f}", file=sys.stderr)

        return False

    @contextmanager
    def io_timer(self):
        """Context manager for I/O timing"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.io_time += elapsed_ms


@contextmanager
def io_timer():
    """Standalone I/O timer context manager"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        add_io_time(elapsed_ms)


def timed_tool(func: Callable) -> Callable:
    """Decorator for timing MCP tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        reset_timing()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            tool_exec_ms = (end_time - start_time) * 1000
            io_time_ms = get_io_time()

            # Output timing markers to stderr
            print(f"---TOOL_EXEC---{tool_exec_ms:.3f}", file=sys.stderr)
            print(f"---IO---{io_time_ms:.3f}", file=sys.stderr)

    return wrapper


class NativeTimingServer:
    """Mixin for adding timing to FastMCP server"""

    _server_start_time: float = 0.0

    @classmethod
    def mark_server_start(cls):
        """Mark server start time (call at beginning of main)"""
        cls._server_start_time = time.perf_counter()

    @classmethod
    def output_total_time(cls):
        """Output total time marker (call at end of request)"""
        if cls._server_start_time > 0:
            total_ms = (time.perf_counter() - cls._server_start_time) * 1000
            print(f"---NATIVE_TOTAL---{total_ms:.3f}", file=sys.stderr)
