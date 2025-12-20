#!/usr/bin/env python3
"""
Timing utilities for Native MCP servers profiling.

Matches WASM timing format exactly:
- ---TIMING---{"tool":"name","fn_total_ms":X,"io_ms":Y,"compute_ms":Z}

Usage:
    from timing import ToolTimer, measure_io

    @mcp.tool()
    def my_tool(path: str) -> dict:
        timer = ToolTimer("my_tool")

        # I/O operation - wrap with measure_io
        content = measure_io(lambda: open(path).read())

        # Compute (automatically tracked)
        result = process(content)

        timer.finish()
        return result
"""

import json
import time
import threading
from contextlib import contextmanager
from typing import Callable, TypeVar, Any
from pathlib import Path

T = TypeVar('T')

# Thread-local storage for I/O accumulator
_timing_data = threading.local()

# Output file (FastMCP captures stderr, so we use a file)
TIMING_FILE = Path("/tmp/mcp_timing.json")


def _get_io_accumulator() -> float:
    """Get thread-local I/O accumulator"""
    if not hasattr(_timing_data, 'io_ms'):
        _timing_data.io_ms = 0.0
    return _timing_data.io_ms


def _reset_io_accumulator():
    """Reset I/O accumulator"""
    _timing_data.io_ms = 0.0


def _add_io_time(ms: float):
    """Add to I/O accumulator"""
    if not hasattr(_timing_data, 'io_ms'):
        _timing_data.io_ms = 0.0
    _timing_data.io_ms += ms


def measure_io(func: Callable[[], T]) -> T:
    """
    Measure an I/O operation and add to the accumulator.

    Usage:
        content = measure_io(lambda: open(path).read())
        data = measure_io(lambda: requests.get(url).json())
    """
    start = time.perf_counter()
    try:
        return func()
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _add_io_time(elapsed_ms)


@contextmanager
def io_timer():
    """
    Context manager for I/O timing.

    Usage:
        with io_timer():
            content = open(path).read()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _add_io_time(elapsed_ms)


class ToolTimer:
    """
    Timer for measuring tool execution (matches WASM ToolTimer).

    Usage:
        def my_tool():
            timer = ToolTimer("my_tool")
            # ... do work, use measure_io() for I/O ...
            timer.finish()
            return result
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.start_time = time.perf_counter()
        _reset_io_accumulator()

    def finish(self) -> dict:
        """Finish timing and output results"""
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

        # Write to both file and stdout (with flush)
        import sys
        try:
            TIMING_FILE.write_text(json.dumps(timing))
        except Exception:
            pass

        # Print to stdout with explicit flush for Docker log capture
        # Using stdout because FastMCP/uvicorn may capture stderr
        timing_line = f"---TIMING---{json.dumps(timing)}"
        print(timing_line, file=sys.stdout, flush=True)
        # Also try stderr as backup
        print(timing_line, file=sys.stderr, flush=True)

        return timing


# Global server start time for total timing
_server_start_time: float = 0.0


def mark_server_start():
    """Mark server start time (call at beginning of main)"""
    global _server_start_time
    _server_start_time = time.perf_counter()


def get_native_total_ms() -> float:
    """Get total time since server start"""
    global _server_start_time
    if _server_start_time > 0:
        return (time.perf_counter() - _server_start_time) * 1000
    return 0.0
