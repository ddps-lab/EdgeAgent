#!/usr/bin/env python3
"""
Log Parser MCP Server

Provides independent, primitive tools for log processing.
Each tool is self-contained - LLM orchestrates the workflow.

Tools:
- parse_logs: Parse raw log content into structured entries
- filter_entries: Filter parsed entries by severity level
- compute_log_statistics: Compute statistics from entries
- search_entries: Search entries by pattern

Design Principle:
- Each tool is INDEPENDENT and takes explicit inputs
- NO internal tool-to-tool calls
- LLM decides the workflow: parse → filter → aggregate

Usage:
    python servers/log_parser_server.py
"""

import re
import json
import sys
import time
from typing import Literal, Any
from collections import Counter
from fastmcp import FastMCP

mcp = FastMCP("log_parser")

# Timing utilities
_tool_start_time = 0.0
_io_time = 0.0

def _reset_timing():
    global _tool_start_time, _io_time
    _tool_start_time = time.perf_counter()
    _io_time = 0.0

def _output_timing():
    global _tool_start_time, _io_time
    tool_exec_ms = (time.perf_counter() - _tool_start_time) * 1000
    print(f"---TOOL_EXEC---{tool_exec_ms:.3f}", file=sys.stderr)
    print(f"---IO---{_io_time:.3f}", file=sys.stderr)

# Common log patterns
LOG_PATTERNS = {
    "apache_combined": re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)"'
    ),
    "apache_common": re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)'
    ),
    "nginx": re.compile(
        r'(?P<ip>\S+) - \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)"'
    ),
    "syslog": re.compile(
        r"(?P<time>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(?P<host>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)"
    ),
    "python": re.compile(
        r"(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<logger>\S+)\s+-\s+(?P<level>\S+)\s+-\s+(?P<message>.*)"
    ),
}

SEVERITY_LEVELS = {
    "debug": 0,
    "info": 1,
    "notice": 2,
    "warning": 3,
    "warn": 3,
    "error": 4,
    "err": 4,
    "critical": 5,
    "crit": 5,
    "alert": 6,
    "emergency": 7,
    "emerg": 7,
}


def _detect_log_format(lines: list[str]) -> str:
    """Auto-detect log format from sample lines"""
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue

        # Try JSON first
        try:
            json.loads(line)
            return "json"
        except json.JSONDecodeError:
            pass

        # Try regex patterns
        for format_name, pattern in LOG_PATTERNS.items():
            if pattern.match(line):
                return format_name

    return "unknown"


def _parse_line(line: str, format_type: str) -> dict | None:
    """Parse a single log line"""
    line = line.strip()
    if not line:
        return None

    if format_type == "json":
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    pattern = LOG_PATTERNS.get(format_type)
    if pattern:
        match = pattern.match(line)
        if match:
            return match.groupdict()

    # Fallback: return raw line
    return {"raw": line, "parsed": False}


def _extract_level(parsed: dict) -> str:
    """Extract log level from parsed log entry"""
    # Check common level field names
    for field in ["level", "severity", "loglevel", "log_level"]:
        if field in parsed:
            return str(parsed[field]).lower()

    # Check status code for HTTP logs
    if "status" in parsed:
        status = int(parsed["status"])
        if status >= 500:
            return "error"
        elif status >= 400:
            return "warning"
        return "info"

    # Check message for level keywords
    message = parsed.get("message", parsed.get("raw", "")).lower()
    for level in ["error", "warning", "warn", "critical", "crit", "debug", "info"]:
        if level in message:
            return level

    return "info"


# ============================================================
# PRIMITIVE TOOLS - Each is independent, LLM orchestrates
# ============================================================

@mcp.tool()
def parse_logs(
    log_content: str,
    format_type: Literal[
        "auto", "apache_combined", "apache_common", "nginx", "syslog", "python", "json"
    ] = "auto",
    max_entries: int = 1000,
) -> dict:
    """
    Parse raw log content into structured entries.

    This is typically the FIRST step in log analysis.
    Output can be passed to filter_entries or compute_log_statistics.

    Args:
        log_content: Raw log content (multi-line string)
        format_type: Log format type or "auto" for auto-detection
        max_entries: Maximum number of entries to parse

    Returns:
        Dictionary with:
        - entries: List of parsed log entries (each with _level field)
        - format_detected: Detected log format
        - parsed_count: Number of successfully parsed entries
        - error_count: Number of lines that couldn't be parsed
    """
    _reset_timing()

    lines = log_content.split("\n")

    # Auto-detect format if needed
    detected_format = format_type
    if format_type == "auto":
        detected_format = _detect_log_format(lines)

    parsed_entries = []
    errors = 0

    for line in lines[:max_entries]:
        result = _parse_line(line, detected_format)
        if result:
            result["_level"] = _extract_level(result)
            parsed_entries.append(result)
        elif line.strip():  # Only count non-empty lines as errors
            errors += 1

    result = {
        "format_detected": detected_format,
        "total_lines": len(lines),
        "parsed_count": len(parsed_entries),
        "error_count": errors,
        "entries": parsed_entries,
    }

    _output_timing()
    return result


@mcp.tool()
def filter_entries(
    entries: list,
    min_level: Literal[
        "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
    ] = "warning",
    include_levels: list[str] | None = None,
) -> dict:
    """
    Filter log entries by severity level.

    IMPORTANT: This tool requires the 'entries' parameter from parse_logs output.
    You MUST call parse_logs first and pass its result['entries'] to this tool.

    Example workflow:
    1. First call parse_logs(log_content="...") -> returns {"entries": [...], ...}
    2. Then call filter_entries(entries=<result from step 1>["entries"], min_level="warning")

    Args:
        entries: REQUIRED. List of parsed log entries - get this from parse_logs() result["entries"].
                 This is a list of dictionaries, each containing log entry data.
        min_level: Minimum severity level to include. Options: debug, info, notice, warning, error, critical, alert, emergency.
                   Default is "warning" to get warnings and above (warning, error, critical, etc.)
        include_levels: Optional. Specific levels to include (overrides min_level).
                       Example: ["error", "critical"] to only get errors and critical.

    Returns:
        Dictionary with:
        - entries: Filtered entries (list of dicts)
        - original_count: Number of input entries
        - filtered_count: Number of entries after filtering
        - by_level: Count per severity level
        - levels_included: Which levels are in the result
    """
    _reset_timing()

    if include_levels:
        # Filter by specific levels
        target_levels = set(level.lower() for level in include_levels)
        filtered = [
            entry for entry in entries
            if entry.get("_level", "info") in target_levels
        ]
    else:
        # Filter by minimum level
        min_severity = SEVERITY_LEVELS.get(min_level, 0)
        filtered = [
            entry for entry in entries
            if SEVERITY_LEVELS.get(entry.get("_level", "info"), 1) >= min_severity
        ]

    # Count levels in result
    level_counts = Counter(entry.get("_level", "unknown") for entry in filtered)

    result = {
        "original_count": len(entries),
        "filtered_count": len(filtered),
        "levels_included": list(level_counts.keys()),
        "by_level": dict(level_counts),
        "entries": filtered,
    }

    _output_timing()
    return result


@mcp.tool()
def compute_log_statistics(entries: list) -> dict:
    """
    Compute statistics from parsed log entries.

    IMPORTANT: This tool requires the 'entries' parameter from parse_logs or filter_entries output.
    You MUST call parse_logs first to get the entries list.

    Example workflow:
    1. parse_logs(log_content="...") -> result with entries
    2. compute_log_statistics(entries=<result>["entries"])

    Or with filtering:
    1. parse_logs(log_content="...") -> parsed_result
    2. filter_entries(entries=parsed_result["entries"]) -> filtered_result
    3. compute_log_statistics(entries=filtered_result["entries"])

    Args:
        entries: REQUIRED. List of parsed log entries - get this from parse_logs()["entries"]
                 or filter_entries()["entries"]. This is a list of dictionaries.

    Returns:
        Dictionary with statistics:
        - entry_count: Total entries analyzed
        - by_level: Count per severity level (e.g., {"error": 5, "warning": 3})
        - by_status: Count per HTTP status (if applicable)
        - top_ips: Most frequent IP addresses (if applicable)
        - top_paths: Most frequent paths (if applicable)
    """
    _reset_timing()

    if not entries:
        _output_timing()
        return {"entry_count": 0, "by_level": {}}

    level_counts = Counter(entry.get("_level", "unknown") for entry in entries)

    # HTTP-specific stats
    status_counts = Counter()
    ip_counts = Counter()
    path_counts = Counter()

    for entry in entries:
        if "status" in entry:
            status_counts[entry["status"]] += 1
        if "ip" in entry:
            ip_counts[entry["ip"]] += 1
        if "path" in entry:
            path_counts[entry["path"]] += 1

    result = {
        "entry_count": len(entries),
        "by_level": dict(level_counts),
        "by_status": dict(status_counts.most_common(10)) if status_counts else None,
        "top_ips": dict(ip_counts.most_common(10)) if ip_counts else None,
        "top_paths": dict(path_counts.most_common(10)) if path_counts else None,
    }

    _output_timing()
    return result


@mcp.tool()
def search_entries(
    entries: list,
    pattern: str,
    fields: list[str] | None = None,
    case_sensitive: bool = False,
) -> dict:
    """
    Search log entries by regex pattern.

    IMPORTANT: This tool requires the 'entries' parameter from parse_logs output.
    You MUST call parse_logs first and pass its result['entries'] to this tool.

    Example workflow:
    1. parse_logs(log_content="...") -> result
    2. search_entries(entries=result["entries"], pattern="error|exception")

    Args:
        entries: REQUIRED. List of parsed log entries - get this from parse_logs()["entries"].
        pattern: Regex pattern to search for (e.g., "error", "timeout", "failed.*connection")
        fields: Optional. Specific fields to search in. Default: ["message", "raw"]
        case_sensitive: Whether search is case-sensitive. Default: False

    Returns:
        Dictionary with:
        - matches: List of matching entries with match details
        - match_count: Number of matches found
        - search_pattern: The pattern used
        - total_entries: Total entries searched
    """
    _reset_timing()

    if fields is None:
        fields = ["message", "raw"]

    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)

    matches = []
    for entry in entries:
        for field in fields:
            text = entry.get(field, "")
            if text and regex.search(str(text)):
                matches.append({
                    "entry": entry,
                    "matched_field": field,
                    "matched_text": str(text)[:200],  # Truncate for output
                })
                break  # One match per entry is enough

    result = {
        "search_pattern": pattern,
        "fields_searched": fields,
        "total_entries": len(entries),
        "match_count": len(matches),
        "matches": matches[:100],  # Limit output
    }

    _output_timing()
    return result


@mcp.tool()
def extract_time_range(entries: list) -> dict:
    """
    Extract time range information from log entries.

    IMPORTANT: This tool requires the 'entries' parameter from parse_logs output.
    You MUST call parse_logs first and pass its result['entries'] to this tool.

    Example workflow:
    1. parse_logs(log_content="...") -> result
    2. extract_time_range(entries=result["entries"])

    Args:
        entries: REQUIRED. List of parsed log entries - get this from parse_logs()["entries"].

    Returns:
        Dictionary with time range info:
        - has_timestamps: Whether timestamps were found
        - first_timestamp: First timestamp in logs
        - last_timestamp: Last timestamp in logs
        - entry_count: Number of entries analyzed
    """
    _reset_timing()

    times = []
    for entry in entries:
        time_val = entry.get("time", entry.get("timestamp"))
        if time_val:
            times.append(str(time_val))

    if not times:
        _output_timing()
        return {"has_timestamps": False, "entry_count": len(entries)}

    result = {
        "has_timestamps": True,
        "entry_count": len(entries),
        "first_timestamp": times[0] if times else None,
        "last_timestamp": times[-1] if times else None,
        "sample_timestamps": times[:5],
    }

    _output_timing()
    return result


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        # Streamable HTTP for serverless/remote deployment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        # stdio for local development / Claude Desktop
        mcp.run()
