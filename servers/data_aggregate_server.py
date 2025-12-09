#!/usr/bin/env python3
"""
Data Aggregate MCP Server

Aggregates, merges, and summarizes structured data from multiple sources.
Used in S2 (Log Analysis), S3 (Research), and S4 (Data Processing) scenarios.

This server is optimized for EDGE deployment with significant data REDUCTION.
Input: Multiple data sources (lists, dicts, summaries)
Output: Aggregated summary (typically 1-10% of input)

Usage:
    python servers/data_aggregate_server.py
"""

import json
from collections import Counter, defaultdict
from typing import Any
from statistics import mean, median, stdev
from fastmcp import FastMCP

mcp = FastMCP("data_aggregate")


def _safe_numeric(value: Any) -> float | None:
    """Safely convert value to numeric"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _compute_stats(values: list[float]) -> dict:
    """Compute statistics for numeric values"""
    if not values:
        return {"count": 0}

    result = {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }

    if len(values) > 1:
        result["stdev"] = stdev(values)

    return result


@mcp.tool()
def aggregate_list(
    items: list[dict],
    group_by: str | None = None,
    count_field: str | None = None,
    sum_fields: list[str] | None = None,
) -> dict:
    """
    Aggregate a list of dictionaries by grouping, counting, or summing.

    This tool takes a list of dictionaries (e.g., from parse_logs, filter_entries, or any other source)
    and produces aggregated statistics.

    Example usage for log analysis:
    1. parse_logs(log_content="...") -> result with entries
    2. filter_entries(entries=result["entries"], min_level="warning") -> filtered
    3. aggregate_list(items=filtered["entries"], group_by="_level")

    Example usage for general aggregation:
    - aggregate_list(items=[{"type": "A", "value": 10}, {"type": "B", "value": 20}], group_by="type")

    Args:
        items: REQUIRED. List of dictionaries to aggregate. This can be:
               - Log entries from parse_logs()["entries"] or filter_entries()["entries"]
               - Any other list of dictionaries you want to aggregate
        group_by: Optional. Field name to group by (e.g., "_level" for log levels, "category" for categories)
        count_field: Optional. Field to count occurrences of unique values
        sum_fields: Optional. List of field names with numeric values to compute sum/avg/min/max statistics

    Returns:
        Dictionary with:
        - total_count: Total number of items
        - groups: Count per group (if group_by specified)
        - counts: Value counts (if count_field specified)
        - field_stats: Statistics for each sum_field (if sum_fields specified)
        - reduction_ratio: Output size / Input size (shows data reduction achieved)
    """
    if not items:
        return {"total_count": 0, "groups": {}}

    result = {
        "total_count": len(items),
        "input_size_estimate": len(json.dumps(items)),
    }

    # Group by field
    if group_by:
        groups = defaultdict(list)
        for item in items:
            key = str(item.get(group_by, "unknown"))
            groups[key].append(item)

        result["group_by"] = group_by
        result["groups"] = {k: len(v) for k, v in groups.items()}
        result["group_count"] = len(groups)

    # Count field occurrences
    if count_field:
        counts = Counter(str(item.get(count_field, "unknown")) for item in items)
        result["count_by"] = count_field
        result["counts"] = dict(counts.most_common(20))

    # Sum numeric fields
    if sum_fields:
        sums = {}
        for field in sum_fields:
            values = [_safe_numeric(item.get(field)) for item in items]
            values = [v for v in values if v is not None]
            sums[field] = _compute_stats(values)
        result["field_stats"] = sums

    result["output_size_estimate"] = len(json.dumps(result))
    result["reduction_ratio"] = (
        result["output_size_estimate"] / result["input_size_estimate"]
        if result["input_size_estimate"] > 0
        else 0
    )

    return result


@mcp.tool()
def merge_summaries(summaries: list[dict], weights: list[float] | None = None) -> dict:
    """
    Merge multiple summary dictionaries into one.

    Args:
        summaries: List of summary dictionaries to merge
        weights: Optional weights for each summary (for weighted averages)

    Returns:
        Merged summary
    """
    if not summaries:
        return {"merged_count": 0}

    if weights is None:
        weights = [1.0] * len(summaries)

    # Collect all keys
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())

    merged = {"merged_count": len(summaries), "source_keys": list(all_keys)}

    # Merge each key
    for key in all_keys:
        values = []
        key_weights = []
        for s, w in zip(summaries, weights):
            if key in s:
                val = s[key]
                if isinstance(val, (int, float)):
                    values.append(val)
                    key_weights.append(w)
                elif isinstance(val, list):
                    values.extend(val)
                elif isinstance(val, dict):
                    # Nested dict - try to merge counts
                    if key not in merged:
                        merged[key] = defaultdict(float)
                    for k, v in val.items():
                        if isinstance(v, (int, float)):
                            merged[key][k] += v * w

        if values and all(isinstance(v, (int, float)) for v in values):
            # Compute weighted average for numeric values
            if sum(key_weights) > 0:
                weighted_sum = sum(v * w for v, w in zip(values, key_weights))
                merged[f"{key}_weighted_avg"] = weighted_sum / sum(key_weights)
            merged[f"{key}_total"] = sum(values)

    # Convert defaultdicts to regular dicts
    for key in list(merged.keys()):
        if isinstance(merged[key], defaultdict):
            merged[key] = dict(merged[key])

    return merged


@mcp.tool()
def combine_research_results(
    results: list[dict],
    title_field: str = "title",
    summary_field: str = "summary",
    score_field: str | None = "relevance_score",
) -> dict:
    """
    Combine multiple research/search results into a coherent summary.

    Args:
        results: List of research result dictionaries
        title_field: Field containing the title
        summary_field: Field containing the summary
        score_field: Optional field for relevance scoring

    Returns:
        Combined research summary
    """
    if not results:
        return {"result_count": 0, "combined_summary": ""}

    # Sort by score if available
    if score_field:
        sorted_results = sorted(
            results,
            key=lambda x: _safe_numeric(x.get(score_field, 0)) or 0,
            reverse=True,
        )
    else:
        sorted_results = results

    # Extract key information
    items = []
    for i, r in enumerate(sorted_results):
        item = {
            "rank": i + 1,
            "title": r.get(title_field, f"Result {i+1}"),
            "summary": r.get(summary_field, ""),
        }
        if score_field and score_field in r:
            item["score"] = r[score_field]
        items.append(item)

    # Create combined summary
    combined_text = "\n\n".join(
        f"[{item['rank']}] {item['title']}\n{item['summary']}" for item in items
    )

    return {
        "result_count": len(results),
        "items": items,
        "combined_text": combined_text,
        "input_size": sum(len(json.dumps(r)) for r in results),
        "output_size": len(combined_text),
    }


@mcp.tool()
def deduplicate(
    items: list[dict], key_fields: list[str], keep: str = "first"
) -> dict:
    """
    Remove duplicate items based on key fields.

    Args:
        items: List of items to deduplicate
        key_fields: Fields to use as the deduplication key
        keep: Which duplicate to keep ("first" or "last")

    Returns:
        Deduplicated items with statistics
    """
    if not items:
        return {"original_count": 0, "unique_count": 0, "items": []}

    seen = {}
    result = []

    for item in items:
        key = tuple(str(item.get(f, "")) for f in key_fields)

        if key not in seen:
            seen[key] = item
            if keep == "first":
                result.append(item)
        elif keep == "last":
            # Update the stored item
            seen[key] = item

    if keep == "last":
        result = list(seen.values())

    return {
        "original_count": len(items),
        "unique_count": len(result),
        "duplicates_removed": len(items) - len(result),
        "key_fields": key_fields,
        "items": result,
    }


@mcp.tool()
def compute_trends(
    time_series: list[dict],
    time_field: str = "timestamp",
    value_field: str = "value",
    bucket_count: int = 10,
) -> dict:
    """
    Compute trends from time-series data.

    Args:
        time_series: List of time-series data points
        time_field: Field containing the timestamp
        value_field: Field containing the value
        bucket_count: Number of time buckets

    Returns:
        Trend analysis
    """
    if not time_series:
        return {"data_points": 0, "trend": "insufficient_data"}

    # Extract and sort values
    data = []
    for item in time_series:
        value = _safe_numeric(item.get(value_field))
        if value is not None:
            data.append({"time": item.get(time_field), "value": value})

    if len(data) < 2:
        return {"data_points": len(data), "trend": "insufficient_data"}

    values = [d["value"] for d in data]

    # Simple trend analysis
    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]

    first_avg = mean(first_half) if first_half else 0
    second_avg = mean(second_half) if second_half else 0

    if second_avg > first_avg * 1.1:
        trend = "increasing"
    elif second_avg < first_avg * 0.9:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "data_points": len(data),
        "trend": trend,
        "stats": _compute_stats(values),
        "first_half_avg": first_avg,
        "second_half_avg": second_avg,
        "change_percent": (
            ((second_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
        ),
    }


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        # Streamable HTTP for serverless/remote deployment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8001"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        # stdio for local development / Claude Desktop
        mcp.run()
