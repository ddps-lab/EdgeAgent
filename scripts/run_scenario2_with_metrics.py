#!/usr/bin/env python3
"""
Scenario 2: Log Analysis Pipeline - With Unified Metrics Collection

Tool Chain:
    filesystem(read) -> log_parser -> data_aggregate -> filesystem(write)
    DEVICE            EDGE         EDGE             DEVICE

This script uses the ScenarioRunner framework for unified metrics collection.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeagent import ScenarioRunner, EdgeAgentMCPClient


def parse_tool_result(result):
    """Parse tool result - handle both dict and JSON string."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return {"raw": str(result)}


class LogAnalysisScenario(ScenarioRunner):
    """Log Analysis Pipeline Scenario"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_log_path = None
        self._report_path = "/tmp/edgeagent_device/log_report.md"
        # Ensure device directory exists
        Path("/tmp/edgeagent_device").mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "log_analysis"

    @property
    def description(self) -> str:
        return "Server log analysis and report generation"

    @property
    def user_request(self) -> str:
        return "Analyze server logs for errors and generate a summary report"

    def get_validation_context(self) -> dict:
        """Provide context for validation"""
        return {
            "report_path": self._report_path,
            "input_log_path": self._input_log_path,
        }

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute the log analysis pipeline"""

        # Find tools by name
        tool_by_name = {t.name: t for t in tools}

        print("Available tools:")
        for name in sorted(tool_by_name.keys()):
            print(f"  - {name}")
        print()

        # Prepare log file - try LogHub first, then sample server.log
        data_dir = Path(__file__).parent.parent / "data" / "scenario2"
        loghub_dir = data_dir / "loghub_samples"
        sample_log = data_dir / "server.log"

        # Check for LogHub data (prefer medium size for reasonable test)
        log_file = None
        data_source = None

        if loghub_dir.exists():
            # Priority: medium_python > small_python > any .log file
            for log_name in ["medium_python.log", "small_python.log"]:
                candidate = loghub_dir / log_name
                if candidate.exists():
                    log_file = candidate
                    data_source = f"LogHub ({log_name})"
                    break
            if log_file is None:
                # Fallback to any .log file in loghub_samples
                log_files = list(loghub_dir.glob("*.log"))
                if log_files:
                    log_file = log_files[0]
                    data_source = f"LogHub ({log_file.name})"

        if log_file is None and sample_log.exists():
            log_file = sample_log
            data_source = "Sample server.log"

        if log_file is None:
            raise FileNotFoundError(
                f"No log file found in {data_dir}\n"
                "Run 'python scripts/download_public_datasets.py -s 2' for LogHub data"
            )

        print(f"Data Source: {data_source}")
        print(f"Log file size: {log_file.stat().st_size:,} bytes")

        device_log = Path("/tmp/edgeagent_device/server.log")
        device_log.parent.mkdir(exist_ok=True)
        device_log.write_text(log_file.read_text())

        # Store input path for validation
        self._input_log_path = str(device_log)

        # Step 1: Read log file (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 1: Read log file (filesystem -> DEVICE)")
        print("-" * 70)

        read_tool = tool_by_name.get("read_file") or tool_by_name.get("read_text_file")
        if not read_tool:
            raise ValueError(f"read_file tool not found. Available: {list(tool_by_name.keys())}")

        log_content = await read_tool.ainvoke({"path": str(device_log)})
        print(f"  Read {len(str(log_content))} chars from {device_log}")
        print()

        # Step 2: Parse logs (log_parser -> EDGE)
        print("-" * 70)
        print("Step 2: Parse logs (log_parser -> EDGE)")
        print("-" * 70)

        parse_tool = tool_by_name.get("parse_logs")
        if parse_tool:
            raw_parsed = await parse_tool.ainvoke({
                "log_content": str(log_content),
                "format_type": "python"
            })
            parsed = parse_tool_result(raw_parsed)
        else:
            # Fallback: direct import
            from servers.log_parser_server import parse_logs
            parsed = parse_logs.fn(str(log_content), "python")

        print(f"  Format detected: {parsed.get('format_detected', 'N/A')}")
        print(f"  Parsed {parsed.get('parsed_count', 0)} entries")

        # Add custom metrics
        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("entries_parsed", parsed.get('parsed_count', 0))
            client.metrics_collector.add_custom_metric("data_source", data_source)
        print()

        # Step 3: Filter errors (log_parser -> EDGE)
        print("-" * 70)
        print("Step 3: Filter errors (log_parser -> EDGE)")
        print("-" * 70)

        filter_tool = tool_by_name.get("filter_entries")
        if filter_tool:
            raw_filtered = await filter_tool.ainvoke({
                "entries": parsed["entries"],
                "min_level": "warning"
            })
            filtered = parse_tool_result(raw_filtered)
        else:
            from servers.log_parser_server import filter_entries
            filtered = filter_entries.fn(parsed["entries"], min_level="warning")

        print(f"  Filtered to {filtered.get('filtered_count', 0)} entries (warnings+)")
        print(f"  By level: {filtered.get('by_level', {})}")

        # Add custom metrics
        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("entries_filtered", filtered.get('filtered_count', 0))
            client.metrics_collector.add_custom_metric("error_count_by_level", filtered.get('by_level', {}))
        print()

        # Step 4: Compute statistics (log_parser -> EDGE)
        print("-" * 70)
        print("Step 4: Compute statistics (log_parser -> EDGE)")
        print("-" * 70)

        stats_tool = tool_by_name.get("compute_log_statistics")
        if stats_tool:
            raw_stats = await stats_tool.ainvoke({"entries": filtered["entries"]})
            stats = parse_tool_result(raw_stats)
        else:
            from servers.log_parser_server import compute_log_statistics
            stats = compute_log_statistics.fn(filtered["entries"])

        print(f"  Total entries: {stats.get('entry_count', 0)}")
        print(f"  By level: {stats.get('by_level', {})}")
        print()

        # Step 5: Aggregate data (data_aggregate -> EDGE)
        print("-" * 70)
        print("Step 5: Aggregate data (data_aggregate -> EDGE)")
        print("-" * 70)

        aggregate_tool = tool_by_name.get("aggregate_list")
        if aggregate_tool:
            raw_aggregated = await aggregate_tool.ainvoke({
                "items": filtered["entries"],
                "group_by": "_level"
            })
            aggregated = parse_tool_result(raw_aggregated)
        else:
            from servers.data_aggregate_server import aggregate_list
            aggregated = aggregate_list.fn(filtered["entries"], group_by="_level")

        print(f"  Groups: {aggregated.get('group_count', 0)}")
        print(f"  Group names: {list(aggregated.get('groups', {}).keys())}")
        print()

        # Step 6: Write report (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 6: Write report (filesystem -> DEVICE)")
        print("-" * 70)

        # Generate report
        report = f"""# Log Analysis Report

## Summary
- Total entries analyzed: {parsed.get('parsed_count', 0)}
- Warnings and above: {filtered.get('filtered_count', 0)}
- Log format: {parsed.get('format_detected', 'N/A')}

## By Severity Level
{chr(10).join(f"- {level}: {count}" for level, count in stats.get('by_level', {}).items())}

## Grouped Analysis
{chr(10).join(f"### {level}: {count} entries" for level, count in aggregated.get('groups', {}).items())}

## Critical Issues
"""
        # Add critical/error entries
        for entry in filtered.get("entries", []):
            if entry.get("_level") in ("critical", "error"):
                report += f"- [{entry.get('_level', 'N/A').upper()}] {entry.get('message', entry.get('raw', 'N/A'))}\n"

        write_tool = tool_by_name.get("write_file")
        if write_tool:
            output_path = "/tmp/edgeagent_device/log_report.md"
            await write_tool.ainvoke({
                "path": output_path,
                "content": report
            })
        else:
            output_path = Path("/tmp/edgeagent_device/log_report.md")
            output_path.write_text(report)

        print(f"  Report written to: {output_path}")
        print(f"  Report size: {len(report)} bytes")
        print()

        # Data reduction stats
        input_size = len(log_file.read_text())
        output_size = len(report)
        reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0

        print("=" * 70)
        print("Data Flow Summary")
        print("=" * 70)
        print(f"  Input size:  {input_size:,} bytes")
        print(f"  Output size: {output_size:,} bytes")
        print(f"  Reduction:   {reduction:.1f}%")
        print()

        return report


async def main():
    """Run the Log Analysis scenario with metrics collection"""
    config_path = Path(__file__).parent.parent / "config" / "tools_scenario2.yaml"

    scenario = LogAnalysisScenario(
        config_path=config_path,
        output_dir="results/scenario2",
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional: Export metrics to CSV for pandas analysis
    if result.metrics:
        csv_path = result.metrics.save_csv("results/scenario2/metrics.csv")
        print(f"Metrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
