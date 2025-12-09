#!/usr/bin/env python3
"""
Scenario 1: Code Review Pipeline - With Unified Metrics Collection

Tool Chain:
    filesystem(search) -> git(diff) -> git(log) -> summarize -> data_aggregate -> filesystem(write)
    DEVICE              DEVICE        DEVICE      EDGE        EDGE             DEVICE

This scenario demonstrates:
- Local Git repository analysis (with real bug data from Defects4J)
- Recent commit history extraction
- Code changes summarization
- Review report generation

Data Sources:
- Primary: Defects4J dataset (download via scripts/download_public_datasets.py -s 1)
  - Real bugs from Apache Commons Lang, Math, Time, Closure, Mockito
- Fallback: Generated sample repository (scripts/generate_test_repo.py)
"""

import asyncio
import json
import shutil
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


class CodeReviewScenario(ScenarioRunner):
    """Code Review Pipeline Scenario"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._repo_path = "/tmp/edgeagent_device/repo"
        self._report_path = "/tmp/edgeagent_device/code_review_report.md"
        # Ensure device directory exists
        Path("/tmp/edgeagent_device").mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "code_review"

    @property
    def description(self) -> str:
        return "Analyze Git repository changes and generate code review report"

    @property
    def user_request(self) -> str:
        return "Review recent code changes in the repository and generate a detailed review report"

    def get_validation_context(self) -> dict:
        """Provide context for validation"""
        return {
            "report_path": self._report_path,
            "repo_path": self._repo_path,
        }

    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """Execute the code review pipeline"""

        # Find tools by name
        tool_by_name = {t.name: t for t in tools}

        print("Available tools:")
        for name in sorted(tool_by_name.keys()):
            print(f"  - {name}")
        print()

        # Prepare Git repository - try Defects4J first, then sample_repo
        data_dir = Path(__file__).parent.parent / "data" / "scenario1"
        defects4j_dir = data_dir / "defects4j"
        sample_repo = data_dir / "sample_repo"

        # Check for available data sources
        # Defects4J: check for subdirectory with .git (e.g., defects4j/lang/)
        repo_source = None
        data_source = None

        if defects4j_dir.exists():
            # Look for git repo in subdirectories (lang, math, etc.)
            for subdir in defects4j_dir.iterdir():
                if subdir.is_dir() and (subdir / ".git").exists():
                    repo_source = subdir
                    data_source = f"Defects4J ({subdir.name})"
                    break

        if repo_source is None and sample_repo.exists() and (sample_repo / ".git").exists():
            repo_source = sample_repo
            data_source = "Generated sample repository"

        if repo_source is None:
            raise FileNotFoundError(
                f"No Git repository found in {data_dir}\n"
                "Run 'python scripts/download_public_datasets.py -s 1' for Defects4J, or\n"
                "Run 'python scripts/generate_test_repo.py' for sample repository"
            )

        print(f"Data Source: {data_source}")

        device_repo = Path(self._repo_path)
        if device_repo.exists():
            shutil.rmtree(device_repo)
        shutil.copytree(repo_source, device_repo)

        print(f"Prepared repository at: {device_repo}")
        print()

        # Step 1: List repository files (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 1: List repository files (filesystem -> DEVICE)")
        print("-" * 70)

        list_tool = tool_by_name.get("list_directory") or tool_by_name.get("search_files")
        if list_tool:
            raw_files = await list_tool.ainvoke({
                "path": str(device_repo),
            })
            files_result = parse_tool_result(raw_files)
        else:
            files_result = {"files": list(device_repo.rglob("*.py"))}

        print(f"  Found files in repository")
        if client.metrics_collector:
            client.metrics_collector.add_custom_metric("repo_path", str(device_repo))
            client.metrics_collector.add_custom_metric("data_source", data_source)
        print()

        # Step 2: Get git status (git -> DEVICE)
        print("-" * 70)
        print("Step 2: Get git status (git -> DEVICE)")
        print("-" * 70)

        status_tool = tool_by_name.get("git_status")
        if status_tool:
            raw_status = await status_tool.ainvoke({
                "repo_path": str(device_repo),
            })
            status_result = parse_tool_result(raw_status)
            print(f"  Git status retrieved")
        else:
            status_result = {"status": "clean"}
            print("  [SKIP] git_status tool not available")
        print()

        # Step 3: Get git log (git -> DEVICE)
        print("-" * 70)
        print("Step 3: Get commit history (git -> DEVICE)")
        print("-" * 70)

        log_tool = tool_by_name.get("git_log")
        if log_tool:
            raw_log = await log_tool.ainvoke({
                "repo_path": str(device_repo),
                "max_count": 10,
            })
            log_result = parse_tool_result(raw_log)
            commits = log_result if isinstance(log_result, list) else log_result.get("commits", [])
            commit_count = len(commits) if isinstance(commits, list) else 0
            print(f"  Retrieved {commit_count} commits")

            if client.metrics_collector:
                client.metrics_collector.add_custom_metric("commit_count", commit_count)
        else:
            log_result = {"raw": "No git_log tool available"}
            commits = []
            print("  [SKIP] git_log tool not available")
        print()

        # Step 4: Get git diff (git -> DEVICE)
        print("-" * 70)
        print("Step 4: Get code diff (git -> DEVICE)")
        print("-" * 70)

        diff_tool = tool_by_name.get("git_diff")
        diff_content = ""
        diff_lines = 0

        if diff_tool:
            try:
                # Get diff - repo_path and target are required
                raw_diff = await diff_tool.ainvoke({
                    "repo_path": str(device_repo),
                    "target": "HEAD~3",  # Compare last 3 commits
                })
                diff_result = parse_tool_result(raw_diff)
                diff_content = diff_result.get("diff", str(raw_diff)) if isinstance(diff_result, dict) else str(raw_diff)
                diff_lines = len(str(diff_content).split('\n'))
                print(f"  Retrieved diff ({diff_lines} lines)")

                if client.metrics_collector:
                    client.metrics_collector.add_custom_metric("diff_lines", diff_lines)
            except Exception as e:
                print(f"  [ERROR] git_diff failed: {e}")
                diff_content = ""
        else:
            print("  [SKIP] git_diff tool not available")
        print()

        # Step 5: Summarize changes (summarize -> EDGE)
        print("-" * 70)
        print("Step 5: Summarize code changes (summarize -> EDGE)")
        print("-" * 70)

        summarize_tool = tool_by_name.get("summarize_text")

        # Prepare content to summarize
        log_text = str(log_result) if log_result else ""
        content_to_summarize = f"Git Log:\n{log_text}\n\nDiff:\n{diff_content[:5000]}"  # Limit diff size

        if summarize_tool:
            try:
                raw_summary = await summarize_tool.ainvoke({
                    "text": content_to_summarize,
                    "max_length": 200,
                    "style": "detailed",
                })
                summary = str(raw_summary)
                print(f"  Generated summary ({len(summary)} chars)")
            except Exception as e:
                summary = f"Summary generation failed: {e}"
                print(f"  [ERROR] {e}")
        else:
            # Fallback: basic summary
            summary = f"Repository contains {commit_count} commits with code changes."
            print("  [SKIP] summarize_text tool not available, using basic summary")
        print()

        # Step 6: Aggregate review data (data_aggregate -> EDGE)
        print("-" * 70)
        print("Step 6: Aggregate review data (data_aggregate -> EDGE)")
        print("-" * 70)

        aggregate_tool = tool_by_name.get("aggregate_list")

        # Prepare review items
        review_items = []
        if isinstance(commits, list):
            for commit in commits[:5]:  # Top 5 commits
                if isinstance(commit, dict):
                    review_items.append({
                        "type": "commit",
                        "hash": commit.get("hash", "unknown")[:7],
                        "message": commit.get("message", "No message"),
                        "author": commit.get("author", "Unknown"),
                    })

        if aggregate_tool and review_items:
            raw_agg = await aggregate_tool.ainvoke({
                "items": review_items,
                "group_by": "type",
            })
            agg_result = parse_tool_result(raw_agg)
            print(f"  Aggregated {len(review_items)} review items")
        else:
            agg_result = {"groups": {"commit": len(review_items)}}
            print("  Using basic aggregation")
        print()

        # Step 7: Generate and write report (filesystem -> DEVICE)
        print("-" * 70)
        print("Step 7: Write review report (filesystem -> DEVICE)")
        print("-" * 70)

        # Generate report
        report = f"""# Code Review Report

## Repository
- Path: {device_repo}
- Total Commits: {commit_count}

## Summary
{summary}

## Recent Commits
"""
        if isinstance(commits, list):
            for commit in commits[:5]:
                if isinstance(commit, dict):
                    report += f"- **{commit.get('hash', 'N/A')[:7]}**: {commit.get('message', 'No message')}\n"
                else:
                    report += f"- {commit}\n"

        # Extract changed files from diff
        changed_files = []
        if diff_content:
            import re
            # Match diff file headers like "diff --git a/file.py b/file.py" or "+++ b/file.py"
            file_patterns = re.findall(r'(?:diff --git a/|[\+\-]{3} [ab]/)([^\s\n]+)', str(diff_content))
            changed_files = list(set(f for f in file_patterns if f and not f.startswith('/dev/null')))

        report += f"""
## Code Review Findings

### Changes Overview
- Total diff lines: {diff_lines if 'diff_lines' in dir() else 'N/A'}
- Review items: {len(review_items)}

### Changed Files
"""
        if changed_files:
            for f in changed_files[:10]:  # Limit to 10 files
                report += f"- `{f}`\n"
        else:
            report += "- No file changes detected\n"

        report += f"""
### Recommendations
1. Review security changes in authentication module
2. Ensure test coverage for new API endpoints
3. Consider adding integration tests

## Conclusion
The repository shows active development with recent commits focusing on:
- Feature additions (API endpoints)
- Bug fixes (data processing efficiency)
- Security improvements (password hashing)
- Test coverage expansion

---
*Generated by EdgeAgent Code Review Pipeline*
"""

        write_tool = tool_by_name.get("write_file")
        output_path = "/tmp/edgeagent_device/code_review_report.md"
        if write_tool:
            await write_tool.ainvoke({
                "path": output_path,
                "content": report
            })
        else:
            Path(output_path).write_text(report)

        print(f"  Report written to: {output_path}")
        print(f"  Report size: {len(report)} bytes")
        print()

        # Data flow summary
        input_size = sum(f.stat().st_size for f in device_repo.rglob("*") if f.is_file())
        output_size = len(report)
        reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0

        print("=" * 70)
        print("Data Flow Summary")
        print("=" * 70)
        print(f"  Input size:  {input_size:,} bytes (repository)")
        print(f"  Output size: {output_size:,} bytes (report)")
        print(f"  Reduction:   {reduction:.1f}%")
        print()

        return report


async def main():
    """Run the Code Review scenario with metrics collection"""
    config_path = Path(__file__).parent.parent / "config" / "tools_scenario1.yaml"

    scenario = CodeReviewScenario(
        config_path=config_path,
        output_dir="results/scenario1",
    )

    result = await scenario.run(
        save_results=True,
        print_summary=True,
    )

    # Additional: Export metrics to CSV for pandas analysis
    if result.metrics:
        csv_path = result.metrics.save_csv("results/scenario1/metrics.csv")
        print(f"Metrics CSV saved to: {csv_path}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
