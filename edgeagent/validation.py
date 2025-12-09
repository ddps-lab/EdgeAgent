"""
Result Validation Framework for EdgeAgent Scenarios

Each scenario has specific validation criteria to ensure the task was completed correctly.
This module provides validators that check:
1. Output file existence
2. Output content structure
3. Expected data presence
4. Quality metrics (optional)
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of validating a scenario output"""

    passed: bool
    score: float  # 0.0 to 1.0
    checks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def add_check(self, name: str, passed: bool, message: str = "", weight: float = 1.0):
        """Add a validation check result"""
        self.checks.append({
            "name": name,
            "passed": passed,
            "message": message,
            "weight": weight,
        })
        if not passed:
            self.errors.append(f"{name}: {message}")

    def add_warning(self, message: str):
        """Add a warning (non-fatal issue)"""
        self.warnings.append(message)

    def compute_score(self) -> float:
        """Compute weighted score from checks"""
        if not self.checks:
            return 0.0

        total_weight = sum(c["weight"] for c in self.checks)
        passed_weight = sum(c["weight"] for c in self.checks if c["passed"])

        self.score = passed_weight / total_weight if total_weight > 0 else 0.0
        self.passed = self.score >= 0.8  # 80% threshold
        return self.score

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }

    def print_summary(self):
        """Print human-readable summary"""
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"Validation Result: [{status}] Score: {self.score:.1%}")
        print(f"{'='*60}")

        for check in self.checks:
            icon = "✓" if check["passed"] else "✗"
            print(f"  {icon} {check['name']}: {check['message']}")

        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  ⚠ {w}")

        if self.errors and not self.passed:
            print(f"\nErrors:")
            for e in self.errors:
                print(f"  ✗ {e}")


class ScenarioValidator(ABC):
    """Base class for scenario validators"""

    @property
    @abstractmethod
    def scenario_name(self) -> str:
        """Name of the scenario being validated"""
        pass

    @abstractmethod
    def validate(self, output: Any, context: dict) -> ValidationResult:
        """
        Validate the scenario output.

        Args:
            output: The final output from the scenario
            context: Additional context (e.g., input data, config)

        Returns:
            ValidationResult with pass/fail status and details
        """
        pass


class LogAnalysisValidator(ScenarioValidator):
    """Validator for S2: Log Analysis Pipeline"""

    @property
    def scenario_name(self) -> str:
        return "log_analysis"

    def validate(self, output: Any, context: dict) -> ValidationResult:
        result = ValidationResult(passed=False, score=0.0)

        # Check 1: Output file exists
        report_path = Path(context.get("report_path", "/tmp/edgeagent_device/log_report.md"))
        agent_report_path = Path("/tmp/edgeagent_device/agent_report.md")

        # Try both paths (script vs agent may use different names)
        actual_path = None
        if report_path.exists():
            actual_path = report_path
        elif agent_report_path.exists():
            actual_path = agent_report_path

        if actual_path:
            result.add_check("output_file_exists", True, f"Report found at {actual_path}")
            report_content = actual_path.read_text()
        else:
            result.add_check("output_file_exists", False, f"Report not found at {report_path}")
            result.compute_score()
            return result

        # Check 2: Report is not empty
        if len(report_content) > 100:
            result.add_check("report_not_empty", True, f"Report has {len(report_content)} bytes")
        else:
            result.add_check("report_not_empty", False, f"Report too short: {len(report_content)} bytes")

        # Check 3: Report contains expected sections
        expected_keywords = ["error", "warning", "log", "analysis", "summary"]
        found_keywords = [kw for kw in expected_keywords if kw.lower() in report_content.lower()]

        if len(found_keywords) >= 3:
            result.add_check("contains_expected_content", True,
                           f"Found keywords: {found_keywords}")
        else:
            result.add_check("contains_expected_content", False,
                           f"Missing keywords. Found only: {found_keywords}")

        # Check 4: Contains error/warning counts
        # Match patterns like "error: 6" or "6 errors" or "errors: 6"
        has_counts = bool(re.search(r'(error|warning|critical)[:\s]+\d+|\d+\s*(error|warning|critical)', report_content, re.IGNORECASE))
        result.add_check("contains_statistics", has_counts,
                        "Report includes error/warning counts" if has_counts else "No statistics found")

        # Check 5: Validate against input log if available
        input_log_path = context.get("input_log_path")
        if input_log_path and Path(input_log_path).exists():
            input_log = Path(input_log_path).read_text()

            # Count actual errors in input
            actual_errors = len(re.findall(r'ERROR', input_log, re.IGNORECASE))
            actual_warnings = len(re.findall(r'WARNING', input_log, re.IGNORECASE))

            result.details["input_error_count"] = actual_errors
            result.details["input_warning_count"] = actual_warnings

            # Check if report mentions similar numbers (allow some variance)
            mentioned_numbers = re.findall(r'\b(\d+)\b', report_content)
            mentioned_numbers = [int(n) for n in mentioned_numbers]

            error_mentioned = any(abs(n - actual_errors) <= 2 for n in mentioned_numbers)
            warning_mentioned = any(abs(n - actual_warnings) <= 2 for n in mentioned_numbers)

            if error_mentioned or warning_mentioned:
                result.add_check("accurate_counts", True,
                               f"Report counts match input (errors={actual_errors}, warnings={actual_warnings})")
            else:
                result.add_warning(f"Could not verify counts (input: {actual_errors} errors, {actual_warnings} warnings)")

        result.compute_score()
        return result


class ImageProcessingValidator(ScenarioValidator):
    """Validator for S4: Image Processing Pipeline"""

    @property
    def scenario_name(self) -> str:
        return "image_processing"

    def validate(self, output: Any, context: dict) -> ValidationResult:
        result = ValidationResult(passed=False, score=0.0)

        # Check 1: Output file exists
        report_path = Path(context.get("report_path", "/tmp/edgeagent_device/image_report.md"))
        agent_report_path = Path("/tmp/edgeagent_device/agent_image_report.md")

        actual_path = None
        if report_path.exists():
            actual_path = report_path
        elif agent_report_path.exists():
            actual_path = agent_report_path

        if actual_path:
            result.add_check("output_file_exists", True, f"Report found at {actual_path}")
            report_content = actual_path.read_text()
        else:
            result.add_check("output_file_exists", False, f"Report not found")
            result.compute_score()
            return result

        # Check 2: Report is not empty
        if len(report_content) > 100:
            result.add_check("report_not_empty", True, f"Report has {len(report_content)} bytes")
        else:
            result.add_check("report_not_empty", False, f"Report too short")

        # Check 3: Contains image processing keywords
        expected_keywords = ["image", "duplicate", "thumbnail", "resize", "hash"]
        found_keywords = [kw for kw in expected_keywords if kw.lower() in report_content.lower()]

        if len(found_keywords) >= 2:
            result.add_check("contains_expected_content", True,
                           f"Found keywords: {found_keywords}")
        else:
            result.add_check("contains_expected_content", False,
                           f"Missing keywords. Found: {found_keywords}")

        # Check 4: Contains numeric statistics
        has_stats = bool(re.search(r'\d+\s*(image|file|duplicate|unique)', report_content, re.IGNORECASE))
        result.add_check("contains_statistics", has_stats,
                        "Report includes image statistics" if has_stats else "No statistics found")

        # Check 5: Validate against input if available
        input_dir = context.get("input_dir", "/tmp/edgeagent_device/images")
        if Path(input_dir).exists():
            input_images = list(Path(input_dir).glob("*"))
            input_count = len([f for f in input_images if f.is_file()])

            result.details["input_image_count"] = input_count

            # Check if report mentions the image count
            if str(input_count) in report_content or str(input_count - 1) in report_content:
                result.add_check("accurate_count", True,
                               f"Report mentions correct image count (~{input_count})")
            else:
                result.add_warning(f"Could not verify image count (input: {input_count})")

        result.compute_score()
        return result


class CodeReviewValidator(ScenarioValidator):
    """Validator for S1: Code Review Pipeline"""

    @property
    def scenario_name(self) -> str:
        return "code_review"

    def validate(self, output: Any, context: dict) -> ValidationResult:
        result = ValidationResult(passed=False, score=0.0)

        # Check 1: Output file exists
        report_path = Path(context.get("report_path", "/tmp/edgeagent_device/code_review_report.md"))
        agent_report_path = Path("/tmp/edgeagent_device/agent_code_review.md")

        actual_path = None
        if report_path.exists():
            actual_path = report_path
        elif agent_report_path.exists():
            actual_path = agent_report_path

        if actual_path:
            result.add_check("output_file_exists", True, f"Report found at {actual_path}")
            report_content = actual_path.read_text()
        else:
            result.add_check("output_file_exists", False, f"Report not found")
            result.compute_score()
            return result

        # Check 2: Report is not empty
        if len(report_content) > 100:
            result.add_check("report_not_empty", True, f"Report has {len(report_content)} bytes")
        else:
            result.add_check("report_not_empty", False, f"Report too short")

        # Check 3: Contains code review keywords
        expected_keywords = ["commit", "change", "code", "review", "file", "diff"]
        found_keywords = [kw for kw in expected_keywords if kw.lower() in report_content.lower()]

        if len(found_keywords) >= 3:
            result.add_check("contains_expected_content", True,
                           f"Found keywords: {found_keywords}")
        else:
            result.add_check("contains_expected_content", False,
                           f"Missing keywords. Found: {found_keywords}")

        # Check 4: Contains file references or code snippets
        has_code_refs = bool(re.search(r'\.(py|js|ts|java|go|rs|cpp|c|h)\b', report_content))
        result.add_check("references_code_files", has_code_refs,
                        "Report references code files" if has_code_refs else "No code file references")

        result.compute_score()
        return result


class ResearchAssistantValidator(ScenarioValidator):
    """Validator for S3: Research Assistant Pipeline"""

    @property
    def scenario_name(self) -> str:
        return "research_assistant"

    def validate(self, output: Any, context: dict) -> ValidationResult:
        result = ValidationResult(passed=False, score=0.0)

        # Check 1: Output file exists
        report_path = Path(context.get("report_path", "/tmp/edgeagent_device/research_report.md"))
        agent_report_path = Path("/tmp/edgeagent_device/agent_research_report.md")

        actual_path = None
        if report_path.exists():
            actual_path = report_path
        elif agent_report_path.exists():
            actual_path = agent_report_path

        if actual_path:
            result.add_check("output_file_exists", True, f"Report found at {actual_path}")
            report_content = actual_path.read_text()
        else:
            result.add_check("output_file_exists", False, f"Report not found")
            result.compute_score()
            return result

        # Check 2: Report is not empty
        if len(report_content) > 200:
            result.add_check("report_not_empty", True, f"Report has {len(report_content)} bytes")
        else:
            result.add_check("report_not_empty", False, f"Report too short")

        # Check 3: Contains research keywords
        expected_keywords = ["agent", "llm", "language model", "ai", "research", "summary"]
        found_keywords = [kw for kw in expected_keywords if kw.lower() in report_content.lower()]

        if len(found_keywords) >= 2:
            result.add_check("contains_expected_content", True,
                           f"Found keywords: {found_keywords}")
        else:
            result.add_check("contains_expected_content", False,
                           f"Missing keywords. Found: {found_keywords}")

        # Check 4: Contains source references
        urls = context.get("urls", [])
        has_refs = bool(re.search(r'(wikipedia|source|reference|http)', report_content, re.IGNORECASE))
        result.add_check("contains_references", has_refs,
                        "Report includes source references" if has_refs else "No source references")

        # Check 5: Multiple sections/summaries
        section_markers = len(re.findall(r'^#{1,3}\s+', report_content, re.MULTILINE))
        if section_markers >= 2:
            result.add_check("structured_report", True,
                           f"Report has {section_markers} sections")
        else:
            result.add_warning("Report may lack proper structure")

        result.compute_score()
        return result


# Registry of validators
VALIDATORS = {
    "log_analysis": LogAnalysisValidator(),
    "log_analysis_agent": LogAnalysisValidator(),
    "image_processing": ImageProcessingValidator(),
    "image_processing_agent": ImageProcessingValidator(),
    "code_review": CodeReviewValidator(),
    "code_review_agent": CodeReviewValidator(),
    "research_assistant": ResearchAssistantValidator(),
    "research_assistant_agent": ResearchAssistantValidator(),
}


def get_validator(scenario_name: str) -> Optional[ScenarioValidator]:
    """Get validator for a scenario by name"""
    # Try exact match first
    if scenario_name in VALIDATORS:
        return VALIDATORS[scenario_name]

    # Try partial match
    for key, validator in VALIDATORS.items():
        if key in scenario_name.lower() or scenario_name.lower() in key:
            return validator

    return None


def validate_scenario(scenario_name: str, output: Any, context: dict) -> ValidationResult:
    """
    Validate a scenario's output.

    Args:
        scenario_name: Name of the scenario
        output: Final output from the scenario
        context: Additional context (input paths, config, etc.)

    Returns:
        ValidationResult with pass/fail status and details
    """
    validator = get_validator(scenario_name)

    if validator is None:
        result = ValidationResult(passed=False, score=0.0)
        result.add_check("validator_found", False, f"No validator for scenario: {scenario_name}")
        result.compute_score()
        return result

    return validator.validate(output, context)
