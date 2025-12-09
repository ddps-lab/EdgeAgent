"""
Scenario Runner Framework

Provides a unified base class for running EdgeAgent experiments/scenarios
with automatic metrics collection.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Load .env file if exists (for API keys)
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of edgeagent package)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

from .middleware import EdgeAgentMCPClient
from .metrics import MetricsCollector, MetricsConfig
from .validation import ValidationResult, validate_scenario


@dataclass
class ScenarioResult:
    """Result of a scenario run with metrics"""

    # Scenario info
    scenario_name: str
    description: str
    user_request: str

    # Timing
    start_time: float
    end_time: float

    # Status
    success: bool
    error: Optional[str] = None

    # Metrics
    metrics: Optional[MetricsCollector] = None

    # Output
    final_output: Any = None

    # Custom scenario-specific data
    custom_data: dict = field(default_factory=dict)

    # Validation
    validation: Optional[ValidationResult] = None
    validation_context: dict = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        """Total end-to-end latency"""
        return (self.end_time - self.start_time) * 1000

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls"""
        if self.metrics:
            return len(self.metrics.entries)
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "user_request": self.user_request,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_latency_ms": self.total_latency_ms,
            "tool_call_count": self.tool_call_count,
            "success": self.success,
            "error": self.error,
            "custom_data": self.custom_data,
        }

        # Add metrics if available
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()

        # Add output preview
        if self.final_output:
            output_str = str(self.final_output)
            result["final_output_preview"] = (
                output_str[:500] if len(output_str) > 500 else output_str
            )

        # Add validation results
        if self.validation:
            result["validation"] = self.validation.to_dict()

        return result

    def save(self, output_dir: str | Path) -> Path:
        """Save result to JSON file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.scenario_name}_{int(self.start_time)}.json"
        output_path = output_dir / filename

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return output_path

    def print_summary(self):
        """Print scenario result summary"""
        print(f"\n{'='*70}")
        print(f"Scenario: {self.scenario_name}")
        print(f"{'='*70}")
        print(f"Description: {self.description}")
        print(f"User Request: {self.user_request}")
        print(f"Success: {self.success}")
        print(f"Total Latency: {self.total_latency_ms:.2f} ms")
        print(f"Tool Calls: {self.tool_call_count}")

        if not self.success and self.error:
            print(f"\nError: {self.error}")

        # Print validation summary if available
        if self.validation:
            self.validation.print_summary()

        # Print metrics summary if available
        if self.metrics:
            print()
            self.metrics.print_summary()

        print(f"{'='*70}\n")


class ScenarioRunner(ABC):
    """
    Base class for EdgeAgent scenario runners.

    Subclasses implement the execute() method with scenario-specific logic.
    The run() method handles:
    - Client initialization
    - Metrics collection
    - Error handling
    - Result generation
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = "results",
        metrics_config: Optional[MetricsConfig] = None,
    ):
        """
        Args:
            config_path: Path to tools YAML config
            output_dir: Directory for saving results
            metrics_config: Metrics collection configuration
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.metrics_config = metrics_config

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name (e.g., 'log_analysis')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Scenario description"""
        pass

    @property
    @abstractmethod
    def user_request(self) -> str:
        """User request that triggers this scenario"""
        pass

    @abstractmethod
    async def execute(
        self,
        client: EdgeAgentMCPClient,
        tools: list,
    ) -> Any:
        """
        Execute the scenario-specific logic.

        Args:
            client: Initialized EdgeAgentMCPClient
            tools: List of available tools

        Returns:
            Final output of the scenario
        """
        pass

    def get_tool_by_name(self, tools: list, name: str) -> Optional[Any]:
        """Helper to find a tool by name"""
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    def get_validation_context(self) -> dict:
        """
        Get context for validation (override in subclass for custom context).

        Returns:
            Dictionary with validation context (e.g., input paths, expected values)
        """
        return {}

    async def run(
        self,
        save_results: bool = True,
        print_summary: bool = True,
        validate: bool = True,
    ) -> ScenarioResult:
        """
        Run the scenario with automatic metrics collection and validation.

        Args:
            save_results: Whether to save results to file
            print_summary: Whether to print summary to console
            validate: Whether to validate the output

        Returns:
            ScenarioResult with metrics, output, and validation
        """
        start_time = time.time()
        success = True
        error = None
        final_output = None
        metrics_collector = None

        try:
            async with EdgeAgentMCPClient(
                self.config_path,
                metrics_config=self.metrics_config,
                collect_metrics=True,
            ) as client:
                # Load tools
                tools = await client.get_tools()

                # Execute scenario
                final_output = await self.execute(client, tools)

                # Get metrics collector
                metrics_collector = client.get_metrics()

        except Exception as e:
            success = False
            error = str(e)
            import traceback
            error += f"\n{traceback.format_exc()}"

        end_time = time.time()

        # Get validation context
        validation_context = self.get_validation_context()

        # Create result
        result = ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            user_request=self.user_request,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error=error,
            metrics=metrics_collector,
            final_output=final_output,
            validation_context=validation_context,
        )

        # Validate output if requested and execution succeeded
        if validate and success:
            try:
                validation_result = validate_scenario(
                    self.name,
                    final_output,
                    validation_context,
                )
                result.validation = validation_result

                # Update success based on validation
                if not validation_result.passed:
                    result.success = False
                    if result.error:
                        result.error += f"\nValidation failed: score={validation_result.score:.1%}"
                    else:
                        result.error = f"Validation failed: score={validation_result.score:.1%}"
            except Exception as e:
                result.validation = ValidationResult(passed=False, score=0.0)
                result.validation.add_check("validation_error", False, str(e))
                result.validation.compute_score()

        # Save results
        if save_results:
            output_path = result.save(self.output_dir)
            print(f"Results saved to: {output_path}")

        # Print summary
        if print_summary:
            result.print_summary()

        return result


class SimpleScenarioRunner(ScenarioRunner):
    """
    Simple scenario runner for quick ad-hoc scenarios.

    Allows defining scenarios without creating a subclass.
    """

    def __init__(
        self,
        config_path: str | Path,
        scenario_name: str,
        scenario_description: str,
        scenario_user_request: str,
        execute_fn,
        output_dir: str | Path = "results",
        metrics_config: Optional[MetricsConfig] = None,
    ):
        super().__init__(config_path, output_dir, metrics_config)
        self._name = scenario_name
        self._description = scenario_description
        self._user_request = scenario_user_request
        self._execute_fn = execute_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def user_request(self) -> str:
        return self._user_request

    async def execute(self, client: EdgeAgentMCPClient, tools: list) -> Any:
        return await self._execute_fn(client, tools, self)
