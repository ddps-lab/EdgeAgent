"""
EdgeAgent: LangChain MCP Scheduling Middleware

Locality-aware serverless execution of MCP tools in the Edge-Cloud continuum.
"""

__version__ = "0.1.0"

# Main exports
from .middleware import EdgeAgentMCPClient
from .registry import ToolRegistry
from .scheduler import (
    BaseScheduler,
    StaticScheduler,
    SchedulingContext,
    SchedulingResult,
    AllDeviceScheduler,
    AllEdgeScheduler,
    AllCloudScheduler,
    HeuristicScheduler,
    SCHEDULER_REGISTRY,
    create_scheduler,
)
from .profiles import ToolProfile, EndpointConfig, ToolConfig
from .proxy_tool import LocationAwareProxyTool
from .types import (
    Location,
    Runtime,
    DataAffinity,
    ComputeIntensity,
    DataFlow,
    TransportType,
)
from .metrics import (
    MetricEntry,
    MetricsConfig,
    MetricsCollector,
    CallContext,
)
from .scenario_runner import (
    ScenarioResult,
    ScenarioRunner,
    SimpleScenarioRunner,
)
from .validation import (
    ValidationResult,
    ScenarioValidator,
    validate_scenario,
    get_validator,
)

__all__ = [
    # Middleware
    "EdgeAgentMCPClient",
    "LocationAwareProxyTool",
    # Registry & Scheduler
    "ToolRegistry",
    "BaseScheduler",
    "StaticScheduler",
    "SchedulingContext",
    "SchedulingResult",
    "AllDeviceScheduler",
    "AllEdgeScheduler",
    "AllCloudScheduler",
    "HeuristicScheduler",
    "SCHEDULER_REGISTRY",
    "create_scheduler",
    # Profiles
    "ToolProfile",
    "EndpointConfig",
    "ToolConfig",
    # Metrics
    "MetricEntry",
    "MetricsConfig",
    "MetricsCollector",
    "CallContext",
    # Scenario Runner
    "ScenarioResult",
    "ScenarioRunner",
    "SimpleScenarioRunner",
    # Validation
    "ValidationResult",
    "ScenarioValidator",
    "validate_scenario",
    "get_validator",
    # Types
    "Location",
    "Runtime",
    "DataAffinity",
    "ComputeIntensity",
    "DataFlow",
    "TransportType",
]
