"""
EdgeAgent: LangChain MCP Scheduling Middleware

Locality-aware serverless execution of MCP tools in the Edge-Cloud continuum.
"""

__version__ = "0.1.0"

# Main exports
from .middleware import EdgeAgentMCPClient
from .registry import ToolRegistry
from .scheduler import BaseScheduler, StaticScheduler, SchedulingContext
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

__all__ = [
    # Middleware
    "EdgeAgentMCPClient",
    "LocationAwareProxyTool",
    # Registry & Scheduler
    "ToolRegistry",
    "BaseScheduler",
    "StaticScheduler",
    "SchedulingContext",
    # Profiles
    "ToolProfile",
    "EndpointConfig",
    "ToolConfig",
    # Types
    "Location",
    "Runtime",
    "DataAffinity",
    "ComputeIntensity",
    "DataFlow",
    "TransportType",
]
