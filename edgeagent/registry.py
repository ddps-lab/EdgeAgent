"""
Tool Registry

Multi-endpoint tool 설정을 관리하는 registry
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional

from .types import Location
from .profiles import ToolProfile, EndpointConfig, ToolConfig, ToolMeasurements


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in string.

    Supports:
    - ${VAR} - required variable
    - ${VAR:-default} - variable with default value
    """
    def replacer(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        else:
            return os.environ.get(var_expr, "")

    return re.sub(r'\$\{([^}]+)\}', replacer, value)


def _expand_env_dict(env_dict: dict) -> dict:
    """Expand environment variables in a dictionary."""
    return {k: _expand_env_vars(v) if isinstance(v, str) else v
            for k, v in env_dict.items()}


class ToolRegistry:
    """
    Tool의 multi-endpoint 설정을 관리하는 registry

    YAML 파일에서 tool 설정을 로드하고, location별 endpoint 정보를 제공
    """

    def __init__(self):
        self.tools: dict[str, ToolConfig] = {}
        # MCP 서버 이름 목록 (개별 tool과 구분)
        self.mcp_servers: set[str] = set()
        # 개별 tool → 부모 MCP 서버 매핑
        self.tool_to_server: dict[str, str] = {}

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ToolRegistry":
        """
        YAML 파일에서 tool 설정 로드

        Args:
            config_path: YAML 설정 파일 경로

        Returns:
            ToolRegistry 인스턴스
        """
        registry = cls()
        registry.load_from_yaml(config_path)
        return registry

    def load_from_yaml(self, config_path: str | Path):
        """YAML 파일에서 tool 설정 로드"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Tools 섹션 파싱
        tools_config = config.get("tools", {})

        for tool_name, tool_data in tools_config.items():
            # Profile 파싱
            profile_data = tool_data.get("profile", {})
            profile = ToolProfile(
                name=tool_name,
                description=profile_data.get("description", ""),
                data_affinity=profile_data.get("data_affinity", "EDGE"),
                compute_intensity=profile_data.get("compute_intensity", "LOW"),
                data_flow=profile_data.get("data_flow", "TRANSFORM"),
                reduction_ratio=profile_data.get("reduction_ratio", 1.0),
                requires_cloud_api=profile_data.get("requires_cloud_api", False),
                privacy_sensitive=profile_data.get("privacy_sensitive", False),
                wasi_compatible=profile_data.get("wasi_compatible", True),
                requires_gpu=profile_data.get("requires_gpu", False),
            )

            # Endpoints 파싱
            endpoints = {}
            endpoints_data = tool_data.get("endpoints", {})

            for location, endpoint_data in endpoints_data.items():
                transport = endpoint_data.get("transport", "stdio")

                # Expand environment variables in env dict
                env = _expand_env_dict(endpoint_data.get("env", {}))

                if transport == "stdio":
                    endpoint = EndpointConfig(
                        location=location,
                        transport=transport,
                        command=endpoint_data.get("command"),
                        args=endpoint_data.get("args", []),
                        env=env,
                    )
                else:  # streamable_http
                    endpoint = EndpointConfig(
                        location=location,
                        transport=transport,
                        url=endpoint_data.get("url"),
                        env=env,
                    )

                endpoints[location] = endpoint

            # ToolConfig 생성 및 등록
            tool_config = ToolConfig(
                name=tool_name,
                profile=profile,
                endpoints=endpoints,
            )

            self.tools[tool_name] = tool_config
            # MCP 서버로 등록
            self.mcp_servers.add(tool_name)

            # tool_profiles 섹션 파싱 (Score-based Scheduling용)
            tool_profiles_data = tool_data.get("tool_profiles", {})
            for individual_tool_name, individual_profile_data in tool_profiles_data.items():
                individual_profile = ToolProfile(
                    name=individual_tool_name,
                    description=individual_profile_data.get("description", ""),
                )

                # Score-based Scheduling 파라미터 (fallback)
                individual_profile.data_locality = individual_profile_data.get("data_locality", "args_only")
                individual_profile.alpha = float(individual_profile_data.get("alpha", 0.5))
                if "P_exec" in individual_profile_data:
                    individual_profile.P_exec = [float(x) for x in individual_profile_data["P_exec"]]

                # 실측 기반 measurements (동적 계산용)
                if "measurements" in individual_profile_data:
                    m = individual_profile_data["measurements"]
                    individual_profile.measurements = ToolMeasurements(
                        input_datasize_mb=float(m.get("input_datasize_mb", 0.001)),
                        output_datasize_mb=float(m.get("output_datasize_mb", 0.001)),
                        t_exec_ms=m.get("T_exec_ms", {}),
                    )

                # 개별 Tool 등록 (endpoints는 서버와 공유)
                individual_tool_config = ToolConfig(
                    name=individual_tool_name,
                    profile=individual_profile,
                    endpoints=endpoints,
                )
                self.tools[individual_tool_name] = individual_tool_config
                # tool → 부모 서버 매핑
                self.tool_to_server[individual_tool_name] = tool_name

    def register_tool(self, tool_config: ToolConfig):
        """Tool 등록"""
        self.tools[tool_config.name] = tool_config

    def get_tool(self, tool_name: str) -> Optional[ToolConfig]:
        """Tool 설정 가져오기"""
        return self.tools.get(tool_name)

    def get_endpoint(
        self, tool_name: str, location: Location
    ) -> Optional[EndpointConfig]:
        """특정 tool의 특정 location endpoint 가져오기"""
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        return tool.get_endpoint(location)

    def get_profile(self, tool_name: str) -> Optional[ToolProfile]:
        """Tool profile 가져오기"""
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        return tool.profile

    def list_tools(self) -> list[str]:
        """등록된 모든 tool 이름 목록 (서버 + 개별 tool)"""
        return list(self.tools.keys())

    def list_servers(self) -> list[str]:
        """MCP 서버 이름만 반환"""
        return list(self.mcp_servers)

    def list_individual_tools(self) -> list[str]:
        """개별 tool 이름만 반환 (서버 제외)"""
        return list(self.tool_to_server.keys())

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """tool의 부모 MCP 서버 이름 반환"""
        # 서버 자체인 경우 자신 반환
        if tool_name in self.mcp_servers:
            return tool_name
        # 개별 tool인 경우 부모 서버 반환
        return self.tool_to_server.get(tool_name)

    def is_server(self, name: str) -> bool:
        """MCP 서버인지 확인"""
        return name in self.mcp_servers

    def get_tools_by_location(self, location: Location) -> list[str]:
        """특정 location에 배포된 tool 목록"""
        result = []
        for tool_name, tool_config in self.tools.items():
            if tool_config.has_location(location):
                result.append(tool_name)
        return result

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self.tools)})"
