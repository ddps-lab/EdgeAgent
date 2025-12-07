"""
Tool Registry

Multi-endpoint tool 설정을 관리하는 registry
"""

import yaml
from pathlib import Path
from typing import Optional

from .types import Location
from .profiles import ToolProfile, EndpointConfig, ToolConfig


class ToolRegistry:
    """
    Tool의 multi-endpoint 설정을 관리하는 registry

    YAML 파일에서 tool 설정을 로드하고, location별 endpoint 정보를 제공
    """

    def __init__(self):
        self.tools: dict[str, ToolConfig] = {}

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

                if transport == "stdio":
                    endpoint = EndpointConfig(
                        location=location,
                        transport=transport,
                        command=endpoint_data.get("command"),
                        args=endpoint_data.get("args", []),
                        env=endpoint_data.get("env", {}),
                    )
                else:  # streamable_http
                    endpoint = EndpointConfig(
                        location=location,
                        transport=transport,
                        url=endpoint_data.get("url"),
                        env=endpoint_data.get("env", {}),
                    )

                endpoints[location] = endpoint

            # ToolConfig 생성 및 등록
            tool_config = ToolConfig(
                name=tool_name,
                profile=profile,
                endpoints=endpoints,
            )

            self.tools[tool_name] = tool_config

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
        """등록된 모든 tool 이름 목록"""
        return list(self.tools.keys())

    def get_tools_by_location(self, location: Location) -> list[str]:
        """특정 location에 배포된 tool 목록"""
        result = []
        for tool_name, tool_config in self.tools.items():
            if tool_config.has_location(location):
                result.append(tool_name)
        return result

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self.tools)})"
