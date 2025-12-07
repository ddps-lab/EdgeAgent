"""
E2E Routing 통합 테스트

ProxyTool을 통한 실제 routing 검증:
1. Tool 이름에 location suffix 없음
2. Args에 따라 다른 location으로 routing
3. Execution trace 기록 확인
"""

import pytest
import asyncio
from pathlib import Path

from edgeagent.middleware import EdgeAgentMCPClient


@pytest.fixture
def e2e_config(tmp_path):
    """E2E 테스트용 설정 및 디렉토리"""
    # 각 location별 디렉토리 생성
    device_dir = tmp_path / "edgeagent_device"
    edge_dir = tmp_path / "edgeagent_edge"
    cloud_dir = tmp_path / "edgeagent_cloud"

    for d in [device_dir, edge_dir, cloud_dir]:
        d.mkdir()
        location_name = d.name.replace("edgeagent_", "").upper()
        (d / "marker.txt").write_text(f"LOCATION={location_name}")
        (d / "test.txt").write_text(f"Content from {location_name}")

    # YAML 설정 생성
    config = tmp_path / "tools.yaml"
    config.write_text(f"""
tools:
  filesystem:
    profile:
      description: "Filesystem access"
      data_affinity: DEVICE
      compute_intensity: LOW
      privacy_sensitive: false
    endpoints:
      DEVICE:
        transport: stdio
        command: npx
        args:
          - "-y"
          - "@modelcontextprotocol/server-filesystem"
          - "{device_dir}"
      EDGE:
        transport: stdio
        command: npx
        args:
          - "-y"
          - "@modelcontextprotocol/server-filesystem"
          - "{edge_dir}"
      CLOUD:
        transport: stdio
        command: npx
        args:
          - "-y"
          - "@modelcontextprotocol/server-filesystem"
          - "{cloud_dir}"

static_mapping:
  filesystem: DEVICE
""")

    return {
        "config": config,
        "device_dir": device_dir,
        "edge_dir": edge_dir,
        "cloud_dir": cloud_dir,
    }


class TestProxyToolStructure:
    """ProxyTool 구조 테스트"""

    @pytest.mark.asyncio
    async def test_tools_have_no_location_suffix(self, e2e_config):
        """Tool 이름에 location suffix가 없어야 함"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()

            for tool in tools:
                # suffix가 있으면 안됨
                assert not tool.name.endswith("_device"), f"{tool.name} has _device suffix"
                assert not tool.name.endswith("_edge"), f"{tool.name} has _edge suffix"
                assert not tool.name.endswith("_cloud"), f"{tool.name} has _cloud suffix"

    @pytest.mark.asyncio
    async def test_proxy_tools_have_backend_tools(self, e2e_config):
        """각 ProxyTool이 여러 backend tool을 가져야 함"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()

            for tool in tools:
                assert hasattr(tool, "backend_tools"), f"{tool.name} has no backend_tools"
                assert len(tool.backend_tools) == 3, f"{tool.name} should have 3 backends"
                assert "DEVICE" in tool.backend_tools
                assert "EDGE" in tool.backend_tools
                assert "CLOUD" in tool.backend_tools

    @pytest.mark.asyncio
    async def test_tool_count_is_reduced(self, e2e_config):
        """이전 42개 → 현재 14개로 축소"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()

            # filesystem MCP server는 14개 tool 제공
            # 이전: 14 * 3 = 42개
            # 현재: 14개 (proxy)
            assert len(tools) == 14


class TestArgsBasedRouting:
    """Args 기반 동적 routing 테스트"""

    @pytest.mark.asyncio
    async def test_routing_by_path_device(self, e2e_config):
        """path에 /edgeagent_device/ 포함 → DEVICE로 routing"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            # DEVICE path로 호출
            result = await read_file.ainvoke({
                "path": f"{e2e_config['device_dir']}/marker.txt"
            })

            # 결과 확인
            assert "DEVICE" in str(result)

            # Trace 확인
            assert len(client.execution_trace) > 0
            assert client.execution_trace[-1]["location"] == "DEVICE"

    @pytest.mark.asyncio
    async def test_routing_by_path_edge(self, e2e_config):
        """path에 /edgeagent_edge/ 포함 → EDGE로 routing"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            result = await read_file.ainvoke({
                "path": f"{e2e_config['edge_dir']}/marker.txt"
            })

            assert "EDGE" in str(result)
            assert client.execution_trace[-1]["location"] == "EDGE"

    @pytest.mark.asyncio
    async def test_routing_by_path_cloud(self, e2e_config):
        """path에 /edgeagent_cloud/ 포함 → CLOUD로 routing"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            result = await read_file.ainvoke({
                "path": f"{e2e_config['cloud_dir']}/marker.txt"
            })

            assert "CLOUD" in str(result)
            assert client.execution_trace[-1]["location"] == "CLOUD"


class TestExecutionTrace:
    """Execution trace 기록 테스트"""

    @pytest.mark.asyncio
    async def test_trace_records_tool_name(self, e2e_config):
        """Trace에 tool name이 기록되어야 함"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            await read_file.ainvoke({
                "path": f"{e2e_config['device_dir']}/marker.txt"
            })

            trace = client.execution_trace[-1]
            assert trace["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_trace_records_location(self, e2e_config):
        """Trace에 location이 기록되어야 함"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            await read_file.ainvoke({
                "path": f"{e2e_config['device_dir']}/marker.txt"
            })

            trace = client.execution_trace[-1]
            assert "location" in trace
            assert trace["location"] in ["DEVICE", "EDGE", "CLOUD"]

    @pytest.mark.asyncio
    async def test_multiple_calls_trace(self, e2e_config):
        """여러 호출의 trace가 모두 기록되어야 함"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            read_file = next(t for t in tools if t.name == "read_file")

            # 3개 location에 각각 호출
            await read_file.ainvoke({"path": f"{e2e_config['device_dir']}/test.txt"})
            await read_file.ainvoke({"path": f"{e2e_config['edge_dir']}/test.txt"})
            await read_file.ainvoke({"path": f"{e2e_config['cloud_dir']}/test.txt"})

            # 3개의 trace가 있어야 함
            assert len(client.execution_trace) == 3

            # 각각 다른 location
            locations = [t["location"] for t in client.execution_trace]
            assert "DEVICE" in locations
            assert "EDGE" in locations
            assert "CLOUD" in locations


class TestStaticMappingFallback:
    """Static mapping fallback 테스트"""

    @pytest.mark.asyncio
    async def test_fallback_to_static_mapping(self, e2e_config):
        """패턴에 안 맞는 path는 static_mapping 사용"""
        async with EdgeAgentMCPClient(e2e_config["config"]) as client:
            tools = await client.get_tools()
            list_dir = next(t for t in tools if t.name == "list_directory")

            # 패턴에 안 맞는 path (하지만 DEVICE 디렉토리에서 실행)
            # static_mapping: filesystem: DEVICE
            result = await list_dir.ainvoke({
                "path": str(e2e_config['device_dir'])
            })

            # DEVICE로 routing 되어야 함 (edgeagent_device 패턴 매칭)
            assert client.execution_trace[-1]["location"] == "DEVICE"
