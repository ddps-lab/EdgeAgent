"""
Pytest Configuration and Fixtures
"""

import pytest
import asyncio
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config_path(tmp_path):
    """테스트용 YAML 설정 파일 생성"""
    config = tmp_path / "tools.yaml"
    config.write_text("""
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
        command: echo
        args: ["device"]
      EDGE:
        transport: stdio
        command: echo
        args: ["edge"]
      CLOUD:
        transport: stdio
        command: echo
        args: ["cloud"]

  slack:
    profile:
      description: "Slack messaging"
      data_affinity: CLOUD
      requires_cloud_api: true
    endpoints:
      CLOUD:
        transport: stdio
        command: echo
        args: ["slack"]

  credentials:
    profile:
      description: "Credential manager"
      data_affinity: DEVICE
      privacy_sensitive: true
    endpoints:
      DEVICE:
        transport: stdio
        command: echo
        args: ["credentials"]
      EDGE:
        transport: stdio
        command: echo
        args: ["credentials"]

static_mapping:
  filesystem: DEVICE
  slack: CLOUD
  credentials: DEVICE
""")
    return config


@pytest.fixture
def test_directories(tmp_path):
    """테스트용 location별 디렉토리 생성"""
    device_dir = tmp_path / "device"
    edge_dir = tmp_path / "edge"
    cloud_dir = tmp_path / "cloud"

    for d in [device_dir, edge_dir, cloud_dir]:
        d.mkdir()
        (d / "marker.txt").write_text(d.name.upper())

    return {
        "device": device_dir,
        "edge": edge_dir,
        "cloud": cloud_dir,
        "root": tmp_path,
    }
