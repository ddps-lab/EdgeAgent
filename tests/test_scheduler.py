"""
Scheduler 단위 테스트

StaticScheduler의 location 결정 로직 검증:
1. Constraints (requires_cloud_api, privacy_sensitive)
2. Args 기반 routing (path 패턴)
3. Static mapping fallback
4. Profile data_affinity fallback
"""

import pytest
from pathlib import Path

from edgeagent.scheduler import StaticScheduler, SchedulingContext
from edgeagent.registry import ToolRegistry


class TestStaticSchedulerConstraints:
    """Constraint 기반 routing 테스트"""

    @pytest.fixture
    def scheduler(self, test_config_path):
        """테스트용 Scheduler 생성"""
        registry = ToolRegistry.from_yaml(test_config_path)
        return StaticScheduler(test_config_path, registry)

    def test_cloud_api_required_routes_to_cloud(self, scheduler):
        """requires_cloud_api=True → 무조건 CLOUD"""
        # slack tool은 requires_cloud_api=True
        location = scheduler.get_location_for_call("slack", {"message": "hello"})
        assert location == "CLOUD"

    def test_cloud_api_required_ignores_args(self, scheduler):
        """requires_cloud_api=True면 args의 path hint 무시"""
        # path에 /device/가 있어도 CLOUD로 routing
        location = scheduler.get_location_for_call(
            "slack",
            {"path": "/device/something", "message": "hello"}
        )
        assert location == "CLOUD"

    def test_privacy_sensitive_excludes_cloud(self, scheduler):
        """privacy_sensitive=True → CLOUD 제외"""
        # credentials tool은 privacy_sensitive=True
        location = scheduler.get_location_for_call("credentials", {})
        assert location in ["DEVICE", "EDGE"]

    def test_privacy_sensitive_ignores_cloud_hint(self, scheduler):
        """privacy_sensitive=True면 CLOUD hint 무시"""
        # path에 /cloud/가 있어도 CLOUD 선택 안함
        location = scheduler.get_location_for_call(
            "credentials",
            {"path": "/cloud/secrets.txt"}
        )
        assert location in ["DEVICE", "EDGE"]


class TestStaticSchedulerArgsRouting:
    """Args 기반 동적 routing 테스트"""

    @pytest.fixture
    def scheduler(self, test_config_path):
        """테스트용 Scheduler 생성"""
        registry = ToolRegistry.from_yaml(test_config_path)
        return StaticScheduler(test_config_path, registry)

    def test_path_based_routing_device(self, scheduler):
        """path에 /edgeagent_device/ 포함 → DEVICE"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/tmp/edgeagent_device/file.txt"}
        )
        assert location == "DEVICE"

    def test_path_based_routing_edge(self, scheduler):
        """path에 /edgeagent_edge/ 포함 → EDGE"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/tmp/edgeagent_edge/file.txt"}
        )
        assert location == "EDGE"

    def test_path_based_routing_cloud(self, scheduler):
        """path에 /edgeagent_cloud/ 포함 → CLOUD"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/tmp/edgeagent_cloud/file.txt"}
        )
        assert location == "CLOUD"

    def test_path_pattern_device_keyword(self, scheduler):
        """path에 /device/ 키워드 포함 → DEVICE"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/data/device/local_file.txt"}
        )
        assert location == "DEVICE"

    def test_path_pattern_edge_keyword(self, scheduler):
        """path에 /edge/ 키워드 포함 → EDGE"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/data/edge/server_file.txt"}
        )
        assert location == "EDGE"

    def test_path_pattern_cloud_keyword(self, scheduler):
        """path에 /cloud/ 키워드 포함 → CLOUD"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/data/cloud/remote_file.txt"}
        )
        assert location == "CLOUD"

    def test_path_case_insensitive(self, scheduler):
        """path 패턴은 대소문자 구분 안함"""
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/data/DEVICE/file.txt"}
        )
        assert location == "DEVICE"


class TestStaticSchedulerFallback:
    """Fallback 로직 테스트"""

    @pytest.fixture
    def scheduler(self, test_config_path):
        """테스트용 Scheduler 생성"""
        registry = ToolRegistry.from_yaml(test_config_path)
        return StaticScheduler(test_config_path, registry)

    def test_static_mapping_fallback(self, scheduler):
        """args에서 hint 없으면 static_mapping 사용"""
        # /some/path는 어떤 패턴에도 안 맞음 → static_mapping
        location = scheduler.get_location_for_call(
            "filesystem",
            {"path": "/some/random/path.txt"}
        )
        assert location == "DEVICE"  # static_mapping: filesystem: DEVICE

    def test_no_args_uses_static_mapping(self, scheduler):
        """args 없으면 static_mapping 사용"""
        location = scheduler.get_location_for_call("filesystem", None)
        assert location == "DEVICE"

    def test_empty_args_uses_static_mapping(self, scheduler):
        """빈 args면 static_mapping 사용"""
        location = scheduler.get_location_for_call("filesystem", {})
        assert location == "DEVICE"

    def test_data_affinity_fallback(self, scheduler):
        """static_mapping에 없는 tool은 data_affinity 사용"""
        # 임시로 static_mapping에서 제거된 상태 시뮬레이션
        # (실제로는 unknown tool 테스트)
        location = scheduler.get_location("unknown_tool")
        assert location == "EDGE"  # 기본값


class TestStaticSchedulerRuntime:
    """Runtime 선택 테스트"""

    @pytest.fixture
    def scheduler(self, test_config_path):
        """테스트용 Scheduler 생성"""
        registry = ToolRegistry.from_yaml(test_config_path)
        return StaticScheduler(test_config_path, registry)

    def test_device_always_wasi(self, scheduler):
        """DEVICE location은 항상 WASI"""
        runtime = scheduler.select_runtime("filesystem", "DEVICE")
        assert runtime == "WASI"

    def test_edge_default_wasi(self, scheduler):
        """EDGE location 기본값은 WASI"""
        runtime = scheduler.select_runtime("filesystem", "EDGE")
        assert runtime == "WASI"

    def test_cloud_default_wasi(self, scheduler):
        """CLOUD location 기본값은 WASI"""
        runtime = scheduler.select_runtime("slack", "CLOUD")
        assert runtime == "WASI"


class TestStaticSchedulerBatch:
    """Batch scheduling 테스트"""

    @pytest.fixture
    def scheduler(self, test_config_path):
        """테스트용 Scheduler 생성"""
        registry = ToolRegistry.from_yaml(test_config_path)
        return StaticScheduler(test_config_path, registry)

    def test_schedule_multiple_tools(self, scheduler):
        """여러 tool 일괄 스케줄링"""
        placements = scheduler.schedule(["filesystem", "slack", "credentials"])

        assert len(placements) == 3

        # 각 tool의 placement 확인
        placement_dict = {name: (loc, rt) for name, loc, rt in placements}

        assert placement_dict["filesystem"][0] == "DEVICE"
        assert placement_dict["slack"][0] == "CLOUD"
        assert placement_dict["credentials"][0] == "DEVICE"
