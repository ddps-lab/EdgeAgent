"""
EdgeAgent Path Configuration

Location별 경로 설정을 제공합니다.
스케줄러 타입에 따라 적절한 경로를 반환합니다.

경로 규칙:
- DEVICE: /tmp/edgeagent_device_hy/...
- EDGE/CLOUD: /edgeagent/data/scenario{N}/...

사용 예시:
    from edgeagent.paths import get_paths

    paths = get_paths("all_edge")
    repo_path = paths.repo           # /edgeagent/data/scenario1/repo
    log_path = paths.log             # /edgeagent/data/scenario2/server.log
    images_dir = paths.images        # /edgeagent/data/scenario4/coco/images
"""

from dataclasses import dataclass


@dataclass
class ScenarioPaths:
    """시나리오별 경로 설정"""

    # Base directory
    base: str

    # Scenario 1: Code Review
    repo: str
    code_review_report: str

    # Scenario 2: Log Analysis
    log: str
    log_report: str

    # Scenario 3: Research
    papers: str
    research_report: str

    # Scenario 4: Image Processing
    images: str
    image_report: str
    thumbnails: str


# DEVICE 경로 (/tmp/edgeagent_device_hy/...)
DEVICE_PATHS = ScenarioPaths(
    base="/tmp/edgeagent_device_hy",
    # Scenario 1
    repo="/tmp/edgeagent_device_hy/repo",
    code_review_report="/tmp/edgeagent_device_hy/code_review_report.md",
    # Scenario 2
    log="/tmp/edgeagent_device_hy/server.log",
    log_report="/tmp/edgeagent_device_hy/log_report.md",
    # Scenario 3
    papers="/tmp/edgeagent_device_hy/papers",
    research_report="/tmp/edgeagent_device_hy/research_report.md",
    # Scenario 4
    images="/tmp/edgeagent_device_hy/images",
    image_report="/tmp/edgeagent_device_hy/image_report.md",
    thumbnails="/tmp/edgeagent_device_hy/thumbnails",
)

# EDGE/CLOUD 경로 (/edgeagent/data/scenario{N}/...)
EDGE_CLOUD_PATHS = ScenarioPaths(
    base="/edgeagent/data",
    # Scenario 1
    repo="/edgeagent/data/scenario1/repo",
    code_review_report="/edgeagent/data/scenario1/code_review_report.md",
    # Scenario 2
    log="/edgeagent/data/scenario2/server.log",
    log_report="/edgeagent/data/scenario2/log_report.md",
    # Scenario 3
    papers="/edgeagent/data/scenario3/papers",
    research_report="/edgeagent/data/scenario3/research_report.md",
    # Scenario 4
    images="/edgeagent/data/scenario4/coco/images",
    image_report="/edgeagent/data/scenario4/image_report.md",
    thumbnails="/edgeagent/data/scenario4/thumbnails",
)

# 스케줄러 타입 → 경로 매핑
SCHEDULER_TO_PATHS = {
    # All-X 스케줄러
    "all_device": DEVICE_PATHS,
    "all_edge": EDGE_CLOUD_PATHS,
    "all_cloud": EDGE_CLOUD_PATHS,
    # Mixed scheduling → DEVICE 경로 사용 (데이터가 DEVICE에 있음)
    "static": DEVICE_PATHS,
    "brute_force": DEVICE_PATHS,
    "heuristic": DEVICE_PATHS,
}


def get_paths(scheduler_type: str = "brute_force") -> ScenarioPaths:
    """
    스케줄러 타입에 따른 경로 설정 반환

    Args:
        scheduler_type: 스케줄러 타입
            - "all_device": DEVICE 경로 사용
            - "all_edge", "all_cloud": EDGE/CLOUD 경로 사용
            - "static", "brute_force", "heuristic": DEVICE 경로 사용

    Returns:
        ScenarioPaths: 해당 location의 경로 설정
    """
    return SCHEDULER_TO_PATHS.get(scheduler_type, DEVICE_PATHS)


def get_paths_for_location(location: str) -> ScenarioPaths:
    """
    Location에 따른 경로 설정 반환

    Args:
        location: "DEVICE", "EDGE", "CLOUD"

    Returns:
        ScenarioPaths: 해당 location의 경로 설정
    """
    location_upper = location.upper()
    if location_upper in ("EDGE", "CLOUD"):
        return EDGE_CLOUD_PATHS
    else:
        return DEVICE_PATHS
