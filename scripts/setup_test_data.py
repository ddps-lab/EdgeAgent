#!/usr/bin/env python3
"""
Setup Test Data for EdgeAgent Scenarios

Unified script that combines data generation and download:
- S1 (Code Review): Generate test Git repository OR download Defects4J
- S2 (Log Analysis): Generate test log files OR download from Loghub
- S3 (Research): Download from Semantic Scholar API
- S4 (Image Processing): Generate test images OR download from COCO

Usage:
    # Setup all scenarios with generated data (fast, no download)
    python scripts/setup_test_data.py

    # Setup specific scenario
    python scripts/setup_test_data.py --scenario 1

    # Download public datasets (slower, requires network)
    python scripts/setup_test_data.py --download

    # Specify size for downloaded datasets
    python scripts/setup_test_data.py --download --size medium
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

# Try to import PIL for image generation
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Scenario 1: Git Repository
# =============================================================================

def run_git(repo_path: Path, *args):
    """Run a git command in the specified repository."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result


def generate_test_repo(output_dir: Path) -> Path:
    """Generate a test Git repository with sample commits."""
    print("  Generating test Git repository...")

    # Clean up if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize repository
    run_git(output_dir, "init")
    run_git(output_dir, "config", "user.email", "test@example.com")
    run_git(output_dir, "config", "user.name", "Test User")

    # Commit 1: Initial project structure
    (output_dir / "README.md").write_text("""# Sample Project

A sample Python project for testing EdgeAgent code review.

## Features
- User authentication
- Data processing
- API endpoints
""")

    (output_dir / "main.py").write_text('''#!/usr/bin/env python3
"""Main entry point for the application."""

from app.auth import authenticate_user
from app.data import process_data


def main():
    """Run the main application."""
    print("Starting application...")
    user = authenticate_user("admin", "password123")
    if user:
        print(f"Welcome, {user.name}!")
        result = process_data([1, 2, 3, 4, 5])
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
''')

    (output_dir / "app").mkdir(exist_ok=True)
    (output_dir / "app" / "__init__.py").write_text("")

    (output_dir / "app" / "auth.py").write_text('''"""Authentication module."""

from dataclasses import dataclass


@dataclass
class User:
    """User model."""
    id: int
    name: str
    email: str


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user with username and password."""
    # TODO: Replace with actual authentication
    if username == "admin" and password == "password123":
        return User(id=1, name="Admin", email="admin@example.com")
    return None
''')

    (output_dir / "app" / "data.py").write_text('''"""Data processing module."""

from typing import List


def process_data(items: List[int]) -> int:
    """Process a list of items and return the sum."""
    total = 0
    for item in items:
        total = total + item
    return total


def filter_data(items: List[int], threshold: int) -> List[int]:
    """Filter items above a threshold."""
    result = []
    for item in items:
        if item > threshold:
            result.append(item)
    return result
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "Initial commit: basic project structure")

    # Commit 2: Add API endpoints
    (output_dir / "app" / "api.py").write_text('''"""API endpoints module."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class APIResponse:
    """API response wrapper."""
    status: int
    data: Any
    message: str = ""


def get_user(user_id: int) -> APIResponse:
    """Get user by ID."""
    if user_id == 1:
        return APIResponse(
            status=200,
            data={"id": 1, "name": "Admin", "email": "admin@example.com"},
        )
    return APIResponse(status=404, data=None, message="User not found")
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "feat: add API endpoints for user management")

    # Commit 3: Bug fix
    (output_dir / "app" / "data.py").write_text('''"""Data processing module."""

from typing import List


def process_data(items: List[int]) -> int:
    """Process a list of items and return the sum."""
    if not items:
        return 0
    return sum(items)  # Fixed: use built-in sum


def filter_data(items: List[int], threshold: int) -> List[int]:
    """Filter items above a threshold."""
    return [item for item in items if item > threshold]  # Fixed: use list comprehension
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "fix: improve data processing efficiency")

    # Commit 4: Security fix
    (output_dir / "app" / "auth.py").write_text('''"""Authentication module."""

import hashlib
import os
from dataclasses import dataclass


@dataclass
class User:
    """User model."""
    id: int
    name: str
    email: str


def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
    """Hash a password with salt."""
    if salt is None:
        salt = os.urandom(32)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hashed.hex(), salt


def verify_password(password: str, hashed: str, salt: bytes) -> bool:
    """Verify a password against its hash."""
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user with username and password."""
    demo_salt = b'demo_salt_for_testing_only_1234'
    demo_hash, _ = hash_password("secure_password_123", demo_salt)

    if username == "admin" and verify_password(password, demo_hash, demo_salt):
        return User(id=1, name="Admin", email="admin@example.com")
    return None
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "security: implement proper password hashing")

    # Commit 5: Add tests
    (output_dir / "tests").mkdir(exist_ok=True)
    (output_dir / "tests" / "__init__.py").write_text("")

    (output_dir / "tests" / "test_data.py").write_text('''"""Tests for data processing module."""

import pytest
from app.data import process_data, filter_data


class TestProcessData:
    def test_empty_list(self):
        assert process_data([]) == 0

    def test_single_item(self):
        assert process_data([5]) == 5

    def test_multiple_items(self):
        assert process_data([1, 2, 3, 4, 5]) == 15


class TestFilterData:
    def test_empty_list(self):
        assert filter_data([], 5) == []

    def test_mixed(self):
        assert filter_data([1, 5, 10, 3, 15], 5) == [10, 15]
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "test: add unit tests for data module")

    # Show git log
    result = run_git(output_dir, "log", "--oneline")
    commits = len(result.stdout.strip().split('\n'))
    print(f"    Created {commits} commits")

    return output_dir


def download_defects4j(output_dir: Path, size: str = "small") -> Path:
    """Download Defects4J dataset for Scenario 1."""
    print("  Downloading Defects4J...")

    defects4j_dir = output_dir / "defects4j"
    if defects4j_dir.exists():
        shutil.rmtree(defects4j_dir)
    defects4j_dir.mkdir(parents=True, exist_ok=True)

    # Define projects based on size
    if size == "small":
        projects = [("Lang", 1)]
    elif size == "medium":
        projects = [("Lang", 3), ("Math", 3)]
    else:
        projects = [("Lang", 10), ("Math", 15), ("Time", 10)]

    # Check if defects4j is installed
    defects4j_path = shutil.which("defects4j")

    if defects4j_path:
        print(f"    Found defects4j at: {defects4j_path}")
        for project, num_bugs in projects:
            for bug_id in range(1, num_bugs + 1):
                bug_dir = defects4j_dir / project.lower() / f"bug_{bug_id}"
                try:
                    subprocess.run(
                        ["defects4j", "checkout", "-p", project, "-v", f"{bug_id}b", "-w", str(bug_dir)],
                        capture_output=True,
                        check=True,
                        timeout=120,
                    )
                    print(f"    Checked out {project}-{bug_id}")
                except Exception as e:
                    print(f"    [ERROR] {project}-{bug_id}: {e}")
    else:
        print("    defects4j not installed, cloning sample project from GitHub...")
        sample_repo = defects4j_dir / "lang"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "100",
                 "https://github.com/apache/commons-lang.git", str(sample_repo)],
                capture_output=True,
                timeout=180,
            )
            print("    Cloned apache/commons-lang (depth=100)")
        except Exception as e:
            print(f"    [ERROR] Clone failed: {e}")

    # Create sample_repo symlink
    sample_repo = output_dir / "sample_repo"
    if sample_repo.exists():
        shutil.rmtree(sample_repo)

    # Use generated repo as fallback
    if not any(defects4j_dir.iterdir()):
        generate_test_repo(sample_repo)
    else:
        first_project = next(defects4j_dir.iterdir())
        shutil.copytree(first_project, sample_repo)

    return output_dir


def setup_scenario1(data_dir: Path, download: bool = False, size: str = "small"):
    """Setup Scenario 1: Code Review."""
    print("\n" + "=" * 60)
    print("Scenario 1: Code Review (Git Repository)")
    print("=" * 60)

    output_dir = data_dir / "scenario1"
    output_dir.mkdir(parents=True, exist_ok=True)

    if download:
        download_defects4j(output_dir, size)
    else:
        sample_repo = output_dir / "sample_repo"
        generate_test_repo(sample_repo)

    print(f"  Output: {output_dir}")


# =============================================================================
# Scenario 2: Log Files
# =============================================================================

LOGGERS = [
    "app.auth", "app.api", "app.db", "app.cache",
    "worker.task", "worker.queue", "scheduler",
]

LEVELS = {
    "debug": 0.30, "info": 0.40, "warning": 0.15,
    "error": 0.12, "critical": 0.03,
}

MESSAGES = {
    "debug": [
        "Processing request with params: {}",
        "Cache lookup for key: user_{}",
        "Database query executed in {} ms",
    ],
    "info": [
        "Request processed successfully",
        "User {} logged in",
        "Task {} completed",
        "API response time: {} ms",
    ],
    "warning": [
        "Slow query detected: {} ms",
        "Rate limit approaching for user {}",
        "Memory usage above threshold: {}%",
    ],
    "error": [
        "Database connection failed: timeout",
        "Authentication failed for user {}",
        "API request failed with status {}",
    ],
    "critical": [
        "Database connection pool exhausted",
        "Out of memory error",
        "Service unavailable: {}",
    ],
}


def weighted_choice(weights: dict) -> str:
    """Choose a key based on weights."""
    total = sum(weights.values())
    r = random.uniform(0, total)
    cumulative = 0
    for key, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return key
    return list(weights.keys())[-1]


def generate_python_log_line(timestamp: datetime) -> str:
    """Generate a Python logging format line."""
    level = weighted_choice(LEVELS)
    logger = random.choice(LOGGERS)
    message_template = random.choice(MESSAGES[level])

    placeholders = message_template.count("{}")
    values = [str(random.randint(1, 1000)) for _ in range(placeholders)]
    message = message_template.format(*values) if values else message_template

    return f"{timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {logger} - {level.upper()} - {message}"


def generate_log_file(output_path: Path, num_lines: int) -> dict:
    """Generate a log file with specified number of lines."""
    start_time = datetime.now() - timedelta(hours=1)
    lines = []
    current_time = start_time

    for _ in range(num_lines):
        line = generate_python_log_line(current_time)
        lines.append(line)
        current_time += timedelta(seconds=random.uniform(0.1, 5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    output_path.write_text(content)

    return {"path": str(output_path), "lines": num_lines, "size_bytes": len(content)}


def setup_scenario2(data_dir: Path, download: bool = False, size: str = "small"):
    """Setup Scenario 2: Log Analysis."""
    print("\n" + "=" * 60)
    print("Scenario 2: Log Analysis")
    print("=" * 60)

    output_dir = data_dir / "scenario2"
    loghub_dir = output_dir / "loghub_samples"
    loghub_dir.mkdir(parents=True, exist_ok=True)

    # Define log sizes
    configs = [
        ("small_python.log", 100),
        ("medium_python.log", 1000),
        ("large_python.log", 10000),
    ]

    print("  Generating log files...")
    for name, lines in configs:
        result = generate_log_file(loghub_dir / name, lines)
        print(f"    {name}: {lines:,} lines, {result['size_bytes']:,} bytes")

    # Create default server.log
    default_log = output_dir / "server.log"
    medium_log = loghub_dir / "medium_python.log"
    if medium_log.exists():
        default_log.write_text(medium_log.read_text())
        print(f"  Default log: {default_log}")

    print(f"  Output: {output_dir}")


# =============================================================================
# Scenario 3: Research Papers (S2ORC)
# =============================================================================

def setup_scenario3(data_dir: Path, download: bool = False, size: str = "small"):
    """Setup Scenario 3: Research Assistant."""
    print("\n" + "=" * 60)
    print("Scenario 3: Research Assistant (S2ORC)")
    print("=" * 60)

    output_dir = data_dir / "scenario3" / "s2orc"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_papers = {"small": 50, "medium": 200, "large": 1000}[size]

    print(f"  Fetching {num_papers} papers from Semantic Scholar API...")

    api_base = "https://api.semanticscholar.org/graph/v1"
    search_queries = [
        "large language model agent",
        "AI agent tool use",
        "edge computing machine learning",
    ]

    papers = []
    papers_per_query = num_papers // len(search_queries) + 1

    for query in search_queries:
        if len(papers) >= num_papers:
            break

        try:
            url = f"{api_base}/paper/search?query={urllib.parse.quote(query)}&limit={papers_per_query}&fields=paperId,title,abstract,year"
            req = urllib.request.Request(url, headers={"User-Agent": "EdgeAgent/1.0"})

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                if "data" in result:
                    for paper in result["data"]:
                        if paper.get("abstract") and len(paper["abstract"]) > 100:
                            papers.append({
                                "id": paper["paperId"],
                                "title": paper["title"],
                                "abstract": paper["abstract"],
                                "year": paper.get("year"),
                            })
                    print(f"    Query '{query[:30]}...': {len(result['data'])} papers")
        except Exception as e:
            print(f"    [ERROR] Query failed: {e}")

    papers = papers[:num_papers]

    # Save papers
    output_file = output_dir / "papers.json"
    with open(output_file, "w") as f:
        json.dump(papers, f, indent=2)

    # Create paper URLs file
    urls_file = output_dir / "paper_urls.txt"
    with open(urls_file, "w") as f:
        for paper in papers:
            f.write(f"https://www.semanticscholar.org/paper/{paper['id']}\n")

    print(f"  Saved {len(papers)} papers")
    print(f"  Output: {output_dir}")


# =============================================================================
# Scenario 4: Images
# =============================================================================

def generate_gradient_image(width: int, height: int, colors: tuple) -> "Image.Image":
    """Generate an image with gradient colors."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()

    c1, c2 = colors
    for y in range(height):
        ratio = y / height
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        for x in range(width):
            pixels[x, y] = (r, g, b)

    return img


def generate_pattern_image(width: int, height: int, pattern: str = "checkerboard") -> "Image.Image":
    """Generate an image with a pattern."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    if pattern == "checkerboard":
        cell_size = min(width, height) // 8
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                if (x // cell_size + y // cell_size) % 2 == 0:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill="black")
    elif pattern == "circles":
        for _ in range(10):
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            r = random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif pattern == "stripes":
        stripe_width = width // 10
        for i in range(0, width, stripe_width * 2):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([i, 0, i + stripe_width, height], fill=color)

    return img


def generate_test_images(output_dir: Path, count: int = 20) -> list:
    """Generate test images."""
    if not HAS_PIL:
        print("    [SKIP] Pillow not installed")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Small images
    for i in range(count // 4):
        colors = (
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        img = generate_gradient_image(100, 100, colors)
        path = output_dir / f"small_{i:02d}.jpg"
        img.save(path, "JPEG", quality=85)
        results.append({"name": path.name, "size": (100, 100), "file_size": path.stat().st_size})

    # Medium images
    patterns = ["checkerboard", "circles", "stripes"]
    for i in range(count // 4):
        img = generate_pattern_image(400, 400, random.choice(patterns))
        path = output_dir / f"medium_{i:02d}.png"
        img.save(path, "PNG")
        results.append({"name": path.name, "size": (400, 400), "file_size": path.stat().st_size})

    # Large images
    for i in range(count // 4):
        colors = (
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        img = generate_gradient_image(800, 600, colors)
        path = output_dir / f"large_{i:02d}.jpg"
        img.save(path, "JPEG", quality=90)
        results.append({"name": path.name, "size": (800, 600), "file_size": path.stat().st_size})

    # Duplicate pairs (for duplicate detection testing)
    for i in range(count // 4):
        random.seed(1000 + i)  # Same seed = same image
        img = generate_pattern_image(300, 300, "circles")
        random.seed()

        path_a = output_dir / f"dup_{i:02d}_a.jpg"
        path_b = output_dir / f"dup_{i:02d}_b.jpg"
        img.save(path_a, "JPEG", quality=85)
        img.save(path_b, "JPEG", quality=80)  # Slightly different quality

        results.append({"name": path_a.name, "size": (300, 300), "file_size": path_a.stat().st_size})
        results.append({"name": path_b.name, "size": (300, 300), "file_size": path_b.stat().st_size})

    return results


def download_coco_images(output_dir: Path, count: int = 30) -> int:
    """Download sample COCO images."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # COCO val2017 sample image IDs (100 IDs for flexibility)
    sample_ids = [
        139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
        1268, 1296, 1353, 1425, 1503, 1532, 1584, 1761, 1818, 1993,
        2006, 2149, 2153, 2157, 2261, 2299, 2431, 2473, 2532, 2587,
        2685, 2867, 3156, 3501, 3553, 3934, 4134, 4495, 4765, 5037,
        5193, 5503, 5529, 5586, 5992, 6040, 6471, 6723, 6894, 7278,
        7386, 7816, 7977, 8021, 8211, 8532, 8690, 8844, 9378, 9448,
        9590, 9769, 9891, 10092, 10125, 10211, 10363, 10583, 10764, 10977,
        11122, 11197, 11511, 11760, 11813, 12062, 12280, 12576, 12670, 12748,
        13177, 13291, 13546, 13659, 13729, 13774, 13923, 14007, 14226, 14380,
        14439, 14473, 14888, 15079, 15254, 15335, 15497, 15660, 15746, 15956,
    ]

    downloaded = 0
    for img_id in sample_ids[:count]:
        img_url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
        img_path = images_dir / f"{img_id:012d}.jpg"

        try:
            urllib.request.urlretrieve(img_url, img_path)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"    Downloaded {downloaded} images...")
        except Exception:
            pass

    print(f"    Downloaded {downloaded} COCO images")
    return downloaded


def create_duplicate_variants(images_dir: Path, num_duplicates: int = 10) -> int:
    """Create duplicate variants of existing images for testing duplicate detection.

    Creates slightly modified copies (resize, quality change) that should be
    detected as duplicates by perceptual hashing.
    """
    if not HAS_PIL:
        print("    [SKIP] Pillow not installed, cannot create duplicates")
        return 0

    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        return 0

    random.seed(42)  # Reproducibility
    source_images = random.sample(images, min(num_duplicates, len(images)))

    created = 0
    for img_path in source_images:
        try:
            img = Image.open(img_path)

            # Create a slightly resized version (95% size) - should still be detected as duplicate
            new_size = (int(img.width * 0.95), int(img.height * 0.95))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save with different name
            dup_name = f"dup_{img_path.stem}.jpg"
            dup_path = images_dir / dup_name
            resized.save(dup_path, "JPEG", quality=85)
            created += 1
        except Exception:
            pass

    random.seed()  # Reset seed
    return created


def setup_scenario4(data_dir: Path, download: bool = False, size: str = "small"):
    """Setup Scenario 4: Image Processing.

    Creates images + duplicate variants for testing duplicate detection.
    Size options:
    - small: 30 images + 10 duplicates = 40 total
    - medium: 60 images + 15 duplicates = 75 total
    - large: 100 images + 20 duplicates = 120 total
    """
    print("\n" + "=" * 60)
    print("Scenario 4: Image Processing")
    print("=" * 60)

    output_dir = data_dir / "scenario4"
    sample_images = output_dir / "sample_images"

    # Size configurations: (base_images, num_duplicates)
    size_config = {
        "small": (30, 10),
        "medium": (60, 15),
        "large": (100, 20),
    }
    base_count, dup_count = size_config[size]

    if sample_images.exists():
        shutil.rmtree(sample_images)
    sample_images.mkdir(parents=True, exist_ok=True)

    if download:
        # Download COCO images
        print(f"  Downloading {base_count} COCO images...")

        coco_dir = output_dir / "coco"
        if coco_dir.exists():
            shutil.rmtree(coco_dir)
        coco_dir.mkdir(parents=True, exist_ok=True)

        downloaded = download_coco_images(coco_dir, base_count)

        # Copy to sample_images
        coco_images = list((coco_dir / "images").glob("*.jpg"))
        for img in coco_images:
            shutil.copy(img, sample_images / img.name)

        print(f"    Copied {len(coco_images)} images to sample_images")
    else:
        # Generate test images
        print(f"  Generating {base_count} test images...")
        results = generate_test_images(sample_images, count=base_count)

        if results:
            total_size = sum(r["file_size"] for r in results)
            print(f"    Generated {len(results)} images ({total_size:,} bytes)")

    # Create duplicate variants for duplicate detection testing
    print(f"  Creating {dup_count} duplicate variants for testing...")
    created_dups = create_duplicate_variants(sample_images, num_duplicates=dup_count)
    print(f"    Created {created_dups} duplicate variants")

    # Final summary
    final_count = len(list(sample_images.glob("*")))
    final_size = sum(f.stat().st_size for f in sample_images.glob("*") if f.is_file())
    print(f"  Total: {final_count} images ({final_size:,} bytes)")
    print(f"  Output: {sample_images}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup test data for EdgeAgent scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: Download/fetch all data
    python scripts/setup_test_data.py

    # Setup specific scenario
    python scripts/setup_test_data.py --scenario 2

    # Specify size
    python scripts/setup_test_data.py --size medium

    # Quick setup with generated data only (no network)
    python scripts/setup_test_data.py --generate
        """,
    )
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        help="Setup specific scenario only (1-4)",
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate test data locally instead of downloading (faster, no network)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size (default: small)",
    )
    args = parser.parse_args()

    # Default is download (not generate)
    download = not args.generate

    # Project paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print()
    print("=" * 60)
    print("EdgeAgent Test Data Setup")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Mode: {'Generate test data (offline)' if args.generate else 'Download/fetch data'}")
    print(f"Size: {args.size}")
    print()

    # Setup scenarios
    if args.scenario is None or args.scenario == 1:
        setup_scenario1(data_dir, download, args.size)

    if args.scenario is None or args.scenario == 2:
        setup_scenario2(data_dir, download, args.size)

    if args.scenario is None or args.scenario == 3:
        setup_scenario3(data_dir, download, args.size)

    if args.scenario is None or args.scenario == 4:
        setup_scenario4(data_dir, download, args.size)

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Scenarios ready:")
    print("  S1: Code Review   - data/scenario1/sample_repo/")
    print("  S2: Log Analysis  - data/scenario2/server.log")
    print("  S3: Research      - data/scenario3/s2orc/papers.json")
    print("  S4: Image Process - data/scenario4/sample_images/")
    print()


if __name__ == "__main__":
    main()
