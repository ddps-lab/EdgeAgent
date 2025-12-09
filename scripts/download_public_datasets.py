#!/usr/bin/env python3
"""
Download Public Datasets for EdgeAgent Scenarios

Downloads academic/benchmark datasets for reproducible experiments:
- S1 (Code Review): Defects4J - Real Java bugs for code review
- S2 (Log Analysis): Loghub - Real-world system logs (already configured)
- S3 (Research): S2ORC - Semantic Scholar Open Research Corpus
- S4 (Image Processing): COCO 2017 - Common Objects in Context

Usage:
    python scripts/download_public_datasets.py [--scenario N] [--size small|medium|large]
"""

import argparse
import subprocess
import shutil
import urllib.request
import zipfile
import tarfile
import json
import os
import random
from pathlib import Path


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress indication."""
    print(f"  Downloading: {description or url}")
    try:
        # Use wget or curl for better progress and resume support
        if shutil.which("wget"):
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(output_path), url],
                check=True,
            )
        elif shutil.which("curl"):
            subprocess.run(
                ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
                check=True,
            )
        else:
            urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def download_s1_defects4j(data_dir: Path, size: str = "small") -> Path:
    """
    Download Defects4J dataset for Scenario 1: Code Review.

    Defects4J is a database of real bugs from real-world Java projects.
    https://github.com/rjust/defects4j

    Size options:
    - small: Lang project, 1 bug (50 files, ~500KB)
    - medium: Math + Time projects, 10 bugs (~5MB)
    - large: 5 projects, 50 bugs (~50MB)
    """
    output_dir = data_dir / "scenario1" / "defects4j"

    print("=" * 60)
    print("Scenario 1: Defects4J Dataset")
    print("=" * 60)
    print(f"Size: {size}")
    print(f"Output directory: {output_dir}")

    # Clean up if exists
    if output_dir.exists():
        print(f"  Removing existing directory...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define projects based on size
    if size == "small":
        projects = [("Lang", 1)]  # 1 bug from Lang
    elif size == "medium":
        projects = [("Lang", 3), ("Math", 3), ("Time", 4)]  # 10 bugs total
    else:  # large
        projects = [("Lang", 10), ("Math", 15), ("Time", 10), ("Closure", 10), ("Mockito", 5)]

    # Check if defects4j is installed
    defects4j_path = shutil.which("defects4j")

    if defects4j_path:
        print(f"\nFound defects4j at: {defects4j_path}")
        print("Checking out bug versions...")

        for project, num_bugs in projects:
            project_dir = output_dir / project.lower()
            project_dir.mkdir(exist_ok=True)

            for bug_id in range(1, num_bugs + 1):
                bug_dir = project_dir / f"bug_{bug_id}"
                print(f"  Checking out {project}-{bug_id}b...")

                try:
                    # Checkout buggy version
                    subprocess.run(
                        ["defects4j", "checkout", "-p", project, "-v", f"{bug_id}b", "-w", str(bug_dir)],
                        capture_output=True,
                        check=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    print(f"    [TIMEOUT] {project}-{bug_id}")
                except subprocess.CalledProcessError as e:
                    print(f"    [ERROR] {project}-{bug_id}: {e.stderr.decode()[:100]}")

    else:
        print("\ndefects4j not installed. Cloning sample projects from GitHub...")

        # Alternative: Clone specific commits from actual Defects4J project repos
        sample_repos = {
            "Lang": {
                "url": "https://github.com/apache/commons-lang.git",
                "commit": "d5b7d7b",  # A known bug-fix commit
                "depth": 10,
            },
            "Math": {
                "url": "https://github.com/apache/commons-math.git",
                "commit": "HEAD",
                "depth": 10,
            },
        }

        for project, info in sample_repos.items():
            if any(p == project for p, _ in projects):
                project_dir = output_dir / project.lower()
                print(f"  Cloning {project}...")

                try:
                    subprocess.run(
                        ["git", "clone", "--depth", str(info["depth"]), info["url"], str(project_dir)],
                        capture_output=True,
                        timeout=180,
                    )
                except Exception as e:
                    print(f"    [ERROR] Clone failed: {e}")

    # Create sample_repo symlink to first project for backward compatibility
    sample_repo = data_dir / "scenario1" / "sample_repo"
    if sample_repo.exists():
        if sample_repo.is_symlink():
            sample_repo.unlink()
        else:
            shutil.rmtree(sample_repo)

    first_project_dir = output_dir / projects[0][0].lower()
    if first_project_dir.exists():
        # Copy instead of symlink for portability
        shutil.copytree(first_project_dir, sample_repo)
        print(f"\nCreated sample_repo from {projects[0][0]}")

    print(f"\nDefects4J setup complete at: {output_dir}")
    return output_dir


def download_s3_s2orc(data_dir: Path, size: str = "small") -> Path:
    """
    Download S2ORC (Semantic Scholar Open Research Corpus) for Scenario 3.

    S2ORC contains 81M+ paper metadata and 8.1M full-text papers.
    https://github.com/allenai/s2orc

    Size options:
    - small: 100 CS papers (2020-2024)
    - medium: 1,000 papers
    - large: 10,000 papers
    """
    output_dir = data_dir / "scenario3" / "s2orc"

    print("=" * 60)
    print("Scenario 3: S2ORC Dataset")
    print("=" * 60)
    print(f"Size: {size}")
    print(f"Output directory: {output_dir}")

    # Clean up if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # S2ORC requires API access or downloading from official source
    # For simplicity, we'll use Semantic Scholar API to get sample papers

    num_papers = {"small": 100, "medium": 1000, "large": 10000}[size]

    print(f"\nFetching {num_papers} papers from Semantic Scholar API...")
    print("(Using public API - no key required for small samples)")

    # Semantic Scholar public API
    api_base = "https://api.semanticscholar.org/graph/v1"

    # Search for AI/ML papers
    search_queries = [
        "large language model agent",
        "AI agent tool use",
        "machine learning edge computing",
        "federated learning",
        "neural network optimization",
    ]

    papers = []
    papers_per_query = num_papers // len(search_queries) + 1

    for query in search_queries:
        if len(papers) >= num_papers:
            break

        print(f"  Searching: {query}")
        try:
            url = f"{api_base}/paper/search?query={urllib.parse.quote(query)}&limit={papers_per_query}&fields=paperId,title,abstract,year,citationCount,authors"

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
                                "citations": paper.get("citationCount", 0),
                                "authors": [a.get("name", "") for a in paper.get("authors", [])[:5]],
                            })
                    print(f"    Found {len(result['data'])} papers")
        except Exception as e:
            print(f"    [ERROR] API call failed: {e}")

    # Limit to requested number
    papers = papers[:num_papers]

    # Save papers to JSON
    output_file = output_dir / "papers.json"
    with open(output_file, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"\nSaved {len(papers)} papers to: {output_file}")

    # Create individual paper files for processing
    papers_dir = output_dir / "papers"
    papers_dir.mkdir(exist_ok=True)

    for i, paper in enumerate(papers):
        paper_file = papers_dir / f"paper_{i:04d}.json"
        with open(paper_file, "w") as f:
            json.dump(paper, f, indent=2)

    # Create index file with URLs for fetch tool
    urls_file = output_dir / "paper_urls.txt"
    with open(urls_file, "w") as f:
        for paper in papers:
            f.write(f"https://www.semanticscholar.org/paper/{paper['id']}\n")

    print(f"Created {len(papers)} paper files in: {papers_dir}")
    print(f"S2ORC setup complete at: {output_dir}")

    return output_dir


def download_s4_coco(data_dir: Path, size: str = "small") -> Path:
    """
    Download COCO 2017 dataset for Scenario 4: Image Processing.

    COCO (Common Objects in Context) is a standard benchmark for
    object detection, segmentation, and captioning.
    https://cocodataset.org/

    Size options:
    - small: 500 images + 50 duplicates (~100MB)
    - medium: 5,000 images (full val2017) + 500 duplicates (~1.1GB)
    - large: Uses Open Images V7 subset (~11GB)
    """
    output_dir = data_dir / "scenario4" / "coco"

    print("=" * 60)
    print("Scenario 4: COCO 2017 Dataset")
    print("=" * 60)
    print(f"Size: {size}")
    print(f"Output directory: {output_dir}")

    # Clean up if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # COCO 2017 validation images URL
    coco_val_url = "http://images.cocodataset.org/zips/val2017.zip"
    coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if size == "small":
        # Download a subset using COCO API or direct image URLs
        print("\nDownloading small COCO sample (500 images)...")

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Use COCO val2017 image IDs
        # These are actual COCO val2017 image IDs
        sample_ids = [
            139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
            1268, 1296, 1353, 1425, 1503, 1532, 1584, 1761, 1818, 1993,
            2006, 2149, 2153, 2157, 2261, 2299, 2431, 2473, 2532, 2587,
        ]

        # Download sample images
        downloaded = 0
        for img_id in sample_ids[:500]:  # Limit for small size
            img_url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
            img_path = images_dir / f"{img_id:012d}.jpg"

            try:
                urllib.request.urlretrieve(img_url, img_path)
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  Downloaded {downloaded} images...")
            except Exception as e:
                pass  # Skip unavailable images

        print(f"  Downloaded {downloaded} COCO images")

        # Create duplicates for testing duplicate detection
        print("\nCreating duplicate variants...")
        create_duplicates(images_dir, num_duplicates=50)

    elif size in ("medium", "large"):
        # Download full val2017
        print("\nDownloading COCO val2017...")
        zip_path = output_dir / "val2017.zip"

        if download_file(coco_val_url, zip_path, "COCO val2017 images"):
            print("  Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()

            # Rename to images
            if (output_dir / "val2017").exists():
                shutil.move(str(output_dir / "val2017"), str(output_dir / "images"))

            # Create duplicates
            num_dups = 500 if size == "medium" else 5000
            print(f"\nCreating {num_dups} duplicate variants...")
            create_duplicates(output_dir / "images", num_duplicates=num_dups)

    # Create sample_images symlink for backward compatibility
    sample_images = data_dir / "scenario4" / "sample_images"
    if sample_images.exists():
        if sample_images.is_symlink():
            sample_images.unlink()
        else:
            shutil.rmtree(sample_images)

    coco_images = output_dir / "images"
    if coco_images.exists():
        # Copy a subset for sample_images
        sample_images.mkdir(parents=True, exist_ok=True)
        images = list(coco_images.glob("*.jpg"))[:30]  # 30 images for quick tests
        for img in images:
            shutil.copy(img, sample_images / img.name)
        print(f"\nCreated sample_images with {len(images)} images")

    print(f"\nCOCO setup complete at: {output_dir}")
    return output_dir


def create_duplicates(images_dir: Path, num_duplicates: int = 50):
    """Create duplicate image variants for testing duplicate detection."""
    try:
        from PIL import Image
    except ImportError:
        print("  [SKIP] PIL not installed, skipping duplicate creation")
        return

    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        return

    random.seed(42)  # For reproducibility
    source_images = random.sample(images, min(num_duplicates, len(images)))

    created = 0
    for img_path in source_images:
        try:
            img = Image.open(img_path)

            # Create variations
            variations = []

            # 1. Slight resize (similar but different file)
            new_size = (int(img.width * 0.95), int(img.height * 0.95))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            variations.append(("resize", resized))

            # 2. Copy with different quality
            variations.append(("copy", img))

            # Save one variation
            var_type, var_img = random.choice(variations)
            dup_name = f"dup_{var_type}_{img_path.stem}.jpg"
            dup_path = images_dir / dup_name
            var_img.save(dup_path, "JPEG", quality=85)
            created += 1

        except Exception as e:
            pass

    print(f"  Created {created} duplicate variants")


def verify_s2_loghub():
    """Verify Scenario 2 Loghub data (already configured)."""
    print("=" * 60)
    print("Scenario 2: Loghub Dataset (Already Configured)")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "scenario2"
    log_file = data_dir / "server.log"

    if log_file.exists():
        size = log_file.stat().st_size
        lines = len(log_file.read_text().splitlines())
        print(f"  Log file: {log_file}")
        print(f"  Size: {size:,} bytes")
        print(f"  Lines: {lines}")
        print("  [OK] Loghub sample data is ready")
    else:
        print("  [WARNING] Log file not found")
        print("  Run: python scripts/generate_log_data.py")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download public datasets for EdgeAgent scenarios"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        help="Download specific scenario dataset only",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size (default: small)",
    )
    args = parser.parse_args()

    # Project root
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print()
    print("EdgeAgent Public Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Size: {args.size}")
    print()

    if args.scenario is None or args.scenario == 1:
        download_s1_defects4j(data_dir, args.size)
        print()

    if args.scenario is None or args.scenario == 2:
        verify_s2_loghub()

    if args.scenario is None or args.scenario == 3:
        download_s3_s2orc(data_dir, args.size)
        print()

    if args.scenario is None or args.scenario == 4:
        download_s4_coco(data_dir, args.size)
        print()

    print("=" * 60)
    print("Dataset download complete!")
    print()
    print("Datasets summary:")
    print("  S1: Defects4J (Java bug database)")
    print("  S2: Loghub (system logs) - already configured")
    print("  S3: S2ORC (academic papers)")
    print("  S4: COCO 2017 (image dataset)")
    print("=" * 60)


if __name__ == "__main__":
    main()
