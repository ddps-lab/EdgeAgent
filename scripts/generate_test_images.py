#!/usr/bin/env python3
"""
Generate Test Image Data for Scenario 4

Creates sample images of various sizes and types for testing:
- Different formats (JPEG, PNG)
- Different sizes (small, medium, large)
- Some duplicates (for duplicate detection testing)
- Different color patterns

This script uses Pillow to generate test images programmatically.
"""

import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("[ERROR] Pillow is required: pip install pillow")
    exit(1)


def generate_gradient_image(width: int, height: int, colors: tuple[tuple, tuple]) -> Image.Image:
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


def generate_pattern_image(width: int, height: int, pattern: str = "checkerboard") -> Image.Image:
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
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    elif pattern == "stripes":
        stripe_width = width // 10
        for i in range(0, width, stripe_width * 2):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([i, 0, i + stripe_width, height], fill=color)

    elif pattern == "noise":
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

    return img


def generate_test_images(output_dir: Path, count: int = 20):
    """Generate a set of test images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []

    # Small images (100x100)
    for i in range(count // 4):
        configs.append({
            "name": f"small_{i:02d}.jpg",
            "size": (100, 100),
            "type": "gradient",
            "colors": (
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ),
            "format": "JPEG",
            "quality": 85,
        })

    # Medium images (400x400)
    for i in range(count // 4):
        patterns = ["checkerboard", "circles", "stripes"]
        configs.append({
            "name": f"medium_{i:02d}.png",
            "size": (400, 400),
            "type": "pattern",
            "pattern": random.choice(patterns),
            "format": "PNG",
        })

    # Large images (800x600)
    for i in range(count // 4):
        configs.append({
            "name": f"large_{i:02d}.jpg",
            "size": (800, 600),
            "type": "gradient",
            "colors": (
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ),
            "format": "JPEG",
            "quality": 90,
        })

    # Generate duplicates (copies with slight modifications)
    for i in range(count // 4):
        configs.append({
            "name": f"dup_{i:02d}_a.jpg",
            "size": (300, 300),
            "type": "pattern",
            "pattern": "circles",
            "format": "JPEG",
            "quality": 85,
            "seed": 1000 + i,  # Same seed = same image
        })
        configs.append({
            "name": f"dup_{i:02d}_b.jpg",
            "size": (300, 300),
            "type": "pattern",
            "pattern": "circles",
            "format": "JPEG",
            "quality": 80,  # Slightly different quality
            "seed": 1000 + i,  # Same seed = same image
        })

    results = []
    for config in configs:
        seed = config.get("seed")
        if seed:
            random.seed(seed)

        width, height = config["size"]

        if config["type"] == "gradient":
            img = generate_gradient_image(width, height, config["colors"])
        else:
            img = generate_pattern_image(width, height, config.get("pattern", "checkerboard"))

        # Reset random seed
        if seed:
            random.seed()

        # Save image
        output_path = output_dir / config["name"]
        save_kwargs = {"format": config["format"]}
        if config["format"] == "JPEG":
            save_kwargs["quality"] = config.get("quality", 85)

        img.save(output_path, **save_kwargs)

        results.append({
            "name": config["name"],
            "size": config["size"],
            "format": config["format"],
            "file_size": output_path.stat().st_size,
        })

    return results


def main():
    """Generate test images for Scenario 4."""
    output_dir = Path(__file__).parent.parent / "data" / "scenario4" / "sample_images"

    print("Generating test images...")
    print()

    results = generate_test_images(output_dir, count=20)

    # Summary
    total_size = sum(r["file_size"] for r in results)

    print(f"Generated {len(results)} images in {output_dir}")
    print()
    print("Summary:")
    print("-" * 60)
    print(f"{'File':<25} {'Size':>12} {'Dimensions':>15} {'Format':<8}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x["file_size"], reverse=True)[:10]:
        print(f"{r['name']:<25} {r['file_size']:>10,} B {str(r['size']):>15} {r['format']:<8}")

    if len(results) > 10:
        print(f"... and {len(results) - 10} more files")

    print("-" * 60)
    print(f"Total: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print()

    # Count duplicates
    dup_count = len([r for r in results if r["name"].startswith("dup_")])
    print(f"Note: {dup_count} images are intentional duplicates (dup_*_a and dup_*_b pairs)")


if __name__ == "__main__":
    main()
