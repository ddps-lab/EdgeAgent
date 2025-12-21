#!/usr/bin/env python3
"""
Image Resize MCP Server

Provides independent, primitive tools for image processing.
Each tool is self-contained - LLM orchestrates the workflow.

Tools:
- get_image_info: Get metadata about an image
- resize_image: Resize a single image
- scan_directory: List images in a directory
- compute_image_hash: Compute perceptual hash for duplicate detection
- compare_hashes: Compare hashes to find duplicates

Design Principle:
- Each tool is INDEPENDENT and takes explicit inputs
- NO internal tool-to-tool calls
- LLM decides the workflow: scan → get_info → resize → compare

Usage:
    python servers/image_resize_server.py
"""

import os
import io
import base64
from pathlib import Path
from typing import Literal
from fastmcp import FastMCP
from timing import ToolTimer, measure_io

mcp = FastMCP("image_resize")

# Lazy imports for optional dependencies
_PIL_AVAILABLE = False
_IMAGEHASH_AVAILABLE = False


def _ensure_pil():
    global _PIL_AVAILABLE
    if not _PIL_AVAILABLE:
        try:
            from PIL import Image
            _PIL_AVAILABLE = True
        except ImportError:
            raise ImportError("Pillow is required: pip install pillow")


def _ensure_imagehash():
    global _IMAGEHASH_AVAILABLE
    if not _IMAGEHASH_AVAILABLE:
        try:
            import imagehash
            _IMAGEHASH_AVAILABLE = True
        except ImportError:
            raise ImportError("imagehash is required: pip install imagehash")


# ============================================================
# PRIMITIVE TOOLS - Each is independent, LLM orchestrates
# ============================================================

@mcp.tool()
def get_image_info(image_path: str) -> dict:
    """
    Get detailed information about an image.

    This is typically the FIRST step when working with an image.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with:
        - path: Image path
        - format: Image format (JPEG, PNG, etc.)
        - width, height: Dimensions
        - size_bytes: File size
        - mode: Color mode (RGB, RGBA, etc.)
    """
    timer = ToolTimer("get_image_info")
    _ensure_pil()
    from PIL import Image

    try:
        img = measure_io(lambda: Image.open(image_path))
        with img:
            size_bytes = measure_io(lambda: os.path.getsize(image_path))
            result = {
                "path": image_path,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": size_bytes,
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 0,
            }
        timer.finish()
        return result
    except Exception as e:
        timer.finish()
        return {"path": image_path, "error": str(e)}


@mcp.tool()
def resize_image(
    image_path: str,
    width: int | None = None,
    height: int | None = None,
    max_size: int | None = None,
    quality: int = 85,
    output_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG",
) -> dict:
    """
    Resize an image and return as base64.

    Provide either (width, height), just width, just height, or max_size.

    Args:
        image_path: Path to the input image
        width: Target width (maintains aspect ratio if height not given)
        height: Target height (maintains aspect ratio if width not given)
        max_size: Maximum dimension (constrains both width and height)
        quality: Output quality (1-100, for JPEG/WEBP)
        output_format: Output format

    Returns:
        Dictionary with:
        - success: Whether resize succeeded
        - original_size: Original dimensions
        - new_size: New dimensions
        - original_bytes, output_bytes: Size comparison
        - data_base64: Base64-encoded output image
    """
    timer = ToolTimer("resize_image")
    _ensure_pil()
    from PIL import Image

    try:
        img = measure_io(lambda: Image.open(image_path))
        with img:
            original_size = img.size
            original_bytes = measure_io(lambda: os.path.getsize(image_path))

            # Calculate new size (compute - no I/O)
            if max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            elif width and height:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            elif width:
                ratio = width / img.width
                img = img.resize(
                    (width, int(img.height * ratio)), Image.Resampling.LANCZOS
                )
            elif height:
                ratio = height / img.height
                img = img.resize(
                    (int(img.width * ratio), height), Image.Resampling.LANCZOS
                )
            else:
                timer.finish()
                return {
                    "success": False,
                    "error": "No resize parameters provided (width, height, or max_size)",
                    "path": image_path,
                }

            # Convert to RGB if needed for JPEG
            if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Save to buffer (in-memory, not I/O)
            buffer = io.BytesIO()
            img.save(buffer, format=output_format, quality=quality)
            buffer.seek(0)

            output_bytes = len(buffer.getvalue())
            output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            result = {
                "success": True,
                "path": image_path,
                "original_size": original_size,
                "new_size": img.size,
                "original_bytes": original_bytes,
                "output_bytes": output_bytes,
                "reduction_ratio": round(output_bytes / original_bytes, 4) if original_bytes > 0 else 0,
                "format": output_format,
                "data_base64": output_b64,
            }
        timer.finish()
        return result
    except Exception as e:
        timer.finish()
        return {"success": False, "error": str(e), "path": image_path}


@mcp.tool()
def scan_directory(
    directory: str,
    extensions: list[str] | None = None,
    recursive: bool = True,
    include_info: bool = False,
) -> dict:
    """
    Scan a directory for image files.

    This is typically the FIRST step for batch operations.

    Args:
        directory: Directory path to scan
        extensions: File extensions to include (default: common image formats)
        recursive: Whether to scan subdirectories
        include_info: Whether to include detailed info per image (slower)

    Returns:
        Dictionary with:
        - directory: Scanned directory
        - image_count: Number of images found
        - image_paths: List of image paths (use these with other tools)
        - total_size_bytes: Combined size of all images
    """
    timer = ToolTimer("scan_directory")

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"]

    path = Path(directory)
    if not path.exists():
        timer.finish()
        return {"error": f"Directory not found: {directory}"}

    image_paths = []
    total_size = 0

    pattern = "**/*" if recursive else "*"
    for file_path in path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_paths.append(str(file_path))
            total_size += measure_io(lambda fp=file_path: fp.stat().st_size)

    result = {
        "directory": directory,
        "image_count": len(image_paths),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "image_paths": image_paths,
    }

    if include_info:
        # Get detailed info for each image (slower but more useful)
        images_info = []
        for img_path in image_paths:
            info = get_image_info.fn(img_path) if hasattr(get_image_info, 'fn') else get_image_info(img_path)
            images_info.append(info)
        result["images"] = images_info

    timer.finish()
    return result


@mcp.tool()
def compute_image_hash(
    image_path: str,
    hash_type: Literal["phash", "dhash", "ahash", "whash"] = "phash",
) -> dict:
    """
    Compute perceptual hash of an image for duplicate detection.

    Use this to get hashes, then use compare_hashes to find duplicates.

    Args:
        image_path: Path to the image
        hash_type: Type of perceptual hash to compute

    Returns:
        Dictionary with:
        - path: Image path
        - hash: Computed hash value (string)
        - hash_type: Type of hash used
    """
    timer = ToolTimer("compute_image_hash")
    _ensure_pil()
    _ensure_imagehash()
    from PIL import Image
    import imagehash

    hash_funcs = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
        "whash": imagehash.whash,
    }

    try:
        img = measure_io(lambda: Image.open(image_path))
        with img:
            hash_func = hash_funcs.get(hash_type, imagehash.phash)
            # hash computation is CPU-bound (compute, not I/O)
            hash_value = hash_func(img)
            result = {
                "path": image_path,
                "hash": str(hash_value),
                "hash_type": hash_type,
            }
        timer.finish()
        return result
    except Exception as e:
        timer.finish()
        return {"path": image_path, "error": str(e)}


@mcp.tool()
def compare_hashes(
    hashes: list[dict],
    threshold: int = 5,
) -> dict:
    """
    Compare image hashes to find duplicates/similar images.

    IMPORTANT: This tool requires the 'hashes' parameter which is a LIST of hash results.
    You MUST call compute_image_hash for EACH image first, collect all results into a list,
    and then pass that list to this tool.

    Example workflow:
    1. scan_directory(directory="/path/to/images") -> result with image_paths
    2. For EACH path in result["image_paths"]:
       compute_image_hash(image_path=path) -> hash_result
       Collect all hash_results into a list
    3. compare_hashes(hashes=[hash_result1, hash_result2, ...], threshold=5)

    Args:
        hashes: REQUIRED. List of hash results from compute_image_hash().
                Each item should be a dict with 'path' and 'hash' keys.
                Example: [{"path": "/img1.jpg", "hash": "abc123"}, {"path": "/img2.jpg", "hash": "abc124"}]
        threshold: Maximum hash difference to consider as duplicate.
                   0 = exact match only, 5 = similar images, 10 = more lenient matching.
                   Default: 5

    Returns:
        Dictionary with:
        - duplicate_groups: List of groups of similar images (each group is a list of paths)
        - unique_paths: List of paths that have no duplicates
        - unique_count: Number of unique images
        - total_compared: Total images compared
    """
    timer = ToolTimer("compare_hashes")
    _ensure_imagehash()
    import imagehash

    # Filter out errors (pure compute, no I/O)
    valid_hashes = {
        h["path"]: h["hash"]
        for h in hashes
        if "hash" in h and "error" not in h
    }

    if len(valid_hashes) < 2:
        timer.finish()
        return {
            "total_compared": len(valid_hashes),
            "duplicate_groups": [],
            "unique_count": len(valid_hashes),
            "errors": [h for h in hashes if "error" in h],
        }

    # Find similar pairs (pure compute)
    groups = []
    processed = set()

    paths = list(valid_hashes.keys())
    for i, path1 in enumerate(paths):
        if path1 in processed:
            continue

        group = [path1]
        hash1 = imagehash.hex_to_hash(valid_hashes[path1])

        for path2 in paths[i + 1:]:
            if path2 in processed:
                continue

            hash2 = imagehash.hex_to_hash(valid_hashes[path2])
            if hash1 - hash2 <= threshold:
                group.append(path2)
                processed.add(path2)

        if len(group) > 1:
            groups.append(group)
            processed.add(path1)

    # Count unique
    all_duplicates = set()
    for group in groups:
        all_duplicates.update(group)

    unique = [p for p in valid_hashes.keys() if p not in all_duplicates]

    result = {
        "total_compared": len(valid_hashes),
        "duplicate_groups": groups,
        "duplicate_group_count": len(groups),
        "unique_paths": unique,
        "unique_count": len(unique),
        "threshold": threshold,
        "errors": [h for h in hashes if "error" in h],
    }
    timer.finish()
    return result


@mcp.tool()
def batch_resize(
    image_paths: list[str],
    max_size: int = 150,
    quality: int = 75,
    output_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG",
) -> dict:
    """
    Resize multiple images at once (e.g., create thumbnails).

    IMPORTANT: This tool requires the 'image_paths' parameter which is a LIST of file paths.
    You can get this list from scan_directory() or compare_hashes() results.

    Example workflow:
    1. scan_directory(directory="/path/to/images") -> result
    2. batch_resize(image_paths=result["image_paths"], max_size=150)

    Or with duplicate filtering:
    1. scan_directory(...) -> scan_result
    2. compute_image_hash for each path -> collect hashes
    3. compare_hashes(hashes=...) -> comparison_result
    4. batch_resize(image_paths=comparison_result["unique_paths"], max_size=150)

    Args:
        image_paths: REQUIRED. List of image file paths to resize.
                     Get this from scan_directory()["image_paths"] or compare_hashes()["unique_paths"].
                     Example: ["/img1.jpg", "/img2.png", "/img3.jpeg"]
        max_size: Maximum dimension for thumbnails. Default: 150 pixels.
                  Both width and height will be constrained to this size (maintains aspect ratio).
        quality: Output quality (1-100). Default: 75. Higher = better quality but larger files.
        output_format: Output format - "JPEG", "PNG", or "WEBP". Default: "JPEG"

    Returns:
        Dictionary with:
        - results: List of resize results per image (path, success, sizes)
        - successful: Number of successful resizes
        - failed: Number of failures
        - overall_reduction: Ratio of output size to input size (e.g., 0.1 = 90% reduction)
    """
    timer = ToolTimer("batch_resize")
    results = []
    total_input = 0
    total_output = 0
    successful = 0
    failed = 0

    for path in image_paths:
        _ensure_pil()
        from PIL import Image

        try:
            img = measure_io(lambda p=path: Image.open(p))
            with img:
                original_bytes = measure_io(lambda p=path: os.path.getsize(p))
                total_input += original_bytes

                # Create thumbnail (compute, not I/O)
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                buffer = io.BytesIO()
                img.save(buffer, format=output_format, quality=quality)
                output_bytes = len(buffer.getvalue())
                total_output += output_bytes

                results.append({
                    "path": path,
                    "success": True,
                    "original_bytes": original_bytes,
                    "output_bytes": output_bytes,
                    "new_size": img.size,
                })
                successful += 1

        except Exception as e:
            results.append({
                "path": path,
                "success": False,
                "error": str(e),
            })
            failed += 1

    result = {
        "total_images": len(image_paths),
        "successful": successful,
        "failed": failed,
        "total_input_bytes": total_input,
        "total_output_bytes": total_output,
        "overall_reduction": round(total_output / total_input, 4) if total_input > 0 else 0,
        "results": results,
    }
    timer.finish()
    return result


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        # Streamable HTTP for serverless/remote deployment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8002"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        # stdio for local development / Claude Desktop
        mcp.run()
