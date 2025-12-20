#!/usr/bin/env python3
"""
Filesystem MCP Server

Provides file system operations matching the WASM filesystem server.
14 tools for reading, writing, and managing files.

Usage:
    python servers/filesystem_server.py
"""

import os
import base64
import json
from pathlib import Path
from typing import Literal
from fastmcp import FastMCP
from timing import ToolTimer, measure_io

mcp = FastMCP("filesystem")


# ============================================================================
# Helper Functions
# ============================================================================

def head_lines(content: str, n: int) -> str:
    """Get first n lines of content"""
    return "\n".join(content.split("\n")[:n])


def tail_lines(content: str, n: int) -> str:
    """Get last n lines of content"""
    lines = content.split("\n")
    return "\n".join(lines[-n:])


def format_size(size: int) -> str:
    """Format file size in human-readable form"""
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
    elif size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size} B"


def get_mime_type(path: str) -> str:
    """Get MIME type from file extension"""
    ext = Path(path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    return mime_types.get(ext, "application/octet-stream")


# ============================================================================
# Tool Implementations (14 tools matching WASM version)
# ============================================================================

@mcp.tool()
def read_file(path: str, head: int | None = None, tail: int | None = None) -> str:
    """
    Read a text file's contents.

    Args:
        path: Path to the file to read
        head: Only return first N lines
        tail: Only return last N lines

    Returns:
        File contents (or portion if head/tail specified)
    """
    timer = ToolTimer("read_file")

    try:
        content = measure_io(lambda: Path(path).read_text())

        if head is not None and tail is not None:
            timer.finish()
            return "Error: Cannot specify both head and tail parameters"

        if head is not None:
            result = head_lines(content, head)
        elif tail is not None:
            result = tail_lines(content, tail)
        else:
            result = content

        timer.finish()
        return result
    except Exception as e:
        timer.finish()
        return f"Error reading file: {e}"


@mcp.tool()
def read_text_file(path: str, head: int | None = None, tail: int | None = None) -> str:
    """
    Read a text file's contents (alias for read_file).

    Args:
        path: Path to the file to read
        head: Only return first N lines
        tail: Only return last N lines

    Returns:
        File contents
    """
    timer = ToolTimer("read_text_file")

    try:
        content = measure_io(lambda: Path(path).read_text())

        if head is not None and tail is not None:
            timer.finish()
            return "Error: Cannot specify both head and tail parameters"

        if head is not None:
            result = head_lines(content, head)
        elif tail is not None:
            result = tail_lines(content, tail)
        else:
            result = content

        timer.finish()
        return result
    except Exception as e:
        timer.finish()
        return f"Error reading file: {e}"


@mcp.tool()
def read_media_file(path: str) -> str:
    """
    Read a media file and return as base64 data URL.

    Args:
        path: Path to the media file

    Returns:
        Base64 data URL (data:mime/type;base64,...)
    """
    timer = ToolTimer("read_media_file")

    try:
        data = measure_io(lambda: Path(path).read_bytes())
        base64_data = base64.b64encode(data).decode("utf-8")
        mime_type = get_mime_type(path)

        timer.finish()
        return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        timer.finish()
        return f"Error reading media file: {e}"


@mcp.tool()
def read_multiple_files(paths: list[str]) -> str:
    """
    Read multiple files at once.

    Args:
        paths: List of file paths to read

    Returns:
        Combined contents with separators
    """
    timer = ToolTimer("read_multiple_files")

    results = []
    for path in paths:
        try:
            content = measure_io(lambda p=path: Path(p).read_text())
            results.append(f"{path}:\n{content}")
        except Exception as e:
            results.append(f"{path}: Error - {e}")

    timer.finish()
    return "\n---\n".join(results)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        path: Path to write to
        content: Content to write

    Returns:
        Success message
    """
    timer = ToolTimer("write_file")

    try:
        measure_io(lambda: Path(path).write_text(content))
        timer.finish()
        return f"Successfully wrote to {path}"
    except Exception as e:
        timer.finish()
        return f"Error writing file: {e}"


@mcp.tool()
def edit_file(
    path: str,
    edits: list[dict],
    dry_run: bool = False
) -> str:
    """
    Apply edits to a file.

    Args:
        path: Path to the file to edit
        edits: List of edit operations, each with 'old_text' and 'new_text'
        dry_run: If True, show what would change without applying

    Returns:
        Summary of changes made
    """
    timer = ToolTimer("edit_file")

    try:
        content = measure_io(lambda: Path(path).read_text())
        original = content
        changes = []

        for edit in edits:
            old_text = edit.get("old_text", "")
            new_text = edit.get("new_text", "")

            if old_text in content:
                content = content.replace(old_text, new_text)
                changes.append(f"- {old_text}\n+ {new_text}")
            else:
                timer.finish()
                return f"Error: Text not found: '{old_text[:50]}...'"

        if dry_run:
            timer.finish()
            return f"Dry run - changes that would be made:\n" + "\n".join(changes)
        else:
            measure_io(lambda: Path(path).write_text(content))
            timer.finish()
            return f"Applied {len(edits)} edit(s) to {path}"
    except Exception as e:
        timer.finish()
        return f"Error editing file: {e}"


@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a directory (including parent directories).

    Args:
        path: Directory path to create

    Returns:
        Success message
    """
    timer = ToolTimer("create_directory")

    try:
        measure_io(lambda: Path(path).mkdir(parents=True, exist_ok=True))
        timer.finish()
        return f"Successfully created directory {path}"
    except Exception as e:
        timer.finish()
        return f"Error creating directory: {e}"


@mcp.tool()
def list_directory(path: str) -> str:
    """
    List contents of a directory.

    Args:
        path: Directory path to list

    Returns:
        List of files and directories
    """
    timer = ToolTimer("list_directory")

    try:
        entries = measure_io(lambda: list(Path(path).iterdir()))

        items = []
        for entry in sorted(entries, key=lambda x: x.name):
            file_type = "[DIR]" if entry.is_dir() else "[FILE]"
            items.append(f"{file_type} {entry.name}")

        timer.finish()
        return "\n".join(items) if items else "Directory is empty"
    except Exception as e:
        timer.finish()
        return f"Error listing directory: {e}"


@mcp.tool()
def list_directory_with_sizes(
    path: str,
    sort_by: Literal["name", "size"] = "name"
) -> str:
    """
    List directory contents with file sizes.

    Args:
        path: Directory path to list
        sort_by: Sort by "name" or "size"

    Returns:
        Formatted list with sizes
    """
    timer = ToolTimer("list_directory_with_sizes")

    try:
        entries = measure_io(lambda: list(Path(path).iterdir()))

        items = []
        total_size = 0
        file_count = 0
        dir_count = 0

        for entry in entries:
            is_dir = entry.is_dir()
            size = 0 if is_dir else measure_io(lambda e=entry: e.stat().st_size)
            items.append((entry.name, is_dir, size))

        # Sort
        if sort_by == "size":
            items.sort(key=lambda x: x[2], reverse=True)
        else:
            items.sort(key=lambda x: x[0])

        result = []
        for name, is_dir, size in items:
            file_type = "[DIR]" if is_dir else "[FILE]"
            size_str = "" if is_dir else format_size(size)
            result.append(f"{file_type} {name:30} {size_str:>10}")

            if is_dir:
                dir_count += 1
            else:
                file_count += 1
                total_size += size

        result.append("")
        result.append(f"Total: {file_count} files, {dir_count} directories")
        result.append(f"Combined size: {format_size(total_size)}")

        timer.finish()
        return "\n".join(result)
    except Exception as e:
        timer.finish()
        return f"Error listing directory: {e}"


def _build_tree(path: Path, exclude: list[str]) -> list:
    """Build directory tree recursively"""
    result = []
    try:
        entries = list(path.iterdir())
    except PermissionError:
        return result

    for entry in sorted(entries, key=lambda x: x.name):
        # Check exclusions
        if any(p in entry.name for p in exclude):
            continue

        item = {
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file"
        }

        if entry.is_dir():
            item["children"] = _build_tree(entry, exclude)

        result.append(item)

    return result


@mcp.tool()
def directory_tree(path: str, exclude_patterns: list[str] | None = None) -> str:
    """
    Get directory tree as JSON.

    Args:
        path: Root directory path
        exclude_patterns: Patterns to exclude (e.g., [".git", "node_modules"])

    Returns:
        JSON representation of directory tree
    """
    timer = ToolTimer("directory_tree")

    try:
        exclude = exclude_patterns or []
        tree = measure_io(lambda: _build_tree(Path(path), exclude))
        timer.finish()
        return json.dumps(tree, indent=2)
    except Exception as e:
        timer.finish()
        return f"Error building tree: {e}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """
    Move or rename a file.

    Args:
        source: Source path
        destination: Destination path

    Returns:
        Success message
    """
    timer = ToolTimer("move_file")

    try:
        measure_io(lambda: Path(source).rename(destination))
        timer.finish()
        return f"Successfully moved {source} to {destination}"
    except Exception as e:
        timer.finish()
        return f"Error moving file: {e}"


def _search_recursive(path: Path, pattern: str, exclude: list[str], results: list):
    """Search files recursively"""
    try:
        for entry in path.iterdir():
            # Check exclusions
            if any(p in entry.name for p in exclude):
                continue

            # Simple glob matching
            matches = False
            if "*" in pattern:
                if pattern.startswith("*."):
                    matches = entry.name.endswith(pattern[1:])
                elif pattern.endswith("*"):
                    matches = entry.name.startswith(pattern[:-1])
                else:
                    matches = pattern.replace("*", "") in entry.name
            else:
                matches = pattern in entry.name

            if matches:
                results.append(str(entry))

            if entry.is_dir():
                _search_recursive(entry, pattern, exclude, results)
    except PermissionError:
        pass


@mcp.tool()
def search_files(
    path: str,
    pattern: str,
    exclude_patterns: list[str] | None = None
) -> str:
    """
    Search for files matching a pattern.

    Args:
        path: Root directory to search
        pattern: Search pattern (supports * wildcards)
        exclude_patterns: Patterns to exclude

    Returns:
        List of matching file paths
    """
    timer = ToolTimer("search_files")

    try:
        exclude = exclude_patterns or []
        results = []
        measure_io(lambda: _search_recursive(Path(path), pattern, exclude, results))

        timer.finish()
        return "\n".join(results) if results else "No matches found"
    except Exception as e:
        timer.finish()
        return f"Error searching: {e}"


@mcp.tool()
def get_file_info(path: str) -> str:
    """
    Get detailed file information.

    Args:
        path: Path to the file

    Returns:
        File metadata
    """
    timer = ToolTimer("get_file_info")

    try:
        p = Path(path)
        stat = measure_io(lambda: p.stat())

        file_type = "directory" if p.is_dir() else "file" if p.is_file() else "other"

        info = [
            f"size: {stat.st_size}",
            f"type: {file_type}",
            f"modified: {stat.st_mtime}",
            f"accessed: {stat.st_atime}",
            f"created: {stat.st_ctime}",
        ]

        timer.finish()
        return "\n".join(info)
    except Exception as e:
        timer.finish()
        return f"Error getting file info: {e}"


@mcp.tool()
def list_allowed_directories() -> str:
    """
    List allowed directories.

    Returns:
        Information about directory access
    """
    timer = ToolTimer("list_allowed_directories")
    result = "Native Python server has access to all directories the process can read."
    timer.finish()
    return result


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8004"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        mcp.run()
