#!/usr/bin/env python3
"""
Generate Test Git Repository for Scenario 1

Creates a sample Git repository with multiple commits for testing:
- Initial commit with basic Python files
- Feature commits with new functionality
- Bug fix commits
- Refactoring commits

This provides realistic git history for code review testing.
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta


def run_git(repo_path: Path, *args):
    """Run a git command in the specified repository."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Git error: {result.stderr}")
    return result


def generate_test_repo(output_dir: Path):
    """Generate a test Git repository with sample commits."""

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

    # Authenticate user
    user = authenticate_user("admin", "password123")
    if user:
        print(f"Welcome, {user.name}!")

        # Process some data
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
    """Authenticate a user with username and password.

    Args:
        username: The username to authenticate
        password: The password to verify

    Returns:
        User object if authenticated, None otherwise
    """
    # TODO: Replace with actual authentication
    if username == "admin" and password == "password123":
        return User(id=1, name="Admin", email="admin@example.com")
    return None
''')

    (output_dir / "app" / "data.py").write_text('''"""Data processing module."""

from typing import List


def process_data(items: List[int]) -> int:
    """Process a list of items and return the sum.

    Args:
        items: List of integers to process

    Returns:
        Sum of all items
    """
    total = 0
    for item in items:
        total = total + item
    return total


def filter_data(items: List[int], threshold: int) -> List[int]:
    """Filter items above a threshold.

    Args:
        items: List of integers to filter
        threshold: Minimum value to include

    Returns:
        Filtered list of items
    """
    result = []
    for item in items:
        if item > threshold:
            result.append(item)
    return result
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "Initial commit: basic project structure")

    # Commit 2: Add new feature
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
    """Get user by ID.

    Args:
        user_id: The user ID to look up

    Returns:
        APIResponse with user data or error
    """
    # Mock implementation
    if user_id == 1:
        return APIResponse(
            status=200,
            data={"id": 1, "name": "Admin", "email": "admin@example.com"},
        )
    return APIResponse(status=404, data=None, message="User not found")


def create_user(data: Dict[str, Any]) -> APIResponse:
    """Create a new user.

    Args:
        data: User data dictionary

    Returns:
        APIResponse with created user or error
    """
    if not data.get("name") or not data.get("email"):
        return APIResponse(status=400, data=None, message="Missing required fields")

    # Mock: Return created user
    return APIResponse(
        status=201,
        data={"id": 2, **data},
        message="User created successfully",
    )
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "feat: add API endpoints for user management")

    # Commit 3: Bug fix
    (output_dir / "app" / "data.py").write_text('''"""Data processing module."""

from typing import List


def process_data(items: List[int]) -> int:
    """Process a list of items and return the sum.

    Args:
        items: List of integers to process

    Returns:
        Sum of all items
    """
    if not items:
        return 0
    return sum(items)  # Fixed: use built-in sum


def filter_data(items: List[int], threshold: int) -> List[int]:
    """Filter items above a threshold.

    Args:
        items: List of integers to filter
        threshold: Minimum value to include

    Returns:
        Filtered list of items
    """
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
    """Hash a password with salt.

    Args:
        password: Plain text password
        salt: Optional salt bytes

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = os.urandom(32)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hashed.hex(), salt


def verify_password(password: str, hashed: str, salt: bytes) -> bool:
    """Verify a password against its hash.

    Args:
        password: Plain text password to verify
        hashed: The stored hash
        salt: The salt used for hashing

    Returns:
        True if password matches, False otherwise
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user with username and password.

    Args:
        username: The username to authenticate
        password: The password to verify

    Returns:
        User object if authenticated, None otherwise
    """
    # TODO: Replace with database lookup
    # For demo: admin user with proper password hashing
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
    """Tests for process_data function."""

    def test_empty_list(self):
        """Test with empty list."""
        assert process_data([]) == 0

    def test_single_item(self):
        """Test with single item."""
        assert process_data([5]) == 5

    def test_multiple_items(self):
        """Test with multiple items."""
        assert process_data([1, 2, 3, 4, 5]) == 15


class TestFilterData:
    """Tests for filter_data function."""

    def test_empty_list(self):
        """Test with empty list."""
        assert filter_data([], 5) == []

    def test_all_above_threshold(self):
        """Test when all items are above threshold."""
        assert filter_data([10, 20, 30], 5) == [10, 20, 30]

    def test_all_below_threshold(self):
        """Test when all items are below threshold."""
        assert filter_data([1, 2, 3], 5) == []

    def test_mixed(self):
        """Test with mixed values."""
        assert filter_data([1, 5, 10, 3, 15], 5) == [10, 15]
''')

    run_git(output_dir, "add", ".")
    run_git(output_dir, "commit", "-m", "test: add unit tests for data module")

    print(f"Generated test repository at: {output_dir}")

    # Show git log
    result = run_git(output_dir, "log", "--oneline")
    print("\nCommit history:")
    print(result.stdout)

    return output_dir


def main():
    """Generate test Git repository for Scenario 1."""
    output_dir = Path(__file__).parent.parent / "data" / "scenario1" / "sample_repo"

    print("Generating test Git repository...")
    print()

    generate_test_repo(output_dir)

    print()
    print("Repository structure:")
    for path in sorted(output_dir.rglob("*")):
        if ".git" not in str(path):
            relative = path.relative_to(output_dir)
            if path.is_file():
                print(f"  {relative} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
