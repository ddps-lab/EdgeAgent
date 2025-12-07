#!/usr/bin/env python3
"""
Mock MCP Server using FastMCP

다양한 tool 타입을 시뮬레이션하는 MCP server입니다.

Usage:
    python mock_mcp_server.py slack --location CLOUD
    python mock_mcp_server.py credentials --location DEVICE
    python mock_mcp_server.py compute --location CLOUD
"""

import sys
import argparse
from fastmcp import FastMCP


def create_slack_server(location: str) -> FastMCP:
    """Slack API 시뮬레이션 서버"""
    mcp = FastMCP(f"mock-slack-{location}")

    @mcp.tool()
    def send_message(channel: str, message: str) -> str:
        """Send a message to a Slack channel"""
        return f"[{location}] Message sent to #{channel}: {message}"

    @mcp.tool()
    def list_channels() -> str:
        """List available Slack channels"""
        return f"[{location}] Channels: #general, #random, #dev"

    @mcp.tool()
    def get_user_info(user_id: str) -> str:
        """Get information about a Slack user"""
        return f"[{location}] User {user_id}: name=TestUser, email=test@example.com"

    return mcp


def create_credentials_server(location: str) -> FastMCP:
    """Credential storage 시뮬레이션 서버"""
    mcp = FastMCP(f"mock-credentials-{location}")

    @mcp.tool()
    def get_secret(key: str) -> str:
        """Retrieve a secret from secure storage"""
        return f"[{location}] Secret '{key}': ***REDACTED***"

    @mcp.tool()
    def store_secret(key: str, value: str) -> str:
        """Store a secret in secure storage"""
        return f"[{location}] Secret '{key}' stored successfully"

    @mcp.tool()
    def list_secrets() -> str:
        """List all secret keys (not values)"""
        return f"[{location}] Keys: api_key, db_password, oauth_token"

    @mcp.tool()
    def delete_secret(key: str) -> str:
        """Delete a secret from storage"""
        return f"[{location}] Secret '{key}' deleted"

    return mcp


def create_compute_server(location: str) -> FastMCP:
    """GPU compute 시뮬레이션 서버"""
    mcp = FastMCP(f"mock-compute-{location}")

    @mcp.tool()
    def matrix_multiply(size: int) -> str:
        """Perform GPU-accelerated matrix multiplication"""
        return f"[{location}] Matrix multiplication {size}x{size} completed (GPU accelerated)"

    @mcp.tool()
    def run_inference(model: str, input_data: str) -> str:
        """Run ML model inference"""
        return f"[{location}] Inference on '{model}' completed. Result: prediction_score=0.95"

    @mcp.tool()
    def train_model(model: str, epochs: int = 10) -> str:
        """Train a machine learning model"""
        return f"[{location}] Training '{model}' for {epochs} epochs completed"

    @mcp.tool()
    def get_gpu_info() -> str:
        """Get GPU information"""
        return f"[{location}] GPU: NVIDIA A100, Memory: 40GB, Utilization: 45%"

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Mock MCP Server")
    parser.add_argument("server_type", choices=["slack", "credentials", "compute"])
    parser.add_argument("--location", default="UNKNOWN", help="Location identifier")
    args = parser.parse_args()

    # 서버 타입에 따라 생성
    if args.server_type == "slack":
        mcp = create_slack_server(args.location)
    elif args.server_type == "credentials":
        mcp = create_credentials_server(args.location)
    elif args.server_type == "compute":
        mcp = create_compute_server(args.location)
    else:
        print(f"Unknown server type: {args.server_type}", file=sys.stderr)
        sys.exit(1)

    # stdio로 실행
    mcp.run()


if __name__ == "__main__":
    main()
