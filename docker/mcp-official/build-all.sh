#!/bin/bash
# Build all MCP official servers with ToolTimer profiling
# Usage: ./build-all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

echo "Building all MCP official servers with ToolTimer..."
echo "Root directory: $ROOT_DIR"
echo ""

# Build all 4 servers in parallel
docker build -f docker/mcp-official/Dockerfile.git -t edgeagent-mcp-git . &
docker build -f docker/mcp-official/Dockerfile.time -t edgeagent-mcp-time . &
docker build -f docker/mcp-official/Dockerfile.filesystem -t edgeagent-mcp-filesystem . &
docker build -f docker/mcp-official/Dockerfile.sequentialthinking -t edgeagent-mcp-sequentialthinking . &

# Wait for all builds to complete
wait

echo ""
echo "=== Build Complete ==="
docker images | grep edgeagent-mcp
