#!/bin/bash
# Build and push WASM MCP server images to private registry

set -e

REGISTRY="srv2.ddps.cloud/wasm-mcp"
WASM_DIR="/home/mhsong/wasm_mcp/target/wasm32-wasip2/release"
DOCKERFILE_DIR="/home/mhsong/edgeagent/docker/wasm"

# Server name : WASM file mapping
declare -A servers=(
    ["time"]="mcp_server_time_http.wasm"
    ["git"]="mcp_server_git_http.wasm"
    ["filesystem"]="mcp_server_filesystem_http.wasm"
    ["log-parser"]="mcp_server_log_parser_http.wasm"
    ["data-aggregate"]="mcp_server_data_aggregate_http.wasm"
    ["image-resize"]="mcp_server_image_resize_http.wasm"
    ["sequential-thinking"]="mcp_server_sequential_thinking_http.wasm"
    ["fetch"]="mcp_server_fetch_http.wasm"
    ["summarize"]="mcp_server_summarize_http.wasm"
)

# Create temporary build context
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "=== Building WASM MCP Server Images ==="
echo "Registry: $REGISTRY"
echo "WASM Directory: $WASM_DIR"
echo ""

for name in "${!servers[@]}"; do
    wasm="${servers[$name]}"
    echo "--- Building $name ---"

    # Copy WASM file to build context
    cp "$WASM_DIR/$wasm" "$BUILD_DIR/"
    cp "$DOCKERFILE_DIR/Dockerfile.$name" "$BUILD_DIR/Dockerfile"

    # Build image with wasi/wasm platform for runwasi detection
    docker buildx build --platform wasi/wasm \
        --provenance=false \
        -t "$REGISTRY/$name:latest" \
        --push \
        "$BUILD_DIR"

    # Clean up build context
    rm "$BUILD_DIR/$wasm" "$BUILD_DIR/Dockerfile"

    echo ""
done

echo "=== All images built and pushed successfully ==="
echo ""
echo "Images:"
for name in "${!servers[@]}"; do
    echo "  - $REGISTRY/$name:latest"
done
