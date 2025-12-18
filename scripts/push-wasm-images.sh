#!/bin/bash
# Build and push WASM MCP server images to private registry using crane
# Creates OCI images with wasm/wasip2 platform for containerd runwasi

set -e

REGISTRY="srv2.ddps.cloud/wasm-mcp"
WASM_DIR="/home/mhsong/edgeagent/wasm_mcp/target/wasm32-wasip2/release"

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

echo "=== Building WASM MCP Server Images ==="
echo "Registry: $REGISTRY"
echo "WASM Directory: $WASM_DIR"
echo ""

for name in "${!servers[@]}"; do
    wasm="${servers[$name]}"
    echo "--- Building $name ---"

    WASM_FILE="$WASM_DIR/$wasm"
    if [ ! -f "$WASM_FILE" ]; then
        echo "ERROR: WASM file not found: $WASM_FILE"
        exit 1
    fi

    # Create temporary OCI layout directory
    BUILD_DIR=$(mktemp -d)
    trap "rm -rf $BUILD_DIR" EXIT
    mkdir -p "$BUILD_DIR/blobs/sha256"

    # Create layer tarball
    LAYER_TAR="$BUILD_DIR/layer.tar"
    tar -cf "$LAYER_TAR" -C "$WASM_DIR" "$wasm"
    LAYER_DIGEST=$(sha256sum "$LAYER_TAR" | cut -d' ' -f1)
    LAYER_SIZE=$(stat -c%s "$LAYER_TAR")
    mv "$LAYER_TAR" "$BUILD_DIR/blobs/sha256/$LAYER_DIGEST"

    # Create config blob with wasm/wasip2 platform
    CONFIG_FILE="$BUILD_DIR/config.json"
    cat > "$CONFIG_FILE" <<EOF
{
  "architecture": "wasm",
  "os": "wasip2",
  "config": {
    "Entrypoint": ["/$wasm"]
  },
  "rootfs": {
    "type": "layers",
    "diff_ids": ["sha256:$LAYER_DIGEST"]
  },
  "history": []
}
EOF
    CONFIG_DIGEST=$(sha256sum "$CONFIG_FILE" | cut -d' ' -f1)
    CONFIG_SIZE=$(stat -c%s "$CONFIG_FILE")
    mv "$CONFIG_FILE" "$BUILD_DIR/blobs/sha256/$CONFIG_DIGEST"

    # Create manifest
    MANIFEST_FILE="$BUILD_DIR/manifest.json"
    cat > "$MANIFEST_FILE" <<EOF
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:$CONFIG_DIGEST",
    "size": $CONFIG_SIZE
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar",
      "digest": "sha256:$LAYER_DIGEST",
      "size": $LAYER_SIZE
    }
  ]
}
EOF
    MANIFEST_DIGEST=$(sha256sum "$MANIFEST_FILE" | cut -d' ' -f1)
    MANIFEST_SIZE=$(stat -c%s "$MANIFEST_FILE")
    mv "$MANIFEST_FILE" "$BUILD_DIR/blobs/sha256/$MANIFEST_DIGEST"

    # Create oci-layout
    echo '{"imageLayoutVersion": "1.0.0"}' > "$BUILD_DIR/oci-layout"

    # Create index.json
    cat > "$BUILD_DIR/index.json" <<EOF
{
  "schemaVersion": 2,
  "manifests": [
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "digest": "sha256:$MANIFEST_DIGEST",
      "size": $MANIFEST_SIZE,
      "platform": {
        "architecture": "wasm",
        "os": "wasip2"
      }
    }
  ]
}
EOF

    # Push OCI layout using crane
    IMAGE_TAG="$REGISTRY/$name:latest"
    crane push "$BUILD_DIR" "$IMAGE_TAG"

    echo "Pushed: $IMAGE_TAG"

    # Clean up
    rm -rf "$BUILD_DIR"
    trap - EXIT

    echo ""
done

echo "=== All images built and pushed successfully ==="
echo ""
echo "Images:"
for name in "${!servers[@]}"; do
    echo "  - $REGISTRY/$name:latest"
done
