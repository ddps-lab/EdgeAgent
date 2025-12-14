#!/bin/bash
# Push EdgeAgent images to Private Container Registry
#
# Supports:
#   - Self-hosted Docker Registry (with or without TLS)
#   - Harbor
#   - GitLab Container Registry
#   - Any Docker-compatible registry
#
# Usage:
#   ./scripts/push-to-registry.sh <registry-url> [project/namespace]
#
# Examples:
#   ./scripts/push-to-registry.sh registry.example.com:5000
#   ./scripts/push-to-registry.sh registry.example.com:5000 edgeagent
#   ./scripts/push-to-registry.sh harbor.example.com/library edgeagent
#   ./scripts/push-to-registry.sh registry.gitlab.com/mygroup edgeagent

set -e

# Configuration
REGISTRY_URL="${1:?Usage: $0 <registry-url> [project]}"
PROJECT="${2:-edgeagent}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Remove trailing slash
REGISTRY_URL="${REGISTRY_URL%/}"

# Full registry path
if [[ -n "${PROJECT}" ]]; then
    REGISTRY_PATH="${REGISTRY_URL}/${PROJECT}"
else
    REGISTRY_PATH="${REGISTRY_URL}"
fi

# Image names (local -> remote mapping)
# Note: SubAgent uses single image, LOCATION env var determines behavior
declare -A IMAGES=(
    # Custom FastMCP servers
    ["edgeagent-mcp-fetch"]="mcp-fetch"
    ["edgeagent-mcp-log-parser"]="mcp-log-parser"
    ["edgeagent-mcp-summarize"]="mcp-summarize"
    ["edgeagent-mcp-data-aggregate"]="mcp-data-aggregate"
    ["edgeagent-mcp-image-resize"]="mcp-image-resize"
    # Official MCP servers (via mcp-proxy)
    ["edgeagent-mcp-filesystem"]="mcp-filesystem"
    ["edgeagent-mcp-time"]="mcp-time"
    ["edgeagent-mcp-sequentialthinking"]="mcp-sequentialthinking"
    ["edgeagent-mcp-git"]="mcp-git"
    # SubAgent
    ["edgeagent-subagent"]="subagent"  # LOCATION=EDGE|CLOUD at runtime
)

echo "========================================"
echo "EdgeAgent Private Registry Push Script"
echo "========================================"
echo "Registry: ${REGISTRY_URL}"
echo "Project:  ${PROJECT}"
echo "Path:     ${REGISTRY_PATH}"
echo "Tag:      ${IMAGE_TAG}"
echo "========================================"
echo ""

# Check if login is needed (skip interactive prompt)
echo "[1/3] Checking registry access..."
# Test push access by checking if we can reach the registry
if ! curl -sf "https://${REGISTRY_URL}/v2/" > /dev/null 2>&1; then
    echo "  WARNING: Cannot reach registry at ${REGISTRY_URL}"
    echo "  You may need to run: docker login ${REGISTRY_URL}"
fi
echo ""

# Tag and push images
echo "[2/3] Tagging and pushing images..."
for local_name in "${!IMAGES[@]}"; do
    remote_name="${IMAGES[$local_name]}"
    local_image="${local_name}:latest"
    remote_image="${REGISTRY_PATH}/${remote_name}:${IMAGE_TAG}"

    echo "  ${local_image} -> ${remote_image}"

    if ! docker image inspect "${local_image}" &>/dev/null; then
        echo "    WARNING: Local image '${local_image}' not found, skipping..."
        continue
    fi

    docker tag "${local_image}" "${remote_image}"
    docker push "${remote_image}"
done
echo ""

# Print summary
echo "[3/3] Push complete!"
echo ""
echo "Images pushed to registry:"
for local_name in "${!IMAGES[@]}"; do
    remote_name="${IMAGES[$local_name]}"
    echo "  ${REGISTRY_PATH}/${remote_name}:${IMAGE_TAG}"
done
echo ""
echo "========================================"
echo "Kubernetes Usage Examples"
echo "========================================"
echo ""
echo "# If using insecure registry (HTTP), add to /etc/docker/daemon.json:"
echo "# {\"insecure-registries\": [\"${REGISTRY_URL}\"]}"
echo ""
echo "# For Kubernetes, create image pull secret:"
echo "kubectl create secret docker-registry regcred \\"
echo "  --docker-server=${REGISTRY_URL} \\"
echo "  --docker-username=<username> \\"
echo "  --docker-password=<password>"
echo ""
echo "# Update deployment with imagePullSecrets:"
echo "# spec:"
echo "#   imagePullSecrets:"
echo "#     - name: regcred"
echo "#   containers:"
echo "#     - image: ${REGISTRY_PATH}/mcp-fetch:${IMAGE_TAG}"
