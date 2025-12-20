#!/bin/bash
# Build and Push EdgeAgent images to AWS ECR
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed
#
# Usage:
#   ./scripts/push-to-ecr.sh                    # Use default region/account
#   ./scripts/push-to-ecr.sh us-west-2 123456789012  # Specify region and account
#   AWS_REGION=ap-northeast-2 ./scripts/push-to-ecr.sh  # Use env var
#   SKIP_BUILD=1 ./scripts/push-to-ecr.sh      # Skip build, only push

set -e

# Configuration
AWS_REGION="${1:-${AWS_REGION:-us-west-2}}"
AWS_ACCOUNT_ID="${2:-${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SKIP_BUILD="${SKIP_BUILD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Image names
# Note: edge-subagent and cloud-subagent use the same image,
# differentiated by LOCATION env var at runtime
IMAGES=(
    # Custom FastMCP servers
    "edgeagent-mcp-fetch"
    "edgeagent-mcp-log-parser"
    "edgeagent-mcp-summarize"
    "edgeagent-mcp-data-aggregate"
    "edgeagent-mcp-image-resize"
    # Official MCP servers (via mcp-proxy)
    "edgeagent-mcp-filesystem"
    "edgeagent-mcp-time"
    "edgeagent-mcp-sequentialthinking"
    "edgeagent-mcp-git"
    # SubAgent
    "edgeagent-subagent"  # Single image, LOCATION=EDGE|CLOUD at runtime
)

echo "========================================"
echo "EdgeAgent ECR Build & Push Script"
echo "========================================"
echo "Region:   ${AWS_REGION}"
echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Registry: ${ECR_REGISTRY}"
echo "Tag:      ${IMAGE_TAG}"
echo "Skip Build: ${SKIP_BUILD}"
echo "========================================"
echo ""

# Build images
if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[1/5] Building Docker images..."

    # Build base image first
    echo "  Building base image..."
    docker build -t edgeagent-mcp-base:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.base" \
        "${PROJECT_ROOT}"

    # Build custom FastMCP servers
    echo "  Building custom FastMCP servers..."
    docker build -t edgeagent-mcp-fetch:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.fetch" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-log-parser:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.log-parser" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-summarize:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.summarize" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-data-aggregate:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.data-aggregate" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-image-resize:latest \
        -f "${PROJECT_ROOT}/docker/mcp-servers/Dockerfile.image-resize" \
        "${PROJECT_ROOT}"

    # Build official MCP servers (via mcp-proxy)
    echo "  Building official MCP servers..."
    docker build -t edgeagent-mcp-filesystem:latest \
        -f "${PROJECT_ROOT}/docker/mcp-official/Dockerfile.filesystem" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-time:latest \
        -f "${PROJECT_ROOT}/docker/mcp-official/Dockerfile.time" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-sequentialthinking:latest \
        -f "${PROJECT_ROOT}/docker/mcp-official/Dockerfile.sequentialthinking" \
        "${PROJECT_ROOT}"
    docker build -t edgeagent-mcp-git:latest \
        -f "${PROJECT_ROOT}/docker/mcp-official/Dockerfile.git" \
        "${PROJECT_ROOT}"

    # Build subagent
    echo "  Building subagent..."
    docker build -t edgeagent-subagent:latest \
        -f "${PROJECT_ROOT}/docker/subagent/Dockerfile" \
        "${PROJECT_ROOT}"

    echo "  All images built successfully!"
    echo ""
else
    echo "[1/5] Skipping build (SKIP_BUILD=1)"
    echo ""
fi

# Login to ECR
echo "[2/5] Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"
echo ""

# Create repositories if they don't exist
echo "[3/5] Creating ECR repositories..."
for image in "${IMAGES[@]}"; do
    repo_name="${image}"
    if aws ecr describe-repositories --repository-names "${repo_name}" --region "${AWS_REGION}" 2>/dev/null; then
        echo "  Repository '${repo_name}' already exists"
    else
        echo "  Creating repository '${repo_name}'..."
        aws ecr create-repository \
            --repository-name "${repo_name}" \
            --region "${AWS_REGION}" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
    fi
done
echo ""

# Tag and push images
echo "[4/5] Tagging and pushing images..."
for image in "${IMAGES[@]}"; do
    local_image="${image}:latest"
    ecr_image="${ECR_REGISTRY}/${image}:${IMAGE_TAG}"

    echo "  ${local_image} -> ${ecr_image}"
    docker tag "${local_image}" "${ecr_image}"
    docker push "${ecr_image}"
done
echo ""

# Print summary
echo "[5/5] Push complete!"
echo ""
echo "Images pushed to ECR:"
for image in "${IMAGES[@]}"; do
    echo "  ${ECR_REGISTRY}/${image}:${IMAGE_TAG}"
done
echo ""
echo "To use in Kubernetes, update your deployment YAML:"
echo "  image: ${ECR_REGISTRY}/<image-name>:${IMAGE_TAG}"
