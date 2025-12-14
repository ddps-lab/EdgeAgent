#!/bin/bash
# Push EdgeAgent images to AWS ECR
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker logged in to ECR
#
# Usage:
#   ./scripts/push-to-ecr.sh                    # Use default region/account
#   ./scripts/push-to-ecr.sh us-west-2 123456789012  # Specify region and account
#   AWS_REGION=ap-northeast-2 ./scripts/push-to-ecr.sh  # Use env var

set -e

# Configuration
AWS_REGION="${1:-${AWS_REGION:-ap-northeast-2}}"
AWS_ACCOUNT_ID="${2:-${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${IMAGE_TAG:-latest}"

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
echo "EdgeAgent ECR Push Script"
echo "========================================"
echo "Region:   ${AWS_REGION}"
echo "Account:  ${AWS_ACCOUNT_ID}"
echo "Registry: ${ECR_REGISTRY}"
echo "Tag:      ${IMAGE_TAG}"
echo "========================================"
echo ""

# Login to ECR
echo "[1/4] Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"
echo ""

# Create repositories if they don't exist
echo "[2/4] Creating ECR repositories..."
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
echo "[3/4] Tagging and pushing images..."
for image in "${IMAGES[@]}"; do
    local_image="${image}:latest"
    ecr_image="${ECR_REGISTRY}/${image}:${IMAGE_TAG}"

    echo "  ${local_image} -> ${ecr_image}"
    docker tag "${local_image}" "${ecr_image}"
    docker push "${ecr_image}"
done
echo ""

# Print summary
echo "[4/4] Push complete!"
echo ""
echo "Images pushed to ECR:"
for image in "${IMAGES[@]}"; do
    echo "  ${ECR_REGISTRY}/${image}:${IMAGE_TAG}"
done
echo ""
echo "To use in Kubernetes, update your deployment YAML:"
echo "  image: ${ECR_REGISTRY}/<image-name>:${IMAGE_TAG}"
