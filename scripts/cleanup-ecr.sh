#!/bin/bash
# Cleanup EdgeAgent ECR repositories
#
# WARNING: This script deletes ECR repositories and all images within them.
#          Use with caution!
#
# Usage:
#   ./scripts/cleanup-ecr.sh              # Dry run (show what would be deleted)
#   ./scripts/cleanup-ecr.sh --force      # Actually delete repositories
#   ./scripts/cleanup-ecr.sh --force --all  # Delete ALL edgeagent-* repos

set -e

# Configuration
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
FORCE="${1:-}"
DELETE_ALL="${2:-}"

# Default repositories to clean up
REPOS=(
    "edgeagent-mcp-fetch"
    "edgeagent-mcp-log-parser"
    "edgeagent-mcp-summarize"
    "edgeagent-mcp-data-aggregate"
    "edgeagent-mcp-image-resize"
    "edgeagent-subagent"
)

echo "========================================"
echo "EdgeAgent ECR Cleanup Script"
echo "========================================"
echo "Region: ${AWS_REGION}"
echo ""

# If --all flag, find all edgeagent-* repositories
if [[ "${DELETE_ALL}" == "--all" ]]; then
    echo "Finding all edgeagent-* repositories..."
    REPOS=($(aws ecr describe-repositories \
        --region "${AWS_REGION}" \
        --query 'repositories[?starts_with(repositoryName, `edgeagent-`)].repositoryName' \
        --output text 2>/dev/null || echo ""))

    if [[ ${#REPOS[@]} -eq 0 || -z "${REPOS[0]}" ]]; then
        echo "No edgeagent-* repositories found."
        exit 0
    fi
fi

echo "Repositories to delete:"
for repo in "${REPOS[@]}"; do
    echo "  - ${repo}"
done
echo ""

# Check if dry run
if [[ "${FORCE}" != "--force" ]]; then
    echo "========================================"
    echo "DRY RUN - No changes will be made"
    echo "========================================"
    echo ""
    echo "The following repositories would be deleted:"
    for repo in "${REPOS[@]}"; do
        # Check if repo exists
        if aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" &>/dev/null; then
            # Get image count
            IMAGE_COUNT=$(aws ecr list-images \
                --repository-name "${repo}" \
                --region "${AWS_REGION}" \
                --query 'imageIds | length(@)' \
                --output text 2>/dev/null || echo "0")
            echo "  ${repo} (${IMAGE_COUNT} images)"
        else
            echo "  ${repo} (not found, will skip)"
        fi
    done
    echo ""
    echo "To actually delete, run:"
    echo "  $0 --force"
    echo ""
    echo "To delete ALL edgeagent-* repos:"
    echo "  $0 --force --all"
    exit 0
fi

# Confirmation for force delete
echo "WARNING: This will permanently delete the above repositories!"
read -p "Type 'yes' to confirm: " CONFIRM
if [[ "${CONFIRM}" != "yes" ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Deleting repositories..."

for repo in "${REPOS[@]}"; do
    echo ""
    echo "Processing: ${repo}"

    # Check if repo exists
    if ! aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" &>/dev/null; then
        echo "  Skipping (repository not found)"
        continue
    fi

    # Delete all images first (required before repo deletion)
    echo "  Deleting images..."
    IMAGE_IDS=$(aws ecr list-images \
        --repository-name "${repo}" \
        --region "${AWS_REGION}" \
        --query 'imageIds[*]' \
        --output json 2>/dev/null || echo "[]")

    if [[ "${IMAGE_IDS}" != "[]" && -n "${IMAGE_IDS}" ]]; then
        aws ecr batch-delete-image \
            --repository-name "${repo}" \
            --region "${AWS_REGION}" \
            --image-ids "${IMAGE_IDS}" \
            --output text > /dev/null
        echo "  Images deleted"
    else
        echo "  No images to delete"
    fi

    # Delete repository
    echo "  Deleting repository..."
    aws ecr delete-repository \
        --repository-name "${repo}" \
        --region "${AWS_REGION}" \
        --force \
        --output text > /dev/null
    echo "  Done"
done

echo ""
echo "========================================"
echo "Cleanup Complete!"
echo "========================================"
echo ""
echo "Deleted repositories:"
for repo in "${REPOS[@]}"; do
    echo "  - ${repo}"
done
