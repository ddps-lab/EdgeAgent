#!/bin/bash
# Setup NFS Server for EdgeAgent (EDGE environment)
#
# Run this on the NFS server node (can be a K8s node or dedicated server)
#
# Usage:
#   sudo ./setup-nfs-server.sh
#   sudo ./setup-nfs-server.sh 192.168.1.0/24  # Custom allowed network

set -e

# Configuration
ALLOWED_NETWORK="${1:-*}"  # Default: allow all, or specify subnet like 192.168.1.0/24
EDGEAGENT_ROOT="/edgeagent"

echo "========================================"
echo "EdgeAgent NFS Server Setup"
echo "========================================"
echo "Root directory: ${EDGEAGENT_ROOT}"
echo "Allowed network: ${ALLOWED_NETWORK}"
echo "========================================"
echo ""

# Step 1: Install NFS server
echo "[1/5] Installing NFS server..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nfs-kernel-server
elif command -v yum &> /dev/null; then
    sudo yum install -y nfs-utils
    sudo systemctl enable nfs-server
fi

# Step 2: Create directory structure
echo ""
echo "[2/5] Creating directory structure..."
sudo mkdir -p "${EDGEAGENT_ROOT}/data/logs"
sudo mkdir -p "${EDGEAGENT_ROOT}/data/images"
sudo mkdir -p "${EDGEAGENT_ROOT}/data/documents"
sudo mkdir -p "${EDGEAGENT_ROOT}/data/temp"
sudo mkdir -p "${EDGEAGENT_ROOT}/repos"
sudo mkdir -p "${EDGEAGENT_ROOT}/results/summaries"
sudo mkdir -p "${EDGEAGENT_ROOT}/results/parsed"
sudo mkdir -p "${EDGEAGENT_ROOT}/results/aggregated"

# Set permissions (allow read/write for all - adjust as needed)
sudo chown -R nobody:nogroup "${EDGEAGENT_ROOT}"
sudo chmod -R 755 "${EDGEAGENT_ROOT}"

echo "Directory structure created:"
tree "${EDGEAGENT_ROOT}" 2>/dev/null || find "${EDGEAGENT_ROOT}" -type d

# Step 3: Configure NFS exports
echo ""
echo "[3/5] Configuring NFS exports..."

# Backup existing exports
if [[ -f /etc/exports ]]; then
    sudo cp /etc/exports /etc/exports.bak.$(date +%Y%m%d)
fi

# Add EdgeAgent export (if not already exists)
if ! grep -q "${EDGEAGENT_ROOT}" /etc/exports 2>/dev/null; then
    echo "${EDGEAGENT_ROOT} ${ALLOWED_NETWORK}(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
fi

echo "NFS exports:"
cat /etc/exports

# Step 4: Export and restart NFS
echo ""
echo "[4/5] Exporting filesystems and restarting NFS..."
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server 2>/dev/null || sudo systemctl restart nfs-server

# Step 5: Verify
echo ""
echo "[5/5] Verifying NFS setup..."
sudo exportfs -v

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "NFS Server IP: ${SERVER_IP}"
echo "Export path:   ${EDGEAGENT_ROOT}"
echo ""
echo "To mount on K8s nodes (install nfs-common first):"
echo "  sudo apt-get install -y nfs-common"
echo "  sudo mount -t nfs ${SERVER_IP}:${EDGEAGENT_ROOT} /mnt/edgeagent"
echo ""
echo "Update k8s/storage/nfs-edge.yaml with:"
echo "  server: ${SERVER_IP}"
echo "  path: ${EDGEAGENT_ROOT}"
echo ""
echo "Apply to K8s:"
echo "  kubectl apply -f k8s/storage/nfs-edge.yaml"
