#!/bin/bash
# Setup Knative configuration for Edge (K3s) cluster
# This script configures Knative Serving to support:
# - WASM runtimeClassName (wasmtime)
# - PersistentVolumeClaim volumes
# - Node selector for scheduling
# - Extended progress deadline for slower edge nodes
#
# Usage:
#   KUBECONFIG=~/.kube/k3s ./scripts/setup-knative-edge.sh

set -e

# Check if KUBECONFIG is set for edge cluster
if [[ -z "${KUBECONFIG}" ]]; then
    echo "Warning: KUBECONFIG not set. Using default kubeconfig."
    echo "For edge cluster, run: KUBECONFIG=~/.kube/k3s $0"
fi

echo "=== Setting up Knative for Edge cluster ==="

# 1. Enable Knative feature flags for WASM and PVC support
echo "[1/3] Enabling Knative feature flags..."
kubectl patch configmap config-features -n knative-serving --type merge -p '{
  "data": {
    "kubernetes.podspec-runtimeclassname": "enabled",
    "kubernetes.podspec-persistent-volume-claim": "enabled",
    "kubernetes.podspec-persistent-volume-write": "enabled",
    "kubernetes.podspec-nodeselector": "enabled"
  }
}'

# 2. Configure deployment settings (optional: extend progress-deadline for slow cold starts)
echo "[2/3] Configuring deployment settings..."
kubectl patch configmap config-deployment -n knative-serving --type merge -p '{
  "data": {
    "progress-deadline": "600s"
  }
}'

# 3. Verify RuntimeClass exists for wasmtime
echo "[3/3] Checking RuntimeClass for wasmtime..."
if kubectl get runtimeclass wasmtime &>/dev/null; then
    echo "RuntimeClass 'wasmtime' exists."
else
    echo "Creating RuntimeClass 'wasmtime'..."
    kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: wasmtime
handler: wasmtime
EOF
fi

echo ""
echo "=== Knative Edge configuration complete ==="
echo ""
echo "Enabled features:"
echo "  - kubernetes.podspec-runtimeclassname: enabled (for WASM)"
echo "  - kubernetes.podspec-persistent-volume-claim: enabled"
echo "  - kubernetes.podspec-persistent-volume-write: enabled"
echo "  - kubernetes.podspec-nodeselector: enabled"
echo "  - progress-deadline: 600s"
echo ""
echo "You can now deploy WASM services with 'runtimeClassName: wasmtime' in the pod spec."
