#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/build_server.sh <server_name>"
    echo "Available: filesystem"
    exit 1
fi

SERVER=$1
cd "$(dirname "$0")/.."

echo "Building $SERVER server..."

cargo build --target wasm32-wasip2 --release -p "mcp-server-$SERVER"

WASM_FILE="target/wasm32-wasip2/release/mcp_server_$SERVER.wasm"

if [ -f "$WASM_FILE" ]; then
    SIZE=$(du -h "$WASM_FILE" | cut -f1)
    echo "✓ Build successful: $WASM_FILE ($SIZE)"
else
    echo "✗ Build failed"
    exit 1
fi
