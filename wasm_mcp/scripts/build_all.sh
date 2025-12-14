#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building all WASM MCP Servers..."

SERVERS=("filesystem")

for server in "${SERVERS[@]}"; do
    if [ -d "servers/$server" ]; then
        echo "Building $server..."
        cargo build --target wasm32-wasip2 --release -p "mcp-server-$server"

        WASM_FILE="target/wasm32-wasip2/release/mcp_server_$server.wasm"
        if [ -f "$WASM_FILE" ]; then
            SIZE=$(du -h "$WASM_FILE" | cut -f1)
            echo "✓ $server build successful: $SIZE"
        else
            echo "✗ $server build failed"
            exit 1
        fi
    fi
done

echo ""
echo "✅ All servers built successfully"
echo "Output: target/wasm32-wasip2/release/"
