#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./scripts/run_server.sh <server_name> [directory]"
    echo "Example: ./scripts/run_server.sh filesystem /tmp/test"
    exit 1
fi

SERVER=$1
DIR="${2:-/tmp/wasm_test}"

# Create directory if it doesn't exist
mkdir -p "$DIR"

WASM_FILE="target/wasm32-wasip2/release/mcp_server_$SERVER.wasm"

if [ ! -f "$WASM_FILE" ]; then
    echo "WASM file not found. Run ./scripts/build_server.sh $SERVER first"
    exit 1
fi

# Use wasmtime if available, otherwise try wasmedge
if command -v wasmtime &> /dev/null; then
    RUNTIME="wasmtime run"
elif [ -f "$HOME/.wasmtime/bin/wasmtime" ]; then
    RUNTIME="$HOME/.wasmtime/bin/wasmtime run"
else
    echo "Wasmtime not found. Please install wasmtime."
    exit 1
fi

echo "Running $SERVER MCP Server with directory: $DIR"
echo "Runtime: $RUNTIME"
$RUNTIME --dir="$DIR" "$WASM_FILE"
