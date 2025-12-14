#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_FILE="${SCRIPT_DIR}/target/wasm32-wasip1/release/mcp-server-filesystem.wasm"
WASMEDGE="${WASMEDGE:-wasmedge}"

# Create a named pipe for communication
PIPE_IN=$(mktemp -u)
PIPE_OUT=$(mktemp -u)
mkfifo "$PIPE_IN"
mkfifo "$PIPE_OUT"

# Start wasmedge in background
$WASMEDGE "$WASM_FILE" < "$PIPE_IN" > "$PIPE_OUT" 2>&1 &
WASM_PID=$!

# Open pipes
exec 3>"$PIPE_IN"
exec 4<"$PIPE_OUT"

# Send initialize request
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' >&3

# Read response
read -t 5 response <&4
echo "Initialize response: $response"

# Send initialized notification
echo '{"jsonrpc":"2.0","method":"notifications/initialized"}' >&3

# Send tools/list request
echo '{"jsonrpc":"2.0","method":"tools/list","id":2}' >&3

# Read response
read -t 5 response <&4
echo "Tools list response: $response"

# Cleanup
exec 3>&-
exec 4<&-
kill $WASM_PID 2>/dev/null
rm -f "$PIPE_IN" "$PIPE_OUT"
