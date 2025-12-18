#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building all WASM MCP Servers..."

# All servers now use unified CLI/HTTP builds (feature flags)
UNIFIED_SERVERS=(
    "time"
    "sequential-thinking"
    "data-aggregate"
    "log-parser"
    "fetch"
    "summarize"
    "git"
    "image-resize"
    "filesystem"
)

# Build all servers (both CLI and HTTP from same crate)
for server in "${UNIFIED_SERVERS[@]}"; do
    if [ -d "servers/$server" ]; then
        # Convert kebab-case to snake_case for crate name
        crate_name=$(echo "$server" | tr '-' '_')

        echo "Building $server (CLI)..."
        cargo build --target wasm32-wasip2 --release -p "mcp-server-$server" --features cli-export --no-default-features

        WASM_FILE="target/wasm32-wasip2/release/mcp_server_$crate_name.wasm"
        if [ -f "$WASM_FILE" ]; then
            # Copy to CLI-specific name
            cp "$WASM_FILE" "target/wasm32-wasip2/release/mcp_server_${crate_name}_cli.wasm"
            SIZE=$(du -h "$WASM_FILE" | cut -f1)
            echo "✓ $server CLI build successful: $SIZE"
        else
            echo "✗ $server CLI build failed"
            exit 1
        fi

        echo "Building $server (HTTP)..."
        cargo build --target wasm32-wasip2 --release -p "mcp-server-$server" --features http-export --no-default-features

        if [ -f "$WASM_FILE" ]; then
            # Copy to HTTP-specific name
            cp "$WASM_FILE" "target/wasm32-wasip2/release/mcp_server_${crate_name}_http.wasm"
            SIZE=$(du -h "$WASM_FILE" | cut -f1)
            echo "✓ $server HTTP build successful: $SIZE"
        else
            echo "✗ $server HTTP build failed"
            exit 1
        fi
    fi
done

echo ""
echo "✅ All servers built successfully"
echo "Output: target/wasm32-wasip2/release/"
echo ""
echo "Built servers (CLI + HTTP):"
for server in "${UNIFIED_SERVERS[@]}"; do
    crate_name=$(echo "$server" | tr '-' '_')
    echo "  - mcp_server_${crate_name}_cli.wasm  (wasmtime run)"
    echo "  - mcp_server_${crate_name}_http.wasm (wasmtime serve)"
done
