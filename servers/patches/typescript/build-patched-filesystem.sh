#!/bin/bash
# Build patched filesystem server with ToolTimer instrumentation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/mcp-filesystem-patched"

echo "=== Building patched MCP filesystem server ==="

# Clone or update source
if [ ! -d "/tmp/mcp_servers" ]; then
    echo "Cloning MCP servers..."
    git clone --depth 1 https://github.com/modelcontextprotocol/servers.git /tmp/mcp_servers
fi

# Create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy filesystem server source
cp -r /tmp/mcp_servers/src/filesystem/* "$BUILD_DIR/"

# Copy timing instrumentation
cp "$SCRIPT_DIR/timing.ts" "$BUILD_DIR/"

# Replace lib.ts with patched version
cp "$SCRIPT_DIR/lib.ts" "$BUILD_DIR/"

# Patch index.ts - add import and monkey-patch after existing imports
echo "Patching index.ts..."

# Read original index.ts and find the last import line
PATCH_IMPORT='import { ToolTimer } from "./timing.js";'
PATCH_CODE='
// ===== ToolTimer Monkey-Patch (added by build script) =====
const _origRegisterTool = McpServer.prototype.registerTool;
(McpServer.prototype as any).registerTool = function(name: string, config: any, handler: any) {
    const wrappedHandler = async (...args: any[]) => {
        const timer = new ToolTimer(name);
        try { return await handler(...args); }
        finally { timer.finish(); }
    };
    return _origRegisterTool.call(this, name, config, wrappedHandler);
};
// ===== End ToolTimer Monkey-Patch =====
'

# Create patched index.ts
{
    # Add timing import after other imports
    head -30 "$BUILD_DIR/index.ts"
    echo "$PATCH_IMPORT"
    echo "$PATCH_CODE"
    tail -n +31 "$BUILD_DIR/index.ts"
} > "$BUILD_DIR/index.ts.patched"
mv "$BUILD_DIR/index.ts.patched" "$BUILD_DIR/index.ts"

# Update package.json
cat > "$BUILD_DIR/package.json" << 'EOF'
{
  "name": "mcp-server-filesystem-patched",
  "version": "0.2.0-patched",
  "type": "module",
  "main": "dist/index.js",
  "bin": {
    "mcp-server-filesystem": "dist/index.js"
  },
  "scripts": {
    "build": "tsc"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "diff": "^5.0.0",
    "minimatch": "^9.0.0",
    "zod": "^3.0.0"
  },
  "devDependencies": {
    "@types/diff": "^5.0.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  }
}
EOF

# Create tsconfig.json
cat > "$BUILD_DIR/tsconfig.json" << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "esModuleInterop": true,
    "strict": true,
    "outDir": "./dist",
    "rootDir": ".",
    "declaration": true,
    "skipLibCheck": true
  },
  "include": ["*.ts"],
  "exclude": ["node_modules", "dist"]
}
EOF

echo "Build directory prepared: $BUILD_DIR"
echo ""
echo "To build, run:"
echo "  cd $BUILD_DIR && npm install && npm run build"
