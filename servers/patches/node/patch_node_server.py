#!/usr/bin/env python3
"""
Patch Node.js MCP server to add ToolTimer profiling.

Usage: python patch_node_server.py <path_to_index.js>

Works with compiled JavaScript from @modelcontextprotocol/server-filesystem
"""

import sys
import re

# Timing code to inject after imports
TIMING_IMPORTS = '''
// ToolTimer imports - injected by patch_node_server.py
import { writeFileSync as _writeFileSync } from "fs";
import { performance as _performance } from "perf_hooks";
'''

TIMING_CLASS = '''
// ToolTimer class - injected by patch_node_server.py
const _TIMING_FILE = "/tmp/mcp_timing.json";
let _currentTimer = null;
class _ToolTimer {
    constructor(name) {
        this.name = name;
        this.start = _performance.now();
        this.ioTime = 0;
    }
    addIO(ms) { this.ioTime += ms; }
    finish() {
        const total = _performance.now() - this.start;
        const timing = {
            tool: this.name,
            fn_total_ms: Math.round(total * 1000) / 1000,
            io_ms: Math.round(this.ioTime * 1000) / 1000,
            compute_ms: Math.round((total - this.ioTime) * 1000) / 1000
        };
        const line = "---TIMING---" + JSON.stringify(timing);
        try { _writeFileSync(_TIMING_FILE, JSON.stringify(timing)); } catch(e) {}
        console.log(line);
        console.error(line);
        return timing;
    }
}
function _wrapHandler(name, handler) {
    return async (...args) => {
        _currentTimer = new _ToolTimer(name);
        try {
            const result = await handler(...args);
            return result;
        } finally {
            if (_currentTimer) _currentTimer.finish();
            _currentTimer = null;
        }
    };
}
// End ToolTimer
'''


def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if '_ToolTimer' in content:
        print(f"Already patched: {filepath}")
        return

    # Find the last import statement
    lines = content.split('\n')
    last_import_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match import statements (ESM)
        if stripped.startswith('import ') or stripped.startswith('import{'):
            last_import_idx = i

    if last_import_idx == -1:
        print(f"No imports found in {filepath}")
        return

    # Insert timing imports and class after all imports
    lines.insert(last_import_idx + 1, TIMING_IMPORTS + TIMING_CLASS)
    content = '\n'.join(lines)

    # Wrap registerTool handlers
    # Pattern: server.registerTool("name", {...}, handler)
    # We need to wrap the handler with _wrapHandler

    # Match: server.registerTool( "tool_name" , { ... } , <handler> )
    # The handler can be: async (args) => {...} or a variable name

    # Simpler approach: wrap all async arrow functions that are third argument to registerTool
    # Pattern: registerTool("name", config, async (

    def wrap_handler(match):
        before = match.group(1)  # registerTool("name",
        tool_name = match.group(2)  # tool name
        middle = match.group(3)  # config object and comma
        return f'{before}"{tool_name}",{middle}_wrapHandler("{tool_name}", async ('

    # Match registerTool("tool_name", {...}, async (
    pattern = r'(registerTool\s*\(\s*)"([^"]+)"(\s*,\s*\{[^}]*\}\s*,\s*)async\s*\('
    content = re.sub(pattern, wrap_handler, content)

    # Handle named handlers like: registerTool("name", {...}, handlerName)
    # Wrap the variable reference: handlerName -> _wrapHandler("name", handlerName)
    def wrap_var_handler(match):
        before = match.group(1)
        tool_name = match.group(2)
        middle = match.group(3)
        handler_name = match.group(4)
        return f'{before}"{tool_name}",{middle}_wrapHandler("{tool_name}", {handler_name})'

    # Pattern: registerTool("name", {...}, handlerVariableName)
    # handlerVariableName is a word that's not 'async'
    pattern2 = r'(registerTool\s*\(\s*)"([^"]+)"(\s*,\s*\{[^}]*\}\s*,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
    content = re.sub(pattern2, lambda m: wrap_var_handler(m) + ')', content)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Patched: {filepath}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_node_server.py <path_to_index.js>")
        sys.exit(1)

    patch_file(sys.argv[1])
