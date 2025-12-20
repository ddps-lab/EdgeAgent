#!/usr/bin/env python3
"""
Patch Node.js MCP server to add ToolTimer profiling.
Uses monkey-patching approach for reliable handler wrapping.
"""

import sys

PATCH_CODE = '''
// ===== ToolTimer Patch (injected by patch_node_server.py) =====
import { writeFileSync as _writeFileSync } from "fs";
import { performance as _performance } from "perf_hooks";

const _TIMING_FILE = "/tmp/mcp_timing.json";
class _ToolTimer {
    constructor(name) { this.name = name; this.start = _performance.now(); this.ioTime = 0; }
    addIO(ms) { this.ioTime += ms; }
    finish() {
        const total = _performance.now() - this.start;
        const timing = { tool: this.name, fn_total_ms: Math.round(total*1000)/1000, io_ms: Math.round(this.ioTime*1000)/1000, compute_ms: Math.round((total-this.ioTime)*1000)/1000 };
        const line = "---TIMING---" + JSON.stringify(timing);
        try { _writeFileSync(_TIMING_FILE, JSON.stringify(timing)); } catch(e) {}
        console.log(line); console.error(line);
    }
}

// Monkey-patch McpServer.prototype.registerTool to wrap all handlers
const _origRegisterTool = McpServer.prototype.registerTool;
McpServer.prototype.registerTool = function(name, config, handler) {
    const wrappedHandler = async (...args) => {
        const timer = new _ToolTimer(name);
        try { return await handler(...args); }
        finally { timer.finish(); }
    };
    return _origRegisterTool.call(this, name, config, wrappedHandler);
};
// ===== End ToolTimer Patch =====
'''

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if '_ToolTimer' in content:
        print(f"Already patched: {filepath}")
        return

    # Find position after the McpServer import
    # Insert patch code right after imports
    lines = content.split('\n')

    # Find last import line
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import '):
            insert_idx = i + 1

    if insert_idx == 0:
        print(f"No imports found: {filepath}")
        return

    # Insert patch after imports
    lines.insert(insert_idx, PATCH_CODE)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Patched: {filepath}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_node_server.py <path_to_index.js>")
        sys.exit(1)
    patch_file(sys.argv[1])
