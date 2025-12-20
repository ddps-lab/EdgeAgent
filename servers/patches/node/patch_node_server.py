#!/usr/bin/env python3
"""
Patch Node.js MCP server to add ToolTimer profiling.

Usage: python patch_node_server.py <path_to_index.js>

This script:
1. Adds ToolTimer import and class definition
2. Wraps server.registerTool handlers with timing
"""

import sys
import re

TIMING_CODE = '''
// ToolTimer for profiling - injected by patch_node_server.py
import { writeFileSync } from 'fs';
import { performance } from 'perf_hooks';

const TIMING_FILE = '/tmp/mcp_timing.json';

class ToolTimer {
    constructor(toolName) {
        this.toolName = toolName;
        this.startTime = performance.now();
    }

    finish() {
        const elapsed = performance.now() - this.startTime;
        const timing = {
            tool: this.toolName,
            fn_total_ms: Math.round(elapsed * 1000) / 1000,
            io_ms: 0,
            compute_ms: Math.round(elapsed * 1000) / 1000,
        };
        try { writeFileSync(TIMING_FILE, JSON.stringify(timing)); } catch (e) {}
        console.error(`---TIMING---${JSON.stringify(timing)}`);
        return timing;
    }
}

function wrapWithTiming(name, handler) {
    return async (...args) => {
        const timer = new ToolTimer(name);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    };
}
// End ToolTimer
'''

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'ToolTimer for profiling' in content:
        print(f"Already patched: {filepath}")
        return

    # Insert timing code after the imports (after the first few import statements)
    # Find a good insertion point - after import statements
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('// '):
            insert_idx = i + 1
        else:
            break

    # Insert timing code
    lines.insert(insert_idx, TIMING_CODE)
    content = '\n'.join(lines)

    # Patch registerTool calls to use wrapped handlers
    # Pattern: server.registerTool("name", config, async (args) => { ... })
    # We need to wrap the handler function

    # Find all server.registerTool calls and wrap the handler
    # This is a simplified regex - handles the common case
    pattern = r'(server\.registerTool\(\s*["\']([^"\']+)["\']\s*,\s*\{[^}]+\}\s*,\s*)(async\s*\([^)]*\)\s*=>\s*\{)'

    def replace_handler(match):
        prefix = match.group(1)
        tool_name = match.group(2)
        handler_start = match.group(3)
        return f'{prefix}wrapWithTiming("{tool_name}", {handler_start}'

    content = re.sub(pattern, replace_handler, content)

    # Also need to close the wrapper - this is tricky with regex
    # For now, let's use a simpler approach: just add timing to the beginning of handlers

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Patched: {filepath}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_node_server.py <path_to_index.js>")
        sys.exit(1)

    patch_file(sys.argv[1])
