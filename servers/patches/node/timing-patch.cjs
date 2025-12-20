/**
 * CommonJS module that patches McpServer at load time.
 * Use with: node --require ./timing-patch.cjs <your-script>
 *
 * This patches the ES module's McpServer class to wrap tool handlers with timing.
 */

const fs = require('fs');
const { performance } = require('perf_hooks');

const TIMING_FILE = '/tmp/mcp_timing.json';

// ToolTimer class
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

        try {
            fs.writeFileSync(TIMING_FILE, JSON.stringify(timing));
        } catch (e) {}

        console.error(`---TIMING---${JSON.stringify(timing)}`);
        return timing;
    }
}

// Make ToolTimer available globally for the patched modules
global.__ToolTimer = ToolTimer;

console.error('[TIMING-PATCH] Loaded. ToolTimer available as global.__ToolTimer');
