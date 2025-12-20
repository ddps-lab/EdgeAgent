#!/usr/bin/env node
/**
 * Wrapper for @modelcontextprotocol/server-filesystem with ToolTimer.
 *
 * This script patches McpServer before importing the filesystem server.
 */

import { performance } from 'perf_hooks';
import fs from 'fs';

const TIMING_FILE = '/tmp/mcp_timing.json';

// ToolTimer class
class ToolTimer {
    constructor(toolName) {
        this.toolName = toolName;
        this.startTime = performance.now();
    }

    finish() {
        const elapsed = performance.now() - this.startTime;
        const fnTotalMs = elapsed;
        const computeMs = fnTotalMs; // Simplified - no I/O tracking

        const timing = {
            tool: this.toolName,
            fn_total_ms: Math.round(fnTotalMs * 1000) / 1000,
            io_ms: 0,
            compute_ms: Math.round(computeMs * 1000) / 1000,
        };

        try {
            fs.writeFileSync(TIMING_FILE, JSON.stringify(timing));
        } catch (e) {}

        console.error(`---TIMING---${JSON.stringify(timing)}`);
        return timing;
    }
}

// Patch McpServer BEFORE importing the filesystem server
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

const originalRegisterTool = McpServer.prototype.registerTool;
McpServer.prototype.registerTool = function(name, config, handler) {
    const wrappedHandler = async (...args) => {
        const timer = new ToolTimer(name);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    };
    return originalRegisterTool.call(this, name, config, wrappedHandler);
};

console.error('[PATCH] Filesystem server patched with ToolTimer');

// Now import and run the actual server
// The filesystem server is typically run via 'mcp-server-filesystem' command
// We need to import its main module

// Since we can't easily import the CLI entry point, let's just run the original server
// after patching is done. The patch will be applied when McpServer is used.

// Re-export for CLI usage or manually start the server here
import('@modelcontextprotocol/server-filesystem');
