/**
 * Monkey-patch for @modelcontextprotocol/sdk McpServer to add timing.
 *
 * This patches the registerTool method to wrap all tool handlers with ToolTimer.
 * Import this BEFORE importing any MCP servers.
 */

import { ToolTimer } from './timing.mjs';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

// Store original registerTool method
const originalRegisterTool = McpServer.prototype.registerTool;

// Patch registerTool to wrap handlers with timing
McpServer.prototype.registerTool = function(name, config, handler) {
    // Wrap the handler with timing
    const wrappedHandler = async (...args) => {
        const timer = new ToolTimer(name);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    };

    // Call original with wrapped handler
    return originalRegisterTool.call(this, name, config, wrappedHandler);
};

console.error('[PATCH] McpServer.registerTool patched with ToolTimer');

export { McpServer };
