/**
 * Patch code to add at the TOP of index.ts (after imports)
 *
 * This monkey-patches McpServer.registerTool to automatically wrap
 * all tool handlers with ToolTimer for profiling.
 */

import { ToolTimer } from './timing.js';

// Monkey-patch McpServer.prototype.registerTool
const originalRegisterTool = McpServer.prototype.registerTool;
McpServer.prototype.registerTool = function(name: string, config: any, handler: any) {
    const wrappedHandler = async (...args: any[]) => {
        const timer = new ToolTimer(name);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    };
    return originalRegisterTool.call(this, name, config, wrappedHandler);
};
