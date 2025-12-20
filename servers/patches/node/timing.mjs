/**
 * Timing utilities for Native MCP servers profiling (ESM version).
 *
 * Matches WASM timing format exactly:
 * - ---TIMING---{"tool":"name","fn_total_ms":X,"io_ms":Y,"compute_ms":Z}
 */

import fs from 'fs';
import { performance } from 'perf_hooks';

const TIMING_FILE = '/tmp/mcp_timing.json';

// I/O accumulator
let ioAccumulator = 0.0;

function resetIoAccumulator() {
    ioAccumulator = 0.0;
}

function addIoTime(ms) {
    ioAccumulator += ms;
}

function getIoAccumulator() {
    return ioAccumulator;
}

/**
 * Measure an async I/O operation and add to the accumulator.
 */
export async function measureIo(fn) {
    const start = performance.now();
    try {
        return await fn();
    } finally {
        const elapsedMs = performance.now() - start;
        addIoTime(elapsedMs);
    }
}

/**
 * Synchronous version of measureIo
 */
export function measureIoSync(fn) {
    const start = performance.now();
    try {
        return fn();
    } finally {
        const elapsedMs = performance.now() - start;
        addIoTime(elapsedMs);
    }
}

/**
 * Timer for measuring tool execution.
 */
export class ToolTimer {
    constructor(toolName) {
        this.toolName = toolName;
        this.startTime = performance.now();
        resetIoAccumulator();
    }

    finish() {
        const elapsed = performance.now() - this.startTime;
        const ioMs = getIoAccumulator();

        const fnTotalMs = elapsed;
        const computeMs = Math.max(0.0, fnTotalMs - ioMs);

        const timing = {
            tool: this.toolName,
            fn_total_ms: Math.round(fnTotalMs * 1000) / 1000,
            io_ms: Math.round(ioMs * 1000) / 1000,
            compute_ms: Math.round(computeMs * 1000) / 1000,
        };

        // Write to file
        try {
            fs.writeFileSync(TIMING_FILE, JSON.stringify(timing));
        } catch (e) {
            // Ignore file write errors
        }

        // Print to stderr
        console.error(`---TIMING---${JSON.stringify(timing)}`);

        return timing;
    }
}

/**
 * Wrapper for async tool handlers
 */
export function withTiming(toolName, handler) {
    return async (...args) => {
        const timer = new ToolTimer(toolName);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    };
}
