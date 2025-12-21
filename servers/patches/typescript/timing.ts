/**
 * ToolTimer for Node.js MCP servers profiling.
 *
 * Matches Python timing format:
 * ---TIMING---{"tool":"name","fn_total_ms":X,"io_ms":Y,"compute_ms":Z}
 */

import { writeFileSync } from 'fs';
import { performance } from 'perf_hooks';

const TIMING_FILE = '/tmp/mcp_timing.json';

// Thread-local equivalent for async context
let currentIOTime = 0;

export function resetIOTime(): void {
    currentIOTime = 0;
}

export function addIOTime(ms: number): void {
    currentIOTime += ms;
}

export function getIOTime(): number {
    return currentIOTime;
}

/**
 * Measure an async I/O operation and accumulate the time.
 */
export async function measureIO<T>(fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
        return await fn();
    } finally {
        const elapsed = performance.now() - start;
        addIOTime(elapsed);
    }
}

/**
 * Measure a sync I/O operation and accumulate the time.
 */
export function measureIOSync<T>(fn: () => T): T {
    const start = performance.now();
    try {
        return fn();
    } finally {
        const elapsed = performance.now() - start;
        addIOTime(elapsed);
    }
}

export class ToolTimer {
    private toolName: string;
    private startTime: number;

    constructor(toolName: string) {
        this.toolName = toolName;
        this.startTime = performance.now();
        resetIOTime();
    }

    finish(): { tool: string; fn_total_ms: number; io_ms: number; compute_ms: number } {
        const elapsed = performance.now() - this.startTime;
        const ioMs = getIOTime();

        const timing = {
            tool: this.toolName,
            fn_total_ms: Math.round(elapsed * 1000) / 1000,
            io_ms: Math.round(ioMs * 1000) / 1000,
            compute_ms: Math.round((elapsed - ioMs) * 1000) / 1000,
        };

        const timingLine = `---TIMING---${JSON.stringify(timing)}`;

        // Write to file
        try {
            writeFileSync(TIMING_FILE, JSON.stringify(timing));
        } catch (e) {
            // Ignore file write errors
        }

        // Output to both stdout and stderr for Docker log capture
        console.log(timingLine);
        console.error(timingLine);

        return timing;
    }
}

/**
 * Wrap an async handler with timing.
 */
export function withTiming<T extends (...args: any[]) => Promise<any>>(
    toolName: string,
    handler: T
): T {
    return (async (...args: Parameters<T>): Promise<ReturnType<T>> => {
        const timer = new ToolTimer(toolName);
        try {
            return await handler(...args);
        } finally {
            timer.finish();
        }
    }) as T;
}
