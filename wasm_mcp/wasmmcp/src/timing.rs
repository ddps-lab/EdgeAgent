//! Timing utilities for WASM MCP tool profiling
//!
//! Provides timing measurement for:
//! - Cold start time (WASM loading + runtime init)
//! - Tool total execution time (fn_total)
//! - Disk I/O time (filesystem operations)
//! - Network I/O time (HTTP requests)
//! - Compute time (fn_total - disk_io - network_io)

use std::cell::RefCell;
use std::time::{Duration, Instant};

// Thread-local storage for I/O timing accumulators
thread_local! {
    static DISK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    static NETWORK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    // For HTTP transport to read timing values set by handle_tools_call
    static TOOL_EXEC_MS: RefCell<f64> = RefCell::new(0.0);
    static DISK_IO_MS: RefCell<f64> = RefCell::new(0.0);
    static NETWORK_IO_MS: RefCell<f64> = RefCell::new(0.0);
}

/// Reset both I/O accumulators (call before each tool execution)
pub fn reset_io_accumulators() {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
}

/// Add duration to disk I/O accumulator
pub fn add_disk_io_duration(duration: Duration) {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Add duration to network I/O accumulator
pub fn add_network_io_duration(duration: Duration) {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Get accumulated disk I/O duration
pub fn get_disk_io_duration() -> Duration {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Get accumulated network I/O duration
pub fn get_network_io_duration() -> Duration {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Measure a disk I/O operation (filesystem read/write)
pub fn measure_disk_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    add_disk_io_duration(start.elapsed());
    result
}

/// Measure a network I/O operation (HTTP requests)
pub fn measure_network_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    add_network_io_duration(start.elapsed());
    result
}

/// Set tool execution time (for HTTP transport to read)
pub fn set_tool_exec_ms(ms: f64) {
    TOOL_EXEC_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get tool execution time
pub fn get_tool_exec_ms() -> f64 {
    TOOL_EXEC_MS.with(|v| *v.borrow())
}

/// Set disk I/O time (for HTTP transport to read)
pub fn set_disk_io_ms(ms: f64) {
    DISK_IO_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get disk I/O time in ms
pub fn get_disk_io_ms() -> f64 {
    DISK_IO_MS.with(|v| *v.borrow())
}

/// Set network I/O time (for HTTP transport to read)
pub fn set_network_io_ms(ms: f64) {
    NETWORK_IO_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get network I/O time in ms
pub fn get_network_io_ms() -> f64 {
    NETWORK_IO_MS.with(|v| *v.borrow())
}

// Backward compatibility aliases
/// Reset I/O accumulator (backward compat - resets both accumulators)
pub fn reset_io_accumulator() {
    reset_io_accumulators();
}

/// Measure I/O operation (backward compat - measures as disk I/O)
pub fn measure_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    measure_disk_io(f)
}

/// Get I/O duration (backward compat - returns disk I/O only)
pub fn get_io_duration() -> Duration {
    get_disk_io_duration()
}

/// Tool timing information
#[derive(Debug, Clone)]
pub struct ToolTiming {
    pub tool_name: String,
    pub fn_total_ms: f64,
    pub disk_io_ms: f64,
    pub network_io_ms: f64,
    pub compute_ms: f64,
}

impl ToolTiming {
    /// Output timing information to stderr in the expected format
    pub fn output(&self) {
        let json = serde_json::json!({
            "tool": self.tool_name,
            "fn_total_ms": self.fn_total_ms,
            "disk_io_ms": self.disk_io_ms,
            "network_io_ms": self.network_io_ms,
            "compute_ms": self.compute_ms,
        });
        eprintln!("---TIMING---{}", json);
    }

    /// Get total I/O time (disk + network) for backward compatibility
    pub fn io_ms(&self) -> f64 {
        self.disk_io_ms + self.network_io_ms
    }
}

/// Timer for measuring tool execution
pub struct ToolTimer {
    start: Instant,
}

impl ToolTimer {
    /// Start a new timer (also resets both I/O accumulators)
    pub fn start() -> Self {
        reset_io_accumulators();
        Self {
            start: Instant::now(),
        }
    }

    /// Finish timing and output the results
    pub fn finish(self, tool_name: &str) -> ToolTiming {
        let elapsed = self.start.elapsed();
        let disk_io = get_disk_io_duration();
        let network_io = get_network_io_duration();

        let fn_total_ms = elapsed.as_secs_f64() * 1000.0;
        let disk_io_ms = disk_io.as_secs_f64() * 1000.0;
        let network_io_ms = network_io.as_secs_f64() * 1000.0;
        let compute_ms = (fn_total_ms - disk_io_ms - network_io_ms).max(0.0);

        let timing = ToolTiming {
            tool_name: tool_name.to_string(),
            fn_total_ms,
            disk_io_ms,
            network_io_ms,
            compute_ms,
        };

        timing.output();
        timing
    }
}

/// Output WASM total execution time
pub fn output_wasm_total(start: Instant) {
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("---WASM_TOTAL---{:.3}", elapsed_ms);
}

/// Global WASM start time (set once at module init)
static mut WASM_START: Option<Instant> = None;

/// Initialize WASM start time (call at the very beginning)
pub fn init_wasm_start() {
    unsafe {
        if WASM_START.is_none() {
            WASM_START = Some(Instant::now());
        }
    }
}

/// Get WASM start time
pub fn get_wasm_start() -> Option<Instant> {
    unsafe { WASM_START }
}

/// Reset WASM start time (call at beginning of each request for per-request timing)
pub fn reset_wasm_start() {
    unsafe {
        WASM_START = Some(Instant::now());
    }
}

/// Get WASM total elapsed time in milliseconds (from request start to now)
pub fn get_wasm_total_ms() -> f64 {
    unsafe {
        WASM_START
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}
