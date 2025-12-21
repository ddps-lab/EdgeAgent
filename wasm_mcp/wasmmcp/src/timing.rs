//! Timing utilities for WASM MCP tool profiling
//!
//! Provides timing measurement for:
//! - Cold start time (WASM loading + runtime init)
//! - Tool total execution time (fn_total)
//! - Disk I/O time (file operations)
//! - Network I/O time (HTTP requests)
//! - Compute time (fn_total - disk_io - network_io)

use std::cell::RefCell;
use std::time::{Duration, Instant};

// Thread-local storage for timing data
thread_local! {
    // Legacy single I/O accumulator (for backward compatibility)
    static IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    // Separate accumulators for disk and network I/O
    static DISK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    static NETWORK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    // Timing values for HTTP transport
    static TOOL_EXEC_MS: RefCell<f64> = RefCell::new(0.0);
    static IO_MS: RefCell<f64> = RefCell::new(0.0);
    static DISK_IO_MS: RefCell<f64> = RefCell::new(0.0);
    static NETWORK_IO_MS: RefCell<f64> = RefCell::new(0.0);
    static COMPUTE_MS: RefCell<f64> = RefCell::new(0.0);
}

/// Reset the I/O accumulator (call before each tool execution)
pub fn reset_io_accumulator() {
    IO_ACCUMULATOR.with(|acc| {
        *acc.borrow_mut() = Duration::ZERO;
    });
}

/// Reset all I/O accumulators (call before each tool execution)
pub fn reset_io_accumulators() {
    IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
}

/// Add duration to the I/O accumulator
pub fn add_io_duration(duration: Duration) {
    IO_ACCUMULATOR.with(|acc| {
        *acc.borrow_mut() += duration;
    });
}

/// Add duration to disk I/O accumulator
pub fn add_disk_io_duration(duration: Duration) {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Add duration to network I/O accumulator
pub fn add_network_io_duration(duration: Duration) {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Get the accumulated I/O duration
pub fn get_io_duration() -> Duration {
    IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Get accumulated disk I/O duration
pub fn get_disk_io_duration() -> Duration {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Get accumulated network I/O duration
pub fn get_network_io_duration() -> Duration {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Set tool execution time (for HTTP transport to read)
pub fn set_tool_exec_ms(ms: f64) {
    TOOL_EXEC_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get tool execution time
pub fn get_tool_exec_ms() -> f64 {
    TOOL_EXEC_MS.with(|v| *v.borrow())
}

/// Set I/O time (for HTTP transport to read)
pub fn set_io_ms(ms: f64) {
    IO_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get I/O time
pub fn get_io_ms() -> f64 {
    IO_MS.with(|v| *v.borrow())
}

/// Set disk I/O time (for HTTP transport to read)
pub fn set_disk_io_ms(ms: f64) {
    DISK_IO_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get disk I/O time
pub fn get_disk_io_ms() -> f64 {
    DISK_IO_MS.with(|v| *v.borrow())
}

/// Set network I/O time (for HTTP transport to read)
pub fn set_network_io_ms(ms: f64) {
    NETWORK_IO_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get network I/O time
pub fn get_network_io_ms() -> f64 {
    NETWORK_IO_MS.with(|v| *v.borrow())
}

/// Set compute time (for HTTP transport to read)
pub fn set_compute_ms(ms: f64) {
    COMPUTE_MS.with(|v| *v.borrow_mut() = ms);
}

/// Get compute time
pub fn get_compute_ms() -> f64 {
    COMPUTE_MS.with(|v| *v.borrow())
}

/// Reset all timing values
pub fn reset_timing() {
    reset_io_accumulators();
    TOOL_EXEC_MS.with(|v| *v.borrow_mut() = 0.0);
    IO_MS.with(|v| *v.borrow_mut() = 0.0);
    DISK_IO_MS.with(|v| *v.borrow_mut() = 0.0);
    NETWORK_IO_MS.with(|v| *v.borrow_mut() = 0.0);
    COMPUTE_MS.with(|v| *v.borrow_mut() = 0.0);
}

/// Measure an I/O operation and add to the accumulator (legacy, defaults to disk I/O)
///
/// Usage:
/// ```ignore
/// let content = measure_io(|| std::fs::read_to_string(path))?;
/// ```
pub fn measure_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    add_io_duration(elapsed);
    add_disk_io_duration(elapsed);  // Also count as disk I/O for new API
    result
}

/// Measure a disk I/O operation (file read/write)
///
/// Usage:
/// ```ignore
/// let content = measure_disk_io(|| std::fs::read_to_string(path))?;
/// ```
pub fn measure_disk_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    add_disk_io_duration(elapsed);
    add_io_duration(elapsed);  // Also update legacy accumulator
    result
}

/// Measure a network I/O operation (HTTP request, etc.)
///
/// Usage:
/// ```ignore
/// let response = measure_network_io(|| http_client.get(url))?;
/// ```
pub fn measure_network_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    add_network_io_duration(elapsed);
    add_io_duration(elapsed);  // Also update legacy accumulator
    result
}

/// Tool timing information
#[derive(Debug, Clone)]
pub struct ToolTiming {
    pub tool_name: String,
    pub fn_total_ms: f64,
    pub io_ms: f64,          // Legacy: total I/O
    pub disk_io_ms: f64,     // Disk I/O (file operations)
    pub network_io_ms: f64,  // Network I/O (HTTP requests)
    pub compute_ms: f64,
}

impl ToolTiming {
    /// Output timing information to stderr in the expected format
    pub fn output(&self) {
        let json = serde_json::json!({
            "tool": self.tool_name,
            "fn_total_ms": self.fn_total_ms,
            "io_ms": self.io_ms,
            "disk_io_ms": self.disk_io_ms,
            "network_io_ms": self.network_io_ms,
            "compute_ms": self.compute_ms,
        });
        eprintln!("---TIMING---{}", json);
    }
}

/// Timer for measuring tool execution
pub struct ToolTimer {
    start: Instant,
}

impl ToolTimer {
    /// Start a new timer (also resets all I/O accumulators)
    pub fn start() -> Self {
        reset_io_accumulators();
        Self {
            start: Instant::now(),
        }
    }

    /// Finish timing and output the results
    pub fn finish(self, tool_name: &str) -> ToolTiming {
        let elapsed = self.start.elapsed();
        let io_duration = get_io_duration();
        let disk_io_duration = get_disk_io_duration();
        let network_io_duration = get_network_io_duration();

        let fn_total_ms = elapsed.as_secs_f64() * 1000.0;
        let io_ms = io_duration.as_secs_f64() * 1000.0;
        let disk_io_ms = disk_io_duration.as_secs_f64() * 1000.0;
        let network_io_ms = network_io_duration.as_secs_f64() * 1000.0;
        let compute_ms = (fn_total_ms - disk_io_ms - network_io_ms).max(0.0);

        // Store values for HTTP transport to read
        set_tool_exec_ms(fn_total_ms);
        set_io_ms(io_ms);
        set_disk_io_ms(disk_io_ms);
        set_network_io_ms(network_io_ms);
        set_compute_ms(compute_ms);

        let timing = ToolTiming {
            tool_name: tool_name.to_string(),
            fn_total_ms,
            io_ms,
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
