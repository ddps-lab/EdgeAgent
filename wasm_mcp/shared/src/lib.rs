//! Shared utilities for MCP WASM servers
//!
//! This crate provides common utilities and traits used across
//! different MCP server implementations.

use anyhow::Result;

/// Timing utilities for profiling MCP tool execution
pub mod timing {
    use std::time::{Duration, Instant};

    /// Result of a timed operation
    #[derive(Debug, Clone)]
    pub struct TimingResult<T> {
        pub result: T,
        pub duration: Duration,
    }

    impl<T> TimingResult<T> {
        pub fn duration_ms(&self) -> f64 {
            self.duration.as_secs_f64() * 1000.0
        }
    }

    /// Measure execution time of a closure (for I/O operations)
    ///
    /// # Example
    /// ```
    /// use mcp_shared::timing::measure;
    ///
    /// let TimingResult { result, duration } = measure(|| {
    ///     std::fs::read_to_string("/tmp/test.txt")
    /// });
    /// ```
    #[inline]
    pub fn measure<F, T>(f: F) -> TimingResult<T>
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        TimingResult { result, duration }
    }

    /// Tool timing context - tracks tool execution and I/O time
    #[derive(Debug)]
    pub struct ToolTimer {
        tool_start: Instant,
        io_total: Duration,
    }

    impl ToolTimer {
        /// Start timing a tool
        pub fn start() -> Self {
            Self {
                tool_start: Instant::now(),
                io_total: Duration::ZERO,
            }
        }

        /// Measure an I/O operation and accumulate time
        pub fn measure_io<F, T>(&mut self, f: F) -> T
        where
            F: FnOnce() -> T,
        {
            let start = Instant::now();
            let result = f();
            self.io_total += start.elapsed();
            result
        }

        /// Finish timing and output results to stderr
        /// Format: ---TIMING---{"tool_ms":X,"io_ms":Y}
        pub fn finish(self, tool_name: &str) {
            let tool_ms = self.tool_start.elapsed().as_secs_f64() * 1000.0;
            let io_ms = self.io_total.as_secs_f64() * 1000.0;
            eprintln!(
                "---TIMING---{{\"tool\":\"{}\",\"tool_ms\":{:.3},\"io_ms\":{:.3}}}",
                tool_name, tool_ms, io_ms
            );
        }

        /// Get current tool execution time in ms
        pub fn tool_ms(&self) -> f64 {
            self.tool_start.elapsed().as_secs_f64() * 1000.0
        }

        /// Get accumulated I/O time in ms
        pub fn io_ms(&self) -> f64 {
            self.io_total.as_secs_f64() * 1000.0
        }
    }
}

/// Common utilities for MCP servers
pub mod utils {
    use anyhow::{anyhow, Result};
    use std::path::Path;

    /// Validate that a path is safe to access
    pub fn validate_path(path: &str) -> Result<()> {
        let path = Path::new(path);

        // Check for path traversal attempts
        for component in path.components() {
            if let std::path::Component::ParentDir = component {
                return Err(anyhow!("Path traversal not allowed: {}", path.display()));
            }
        }

        Ok(())
    }

    /// Normalize a path by removing redundant components
    pub fn normalize_path(path: &str) -> String {
        let path = Path::new(path);
        let mut components = Vec::new();

        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    components.pop();
                }
                std::path::Component::CurDir => {}
                _ => {
                    components.push(component);
                }
            }
        }

        let normalized: std::path::PathBuf = components.iter().collect();
        normalized.to_string_lossy().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_validate_path_normal() {
        assert!(validate_path("/tmp/test.txt").is_ok());
        assert!(validate_path("test.txt").is_ok());
    }

    #[test]
    fn test_validate_path_traversal() {
        assert!(validate_path("../etc/passwd").is_err());
        assert!(validate_path("/tmp/../etc/passwd").is_err());
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("/tmp/./test.txt"), "/tmp/test.txt");
    }
}
