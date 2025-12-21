//! Sequential Thinking tools - Pure business logic
//!
//! Shared between CLI and HTTP transports.

use serde_json::json;
#[allow(unused_imports)]
use wasmmcp::timing::{measure_io, ToolTimer, get_wasm_total_ms};

/// Process a sequential thinking step
///
/// Validates input and builds response matching Node mcp-server-sequential-thinking format.
pub fn sequentialthinking(
    thought: &str,
    next_thought_needed: bool,
    thought_number: i32,
    total_thoughts: i32,
    _is_revision: Option<bool>,
    _revises_thought: Option<i32>,
    _branch_from_thought: Option<i32>,
    _branch_id: Option<&str>,
    _needs_more_thoughts: Option<bool>,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    // Validate inputs
    if thought_number < 1 {
        return Err("thoughtNumber must be >= 1".to_string());
    }
    if total_thoughts < 1 {
        return Err("totalThoughts must be >= 1".to_string());
    }

    // Thought content is used for validation but not stored (stateless)
    let _ = thought;

    // Build branches array (empty for now, matching Node server)
    let branches: Vec<String> = Vec::new();

    let timing = timer.finish("sequentialthinking");
    // Build response - matches Node mcp-server-sequential-thinking output format
    let response = json!({
        "thoughtNumber": thought_number,
        "totalThoughts": total_thoughts,
        "nextThoughtNeeded": next_thought_needed,
        "branches": branches,
        "thoughtHistoryLength": thought_number,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    });

    Ok(response.to_string())
}
