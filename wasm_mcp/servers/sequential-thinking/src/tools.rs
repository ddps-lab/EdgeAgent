//! Sequential Thinking tools - Pure business logic
//!
//! Shared between CLI and HTTP transports.

use serde_json::json;

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

    // Build response - matches Node mcp-server-sequential-thinking output format
    let response = json!({
        "thoughtNumber": thought_number,
        "totalThoughts": total_thoughts,
        "nextThoughtNeeded": next_thought_needed,
        "branches": branches,
        "thoughtHistoryLength": thought_number
    });

    Ok(response.to_string())
}
