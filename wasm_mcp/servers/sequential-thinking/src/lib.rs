//! Sequential Thinking MCP Server - WASM compatible (wasip2)
//!
//! A tool for dynamic and reflective problem-solving through structured thinking.
//! Designed to run as a WASM component with Wasmtime runtime.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run)
//! - `cargo build --features http-export` → HTTP server (wasmtime serve)
//!
//! Both use the same `create_server()` function with shared business logic.

pub mod tools;

// Keep service module for backward compatibility (rmcp-based)
#[cfg(feature = "rmcp-service")]
pub mod service;

use wasmmcp::schemars::JsonSchema;
use wasmmcp::serde::Deserialize;
use wasmmcp::prelude::*;

// ==========================================
// Parameter structs for JSON Schema generation
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SequentialThinkingParams {
    /// The current thinking step content
    pub thought: String,

    /// Whether another thought step is needed
    #[serde(rename = "nextThoughtNeeded")]
    pub next_thought_needed: bool,

    /// Current thought number in the sequence
    #[serde(rename = "thoughtNumber")]
    pub thought_number: i32,

    /// Estimated total number of thoughts needed
    #[serde(rename = "totalThoughts")]
    pub total_thoughts: i32,

    /// Whether this thought revises previous thinking
    #[serde(rename = "isRevision", default)]
    pub is_revision: Option<bool>,

    /// Which thought number is being revised
    #[serde(rename = "revisesThought")]
    pub revises_thought: Option<i32>,

    /// Thought number to branch from
    #[serde(rename = "branchFromThought")]
    pub branch_from_thought: Option<i32>,

    /// Identifier for the branch
    #[serde(rename = "branchId")]
    pub branch_id: Option<String>,

    /// Whether more thoughts are needed beyond the original estimate
    #[serde(rename = "needsMoreThoughts", default)]
    pub needs_more_thoughts: Option<bool>,
}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-sequential-thinking")
        .version("1.0.0")
        .description("Sequential Thinking MCP Server - A tool for dynamic and reflective problem-solving through structured thinking")
        .tool::<SequentialThinkingParams, _>(
            "sequentialthinking",
            "A detailed tool for dynamic and reflective problem-solving through structured thinking. Facilitates step-by-step analysis with support for revisions, branching into alternative paths, and dynamic adjustment of the thinking process. Use this for complex problems requiring careful reasoning.",
            |params| tools::sequentialthinking(
                &params.thought,
                params.next_thought_needed,
                params.thought_number,
                params.total_thoughts,
                params.is_revision,
                params.revises_thought,
                params.branch_from_thought,
                params.branch_id.as_deref(),
                params.needs_more_thoughts,
            )
        )
        .build()
}

// ==========================================
// CLI Export (wasmtime run)
// ==========================================

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// ==========================================
// HTTP Export (wasmtime serve)
// ==========================================

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
