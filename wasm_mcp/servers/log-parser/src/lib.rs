//! Log Parser MCP Server - WASM compatible (wasip2)
//!
//! A stateless MCP server that provides log parsing and analysis.
//! Designed to run as a WASM component with Wasmtime runtime.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run)
//! - `cargo build --features http-export` → HTTP server (wasmtime serve)

pub mod tools;

#[cfg(feature = "rmcp-service")]
pub mod service;

use wasmmcp::schemars::JsonSchema;
use wasmmcp::serde::Deserialize;
use wasmmcp::prelude::*;
use serde_json::Value;

// ==========================================
// Parameter structs for JSON Schema generation
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ParseLogsParams {
    /// Raw log content (multi-line string)
    pub log_content: String,
    /// Log format type or "auto" for auto-detection
    #[serde(default = "default_format")]
    pub format_type: String,
    /// Maximum entries to parse
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
}

fn default_format() -> String { "auto".to_string() }
fn default_max_entries() -> usize { 1000 }

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterEntriesParams {
    /// Parsed log entries
    pub entries: Vec<Value>,
    /// Minimum severity level
    #[serde(default = "default_min_level")]
    pub min_level: String,
    /// Specific levels to include
    pub include_levels: Option<Vec<String>>,
}

fn default_min_level() -> String { "warning".to_string() }

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeStatisticsParams {
    /// Parsed log entries
    pub entries: Vec<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchEntriesParams {
    /// Parsed log entries
    pub entries: Vec<Value>,
    /// Regex pattern to search
    pub pattern: String,
    /// Fields to search in
    pub fields: Option<Vec<String>>,
    /// Case sensitive search
    #[serde(default)]
    pub case_sensitive: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractTimeRangeParams {
    /// Parsed log entries
    pub entries: Vec<Value>,
}

// ==========================================
// Unified Server Factory
// ==========================================

pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-log-parser")
        .version("1.0.0")
        .description("Log Parser MCP Server - Parse various log formats and analyze entries")
        .tool::<ParseLogsParams, _>(
            "parse_logs",
            "Parse raw log content into structured entries. Returns entries with _level field added.",
            |params| tools::parse_logs(&params.log_content, &params.format_type, params.max_entries)
        )
        .tool::<FilterEntriesParams, _>(
            "filter_entries",
            "Filter log entries by severity level. Pass entries from parse_logs result.",
            |params| tools::filter_entries(
                &params.entries,
                &params.min_level,
                params.include_levels.as_ref().map(|v| v.as_slice()),
            )
        )
        .tool::<ComputeStatisticsParams, _>(
            "compute_log_statistics",
            "Compute statistics from parsed log entries.",
            |params| tools::compute_log_statistics(&params.entries)
        )
        .tool::<SearchEntriesParams, _>(
            "search_entries",
            "Search log entries by regex pattern.",
            |params| tools::search_entries(
                &params.entries,
                &params.pattern,
                params.fields.as_ref().map(|v| v.as_slice()),
                params.case_sensitive,
            )
        )
        .tool::<ExtractTimeRangeParams, _>(
            "extract_time_range",
            "Extract time range information from log entries.",
            |params| tools::extract_time_range(&params.entries)
        )
        .build()
}

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
