//! Data Aggregate MCP Server - WASM compatible (wasip2)
//!
//! A stateless MCP server that aggregates, merges, and summarizes structured data.
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
use serde_json::Value;

// ==========================================
// Parameter structs for JSON Schema generation
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AggregateListParams {
    /// List of dictionaries to aggregate
    pub items: Vec<Value>,
    /// Field name to group by
    pub group_by: Option<String>,
    /// Field to count occurrences of unique values
    pub count_field: Option<String>,
    /// List of field names with numeric values to compute statistics
    pub sum_fields: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MergeSummariesParams {
    /// List of summary dictionaries to merge
    pub summaries: Vec<Value>,
    /// Optional weights for each summary (for weighted averages)
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CombineResearchResultsParams {
    /// List of research result dictionaries
    pub results: Vec<Value>,
    /// Field containing the title
    #[serde(default = "default_title_field")]
    pub title_field: String,
    /// Field containing the summary
    #[serde(default = "default_summary_field")]
    pub summary_field: String,
    /// Optional field for relevance scoring
    #[serde(default = "default_score_field")]
    pub score_field: Option<String>,
}

fn default_title_field() -> String { "title".to_string() }
fn default_summary_field() -> String { "summary".to_string() }
fn default_score_field() -> Option<String> { Some("relevance_score".to_string()) }

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeduplicateParams {
    /// List of items to deduplicate
    pub items: Vec<Value>,
    /// Fields to use as the deduplication key
    pub key_fields: Vec<String>,
    /// Which duplicate to keep (first or last)
    #[serde(default = "default_keep")]
    pub keep: String,
}

fn default_keep() -> String { "first".to_string() }

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeTrendsParams {
    /// List of time-series data points
    pub time_series: Vec<Value>,
    /// Field containing the timestamp
    #[serde(default = "default_time_field")]
    pub time_field: String,
    /// Field containing the value
    #[serde(default = "default_value_field")]
    pub value_field: String,
    /// Number of time buckets
    #[serde(default = "default_bucket_count")]
    pub bucket_count: usize,
}

fn default_time_field() -> String { "timestamp".to_string() }
fn default_value_field() -> String { "value".to_string() }
fn default_bucket_count() -> usize { 10 }

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-data-aggregate")
        .version("1.0.0")
        .description("Data Aggregate MCP Server - Aggregates, merges, and summarizes structured data")
        .tool::<AggregateListParams, _>(
            "aggregate_list",
            "Aggregate a list of dictionaries by grouping, counting, or summing. Returns statistics like counts per group, value distributions, and numeric field stats.",
            |params| tools::aggregate_list(
                &params.items,
                params.group_by.as_deref(),
                params.count_field.as_deref(),
                params.sum_fields.as_ref().map(|v| v.as_slice()),
            )
        )
        .tool::<MergeSummariesParams, _>(
            "merge_summaries",
            "Merge multiple summary dictionaries into one. Supports weighted averages for numeric values.",
            |params| tools::merge_summaries(
                &params.summaries,
                params.weights.as_ref().map(|v| v.as_slice()),
            )
        )
        .tool::<CombineResearchResultsParams, _>(
            "combine_research_results",
            "Combine multiple research/search results into a coherent summary. Sorts by relevance score and creates combined text.",
            |params| tools::combine_research_results(
                &params.results,
                &params.title_field,
                &params.summary_field,
                params.score_field.as_deref(),
            )
        )
        .tool::<DeduplicateParams, _>(
            "deduplicate",
            "Remove duplicate items based on key fields. Returns deduplicated items with statistics.",
            |params| tools::deduplicate(
                &params.items,
                &params.key_fields,
                &params.keep,
            )
        )
        .tool::<ComputeTrendsParams, _>(
            "compute_trends",
            "Compute trends from time-series data. Analyzes whether values are increasing, decreasing, or stable.",
            |params| tools::compute_trends(
                &params.time_series,
                &params.time_field,
                &params.value_field,
                params.bucket_count,
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
