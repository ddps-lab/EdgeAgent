//! Time MCP Server - WASM compatible (wasip2)
//!
//! A stateless MCP server that provides time operations and timezone conversion.
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
pub struct GetCurrentTimeParams {
    /// IANA timezone name (e.g., 'America/New_York', 'Asia/Seoul', 'UTC')
    pub timezone: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConvertTimeParams {
    /// Source IANA timezone name (e.g., 'UTC', 'America/New_York')
    pub source_timezone: String,
    /// Time to convert in 24-hour format (HH:MM)
    pub time: String,
    /// Target IANA timezone name (e.g., 'Asia/Seoul', 'Europe/London')
    pub target_timezone: String,
}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-time")
        .version("1.0.0")
        .description("Time MCP Server - Get current time and convert between timezones")
        .tool::<GetCurrentTimeParams, _>(
            "get_current_time",
            "Get the current time in a specific timezone. Returns ISO 8601 formatted time with timezone offset.",
            |params| tools::get_current_time(&params.timezone)
        )
        .tool::<ConvertTimeParams, _>(
            "convert_time",
            "Convert a time from one timezone to another. Input time should be in HH:MM 24-hour format.",
            |params| tools::convert_time(&params.source_timezone, &params.time, &params.target_timezone)
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
