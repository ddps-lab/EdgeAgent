//! Fetch MCP Server - WASM compatible (wasip2)
//!
//! Fetches web pages and converts HTML to markdown using wasi:http/outgoing-handler.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run -S http)
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
pub struct FetchParams {
    /// URL to fetch
    pub url: String,

    /// Maximum length of returned content (default: 50000)
    pub max_length: Option<usize>,
}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-fetch")
        .version("1.0.0")
        .description("Fetch MCP Server - Retrieves web content and converts to markdown")
        .tool::<FetchParams, _>(
            "fetch",
            "Fetches a URL from the internet and extracts its contents as markdown",
            |params| {
                let max_length = params.max_length.unwrap_or(50000);
                tools::fetch(&params.url, max_length)
            }
        )
        .build()
}

// ==========================================
// CLI Export (wasmtime run -S http)
// ==========================================

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// ==========================================
// HTTP Export (wasmtime serve)
// ==========================================

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
