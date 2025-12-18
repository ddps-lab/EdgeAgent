//! Summarize MCP Server - WASM compatible (wasip2)
//!
//! Uses OpenAI or Upstage API for text summarization via wasi:http/outgoing-handler.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run -S http)
//! - `cargo build --features http-export` → HTTP server (wasmtime serve)
//!
//! # Environment Variables
//!
//! - `OPENAI_API_KEY` - OpenAI API key (default provider)
//! - `UPSTAGE_API_KEY` - Upstage API key
//! - `SUMMARIZE_PROVIDER` - Provider to use: "openai" (default) or "upstage"

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

fn default_max_length() -> i32 { 150 }
fn default_style() -> String { "concise".to_string() }
fn default_max_length_per_doc() -> i32 { 100 }

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SummarizeTextParams {
    /// The text to summarize
    pub text: String,

    /// Maximum length of summary in words (default: 150)
    #[serde(default = "default_max_length")]
    pub max_length: i32,

    /// Summary style - "concise", "detailed", or "bullet"
    #[serde(default = "default_style")]
    pub style: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SummarizeDocumentsParams {
    /// List of documents to summarize
    pub documents: Vec<String>,

    /// Maximum length per summary in words
    #[serde(default = "default_max_length_per_doc")]
    pub max_length_per_doc: i32,

    /// Summary style for all documents
    #[serde(default = "default_style")]
    pub style: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetProviderInfoParams {}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    // Read provider from environment (default: openai)
    let provider = std::env::var("SUMMARIZE_PROVIDER")
        .unwrap_or_else(|_| "openai".to_string());

    // Read API key from environment
    let api_key: Option<String> = match provider.as_str() {
        "upstage" => std::env::var("UPSTAGE_API_KEY").ok(),
        _ => std::env::var("OPENAI_API_KEY").ok(),
    };

    // Clone for closures
    let provider1 = provider.clone();
    let provider2 = provider.clone();
    let provider3 = provider.clone();
    let api_key1 = api_key.clone();
    let api_key2 = api_key.clone();

    McpServer::builder("wasmmcp-summarize")
        .version("1.0.0")
        .description("Summarize MCP Server - Summarizes text using LLM APIs (OpenAI/Upstage)")
        .tool::<SummarizeTextParams, _>(
            "summarize_text",
            "Summarize the given text",
            move |params| {
                tools::summarize_text(
                    &provider1,
                    api_key1.as_deref(),
                    &params.text,
                    params.max_length,
                    &params.style
                )
            }
        )
        .tool::<SummarizeDocumentsParams, _>(
            "summarize_documents",
            "Summarize multiple documents",
            move |params| {
                tools::summarize_documents(
                    &provider2,
                    api_key2.as_deref(),
                    &params.documents,
                    params.max_length_per_doc,
                    &params.style
                )
            }
        )
        .tool::<GetProviderInfoParams, _>(
            "get_provider_info",
            "Get information about the current summarization provider",
            move |_params| {
                tools::get_provider_info(&provider3)
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
