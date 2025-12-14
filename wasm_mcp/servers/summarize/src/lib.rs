//! Summarize MCP Server - WASM Entry Point
//!
//! Uses wasi:cli/run for stdio transport.
//! Run with: wasmtime run target/wasm32-wasip2/release/mcp_server_summarize.wasm
//!
//! Environment variables:
//!   OPENAI_API_KEY - OpenAI API key (default provider)
//!   UPSTAGE_API_KEY - Upstage API key
//!   SUMMARIZE_PROVIDER - Provider to use: "openai" (default) or "upstage"

mod service;

use service::SummarizeService;
use wasmmcp::prelude::{StdioTransport, Transport};
use rmcp::ServiceExt;

struct TokioCliRunner;

impl wasi::exports::cli::run::Guest for TokioCliRunner {
    fn run() -> Result<(), ()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let transport = StdioTransport::new();
            let (input, output) = transport.streams();

            match SummarizeService::new().serve((input, output)).await {
                Ok(server) => {
                    let _ = server.waiting().await;
                }
                Err(_) => {}
            }
        });

        Ok(())
    }
}

wasi::cli::command::export!(TokioCliRunner);
