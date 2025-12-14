//! Sequential Thinking MCP Server - WASM compatible (wasip2)
//!
//! A tool for dynamic and reflective problem-solving through structured thinking.
//! Designed to run as a WASM component with Wasmtime runtime.

pub mod service;

use rmcp::ServiceExt;
use wasmmcp::transport::{StdioTransport, Transport};

use service::SequentialThinkingService;

/// WASI CLI runner that sets up the Tokio runtime and runs the MCP server
struct TokioCliRunner;

impl wasi::exports::cli::run::Guest for TokioCliRunner {
    fn run() -> Result<(), ()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            // Use wasmmcp's stdio transport
            let transport = StdioTransport::new();
            let (input, output) = transport.streams();

            match SequentialThinkingService::new().serve((input, output)).await {
                Ok(server) => {
                    // Gracefully handle connection close
                    let _ = server.waiting().await;
                }
                Err(_) => {
                    // Connection failed or closed early - exit gracefully
                }
            }
        });
        Ok(())
    }
}

wasi::cli::command::export!(TokioCliRunner);
