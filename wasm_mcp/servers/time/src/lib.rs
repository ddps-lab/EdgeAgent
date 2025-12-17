//! Time MCP Server - WASM compatible (wasip2)
//!
//! A stateless MCP server that provides time operations and timezone conversion.
//! Designed to run as a WASM component with Wasmtime runtime.

pub mod service;

use rmcp::ServiceExt;
use wasmmcp::transport::{StdioTransport, Transport};
use std::time::Instant;

use service::TimeService;

/// WASI CLI runner that sets up the Tokio runtime and runs the MCP server
struct TokioCliRunner;

impl wasi::exports::cli::run::Guest for TokioCliRunner {
    fn run() -> Result<(), ()> {
        let wasm_start = Instant::now();  // WASM 코드 실행 시작 시점

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let runtime_init_ms = wasm_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("---WASM_INIT---{:.3}", runtime_init_ms);

            // Use wasmmcp's stdio transport
            let transport = StdioTransport::new();
            let (input, output) = transport.streams();

            match TimeService::new().serve((input, output)).await {
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
