//! Fetch MCP Server - WASM Entry Point
//!
//! Uses wasi:cli/run for stdio transport and wasi:http/outgoing-handler for fetching.
//! Run with: wasmtime run -S http target/wasm32-wasip2/release/mcp_server_fetch.wasm

mod service;

use service::FetchService;
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

            match FetchService::new().serve((input, output)).await {
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
