//! Git MCP Server - WASM Entry Point
//!
//! Uses wasi:cli/run for stdio transport.
//! Run with: wasmtime run --dir /path/to/repo target/wasm32-wasip2/release/mcp_server_git.wasm

mod service;

use service::GitService;
use wasmmcp::prelude::{StdioTransport, Transport};
use rmcp::ServiceExt;
use std::time::Instant;

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

            let transport = StdioTransport::new();
            let (input, output) = transport.streams();

            match GitService::new().serve((input, output)).await {
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
