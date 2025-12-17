//! Fetch MCP Server - WASM Entry Point
//!
//! Uses wasi:cli/run for stdio transport and wasi:http/outgoing-handler for fetching.
//! Run with: wasmtime run -S http target/wasm32-wasip2/release/mcp_server_fetch.wasm

mod service;

use service::FetchService;
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
            let transport = StdioTransport::new();
            let (input, output) = transport.streams();

            match FetchService::new().serve((input, output)).await {
                Ok(server) => {
                    let _ = server.waiting().await;
                }
                Err(_) => {}
            }
        });

        // serve() 완료 후 WASM 전체 실행 시간 출력
        let wasm_total_ms = wasm_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("---WASM_TOTAL---{:.3}", wasm_total_ms);
        Ok(())
    }
}

wasi::cli::command::export!(TokioCliRunner);
