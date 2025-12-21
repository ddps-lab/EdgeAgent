//! WasmMCP - FastMCP-like framework for building MCP servers in WASM
//!
//! A simple, ergonomic framework for creating Model Context Protocol (MCP) servers
//! that compile to WebAssembly and run on WASI-compatible runtimes.
//!
//! # Features
//!
//! - **Simple API**: Define tools with `#[mcp_tool]` attribute
//! - **Dual Transport**: Supports both stdio (CLI) and HTTP (serverless)
//! - **Stateless**: Optimized for serverless deployments
//! - **Type-safe**: Automatic JSON schema generation from Rust types
//!
//! # Example
//!
//! ```rust,ignore
//! use wasmmcp::prelude::*;
//!
//! #[wasmmcp_main]
//! async fn main() {
//!     let server = WasmMcp::builder("my-server")
//!         .tool::<ReadFileParams>(read_file)
//!         .build();
//!
//!     server.run().await.unwrap();
//! }
//!
//! #[mcp_tool(description = "Read a file")]
//! fn read_file(path: String) -> Result<String, String> {
//!     std::fs::read_to_string(&path).map_err(|e| e.to_string())
//! }
//! ```

pub mod server;
pub mod transport;
pub mod protocol;
pub mod registry;
pub mod builder;
pub mod timing;

// Re-export macros
pub use wasmmcp_macros::{mcp_tool, wasmmcp_main, wasmmcp_http, wasmmcp_tool, export_cli, export_http};

// Re-export rmcp entirely so users don't need to depend on rmcp directly
pub use rmcp;

// Re-export commonly used types from rmcp at top level for convenience
pub use rmcp::{
    ServerHandler,
    ServiceExt,
    model::{ServerCapabilities, ServerInfo},
    schemars,
    tool, tool_router, tool_handler,
};

// Re-export serde for tool parameter structs
pub use serde;
pub use serde_json;

/// Prelude module for convenient imports
pub mod prelude {
    // WasmMCP types
    pub use crate::server::{WasmMcp, WasmMcpBuilder};
    pub use crate::transport::{Transport, StdioTransport};
    #[cfg(feature = "transport-http")]
    pub use crate::transport::HttpTransport;
    pub use crate::registry::{Tool, ToolRegistry, ToolInfo, FnTool};
    pub use crate::builder::{McpServer, McpServerBuilder, JsonRpcResult};

    // WasmMCP macros
    pub use crate::{mcp_tool, wasmmcp_main, wasmmcp_tool, export_cli, export_http};

    // Re-exports from rmcp (so users don't need rmcp dependency)
    pub use crate::rmcp::ServiceExt;
    pub use crate::rmcp::handler::server::wrapper::Parameters;
    pub use crate::{tool, tool_router, tool_handler};
    pub use crate::{ServerHandler, ServerCapabilities, ServerInfo};

    // Re-exports for derive macros
    pub use crate::schemars;
    pub use crate::serde;

    // Timing utilities
    pub use crate::timing::{measure_io, ToolTimer, get_tool_exec_ms, get_io_ms, set_tool_exec_ms, set_io_ms, reset_timing};
}

/// Error type for WasmMCP operations
#[derive(Debug)]
pub enum Error {
    /// Transport error
    Transport(String),
    /// Protocol error
    Protocol(String),
    /// IO error
    Io(std::io::Error),
    /// Serialization error
    Serde(serde_json::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Transport(msg) => write!(f, "Transport error: {}", msg),
            Error::Protocol(msg) => write!(f, "Protocol error: {}", msg),
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Serde(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serde(e)
    }
}

/// Result type alias for WasmMCP operations
pub type Result<T> = std::result::Result<T, Error>;
