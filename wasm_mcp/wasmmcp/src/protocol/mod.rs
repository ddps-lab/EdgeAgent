//! MCP Protocol types and handling
//!
//! Re-exports and extends types from rmcp for MCP protocol handling.

pub mod jsonrpc;

// Re-export useful types from rmcp
pub use rmcp::model::{
    ServerCapabilities,
    ServerInfo,
};
