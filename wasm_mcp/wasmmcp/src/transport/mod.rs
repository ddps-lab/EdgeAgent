//! Transport layer for WasmMCP
//!
//! Provides abstraction over different transport mechanisms:
//! - `StdioTransport`: For CLI usage via WASI stdin/stdout
//! - `HttpTransport`: For HTTP servers via wasi:http (serverless)

#[cfg(feature = "transport-stdio")]
mod stdio;

#[cfg(feature = "transport-http")]
mod http;

#[cfg(feature = "transport-stdio")]
pub use stdio::{StdioTransport, AsyncInputStream, AsyncOutputStream};

#[cfg(feature = "transport-http")]
pub use http::HttpTransport;

use tokio::io::{AsyncRead, AsyncWrite};

/// Transport trait for different communication mechanisms
pub trait Transport {
    /// The input stream type
    type Input: AsyncRead + Unpin + Send + 'static;
    /// The output stream type
    type Output: AsyncWrite + Unpin + Send + 'static;

    /// Get the input and output streams
    fn streams(self) -> (Self::Input, Self::Output);
}
