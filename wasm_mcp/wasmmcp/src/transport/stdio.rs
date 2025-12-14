//! WASI stdio transport implementation
//!
//! Provides async wrappers around WASI stdin/stdout for JSON-RPC communication.

use std::task::{Poll, Waker};
use tokio::io::{AsyncRead, AsyncWrite};
use wasi::{
    cli::{
        stdin::{get_stdin, InputStream},
        stdout::{get_stdout, OutputStream},
    },
    io::streams::Pollable,
};

use super::Transport;

/// Stdio transport for WASI CLI mode
///
/// Uses WASI stdin/stdout for communication with MCP clients.
pub struct StdioTransport;

impl StdioTransport {
    /// Create a new stdio transport
    pub fn new() -> Self {
        Self
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for StdioTransport {
    type Input = AsyncInputStream;
    type Output = AsyncOutputStream;

    fn streams(self) -> (Self::Input, Self::Output) {
        let input = AsyncInputStream { inner: get_stdin() };
        let output = AsyncOutputStream { inner: get_stdout() };
        (input, output)
    }
}

/// Async wrapper for WASI InputStream
pub struct AsyncInputStream {
    inner: InputStream,
}

impl AsyncRead for AsyncInputStream {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let bytes = self
            .inner
            .read(buf.remaining() as u64)
            .map_err(std::io::Error::other)?;

        if bytes.is_empty() {
            let pollable = self.inner.subscribe();
            let waker = cx.waker().clone();
            runtime_poll(waker, pollable);
            return Poll::Pending;
        }

        buf.put_slice(&bytes);
        std::task::Poll::Ready(Ok(()))
    }
}

/// Async wrapper for WASI OutputStream
pub struct AsyncOutputStream {
    inner: OutputStream,
}

impl AsyncWrite for AsyncOutputStream {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        let writable_len = self.inner.check_write().map_err(std::io::Error::other)?;

        if writable_len == 0 {
            let pollable = self.inner.subscribe();
            let waker = cx.waker().clone();
            runtime_poll(waker, pollable);
            return Poll::Pending;
        }

        let bytes_to_write = buf.len().min(writable_len as usize);
        self.inner
            .write(&buf[0..bytes_to_write])
            .map_err(std::io::Error::other)?;

        Poll::Ready(Ok(bytes_to_write))
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        self.inner.flush().map_err(std::io::Error::other)?;
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        self.poll_flush(cx)
    }
}

/// Poll the WASI pollable using tokio runtime
fn runtime_poll(waker: Waker, pollable: Pollable) {
    tokio::task::spawn(async move {
        loop {
            if pollable.ready() {
                waker.wake();
                break;
            } else {
                tokio::task::yield_now().await;
            }
        }
    });
}
