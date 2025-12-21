//! HTTP transport implementation using wasi:http/incoming-handler
//!
//! Provides HTTP server functionality for serverless deployments.

use wasi::http::types::{
    Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam,
};
use crate::timing::{get_tool_exec_ms, get_io_ms};

/// HTTP transport configuration
pub struct HttpTransport {
    path: String,
}

impl HttpTransport {
    /// Create a new HTTP transport with the default path "/mcp"
    pub fn new() -> Self {
        Self {
            path: "/mcp".to_string(),
        }
    }

    /// Set the MCP endpoint path
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    /// Get the configured path
    pub fn get_path(&self) -> &str {
        &self.path
    }
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle an incoming HTTP request and dispatch to MCP handler
///
/// This function is called from the generated wasi:http/incoming-handler export.
pub fn handle_http_request<F>(
    handler_factory: F,
    request: IncomingRequest,
    response_out: ResponseOutparam,
) where
    F: FnOnce() -> Vec<u8>,
{
    // Get request method and path
    let method = request.method();
    let path_with_query = request.path_with_query().unwrap_or_default();

    // Read request body
    let body_content = read_request_body(&request);

    // Create response headers
    let headers = Fields::new();
    let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
    let _ = headers.set(&"Access-Control-Allow-Origin".to_string(), &[b"*".to_vec()]);
    let _ = headers.set(&"Access-Control-Allow-Methods".to_string(), &[b"GET, POST, OPTIONS".to_vec()]);
    let _ = headers.set(&"Access-Control-Allow-Headers".to_string(), &[b"Content-Type".to_vec()]);

    // Handle CORS preflight
    if matches!(method, wasi::http::types::Method::Options) {
        send_response(response_out, 204, headers, b"");
        return;
    }

    // Process MCP request
    let response_body = handler_factory();

    // Add timing headers (values set by handler during tool execution)
    let tool_exec_ms = get_tool_exec_ms();
    let io_ms = get_io_ms();
    let _ = headers.set(
        &"X-Tool-Exec-Ms".to_string(),
        &[format!("{:.3}", tool_exec_ms).into_bytes()],
    );
    let _ = headers.set(
        &"X-IO-Ms".to_string(),
        &[format!("{:.3}", io_ms).into_bytes()],
    );

    // Send response
    send_response(response_out, 200, headers, &response_body);
}

/// Read the body of an incoming HTTP request
fn read_request_body(request: &IncomingRequest) -> Vec<u8> {
    let mut body_content = Vec::new();

    if let Some(incoming_body) = request.consume().ok() {
        if let Ok(stream) = incoming_body.stream() {
            loop {
                match stream.blocking_read(4096) {
                    Ok(chunk) => {
                        if chunk.is_empty() {
                            break;
                        }
                        body_content.extend_from_slice(&chunk);
                    }
                    Err(_) => break,
                }
            }
            drop(stream);
        }
    }

    body_content
}

/// Send an HTTP response
fn send_response(response_out: ResponseOutparam, status: u16, headers: Fields, body: &[u8]) {
    let resp = OutgoingResponse::new(headers);
    let _ = resp.set_status_code(status);

    let outgoing_body = resp.body().unwrap();
    ResponseOutparam::set(response_out, Ok(resp));

    if !body.is_empty() {
        let out = outgoing_body.write().unwrap();
        // Write in chunks due to WASI stream 4KB limit
        // See: https://github.com/bytecodealliance/wasmtime/issues/9653
        const CHUNK_SIZE: usize = 4096;
        for chunk in body.chunks(CHUNK_SIZE) {
            if out.blocking_write_and_flush(chunk).is_err() {
                break;
            }
        }
        drop(out);
    }

    let _ = OutgoingBody::finish(outgoing_body, None);
}

/// MCP over HTTP request handler
///
/// Handles the MCP Streamable HTTP transport specification:
/// - POST /mcp: Tool calls and other requests
/// - GET /mcp: SSE stream for server-initiated messages (optional)
pub struct McpHttpHandler {
    path: String,
}

impl McpHttpHandler {
    /// Create a new MCP HTTP handler
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// Process an MCP request over HTTP
    pub fn process_request(&self, method: &str, path: &str, body: &[u8]) -> (u16, Vec<u8>) {
        // Check if path matches
        if !path.starts_with(&self.path) {
            return (404, b"Not Found".to_vec());
        }

        match method {
            "POST" => {
                // Parse JSON-RPC request and dispatch
                // For now, return a placeholder
                (200, body.to_vec())
            }
            "GET" => {
                // SSE endpoint for streaming (stateful mode)
                (501, b"SSE not implemented in stateless mode".to_vec())
            }
            _ => (405, b"Method Not Allowed".to_vec()),
        }
    }
}
