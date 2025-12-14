//! HTTP entry point for Fetch MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve -S http target/wasm32-wasip2/release/mcp_server_fetch_http.wasm

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use url::Url;
use wasi::http::types::{
    Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam, Method,
};

/// JSON-RPC Request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    method: String,
    #[serde(default)]
    params: Option<Value>,
    id: Option<Value>,
}

/// JSON-RPC Response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

/// HTTP Handler for MCP
struct McpHttpHandler;

impl McpHttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        let method = request.method();
        let headers = Fields::new();
        let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Origin".to_string(), &[b"*".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Methods".to_string(), &[b"GET, POST, OPTIONS".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Headers".to_string(), &[b"Content-Type".to_vec()]);

        if matches!(method, Method::Options) {
            Self::send_response(response_out, 204, headers, b"");
            return;
        }
        if !matches!(method, Method::Post) {
            Self::send_response(response_out, 405, headers, b"Method Not Allowed");
            return;
        }

        let body = Self::read_body(&request);
        let response = match serde_json::from_slice::<JsonRpcRequest>(&body) {
            Ok(req) => Self::handle_jsonrpc(req),
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError { code: -32700, message: format!("Parse error: {}", e) }),
                id: Value::Null,
            },
        };
        let response_body = serde_json::to_vec(&response).unwrap_or_default();
        Self::send_response(response_out, 200, headers, &response_body);
    }

    fn handle_jsonrpc(req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);
        match req.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": { "name": "wasmmcp-fetch-http", "version": "0.1.0" },
                    "capabilities": { "tools": {} }
                })),
                error: None,
                id,
            },
            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: Some(json!({ "tools": Self::get_tool_list() })),
                error: None,
                id,
            },
            "tools/call" => {
                let params = req.params.unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_args = params.get("arguments").cloned().unwrap_or(json!({}));
                match Self::call_tool(tool_name, tool_args) {
                    Ok(result) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({ "content": [{ "type": "text", "text": result }] })),
                        error: None, id,
                    },
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({ "content": [{ "type": "text", "text": e }], "isError": true })),
                        error: None, id,
                    },
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError { code: -32601, message: format!("Method not found: {}", req.method) }),
                id,
            },
        }
    }

    fn get_tool_list() -> Vec<Value> {
        vec![
            json!({
                "name": "fetch",
                "description": "Fetch a URL and return its content as markdown",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum content length to return (default 50000)"
                        }
                    },
                    "required": ["url"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "fetch" => {
                let url = args.get("url")
                    .and_then(|v| v.as_str())
                    .ok_or("url is required")?;
                let max_length = args.get("max_length")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(50000) as usize;

                Self::fetch_url(url, max_length)
            }
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    fn fetch_url(url_str: &str, max_length: usize) -> Result<String, String> {
        use wasi::http::types::{
            OutgoingRequest, Scheme,
        };
        use wasi::http::outgoing_handler;

        let parsed_url = Url::parse(url_str)
            .map_err(|e| format!("Invalid URL: {}", e))?;

        let scheme = match parsed_url.scheme() {
            "http" => Scheme::Http,
            "https" => Scheme::Https,
            s => return Err(format!("Unsupported scheme: {}", s)),
        };

        let host = parsed_url.host_str()
            .ok_or_else(|| "URL has no host".to_string())?;

        let path_and_query = if let Some(query) = parsed_url.query() {
            format!("{}?{}", parsed_url.path(), query)
        } else {
            parsed_url.path().to_string()
        };

        // Create headers (matching Python fetch_server.py)
        let headers = Fields::new();
        let _ = headers.set(&"User-Agent".to_string(),
            &[b"Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0".to_vec()]);
        let _ = headers.set(&"Accept".to_string(),
            &[b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_vec()]);

        // Create outgoing request
        let request = OutgoingRequest::new(headers);
        request.set_method(&Method::Get).map_err(|_| "Failed to set method")?;
        request.set_scheme(Some(&scheme)).map_err(|_| "Failed to set scheme")?;
        request.set_authority(Some(host)).map_err(|_| "Failed to set authority")?;
        request.set_path_with_query(Some(&path_and_query)).map_err(|_| "Failed to set path")?;

        // Send request
        let future_response = outgoing_handler::handle(request, None)
            .map_err(|e| format!("Failed to send request: {:?}", e))?;

        // Wait for response
        let response = loop {
            if let Some(result) = future_response.get() {
                break result
                    .map_err(|_| "Response error".to_string())?
                    .map_err(|e| format!("HTTP error: {:?}", e))?;
            }
            future_response.subscribe().block();
        };

        let status = response.status();

        // Handle HTTP 202 (matching Python behavior)
        if status == 202 {
            return Err(format!("HTTP 202 Accepted - Content not ready. This URL ({}) requires server-side processing.", url_str));
        }

        if status >= 400 {
            return Err(format!("HTTP Error {}", status));
        }

        // Read response body
        let body = response.consume().map_err(|_| "Failed to consume response body")?;
        let stream = body.stream().map_err(|_| "Failed to get body stream")?;

        let mut content = Vec::new();
        loop {
            match stream.blocking_read(65536) {
                Ok(chunk) if !chunk.is_empty() => content.extend_from_slice(&chunk),
                _ => break,
            }
        }

        let content_str = String::from_utf8_lossy(&content).to_string();

        // Convert HTML to markdown (matching Python fetch_server behavior)
        let processed = if content_str.trim_start().starts_with("<!") ||
                          content_str.trim_start().starts_with("<html") ||
                          content_str.contains("<head") {
            Self::html_to_markdown(&content_str)
        } else {
            content_str
        };

        // Truncate if needed (matching Python behavior)
        let result = if processed.len() > max_length {
            format!("{}\n\n[Content truncated...]", &processed[..max_length])
        } else {
            processed
        };

        // Return plain text (matching Python fetch_server output format)
        Ok(result)
    }

    fn html_to_markdown(html: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        let mut current_tag = String::new();
        let mut skip_content = false;

        for c in html.chars() {
            if c == '<' {
                in_tag = true;
                current_tag.clear();
            } else if c == '>' {
                in_tag = false;
                let tag_lower = current_tag.to_lowercase();

                if tag_lower.starts_with("script") || tag_lower.starts_with("style") {
                    skip_content = true;
                } else if tag_lower.starts_with("/script") || tag_lower.starts_with("/style") {
                    skip_content = false;
                }

                if !skip_content {
                    if tag_lower.starts_with("h1") {
                        result.push_str("\n# ");
                    } else if tag_lower.starts_with("h2") {
                        result.push_str("\n## ");
                    } else if tag_lower.starts_with("h3") {
                        result.push_str("\n### ");
                    } else if tag_lower.starts_with("p") || tag_lower.starts_with("/p") {
                        result.push_str("\n\n");
                    } else if tag_lower.starts_with("br") {
                        result.push('\n');
                    } else if tag_lower.starts_with("li") {
                        result.push_str("\n- ");
                    } else if tag_lower.starts_with("/h") {
                        result.push('\n');
                    }
                }
            } else if in_tag {
                current_tag.push(c);
            } else if !skip_content {
                result.push(c);
            }
        }

        result
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn read_body(request: &IncomingRequest) -> Vec<u8> {
        let mut body = Vec::new();
        if let Some(incoming) = request.consume().ok() {
            if let Ok(stream) = incoming.stream() {
                loop {
                    match stream.blocking_read(4096) {
                        Ok(chunk) if !chunk.is_empty() => body.extend_from_slice(&chunk),
                        _ => break,
                    }
                }
            }
        }
        body
    }

    fn send_response(out: ResponseOutparam, status: u16, headers: Fields, body: &[u8]) {
        let resp = OutgoingResponse::new(headers);
        let _ = resp.set_status_code(status);
        let outgoing_body = resp.body().unwrap();
        ResponseOutparam::set(out, Ok(resp));
        if !body.is_empty() {
            if let Ok(stream) = outgoing_body.write() { let _ = stream.blocking_write_and_flush(body); }
        }
        let _ = OutgoingBody::finish(outgoing_body, None);
    }
}

struct HttpHandler;
impl wasi::exports::http::incoming_handler::Guest for HttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        McpHttpHandler::handle(request, response_out);
    }
}
wasi::http::proxy::export!(HttpHandler);
