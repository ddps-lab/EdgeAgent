//! HTTP entry point for Sequential Thinking MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve target/wasm32-wasip2/release/mcp_server_sequential_thinking_http.wasm

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
                    "serverInfo": { "name": "wasmmcp-sequential-thinking-http", "version": "1.0.0" },
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
                "name": "sequential_thinking",
                "description": "A detailed tool for dynamic and reflective problem-solving through structured thinking. Facilitates step-by-step analysis with support for revisions, branching into alternative paths, and dynamic adjustment of the thinking process.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "The current thinking step content"
                        },
                        "nextThoughtNeeded": {
                            "type": "boolean",
                            "description": "Whether another thought step is needed after this one"
                        },
                        "thoughtNumber": {
                            "type": "integer",
                            "description": "Current thought number (1-indexed)"
                        },
                        "totalThoughts": {
                            "type": "integer",
                            "description": "Estimated total number of thoughts needed"
                        },
                        "isRevision": {
                            "type": "boolean",
                            "description": "Whether this thought revises previous thinking"
                        },
                        "revisesThought": {
                            "type": "integer",
                            "description": "Which thought number is being reconsidered"
                        },
                        "branchFromThought": {
                            "type": "integer",
                            "description": "Thought number to branch from"
                        },
                        "branchId": {
                            "type": "string",
                            "description": "Unique identifier for this reasoning branch"
                        },
                        "needsMoreThoughts": {
                            "type": "boolean",
                            "description": "Whether more thoughts are needed beyond the original estimate"
                        }
                    },
                    "required": ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "sequential_thinking" => {
                let thought = args.get("thought").and_then(|v| v.as_str()).ok_or("thought required")?;
                let next_thought_needed = args.get("nextThoughtNeeded").and_then(|v| v.as_bool()).ok_or("nextThoughtNeeded required")?;
                let thought_number = args.get("thoughtNumber").and_then(|v| v.as_i64()).ok_or("thoughtNumber required")? as i32;
                let total_thoughts = args.get("totalThoughts").and_then(|v| v.as_i64()).ok_or("totalThoughts required")? as i32;

                let is_revision = args.get("isRevision").and_then(|v| v.as_bool()).unwrap_or(false);
                let revises_thought = args.get("revisesThought").and_then(|v| v.as_i64()).map(|v| v as i32);
                let branch_from_thought = args.get("branchFromThought").and_then(|v| v.as_i64()).map(|v| v as i32);
                let branch_id = args.get("branchId").and_then(|v| v.as_str()).map(String::from);
                let needs_more_thoughts = args.get("needsMoreThoughts").and_then(|v| v.as_bool()).unwrap_or(false);

                // Validate
                if thought_number < 1 {
                    return Err("thoughtNumber must be at least 1".to_string());
                }
                if total_thoughts < 1 {
                    return Err("totalThoughts must be at least 1".to_string());
                }
                if thought.trim().is_empty() {
                    return Err("thought cannot be empty".to_string());
                }
                if is_revision && revises_thought.is_none() {
                    return Err("revisesThought is required when isRevision is true".to_string());
                }
                if branch_id.is_some() && branch_from_thought.is_none() {
                    return Err("branchFromThought is required when branchId is provided".to_string());
                }

                let is_final = !next_thought_needed;

                // Build formatted output
                let header = if is_revision {
                    format!("Thought {} (Revision of thought {})", thought_number, revises_thought.unwrap_or(0))
                } else if branch_id.is_some() {
                    format!("Thought {} [Branch: {} from thought {}]", thought_number, branch_id.as_ref().unwrap(), branch_from_thought.unwrap_or(0))
                } else {
                    format!("Thought {}/{}", thought_number, total_thoughts)
                };

                let mut formatted = format!("=== {} ===\n{}", header, thought);
                if is_final {
                    formatted.push_str("\n\n[Thinking process complete]");
                } else if needs_more_thoughts {
                    formatted.push_str(&format!("\n\n[Continuing... (estimated {} more thoughts needed)]", total_thoughts - thought_number));
                }

                Ok(json!({
                    "thoughtNumber": thought_number,
                    "totalThoughts": total_thoughts,
                    "nextThoughtNeeded": next_thought_needed,
                    "isRevision": is_revision,
                    "revisesThought": revises_thought,
                    "branchId": branch_id,
                    "branchFromThought": branch_from_thought,
                    "needsMoreThoughts": needs_more_thoughts,
                    "formattedThought": formatted,
                    "status": if is_final { "complete" } else { "in_progress" }
                }).to_string())
            }
            _ => Err(format!("Unknown tool: {}", name)),
        }
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
