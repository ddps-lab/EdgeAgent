//! HTTP entry point for Time MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve target/wasm32-wasip2/release/mcp_server_time_http.wasm

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use wasi::http::types::{
    Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam, Method,
};

/// JSON-RPC Request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
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

/// Parse timezone string to Tz
fn parse_timezone(tz_str: &str) -> Result<Tz, String> {
    tz_str.parse::<Tz>().map_err(|_| {
        format!("Invalid timezone: '{}'. Use IANA timezone names like 'America/New_York', 'Asia/Seoul', 'UTC'", tz_str)
    })
}

/// Format datetime with timezone info
fn format_datetime<T: TimeZone>(dt: DateTime<T>) -> String
where
    T::Offset: std::fmt::Display
{
    dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

/// HTTP Handler for MCP
struct McpHttpHandler;

impl McpHttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        let method = request.method();

        // CORS headers
        let headers = Fields::new();
        let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Origin".to_string(), &[b"*".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Methods".to_string(), &[b"GET, POST, OPTIONS".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Headers".to_string(), &[b"Content-Type".to_vec()]);

        // Handle CORS preflight
        if matches!(method, Method::Options) {
            Self::send_response(response_out, 204, headers, b"");
            return;
        }

        // Only handle POST
        if !matches!(method, Method::Post) {
            Self::send_response(response_out, 405, headers, b"Method Not Allowed");
            return;
        }

        // Read request body
        let body = Self::read_body(&request);

        // Parse JSON-RPC request
        let response = match serde_json::from_slice::<JsonRpcRequest>(&body) {
            Ok(req) => Self::handle_jsonrpc(req),
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: format!("Parse error: {}", e),
                }),
                id: Value::Null,
            },
        };

        let response_body = serde_json::to_vec(&response).unwrap_or_default();
        Self::send_response(response_out, 200, headers, &response_body);
    }

    fn handle_jsonrpc(req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);

        match req.method.as_str() {
            "initialize" => {
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    result: Some(json!({
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "wasmmcp-time-http",
                            "version": "1.0.0"
                        },
                        "capabilities": {
                            "tools": {}
                        }
                    })),
                    error: None,
                    id,
                }
            }

            "tools/list" => {
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    result: Some(json!({ "tools": Self::get_tool_list() })),
                    error: None,
                    id,
                }
            }

            "tools/call" => {
                let params = req.params.unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_args = params.get("arguments").cloned().unwrap_or(json!({}));

                match Self::call_tool(tool_name, tool_args) {
                    Ok(result) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": result
                            }]
                        })),
                        error: None,
                        id,
                    },
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": e
                            }],
                            "isError": true
                        })),
                        error: None,
                        id,
                    },
                }
            }

            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", req.method),
                }),
                id,
            },
        }
    }

    fn get_tool_list() -> Vec<Value> {
        vec![
            json!({
                "name": "get_current_time",
                "description": "Get the current time in a specific timezone. Returns ISO 8601 formatted time with timezone offset.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone name (e.g., 'America/New_York', 'Asia/Seoul', 'UTC'). Defaults to UTC if not provided."
                        }
                    }
                }
            }),
            json!({
                "name": "convert_time",
                "description": "Convert a datetime from one timezone to another. Input time should be in ISO 8601 format.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Time string in ISO 8601 format (e.g., '2025-01-01T00:00:00')"
                        },
                        "sourceTimezone": {
                            "type": "string",
                            "description": "Source timezone name (e.g., 'UTC', 'America/New_York')"
                        },
                        "targetTimezone": {
                            "type": "string",
                            "description": "Target timezone name (e.g., 'Asia/Seoul', 'Europe/London')"
                        }
                    },
                    "required": ["time", "sourceTimezone", "targetTimezone"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "get_current_time" => {
                let tz_str = args.get("timezone")
                    .and_then(|v| v.as_str())
                    .unwrap_or("UTC");
                let tz = parse_timezone(tz_str)?;

                let now_utc = Utc::now();
                let now_local = now_utc.with_timezone(&tz);

                Ok(json!({
                    "timezone": tz_str,
                    "datetime": format_datetime(now_local),
                    "is_dst": false
                }).to_string())
            }

            "convert_time" => {
                let time_str = args.get("time")
                    .and_then(|v| v.as_str())
                    .ok_or("time parameter is required")?;
                let source_tz_str = args.get("sourceTimezone")
                    .and_then(|v| v.as_str())
                    .ok_or("sourceTimezone parameter is required")?;
                let target_tz_str = args.get("targetTimezone")
                    .and_then(|v| v.as_str())
                    .ok_or("targetTimezone parameter is required")?;

                let source_tz = parse_timezone(source_tz_str)?;
                let target_tz = parse_timezone(target_tz_str)?;

                // Parse the input time
                let naive = NaiveDateTime::parse_from_str(time_str, "%Y-%m-%dT%H:%M:%S")
                    .or_else(|_| NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S"))
                    .map_err(|e| format!("Invalid time format: '{}'. Expected ISO 8601 format like '2025-01-01T00:00:00'. Error: {}", time_str, e))?;

                // Create datetime in source timezone
                let source_dt = source_tz.from_local_datetime(&naive)
                    .single()
                    .ok_or_else(|| format!("Ambiguous or invalid time '{}' in timezone '{}'", time_str, source_tz_str))?;

                // Convert to target timezone
                let target_dt = source_dt.with_timezone(&target_tz);

                Ok(json!({
                    "source": {
                        "timezone": source_tz_str,
                        "datetime": format_datetime(source_dt)
                    },
                    "target": {
                        "timezone": target_tz_str,
                        "datetime": format_datetime(target_dt)
                    }
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
            if let Ok(stream) = outgoing_body.write() {
                let _ = stream.blocking_write_and_flush(body);
            }
        }

        let _ = OutgoingBody::finish(outgoing_body, None);
    }
}

// WASI HTTP Handler export
struct HttpHandler;

impl wasi::exports::http::incoming_handler::Guest for HttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        McpHttpHandler::handle(request, response_out);
    }
}

wasi::http::proxy::export!(HttpHandler);
