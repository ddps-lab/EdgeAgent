//! HTTP entry point for Log Parser MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve target/wasm32-wasip2/release/mcp_server_log_parser_http.wasm

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
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

// Severity levels
fn severity_level(level: &str) -> i32 {
    match level.to_lowercase().as_str() {
        "debug" => 0, "info" => 1, "notice" => 2,
        "warning" | "warn" => 3, "error" | "err" => 4,
        "critical" | "crit" => 5, "alert" => 6, "emergency" | "emerg" => 7,
        _ => 1,
    }
}

// Log format patterns
fn get_log_patterns() -> Vec<(&'static str, Regex)> {
    vec![
        ("apache_combined", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("apache_common", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)"#).unwrap()),
        ("nginx", Regex::new(r#"(?P<ip>\S+) - \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("syslog", Regex::new(r"(?P<time>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(?P<host>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)").unwrap()),
        ("python", Regex::new(r"(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<logger>\S+)\s+-\s+(?P<level>\S+)\s+-\s+(?P<message>.*)").unwrap()),
    ]
}

fn detect_format(lines: &[&str]) -> String {
    let patterns = get_log_patterns();
    for line in lines.iter().take(10) {
        let line = line.trim();
        if line.is_empty() { continue; }
        if serde_json::from_str::<Value>(line).is_ok() { return "json".to_string(); }
        for (name, pattern) in &patterns {
            if pattern.is_match(line) { return name.to_string(); }
        }
    }
    "unknown".to_string()
}

fn parse_line(line: &str, format: &str) -> Option<Value> {
    let line = line.trim();
    if line.is_empty() { return None; }
    if format == "json" { return serde_json::from_str(line).ok(); }
    let patterns = get_log_patterns();
    for (name, pattern) in &patterns {
        if *name == format {
            if let Some(caps) = pattern.captures(line) {
                let mut entry = serde_json::Map::new();
                for name in pattern.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        entry.insert(name.to_string(), Value::String(m.as_str().to_string()));
                    }
                }
                return Some(Value::Object(entry));
            }
        }
    }
    Some(json!({"raw": line, "parsed": false}))
}

fn extract_level(entry: &Value) -> String {
    for field in &["level", "severity", "loglevel", "log_level"] {
        if let Some(val) = entry.get(field).and_then(|v| v.as_str()) {
            return val.to_lowercase();
        }
    }
    if let Some(status) = entry.get("status").and_then(|v| v.as_str()).and_then(|s| s.parse::<i32>().ok()) {
        if status >= 500 { return "error".to_string(); }
        if status >= 400 { return "warning".to_string(); }
        return "info".to_string();
    }
    let message = entry.get("message").or_else(|| entry.get("raw")).and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
    for level in &["error", "warning", "warn", "critical", "crit", "debug", "info"] {
        if message.contains(level) { return level.to_string(); }
    }
    "info".to_string()
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
                    "serverInfo": { "name": "wasmmcp-log-parser-http", "version": "1.0.0" },
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
            json!({"name": "parse_logs", "description": "Parse raw log content into structured entries", "inputSchema": {"type": "object", "properties": {"logContent": {"type": "string"}, "formatType": {"type": "string", "default": "auto"}, "maxEntries": {"type": "integer", "default": 1000}}, "required": ["logContent"]}}),
            json!({"name": "filter_entries", "description": "Filter log entries by severity level", "inputSchema": {"type": "object", "properties": {"entries": {"type": "array"}, "minLevel": {"type": "string", "default": "warning"}, "includeLevels": {"type": "array"}}, "required": ["entries"]}}),
            json!({"name": "compute_log_statistics", "description": "Compute statistics from log entries", "inputSchema": {"type": "object", "properties": {"entries": {"type": "array"}}, "required": ["entries"]}}),
            json!({"name": "search_entries", "description": "Search log entries by regex pattern", "inputSchema": {"type": "object", "properties": {"entries": {"type": "array"}, "pattern": {"type": "string"}, "fields": {"type": "array"}, "caseSensitive": {"type": "boolean", "default": false}}, "required": ["entries", "pattern"]}}),
            json!({"name": "extract_time_range", "description": "Extract time range from log entries", "inputSchema": {"type": "object", "properties": {"entries": {"type": "array"}}, "required": ["entries"]}}),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "parse_logs" => {
                let log_content = args.get("logContent").and_then(|v| v.as_str()).ok_or("logContent required")?;
                let format_type = args.get("formatType").and_then(|v| v.as_str()).unwrap_or("auto");
                let max_entries = args.get("maxEntries").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;

                let lines: Vec<&str> = log_content.lines().collect();
                let format = if format_type == "auto" { detect_format(&lines) } else { format_type.to_string() };

                let mut entries = Vec::new();
                let mut errors = 0;
                for line in lines.iter().take(max_entries) {
                    if let Some(mut entry) = parse_line(line, &format) {
                        let level = extract_level(&entry);
                        if let Value::Object(ref mut map) = entry {
                            map.insert("_level".to_string(), Value::String(level));
                        }
                        entries.push(entry);
                    } else if !line.trim().is_empty() { errors += 1; }
                }
                Ok(json!({"format_detected": format, "total_lines": lines.len(), "parsed_count": entries.len(), "error_count": errors, "entries": entries}).to_string())
            }

            "filter_entries" => {
                let entries = args.get("entries").and_then(|v| v.as_array()).ok_or("entries required")?;
                let min_level = args.get("minLevel").and_then(|v| v.as_str()).unwrap_or("warning");
                let include_levels: Option<Vec<String>> = args.get("includeLevels").and_then(|v| v.as_array()).map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());

                let mut filtered = Vec::new();
                let mut level_counts: HashMap<String, i32> = HashMap::new();
                for entry in entries {
                    let level = entry.get("_level").and_then(|v| v.as_str()).unwrap_or("info").to_string();
                    let include = if let Some(ref levels) = include_levels {
                        levels.iter().any(|l| l.to_lowercase() == level)
                    } else {
                        severity_level(&level) >= severity_level(min_level)
                    };
                    if include {
                        *level_counts.entry(level).or_insert(0) += 1;
                        filtered.push(entry.clone());
                    }
                }
                Ok(json!({"original_count": entries.len(), "filtered_count": filtered.len(), "levels_included": level_counts.keys().collect::<Vec<_>>(), "by_level": level_counts, "entries": filtered}).to_string())
            }

            "compute_log_statistics" => {
                let entries = args.get("entries").and_then(|v| v.as_array()).ok_or("entries required")?;
                if entries.is_empty() { return Ok(json!({"entry_count": 0, "by_level": {}}).to_string()); }

                let mut level_counts: HashMap<String, i32> = HashMap::new();
                let mut status_counts: HashMap<String, i32> = HashMap::new();
                let mut ip_counts: HashMap<String, i32> = HashMap::new();
                let mut path_counts: HashMap<String, i32> = HashMap::new();

                for entry in entries {
                    let level = entry.get("_level").and_then(|v| v.as_str()).unwrap_or("unknown");
                    *level_counts.entry(level.to_string()).or_insert(0) += 1;
                    if let Some(status) = entry.get("status").and_then(|v| v.as_str()) { *status_counts.entry(status.to_string()).or_insert(0) += 1; }
                    if let Some(ip) = entry.get("ip").and_then(|v| v.as_str()) { *ip_counts.entry(ip.to_string()).or_insert(0) += 1; }
                    if let Some(path) = entry.get("path").and_then(|v| v.as_str()) { *path_counts.entry(path.to_string()).or_insert(0) += 1; }
                }

                let mut result = json!({"entry_count": entries.len(), "by_level": level_counts});
                if !status_counts.is_empty() { result["by_status"] = json!(status_counts); }
                if !ip_counts.is_empty() { result["top_ips"] = json!(ip_counts); }
                if !path_counts.is_empty() { result["top_paths"] = json!(path_counts); }
                Ok(result.to_string())
            }

            "search_entries" => {
                let entries = args.get("entries").and_then(|v| v.as_array()).ok_or("entries required")?;
                let pattern = args.get("pattern").and_then(|v| v.as_str()).ok_or("pattern required")?;
                let fields: Vec<String> = args.get("fields").and_then(|v| v.as_array()).map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_else(|| vec!["message".to_string(), "raw".to_string()]);
                let case_sensitive = args.get("caseSensitive").and_then(|v| v.as_bool()).unwrap_or(false);

                let regex = if case_sensitive { Regex::new(pattern) } else { Regex::new(&format!("(?i){}", pattern)) }.map_err(|e| format!("Invalid regex: {}", e))?;

                let mut matches = Vec::new();
                for entry in entries {
                    for field in &fields {
                        if let Some(text) = entry.get(field).and_then(|v| v.as_str()) {
                            if regex.is_match(text) {
                                matches.push(json!({"entry": entry, "matched_field": field, "matched_text": &text[..text.len().min(200)]}));
                                break;
                            }
                        }
                    }
                    if matches.len() >= 100 { break; }
                }
                Ok(json!({"search_pattern": pattern, "fields_searched": fields, "total_entries": entries.len(), "match_count": matches.len(), "matches": matches}).to_string())
            }

            "extract_time_range" => {
                let entries = args.get("entries").and_then(|v| v.as_array()).ok_or("entries required")?;
                let mut times = Vec::new();
                for entry in entries {
                    if let Some(time) = entry.get("time").or_else(|| entry.get("timestamp")).and_then(|v| v.as_str()) {
                        times.push(time.to_string());
                    }
                }
                if times.is_empty() { return Ok(json!({"has_timestamps": false, "entry_count": entries.len()}).to_string()); }
                Ok(json!({"has_timestamps": true, "entry_count": entries.len(), "first_timestamp": times.first(), "last_timestamp": times.last(), "sample_timestamps": times.iter().take(5).collect::<Vec<_>>()}).to_string())
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
