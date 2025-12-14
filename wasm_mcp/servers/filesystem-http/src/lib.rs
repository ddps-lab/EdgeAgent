//! HTTP entry point for Filesystem MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve --dir=/path target/wasm32-wasip2/release/mcp_server_filesystem_http.wasm

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
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
                            "name": "wasmmcp-filesystem-http",
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
            json!({"name": "read_file", "description": "Read a file (deprecated)", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}),
            json!({"name": "read_text_file", "description": "Read a text file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "head": {"type": "integer"}, "tail": {"type": "integer"}}, "required": ["path"]}}),
            json!({"name": "read_media_file", "description": "Read media file as base64", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}),
            json!({"name": "read_multiple_files", "description": "Read multiple files", "inputSchema": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}}),
            json!({"name": "write_file", "description": "Write content to file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}),
            json!({"name": "edit_file", "description": "Edit file with replacements", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["oldText", "newText"]}}, "dryRun": {"type": "boolean", "default": false}}, "required": ["path", "edits"]}}),
            json!({"name": "create_directory", "description": "Create directory", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}),
            json!({"name": "list_directory", "description": "List directory contents", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}),
            json!({"name": "list_directory_with_sizes", "description": "List directory with sizes", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "sortBy": {"type": "string", "enum": ["name", "size"], "default": "name"}}, "required": ["path"]}}),
            json!({"name": "directory_tree", "description": "Get directory tree as JSON", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "excludePatterns": {"type": "array", "items": {"type": "string"}, "default": []}}, "required": ["path"]}}),
            json!({"name": "move_file", "description": "Move/rename file", "inputSchema": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}),
            json!({"name": "search_files", "description": "Search files by pattern", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}, "excludePatterns": {"type": "array", "items": {"type": "string"}, "default": []}}, "required": ["path", "pattern"]}}),
            json!({"name": "get_file_info", "description": "Get file metadata", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}),
            json!({"name": "list_allowed_directories", "description": "List allowed directories", "inputSchema": {"type": "object", "properties": {}}}),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "read_file" | "read_text_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let content = fs::read_to_string(path).map_err(|e| e.to_string())?;

                let head = args.get("head").and_then(|v| v.as_u64()).map(|n| n as usize);
                let tail = args.get("tail").and_then(|v| v.as_u64()).map(|n| n as usize);

                if let Some(n) = head {
                    Ok(content.lines().take(n).collect::<Vec<_>>().join("\n"))
                } else if let Some(n) = tail {
                    let lines: Vec<&str> = content.lines().collect();
                    let start = lines.len().saturating_sub(n);
                    Ok(lines[start..].join("\n"))
                } else {
                    Ok(content)
                }
            }

            "read_media_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let data = fs::read(path).map_err(|e| e.to_string())?;
                let b64 = base64_encode(&data);
                let mime = get_mime_type(path);
                Ok(format!("data:{};base64,{}", mime, b64))
            }

            "read_multiple_files" => {
                let paths = args.get("paths")
                    .and_then(|v| v.as_array())
                    .ok_or("paths required")?;
                let results: Vec<String> = paths.iter()
                    .filter_map(|p| p.as_str())
                    .map(|path| {
                        match fs::read_to_string(path) {
                            Ok(content) => format!("{}:\n{}", path, content),
                            Err(e) => format!("{}: Error - {}", path, e),
                        }
                    })
                    .collect();
                Ok(results.join("\n---\n"))
            }

            "write_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let content = args.get("content").and_then(|v| v.as_str()).ok_or("content required")?;
                fs::write(path, content).map_err(|e| e.to_string())?;
                Ok(format!("Successfully wrote to {}", path))
            }

            "edit_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let edits = args.get("edits").and_then(|v| v.as_array()).ok_or("edits required")?;
                let dry_run = args.get("dryRun").and_then(|v| v.as_bool()).unwrap_or(false);

                let mut content = fs::read_to_string(path).map_err(|e| e.to_string())?;
                let mut changes = Vec::new();

                for edit in edits {
                    // Support both camelCase (Node.js compatible) and snake_case
                    let old_text = edit.get("oldText").or_else(|| edit.get("old_text"))
                        .and_then(|v| v.as_str()).ok_or("oldText required")?;
                    let new_text = edit.get("newText").or_else(|| edit.get("new_text"))
                        .and_then(|v| v.as_str()).ok_or("newText required")?;
                    if content.contains(old_text) {
                        content = content.replace(old_text, new_text);
                        changes.push(format!("- {}\n+ {}", old_text, new_text));
                    } else {
                        return Err(format!("Text not found: '{}'", old_text));
                    }
                }

                if dry_run {
                    Ok(format!("Dry run - changes:\n{}", changes.join("\n")))
                } else {
                    fs::write(path, &content).map_err(|e| e.to_string())?;
                    Ok(format!("Applied {} edit(s) to {}", edits.len(), path))
                }
            }

            "create_directory" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                fs::create_dir_all(path).map_err(|e| e.to_string())?;
                Ok(format!("Created directory {}", path))
            }

            "list_directory" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let entries = fs::read_dir(path).map_err(|e| e.to_string())?;
                let mut items = Vec::new();
                for entry in entries.filter_map(|e| e.ok()) {
                    let p = entry.path();
                    let prefix = if p.is_dir() { "[DIR]" } else { "[FILE]" };
                    items.push(format!("{} {}", prefix, entry.file_name().to_string_lossy()));
                }
                items.sort();
                Ok(if items.is_empty() { "Directory is empty".into() } else { items.join("\n") })
            }

            "list_directory_with_sizes" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                // Support both camelCase (Node.js compatible) and snake_case
                let sort_by = args.get("sortBy").or_else(|| args.get("sort_by"))
                    .and_then(|v| v.as_str()).unwrap_or("name");
                let entries = fs::read_dir(path).map_err(|e| e.to_string())?;

                let mut items: Vec<(String, bool, u64)> = Vec::new();
                for entry in entries.filter_map(|e| e.ok()) {
                    let p = entry.path();
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = p.is_dir();
                    let size = if is_dir { 0 } else { fs::metadata(&p).map(|m| m.len()).unwrap_or(0) };
                    items.push((name, is_dir, size));
                }

                if sort_by == "size" {
                    items.sort_by(|a, b| b.2.cmp(&a.2));
                } else {
                    items.sort_by(|a, b| a.0.cmp(&b.0));
                }

                let mut result = Vec::new();
                let mut total_size: u64 = 0;
                let (mut files, mut dirs) = (0, 0);

                for (name, is_dir, size) in &items {
                    let prefix = if *is_dir { "[DIR]" } else { "[FILE]" };
                    let size_str = if *is_dir { "".into() } else { format_size(*size) };
                    result.push(format!("{} {:30} {:>10}", prefix, name, size_str));
                    if *is_dir { dirs += 1; } else { files += 1; total_size += size; }
                }

                result.push(String::new());
                result.push(format!("Total: {} files, {} directories", files, dirs));
                result.push(format!("Combined size: {}", format_size(total_size)));
                Ok(result.join("\n"))
            }

            "directory_tree" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                // Support both camelCase (Node.js compatible) and snake_case
                let exclude = args.get("excludePatterns").or_else(|| args.get("exclude_patterns"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_else(Vec::new);
                let tree = build_tree(Path::new(path), &exclude)?;
                Ok(serde_json::to_string_pretty(&tree).unwrap_or("[]".into()))
            }

            "move_file" => {
                let source = args.get("source").and_then(|v| v.as_str()).ok_or("source required")?;
                let dest = args.get("destination").and_then(|v| v.as_str()).ok_or("destination required")?;
                fs::rename(source, dest).map_err(|e| e.to_string())?;
                Ok(format!("Moved {} to {}", source, dest))
            }

            "search_files" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let pattern = args.get("pattern").and_then(|v| v.as_str()).ok_or("pattern required")?;
                // Support both camelCase (Node.js compatible) and snake_case
                let exclude = args.get("excludePatterns").or_else(|| args.get("exclude_patterns"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_else(Vec::new);
                let mut results = Vec::new();
                search_recursive(Path::new(path), pattern, &exclude, &mut results);
                Ok(if results.is_empty() { "No matches found".into() } else { results.join("\n") })
            }

            "get_file_info" => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or("path required")?;
                let meta = fs::metadata(path).map_err(|e| e.to_string())?;
                let file_type = if meta.is_dir() { "directory" } else if meta.is_file() { "file" } else { "other" };
                Ok(format!("size: {}\ntype: {}\nreadonly: {}", meta.len(), file_type, meta.permissions().readonly()))
            }

            "list_allowed_directories" => {
                Ok("Allowed directories are controlled by WASI runtime via --dir flag.".into())
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
            let stream = outgoing_body.write().unwrap();
            let _ = stream.blocking_write_and_flush(body);
            drop(stream);
        }
        let _ = OutgoingBody::finish(outgoing_body, None);
    }
}

// Helper functions
fn format_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if size >= GB { format!("{:.2} GB", size as f64 / GB as f64) }
    else if size >= MB { format!("{:.2} MB", size as f64 / MB as f64) }
    else if size >= KB { format!("{:.2} KB", size as f64 / KB as f64) }
    else { format!("{} B", size) }
}

fn get_mime_type(path: &str) -> &'static str {
    let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "png" => "image/png", "jpg" | "jpeg" => "image/jpeg", "gif" => "image/gif",
        "webp" => "image/webp", "svg" => "image/svg+xml", "mp3" => "audio/mpeg",
        "wav" => "audio/wav", _ => "application/octet-stream",
    }
}

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;
        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);
        result.push(if chunk.len() > 1 { ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char } else { '=' });
        result.push(if chunk.len() > 2 { ALPHABET[b2 & 0x3f] as char } else { '=' });
    }
    result
}

fn build_tree(path: &Path, exclude: &[String]) -> Result<Value, String> {
    let entries = fs::read_dir(path).map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if exclude.iter().any(|p| name.contains(p)) { continue; }
        let p = entry.path();
        let is_dir = p.is_dir();
        let mut node = json!({"name": name, "type": if is_dir { "directory" } else { "file" }});
        if is_dir { node["children"] = build_tree(&p, exclude)?; }
        result.push(node);
    }
    Ok(Value::Array(result))
}

fn search_recursive(dir: &Path, pattern: &str, exclude: &[String], results: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if exclude.iter().any(|p| name.contains(p)) { continue; }
            let path = entry.path();
            let matches = if pattern.starts_with("*.") { name.ends_with(&pattern[1..]) }
                else if pattern.starts_with("**/*.") { name.ends_with(&pattern[4..]) }
                else { name.contains(pattern) };
            if matches { results.push(path.to_string_lossy().to_string()); }
            if path.is_dir() { search_recursive(&path, pattern, exclude, results); }
        }
    }
}

// HTTP handler export for wasi:http/proxy world
struct HttpHandler;

impl wasi::exports::http::incoming_handler::Guest for HttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        McpHttpHandler::handle(request, response_out);
    }
}

wasi::http::proxy::export!(HttpHandler);
