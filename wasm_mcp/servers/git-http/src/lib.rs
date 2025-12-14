//! HTTP entry point for Git MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve --dir /path/to/repo target/wasm32-wasip2/release/mcp_server_git_http.wasm

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use wasi::http::types::{
    Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam, Method,
};
use flate2::read::ZlibDecoder;

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
                    "serverInfo": { "name": "wasmmcp-git-http", "version": "0.1.0" },
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
                "name": "git_status",
                "description": "Shows the working tree status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": { "type": "string", "description": "Path to the git repository" }
                    },
                    "required": ["repo_path"]
                }
            }),
            json!({
                "name": "git_log",
                "description": "Shows the commit logs",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": { "type": "string", "description": "Path to the git repository" },
                        "max_count": { "type": "integer", "description": "Maximum number of commits to show" }
                    },
                    "required": ["repo_path"]
                }
            }),
            json!({
                "name": "git_show",
                "description": "Shows the contents of a commit",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": { "type": "string", "description": "Path to the git repository" },
                        "revision": { "type": "string", "description": "Commit SHA or reference" }
                    },
                    "required": ["repo_path", "revision"]
                }
            }),
            json!({
                "name": "git_branch",
                "description": "List Git branches",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": { "type": "string", "description": "Path to the git repository" },
                        "branch_type": { "type": "string", "description": "Branch type: local, remote, or all" }
                    },
                    "required": ["repo_path"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        let repo_path = args.get("repo_path")
            .and_then(|v| v.as_str())
            .ok_or("repo_path is required")?;

        match name {
            "git_status" => Self::git_status(repo_path),
            "git_log" => {
                let max_count = args.get("max_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                Self::git_log(repo_path, max_count)
            }
            "git_show" => {
                let revision = args.get("revision")
                    .and_then(|v| v.as_str())
                    .ok_or("revision is required")?;
                Self::git_show(repo_path, revision)
            }
            "git_branch" => {
                let branch_type = args.get("branch_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local");
                Self::git_branch(repo_path, branch_type)
            }
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // Git helper functions
    fn git_dir(repo_path: &str) -> Result<PathBuf, String> {
        let path = Path::new(repo_path);
        let git_dir = path.join(".git");
        if git_dir.exists() {
            Ok(git_dir)
        } else if path.join("HEAD").exists() {
            Ok(path.to_path_buf())
        } else {
            Err(format!("Not a git repository: {}", repo_path))
        }
    }

    fn read_head(git_dir: &Path) -> Result<String, String> {
        let head_path = git_dir.join("HEAD");
        fs::read_to_string(&head_path)
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("Failed to read HEAD: {}", e))
    }

    fn resolve_ref(git_dir: &Path, ref_str: &str) -> Result<String, String> {
        if ref_str.starts_with("ref: ") {
            let ref_name = &ref_str[5..];
            let ref_path = git_dir.join(ref_name);
            fs::read_to_string(&ref_path)
                .map(|s| s.trim().to_string())
                .map_err(|e| format!("Failed to resolve ref {}: {}", ref_name, e))
        } else {
            Ok(ref_str.to_string())
        }
    }

    fn read_object(git_dir: &Path, sha: &str) -> Result<Vec<u8>, String> {
        if sha.len() < 4 {
            return Err("SHA too short".to_string());
        }

        let loose_path = git_dir.join("objects").join(&sha[..2]).join(&sha[2..]);
        if loose_path.exists() {
            let compressed = fs::read(&loose_path)
                .map_err(|e| format!("Failed to read object {}: {}", sha, e))?;
            let mut decoder = ZlibDecoder::new(&compressed[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| format!("Failed to decompress object: {}", e))?;
            return Ok(decompressed);
        }

        Err(format!("Object {} not found in loose objects", sha))
    }

    fn parse_commit(data: &[u8]) -> Result<(String, Vec<String>, String, String), String> {
        let content = data.iter()
            .position(|&b| b == 0)
            .map(|pos| &data[pos + 1..])
            .ok_or("Invalid object format")?;

        let content_str = String::from_utf8_lossy(content);

        let mut parents = Vec::new();
        let mut author = String::new();
        let mut message_lines = Vec::new();
        let mut in_message = false;

        for line in content_str.lines() {
            if in_message {
                message_lines.push(line);
            } else if line.is_empty() {
                in_message = true;
            } else if let Some(rest) = line.strip_prefix("parent ") {
                parents.push(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("author ") {
                if let Some(email_end) = rest.find('>') {
                    author = rest[..=email_end].to_string();
                }
            }
        }

        Ok((message_lines.join("\n"), parents, author, String::new()))
    }

    fn git_status(repo_path: &str) -> Result<String, String> {
        let git_dir = Self::git_dir(repo_path)?;
        let head = Self::read_head(&git_dir)?;

        let mut result = serde_json::Map::new();

        if head.starts_with("ref: refs/heads/") {
            let branch = head.strip_prefix("ref: refs/heads/").unwrap();
            result.insert("branch".to_string(), Value::String(branch.to_string()));
        } else {
            result.insert("detached_head".to_string(), Value::String(head.clone()));
        }

        let head_sha = Self::resolve_ref(&git_dir, &head)?;
        result.insert("head_sha".to_string(), Value::String(head_sha));
        result.insert("working_directory".to_string(), Value::String(repo_path.to_string()));

        serde_json::to_string_pretty(&result)
            .map_err(|e| format!("JSON error: {}", e))
    }

    fn git_log(repo_path: &str, max_count: usize) -> Result<String, String> {
        let git_dir = Self::git_dir(repo_path)?;
        let head = Self::read_head(&git_dir)?;
        let mut current_sha = Self::resolve_ref(&git_dir, &head)?;

        let mut commits = Vec::new();
        let mut count = 0;

        while count < max_count && !current_sha.is_empty() {
            match Self::read_object(&git_dir, &current_sha) {
                Ok(data) => {
                    if let Ok((message, parents, author, _)) = Self::parse_commit(&data) {
                        let title = message.lines().next().unwrap_or("").to_string();

                        commits.push(json!({
                            "sha": current_sha,
                            "message": title,
                            "author": author,
                            "parents": parents.len()
                        }));

                        current_sha = parents.first().cloned().unwrap_or_default();
                        count += 1;
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        serde_json::to_string_pretty(&json!({
            "commits": commits,
            "count": count
        })).map_err(|e| format!("JSON error: {}", e))
    }

    fn git_show(repo_path: &str, revision: &str) -> Result<String, String> {
        let git_dir = Self::git_dir(repo_path)?;

        let sha = if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
            revision.to_string()
        } else if revision == "HEAD" {
            let head = Self::read_head(&git_dir)?;
            Self::resolve_ref(&git_dir, &head)?
        } else {
            let ref_path = git_dir.join("refs/heads").join(revision);
            if ref_path.exists() {
                fs::read_to_string(&ref_path)
                    .map(|s| s.trim().to_string())
                    .map_err(|e| format!("Failed to read ref: {}", e))?
            } else {
                return Err(format!("Cannot resolve revision: {}", revision));
            }
        };

        let data = Self::read_object(&git_dir, &sha)?;
        let (message, parents, author, _) = Self::parse_commit(&data)?;

        serde_json::to_string_pretty(&json!({
            "sha": sha,
            "parents": parents,
            "author": author,
            "message": message
        })).map_err(|e| format!("JSON error: {}", e))
    }

    fn git_branch(repo_path: &str, branch_type: &str) -> Result<String, String> {
        let git_dir = Self::git_dir(repo_path)?;

        let head = Self::read_head(&git_dir)?;
        let current_branch = if head.starts_with("ref: refs/heads/") {
            Some(head.strip_prefix("ref: refs/heads/").unwrap().to_string())
        } else {
            None
        };

        let mut branches = Vec::new();

        if branch_type == "local" || branch_type == "all" {
            let heads_dir = git_dir.join("refs/heads");
            if heads_dir.exists() {
                Self::collect_branches(&heads_dir, "", "local", &current_branch, &mut branches)?;
            }
        }

        if branch_type == "remote" || branch_type == "all" {
            let remotes_dir = git_dir.join("refs/remotes");
            if remotes_dir.exists() {
                for entry in fs::read_dir(&remotes_dir).map_err(|e| e.to_string())? {
                    if let Ok(entry) = entry {
                        let remote_name = entry.file_name().to_string_lossy().to_string();
                        Self::collect_branches(&entry.path(), &remote_name, "remote", &None, &mut branches)?;
                    }
                }
            }
        }

        serde_json::to_string_pretty(&json!({
            "branches": branches,
            "current": current_branch,
            "branch_type": branch_type
        })).map_err(|e| format!("JSON error: {}", e))
    }

    fn collect_branches(
        dir: &Path,
        prefix: &str,
        ref_type: &str,
        current: &Option<String>,
        branches: &mut Vec<Value>,
    ) -> Result<(), String> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            if let Ok(entry) = entry {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                if path.is_dir() {
                    let new_prefix = if prefix.is_empty() { name } else { format!("{}/{}", prefix, name) };
                    Self::collect_branches(&path, &new_prefix, ref_type, current, branches)?;
                } else {
                    let branch_name = if prefix.is_empty() { name } else { format!("{}/{}", prefix, name) };
                    let is_current = current.as_ref().map(|c| c == &branch_name).unwrap_or(false);
                    let sha = fs::read_to_string(&path).map(|s| s.trim().to_string()).unwrap_or_default();

                    branches.push(json!({
                        "name": branch_name,
                        "sha": sha,
                        "is_current": is_current,
                        "type": ref_type
                    }));
                }
            }
        }

        Ok(())
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
