//! HTTP entry point for Summarize MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve -S cli target/wasm32-wasip2/release/mcp_server_summarize_http.wasm
//!
//! Environment variables:
//!   OPENAI_API_KEY - OpenAI API key (default provider)
//!   UPSTAGE_API_KEY - Upstage API key
//!   SUMMARIZE_PROVIDER - Provider to use: "openai" (default) or "upstage"

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
                    "serverInfo": { "name": "wasmmcp-summarize-http", "version": "0.1.0" },
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
        // Match Python summarize_server.py exactly
        vec![
            json!({
                "name": "summarize_text",
                "description": "Summarize the given text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to summarize"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of summary in words (default: 150)"
                        },
                        "style": {
                            "type": "string",
                            "description": "Summary style - \"concise\", \"detailed\", or \"bullet\""
                        }
                    },
                    "required": ["text"]
                }
            }),
            json!({
                "name": "summarize_documents",
                "description": "Summarize multiple documents.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "List of documents to summarize",
                            "items": {
                                "type": "string"
                            }
                        },
                        "max_length_per_doc": {
                            "type": "integer",
                            "description": "Maximum length per summary in words"
                        },
                        "style": {
                            "type": "string",
                            "description": "Summary style for all documents"
                        }
                    },
                    "required": ["documents"]
                }
            }),
            json!({
                "name": "get_provider_info",
                "description": "Get information about the current summarization provider.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "summarize_text" => {
                let text = args.get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("text is required")?;
                // Default 150 words (matching Python)
                let max_length = args.get("max_length")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(150);
                // Default "concise" (matching Python)
                let style = args.get("style")
                    .and_then(|v| v.as_str())
                    .unwrap_or("concise");

                Self::summarize_text(text, max_length, style)
            }
            "summarize_documents" => {
                let documents = args.get("documents")
                    .and_then(|v| v.as_array())
                    .ok_or("documents array is required")?;
                // Default 100 words (matching Python)
                let max_length_per_doc = args.get("max_length_per_doc")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(100);
                let style = args.get("style")
                    .and_then(|v| v.as_str())
                    .unwrap_or("concise");

                Self::summarize_documents(documents, max_length_per_doc, style)
            }
            "get_provider_info" => Self::get_provider_info(),
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    fn get_provider_and_key() -> (String, Option<String>) {
        let provider = std::env::var("SUMMARIZE_PROVIDER")
            .unwrap_or_else(|_| "openai".to_string());

        let api_key = match provider.as_str() {
            "upstage" => std::env::var("UPSTAGE_API_KEY").ok(),
            _ => std::env::var("OPENAI_API_KEY").ok(),
        };

        (provider, api_key)
    }

    fn call_llm(text: &str, max_length: Option<i32>, style: &str) -> Result<String, String> {
        use wasi::http::types::{OutgoingRequest, Scheme};
        use wasi::http::outgoing_handler;

        let (provider, api_key) = Self::get_provider_and_key();
        let api_key = api_key.ok_or_else(|| format!("API key not set for provider: {}", provider))?;

        let (host, path, model) = match provider.as_str() {
            "upstage" => ("api.upstage.ai", "/v1/solar/chat/completions", "solar-pro"),
            _ => ("api.openai.com", "/v1/chat/completions", "gpt-4o-mini"),
        };

        let length_instruction = max_length
            .map(|l| format!(" Keep the summary under {} characters.", l))
            .unwrap_or_default();

        // Match Python summarize_server.py styles
        let style_instruction = match style {
            "concise" => "Provide a brief, concise summary.",
            "detailed" => "Provide a detailed summary with key points.",
            "bullet" => "Summarize as bullet points.",
            _ => "Provide a brief, concise summary.",
        };

        let system_prompt = format!(
            "You are a summarization assistant. {} {}",
            style_instruction, length_instruction
        );

        let request_body = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format!("Summarize the following text:\n\n{}", text)}
            ],
            "max_tokens": max_length.unwrap_or(500) as i64
        });

        let body_bytes = request_body.to_string();

        // Create headers
        let headers = Fields::new();
        let _ = headers.set(&"Host".to_string(), &[host.as_bytes().to_vec()]);
        let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
        let _ = headers.set(&"Accept".to_string(), &[b"application/json".to_vec()]);
        let auth_value = format!("Bearer {}", api_key);
        let _ = headers.set(&"Authorization".to_string(), &[auth_value.as_bytes().to_vec()]);

        // Create outgoing request
        let request = OutgoingRequest::new(headers);
        request.set_method(&Method::Post).map_err(|_| "Failed to set method")?;
        request.set_scheme(Some(&Scheme::Https)).map_err(|_| "Failed to set scheme")?;
        request.set_authority(Some(host)).map_err(|_| "Failed to set authority")?;
        request.set_path_with_query(Some(path)).map_err(|_| "Failed to set path")?;

        // Set request body
        let outgoing_body = request.body().map_err(|_| "Failed to get request body")?;
        {
            let stream = outgoing_body.write().map_err(|_| "Failed to get body stream")?;
            stream.blocking_write_and_flush(body_bytes.as_bytes())
                .map_err(|e| format!("Failed to write body: {:?}", e))?;
        }
        OutgoingBody::finish(outgoing_body, None).map_err(|_| "Failed to finish body")?;

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

        // Read response body
        let resp_body = response.consume().map_err(|_| "Failed to consume response body")?;
        let stream = resp_body.stream().map_err(|_| "Failed to get body stream")?;

        let mut content = Vec::new();
        loop {
            match stream.blocking_read(65536) {
                Ok(chunk) if !chunk.is_empty() => content.extend_from_slice(&chunk),
                _ => break,
            }
        }

        let content_str = String::from_utf8_lossy(&content).to_string();

        if status >= 400 {
            return Err(format!("LLM API error (status {}): {}", status, content_str));
        }

        // Parse response
        let response_json: Value = serde_json::from_str(&content_str)
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        let summary = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| "No content in LLM response".to_string())?;

        Ok(summary.to_string())
    }

    fn summarize_text(text: &str, max_length: i32, style: &str) -> Result<String, String> {
        // Match Python: empty text returns error
        if text.trim().is_empty() {
            return Err("Error: Empty text provided".to_string());
        }

        // Match Python: text too short to summarize
        if text.len() < 100 {
            return Ok(text.to_string());
        }

        let summary = Self::call_llm(text, Some(max_length), style)?;

        // Return plain text (matching Python output format)
        Ok(summary)
    }

    fn summarize_documents(documents: &[Value], max_length_per_doc: i32, style: &str) -> Result<String, String> {
        // Match Python: accepts list[str]
        let mut summaries = Vec::new();

        for doc in documents {
            let content = match doc {
                Value::String(s) => s.clone(),
                _ => doc.to_string(),
            };

            let summary = if content.trim().is_empty() || content.len() < 100 {
                content.clone()
            } else {
                Self::call_llm(&content, Some(max_length_per_doc), style)?
            };

            summaries.push(Value::String(summary));
        }

        // Return list of summaries (matching Python output format)
        Ok(serde_json::to_string(&summaries).unwrap_or_default())
    }

    fn get_provider_info() -> Result<String, String> {
        let (provider, _) = Self::get_provider_and_key();

        // Match Python output format exactly
        Ok(json!({
            "provider": provider,
            "available_styles": ["concise", "detailed", "bullet"],
            "default_max_length": 150
        }).to_string())
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
