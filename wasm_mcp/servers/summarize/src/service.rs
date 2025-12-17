//! Summarize Service - Text summarization with LLM APIs
//!
//! Uses OpenAI or Upstage API for text summarization via wasi:http/outgoing-handler.

use rmcp::{
    ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;  // profiling

/// Summarize MCP Service
#[derive(Debug, Clone)]
pub struct SummarizeService {
    tool_router: ToolRouter<Self>,
    provider: String,
    api_key: Option<String>,
}

impl SummarizeService {
    pub fn new() -> Self {
        // Read provider from environment (default: openai)
        let provider = std::env::var("SUMMARIZE_PROVIDER")
            .unwrap_or_else(|_| "openai".to_string());

        // Read API key from environment
        let api_key = match provider.as_str() {
            "upstage" => std::env::var("UPSTAGE_API_KEY").ok(),
            _ => std::env::var("OPENAI_API_KEY").ok(),
        };

        Self {
            tool_router: Self::tool_router(),
            provider,
            api_key,
        }
    }

    /// Make HTTP POST request using wasi:http/outgoing-handler
    fn http_post(url: &str, headers: &[(&str, &str)], body: &str) -> Result<(u16, String), String> {
        use wasi::http::types::{
            Fields, Method, OutgoingRequest, Scheme, OutgoingBody,
        };
        use wasi::http::outgoing_handler;

        // Parse URL
        let (host, path) = if url.starts_with("https://") {
            let rest = &url[8..];
            let slash_pos = rest.find('/').unwrap_or(rest.len());
            (&rest[..slash_pos], if slash_pos < rest.len() { &rest[slash_pos..] } else { "/" })
        } else {
            return Err("Only https URLs are supported".to_string());
        };

        // Create headers
        let fields = Fields::new();
        let _ = fields.set(&"Host".to_string(), &[host.as_bytes().to_vec()]);
        let _ = fields.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
        let _ = fields.set(&"Accept".to_string(), &[b"application/json".to_vec()]);

        for (key, value) in headers {
            let _ = fields.set(&key.to_string(), &[value.as_bytes().to_vec()]);
        }

        // Create outgoing request
        let request = OutgoingRequest::new(fields);
        request.set_method(&Method::Post)
            .map_err(|_| "Failed to set method")?;
        request.set_scheme(Some(&Scheme::Https))
            .map_err(|_| "Failed to set scheme")?;
        request.set_authority(Some(host))
            .map_err(|_| "Failed to set authority")?;
        request.set_path_with_query(Some(path))
            .map_err(|_| "Failed to set path")?;

        // Set request body
        let outgoing_body = request.body()
            .map_err(|_| "Failed to get request body")?;

        {
            let stream = outgoing_body.write()
                .map_err(|_| "Failed to get body stream")?;
            stream.blocking_write_and_flush(body.as_bytes())
                .map_err(|e| format!("Failed to write body: {:?}", e))?;
        }

        OutgoingBody::finish(outgoing_body, None)
            .map_err(|_| "Failed to finish body")?;

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
        let body = response.consume()
            .map_err(|_| "Failed to consume response body")?;

        let stream = body.stream()
            .map_err(|_| "Failed to get body stream")?;

        let mut content = Vec::new();
        loop {
            match stream.blocking_read(65536) {
                Ok(chunk) if !chunk.is_empty() => content.extend_from_slice(&chunk),
                _ => break,
            }
        }

        let content_str = String::from_utf8_lossy(&content).to_string();

        Ok((status, content_str))
    }

    /// Call LLM API for summarization
    fn call_llm(&self, text: &str, max_length: Option<i32>, style: &str) -> Result<String, String> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| format!("API key not set for provider: {}", self.provider))?;

        let (url, model) = match self.provider.as_str() {
            "upstage" => (
                "https://api.upstage.ai/v1/solar/chat/completions",
                "solar-pro"
            ),
            _ => (
                "https://api.openai.com/v1/chat/completions",
                "gpt-4o-mini"
            ),
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

        let request_body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format!("Summarize the following text:\n\n{}", text)}
            ],
            "max_tokens": max_length.unwrap_or(500) as i64
        });

        let auth_header = format!("Bearer {}", api_key);
        let headers = [("Authorization", auth_header.as_str())];

        let (status, response) = Self::http_post(url, &headers, &request_body.to_string())?;

        if status >= 400 {
            return Err(format!("LLM API error (status {}): {}", status, response));
        }

        // Parse response
        let response_json: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        let summary = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| "No content in LLM response".to_string())?;

        Ok(summary.to_string())
    }
}

impl Default for SummarizeService {
    fn default() -> Self {
        Self::new()
    }
}

// Tool parameters - matching Python summarize_server.py

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SummarizeTextParams {
    #[schemars(description = "The text to summarize")]
    pub text: String,

    #[schemars(description = "Maximum length of summary in words (default: 150)")]
    #[serde(default = "default_max_length")]
    pub max_length: i32,

    #[schemars(description = "Summary style - \"concise\", \"detailed\", or \"bullet\"")]
    #[serde(default = "default_style")]
    pub style: String,
}

fn default_max_length() -> i32 { 150 }
fn default_style() -> String { "concise".to_string() }
fn default_max_length_per_doc() -> i32 { 100 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SummarizeDocumentsParams {
    #[schemars(description = "List of documents to summarize")]
    pub documents: Vec<String>,

    #[schemars(description = "Maximum length per summary in words")]
    #[serde(default = "default_max_length_per_doc")]
    pub max_length_per_doc: i32,

    #[schemars(description = "Summary style for all documents")]
    #[serde(default = "default_style")]
    pub style: String,
}

// Response types - matching Python summarize_server.py

#[derive(Debug, Serialize)]
pub struct ProviderInfo {
    pub provider: String,
    pub available_styles: Vec<String>,
    pub default_max_length: i32,
}

#[tool_router]
impl SummarizeService {
    /// Summarize a single text - matching Python summarize_server.py
    #[tool(description = "Summarize the given text.")]
    fn summarize_text(&self, Parameters(params): Parameters<SummarizeTextParams>) -> Result<String, String> {
        let fn_start = Instant::now();
        let compute_start = Instant::now();
        // Match Python: empty text returns error
        if params.text.trim().is_empty() {
            let fn_total_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("---TIMING---{{\"fn_total_ms\":{:.3},\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}", fn_total_ms);
            return Err("Error: Empty text provided".to_string());
        }

        // Match Python: text too short to summarize
        if params.text.len() < 100 {
            let fn_total_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("---TIMING---{{\"fn_total_ms\":{:.3},\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}", fn_total_ms);
            return Ok(params.text);
        }

        let io_start = Instant::now();
        let summary = self.call_llm(&params.text, Some(params.max_length), &params.style)?;
        let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0 - io_ms;

        let serialize_start = Instant::now();
        let output = summary.clone();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        let fn_total_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("---TIMING---{{\"fn_total_ms\":{:.3},\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", fn_total_ms, io_ms, compute_ms, serialize_ms);

        // Return plain text (matching Python output format)
        Ok(output)
    }

    /// Summarize multiple documents - matching Python summarize_server.py
    #[tool(description = "Summarize multiple documents.")]
    fn summarize_documents(&self, Parameters(params): Parameters<SummarizeDocumentsParams>) -> Result<String, String> {
        let fn_start = Instant::now();
        let compute_start = Instant::now();
        let mut io_ms = 0.0;

        // Match Python: accepts list[str]
        let mut summaries = Vec::new();

        for doc in &params.documents {
            let summary = if doc.trim().is_empty() || doc.len() < 100 {
                doc.clone()
            } else {
                let io_start = Instant::now();
                let result = self.call_llm(doc, Some(params.max_length_per_doc), &params.style)?;
                io_ms += io_start.elapsed().as_secs_f64() * 1000.0;
                result
            };

            summaries.push(summary);
        }

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0 - io_ms;

        let serialize_start = Instant::now();
        let output = serde_json::to_string(&summaries)
            .map_err(|e| format!("Failed to serialize result: {}", e))?;
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        let fn_total_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("---TIMING---{{\"fn_total_ms\":{:.3},\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", fn_total_ms, io_ms, compute_ms, serialize_ms);

        // Return list of summaries (matching Python output format)
        Ok(output)
    }

    /// Get information about the summarization provider - matching Python summarize_server.py
    #[tool(description = "Get information about the current summarization provider.")]
    fn get_provider_info(&self) -> Result<String, String> {
        let fn_start = Instant::now();
        let compute_start = Instant::now();
        let result = ProviderInfo {
            provider: self.provider.clone(),
            available_styles: vec![
                "concise".to_string(),
                "detailed".to_string(),
                "bullet".to_string(),
            ],
            default_max_length: 150,
        };

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = serde_json::to_string(&result)
            .map_err(|e| format!("Failed to serialize result: {}", e))?;
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        let fn_total_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("---TIMING---{{\"fn_total_ms\":{:.3},\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", fn_total_ms, compute_ms, serialize_ms);
        Ok(output)
    }
}

#[tool_handler]
impl ServerHandler for SummarizeService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Summarize MCP Server - Summarizes text using LLM APIs. \
                Supports OpenAI (gpt-4o-mini) and Upstage (solar-pro) providers. \
                Set OPENAI_API_KEY or UPSTAGE_API_KEY environment variable. \
                Use SUMMARIZE_PROVIDER=upstage to use Upstage instead of OpenAI.".into()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
