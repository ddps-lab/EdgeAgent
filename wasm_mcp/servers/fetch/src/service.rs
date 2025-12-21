//! Fetch Service - HTTP fetching with wasi:http/outgoing-handler
//!
//! Fetches web pages and converts HTML to markdown.

use wasmmcp::timing::{ToolTimer, get_wasm_total_ms};
use rmcp::{
    ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use url::Url;

/// Fetch MCP Service
#[derive(Debug, Clone)]
pub struct FetchService {
    tool_router: ToolRouter<Self>,
}

impl FetchService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Make HTTP request using wasi:http/outgoing-handler
    fn http_get(url_str: &str) -> Result<(u16, String), String> {
        use wasi::http::types::{
            Fields, Method, OutgoingRequest, Scheme,
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

        // Create headers
        let headers = Fields::new();
        let _ = headers.set(
            &"User-Agent".to_string(),
            &[b"WasmMCP-Fetch/1.0".to_vec()]
        );
        let _ = headers.set(
            &"Accept".to_string(),
            &[b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_vec()]
        );

        // Create outgoing request
        let request = OutgoingRequest::new(headers);
        request.set_method(&Method::Get)
            .map_err(|_| "Failed to set method")?;
        request.set_scheme(Some(&scheme))
            .map_err(|_| "Failed to set scheme")?;
        request.set_authority(Some(host))
            .map_err(|_| "Failed to set authority")?;
        request.set_path_with_query(Some(&path_and_query))
            .map_err(|_| "Failed to set path")?;

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
            // Poll
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

    /// Simple HTML to text/markdown conversion
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

                // Skip script and style content
                if tag_lower.starts_with("script") || tag_lower.starts_with("style") {
                    skip_content = true;
                } else if tag_lower.starts_with("/script") || tag_lower.starts_with("/style") {
                    skip_content = false;
                }

                // Convert tags to markdown
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
                // Decode common HTML entities
                result.push(c);
            }
        }

        // Clean up HTML entities
        let result = result
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'");

        // Clean up whitespace
        let lines: Vec<&str> = result.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        lines.join("\n")
    }
}

impl Default for FetchService {
    fn default() -> Self {
        Self::new()
    }
}

// Parameter struct - matches Python fetch_server.py
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FetchParams {
    #[schemars(description = "URL to fetch")]
    pub url: String,

    #[schemars(description = "Maximum length of returned content (default: 50000)")]
    pub max_length: Option<usize>,
}

// Tool implementation - Output matches Python fetch_server.py (plain text markdown)
#[tool_router]
impl FetchService {
    /// Fetch a URL and return its contents as markdown
    /// Output format matches Python fetch_server.py
    #[tool(description = "Fetches a URL from the internet and extracts its contents as markdown")]
    fn fetch(&self, Parameters(params): Parameters<FetchParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let max_length = params.max_length.unwrap_or(50000);

        // Validate URL
        let url = Url::parse(&params.url)
            .map_err(|e| format!("Invalid URL: {}", e))?;

        if url.scheme() != "http" && url.scheme() != "https" {
            return Err(format!("Only http and https URLs are supported, got: {}", url.scheme()));
        }

        // Fetch the URL
        let (status, content) = Self::http_get(&params.url)?;

        if status >= 400 {
            return Err(format!("HTTP error: status {}", status));
        }

        // Convert HTML to markdown (matching Python fetch_server behavior)
        let processed = if content.trim_start().starts_with("<!") ||
                          content.trim_start().starts_with("<html") ||
                          content.contains("<head") {
            Self::html_to_markdown(&content)
        } else {
            content
        };

        // Truncate if needed
        let result = if processed.len() > max_length {
            processed[..max_length].to_string()
        } else {
            processed
        };

        let timing = timer.finish("fetch");
        Ok(serde_json::json!({
            "content": result,
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }
}

#[tool_handler]
impl ServerHandler for FetchService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Fetch MCP Server - Retrieves web content and converts to markdown. \
                Use the fetch tool to get content from URLs. Supports pagination for large pages.".into()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
