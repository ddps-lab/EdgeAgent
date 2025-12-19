//! Fetch tools - Pure business logic (with WASI HTTP dependency)
//!
//! Shared between CLI and HTTP transports.

use url::Url;
use wasmmcp::timing::measure_io;

/// Make HTTP request using wasi:http/outgoing-handler
pub fn http_get(url_str: &str) -> Result<(u16, String), String> {
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

    let headers = Fields::new();
    let _ = headers.set(
        &"User-Agent".to_string(),
        &[b"WasmMCP-Fetch/1.0".to_vec()]
    );
    let _ = headers.set(
        &"Accept".to_string(),
        &[b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_vec()]
    );

    let request = OutgoingRequest::new(headers);
    request.set_method(&Method::Get)
        .map_err(|_| "Failed to set method")?;
    request.set_scheme(Some(&scheme))
        .map_err(|_| "Failed to set scheme")?;
    request.set_authority(Some(host))
        .map_err(|_| "Failed to set authority")?;
    request.set_path_with_query(Some(&path_and_query))
        .map_err(|_| "Failed to set path")?;

    // Measure the entire HTTP I/O operation
    let (status, content_str) = measure_io(|| {
        let future_response = outgoing_handler::handle(request, None)
            .map_err(|e| format!("Failed to send request: {:?}", e))?;

        let response = loop {
            if let Some(result) = future_response.get() {
                break result
                    .map_err(|_| "Response error".to_string())?
                    .map_err(|e| format!("HTTP error: {:?}", e))?;
            }
            future_response.subscribe().block();
        };

        let status = response.status();

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

        Ok::<_, String>((status, content_str))
    })?;

    Ok((status, content_str))
}

/// Simple HTML to text/markdown conversion
pub fn html_to_markdown(html: &str) -> String {
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

    let result = result
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'");

    let lines: Vec<&str> = result.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    lines.join("\n")
}

/// Fetch a URL and return its contents as markdown
pub fn fetch(url_str: &str, max_length: usize) -> Result<String, String> {
    let url = Url::parse(url_str)
        .map_err(|e| format!("Invalid URL: {}", e))?;

    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(format!("Only http and https URLs are supported, got: {}", url.scheme()));
    }

    let (status, content) = http_get(url_str)?;

    if status >= 400 {
        return Err(format!("HTTP error: status {}", status));
    }

    let processed = if content.trim_start().starts_with("<!") ||
                      content.trim_start().starts_with("<html") ||
                      content.contains("<head") {
        html_to_markdown(&content)
    } else {
        content
    };

    let result = if processed.len() > max_length {
        processed[..max_length].to_string()
    } else {
        processed
    };

    Ok(result)
}
