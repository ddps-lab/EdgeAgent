//! Fetch tools - Pure business logic (with WASI HTTP dependency)
//!
//! Shared between CLI and HTTP transports.

use url::Url;
use wasmmcp::timing::{measure_network_io, ToolTimer, get_wasm_total_ms};
use regex::Regex;

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
    let (status, content_str) = measure_network_io(|| {
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

/// Extract Semantic Scholar paper ID from URL
/// Supports:
/// - https://www.semanticscholar.org/paper/TITLE/PAPER_ID
/// - https://www.semanticscholar.org/paper/PAPER_ID
fn extract_s2_paper_id(url: &str) -> Option<String> {
    let re = Regex::new(r"/paper/(?:[^/]+/)?([a-f0-9]{40})/?$").ok()?;
    re.captures(url)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
}

/// Simple sleep using WASI monotonic clock (busy wait)
fn wasi_sleep_ms(ms: u64) {
    use wasi::clocks::monotonic_clock;
    let duration_ns = ms * 1_000_000;
    let start = monotonic_clock::now();
    while monotonic_clock::now() - start < duration_ns {
        // Busy wait - WASM doesn't have proper sleep
    }
}

/// Fetch Semantic Scholar paper via API with retry on 429
fn fetch_s2_paper_via_api(paper_id: &str, max_length: usize) -> Result<String, String> {
    let api_url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/{}?fields=title,abstract,year,authors,venue,citationCount,tldr",
        paper_id
    );

    // Retry with exponential backoff for rate limiting (429)
    const MAX_RETRIES: u32 = 3;
    let mut retries: u32 = 0;
    let mut retry_delay_ms: u64 = 1000; // Start with 1 second

    let (status, content) = loop {
        let result = http_get(&api_url)?;

        if result.0 == 429 {
            // Rate limited - retry with backoff
            retries += 1;
            if retries > MAX_RETRIES {
                return Err("Semantic Scholar API rate limited (429). Max retries exceeded.".to_string());
            }

            // Wait and retry
            wasi_sleep_ms(retry_delay_ms);
            retry_delay_ms *= 2; // Exponential backoff
            continue;
        }

        break result;
    };

    if status >= 400 {
        return Err(format!("Semantic Scholar API error: status {}", status));
    }

    // Parse JSON and format as markdown
    // Simple JSON parsing without serde (to keep WASM small)
    let mut result = String::new();

    // Extract title
    if let Some(title) = extract_json_string(&content, "title") {
        result.push_str(&format!("# {}\n\n", title));
    }

    // Extract year
    if let Some(year) = extract_json_value(&content, "year") {
        result.push_str(&format!("**Year:** {}\n", year));
    }

    // Extract venue
    if let Some(venue) = extract_json_string(&content, "venue") {
        if !venue.is_empty() {
            result.push_str(&format!("**Venue:** {}\n", venue));
        }
    }

    // Extract citation count
    if let Some(citations) = extract_json_value(&content, "citationCount") {
        result.push_str(&format!("**Citations:** {}\n", citations));
    }

    result.push('\n');

    // Extract tldr
    if content.contains("\"tldr\"") {
        if let Some(tldr_start) = content.find("\"tldr\"") {
            if let Some(text_start) = content[tldr_start..].find("\"text\"") {
                let text_pos = tldr_start + text_start;
                if let Some(tldr_text) = extract_json_string(&content[text_pos..], "text") {
                    result.push_str("## TL;DR\n");
                    result.push_str(&tldr_text);
                    result.push_str("\n\n");
                }
            }
        }
    }

    // Extract abstract
    if let Some(abstract_text) = extract_json_string(&content, "abstract") {
        result.push_str("## Abstract\n");
        result.push_str(&abstract_text);
    }

    let result = if result.len() > max_length {
        result[..max_length].to_string()
    } else {
        result
    };

    Ok(result)
}

/// Simple JSON string extraction (without full parser)
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];

    // Skip whitespace
    let after_key = after_key.trim_start();

    // Check if value is a string
    if !after_key.starts_with('"') {
        return None;
    }

    // Find end of string (handle escaped quotes)
    let value_start = 1;
    let mut value_end = value_start;
    let chars: Vec<char> = after_key.chars().collect();

    while value_end < chars.len() {
        if chars[value_end] == '"' && (value_end == 0 || chars[value_end - 1] != '\\') {
            break;
        }
        value_end += 1;
    }

    let value: String = chars[value_start..value_end].iter().collect();
    Some(value.replace("\\\"", "\"").replace("\\n", "\n"))
}

/// Simple JSON value extraction (for numbers, etc.)
fn extract_json_value(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];
    let after_key = after_key.trim_start();

    // Find end of value (comma, }, or ])
    let end = after_key.find(|c| c == ',' || c == '}' || c == ']')?;
    let value = after_key[..end].trim();

    if value == "null" {
        None
    } else {
        Some(value.trim_matches('"').to_string())
    }
}

/// Fetch a URL and return its contents as markdown
pub fn fetch(url_str: &str, max_length: usize) -> Result<String, String> {
    let timer = ToolTimer::start();
    let url = Url::parse(url_str)
        .map_err(|e| format!("Invalid URL: {}", e))?;

    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(format!("Only http and https URLs are supported, got: {}", url.scheme()));
    }

    // Check if this is a Semantic Scholar paper URL - use API instead
    if let Some(paper_id) = extract_s2_paper_id(url_str) {
        let result = fetch_s2_paper_via_api(&paper_id, max_length)?;
        let timing = timer.finish("fetch");
        return Ok(serde_json::json!({
            "content": result,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "disk_io_ms": timing.disk_io_ms,
                "network_io_ms": timing.network_io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string());
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

    let timing = timer.finish("fetch");
    Ok(serde_json::json!({
        "content": result,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}
