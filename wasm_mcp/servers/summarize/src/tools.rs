//! Summarize tools - Pure business logic (with WASI HTTP dependency)
//!
//! Shared between CLI and HTTP transports.

use serde::Serialize;

/// Provider info response
#[derive(Debug, Serialize)]
pub struct ProviderInfo {
    pub provider: String,
    pub available_styles: Vec<String>,
    pub default_max_length: i32,
}

/// Make HTTP POST request using wasi:http/outgoing-handler
pub fn http_post(url: &str, headers: &[(&str, &str)], body: &str) -> Result<(u16, String), String> {
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
pub fn call_llm(
    provider: &str,
    api_key: &str,
    text: &str,
    max_length: Option<i32>,
    style: &str
) -> Result<String, String> {
    let (url, model) = match provider {
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

    let (status, response) = http_post(url, &headers, &request_body.to_string())?;

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

/// Summarize a single text
pub fn summarize_text(
    provider: &str,
    api_key: Option<&str>,
    text: &str,
    max_length: i32,
    style: &str
) -> Result<String, String> {
    // Match Python: empty text returns error
    if text.trim().is_empty() {
        return Err("Error: Empty text provided".to_string());
    }

    // Match Python: text too short to summarize
    if text.len() < 100 {
        return Ok(text.to_string());
    }

    let api_key = api_key
        .ok_or_else(|| format!("API key not set for provider: {}", provider))?;

    call_llm(provider, api_key, text, Some(max_length), style)
}

/// Summarize multiple documents
pub fn summarize_documents(
    provider: &str,
    api_key: Option<&str>,
    documents: &[String],
    max_length_per_doc: i32,
    style: &str
) -> Result<String, String> {
    let api_key = api_key
        .ok_or_else(|| format!("API key not set for provider: {}", provider))?;

    let mut summaries = Vec::new();

    for doc in documents {
        let summary = if doc.trim().is_empty() || doc.len() < 100 {
            doc.clone()
        } else {
            call_llm(provider, api_key, doc, Some(max_length_per_doc), style)?
        };

        summaries.push(summary);
    }

    // Return list of summaries (matching Python output format)
    serde_json::to_string(&summaries)
        .map_err(|e| format!("Failed to serialize result: {}", e))
}

/// Get information about the summarization provider
pub fn get_provider_info(provider: &str) -> Result<String, String> {
    let result = ProviderInfo {
        provider: provider.to_string(),
        available_styles: vec![
            "concise".to_string(),
            "detailed".to_string(),
            "bullet".to_string(),
        ],
        default_max_length: 150,
    };

    serde_json::to_string(&result)
        .map_err(|e| format!("Failed to serialize result: {}", e))
}
