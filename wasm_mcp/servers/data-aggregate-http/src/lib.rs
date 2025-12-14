//! HTTP entry point for Data Aggregate MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve target/wasm32-wasip2/release/mcp_server_data_aggregate_http.wasm

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

// Helper functions

fn safe_numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.replace(",", "").parse().ok(),
        _ => None,
    }
}

fn compute_stats(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"count": 0});
    }

    let count = values.len();
    let sum: f64 = values.iter().sum();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = sum / count as f64;

    // Median
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };

    let mut result = json!({
        "count": count,
        "sum": sum,
        "min": min,
        "max": max,
        "mean": mean,
        "median": median,
    });

    // Standard deviation
    if count > 1 {
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        result["stdev"] = json!(variance.sqrt());
    }

    result
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
                    "serverInfo": { "name": "wasmmcp-data-aggregate-http", "version": "1.0.0" },
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
                "name": "aggregate_list",
                "description": "Aggregate a list of dictionaries by grouping, counting, or summing. Returns statistics like counts per group, value distributions, and numeric field stats.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "List of dictionaries to aggregate"},
                        "groupBy": {"type": "string", "description": "Field name to group by"},
                        "countField": {"type": "string", "description": "Field to count occurrences"},
                        "sumFields": {"type": "array", "items": {"type": "string"}, "description": "Fields for numeric statistics"}
                    },
                    "required": ["items"]
                }
            }),
            json!({
                "name": "merge_summaries",
                "description": "Merge multiple summary dictionaries into one. Supports weighted averages for numeric values.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "summaries": {"type": "array", "description": "List of summary dictionaries to merge"},
                        "weights": {"type": "array", "items": {"type": "number"}, "description": "Optional weights for weighted averages"}
                    },
                    "required": ["summaries"]
                }
            }),
            json!({
                "name": "combine_research_results",
                "description": "Combine multiple research/search results into a coherent summary. Sorts by relevance score.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array", "description": "List of research results"},
                        "titleField": {"type": "string", "default": "title"},
                        "summaryField": {"type": "string", "default": "summary"},
                        "scoreField": {"type": "string", "default": "relevance_score"}
                    },
                    "required": ["results"]
                }
            }),
            json!({
                "name": "deduplicate",
                "description": "Remove duplicate items based on key fields. Returns deduplicated items with statistics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "List of items to deduplicate"},
                        "keyFields": {"type": "array", "items": {"type": "string"}, "description": "Fields to use as deduplication key"},
                        "keep": {"type": "string", "enum": ["first", "last"], "default": "first"}
                    },
                    "required": ["items", "keyFields"]
                }
            }),
            json!({
                "name": "compute_trends",
                "description": "Compute trends from time-series data. Analyzes whether values are increasing, decreasing, or stable.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "timeSeries": {"type": "array", "description": "List of time-series data points"},
                        "timeField": {"type": "string", "default": "timestamp"},
                        "valueField": {"type": "string", "default": "value"},
                        "bucketCount": {"type": "integer", "default": 10}
                    },
                    "required": ["timeSeries"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "aggregate_list" => {
                let items = args.get("items").and_then(|v| v.as_array()).ok_or("items required")?;
                if items.is_empty() {
                    return Ok(json!({"total_count": 0, "groups": {}}).to_string());
                }

                let input_size = serde_json::to_string(&items).map(|s| s.len()).unwrap_or(0);
                let mut result = json!({
                    "total_count": items.len(),
                    "input_size_estimate": input_size,
                });

                // Group by
                if let Some(group_by) = args.get("groupBy").and_then(|v| v.as_str()) {
                    let mut groups: HashMap<String, usize> = HashMap::new();
                    for item in items {
                        let key = item.get(group_by)
                            .map(|v| match v { Value::String(s) => s.clone(), _ => v.to_string() })
                            .unwrap_or_else(|| "unknown".to_string());
                        *groups.entry(key).or_insert(0) += 1;
                    }
                    result["group_by"] = json!(group_by);
                    result["groups"] = json!(groups);
                    result["group_count"] = json!(groups.len());
                }

                // Count field
                if let Some(count_field) = args.get("countField").and_then(|v| v.as_str()) {
                    let mut counts: HashMap<String, usize> = HashMap::new();
                    for item in items {
                        let key = item.get(count_field)
                            .map(|v| match v { Value::String(s) => s.clone(), _ => v.to_string() })
                            .unwrap_or_else(|| "unknown".to_string());
                        *counts.entry(key).or_insert(0) += 1;
                    }
                    let mut sorted: Vec<_> = counts.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(&a.1));
                    let top_counts: HashMap<String, usize> = sorted.into_iter().take(20).collect();
                    result["count_by"] = json!(count_field);
                    result["counts"] = json!(top_counts);
                }

                // Sum fields
                if let Some(sum_fields) = args.get("sumFields").and_then(|v| v.as_array()) {
                    let mut field_stats = json!({});
                    for field in sum_fields {
                        if let Some(field_name) = field.as_str() {
                            let values: Vec<f64> = items.iter()
                                .filter_map(|item| item.get(field_name))
                                .filter_map(|v| safe_numeric(v))
                                .collect();
                            field_stats[field_name] = compute_stats(&values);
                        }
                    }
                    result["field_stats"] = field_stats;
                }

                let output_size = serde_json::to_string(&result).map(|s| s.len()).unwrap_or(0);
                result["output_size_estimate"] = json!(output_size);
                result["reduction_ratio"] = json!(if input_size > 0 { output_size as f64 / input_size as f64 } else { 0.0 });

                Ok(result.to_string())
            }

            "merge_summaries" => {
                let summaries = args.get("summaries").and_then(|v| v.as_array()).ok_or("summaries required")?;
                if summaries.is_empty() {
                    return Ok(json!({"merged_count": 0}).to_string());
                }

                let weights: Vec<f64> = args.get("weights")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                    .unwrap_or_else(|| vec![1.0; summaries.len()]);

                let mut all_keys: Vec<String> = Vec::new();
                for s in summaries {
                    if let Value::Object(map) = s {
                        for key in map.keys() {
                            if !all_keys.contains(key) {
                                all_keys.push(key.clone());
                            }
                        }
                    }
                }

                let mut merged = json!({
                    "merged_count": summaries.len(),
                    "source_keys": all_keys.clone(),
                });

                for key in &all_keys {
                    let mut numeric_values: Vec<f64> = Vec::new();
                    let mut key_weights: Vec<f64> = Vec::new();
                    let mut nested_counts: HashMap<String, f64> = HashMap::new();

                    for (i, s) in summaries.iter().enumerate() {
                        let w = weights.get(i).copied().unwrap_or(1.0);
                        if let Some(val) = s.get(key) {
                            if let Some(f) = val.as_f64() {
                                numeric_values.push(f);
                                key_weights.push(w);
                            } else if let Value::Object(map) = val {
                                for (k, v) in map {
                                    if let Some(f) = safe_numeric(v) {
                                        *nested_counts.entry(k.clone()).or_insert(0.0) += f * w;
                                    }
                                }
                            }
                        }
                    }

                    if !numeric_values.is_empty() {
                        let weight_sum: f64 = key_weights.iter().sum();
                        if weight_sum > 0.0 {
                            let weighted_sum: f64 = numeric_values.iter().zip(key_weights.iter())
                                .map(|(v, w)| v * w)
                                .sum();
                            merged[format!("{}_weighted_avg", key)] = json!(weighted_sum / weight_sum);
                        }
                        merged[format!("{}_total", key)] = json!(numeric_values.iter().sum::<f64>());
                    }

                    if !nested_counts.is_empty() {
                        merged[key.clone()] = json!(nested_counts);
                    }
                }

                Ok(merged.to_string())
            }

            "combine_research_results" => {
                let results = args.get("results").and_then(|v| v.as_array()).ok_or("results required")?;
                if results.is_empty() {
                    return Ok(json!({"result_count": 0, "combined_summary": ""}).to_string());
                }

                let title_field = args.get("titleField").and_then(|v| v.as_str()).unwrap_or("title");
                let summary_field = args.get("summaryField").and_then(|v| v.as_str()).unwrap_or("summary");
                let score_field = args.get("scoreField").and_then(|v| v.as_str()).unwrap_or("relevance_score");

                let mut sorted_results: Vec<&Value> = results.iter().collect();
                sorted_results.sort_by(|a, b| {
                    let score_a = a.get(score_field).and_then(safe_numeric).unwrap_or(0.0);
                    let score_b = b.get(score_field).and_then(safe_numeric).unwrap_or(0.0);
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut items: Vec<Value> = Vec::new();
                let mut combined_parts: Vec<String> = Vec::new();

                for (i, r) in sorted_results.iter().enumerate() {
                    let title = r.get(title_field).and_then(|v| v.as_str()).unwrap_or(&format!("Result {}", i + 1)).to_string();
                    let summary = r.get(summary_field).and_then(|v| v.as_str()).unwrap_or("").to_string();

                    let mut item = json!({"rank": i + 1, "title": title.clone(), "summary": summary.clone()});
                    if let Some(score) = r.get(score_field).and_then(safe_numeric) {
                        item["score"] = json!(score);
                    }

                    combined_parts.push(format!("[{}] {}\n{}", i + 1, title, summary));
                    items.push(item);
                }

                let combined_text = combined_parts.join("\n\n");
                let input_size: usize = results.iter()
                    .map(|r| serde_json::to_string(r).map(|s| s.len()).unwrap_or(0))
                    .sum();

                Ok(json!({
                    "result_count": results.len(),
                    "items": items,
                    "combined_text": combined_text,
                    "input_size": input_size,
                    "output_size": combined_text.len(),
                }).to_string())
            }

            "deduplicate" => {
                let items = args.get("items").and_then(|v| v.as_array()).ok_or("items required")?;
                let key_fields: Vec<String> = args.get("keyFields")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .ok_or("keyFields required")?;
                let keep = args.get("keep").and_then(|v| v.as_str()).unwrap_or("first");

                if items.is_empty() {
                    return Ok(json!({"original_count": 0, "unique_count": 0, "items": []}).to_string());
                }

                let mut seen: HashMap<String, Value> = HashMap::new();
                let mut result: Vec<Value> = Vec::new();

                for item in items {
                    let key: String = key_fields.iter()
                        .map(|f| item.get(f).map(|v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        }).unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("|");

                    if !seen.contains_key(&key) {
                        seen.insert(key.clone(), item.clone());
                        if keep == "first" {
                            result.push(item.clone());
                        }
                    } else if keep == "last" {
                        seen.insert(key, item.clone());
                    }
                }

                if keep == "last" {
                    result = seen.into_values().collect();
                }

                let unique_count = result.len();
                Ok(json!({
                    "original_count": items.len(),
                    "unique_count": unique_count,
                    "duplicates_removed": items.len() - unique_count,
                    "key_fields": key_fields,
                    "items": result,
                }).to_string())
            }

            "compute_trends" => {
                let time_series = args.get("timeSeries").and_then(|v| v.as_array()).ok_or("timeSeries required")?;
                if time_series.is_empty() {
                    return Ok(json!({"data_points": 0, "trend": "insufficient_data"}).to_string());
                }

                let time_field = args.get("timeField").and_then(|v| v.as_str()).unwrap_or("timestamp");
                let value_field = args.get("valueField").and_then(|v| v.as_str()).unwrap_or("value");

                let mut data: Vec<(String, f64)> = Vec::new();
                for item in time_series {
                    if let Some(value) = item.get(value_field).and_then(safe_numeric) {
                        let time = item.get(time_field)
                            .map(|v| match v { Value::String(s) => s.clone(), _ => v.to_string() })
                            .unwrap_or_default();
                        data.push((time, value));
                    }
                }

                if data.len() < 2 {
                    return Ok(json!({"data_points": data.len(), "trend": "insufficient_data"}).to_string());
                }

                let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
                let mid = values.len() / 2;
                let first_half: Vec<f64> = values[..mid].to_vec();
                let second_half: Vec<f64> = values[mid..].to_vec();

                let first_avg = if first_half.is_empty() { 0.0 } else { first_half.iter().sum::<f64>() / first_half.len() as f64 };
                let second_avg = if second_half.is_empty() { 0.0 } else { second_half.iter().sum::<f64>() / second_half.len() as f64 };

                let trend = if second_avg > first_avg * 1.1 {
                    "increasing"
                } else if second_avg < first_avg * 0.9 {
                    "decreasing"
                } else {
                    "stable"
                };

                let change_percent = if first_avg != 0.0 { (second_avg - first_avg) / first_avg * 100.0 } else { 0.0 };
                let stats = compute_stats(&values);

                Ok(json!({
                    "data_points": data.len(),
                    "trend": trend,
                    "stats": stats,
                    "first_half_avg": first_avg,
                    "second_half_avg": second_avg,
                    "change_percent": change_percent,
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
