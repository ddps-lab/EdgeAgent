//! Data Aggregate MCP Service - aggregates, merges, and summarizes structured data
//!
//! Based on edgeagent's data_aggregate_server.py (FastMCP)

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
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;  // profiling

/// Data Aggregate MCP Service
#[derive(Debug, Clone)]
pub struct DataAggregateService {
    tool_router: ToolRouter<Self>,
}

impl DataAggregateService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for DataAggregateService {
    fn default() -> Self {
        Self::new()
    }
}

// Parameter structs for MCP tools - matching Python FastMCP data_aggregate_server.py (snake_case)

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AggregateListParams {
    /// List of dictionaries to aggregate
    #[schemars(description = "List of dictionaries to aggregate")]
    pub items: Vec<Value>,

    /// Field name to group by
    #[schemars(description = "Field name to group by (e.g., '_level' for log levels)")]
    pub group_by: Option<String>,

    /// Field to count occurrences of unique values
    #[schemars(description = "Field to count occurrences of unique values")]
    pub count_field: Option<String>,

    /// List of field names with numeric values to compute statistics
    #[schemars(description = "List of field names with numeric values to compute sum/avg/min/max statistics")]
    pub sum_fields: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MergeSummariesParams {
    /// List of summary dictionaries to merge
    #[schemars(description = "List of summary dictionaries to merge")]
    pub summaries: Vec<Value>,

    /// Optional weights for each summary (for weighted averages)
    #[schemars(description = "Optional weights for each summary (for weighted averages)")]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CombineResearchResultsParams {
    /// List of research result dictionaries
    #[schemars(description = "List of research result dictionaries")]
    pub results: Vec<Value>,

    /// Field containing the title
    #[serde(default = "default_title_field")]
    #[schemars(description = "Field containing the title")]
    pub title_field: String,

    /// Field containing the summary
    #[serde(default = "default_summary_field")]
    #[schemars(description = "Field containing the summary")]
    pub summary_field: String,

    /// Optional field for relevance scoring
    #[serde(default = "default_score_field")]
    #[schemars(description = "Optional field for relevance scoring")]
    pub score_field: Option<String>,
}

fn default_title_field() -> String { "title".to_string() }
fn default_summary_field() -> String { "summary".to_string() }
fn default_score_field() -> Option<String> { Some("relevance_score".to_string()) }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeduplicateParams {
    /// List of items to deduplicate
    #[schemars(description = "List of items to deduplicate")]
    pub items: Vec<Value>,

    /// Fields to use as the deduplication key
    #[schemars(description = "Fields to use as the deduplication key")]
    pub key_fields: Vec<String>,

    /// Which duplicate to keep (first or last)
    #[serde(default = "default_keep")]
    #[schemars(description = "Which duplicate to keep ('first' or 'last')")]
    pub keep: String,
}

fn default_keep() -> String { "first".to_string() }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ComputeTrendsParams {
    /// List of time-series data points
    #[schemars(description = "List of time-series data points")]
    pub time_series: Vec<Value>,

    /// Field containing the timestamp
    #[serde(default = "default_time_field")]
    #[schemars(description = "Field containing the timestamp")]
    pub time_field: String,

    /// Field containing the value
    #[serde(default = "default_value_field")]
    #[schemars(description = "Field containing the value")]
    pub value_field: String,

    /// Number of time buckets
    #[serde(default = "default_bucket_count")]
    #[schemars(description = "Number of time buckets")]
    pub bucket_count: usize,
}

fn default_time_field() -> String { "timestamp".to_string() }
fn default_value_field() -> String { "value".to_string() }
fn default_bucket_count() -> usize { 10 }

// Helper functions

fn safe_numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.replace(",", "").parse().ok(),
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct Stats {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    stdev: Option<f64>,
}

fn compute_stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            stdev: None,
        };
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

    // Standard deviation
    let stdev = if count > 1 {
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        Some(variance.sqrt())
    } else {
        None
    };

    Stats {
        count,
        sum,
        min,
        max,
        mean,
        median,
        stdev,
    }
}

// Tool implementations

#[tool_router]
impl DataAggregateService {
    /// Aggregate a list of dictionaries by grouping, counting, or summing
    #[tool(description = "Aggregate a list of dictionaries by grouping, counting, or summing. Returns statistics like counts per group, value distributions, and numeric field stats.")]
    fn aggregate_list(
        &self,
        Parameters(params): Parameters<AggregateListParams>,
    ) -> Result<String, String> {
        let items = params.items;
        if items.is_empty() {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(json!({"total_count": 0, "groups": {}}).to_string());
        }

        // input_size 계산은 compute 시간 측정 전에 수행
        let input_size = serde_json::to_string(&items).map(|s| s.len()).unwrap_or(0);

        let compute_start = Instant::now();
        let mut result = json!({
            "total_count": items.len(),
            "input_size_estimate": input_size,
        });

        // Group by field
        if let Some(ref group_by) = params.group_by {
            let mut groups: HashMap<String, Vec<&Value>> = HashMap::new();
            for item in &items {
                let key = item.get(group_by)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                groups.entry(key).or_default().push(item);
            }

            let group_counts: HashMap<String, usize> = groups.iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect();

            result["group_by"] = json!(group_by);
            result["groups"] = json!(group_counts);
            result["group_count"] = json!(groups.len());
        }

        // Count field occurrences
        if let Some(ref count_field) = params.count_field {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for item in &items {
                let key = item.get(count_field)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                *counts.entry(key).or_insert(0) += 1;
            }

            // Get top 20
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let top_counts: HashMap<String, usize> = sorted.into_iter().take(20).collect();

            result["count_by"] = json!(count_field);
            result["counts"] = json!(top_counts);
        }

        // Sum numeric fields
        if let Some(ref sum_fields) = params.sum_fields {
            let mut field_stats = json!({});
            for field in sum_fields {
                let values: Vec<f64> = items.iter()
                    .filter_map(|item| item.get(field))
                    .filter_map(|v| safe_numeric(v))
                    .collect();
                let stats = compute_stats(&values);
                field_stats[field] = serde_json::to_value(stats).unwrap_or(json!({}));
            }
            result["field_stats"] = field_stats;
        }

        // output_size 계산을 위해 임시로 serialize (compute 시간에 포함 안 됨)
        let output_size = serde_json::to_string(&result).map(|s| s.len()).unwrap_or(0);
        result["output_size_estimate"] = json!(output_size);
        result["reduction_ratio"] = json!(if input_size > 0 { output_size as f64 / input_size as f64 } else { 0.0 });

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        // serialize_ms: 최종 결과를 JSON 문자열로 변환하는 시간
        let serialize_start = Instant::now();
        let output = result.to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }

    /// Merge multiple summary dictionaries into one
    #[tool(description = "Merge multiple summary dictionaries into one. Supports weighted averages for numeric values.")]
    fn merge_summaries(
        &self,
        Parameters(params): Parameters<MergeSummariesParams>,
    ) -> Result<String, String> {
        let compute_start = Instant::now();
        let summaries = params.summaries;
        if summaries.is_empty() {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(json!({"merged_count": 0}).to_string());
        }

        let weights = params.weights.unwrap_or_else(|| vec![1.0; summaries.len()]);

        // Collect all keys
        let mut all_keys: Vec<String> = Vec::new();
        for s in &summaries {
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

        // Merge each key
        for key in &all_keys {
            let mut numeric_values: Vec<f64> = Vec::new();
            let mut key_weights: Vec<f64> = Vec::new();
            let mut nested_counts: HashMap<String, f64> = HashMap::new();

            for (i, s) in summaries.iter().enumerate() {
                let w = weights.get(i).copied().unwrap_or(1.0);
                if let Some(val) = s.get(key) {
                    match val {
                        Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                numeric_values.push(f);
                                key_weights.push(w);
                            }
                        }
                        Value::Object(map) => {
                            for (k, v) in map {
                                if let Some(f) = safe_numeric(v) {
                                    *nested_counts.entry(k.clone()).or_insert(0.0) += f * w;
                                }
                            }
                        }
                        _ => {}
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

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = merged.to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }

    /// Combine multiple research/search results into a coherent summary
    #[tool(description = "Combine multiple research/search results into a coherent summary. Sorts by relevance score and creates combined text.")]
    fn combine_research_results(
        &self,
        Parameters(params): Parameters<CombineResearchResultsParams>,
    ) -> Result<String, String> {
        let results = params.results;
        if results.is_empty() {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(json!({"result_count": 0, "combined_summary": ""}).to_string());
        }

        // input_size 계산은 compute 시간 측정 전에 수행
        let input_size: usize = results.iter()
            .map(|r| serde_json::to_string(r).map(|s| s.len()).unwrap_or(0))
            .sum();

        let compute_start = Instant::now();
        let title_field = &params.title_field;
        let summary_field = &params.summary_field;
        let score_field = params.score_field.as_deref().unwrap_or("relevance_score");

        // Sort by score
        let mut sorted_results: Vec<&Value> = results.iter().collect();
        sorted_results.sort_by(|a, b| {
            let score_a = a.get(score_field).and_then(safe_numeric).unwrap_or(0.0);
            let score_b = b.get(score_field).and_then(safe_numeric).unwrap_or(0.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract key information
        let mut items: Vec<Value> = Vec::new();
        let mut combined_parts: Vec<String> = Vec::new();

        for (i, r) in sorted_results.iter().enumerate() {
            let title = r.get(title_field)
                .and_then(|v| v.as_str())
                .unwrap_or(&format!("Result {}", i + 1))
                .to_string();
            let summary = r.get(summary_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mut item = json!({
                "rank": i + 1,
                "title": title.clone(),
                "summary": summary.clone(),
            });

            if let Some(score) = r.get(score_field).and_then(safe_numeric) {
                item["score"] = json!(score);
            }

            combined_parts.push(format!("[{}] {}\n{}", i + 1, title, summary));
            items.push(item);
        }

        let combined_text = combined_parts.join("\n\n");

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = json!({
            "result_count": results.len(),
            "items": items,
            "combined_text": combined_text,
            "input_size": input_size,
            "output_size": combined_text.len(),
        }).to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }

    /// Remove duplicate items based on key fields
    #[tool(description = "Remove duplicate items based on key fields. Returns deduplicated items with statistics.")]
    fn deduplicate(
        &self,
        Parameters(params): Parameters<DeduplicateParams>,
    ) -> Result<String, String> {
        let compute_start = Instant::now();
        let items = params.items;
        if items.is_empty() {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(json!({"original_count": 0, "unique_count": 0, "items": []}).to_string());
        }

        let key_fields = params.key_fields;
        let keep = &params.keep;

        let mut seen: HashMap<String, Value> = HashMap::new();
        let mut result: Vec<Value> = Vec::new();

        for item in &items {
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
        let duplicates_removed = items.len() - unique_count;

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = json!({
            "original_count": items.len(),
            "unique_count": unique_count,
            "duplicates_removed": duplicates_removed,
            "key_fields": key_fields,
            "items": result,
        }).to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }

    /// Compute trends from time-series data
    #[tool(description = "Compute trends from time-series data. Analyzes whether values are increasing, decreasing, or stable.")]
    fn compute_trends(
        &self,
        Parameters(params): Parameters<ComputeTrendsParams>,
    ) -> Result<String, String> {
        let compute_start = Instant::now();
        let time_series = params.time_series;
        if time_series.is_empty() {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(json!({"data_points": 0, "trend": "insufficient_data"}).to_string());
        }

        let time_field = &params.time_field;
        let value_field = &params.value_field;

        // Extract values
        let mut data: Vec<(String, f64)> = Vec::new();
        for item in &time_series {
            if let Some(value) = item.get(value_field).and_then(safe_numeric) {
                let time = item.get(time_field)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .unwrap_or_default();
                data.push((time, value));
            }
        }

        if data.len() < 2 {
            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
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

        let change_percent = if first_avg != 0.0 {
            (second_avg - first_avg) / first_avg * 100.0
        } else {
            0.0
        };

        let stats = compute_stats(&values);

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = json!({
            "data_points": data.len(),
            "trend": trend,
            "stats": stats,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "change_percent": change_percent,
        }).to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }
}

#[tool_handler]
impl ServerHandler for DataAggregateService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Data Aggregate MCP Server - Aggregates, merges, and summarizes structured data. Used for log analysis, research results, and data processing.".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
