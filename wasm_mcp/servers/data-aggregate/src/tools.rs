//! Data Aggregate tools - Pure business logic
//!
//! Shared between CLI and HTTP transports.

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;

// ==========================================
// Helper functions
// ==========================================

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

// ==========================================
// Tool implementations
// ==========================================

/// Aggregate a list of dictionaries by grouping, counting, or summing
pub fn aggregate_list(
    items: &[Value],
    group_by: Option<&str>,
    count_field: Option<&str>,
    sum_fields: Option<&[String]>,
) -> Result<String, String> {
    if items.is_empty() {
        return Ok(json!({"total_count": 0, "groups": {}}).to_string());
    }

    let input_size = serde_json::to_string(&items).map(|s| s.len()).unwrap_or(0);
    let mut result = json!({
        "total_count": items.len(),
        "input_size_estimate": input_size,
    });

    // Group by field
    if let Some(group_by) = group_by {
        let mut groups: HashMap<String, Vec<&Value>> = HashMap::new();
        for item in items {
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
    if let Some(count_field) = count_field {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for item in items {
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
    if let Some(sum_fields) = sum_fields {
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

    let output_size = serde_json::to_string(&result).map(|s| s.len()).unwrap_or(0);
    result["output_size_estimate"] = json!(output_size);
    result["reduction_ratio"] = json!(if input_size > 0 { output_size as f64 / input_size as f64 } else { 0.0 });

    Ok(result.to_string())
}

/// Merge multiple summary dictionaries into one
pub fn merge_summaries(
    summaries: &[Value],
    weights: Option<&[f64]>,
) -> Result<String, String> {
    if summaries.is_empty() {
        return Ok(json!({"merged_count": 0}).to_string());
    }

    let default_weights: Vec<f64> = vec![1.0; summaries.len()];
    let weights = weights.unwrap_or(&default_weights);

    // Collect all keys
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

    Ok(merged.to_string())
}

/// Combine multiple research/search results into a coherent summary
pub fn combine_research_results(
    results: &[Value],
    title_field: &str,
    summary_field: &str,
    score_field: Option<&str>,
) -> Result<String, String> {
    if results.is_empty() {
        return Ok(json!({"result_count": 0, "combined_summary": ""}).to_string());
    }

    let score_field = score_field.unwrap_or("relevance_score");

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

/// Remove duplicate items based on key fields
pub fn deduplicate(
    items: &[Value],
    key_fields: &[String],
    keep: &str,
) -> Result<String, String> {
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
    let duplicates_removed = items.len() - unique_count;

    Ok(json!({
        "original_count": items.len(),
        "unique_count": unique_count,
        "duplicates_removed": duplicates_removed,
        "key_fields": key_fields,
        "items": result,
    }).to_string())
}

/// Compute trends from time-series data
pub fn compute_trends(
    time_series: &[Value],
    time_field: &str,
    value_field: &str,
    _bucket_count: usize,
) -> Result<String, String> {
    if time_series.is_empty() {
        return Ok(json!({"data_points": 0, "trend": "insufficient_data"}).to_string());
    }

    // Extract values
    let mut data: Vec<(String, f64)> = Vec::new();
    for item in time_series {
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

    Ok(json!({
        "data_points": data.len(),
        "trend": trend,
        "stats": stats,
        "first_half_avg": first_avg,
        "second_half_avg": second_avg,
        "change_percent": change_percent,
    }).to_string())
}
