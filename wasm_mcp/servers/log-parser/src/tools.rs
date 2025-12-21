//! Log Parser tools - Pure business logic
//!
//! Shared between CLI and HTTP transports.

use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use wasmmcp::timing::{measure_io, ToolTimer, get_wasm_total_ms};

// ==========================================
// Helper functions
// ==========================================

fn severity_level(level: &str) -> i32 {
    match level.to_lowercase().as_str() {
        "debug" => 0,
        "info" => 1,
        "notice" => 2,
        "warning" | "warn" => 3,
        "error" | "err" => 4,
        "critical" | "crit" => 5,
        "alert" => 6,
        "emergency" | "emerg" => 7,
        _ => 1,
    }
}

fn get_log_patterns() -> Vec<(&'static str, Regex)> {
    vec![
        ("apache_combined", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("apache_common", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)"#).unwrap()),
        ("nginx", Regex::new(r#"(?P<ip>\S+) - \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("syslog", Regex::new(r"(?P<time>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(?P<host>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)").unwrap()),
        ("python", Regex::new(r"(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<logger>\S+)\s+-\s+(?P<level>\S+)\s+-\s+(?P<message>.*)").unwrap()),
    ]
}

fn detect_format(lines: &[&str]) -> String {
    let patterns = get_log_patterns();

    for line in lines.iter().take(10) {
        let line = line.trim();
        if line.is_empty() { continue; }

        if serde_json::from_str::<Value>(line).is_ok() {
            return "json".to_string();
        }

        for (name, pattern) in &patterns {
            if pattern.is_match(line) {
                return name.to_string();
            }
        }
    }

    "unknown".to_string()
}

fn parse_line(line: &str, format: &str) -> Option<Value> {
    let line = line.trim();
    if line.is_empty() { return None; }

    if format == "json" {
        return serde_json::from_str(line).ok();
    }

    let patterns = get_log_patterns();
    for (name, pattern) in &patterns {
        if *name == format {
            if let Some(caps) = pattern.captures(line) {
                let mut entry = serde_json::Map::new();
                for name in pattern.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        entry.insert(name.to_string(), Value::String(m.as_str().to_string()));
                    }
                }
                return Some(Value::Object(entry));
            }
        }
    }

    Some(json!({"raw": line, "parsed": false}))
}

fn extract_level(entry: &Value) -> String {
    for field in &["level", "severity", "loglevel", "log_level"] {
        if let Some(val) = entry.get(field).and_then(|v| v.as_str()) {
            return val.to_lowercase();
        }
    }

    if let Some(status) = entry.get("status").and_then(|v| v.as_str()).and_then(|s| s.parse::<i32>().ok()) {
        if status >= 500 { return "error".to_string(); }
        if status >= 400 { return "warning".to_string(); }
        return "info".to_string();
    }

    let message = entry.get("message")
        .or_else(|| entry.get("raw"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    for level in &["error", "warning", "warn", "critical", "crit", "debug", "info"] {
        if message.contains(level) {
            return level.to_string();
        }
    }

    "info".to_string()
}

// ==========================================
// Tool implementations
// ==========================================

/// Parse raw log content into structured entries
pub fn parse_logs(
    log_content: &str,
    format_type: &str,
    max_entries: usize,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let lines: Vec<&str> = log_content.lines().collect();

    let format = if format_type == "auto" {
        detect_format(&lines)
    } else {
        format_type.to_string()
    };

    let mut entries = Vec::new();
    let mut errors = 0;

    for line in lines.iter().take(max_entries) {
        if let Some(mut entry) = parse_line(line, &format) {
            let level = extract_level(&entry);
            if let Value::Object(ref mut map) = entry {
                map.insert("_level".to_string(), Value::String(level));
            }
            entries.push(entry);
        } else if !line.trim().is_empty() {
            errors += 1;
        }
    }

    let timing = timer.finish("parse_logs");
    Ok(json!({
        "format_detected": format,
        "total_lines": lines.len(),
        "parsed_count": entries.len(),
        "error_count": errors,
        "entries": entries,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Filter log entries by severity level
pub fn filter_entries(
    entries: &[Value],
    min_level: &str,
    include_levels: Option<&[String]>,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let mut filtered = Vec::new();
    let mut level_counts: HashMap<String, i32> = HashMap::new();

    for entry in entries {
        let level = entry.get("_level")
            .and_then(|v| v.as_str())
            .unwrap_or("info")
            .to_string();

        let include = if let Some(levels) = include_levels {
            levels.iter().any(|l| l.to_lowercase() == level)
        } else {
            severity_level(&level) >= severity_level(min_level)
        };

        if include {
            *level_counts.entry(level).or_insert(0) += 1;
            filtered.push(entry.clone());
        }
    }

    let timing = timer.finish("filter_entries");
    Ok(json!({
        "original_count": entries.len(),
        "filtered_count": filtered.len(),
        "levels_included": level_counts.keys().collect::<Vec<_>>(),
        "by_level": level_counts,
        "entries": filtered,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Compute statistics from parsed log entries
pub fn compute_log_statistics(entries: &[Value]) -> Result<String, String> {
    let timer = ToolTimer::start();
    if entries.is_empty() {
        let timing = timer.finish("compute_log_statistics");
        return Ok(json!({
            "entry_count": 0,
            "by_level": {},
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string());
    }

    let mut level_counts: HashMap<String, i32> = HashMap::new();
    let mut status_counts: HashMap<String, i32> = HashMap::new();
    let mut ip_counts: HashMap<String, i32> = HashMap::new();
    let mut path_counts: HashMap<String, i32> = HashMap::new();

    for entry in entries {
        let level = entry.get("_level")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        *level_counts.entry(level.to_string()).or_insert(0) += 1;

        if let Some(status) = entry.get("status").and_then(|v| v.as_str()) {
            *status_counts.entry(status.to_string()).or_insert(0) += 1;
        }
        if let Some(ip) = entry.get("ip").and_then(|v| v.as_str()) {
            *ip_counts.entry(ip.to_string()).or_insert(0) += 1;
        }
        if let Some(path) = entry.get("path").and_then(|v| v.as_str()) {
            *path_counts.entry(path.to_string()).or_insert(0) += 1;
        }
    }

    let mut top_ips: Vec<_> = ip_counts.into_iter().collect();
    top_ips.sort_by(|a, b| b.1.cmp(&a.1));
    let top_ips: HashMap<_, _> = top_ips.into_iter().take(10).collect();

    let mut top_paths: Vec<_> = path_counts.into_iter().collect();
    top_paths.sort_by(|a, b| b.1.cmp(&a.1));
    let top_paths: HashMap<_, _> = top_paths.into_iter().take(10).collect();

    let timing = timer.finish("compute_log_statistics");
    let mut result = json!({
        "entry_count": entries.len(),
        "by_level": level_counts,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    });

    if !status_counts.is_empty() {
        result["by_status"] = json!(status_counts);
    }
    if !top_ips.is_empty() {
        result["top_ips"] = json!(top_ips);
    }
    if !top_paths.is_empty() {
        result["top_paths"] = json!(top_paths);
    }

    Ok(result.to_string())
}

/// Search log entries by regex pattern
pub fn search_entries(
    entries: &[Value],
    pattern: &str,
    fields: Option<&[String]>,
    case_sensitive: bool,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let default_fields = vec!["message".to_string(), "raw".to_string()];
    let fields = fields.unwrap_or(&default_fields);

    let regex = if case_sensitive {
        Regex::new(pattern)
    } else {
        Regex::new(&format!("(?i){}", pattern))
    }.map_err(|e| format!("Invalid regex: {}", e))?;

    let mut matches = Vec::new();

    for entry in entries {
        for field in fields {
            if let Some(text) = entry.get(field).and_then(|v| v.as_str()) {
                if regex.is_match(text) {
                    matches.push(json!({
                        "entry": entry,
                        "matched_field": field,
                        "matched_text": &text[..text.len().min(200)]
                    }));
                    break;
                }
            }
        }
        if matches.len() >= 100 { break; }
    }

    let timing = timer.finish("search_entries");
    Ok(json!({
        "search_pattern": pattern,
        "fields_searched": fields,
        "total_entries": entries.len(),
        "match_count": matches.len(),
        "matches": matches,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Extract time range information from log entries
pub fn extract_time_range(entries: &[Value]) -> Result<String, String> {
    let timer = ToolTimer::start();
    let mut times = Vec::new();

    for entry in entries {
        if let Some(time) = entry.get("time").or_else(|| entry.get("timestamp")).and_then(|v| v.as_str()) {
            times.push(time.to_string());
        }
    }

    let timing = timer.finish("extract_time_range");
    if times.is_empty() {
        return Ok(json!({
            "has_timestamps": false,
            "entry_count": entries.len(),
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string());
    }

    Ok(json!({
        "has_timestamps": true,
        "entry_count": entries.len(),
        "first_timestamp": times.first(),
        "last_timestamp": times.last(),
        "sample_timestamps": times.iter().take(5).collect::<Vec<_>>(),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}
