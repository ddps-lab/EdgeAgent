//! Log Parser MCP Service - parse and analyze log files

use regex::Regex;
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
use serde_json::{json, Value};
use std::collections::HashMap;

/// Log Parser MCP Service
#[derive(Debug, Clone)]
pub struct LogParserService {
    tool_router: ToolRouter<Self>,
}

impl LogParserService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for LogParserService {
    fn default() -> Self {
        Self::new()
    }
}

// Severity levels
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

// Log format patterns
fn get_log_patterns() -> Vec<(&'static str, Regex)> {
    vec![
        ("apache_combined", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("apache_common", Regex::new(r#"(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\S+)"#).unwrap()),
        ("nginx", Regex::new(r#"(?P<ip>\S+) - \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)""#).unwrap()),
        ("syslog", Regex::new(r"(?P<time>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(?P<host>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)").unwrap()),
        ("python", Regex::new(r"(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<logger>\S+)\s+-\s+(?P<level>\S+)\s+-\s+(?P<message>.*)").unwrap()),
    ]
}

// Detect log format
fn detect_format(lines: &[&str]) -> String {
    let patterns = get_log_patterns();

    for line in lines.iter().take(10) {
        let line = line.trim();
        if line.is_empty() { continue; }

        // Try JSON
        if serde_json::from_str::<Value>(line).is_ok() {
            return "json".to_string();
        }

        // Try regex patterns
        for (name, pattern) in &patterns {
            if pattern.is_match(line) {
                return name.to_string();
            }
        }
    }

    "unknown".to_string()
}

// Parse a single line
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

    // Fallback
    Some(json!({"raw": line, "parsed": false}))
}

// Extract level from entry
fn extract_level(entry: &Value) -> String {
    // Check level fields
    for field in &["level", "severity", "loglevel", "log_level"] {
        if let Some(val) = entry.get(field).and_then(|v| v.as_str()) {
            return val.to_lowercase();
        }
    }

    // Check HTTP status
    if let Some(status) = entry.get("status").and_then(|v| v.as_str()).and_then(|s| s.parse::<i32>().ok()) {
        if status >= 500 { return "error".to_string(); }
        if status >= 400 { return "warning".to_string(); }
        return "info".to_string();
    }

    // Check message for level keywords
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

// Parameter structs - matching Python FastMCP log_parser_server.py (snake_case)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ParseLogsParams {
    /// Raw log content (multi-line string)
    #[schemars(description = "Raw log content (multi-line string)")]
    pub log_content: String,

    /// Log format type or "auto" for auto-detection
    #[serde(default = "default_format")]
    #[schemars(description = "Log format: auto, apache_combined, apache_common, nginx, syslog, python, json")]
    pub format_type: String,

    /// Maximum entries to parse
    #[serde(default = "default_max_entries")]
    #[schemars(description = "Maximum entries to parse (default: 1000)")]
    pub max_entries: usize,
}

fn default_format() -> String { "auto".to_string() }
fn default_max_entries() -> usize { 1000 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FilterEntriesParams {
    /// Parsed log entries
    #[schemars(description = "List of parsed log entries from parse_logs")]
    pub entries: Vec<Value>,

    /// Minimum severity level
    #[serde(default = "default_min_level")]
    #[schemars(description = "Minimum severity: debug, info, notice, warning, error, critical, alert, emergency")]
    pub min_level: String,

    /// Specific levels to include
    #[schemars(description = "Specific levels to include (overrides min_level)")]
    pub include_levels: Option<Vec<String>>,
}

fn default_min_level() -> String { "warning".to_string() }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ComputeStatisticsParams {
    /// Parsed log entries
    #[schemars(description = "List of parsed log entries")]
    pub entries: Vec<Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchEntriesParams {
    /// Parsed log entries
    #[schemars(description = "List of parsed log entries")]
    pub entries: Vec<Value>,

    /// Regex pattern to search
    #[schemars(description = "Regex pattern to search for")]
    pub pattern: String,

    /// Fields to search in
    #[schemars(description = "Fields to search in (default: message, raw)")]
    pub fields: Option<Vec<String>>,

    /// Case sensitive search
    #[serde(default)]
    #[schemars(description = "Case sensitive search (default: false)")]
    pub case_sensitive: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExtractTimeRangeParams {
    /// Parsed log entries
    #[schemars(description = "List of parsed log entries")]
    pub entries: Vec<Value>,
}

// Tool implementations
#[tool_router]
impl LogParserService {
    #[tool(description = "Parse raw log content into structured entries. Returns entries with _level field added.")]
    fn parse_logs(&self, Parameters(params): Parameters<ParseLogsParams>) -> Result<String, String> {
        let timer = ToolTimer::start();

        let lines: Vec<&str> = params.log_content.lines().collect();

        // Auto-detect format if needed
        let format = if params.format_type == "auto" {
            detect_format(&lines)
        } else {
            params.format_type.clone()
        };

        let mut entries = Vec::new();
        let mut errors = 0;

        for line in lines.iter().take(params.max_entries) {
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
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    #[tool(description = "Filter log entries by severity level. Pass entries from parse_logs result.")]
    fn filter_entries(&self, Parameters(params): Parameters<FilterEntriesParams>) -> Result<String, String> {
        let timer = ToolTimer::start();

        let mut filtered = Vec::new();
        let mut level_counts: HashMap<String, i32> = HashMap::new();

        for entry in &params.entries {
            let level = entry.get("_level")
                .and_then(|v| v.as_str())
                .unwrap_or("info")
                .to_string();

            let include = if let Some(ref levels) = params.include_levels {
                levels.iter().any(|l| l.to_lowercase() == level)
            } else {
                severity_level(&level) >= severity_level(&params.min_level)
            };

            if include {
                *level_counts.entry(level).or_insert(0) += 1;
                filtered.push(entry.clone());
            }
        }

        let timing = timer.finish("filter_entries");
        Ok(json!({
            "original_count": params.entries.len(),
            "filtered_count": filtered.len(),
            "levels_included": level_counts.keys().collect::<Vec<_>>(),
            "by_level": level_counts,
            "entries": filtered,
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    #[tool(description = "Compute statistics from parsed log entries.")]
    fn compute_log_statistics(&self, Parameters(params): Parameters<ComputeStatisticsParams>) -> Result<String, String> {
        let timer = ToolTimer::start();

        if params.entries.is_empty() {
            let timing = timer.finish("compute_log_statistics");
            return Ok(json!({
                "entry_count": 0,
                "by_level": {},
                "_timing": {
                    "wasm_total_ms": get_wasm_total_ms(),
                    "fn_total_ms": timing.fn_total_ms,
                    "io_ms": timing.io_ms,
                    "compute_ms": timing.compute_ms
                }
            }).to_string());
        }

        let mut level_counts: HashMap<String, i32> = HashMap::new();
        let mut status_counts: HashMap<String, i32> = HashMap::new();
        let mut ip_counts: HashMap<String, i32> = HashMap::new();
        let mut path_counts: HashMap<String, i32> = HashMap::new();

        for entry in &params.entries {
            // Count by level
            let level = entry.get("_level")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            *level_counts.entry(level.to_string()).or_insert(0) += 1;

            // HTTP stats
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

        // Get top 10 for each
        let mut top_ips: Vec<_> = ip_counts.into_iter().collect();
        top_ips.sort_by(|a, b| b.1.cmp(&a.1));
        let top_ips: HashMap<_, _> = top_ips.into_iter().take(10).collect();

        let mut top_paths: Vec<_> = path_counts.into_iter().collect();
        top_paths.sort_by(|a, b| b.1.cmp(&a.1));
        let top_paths: HashMap<_, _> = top_paths.into_iter().take(10).collect();

        let mut result = json!({
            "entry_count": params.entries.len(),
            "by_level": level_counts,
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

        let timing = timer.finish("compute_log_statistics");
        result["_timing"] = json!({
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        });

        Ok(result.to_string())
    }

    #[tool(description = "Search log entries by regex pattern.")]
    fn search_entries(&self, Parameters(params): Parameters<SearchEntriesParams>) -> Result<String, String> {
        let timer = ToolTimer::start();

        let fields = params.fields.unwrap_or_else(|| vec!["message".to_string(), "raw".to_string()]);

        let regex = if params.case_sensitive {
            Regex::new(&params.pattern)
        } else {
            Regex::new(&format!("(?i){}", params.pattern))
        }.map_err(|e| format!("Invalid regex: {}", e))?;

        let mut matches = Vec::new();

        for entry in &params.entries {
            for field in &fields {
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
            "search_pattern": params.pattern,
            "fields_searched": fields,
            "total_entries": params.entries.len(),
            "match_count": matches.len(),
            "matches": matches,
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    #[tool(description = "Extract time range information from log entries.")]
    fn extract_time_range(&self, Parameters(params): Parameters<ExtractTimeRangeParams>) -> Result<String, String> {
        let timer = ToolTimer::start();

        let mut times = Vec::new();

        for entry in &params.entries {
            if let Some(time) = entry.get("time").or_else(|| entry.get("timestamp")).and_then(|v| v.as_str()) {
                times.push(time.to_string());
            }
        }

        if times.is_empty() {
            let timing = timer.finish("extract_time_range");
            return Ok(json!({
                "has_timestamps": false,
                "entry_count": params.entries.len(),
                "_timing": {
                    "wasm_total_ms": get_wasm_total_ms(),
                    "fn_total_ms": timing.fn_total_ms,
                    "io_ms": timing.io_ms,
                    "compute_ms": timing.compute_ms
                }
            }).to_string());
        }

        let timing = timer.finish("extract_time_range");
        Ok(json!({
            "has_timestamps": true,
            "entry_count": params.entries.len(),
            "first_timestamp": times.first(),
            "last_timestamp": times.last(),
            "sample_timestamps": times.iter().take(5).collect::<Vec<_>>(),
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
impl ServerHandler for LogParserService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Log Parser MCP Server - Parse various log formats (Apache, nginx, syslog, Python, JSON) and analyze entries.".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
