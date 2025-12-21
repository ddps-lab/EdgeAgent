//! Filesystem tools - Pure business logic
//!
//! Implements the same 14 tools as the NPM @modelcontextprotocol/server-filesystem

use std::fs;
use std::path::Path;
use serde_json::json;
use wasmmcp::timing::{measure_disk_io, ToolTimer, get_wasm_total_ms};

// ============================================================================
// Helper Functions
// ============================================================================

pub fn head_lines(content: &str, n: usize) -> String {
    content.lines().take(n).collect::<Vec<_>>().join("\n")
}

pub fn tail_lines(content: &str, n: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

pub fn format_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if size >= GB {
        format!("{:.2} GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.2} MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.2} KB", size as f64 / KB as f64)
    } else {
        format!("{} B", size)
    }
}

pub fn get_mime_type(path: &str) -> &'static str {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

// Simple base64 encoding (avoiding external dependencies for WASM size)
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

fn build_directory_tree_json(path: &Path, exclude_patterns: &[String]) -> Result<serde_json::Value, String> {
    let entries: Vec<_> = measure_disk_io(|| fs::read_dir(path))
        .map_err(|e| format!("Failed to read directory: {}", e))?
        .filter_map(|e| e.ok())
        .collect();

    let mut result = Vec::new();

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();

        // Check exclude patterns
        if exclude_patterns.iter().any(|p| {
            if p.contains('*') {
                let pattern = p.replace('*', "");
                name.contains(&pattern)
            } else {
                name == *p || name.contains(p)
            }
        }) {
            continue;
        }

        let entry_path = entry.path();
        let is_dir = entry_path.is_dir();

        let mut entry_json = serde_json::json!({
            "name": name,
            "type": if is_dir { "directory" } else { "file" }
        });

        if is_dir {
            let children = build_directory_tree_json(&entry_path, exclude_patterns)?;
            entry_json["children"] = children;
        }

        result.push(entry_json);
    }

    Ok(serde_json::Value::Array(result))
}

fn search_files_recursive(dir: &Path, pattern: &str, exclude_patterns: &[String], results: &mut Vec<String>) -> Result<(), String> {
    let entries = measure_disk_io(|| fs::read_dir(dir))
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Check exclude patterns
        if exclude_patterns.iter().any(|p| name.contains(p)) {
            continue;
        }

        // Glob pattern matching
        let matches = if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                name.starts_with(parts[0]) && name.ends_with(parts[1])
            } else if pattern.starts_with("**/*.") {
                let ext = &pattern[4..];
                name.ends_with(ext)
            } else if pattern.starts_with('*') {
                name.ends_with(&pattern[1..])
            } else if pattern.ends_with('*') {
                name.starts_with(&pattern[..pattern.len()-1])
            } else {
                name.contains(&pattern.replace('*', ""))
            }
        } else {
            name.contains(pattern)
        };

        if matches {
            results.push(path.to_string_lossy().to_string());
        }

        if path.is_dir() {
            search_files_recursive(&path, pattern, exclude_patterns, results)?;
        }
    }

    Ok(())
}

// ============================================================================
// Tool Implementations (14 tools matching NPM version)
// ============================================================================

/// 1. read_file (deprecated)
pub fn read_file(path: &str, head: Option<usize>, tail: Option<usize>) -> Result<String, String> {
    let timer = ToolTimer::start();
    let content = measure_disk_io(|| fs::read_to_string(path))
        .map_err(|e| format!("Failed to read file: {}", e))?;

    if head.is_some() && tail.is_some() {
        return Err("Cannot specify both head and tail parameters simultaneously".to_string());
    }

    let result = if let Some(n) = head {
        head_lines(&content, n)
    } else if let Some(n) = tail {
        tail_lines(&content, n)
    } else {
        content
    };

    let timing = timer.finish("read_file");
    Ok(json!({
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

/// 2. read_text_file
pub fn read_text_file(path: &str, head: Option<usize>, tail: Option<usize>) -> Result<String, String> {
    read_file(path, head, tail)
}

/// 3. read_media_file
pub fn read_media_file(path: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    let data = measure_disk_io(|| fs::read(path))
        .map_err(|e| format!("Failed to read media file: {}", e))?;

    let base64_data = base64_encode(&data);
    let mime_type = get_mime_type(path);

    let timing = timer.finish("read_media_file");
    Ok(json!({
        "content": format!("data:{};base64,{}", mime_type, base64_data),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 4. read_multiple_files
pub fn read_multiple_files(paths: &[String]) -> Result<String, String> {
    let timer = ToolTimer::start();
    let results: Vec<String> = paths.iter().map(|path| {
        match measure_disk_io(|| fs::read_to_string(path)) {
            Ok(content) => format!("{}:\n{}", path, content),
            Err(e) => format!("{}: Error - {}", path, e),
        }
    }).collect();

    let timing = timer.finish("read_multiple_files");
    Ok(json!({
        "content": results.join("\n---\n"),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 5. write_file
pub fn write_file(path: &str, content: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    measure_disk_io(|| fs::write(path, content))
        .map_err(|e| format!("Failed to write file: {}", e))?;

    let timing = timer.finish("write_file");
    Ok(json!({
        "content": format!("Successfully wrote to {}", path),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Edit operation struct
pub struct EditOp {
    pub old_text: String,
    pub new_text: String,
}

/// 6. edit_file
pub fn edit_file(path: &str, edits: &[EditOp], dry_run: bool) -> Result<String, String> {
    let timer = ToolTimer::start();
    let original = measure_disk_io(|| fs::read_to_string(path))
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let mut content = original.clone();
    let mut changes = Vec::new();

    for edit in edits {
        if content.contains(&edit.old_text) {
            content = content.replace(&edit.old_text, &edit.new_text);
            changes.push(format!("- {}\n+ {}", edit.old_text, edit.new_text));
        } else {
            return Err(format!("Text not found: '{}'", edit.old_text));
        }
    }

    let result = if dry_run {
        format!("Dry run - changes that would be made:\n{}", changes.join("\n"))
    } else {
        measure_disk_io(|| fs::write(path, &content))
            .map_err(|e| format!("Failed to write file: {}", e))?;
        format!("Applied {} edit(s) to {}", edits.len(), path)
    };

    let timing = timer.finish("edit_file");
    Ok(json!({
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

/// 7. create_directory
pub fn create_directory(path: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    measure_disk_io(|| fs::create_dir_all(path))
        .map_err(|e| format!("Failed to create directory: {}", e))?;

    let timing = timer.finish("create_directory");
    Ok(json!({
        "content": format!("Successfully created directory {}", path),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 8. list_directory
pub fn list_directory(path: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    let entries = measure_disk_io(|| fs::read_dir(path))
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    let mut items = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;

        let entry_path = entry.path();
        let file_type = if entry_path.is_dir() { "[DIR]" } else { "[FILE]" };
        let name = entry.file_name().to_string_lossy().to_string();

        items.push(format!("{} {}", file_type, name));
    }

    items.sort();

    let result = if items.is_empty() {
        "Directory is empty".to_string()
    } else {
        items.join("\n")
    };

    let timing = timer.finish("list_directory");
    Ok(json!({
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

/// 9. list_directory_with_sizes
pub fn list_directory_with_sizes(path: &str, sort_by: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    let entries = measure_disk_io(|| fs::read_dir(path))
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    let mut items: Vec<(String, bool, u64)> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry_path.is_dir();
        let size = if is_dir {
            0
        } else {
            measure_disk_io(|| fs::metadata(&entry_path)).map(|m| m.len()).unwrap_or(0)
        };
        items.push((name, is_dir, size));
    }

    // Sort by name or size
    if sort_by == "size" {
        items.sort_by(|a, b| b.2.cmp(&a.2));
    } else {
        items.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let mut result = Vec::new();
    let mut total_size: u64 = 0;
    let mut file_count = 0;
    let mut dir_count = 0;

    for (name, is_dir, size) in &items {
        let file_type = if *is_dir { "[DIR]" } else { "[FILE]" };
        let size_str = if *is_dir {
            "".to_string()
        } else {
            format_size(*size)
        };
        result.push(format!("{} {:30} {:>10}", file_type, name, size_str));

        if *is_dir {
            dir_count += 1;
        } else {
            file_count += 1;
            total_size += size;
        }
    }

    result.push(String::new());
    result.push(format!("Total: {} files, {} directories", file_count, dir_count));
    result.push(format!("Combined size: {}", format_size(total_size)));

    let timing = timer.finish("list_directory_with_sizes");
    Ok(json!({
        "content": result.join("\n"),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 10. directory_tree
pub fn directory_tree(path: &str, exclude_patterns: &[String]) -> Result<String, String> {
    let timer = ToolTimer::start();
    let path = Path::new(path);
    let tree = build_directory_tree_json(path, exclude_patterns)?;
    let timing = timer.finish("directory_tree");
    Ok(json!({
        "tree": tree,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 11. move_file
pub fn move_file(source: &str, destination: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    measure_disk_io(|| fs::rename(source, destination))
        .map_err(|e| format!("Failed to move file: {}", e))?;

    let timing = timer.finish("move_file");
    Ok(json!({
        "message": format!("Successfully moved {} to {}", source, destination),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 12. search_files
pub fn search_files(path: &str, pattern: &str, exclude_patterns: &[String]) -> Result<String, String> {
    let timer = ToolTimer::start();
    let path = Path::new(path);
    let mut results = Vec::new();

    search_files_recursive(path, pattern, exclude_patterns, &mut results)?;

    let timing = timer.finish("search_files");
    Ok(json!({
        "matches": results,
        "count": results.len(),
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// 13. get_file_info
pub fn get_file_info(path: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    let path = Path::new(path);
    let metadata = measure_disk_io(|| fs::metadata(path))
        .map_err(|e| format!("Failed to get file info: {}", e))?;

    let file_type = if metadata.is_dir() {
        "directory"
    } else if metadata.is_file() {
        "file"
    } else if metadata.is_symlink() {
        "symlink"
    } else {
        "other"
    };

    let mut info = json!({
        "size": metadata.len(),
        "type": file_type,
        "readonly": metadata.permissions().readonly()
    });

    // Try to get timestamps (may not be available on all platforms)
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        info["modified"] = json!(metadata.mtime());
        info["accessed"] = json!(metadata.atime());
        info["created"] = json!(metadata.ctime());
    }

    let timing = timer.finish("get_file_info");
    info["timing"] = json!({
        "wasm_total_ms": get_wasm_total_ms(),
        "fn_total_ms": timing.fn_total_ms,
        "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
        "compute_ms": timing.compute_ms
    });

    Ok(info.to_string())
}

/// 14. list_allowed_directories
pub fn list_allowed_directories() -> String {
    let timer = ToolTimer::start();
    let message = "Allowed directories are controlled by the WASI runtime.\nUse --dir flag when running with wasmtime to specify allowed directories.";
    let timing = timer.finish("list_allowed_directories");
    json!({
        "message": message,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "disk_io_ms": timing.disk_io_ms,
            "network_io_ms": timing.network_io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string()
}
