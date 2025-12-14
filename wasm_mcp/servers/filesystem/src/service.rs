//! Filesystem service implementation
//!
//! Implements the same 14 tools as the NPM @modelcontextprotocol/server-filesystem

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
use std::fs;
use std::path::Path;

/// Filesystem MCP Service
#[derive(Debug, Clone)]
pub struct FilesystemService {
    tool_router: ToolRouter<Self>,
}

impl FilesystemService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for FilesystemService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tool Parameters (matching NPM version exactly)
// ============================================================================

// 1. read_file (deprecated)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadFileParams {
    #[schemars(description = "Path to the file to read")]
    pub path: String,
    #[schemars(description = "If provided, returns only the last N lines of the file")]
    pub tail: Option<usize>,
    #[schemars(description = "If provided, returns only the first N lines of the file")]
    pub head: Option<usize>,
}

// 2. read_text_file
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadTextFileParams {
    #[schemars(description = "Path to the file to read")]
    pub path: String,
    #[schemars(description = "If provided, returns only the last N lines of the file")]
    pub tail: Option<usize>,
    #[schemars(description = "If provided, returns only the first N lines of the file")]
    pub head: Option<usize>,
}

// 3. read_media_file
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadMediaFileParams {
    #[schemars(description = "Path to the media file to read")]
    pub path: String,
}

// 4. read_multiple_files
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadMultipleFilesParams {
    #[schemars(description = "Array of file paths to read")]
    pub paths: Vec<String>,
}

// 5. write_file
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WriteFileParams {
    #[schemars(description = "Path to the file to write")]
    pub path: String,
    #[schemars(description = "Content to write to the file")]
    pub content: String,
}

// 6. edit_file (matching Node.js camelCase schema)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EditOperation {
    #[serde(rename = "oldText")]
    #[schemars(description = "Text to search for - must match exactly")]
    pub old_text: String,
    #[serde(rename = "newText")]
    #[schemars(description = "Text to replace with")]
    pub new_text: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EditFileParams {
    #[schemars(description = "Path to the file to edit")]
    pub path: String,
    #[schemars(description = "Array of edit operations to apply")]
    pub edits: Vec<EditOperation>,
    #[serde(rename = "dryRun")]
    #[schemars(description = "Preview changes using git-style diff format")]
    #[serde(default)]
    pub dry_run: bool,
}

// 7. create_directory
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateDirectoryParams {
    #[schemars(description = "Path of the directory to create")]
    pub path: String,
}

// 8. list_directory
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListDirectoryParams {
    #[schemars(description = "Path to the directory to list")]
    pub path: String,
}

// 9. list_directory_with_sizes (matching Node.js camelCase schema)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListDirectoryWithSizesParams {
    #[schemars(description = "Path to the directory to list")]
    pub path: String,
    #[serde(rename = "sortBy")]
    #[schemars(description = "Sort entries by name or size")]
    #[serde(default = "default_sort_by")]
    pub sort_by: String,
}

fn default_sort_by() -> String {
    "name".to_string()
}

// 10. directory_tree (matching Node.js camelCase schema)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DirectoryTreeParams {
    #[schemars(description = "Path to the directory")]
    pub path: String,
    #[serde(rename = "excludePatterns")]
    #[schemars(description = "Glob patterns to exclude from the tree")]
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

// 11. move_file
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MoveFileParams {
    #[schemars(description = "Source path")]
    pub source: String,
    #[schemars(description = "Destination path")]
    pub destination: String,
}

// 12. search_files (matching Node.js camelCase schema)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchFilesParams {
    #[schemars(description = "Directory to search in")]
    pub path: String,
    #[schemars(description = "Glob pattern to match files")]
    pub pattern: String,
    #[serde(rename = "excludePatterns")]
    #[schemars(description = "Patterns to exclude")]
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

// 13. get_file_info
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetFileInfoParams {
    #[schemars(description = "Path to get information about")]
    pub path: String,
}

// 14. list_allowed_directories (no params)

// ============================================================================
// Helper Functions
// ============================================================================

fn head_lines(content: &str, n: usize) -> String {
    content.lines().take(n).collect::<Vec<_>>().join("\n")
}

fn tail_lines(content: &str, n: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

fn format_size(size: u64) -> String {
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

fn get_mime_type(path: &str) -> &'static str {
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

fn build_directory_tree_json(path: &Path, exclude_patterns: &[String]) -> Result<serde_json::Value, String> {
    let entries: Vec<_> = fs::read_dir(path)
        .map_err(|e| format!("Failed to read directory: {}", e))?
        .filter_map(|e| e.ok())
        .collect();

    let mut result = Vec::new();

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();

        // Check exclude patterns
        if exclude_patterns.iter().any(|p| {
            if p.contains('*') {
                // Simple glob matching
                let pattern = p.replace("*", "");
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
    let entries = fs::read_dir(dir)
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
                // Handle **/*.ext pattern
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
// Tool Implementation (14 tools matching NPM version)
// ============================================================================

#[tool_router]
impl FilesystemService {
    // 1. read_file (deprecated)
    #[tool(description = "Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.")]
    fn read_file(&self, Parameters(params): Parameters<ReadFileParams>) -> Result<String, String> {
        let content = fs::read_to_string(&params.path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        if params.head.is_some() && params.tail.is_some() {
            return Err("Cannot specify both head and tail parameters simultaneously".to_string());
        }

        if let Some(n) = params.head {
            Ok(head_lines(&content, n))
        } else if let Some(n) = params.tail {
            Ok(tail_lines(&content, n))
        } else {
            Ok(content)
        }
    }

    // 2. read_text_file
    #[tool(description = "Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Only works within allowed directories.")]
    fn read_text_file(&self, Parameters(params): Parameters<ReadTextFileParams>) -> Result<String, String> {
        let content = fs::read_to_string(&params.path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        if params.head.is_some() && params.tail.is_some() {
            return Err("Cannot specify both head and tail parameters simultaneously".to_string());
        }

        if let Some(n) = params.head {
            Ok(head_lines(&content, n))
        } else if let Some(n) = params.tail {
            Ok(tail_lines(&content, n))
        } else {
            Ok(content)
        }
    }

    // 3. read_media_file
    #[tool(description = "Read an image or audio file. Returns the base64 encoded data and MIME type. Only works within allowed directories.")]
    fn read_media_file(&self, Parameters(params): Parameters<ReadMediaFileParams>) -> Result<String, String> {
        let data = fs::read(&params.path)
            .map_err(|e| format!("Failed to read media file: {}", e))?;

        let base64_data = base64_encode(&data);
        let mime_type = get_mime_type(&params.path);

        Ok(format!("data:{};base64,{}", mime_type, base64_data))
    }

    // 4. read_multiple_files
    #[tool(description = "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.")]
    fn read_multiple_files(&self, Parameters(params): Parameters<ReadMultipleFilesParams>) -> Result<String, String> {
        let results: Vec<String> = params.paths.iter().map(|path| {
            match fs::read_to_string(path) {
                Ok(content) => format!("{}:\n{}", path, content),
                Err(e) => format!("{}: Error - {}", path, e),
            }
        }).collect();

        Ok(results.join("\n---\n"))
    }

    // 5. write_file
    #[tool(description = "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.")]
    fn write_file(&self, Parameters(params): Parameters<WriteFileParams>) -> Result<String, String> {
        fs::write(&params.path, &params.content)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(format!("Successfully wrote to {}", params.path))
    }

    // 6. edit_file
    #[tool(description = "Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.")]
    fn edit_file(&self, Parameters(params): Parameters<EditFileParams>) -> Result<String, String> {
        let original = fs::read_to_string(&params.path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut content = original.clone();
        let mut changes = Vec::new();

        for edit in &params.edits {
            if content.contains(&edit.old_text) {
                content = content.replace(&edit.old_text, &edit.new_text);
                changes.push(format!("- {}\n+ {}", edit.old_text, edit.new_text));
            } else {
                return Err(format!("Text not found: '{}'", edit.old_text));
            }
        }

        if params.dry_run {
            Ok(format!("Dry run - changes that would be made:\n{}", changes.join("\n")))
        } else {
            fs::write(&params.path, &content)
                .map_err(|e| format!("Failed to write file: {}", e))?;
            Ok(format!("Applied {} edit(s) to {}", params.edits.len(), params.path))
        }
    }

    // 7. create_directory
    #[tool(description = "Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.")]
    fn create_directory(&self, Parameters(params): Parameters<CreateDirectoryParams>) -> Result<String, String> {
        fs::create_dir_all(&params.path)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        Ok(format!("Successfully created directory {}", params.path))
    }

    // 8. list_directory
    #[tool(description = "Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.")]
    fn list_directory(&self, Parameters(params): Parameters<ListDirectoryParams>) -> Result<String, String> {
        let entries = fs::read_dir(&params.path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        let mut items = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;

            let path = entry.path();
            let file_type = if path.is_dir() { "[DIR]" } else { "[FILE]" };
            let name = entry.file_name().to_string_lossy().to_string();

            items.push(format!("{} {}", file_type, name));
        }

        items.sort();

        Ok(if items.is_empty() {
            "Directory is empty".to_string()
        } else {
            items.join("\n")
        })
    }

    // 9. list_directory_with_sizes
    #[tool(description = "Get a detailed listing of all files and directories in a specified path, including sizes. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is useful for understanding directory structure and finding specific files within a directory. Only works within allowed directories.")]
    fn list_directory_with_sizes(&self, Parameters(params): Parameters<ListDirectoryWithSizesParams>) -> Result<String, String> {
        let entries = fs::read_dir(&params.path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        let mut items: Vec<(String, bool, u64)> = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = path.is_dir();
            let size = if is_dir {
                0
            } else {
                fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            };
            items.push((name, is_dir, size));
        }

        // Sort by name or size
        if params.sort_by == "size" {
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

        Ok(result.join("\n"))
    }

    // 10. directory_tree
    #[tool(description = "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.")]
    fn directory_tree(&self, Parameters(params): Parameters<DirectoryTreeParams>) -> Result<String, String> {
        let path = Path::new(&params.path);
        let tree = build_directory_tree_json(path, &params.exclude_patterns)?;
        Ok(serde_json::to_string_pretty(&tree).unwrap_or_else(|_| "[]".to_string()))
    }

    // 11. move_file
    #[tool(description = "Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Works across different directories and can be used for simple renaming within the same directory. Both source and destination must be within allowed directories.")]
    fn move_file(&self, Parameters(params): Parameters<MoveFileParams>) -> Result<String, String> {
        fs::rename(&params.source, &params.destination)
            .map_err(|e| format!("Failed to move file: {}", e))?;

        Ok(format!("Successfully moved {} to {}", params.source, params.destination))
    }

    // 12. search_files
    #[tool(description = "Recursively search for files and directories matching a pattern. The patterns should be glob-style patterns that match paths relative to the working directory. Use pattern like '*.ext' to match files in current directory, and '**/*.ext' to match files in all subdirectories. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.")]
    fn search_files(&self, Parameters(params): Parameters<SearchFilesParams>) -> Result<String, String> {
        let path = Path::new(&params.path);
        let mut results = Vec::new();

        search_files_recursive(path, &params.pattern, &params.exclude_patterns, &mut results)?;

        if results.is_empty() {
            Ok("No matches found".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }

    // 13. get_file_info
    #[tool(description = "Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.")]
    fn get_file_info(&self, Parameters(params): Parameters<GetFileInfoParams>) -> Result<String, String> {
        let path = Path::new(&params.path);
        let metadata = fs::metadata(path)
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

        let mut info = Vec::new();
        info.push(format!("size: {}", metadata.len()));
        info.push(format!("type: {}", file_type));
        info.push(format!("readonly: {}", metadata.permissions().readonly()));

        // Try to get timestamps (may not be available on all platforms)
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            info.push(format!("modified: {}", metadata.mtime()));
            info.push(format!("accessed: {}", metadata.atime()));
            info.push(format!("created: {}", metadata.ctime()));
        }

        Ok(info.join("\n"))
    }

    // 14. list_allowed_directories
    #[tool(description = "Returns the list of directories that this server is allowed to access. Subdirectories within these allowed directories are also accessible. Use this to understand which directories and their nested paths are available before trying to access files.")]
    fn list_allowed_directories(&self) -> String {
        // In WASM/WASI context, allowed directories are controlled by the runtime via --dir flag
        "Allowed directories are controlled by the WASI runtime.\nUse --dir flag when running with wasmtime to specify allowed directories.".to_string()
    }
}

// Simple base64 encoding (avoiding external dependencies for WASM size)
fn base64_encode(data: &[u8]) -> String {
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

// ============================================================================
// Server Handler Implementation
// ============================================================================

#[tool_handler]
impl ServerHandler for FilesystemService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "A WASM-based filesystem MCP server. Provides read, write, list, search, and other operations for files and directories. Compatible with @modelcontextprotocol/server-filesystem.".to_string()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
