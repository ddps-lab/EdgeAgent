//! Filesystem MCP Server - WASM compatible (wasip2)
//!
//! A stateless MCP server that provides filesystem operations.
//! Implements the same 14 tools as the NPM @modelcontextprotocol/server-filesystem
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run --dir /path)
//! - `cargo build --features http-export` → HTTP server (wasmtime serve)
//!
//! Both use the same `create_server()` function with shared business logic.

pub mod tools;

// Keep service module for backward compatibility (rmcp-based)
#[cfg(feature = "rmcp-service")]
pub mod service;

use wasmmcp::schemars::JsonSchema;
use wasmmcp::serde::Deserialize;
use wasmmcp::prelude::*;

// ============================================================================
// Tool Parameters (matching NPM version exactly)
// ============================================================================

fn default_sort_by() -> String {
    "name".to_string()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadFileParams {
    /// Path to the file to read
    pub path: String,
    /// If provided, returns only the last N lines of the file
    pub tail: Option<usize>,
    /// If provided, returns only the first N lines of the file
    pub head: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadTextFileParams {
    /// Path to the file to read
    pub path: String,
    /// If provided, returns only the last N lines of the file
    pub tail: Option<usize>,
    /// If provided, returns only the first N lines of the file
    pub head: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadMediaFileParams {
    /// Path to the media file to read
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadMultipleFilesParams {
    /// Array of file paths to read
    pub paths: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteFileParams {
    /// Path to the file to write
    pub path: String,
    /// Content to write to the file
    pub content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditOperation {
    /// Text to search for - must match exactly
    #[serde(rename = "oldText")]
    pub old_text: String,
    /// Text to replace with
    #[serde(rename = "newText")]
    pub new_text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditFileParams {
    /// Path to the file to edit
    pub path: String,
    /// Array of edit operations to apply
    pub edits: Vec<EditOperation>,
    /// Preview changes using git-style diff format
    #[serde(rename = "dryRun")]
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateDirectoryParams {
    /// Path of the directory to create
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListDirectoryParams {
    /// Path to the directory to list
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListDirectoryWithSizesParams {
    /// Path to the directory to list
    pub path: String,
    /// Sort entries by name or size
    #[serde(rename = "sortBy")]
    #[serde(default = "default_sort_by")]
    pub sort_by: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DirectoryTreeParams {
    /// Path to the directory
    pub path: String,
    /// Glob patterns to exclude from the tree
    #[serde(rename = "excludePatterns")]
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoveFileParams {
    /// Source path
    pub source: String,
    /// Destination path
    pub destination: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchFilesParams {
    /// Directory to search in
    pub path: String,
    /// Glob pattern to match files
    pub pattern: String,
    /// Patterns to exclude
    #[serde(rename = "excludePatterns")]
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFileInfoParams {
    /// Path to get information about
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmptyParams {}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-filesystem")
        .version("1.0.0")
        .description("Filesystem MCP Server - WASM-based file operations compatible with @modelcontextprotocol/server-filesystem")
        // 1. read_file (deprecated)
        .tool::<ReadFileParams, _>(
            "read_file",
            "Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.",
            |params| tools::read_file(&params.path, params.head, params.tail)
        )
        // 2. read_text_file
        .tool::<ReadTextFileParams, _>(
            "read_text_file",
            "Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read.",
            |params| tools::read_text_file(&params.path, params.head, params.tail)
        )
        // 3. read_media_file
        .tool::<ReadMediaFileParams, _>(
            "read_media_file",
            "Read an image or audio file. Returns the base64 encoded data and MIME type.",
            |params| tools::read_media_file(&params.path)
        )
        // 4. read_multiple_files
        .tool::<ReadMultipleFilesParams, _>(
            "read_multiple_files",
            "Read the contents of multiple files simultaneously. More efficient than reading files one by one.",
            |params| tools::read_multiple_files(&params.paths)
        )
        // 5. write_file
        .tool::<WriteFileParams, _>(
            "write_file",
            "Create a new file or completely overwrite an existing file with new content.",
            |params| tools::write_file(&params.path, &params.content)
        )
        // 6. edit_file
        .tool::<EditFileParams, _>(
            "edit_file",
            "Make line-based edits to a text file. Each edit replaces exact line sequences with new content.",
            |params| {
                let edits: Vec<tools::EditOp> = params.edits.iter().map(|e| {
                    tools::EditOp {
                        old_text: e.old_text.clone(),
                        new_text: e.new_text.clone(),
                    }
                }).collect();
                tools::edit_file(&params.path, &edits, params.dry_run)
            }
        )
        // 7. create_directory
        .tool::<CreateDirectoryParams, _>(
            "create_directory",
            "Create a new directory or ensure a directory exists. Can create multiple nested directories.",
            |params| tools::create_directory(&params.path)
        )
        // 8. list_directory
        .tool::<ListDirectoryParams, _>(
            "list_directory",
            "Get a detailed listing of all files and directories in a specified path.",
            |params| tools::list_directory(&params.path)
        )
        // 9. list_directory_with_sizes
        .tool::<ListDirectoryWithSizesParams, _>(
            "list_directory_with_sizes",
            "Get a detailed listing of all files and directories in a specified path, including sizes.",
            |params| tools::list_directory_with_sizes(&params.path, &params.sort_by)
        )
        // 10. directory_tree
        .tool::<DirectoryTreeParams, _>(
            "directory_tree",
            "Get a recursive tree view of files and directories as a JSON structure.",
            |params| tools::directory_tree(&params.path, &params.exclude_patterns)
        )
        // 11. move_file
        .tool::<MoveFileParams, _>(
            "move_file",
            "Move or rename files and directories.",
            |params| tools::move_file(&params.source, &params.destination)
        )
        // 12. search_files
        .tool::<SearchFilesParams, _>(
            "search_files",
            "Recursively search for files and directories matching a pattern.",
            |params| tools::search_files(&params.path, &params.pattern, &params.exclude_patterns)
        )
        // 13. get_file_info
        .tool::<GetFileInfoParams, _>(
            "get_file_info",
            "Retrieve detailed metadata about a file or directory.",
            |params| tools::get_file_info(&params.path)
        )
        // 14. list_allowed_directories
        .tool::<EmptyParams, _>(
            "list_allowed_directories",
            "Returns the list of directories that this server is allowed to access.",
            |_params| Ok(tools::list_allowed_directories())
        )
        .build()
}

// ==========================================
// CLI Export (wasmtime run --dir /path)
// ==========================================

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// ==========================================
// HTTP Export (wasmtime serve)
// ==========================================

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
