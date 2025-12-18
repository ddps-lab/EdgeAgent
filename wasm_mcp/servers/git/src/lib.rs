//! Git MCP Server - WASM compatible (wasip2)
//!
//! Pure Rust git operations for WASM - reads git repository data directly.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run --dir /repo)
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

// ==========================================
// Parameter structs for JSON Schema generation
// ==========================================

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RepoPathParams {
    /// Path to the git repository
    pub repo_path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Maximum number of commits to show
    pub max_count: Option<usize>,
    /// Start timestamp for filtering commits (ISO format)
    pub start_timestamp: Option<String>,
    /// End timestamp for filtering commits (ISO format)
    pub end_timestamp: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShowParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Commit SHA, branch name, or HEAD
    pub revision: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BranchListParams {
    /// The path to the Git repository.
    pub repo_path: String,
    /// Whether to list local branches ('local'), remote branches ('remote') or all branches('all').
    pub branch_type: String,
    /// Filter branches containing this commit
    pub contains: Option<String>,
    /// Filter branches not containing this commit
    pub not_contains: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommitParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Commit message
    pub message: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Files to add to staging area
    pub files: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Target branch or commit to compare with
    pub target: String,
    /// Number of context lines
    pub context_lines: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffUnstagedParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Number of context lines
    pub context_lines: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateBranchParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Name of the new branch
    pub branch_name: String,
    /// Base branch to create from (optional, defaults to current branch)
    pub base_branch: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CheckoutParams {
    /// Path to the git repository
    pub repo_path: String,
    /// Branch name to checkout
    pub branch_name: String,
}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-git")
        .version("1.0.0")
        .description("Git MCP Server - Read-only git repository tools for WASM")
        .tool::<RepoPathParams, _>(
            "git_status",
            "Shows the working tree status",
            |params| tools::git_status(&params.repo_path)
        )
        .tool::<LogParams, _>(
            "git_log",
            "Shows the commit logs",
            |params| tools::git_log(&params.repo_path, params.max_count)
        )
        .tool::<ShowParams, _>(
            "git_show",
            "Shows a commit or other object",
            |params| tools::git_show(&params.repo_path, &params.revision)
        )
        .tool::<BranchListParams, _>(
            "git_branch",
            "Lists repository branches",
            |params| tools::git_branch(&params.repo_path, &params.branch_type)
        )
        .tool::<DiffUnstagedParams, _>(
            "git_diff_unstaged",
            "Shows changes not yet staged",
            |params| tools::git_diff_unstaged(&params.repo_path, params.context_lines)
        )
        .tool::<DiffUnstagedParams, _>(
            "git_diff_staged",
            "Shows staged changes",
            |params| tools::git_diff_staged(&params.repo_path, params.context_lines)
        )
        .tool::<DiffParams, _>(
            "git_diff",
            "Shows differences between commits",
            |params| tools::git_diff(&params.repo_path, &params.target, params.context_lines)
        )
        .tool::<CommitParams, _>(
            "git_commit",
            "Records changes to the repository",
            |params| tools::git_commit(&params.repo_path, &params.message)
        )
        .tool::<AddParams, _>(
            "git_add",
            "Adds file contents to the index",
            |params| tools::git_add(&params.repo_path, &params.files)
        )
        .tool::<RepoPathParams, _>(
            "git_reset",
            "Unstages all staged changes",
            |params| tools::git_reset(&params.repo_path)
        )
        .tool::<CreateBranchParams, _>(
            "git_create_branch",
            "Creates a new branch",
            |params| tools::git_create_branch(
                &params.repo_path,
                &params.branch_name,
                params.base_branch.as_deref()
            )
        )
        .tool::<CheckoutParams, _>(
            "git_checkout",
            "Switches branches or restores working tree files",
            |params| tools::git_checkout(&params.repo_path, &params.branch_name)
        )
        .build()
}

// ==========================================
// CLI Export (wasmtime run --dir /repo)
// ==========================================

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// ==========================================
// HTTP Export (wasmtime serve)
// ==========================================

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
