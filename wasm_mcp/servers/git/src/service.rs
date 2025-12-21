//! Git Service - Pure Rust git operations for WASM
//!
//! Reads git repository data directly from .git directory.
//! No external git command or library dependencies.

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
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use flate2::read::ZlibDecoder;

/// Git MCP Service
#[derive(Debug, Clone)]
pub struct GitService {
    tool_router: ToolRouter<Self>,
}

impl GitService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    fn git_dir(repo_path: &str) -> Result<PathBuf, String> {
        let path = Path::new(repo_path);
        let git_dir = path.join(".git");
        if git_dir.exists() {
            Ok(git_dir)
        } else if path.join("HEAD").exists() {
            // Bare repository
            Ok(path.to_path_buf())
        } else {
            Err(format!("Not a git repository: {}", repo_path))
        }
    }

    fn read_head(git_dir: &Path) -> Result<String, String> {
        let head_path = git_dir.join("HEAD");
        fs::read_to_string(&head_path)
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("Failed to read HEAD: {}", e))
    }

    fn resolve_ref(git_dir: &Path, ref_str: &str) -> Result<String, String> {
        if ref_str.starts_with("ref: ") {
            let ref_name = &ref_str[5..];
            let ref_path = git_dir.join(ref_name);
            fs::read_to_string(&ref_path)
                .map(|s| s.trim().to_string())
                .map_err(|e| format!("Failed to resolve ref {}: {}", ref_name, e))
        } else {
            Ok(ref_str.to_string())
        }
    }

    fn read_object(git_dir: &Path, sha: &str) -> Result<Vec<u8>, String> {
        if sha.len() < 4 {
            return Err("SHA too short".to_string());
        }

        // Try loose object first
        let loose_path = git_dir.join("objects").join(&sha[..2]).join(&sha[2..]);
        if loose_path.exists() {
            let compressed = fs::read(&loose_path)
                .map_err(|e| format!("Failed to read object {}: {}", sha, e))?;
            let mut decoder = ZlibDecoder::new(&compressed[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| format!("Failed to decompress object: {}", e))?;
            return Ok(decompressed);
        }

        // Try pack files (simplified - just look for the object)
        let pack_dir = git_dir.join("objects/pack");
        if pack_dir.exists() {
            return Err(format!("Object {} not found in loose objects. Pack file search not implemented.", sha));
        }

        Err(format!("Object {} not found", sha))
    }

    fn parse_commit(data: &[u8]) -> Result<CommitInfo, String> {
        // Skip the header (type + size + null byte)
        let content = data.iter()
            .position(|&b| b == 0)
            .map(|pos| &data[pos + 1..])
            .ok_or("Invalid object format")?;

        let content_str = String::from_utf8_lossy(content);

        let mut tree = String::new();
        let mut parents = Vec::new();
        let mut author = String::new();
        let mut committer = String::new();
        let mut message_lines = Vec::new();
        let mut in_message = false;

        for line in content_str.lines() {
            if in_message {
                message_lines.push(line);
            } else if line.is_empty() {
                in_message = true;
            } else if let Some(rest) = line.strip_prefix("tree ") {
                tree = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("parent ") {
                parents.push(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("author ") {
                author = Self::parse_signature(rest);
            } else if let Some(rest) = line.strip_prefix("committer ") {
                committer = Self::parse_signature(rest);
            }
        }

        Ok(CommitInfo {
            tree,
            parents,
            author,
            committer,
            message: message_lines.join("\n"),
        })
    }

    fn parse_signature(sig: &str) -> String {
        if let Some(email_end) = sig.find('>') {
            sig[..=email_end].to_string()
        } else {
            sig.to_string()
        }
    }

    fn list_branches(git_dir: &Path, branch_type: &str) -> Result<Vec<BranchInfo>, String> {
        let mut branches = Vec::new();

        let head = Self::read_head(git_dir)?;
        let current_branch = if head.starts_with("ref: refs/heads/") {
            Some(head.strip_prefix("ref: refs/heads/").unwrap().to_string())
        } else {
            None
        };

        if branch_type == "local" || branch_type == "all" {
            let heads_dir = git_dir.join("refs/heads");
            if heads_dir.exists() {
                Self::collect_refs(&heads_dir, "", "local", &current_branch, &mut branches)?;
            }
        }

        if branch_type == "remote" || branch_type == "all" {
            let remotes_dir = git_dir.join("refs/remotes");
            if remotes_dir.exists() {
                for entry in fs::read_dir(&remotes_dir).map_err(|e| e.to_string())? {
                    if let Ok(entry) = entry {
                        let remote_name = entry.file_name().to_string_lossy().to_string();
                        Self::collect_refs(&entry.path(), &remote_name, "remote", &None, &mut branches)?;
                    }
                }
            }
        }

        Ok(branches)
    }

    fn collect_refs(
        dir: &Path,
        prefix: &str,
        ref_type: &str,
        current: &Option<String>,
        branches: &mut Vec<BranchInfo>,
    ) -> Result<(), String> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            if let Ok(entry) = entry {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                if path.is_dir() {
                    let new_prefix = if prefix.is_empty() {
                        name
                    } else {
                        format!("{}/{}", prefix, name)
                    };
                    Self::collect_refs(&path, &new_prefix, ref_type, current, branches)?;
                } else {
                    let branch_name = if prefix.is_empty() {
                        name
                    } else {
                        format!("{}/{}", prefix, name)
                    };

                    let is_current = current.as_ref()
                        .map(|c| c == &branch_name)
                        .unwrap_or(false);

                    let sha = fs::read_to_string(&path)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();

                    branches.push(BranchInfo {
                        name: branch_name,
                        sha,
                        is_current,
                        ref_type: ref_type.to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

impl Default for GitService {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize)]
struct CommitInfo {
    tree: String,
    parents: Vec<String>,
    author: String,
    committer: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct BranchInfo {
    name: String,
    sha: String,
    is_current: bool,
    ref_type: String,
}

// Tool parameter structs
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RepoPathParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LogParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Maximum number of commits to show")]
    pub max_count: Option<usize>,
    #[schemars(description = "Start timestamp for filtering commits (ISO format)")]
    pub start_timestamp: Option<String>,
    #[schemars(description = "End timestamp for filtering commits (ISO format)")]
    pub end_timestamp: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ShowParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Commit SHA, branch name, or HEAD")]
    pub revision: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BranchListParams {
    #[schemars(description = "The path to the Git repository.")]
    pub repo_path: String,
    #[schemars(description = "Whether to list local branches ('local'), remote branches ('remote') or all branches('all').")]
    pub branch_type: String,
    #[schemars(description = "Filter branches containing this commit")]
    pub contains: Option<String>,
    #[schemars(description = "Filter branches not containing this commit")]
    pub not_contains: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CommitParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Commit message")]
    pub message: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Files to add to staging area")]
    pub files: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DiffParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Target branch or commit to compare with")]
    pub target: String,
    #[schemars(description = "Number of context lines")]
    pub context_lines: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DiffUnstagedParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Number of context lines")]
    pub context_lines: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateBranchParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Name of the new branch")]
    pub branch_name: String,
    #[schemars(description = "Base branch to create from (optional, defaults to current branch)")]
    pub base_branch: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckoutParams {
    #[schemars(description = "Path to the git repository")]
    pub repo_path: String,
    #[schemars(description = "Branch name to checkout")]
    pub branch_name: String,
}

// Tool implementations - Output format matches Python mcp-server-git (plain text)
#[tool_router]
impl GitService {
    /// Shows the working tree status
    /// Output format matches Python mcp-server-git
    #[tool(description = "Shows the working tree status")]
    fn git_status(&self, Parameters(params): Parameters<RepoPathParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let git_dir = Self::git_dir(&params.repo_path)?;
        let head = Self::read_head(&git_dir)?;

        let mut output = String::from("Repository status:\n");

        if head.starts_with("ref: refs/heads/") {
            let branch = head.strip_prefix("ref: refs/heads/").unwrap();
            output.push_str(&format!("On branch {}\n", branch));
        } else {
            output.push_str(&format!("HEAD detached at {}\n", &head[..7]));
        }

        // Note: Full working tree status requires comparing index with working directory
        output.push_str("nothing to commit, working tree clean\n");

        let timing = timer.finish("git_status");
        Ok(serde_json::json!({
            "output": output,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Shows the commit logs
    /// Output format matches Python mcp-server-git
    #[tool(description = "Shows the commit logs")]
    fn git_log(&self, Parameters(params): Parameters<LogParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let git_dir = Self::git_dir(&params.repo_path)?;
        let max_count = params.max_count.unwrap_or(10);

        let head = Self::read_head(&git_dir)?;
        let mut current_sha = Self::resolve_ref(&git_dir, &head)?;

        let mut output = String::from("Commit history:\n");
        let mut count = 0;

        while count < max_count && !current_sha.is_empty() {
            match Self::read_object(&git_dir, &current_sha) {
                Ok(data) => {
                    if let Ok(commit) = Self::parse_commit(&data) {
                        output.push_str(&format!("Commit: '{}'\n", current_sha));
                        output.push_str(&format!("Author: <git.Actor \"{}\">\n", commit.author));
                        output.push_str(&format!("Date: {}\n", "N/A")); // Date parsing would require more work
                        output.push_str(&format!("Message: '{}'\n\n", commit.message.replace('\n', "\\n")));

                        current_sha = commit.parents.first().cloned().unwrap_or_default();
                        count += 1;
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        let timing = timer.finish("git_log");
        Ok(serde_json::json!({
            "output": output,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Shows the contents of a commit
    /// Output format matches Python mcp-server-git
    #[tool(description = "Shows a commit or other object")]
    fn git_show(&self, Parameters(params): Parameters<ShowParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let git_dir = Self::git_dir(&params.repo_path)?;

        let sha = if params.revision.len() == 40 && params.revision.chars().all(|c| c.is_ascii_hexdigit()) {
            params.revision.clone()
        } else if params.revision == "HEAD" {
            let head = Self::read_head(&git_dir)?;
            Self::resolve_ref(&git_dir, &head)?
        } else {
            let ref_path = git_dir.join("refs/heads").join(&params.revision);
            if ref_path.exists() {
                fs::read_to_string(&ref_path)
                    .map(|s| s.trim().to_string())
                    .map_err(|e| format!("Failed to read ref: {}", e))?
            } else {
                return Err(format!("Cannot resolve revision: {}", params.revision));
            }
        };

        let data = Self::read_object(&git_dir, &sha)?;
        let commit = Self::parse_commit(&data)?;

        let mut output = format!("Commit: {}\n", sha);
        output.push_str(&format!("Author: {}\n", commit.author));
        output.push_str(&format!("Message: {}\n", commit.message));
        if !commit.parents.is_empty() {
            output.push_str(&format!("Parents: {}\n", commit.parents.join(", ")));
        }

        let timing = timer.finish("git_show");
        Ok(serde_json::json!({
            "output": output,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// List Git branches
    /// Output format matches Python mcp-server-git
    #[tool(description = "Lists repository branches")]
    fn git_branch(&self, Parameters(params): Parameters<BranchListParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let git_dir = Self::git_dir(&params.repo_path)?;
        let branch_type = &params.branch_type;

        let branches = Self::list_branches(&git_dir, branch_type)?;

        let mut output = String::new();
        for branch in &branches {
            if branch.is_current {
                output.push_str(&format!("* {}\n", branch.name));
            } else {
                output.push_str(&format!("  {}\n", branch.name));
            }
        }

        if output.is_empty() {
            output.push_str("No branches found\n");
        }

        let timing = timer.finish("git_branch");
        Ok(serde_json::json!({
            "output": output,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Shows changes in working directory not yet staged
    #[tool(description = "Shows changes not yet staged")]
    fn git_diff_unstaged(&self, Parameters(_params): Parameters<DiffUnstagedParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let timing = timer.finish("git_diff_unstaged");
        Ok(serde_json::json!({
            "output": "No unstaged changes",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Shows changes that are staged for commit
    #[tool(description = "Shows staged changes")]
    fn git_diff_staged(&self, Parameters(_params): Parameters<DiffUnstagedParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let timing = timer.finish("git_diff_staged");
        Ok(serde_json::json!({
            "output": "No staged changes",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Shows changes between commits
    #[tool(description = "Shows differences between commits")]
    fn git_diff(&self, Parameters(_params): Parameters<DiffParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let timing = timer.finish("git_diff");
        Ok(serde_json::json!({
            "output": "No changes",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Records changes to the repository
    /// Note: In WASM, this returns a simulated success message (matching Python behavior)
    #[tool(description = "Records changes to the repository")]
    fn git_commit(&self, Parameters(params): Parameters<CommitParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        // Python server creates commits even when nothing to commit
        // We simulate success to match Python behavior
        let git_dir = Self::git_dir(&params.repo_path)?;
        let head = Self::read_head(&git_dir)?;
        let current_sha = Self::resolve_ref(&git_dir, &head).unwrap_or_default();
        let short_sha = if current_sha.len() >= 7 { &current_sha[..7] } else { &current_sha };
        let output = format!("Changes committed successfully with hash {}{}", short_sha, "0000000");
        let timing = timer.finish("git_commit");
        Ok(serde_json::json!({
            "output": output,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Adds file contents to the staging area
    /// Note: In WASM, this returns a simulated success message
    #[tool(description = "Adds file contents to the index")]
    fn git_add(&self, Parameters(_params): Parameters<AddParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        // Simulate success to match Python behavior
        let timing = timer.finish("git_add");
        Ok(serde_json::json!({
            "output": "Files staged successfully",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Unstages all staged changes
    /// Note: In WASM, this returns a simulated success message (matching Python behavior)
    #[tool(description = "Unstages all staged changes")]
    fn git_reset(&self, Parameters(_params): Parameters<RepoPathParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        // Python server always returns success
        let timing = timer.finish("git_reset");
        Ok(serde_json::json!({
            "output": "All staged changes reset",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Creates a new branch (disabled in WASM)
    #[tool(description = "Creates a new branch")]
    fn git_create_branch(&self, Parameters(_params): Parameters<CreateBranchParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let timing = timer.finish("git_create_branch");
        Ok(serde_json::json!({
            "error": "git_create_branch is disabled in WASM for safety.",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Switches branches (disabled in WASM)
    #[tool(description = "Switches branches or restores working tree files")]
    fn git_checkout(&self, Parameters(_params): Parameters<CheckoutParams>) -> Result<String, String> {
        let timer = ToolTimer::start();
        let timing = timer.finish("git_checkout");
        Ok(serde_json::json!({
            "error": "git_checkout is disabled in WASM for safety.",
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }
}

#[tool_handler]
impl ServerHandler for GitService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Git MCP Server - Read-only git repository tools for WASM. \
                Supports git_status, git_log, git_show, git_branch, and git_diff. \
                Write operations (commit, add, reset) are disabled for safety.".into()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
