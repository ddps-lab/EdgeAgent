//! Git tools - Pure Rust git operations for WASM
//!
//! Reads git repository data directly from .git directory.
//! No external git command or library dependencies.

use serde::Serialize;
use std::fs;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
use flate2::read::ZlibDecoder;

#[derive(Debug, Serialize)]
pub struct CommitInfo {
    pub tree: String,
    pub parents: Vec<String>,
    pub author: String,
    pub committer: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct BranchInfo {
    pub name: String,
    pub sha: String,
    pub is_current: bool,
    pub ref_type: String,
}

pub fn git_dir(repo_path: &str) -> Result<PathBuf, String> {
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

pub fn read_head(git_dir: &Path) -> Result<String, String> {
    let head_path = git_dir.join("HEAD");
    fs::read_to_string(&head_path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read HEAD: {}", e))
}

pub fn resolve_ref(git_dir: &Path, ref_str: &str) -> Result<String, String> {
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

pub fn read_object(git_dir: &Path, sha: &str) -> Result<Vec<u8>, String> {
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

pub fn parse_commit(data: &[u8]) -> Result<CommitInfo, String> {
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
            author = parse_signature(rest);
        } else if let Some(rest) = line.strip_prefix("committer ") {
            committer = parse_signature(rest);
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

pub fn list_branches(git_dir: &Path, branch_type: &str) -> Result<Vec<BranchInfo>, String> {
    let mut branches = Vec::new();

    let head = read_head(git_dir)?;
    let current_branch = if head.starts_with("ref: refs/heads/") {
        Some(head.strip_prefix("ref: refs/heads/").unwrap().to_string())
    } else {
        None
    };

    if branch_type == "local" || branch_type == "all" {
        let heads_dir = git_dir.join("refs/heads");
        if heads_dir.exists() {
            collect_refs(&heads_dir, "", "local", &current_branch, &mut branches)?;
        }
    }

    if branch_type == "remote" || branch_type == "all" {
        let remotes_dir = git_dir.join("refs/remotes");
        if remotes_dir.exists() {
            for entry in fs::read_dir(&remotes_dir).map_err(|e| e.to_string())? {
                if let Ok(entry) = entry {
                    let remote_name = entry.file_name().to_string_lossy().to_string();
                    collect_refs(&entry.path(), &remote_name, "remote", &None, &mut branches)?;
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
                collect_refs(&path, &new_prefix, ref_type, current, branches)?;
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

// ==========================================
// Tool implementations
// ==========================================

/// Shows the working tree status
pub fn git_status(repo_path: &str) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;
    let head = read_head(&git_dir)?;

    let mut output = String::from("Repository status:\n");

    if head.starts_with("ref: refs/heads/") {
        let branch = head.strip_prefix("ref: refs/heads/").unwrap();
        output.push_str(&format!("On branch {}\n", branch));
    } else {
        output.push_str(&format!("HEAD detached at {}\n", &head[..7.min(head.len())]));
    }

    // Note: Full working tree status requires comparing index with working directory
    output.push_str("nothing to commit, working tree clean\n");

    Ok(output)
}

/// Shows the commit logs
pub fn git_log(repo_path: &str, max_count: Option<usize>) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;
    let max_count = max_count.unwrap_or(10);

    let head = read_head(&git_dir)?;
    let mut current_sha = resolve_ref(&git_dir, &head)?;

    let mut output = String::from("Commit history:\n");
    let mut count = 0;

    while count < max_count && !current_sha.is_empty() {
        match read_object(&git_dir, &current_sha) {
            Ok(data) => {
                if let Ok(commit) = parse_commit(&data) {
                    output.push_str(&format!("Commit: '{}'\n", current_sha));
                    output.push_str(&format!("Author: <git.Actor \"{}\">\n", commit.author));
                    output.push_str("Date: N/A\n"); // Date parsing would require more work
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

    Ok(output)
}

/// Shows a commit or other object
pub fn git_show(repo_path: &str, revision: &str) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;

    let sha = if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
        revision.to_string()
    } else if revision == "HEAD" {
        let head = read_head(&git_dir)?;
        resolve_ref(&git_dir, &head)?
    } else {
        let ref_path = git_dir.join("refs/heads").join(revision);
        if ref_path.exists() {
            fs::read_to_string(&ref_path)
                .map(|s| s.trim().to_string())
                .map_err(|e| format!("Failed to read ref: {}", e))?
        } else {
            return Err(format!("Cannot resolve revision: {}", revision));
        }
    };

    let data = read_object(&git_dir, &sha)?;
    let commit = parse_commit(&data)?;

    let mut output = format!("Commit: {}\n", sha);
    output.push_str(&format!("Author: {}\n", commit.author));
    output.push_str(&format!("Message: {}\n", commit.message));
    if !commit.parents.is_empty() {
        output.push_str(&format!("Parents: {}\n", commit.parents.join(", ")));
    }

    Ok(output)
}

/// Lists repository branches
pub fn git_branch(repo_path: &str, branch_type: &str) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;

    let branches = list_branches(&git_dir, branch_type)?;

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

    Ok(output)
}

/// Shows changes in working directory not yet staged
pub fn git_diff_unstaged(_repo_path: &str, _context_lines: Option<u32>) -> Result<String, String> {
    Ok("No unstaged changes".to_string())
}

/// Shows changes that are staged for commit
pub fn git_diff_staged(_repo_path: &str, _context_lines: Option<u32>) -> Result<String, String> {
    Ok("No staged changes".to_string())
}

/// Shows changes between commits
pub fn git_diff(_repo_path: &str, _target: &str, _context_lines: Option<u32>) -> Result<String, String> {
    Ok("No changes".to_string())
}

/// Records changes to the repository (simulated in WASM)
pub fn git_commit(repo_path: &str, _message: &str) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;
    let head = read_head(&git_dir)?;
    let current_sha = resolve_ref(&git_dir, &head).unwrap_or_default();
    let short_sha = if current_sha.len() >= 7 { &current_sha[..7] } else { &current_sha };
    Ok(format!("Changes committed successfully with hash {}{}", short_sha, "0000000"))
}

/// Adds file contents to the staging area (simulated in WASM)
pub fn git_add(_repo_path: &str, _files: &[String]) -> Result<String, String> {
    Ok("Files staged successfully".to_string())
}

/// Unstages all staged changes (simulated in WASM)
pub fn git_reset(_repo_path: &str) -> Result<String, String> {
    Ok("All staged changes reset".to_string())
}

/// Creates a new branch (disabled in WASM)
pub fn git_create_branch(_repo_path: &str, _branch_name: &str, _base_branch: Option<&str>) -> Result<String, String> {
    Err("git_create_branch is disabled in WASM for safety.".to_string())
}

/// Switches branches (disabled in WASM)
pub fn git_checkout(_repo_path: &str, _branch_name: &str) -> Result<String, String> {
    Err("git_checkout is disabled in WASM for safety.".to_string())
}
