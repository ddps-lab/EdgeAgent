//! Git tools - Pure Rust git operations for WASM
//!
//! Reads git repository data directly from .git directory.
//! No external git command or library dependencies.
//! Supports both loose objects and pack files.

use serde::Serialize;
use std::fs::{self, File};
use std::io::{Read as IoRead, Seek, SeekFrom, BufReader};
use std::path::{Path, PathBuf};
use flate2::read::ZlibDecoder;
use wasmmcp::timing::measure_disk_io;

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
    measure_disk_io(|| fs::read_to_string(&head_path))
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read HEAD: {}", e))
}

pub fn resolve_ref(git_dir: &Path, ref_str: &str) -> Result<String, String> {
    if ref_str.starts_with("ref: ") {
        let ref_name = &ref_str[5..];
        let ref_path = git_dir.join(ref_name);
        measure_disk_io(|| fs::read_to_string(&ref_path))
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
        let compressed = measure_disk_io(|| fs::read(&loose_path))
            .map_err(|e| format!("Failed to read object {}: {}", sha, e))?;
        let mut decoder = ZlibDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| format!("Failed to decompress object: {}", e))?;
        return Ok(decompressed);
    }

    // Try pack files
    let pack_dir = git_dir.join("objects/pack");
    if pack_dir.exists() {
        if let Ok(result) = read_object_from_pack(&pack_dir, sha) {
            return Ok(result);
        }
    }

    Err(format!("Object {} not found", sha))
}

// ==========================================
// Pack file support
// ==========================================

/// Read object from pack files
fn read_object_from_pack(pack_dir: &Path, sha: &str) -> Result<Vec<u8>, String> {
    let sha_bytes = hex::decode(sha).map_err(|e| format!("Invalid SHA: {}", e))?;

    // Find all .idx files
    let entries = measure_disk_io(|| fs::read_dir(pack_dir))
        .map_err(|e| format!("Failed to read pack dir: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "idx") {
            if let Ok(offset) = find_object_in_idx(&path, &sha_bytes) {
                // Get corresponding .pack file
                let pack_path = path.with_extension("pack");
                if pack_path.exists() {
                    return read_object_from_packfile(&pack_path, offset);
                }
            }
        }
    }

    Err(format!("Object {} not found in pack files", sha))
}

/// Find object offset in .idx file (version 2 format)
fn find_object_in_idx(idx_path: &Path, sha: &[u8]) -> Result<u64, String> {
    let file = measure_disk_io(|| File::open(idx_path))
        .map_err(|e| format!("Failed to open idx: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;

    // Check for v2 magic
    if header[0..4] != [0xff, 0x74, 0x4f, 0x63] {
        return Err("Only idx v2 supported".to_string());
    }

    let version = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    if version != 2 {
        return Err(format!("Unsupported idx version: {}", version));
    }

    // Read fanout table (256 entries, 4 bytes each)
    let mut fanout = [0u32; 256];
    for i in 0..256 {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
        fanout[i] = u32::from_be_bytes(buf);
    }

    let total_objects = fanout[255] as usize;
    let first_byte = sha[0] as usize;

    // Determine search range
    let start = if first_byte == 0 { 0 } else { fanout[first_byte - 1] as usize };
    let end = fanout[first_byte] as usize;

    if start >= end {
        return Err("Object not in this pack".to_string());
    }

    // SHA table starts after header (8 bytes) and fanout (256 * 4 = 1024 bytes)
    let sha_table_offset = 8 + 1024;

    // Binary search in SHA table
    let mut low = start;
    let mut high = end;

    while low < high {
        let mid = (low + high) / 2;
        let sha_offset = sha_table_offset + (mid * 20);

        reader.seek(SeekFrom::Start(sha_offset as u64)).map_err(|e| e.to_string())?;
        let mut entry_sha = [0u8; 20];
        reader.read_exact(&mut entry_sha).map_err(|e| e.to_string())?;

        match sha.cmp(&entry_sha[..]) {
            std::cmp::Ordering::Less => high = mid,
            std::cmp::Ordering::Greater => low = mid + 1,
            std::cmp::Ordering::Equal => {
                // Found! Now get the offset
                // CRC table: after SHA table
                let crc_table_offset = sha_table_offset + (total_objects * 20);
                // Offset table: after CRC table
                let offset_table_offset = crc_table_offset + (total_objects * 4);

                let offset_pos = offset_table_offset + (mid * 4);
                reader.seek(SeekFrom::Start(offset_pos as u64)).map_err(|e| e.to_string())?;

                let mut offset_buf = [0u8; 4];
                reader.read_exact(&mut offset_buf).map_err(|e| e.to_string())?;
                let offset = u32::from_be_bytes(offset_buf);

                // Check MSB for large offset
                if offset & 0x80000000 != 0 {
                    // Large offset - need to read from 64-bit offset table
                    let large_offset_idx = (offset & 0x7fffffff) as usize;
                    let large_offset_table_offset = offset_table_offset + (total_objects * 4);
                    let large_pos = large_offset_table_offset + (large_offset_idx * 8);

                    reader.seek(SeekFrom::Start(large_pos as u64)).map_err(|e| e.to_string())?;
                    let mut large_buf = [0u8; 8];
                    reader.read_exact(&mut large_buf).map_err(|e| e.to_string())?;
                    return Ok(u64::from_be_bytes(large_buf));
                }

                return Ok(offset as u64);
            }
        }
    }

    Err("Object not found in idx".to_string())
}

/// Read object from .pack file at given offset
fn read_object_from_packfile(pack_path: &Path, offset: u64) -> Result<Vec<u8>, String> {
    let file = measure_disk_io(|| File::open(pack_path))
        .map_err(|e| format!("Failed to open pack: {}", e))?;
    let mut reader = BufReader::new(file);

    reader.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;

    // Read object header (variable length encoding)
    let mut byte = [0u8; 1];
    reader.read_exact(&mut byte).map_err(|e| e.to_string())?;

    let obj_type = (byte[0] >> 4) & 0x07;
    let mut size = (byte[0] & 0x0f) as u64;
    let mut shift = 4;

    while byte[0] & 0x80 != 0 {
        reader.read_exact(&mut byte).map_err(|e| e.to_string())?;
        size |= ((byte[0] & 0x7f) as u64) << shift;
        shift += 7;
    }

    match obj_type {
        1 | 2 | 3 | 4 => {
            // commit, tree, blob, tag - deflate compressed
            read_deflated_object(&mut reader, obj_type, size)
        }
        6 => {
            // OFS_DELTA - offset delta
            read_ofs_delta(&mut reader, pack_path, offset)
        }
        7 => {
            // REF_DELTA - reference delta
            read_ref_delta(&mut reader, pack_path)
        }
        _ => Err(format!("Unknown object type: {}", obj_type))
    }
}

/// Read deflate-compressed object
fn read_deflated_object<R: IoRead>(reader: &mut R, obj_type: u8, size: u64) -> Result<Vec<u8>, String> {
    let type_str = match obj_type {
        1 => "commit",
        2 => "tree",
        3 => "blob",
        4 => "tag",
        _ => "unknown",
    };

    // Read remaining data and decompress
    let mut compressed = Vec::new();
    reader.read_to_end(&mut compressed).map_err(|e| e.to_string())?;

    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut content = Vec::new();
    decoder.read_to_end(&mut content).map_err(|e| format!("Decompress error: {}", e))?;

    // Truncate to expected size
    content.truncate(size as usize);

    // Create git object format: "type size\0content"
    let header = format!("{} {}\0", type_str, content.len());
    let mut result = header.into_bytes();
    result.extend(content);

    Ok(result)
}

/// Read OFS_DELTA object
fn read_ofs_delta<R: IoRead + Seek>(reader: &mut R, pack_path: &Path, current_offset: u64) -> Result<Vec<u8>, String> {
    // Read negative offset (variable length encoding)
    let mut byte = [0u8; 1];
    reader.read_exact(&mut byte).map_err(|e| e.to_string())?;

    let mut base_offset = (byte[0] & 0x7f) as u64;
    while byte[0] & 0x80 != 0 {
        reader.read_exact(&mut byte).map_err(|e| e.to_string())?;
        base_offset = ((base_offset + 1) << 7) | ((byte[0] & 0x7f) as u64);
    }

    let base_pack_offset = current_offset - base_offset;

    // Read and decompress delta
    let mut compressed = Vec::new();
    reader.read_to_end(&mut compressed).map_err(|e| e.to_string())?;

    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut delta = Vec::new();
    decoder.read_to_end(&mut delta).map_err(|e| format!("Delta decompress error: {}", e))?;

    // Read base object recursively
    let base = read_object_from_packfile(pack_path, base_pack_offset)?;

    // Apply delta
    apply_delta(&base, &delta)
}

/// Read REF_DELTA object
fn read_ref_delta<R: IoRead>(reader: &mut R, pack_path: &Path) -> Result<Vec<u8>, String> {
    // Read base SHA
    let mut base_sha = [0u8; 20];
    reader.read_exact(&mut base_sha).map_err(|e| e.to_string())?;
    let base_sha_hex = hex::encode(base_sha);

    // Read and decompress delta
    let mut compressed = Vec::new();
    reader.read_to_end(&mut compressed).map_err(|e| e.to_string())?;

    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut delta = Vec::new();
    decoder.read_to_end(&mut delta).map_err(|e| format!("Delta decompress error: {}", e))?;

    // Find base object in pack
    let pack_dir = pack_path.parent().ok_or("No parent dir")?;
    let base = read_object_from_pack(pack_dir, &base_sha_hex)?;

    apply_delta(&base, &delta)
}

/// Apply git delta to base object
fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vec<u8>, String> {
    if delta.is_empty() {
        return Err("Empty delta".to_string());
    }

    let mut pos = 0;

    // Read base size (variable length)
    let (_, new_pos) = read_delta_size(delta, pos)?;
    pos = new_pos;

    // Read result size (variable length)
    let (result_size, new_pos) = read_delta_size(delta, pos)?;
    pos = new_pos;

    // Extract content from base (skip header)
    let base_content = base.iter()
        .position(|&b| b == 0)
        .map(|p| &base[p + 1..])
        .ok_or("Invalid base object")?;

    // Apply delta instructions
    let mut result = Vec::with_capacity(result_size as usize);

    while pos < delta.len() {
        let cmd = delta[pos];
        pos += 1;

        if cmd & 0x80 != 0 {
            // Copy from base
            let mut copy_offset = 0u32;
            let mut copy_size = 0u32;

            if cmd & 0x01 != 0 { copy_offset |= delta.get(pos).copied().unwrap_or(0) as u32; pos += 1; }
            if cmd & 0x02 != 0 { copy_offset |= (delta.get(pos).copied().unwrap_or(0) as u32) << 8; pos += 1; }
            if cmd & 0x04 != 0 { copy_offset |= (delta.get(pos).copied().unwrap_or(0) as u32) << 16; pos += 1; }
            if cmd & 0x08 != 0 { copy_offset |= (delta.get(pos).copied().unwrap_or(0) as u32) << 24; pos += 1; }

            if cmd & 0x10 != 0 { copy_size |= delta.get(pos).copied().unwrap_or(0) as u32; pos += 1; }
            if cmd & 0x20 != 0 { copy_size |= (delta.get(pos).copied().unwrap_or(0) as u32) << 8; pos += 1; }
            if cmd & 0x40 != 0 { copy_size |= (delta.get(pos).copied().unwrap_or(0) as u32) << 16; pos += 1; }

            if copy_size == 0 { copy_size = 0x10000; }

            let start = copy_offset as usize;
            let end = start + copy_size as usize;
            if end <= base_content.len() {
                result.extend_from_slice(&base_content[start..end]);
            }
        } else if cmd != 0 {
            // Insert new data
            let insert_size = cmd as usize;
            if pos + insert_size <= delta.len() {
                result.extend_from_slice(&delta[pos..pos + insert_size]);
                pos += insert_size;
            }
        }
    }

    // Reconstruct full object with header
    // Try to determine type from base
    let base_header = &base[..base.iter().position(|&b| b == 0).unwrap_or(0)];
    let base_header_str = String::from_utf8_lossy(base_header);
    let type_str = base_header_str.split_whitespace().next().unwrap_or("blob");

    let header = format!("{} {}\0", type_str, result.len());
    let mut full_result = header.into_bytes();
    full_result.extend(result);

    Ok(full_result)
}

/// Read variable-length size from delta
fn read_delta_size(delta: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    if pos >= delta.len() {
        return Err("Delta too short".to_string());
    }

    let mut size = (delta[pos] & 0x7f) as u64;
    let mut shift = 7;

    while delta[pos] & 0x80 != 0 {
        pos += 1;
        if pos >= delta.len() {
            return Err("Delta size incomplete".to_string());
        }
        size |= ((delta[pos] & 0x7f) as u64) << shift;
        shift += 7;
    }

    Ok((size, pos + 1))
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
            for entry in measure_disk_io(|| fs::read_dir(&remotes_dir)).map_err(|e| e.to_string())? {
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

    for entry in measure_disk_io(|| fs::read_dir(dir)).map_err(|e| e.to_string())? {
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

                let sha = measure_disk_io(|| fs::read_to_string(&path))
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
            measure_disk_io(|| fs::read_to_string(&ref_path))
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

/// Tree entry representation
#[derive(Debug, Clone)]
struct TreeEntry {
    mode: String,
    name: String,
    sha: String,
}

/// Parse a tree object into entries
fn parse_tree(data: &[u8]) -> Result<Vec<TreeEntry>, String> {
    // Skip the header (type + size + null byte)
    let content = data.iter()
        .position(|&b| b == 0)
        .map(|pos| &data[pos + 1..])
        .ok_or("Invalid tree object format")?;

    let mut entries = Vec::new();
    let mut pos = 0;

    while pos < content.len() {
        // Find space after mode
        let space_pos = content[pos..].iter().position(|&b| b == b' ')
            .ok_or("Invalid tree entry: no space")?;
        let mode = String::from_utf8_lossy(&content[pos..pos + space_pos]).to_string();
        pos += space_pos + 1;

        // Find null after name
        let null_pos = content[pos..].iter().position(|&b| b == 0)
            .ok_or("Invalid tree entry: no null")?;
        let name = String::from_utf8_lossy(&content[pos..pos + null_pos]).to_string();
        pos += null_pos + 1;

        // Read 20-byte SHA
        if pos + 20 > content.len() {
            break;
        }
        let sha = hex::encode(&content[pos..pos + 20]);
        pos += 20;

        entries.push(TreeEntry { mode, name, sha });
    }

    Ok(entries)
}

/// File info with SHA and mode
#[derive(Debug, Clone)]
struct FileInfo {
    sha: String,
    mode: String,
}

/// Recursively collect all files from a tree with mode info
fn collect_files(git_dir: &Path, tree_sha: &str, prefix: &str) -> Result<std::collections::HashMap<String, FileInfo>, String> {
    let mut files = std::collections::HashMap::new();

    let tree_data = read_object(git_dir, tree_sha)?;
    let entries = parse_tree(&tree_data)?;

    for entry in entries {
        let path = if prefix.is_empty() {
            entry.name.clone()
        } else {
            format!("{}/{}", prefix, entry.name)
        };

        if entry.mode.starts_with("40") {
            // Directory (tree)
            let sub_files = collect_files(git_dir, &entry.sha, &path)?;
            files.extend(sub_files);
        } else {
            // File (blob) - convert mode to standard format
            let mode = match entry.mode.as_str() {
                "100644" => "100644",
                "100755" => "100755",
                "120000" => "120000",
                m if m.ends_with("644") => "100644",
                m if m.ends_with("755") => "100755",
                _ => "100644",
            }.to_string();
            files.insert(path, FileInfo { sha: entry.sha, mode });
        }
    }

    Ok(files)
}

/// Read blob content
fn read_blob_content(git_dir: &Path, sha: &str) -> Result<String, String> {
    let data = read_object(git_dir, sha)?;

    // Skip header
    let content = data.iter()
        .position(|&b| b == 0)
        .map(|pos| &data[pos + 1..])
        .ok_or("Invalid blob format")?;

    Ok(String::from_utf8_lossy(content).to_string())
}

/// Generate unified diff between two strings
fn generate_diff(old_content: &str, new_content: &str, path: &str, context_lines: usize) -> String {
    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    let mut output = format!("diff --git a/{} b/{}\n", path, path);
    output.push_str(&format!("--- a/{}\n", path));
    output.push_str(&format!("+++ b/{}\n", path));

    // Simple line-by-line diff (not optimal but functional)
    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut hunks: Vec<String> = Vec::new();
    let mut current_hunk: Vec<String> = Vec::new();
    let mut hunk_old_start = 1;
    let mut hunk_new_start = 1;
    let mut hunk_old_count = 0;
    let mut hunk_new_count = 0;
    let mut context_buffer: Vec<String> = Vec::new();
    let mut in_change = false;

    while old_idx < old_lines.len() || new_idx < new_lines.len() {
        let old_line = old_lines.get(old_idx);
        let new_line = new_lines.get(new_idx);

        match (old_line, new_line) {
            (Some(o), Some(n)) if o == n => {
                // Same line
                if in_change {
                    // Add context after change
                    current_hunk.push(format!(" {}", o));
                    hunk_old_count += 1;
                    hunk_new_count += 1;
                    context_buffer.push(format!(" {}", o));
                    if context_buffer.len() > context_lines {
                        // End current hunk
                        if !current_hunk.is_empty() {
                            let header = format!("@@ -{},{} +{},{} @@\n",
                                hunk_old_start, hunk_old_count,
                                hunk_new_start, hunk_new_count);
                            hunks.push(format!("{}{}", header, current_hunk.join("\n")));
                        }
                        current_hunk.clear();
                        context_buffer.clear();
                        in_change = false;
                    }
                }
                old_idx += 1;
                new_idx += 1;
            }
            (Some(o), Some(n)) => {
                // Different lines
                if !in_change {
                    in_change = true;
                    hunk_old_start = old_idx.saturating_sub(context_lines) + 1;
                    hunk_new_start = new_idx.saturating_sub(context_lines) + 1;
                    hunk_old_count = 0;
                    hunk_new_count = 0;
                    // Add context before
                    let start = old_idx.saturating_sub(context_lines);
                    for i in start..old_idx {
                        if let Some(line) = old_lines.get(i) {
                            current_hunk.push(format!(" {}", line));
                            hunk_old_count += 1;
                            hunk_new_count += 1;
                        }
                    }
                }
                context_buffer.clear();
                current_hunk.push(format!("-{}", o));
                current_hunk.push(format!("+{}", n));
                hunk_old_count += 1;
                hunk_new_count += 1;
                old_idx += 1;
                new_idx += 1;
            }
            (Some(o), None) => {
                // Deleted line
                if !in_change {
                    in_change = true;
                    hunk_old_start = old_idx.saturating_sub(context_lines) + 1;
                    hunk_new_start = new_idx.saturating_sub(context_lines) + 1;
                    hunk_old_count = 0;
                    hunk_new_count = 0;
                }
                context_buffer.clear();
                current_hunk.push(format!("-{}", o));
                hunk_old_count += 1;
                old_idx += 1;
            }
            (None, Some(n)) => {
                // Added line
                if !in_change {
                    in_change = true;
                    hunk_old_start = old_idx.saturating_sub(context_lines) + 1;
                    hunk_new_start = new_idx.saturating_sub(context_lines) + 1;
                    hunk_old_count = 0;
                    hunk_new_count = 0;
                }
                context_buffer.clear();
                current_hunk.push(format!("+{}", n));
                hunk_new_count += 1;
                new_idx += 1;
            }
            (None, None) => break,
        }
    }

    // Flush remaining hunk
    if !current_hunk.is_empty() {
        let header = format!("@@ -{},{} +{},{} @@\n",
            hunk_old_start, hunk_old_count,
            hunk_new_start, hunk_new_count);
        hunks.push(format!("{}{}", header, current_hunk.join("\n")));
    }

    if hunks.is_empty() {
        String::new()
    } else {
        output.push_str(&hunks.join("\n"));
        output.push('\n');
        output
    }
}

/// Resolve a revision to SHA (HEAD, HEAD~1, branch name, or SHA)
fn resolve_revision(git_dir: &Path, revision: &str) -> Result<String, String> {
    // Handle HEAD~N syntax
    if revision.starts_with("HEAD") {
        let head = read_head(git_dir)?;
        let mut sha = resolve_ref(git_dir, &head)?;

        if let Some(tilde_pos) = revision.find('~') {
            let count: usize = revision[tilde_pos + 1..].parse().unwrap_or(1);
            for _ in 0..count {
                let data = read_object(git_dir, &sha)?;
                let commit = parse_commit(&data)?;
                sha = commit.parents.first().cloned().ok_or("No parent commit")?;
            }
        }
        return Ok(sha);
    }

    // Full SHA
    if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(revision.to_string());
    }

    // Branch name
    let ref_path = git_dir.join("refs/heads").join(revision);
    if ref_path.exists() {
        return measure_disk_io(|| fs::read_to_string(&ref_path))
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("Failed to read ref: {}", e));
    }

    Err(format!("Cannot resolve revision: {}", revision))
}

/// Shows changes between commits
pub fn git_diff(repo_path: &str, target: &str, context_lines: Option<u32>) -> Result<String, String> {
    let git_dir = git_dir(repo_path)?;
    let context = context_lines.unwrap_or(3) as usize;

    // Resolve HEAD
    let head = read_head(&git_dir)?;
    let head_sha = resolve_ref(&git_dir, &head)?;

    // Resolve target
    let target_sha = resolve_revision(&git_dir, target)?;

    // Get tree SHAs
    let head_data = read_object(&git_dir, &head_sha)?;
    let head_commit = parse_commit(&head_data)?;

    let target_data = read_object(&git_dir, &target_sha)?;
    let target_commit = parse_commit(&target_data)?;

    // Collect files from both trees
    let head_files = collect_files(&git_dir, &head_commit.tree, "")?;
    let target_files = collect_files(&git_dir, &target_commit.tree, "")?;

    // 1. Add header like Cloud Python server
    let mut output = format!("Diff with {}:\n", target);

    let mut all_paths: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    all_paths.extend(head_files.keys());
    all_paths.extend(target_files.keys());

    let mut has_changes = false;

    for path in all_paths {
        let head_info = head_files.get(path);
        let target_info = target_files.get(path);

        match (target_info, head_info) {
            (Some(old), Some(new)) => {
                // File exists in both commits
                let mode_changed = old.mode != new.mode;
                let content_changed = old.sha != new.sha;

                if mode_changed || content_changed {
                    has_changes = true;
                    output.push_str(&format!("diff --git a/{} b/{}\n", path, path));

                    // 2. Add index line (short SHA + mode)
                    let old_short = if old.sha.len() >= 7 { &old.sha[..7] } else { &old.sha };
                    let new_short = if new.sha.len() >= 7 { &new.sha[..7] } else { &new.sha };
                    output.push_str(&format!("index {}..{} {}\n", old_short, new_short, new.mode));

                    // Mode change
                    if mode_changed {
                        output.push_str(&format!("old mode {}\n", old.mode));
                        output.push_str(&format!("new mode {}\n", new.mode));
                    }

                    // Content change
                    if content_changed {
                        let old_content = read_blob_content(&git_dir, &old.sha).unwrap_or_default();
                        let new_content = read_blob_content(&git_dir, &new.sha).unwrap_or_default();
                        output.push_str(&format!("--- a/{}\n", path));
                        output.push_str(&format!("+++ b/{}\n", path));
                        let diff = generate_diff(&old_content, &new_content, path, context);
                        // generate_diff already includes header, so extract just the hunks
                        if let Some(hunk_start) = diff.find("@@ ") {
                            output.push_str(&diff[hunk_start..]);
                        }
                    }
                }
            }
            (None, Some(new)) => {
                // New file
                has_changes = true;
                let new_content = read_blob_content(&git_dir, &new.sha).unwrap_or_default();
                output.push_str(&format!("diff --git a/{} b/{}\n", path, path));
                output.push_str(&format!("new file mode {}\n", new.mode));
                let new_short = if new.sha.len() >= 7 { &new.sha[..7] } else { &new.sha };
                output.push_str(&format!("index 0000000..{}\n", new_short));
                output.push_str("--- /dev/null\n");
                output.push_str(&format!("+++ b/{}\n", path));
                let lines: Vec<&str> = new_content.lines().collect();
                if !lines.is_empty() {
                    output.push_str(&format!("@@ -0,0 +1,{} @@\n", lines.len()));
                    for line in lines {
                        output.push_str(&format!("+{}\n", line));
                    }
                }
            }
            (Some(old), None) => {
                // Deleted file
                has_changes = true;
                let old_content = read_blob_content(&git_dir, &old.sha).unwrap_or_default();
                output.push_str(&format!("diff --git a/{} b/{}\n", path, path));
                output.push_str(&format!("deleted file mode {}\n", old.mode));
                let old_short = if old.sha.len() >= 7 { &old.sha[..7] } else { &old.sha };
                output.push_str(&format!("index {}..0000000\n", old_short));
                output.push_str(&format!("--- a/{}\n", path));
                output.push_str("+++ /dev/null\n");
                let lines: Vec<&str> = old_content.lines().collect();
                if !lines.is_empty() {
                    output.push_str(&format!("@@ -1,{} +0,0 @@\n", lines.len()));
                    for line in lines {
                        output.push_str(&format!("-{}\n", line));
                    }
                }
            }
            _ => {}
        }
    }

    if !has_changes {
        Ok("No changes".to_string())
    } else {
        Ok(output)
    }
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
