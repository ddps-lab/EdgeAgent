//! Shared utilities for MCP WASM servers
//!
//! This crate provides common utilities and traits used across
//! different MCP server implementations.

use anyhow::Result;

/// Common utilities for MCP servers
pub mod utils {
    use anyhow::{anyhow, Result};
    use std::path::Path;

    /// Validate that a path is safe to access
    pub fn validate_path(path: &str) -> Result<()> {
        let path = Path::new(path);

        // Check for path traversal attempts
        for component in path.components() {
            if let std::path::Component::ParentDir = component {
                return Err(anyhow!("Path traversal not allowed: {}", path.display()));
            }
        }

        Ok(())
    }

    /// Normalize a path by removing redundant components
    pub fn normalize_path(path: &str) -> String {
        let path = Path::new(path);
        let mut components = Vec::new();

        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    components.pop();
                }
                std::path::Component::CurDir => {}
                _ => {
                    components.push(component);
                }
            }
        }

        let normalized: std::path::PathBuf = components.iter().collect();
        normalized.to_string_lossy().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_validate_path_normal() {
        assert!(validate_path("/tmp/test.txt").is_ok());
        assert!(validate_path("test.txt").is_ok());
    }

    #[test]
    fn test_validate_path_traversal() {
        assert!(validate_path("../etc/passwd").is_err());
        assert!(validate_path("/tmp/../etc/passwd").is_err());
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("/tmp/./test.txt"), "/tmp/test.txt");
    }
}
