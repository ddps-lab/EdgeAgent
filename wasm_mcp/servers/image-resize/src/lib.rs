//! Image Resize MCP Server - WASM compatible (wasip2)
//!
//! Provides tools for image processing: info, resize, scan, hash, and batch operations.
//!
//! # Build Options
//!
//! - `cargo build --features cli-export` → stdio server (wasmtime run --dir /images)
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
pub struct GetImageInfoParams {
    /// Path to the image file
    pub image_path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResizeImageParams {
    /// Path to the input image
    pub image_path: String,

    /// Target width (maintains aspect ratio if height not given)
    pub width: Option<u32>,

    /// Target height (maintains aspect ratio if width not given)
    pub height: Option<u32>,

    /// Maximum dimension (constrains both width and height)
    pub max_size: Option<u32>,

    /// Output quality (1-100, for JPEG/WEBP). Default: 85
    pub quality: Option<u8>,

    /// Output format: JPEG, PNG, or WEBP. Default: JPEG
    pub output_format: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScanDirectoryParams {
    /// Directory path to scan
    pub directory: String,

    /// File extensions to include (default: common image formats)
    pub extensions: Option<Vec<String>>,

    /// Whether to scan subdirectories. Default: true
    pub recursive: Option<bool>,

    /// Whether to include detailed info per image. Default: false
    pub include_info: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeHashParams {
    /// Path to the image
    pub image_path: String,

    /// Type of hash: phash, dhash, or ahash. Default: phash
    pub hash_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HashResultInput {
    pub path: String,
    pub hash: Option<String>,
    pub hash_type: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareHashesParams {
    /// List of hash results from compute_image_hash
    pub hashes: Vec<HashResultInput>,

    /// Maximum hash difference to consider as duplicate. Default: 5
    pub threshold: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchResizeParams {
    /// List of image file paths to resize
    pub image_paths: Vec<String>,

    /// Maximum dimension for thumbnails. Default: 150
    pub max_size: Option<u32>,

    /// Output quality (1-100). Default: 75
    pub quality: Option<u8>,

    /// Output format: JPEG, PNG, or WEBP. Default: JPEG
    pub output_format: Option<String>,
}

// ==========================================
// Unified Server Factory
// ==========================================

/// Create the MCP server with all tools registered.
/// This is shared between CLI and HTTP transports.
pub fn create_server() -> McpServer {
    McpServer::builder("wasmmcp-image-resize")
        .version("1.0.0")
        .description("Image Resize MCP Server - Image processing tools for WASM")
        .tool::<GetImageInfoParams, _>(
            "get_image_info",
            "Get detailed information about an image file",
            |params| tools::get_image_info(&params.image_path)
        )
        .tool::<ResizeImageParams, _>(
            "resize_image",
            "Resize an image and return the result as base64-encoded data",
            |params| tools::resize_image(
                &params.image_path,
                params.width,
                params.height,
                params.max_size,
                params.quality,
                params.output_format.as_deref()
            )
        )
        .tool::<ScanDirectoryParams, _>(
            "scan_directory",
            "Scan a directory for image files",
            |params| tools::scan_directory(
                &params.directory,
                params.extensions.clone(),
                params.recursive,
                params.include_info
            )
        )
        .tool::<ComputeHashParams, _>(
            "compute_image_hash",
            "Compute perceptual hash of an image for duplicate detection",
            |params| tools::compute_image_hash(
                &params.image_path,
                params.hash_type.as_deref()
            )
        )
        .tool::<CompareHashesParams, _>(
            "compare_hashes",
            "Compare image hashes to find duplicates/similar images",
            |params| {
                let hashes: Vec<tools::HashResult> = params.hashes.iter().map(|h| {
                    tools::HashResult {
                        path: h.path.clone(),
                        hash: h.hash.clone(),
                        hash_type: h.hash_type.clone(),
                        error: h.error.clone(),
                    }
                }).collect();
                tools::compare_hashes(&hashes, params.threshold)
            }
        )
        .tool::<BatchResizeParams, _>(
            "batch_resize",
            "Resize multiple images at once (e.g., create thumbnails)",
            |params| tools::batch_resize(
                &params.image_paths,
                params.max_size,
                params.quality,
                params.output_format.as_deref()
            )
        )
        .build()
}

// ==========================================
// CLI Export (wasmtime run --dir /images)
// ==========================================

#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// ==========================================
// HTTP Export (wasmtime serve)
// ==========================================

#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
