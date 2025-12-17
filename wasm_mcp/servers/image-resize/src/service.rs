//! Image Resize Service - Image processing tools
//!
//! Provides tools for image information, resizing, directory scanning,
//! and perceptual hashing for duplicate detection.

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
use std::path::Path;
use std::fs;
use std::io::Cursor;
use std::time::Instant;  // profiling
use image::{GenericImageView, ImageFormat, imageops::FilterType};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Image Resize MCP Service
#[derive(Debug, Clone)]
pub struct ImageResizeService {
    tool_router: ToolRouter<Self>,
}

impl ImageResizeService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    fn get_image_format(path: &str) -> Option<ImageFormat> {
        let ext = Path::new(path).extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            "gif" => Some(ImageFormat::Gif),
            "webp" => Some(ImageFormat::WebP),
            "bmp" => Some(ImageFormat::Bmp),
            _ => None,
        }
    }

    fn output_format_from_str(s: &str) -> ImageFormat {
        match s.to_uppercase().as_str() {
            "PNG" => ImageFormat::Png,
            "WEBP" => ImageFormat::WebP,
            "GIF" => ImageFormat::Gif,
            _ => ImageFormat::Jpeg,
        }
    }

    /// Simple perceptual hash (average hash - aHash)
    /// Returns a 64-bit hash as hex string
    fn compute_ahash(img: &image::DynamicImage) -> String {
        // Resize to 8x8
        let small = img.resize_exact(8, 8, FilterType::Lanczos3);
        let gray = small.to_luma8();

        // Compute average
        let pixels: Vec<u8> = gray.pixels().map(|p| p.0[0]).collect();
        let avg: u64 = pixels.iter().map(|&p| p as u64).sum::<u64>() / pixels.len() as u64;

        // Generate hash bits
        let mut hash: u64 = 0;
        for (i, &pixel) in pixels.iter().enumerate() {
            if pixel as u64 >= avg {
                hash |= 1 << i;
            }
        }

        format!("{:016x}", hash)
    }

    /// Difference hash (dHash)
    fn compute_dhash(img: &image::DynamicImage) -> String {
        // Resize to 9x8 (one extra column for comparison)
        let small = img.resize_exact(9, 8, FilterType::Lanczos3);
        let gray = small.to_luma8();

        let mut hash: u64 = 0;
        let mut bit = 0;

        for y in 0..8 {
            for x in 0..8 {
                let left = gray.get_pixel(x, y).0[0];
                let right = gray.get_pixel(x + 1, y).0[0];
                if left > right {
                    hash |= 1 << bit;
                }
                bit += 1;
            }
        }

        format!("{:016x}", hash)
    }

    /// Perceptual hash (simplified pHash using DCT-like approach)
    fn compute_phash(img: &image::DynamicImage) -> String {
        // Resize to 32x32 for better accuracy
        let small = img.resize_exact(32, 32, FilterType::Lanczos3);
        let gray = small.to_luma8();

        // Get top-left 8x8 as simplified DCT
        let mut values: Vec<f64> = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                values.push(gray.get_pixel(x, y).0[0] as f64);
            }
        }

        // Compute average (excluding first value which is DC component)
        let avg: f64 = values[1..].iter().sum::<f64>() / 63.0;

        // Generate hash
        let mut hash: u64 = 0;
        for (i, &val) in values.iter().enumerate() {
            if val > avg {
                hash |= 1 << i;
            }
        }

        format!("{:016x}", hash)
    }

    /// Compute hamming distance between two hex hashes
    fn hash_distance(hash1: &str, hash2: &str) -> u32 {
        let h1 = u64::from_str_radix(hash1, 16).unwrap_or(0);
        let h2 = u64::from_str_radix(hash2, 16).unwrap_or(0);
        (h1 ^ h2).count_ones()
    }
}

impl Default for ImageResizeService {
    fn default() -> Self {
        Self::new()
    }
}

// Tool parameters

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetImageInfoParams {
    #[schemars(description = "Path to the image file")]
    pub image_path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ResizeImageParams {
    #[schemars(description = "Path to the input image")]
    pub image_path: String,

    #[schemars(description = "Target width (maintains aspect ratio if height not given)")]
    pub width: Option<u32>,

    #[schemars(description = "Target height (maintains aspect ratio if width not given)")]
    pub height: Option<u32>,

    #[schemars(description = "Maximum dimension (constrains both width and height)")]
    pub max_size: Option<u32>,

    #[schemars(description = "Output quality (1-100, for JPEG/WEBP). Default: 85")]
    pub quality: Option<u8>,

    #[schemars(description = "Output format: JPEG, PNG, or WEBP. Default: JPEG")]
    pub output_format: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ScanDirectoryParams {
    #[schemars(description = "Directory path to scan")]
    pub directory: String,

    #[schemars(description = "File extensions to include (default: common image formats)")]
    pub extensions: Option<Vec<String>>,

    #[schemars(description = "Whether to scan subdirectories. Default: true")]
    pub recursive: Option<bool>,

    #[schemars(description = "Whether to include detailed info per image. Default: false")]
    pub include_info: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ComputeHashParams {
    #[schemars(description = "Path to the image")]
    pub image_path: String,

    #[schemars(description = "Type of hash: phash, dhash, or ahash. Default: phash")]
    pub hash_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct HashResult {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CompareHashesParams {
    #[schemars(description = "List of hash results from compute_image_hash")]
    pub hashes: Vec<HashResult>,

    #[schemars(description = "Maximum hash difference to consider as duplicate. Default: 5")]
    pub threshold: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BatchResizeParams {
    #[schemars(description = "List of image file paths to resize")]
    pub image_paths: Vec<String>,

    #[schemars(description = "Maximum dimension for thumbnails. Default: 150")]
    pub max_size: Option<u32>,

    #[schemars(description = "Output quality (1-100). Default: 75")]
    pub quality: Option<u8>,

    #[schemars(description = "Output format: JPEG, PNG, or WEBP. Default: JPEG")]
    pub output_format: Option<String>,
}

#[tool_router]
impl ImageResizeService {
    /// Get detailed information about an image
    #[tool(description = "Get detailed information about an image file")]
    fn get_image_info(&self, Parameters(params): Parameters<GetImageInfoParams>) -> Result<String, String> {
        let io_start = Instant::now();
        let path = &params.image_path;

        // Check file exists
        let metadata = fs::metadata(path)
            .map_err(|e| format!("Cannot access file: {}", e))?;

        let size_bytes = metadata.len();

        // Open image
        let img = image::open(path)
            .map_err(|e| format!("Cannot open image: {}", e))?;
        let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

        let compute_start = Instant::now();
        let (width, height) = img.dimensions();
        let format = Self::get_image_format(path)
            .map(|f| format!("{:?}", f))
            .unwrap_or_else(|| "Unknown".to_string());

        let color_type = format!("{:?}", img.color());

        let result = serde_json::json!({
            "path": path,
            "format": format,
            "mode": color_type,
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "aspect_ratio": if height > 0 { (width as f64 / height as f64 * 100.0).round() / 100.0 } else { 0.0 },
        });
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = result.to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", io_ms, compute_ms, serialize_ms);
        Ok(output)
    }

    /// Resize an image and return as base64
    #[tool(description = "Resize an image and return the result as base64-encoded data")]
    fn resize_image(&self, Parameters(params): Parameters<ResizeImageParams>) -> Result<String, String> {
        let io_start = Instant::now();
        let path = &params.image_path;
        let quality = params.quality.unwrap_or(85);
        let output_format = params.output_format.as_deref().unwrap_or("JPEG");

        // Get original size
        let original_bytes = fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Open image
        let img = image::open(path)
            .map_err(|e| format!("Cannot open image: {}", e))?;
        let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

        let compute_start = Instant::now();
        let (orig_width, orig_height) = img.dimensions();

        // Calculate new dimensions
        let (new_width, new_height) = if let Some(max_size) = params.max_size {
            let ratio = (max_size as f64 / orig_width.max(orig_height) as f64).min(1.0);
            ((orig_width as f64 * ratio) as u32, (orig_height as f64 * ratio) as u32)
        } else if let (Some(w), Some(h)) = (params.width, params.height) {
            (w, h)
        } else if let Some(w) = params.width {
            let ratio = w as f64 / orig_width as f64;
            (w, (orig_height as f64 * ratio) as u32)
        } else if let Some(h) = params.height {
            let ratio = h as f64 / orig_height as f64;
            ((orig_width as f64 * ratio) as u32, h)
        } else {
            return Err("No resize parameters provided (width, height, or max_size)".to_string());
        };

        // Resize
        let resized = img.resize(new_width, new_height, FilterType::Lanczos3);

        // Convert to RGB if needed for JPEG
        let format = Self::output_format_from_str(output_format);
        let output_img = if format == ImageFormat::Jpeg {
            image::DynamicImage::ImageRgb8(resized.to_rgb8())
        } else {
            resized
        };

        // Encode to buffer
        let mut buffer = Cursor::new(Vec::new());

        // For JPEG with quality
        if format == ImageFormat::Jpeg {
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
            output_img.write_with_encoder(encoder)
                .map_err(|e| format!("Failed to encode: {}", e))?;
        } else {
            output_img.write_to(&mut buffer, format)
                .map_err(|e| format!("Failed to encode: {}", e))?;
        }

        let output_bytes = buffer.get_ref().len() as u64;
        let data_base64 = BASE64.encode(buffer.get_ref());

        let result = serde_json::json!({
            "success": true,
            "path": path,
            "original_size": [orig_width, orig_height],
            "new_size": [new_width, new_height],
            "original_bytes": original_bytes,
            "output_bytes": output_bytes,
            "reduction_ratio": if original_bytes > 0 { (output_bytes as f64 / original_bytes as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
            "format": output_format,
            "data_base64": data_base64,
        });
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = result.to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", io_ms, compute_ms, serialize_ms);
        Ok(output)
    }

    /// Scan directory for image files
    #[tool(description = "Scan a directory for image files")]
    fn scan_directory(&self, Parameters(params): Parameters<ScanDirectoryParams>) -> Result<String, String> {
        let io_start = Instant::now();
        let directory = &params.directory;
        let recursive = params.recursive.unwrap_or(true);
        let include_info = params.include_info.unwrap_or(false);

        let extensions: Vec<String> = params.extensions.unwrap_or_else(|| {
            vec![
                ".jpg".to_string(), ".jpeg".to_string(), ".png".to_string(),
                ".gif".to_string(), ".bmp".to_string(), ".webp".to_string(),
            ]
        });

        let dir_path = Path::new(directory);
        if !dir_path.exists() {
            return Err(format!("Directory not found: {}", directory));
        }

        let mut image_paths: Vec<String> = Vec::new();
        let mut total_size: u64 = 0;

        fn scan_dir(
            dir: &Path,
            extensions: &[String],
            recursive: bool,
            paths: &mut Vec<String>,
            total: &mut u64,
        ) -> Result<(), String> {
            let entries = fs::read_dir(dir)
                .map_err(|e| format!("Cannot read directory: {}", e))?;

            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_str = format!(".{}", ext.to_string_lossy().to_lowercase());
                        if extensions.iter().any(|e| e.to_lowercase() == ext_str) {
                            if let Ok(meta) = fs::metadata(&path) {
                                *total += meta.len();
                            }
                            paths.push(path.to_string_lossy().to_string());
                        }
                    }
                } else if path.is_dir() && recursive {
                    scan_dir(&path, extensions, recursive, paths, total)?;
                }
            }
            Ok(())
        }

        scan_dir(dir_path, &extensions, recursive, &mut image_paths, &mut total_size)?;
        let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

        let compute_start = Instant::now();
        let mut result = serde_json::json!({
            "directory": directory,
            "image_count": image_paths.len(),
            "total_size_bytes": total_size,
            "total_size_mb": (total_size as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0,
            "image_paths": image_paths,
        });

        if include_info {
            let mut images_info = Vec::new();
            for img_path in &image_paths {
                if let Ok(img) = image::open(img_path) {
                    let (w, h) = img.dimensions();
                    let size = fs::metadata(img_path).map(|m| m.len()).unwrap_or(0);
                    images_info.push(serde_json::json!({
                        "path": img_path,
                        "width": w,
                        "height": h,
                        "size_bytes": size,
                    }));
                }
            }
            result["images"] = serde_json::json!(images_info);
        }
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = result.to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", io_ms, compute_ms, serialize_ms);
        Ok(output)
    }

    /// Compute perceptual hash of an image
    #[tool(description = "Compute perceptual hash of an image for duplicate detection")]
    fn compute_image_hash(&self, Parameters(params): Parameters<ComputeHashParams>) -> Result<String, String> {
        let io_start = Instant::now();
        let path = &params.image_path;
        let hash_type = params.hash_type.as_deref().unwrap_or("phash");

        let img = match image::open(path) {
            Ok(img) => img,
            Err(e) => {
                let result = HashResult {
                    path: path.clone(),
                    hash: None,
                    hash_type: None,
                    error: Some(format!("Cannot open image: {}", e)),
                };
                eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
                return Ok(serde_json::to_string(&result).unwrap());
            }
        };
        let io_ms = io_start.elapsed().as_secs_f64() * 1000.0;

        let compute_start = Instant::now();
        let hash_value = match hash_type.to_lowercase().as_str() {
            "ahash" => Self::compute_ahash(&img),
            "dhash" => Self::compute_dhash(&img),
            _ => Self::compute_phash(&img),
        };

        let result = HashResult {
            path: path.clone(),
            hash: Some(hash_value),
            hash_type: Some(hash_type.to_string()),
            error: None,
        };
        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = serde_json::to_string(&result).unwrap();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", io_ms, compute_ms, serialize_ms);
        Ok(output)
    }

    /// Compare image hashes to find duplicates
    #[tool(description = "Compare image hashes to find duplicates/similar images")]
    fn compare_hashes(&self, Parameters(params): Parameters<CompareHashesParams>) -> Result<String, String> {
        let compute_start = Instant::now();
        let threshold = params.threshold.unwrap_or(5);

        // Filter valid hashes
        let valid_hashes: Vec<(&str, &str)> = params.hashes.iter()
            .filter(|h| h.hash.is_some() && h.error.is_none())
            .map(|h| (h.path.as_str(), h.hash.as_ref().unwrap().as_str()))
            .collect();

        if valid_hashes.len() < 2 {
            let errors: Vec<_> = params.hashes.iter()
                .filter(|h| h.error.is_some())
                .collect();

            eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":0.0,\"serialize_ms\":0.0}}");
            return Ok(serde_json::json!({
                "total_compared": valid_hashes.len(),
                "duplicate_groups": [],
                "unique_count": valid_hashes.len(),
                "errors": errors,
            }).to_string());
        }

        // Find similar pairs
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut processed: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for i in 0..valid_hashes.len() {
            let (path1, hash1) = valid_hashes[i];
            if processed.contains(path1) {
                continue;
            }

            let mut group = vec![path1.to_string()];

            for j in (i + 1)..valid_hashes.len() {
                let (path2, hash2) = valid_hashes[j];
                if processed.contains(path2) {
                    continue;
                }

                let distance = Self::hash_distance(hash1, hash2);
                if distance <= threshold {
                    group.push(path2.to_string());
                    processed.insert(path2);
                }
            }

            if group.len() > 1 {
                processed.insert(path1);
                groups.push(group);
            }
        }

        // Find unique paths
        let all_duplicates: std::collections::HashSet<&str> = groups.iter()
            .flat_map(|g| g.iter().map(|s| s.as_str()))
            .collect();

        let unique: Vec<String> = valid_hashes.iter()
            .filter(|(p, _)| !all_duplicates.contains(*p))
            .map(|(p, _)| p.to_string())
            .collect();

        let errors: Vec<_> = params.hashes.iter()
            .filter(|h| h.error.is_some())
            .collect();

        let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        let serialize_start = Instant::now();
        let output = serde_json::json!({
            "total_compared": valid_hashes.len(),
            "duplicate_groups": groups,
            "duplicate_group_count": groups.len(),
            "unique_paths": unique,
            "unique_count": unique.len(),
            "threshold": threshold,
            "errors": errors,
        }).to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":0.0,\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", compute_ms, serialize_ms);
        Ok(output)
    }

    /// Batch resize multiple images
    #[tool(description = "Resize multiple images at once (e.g., create thumbnails)")]
    fn batch_resize(&self, Parameters(params): Parameters<BatchResizeParams>) -> Result<String, String> {
        let mut io_ms = 0.0;
        let mut compute_ms = 0.0;

        let max_size = params.max_size.unwrap_or(150);
        let quality = params.quality.unwrap_or(75);
        let output_format = params.output_format.as_deref().unwrap_or("JPEG");
        let format = Self::output_format_from_str(output_format);

        let mut results = Vec::new();
        let mut total_input: u64 = 0;
        let mut total_output: u64 = 0;
        let mut successful = 0;
        let mut failed = 0;

        for path in &params.image_paths {
            let io_start = Instant::now();
            let original_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            total_input += original_bytes;

            match image::open(path) {
                Ok(img) => {
                    io_ms += io_start.elapsed().as_secs_f64() * 1000.0;

                    let compute_start = Instant::now();
                    let (orig_w, orig_h) = img.dimensions();
                    let ratio = (max_size as f64 / orig_w.max(orig_h) as f64).min(1.0);
                    let new_w = (orig_w as f64 * ratio) as u32;
                    let new_h = (orig_h as f64 * ratio) as u32;

                    let resized = img.resize(new_w, new_h, FilterType::Lanczos3);

                    let output_img = if format == ImageFormat::Jpeg {
                        image::DynamicImage::ImageRgb8(resized.to_rgb8())
                    } else {
                        resized
                    };

                    let mut buffer = Cursor::new(Vec::new());
                    let encode_result = if format == ImageFormat::Jpeg {
                        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
                        output_img.write_with_encoder(encoder)
                    } else {
                        output_img.write_to(&mut buffer, format)
                    };
                    compute_ms += compute_start.elapsed().as_secs_f64() * 1000.0;

                    match encode_result {
                        Ok(_) => {
                            let output_bytes = buffer.get_ref().len() as u64;
                            total_output += output_bytes;
                            successful += 1;

                            results.push(serde_json::json!({
                                "path": path,
                                "success": true,
                                "original_bytes": original_bytes,
                                "output_bytes": output_bytes,
                                "new_size": [new_w, new_h],
                            }));
                        }
                        Err(e) => {
                            failed += 1;
                            results.push(serde_json::json!({
                                "path": path,
                                "success": false,
                                "error": format!("Encode error: {}", e),
                            }));
                        }
                    }
                }
                Err(e) => {
                    io_ms += io_start.elapsed().as_secs_f64() * 1000.0;
                    failed += 1;
                    results.push(serde_json::json!({
                        "path": path,
                        "success": false,
                        "error": format!("Cannot open: {}", e),
                    }));
                }
            }
        }

        let serialize_start = Instant::now();
        let output = serde_json::json!({
            "total_images": params.image_paths.len(),
            "successful": successful,
            "failed": failed,
            "total_input_bytes": total_input,
            "total_output_bytes": total_output,
            "overall_reduction": if total_input > 0 { (total_output as f64 / total_input as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
            "results": results,
        }).to_string();
        let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("---TIMING---{{\"io_ms\":{:.3},\"compute_ms\":{:.3},\"serialize_ms\":{:.3}}}", io_ms, compute_ms, serialize_ms);
        Ok(output)
    }
}

#[tool_handler]
impl ServerHandler for ImageResizeService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Image Resize MCP Server - Provides tools for image processing. \
                Use get_image_info to get metadata, resize_image to resize single images, \
                scan_directory to find images, compute_image_hash for duplicate detection, \
                compare_hashes to find duplicates, and batch_resize for bulk operations.".into()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
