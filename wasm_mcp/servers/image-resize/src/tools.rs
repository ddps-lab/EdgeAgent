//! Image tools - Pure business logic for image processing
//!
//! Shared between CLI and HTTP transports.

use std::path::Path;
use std::fs;
use std::io::Cursor;
use image::{GenericImageView, ImageFormat, imageops::FilterType};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Serialize;
use wasmmcp::timing::{measure_io, ToolTimer, get_wasm_total_ms};

#[derive(Debug, Serialize)]
pub struct HashResult {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn get_image_format(path: &str) -> Option<ImageFormat> {
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

pub fn output_format_from_str(s: &str) -> ImageFormat {
    match s.to_uppercase().as_str() {
        "PNG" => ImageFormat::Png,
        "WEBP" => ImageFormat::WebP,
        "GIF" => ImageFormat::Gif,
        _ => ImageFormat::Jpeg,
    }
}

/// Simple perceptual hash (average hash - aHash)
pub fn compute_ahash(img: &image::DynamicImage) -> String {
    let small = img.resize_exact(8, 8, FilterType::Lanczos3);
    let gray = small.to_luma8();

    let pixels: Vec<u8> = gray.pixels().map(|p| p.0[0]).collect();
    let avg: u64 = pixels.iter().map(|&p| p as u64).sum::<u64>() / pixels.len() as u64;

    let mut hash: u64 = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel as u64 >= avg {
            hash |= 1 << i;
        }
    }

    format!("{:016x}", hash)
}

/// Difference hash (dHash)
pub fn compute_dhash(img: &image::DynamicImage) -> String {
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

/// Perceptual hash (simplified pHash)
pub fn compute_phash(img: &image::DynamicImage) -> String {
    let small = img.resize_exact(32, 32, FilterType::Lanczos3);
    let gray = small.to_luma8();

    let mut values: Vec<f64> = Vec::with_capacity(64);
    for y in 0..8 {
        for x in 0..8 {
            values.push(gray.get_pixel(x, y).0[0] as f64);
        }
    }

    let avg: f64 = values[1..].iter().sum::<f64>() / 63.0;

    let mut hash: u64 = 0;
    for (i, &val) in values.iter().enumerate() {
        if val > avg {
            hash |= 1 << i;
        }
    }

    format!("{:016x}", hash)
}

/// Compute hamming distance between two hex hashes
pub fn hash_distance(hash1: &str, hash2: &str) -> u32 {
    let h1 = u64::from_str_radix(hash1, 16).unwrap_or(0);
    let h2 = u64::from_str_radix(hash2, 16).unwrap_or(0);
    (h1 ^ h2).count_ones()
}

// ==========================================
// Tool implementations
// ==========================================

/// Get detailed information about an image
pub fn get_image_info(image_path: &str) -> Result<String, String> {
    let timer = ToolTimer::start();
    let metadata = measure_io(|| fs::metadata(image_path))
        .map_err(|e| format!("Cannot access file: {}", e))?;

    let size_bytes = metadata.len();

    let img = measure_io(|| image::open(image_path))
        .map_err(|e| format!("Cannot open image: {}", e))?;

    let (width, height) = img.dimensions();
    let format = get_image_format(image_path)
        .map(|f| format!("{:?}", f))
        .unwrap_or_else(|| "Unknown".to_string());

    let color_type = format!("{:?}", img.color());

    let timing = timer.finish("get_image_info");
    let result = serde_json::json!({
        "path": image_path,
        "format": format,
        "mode": color_type,
        "width": width,
        "height": height,
        "size_bytes": size_bytes,
        "aspect_ratio": if height > 0 { (width as f64 / height as f64 * 100.0).round() / 100.0 } else { 0.0 },
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        }
    });

    Ok(result.to_string())
}

/// Resize an image and return as base64
pub fn resize_image(
    image_path: &str,
    width: Option<u32>,
    height: Option<u32>,
    max_size: Option<u32>,
    quality: Option<u8>,
    output_format: Option<&str>,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let quality = quality.unwrap_or(85);
    let output_format = output_format.unwrap_or("JPEG");

    let original_bytes = measure_io(|| fs::metadata(image_path))
        .map(|m| m.len())
        .unwrap_or(0);

    let img = measure_io(|| image::open(image_path))
        .map_err(|e| format!("Cannot open image: {}", e))?;

    let (orig_width, orig_height) = img.dimensions();

    let (new_width, new_height) = if let Some(max_size) = max_size {
        let ratio = (max_size as f64 / orig_width.max(orig_height) as f64).min(1.0);
        ((orig_width as f64 * ratio) as u32, (orig_height as f64 * ratio) as u32)
    } else if let (Some(w), Some(h)) = (width, height) {
        (w, h)
    } else if let Some(w) = width {
        let ratio = w as f64 / orig_width as f64;
        (w, (orig_height as f64 * ratio) as u32)
    } else if let Some(h) = height {
        let ratio = h as f64 / orig_height as f64;
        ((orig_width as f64 * ratio) as u32, h)
    } else {
        return Err("No resize parameters provided (width, height, or max_size)".to_string());
    };

    let resized = img.resize(new_width, new_height, FilterType::Lanczos3);

    let format = output_format_from_str(output_format);
    let output_img = if format == ImageFormat::Jpeg {
        image::DynamicImage::ImageRgb8(resized.to_rgb8())
    } else {
        resized
    };

    let mut buffer = Cursor::new(Vec::new());

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

    let timing = timer.finish("resize_image");
    let result = serde_json::json!({
        "success": true,
        "path": image_path,
        "original_size": [orig_width, orig_height],
        "new_size": [new_width, new_height],
        "original_bytes": original_bytes,
        "output_bytes": output_bytes,
        "reduction_ratio": if original_bytes > 0 { (output_bytes as f64 / original_bytes as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
        "format": output_format,
        "data_base64": data_base64,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        }
    });

    Ok(result.to_string())
}

/// Scan directory for image files
pub fn scan_directory(
    directory: &str,
    extensions: Option<Vec<String>>,
    recursive: Option<bool>,
    include_info: Option<bool>,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let recursive = recursive.unwrap_or(true);
    let include_info = include_info.unwrap_or(false);

    let extensions: Vec<String> = extensions.unwrap_or_else(|| {
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
        let entries = measure_io(|| fs::read_dir(dir))
            .map_err(|e| format!("Cannot read directory: {}", e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = format!(".{}", ext.to_string_lossy().to_lowercase());
                    if extensions.iter().any(|e| e.to_lowercase() == ext_str) {
                        if let Ok(meta) = measure_io(|| fs::metadata(&path)) {
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
            if let Ok(img) = measure_io(|| image::open(img_path)) {
                let (w, h) = img.dimensions();
                let size = measure_io(|| fs::metadata(img_path)).map(|m| m.len()).unwrap_or(0);
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

    let timing = timer.finish("scan_directory");
    result["timing"] = serde_json::json!({
        "wasm_total_ms": get_wasm_total_ms(),
        "fn_total_ms": timing.fn_total_ms,
        "io_ms": timing.io_ms,
        "compute_ms": timing.compute_ms
    });

    Ok(result.to_string())
}

/// Compute perceptual hash of an image
pub fn compute_image_hash(image_path: &str, hash_type: Option<&str>) -> Result<String, String> {
    let timer = ToolTimer::start();
    let hash_type = hash_type.unwrap_or("phash");

    let img = match measure_io(|| image::open(image_path)) {
        Ok(img) => img,
        Err(e) => {
            let timing = timer.finish("compute_image_hash");
            return Ok(serde_json::json!({
                "path": image_path,
                "hash": null,
                "hash_type": null,
                "error": format!("Cannot open image: {}", e),
                "timing": {
                    "wasm_total_ms": get_wasm_total_ms(),
                    "fn_total_ms": timing.fn_total_ms,
                    "io_ms": timing.io_ms,
                    "compute_ms": timing.compute_ms
                }
            }).to_string());
        }
    };

    let hash_value = match hash_type.to_lowercase().as_str() {
        "ahash" => compute_ahash(&img),
        "dhash" => compute_dhash(&img),
        _ => compute_phash(&img),
    };

    let timing = timer.finish("compute_image_hash");
    Ok(serde_json::json!({
        "path": image_path,
        "hash": hash_value,
        "hash_type": hash_type,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Compare image hashes to find duplicates
pub fn compare_hashes(hashes: &[HashResult], threshold: Option<u32>) -> Result<String, String> {
    let timer = ToolTimer::start();
    let threshold = threshold.unwrap_or(5);

    // Filter valid hashes
    let valid_hashes: Vec<(&str, &str)> = hashes.iter()
        .filter(|h| h.hash.is_some() && h.error.is_none())
        .map(|h| (h.path.as_str(), h.hash.as_ref().unwrap().as_str()))
        .collect();

    if valid_hashes.len() < 2 {
        let errors: Vec<_> = hashes.iter()
            .filter(|h| h.error.is_some())
            .collect();

        let timing = timer.finish("compare_hashes");
        return Ok(serde_json::json!({
            "total_compared": valid_hashes.len(),
            "duplicate_groups": [],
            "unique_count": valid_hashes.len(),
            "errors": errors,
            "timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
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

            let distance = hash_distance(hash1, hash2);
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

    let errors: Vec<_> = hashes.iter()
        .filter(|h| h.error.is_some())
        .collect();

    let timing = timer.finish("compare_hashes");
    Ok(serde_json::json!({
        "total_compared": valid_hashes.len(),
        "duplicate_groups": groups,
        "duplicate_group_count": groups.len(),
        "unique_paths": unique,
        "unique_count": unique.len(),
        "threshold": threshold,
        "errors": errors,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}

/// Batch resize multiple images
pub fn batch_resize(
    image_paths: &[String],
    max_size: Option<u32>,
    quality: Option<u8>,
    output_format: Option<&str>,
) -> Result<String, String> {
    let timer = ToolTimer::start();
    let max_size = max_size.unwrap_or(150);
    let quality = quality.unwrap_or(75);
    let output_format = output_format.unwrap_or("JPEG");
    let format = output_format_from_str(output_format);

    let mut results = Vec::new();
    let mut total_input: u64 = 0;
    let mut total_output: u64 = 0;
    let mut successful = 0;
    let mut failed = 0;

    for path in image_paths {
        let original_bytes = measure_io(|| fs::metadata(path)).map(|m| m.len()).unwrap_or(0);
        total_input += original_bytes;

        match measure_io(|| image::open(path)) {
            Ok(img) => {
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
                failed += 1;
                results.push(serde_json::json!({
                    "path": path,
                    "success": false,
                    "error": format!("Cannot open: {}", e),
                }));
            }
        }
    }

    let timing = timer.finish("batch_resize");
    Ok(serde_json::json!({
        "total_images": image_paths.len(),
        "successful": successful,
        "failed": failed,
        "total_input_bytes": total_input,
        "total_output_bytes": total_output,
        "overall_reduction": if total_input > 0 { (total_output as f64 / total_input as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
        "results": results,
        "timing": {
            "wasm_total_ms": get_wasm_total_ms(),
            "fn_total_ms": timing.fn_total_ms,
            "io_ms": timing.io_ms,
            "compute_ms": timing.compute_ms
        }
    }).to_string())
}
