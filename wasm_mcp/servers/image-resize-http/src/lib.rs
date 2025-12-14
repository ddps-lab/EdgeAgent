//! HTTP entry point for Image Resize MCP Server
//!
//! Uses wasi:http/incoming-handler for serverless deployment.
//! Run with: wasmtime serve --dir /path/to/images -S cli target/wasm32-wasip2/release/mcp_server_image_resize_http.wasm

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::fs;
use std::io::Cursor;
use std::collections::HashSet;
use image::{GenericImageView, ImageFormat, imageops::FilterType};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use wasi::http::types::{
    Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam, Method,
};

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    method: String,
    #[serde(default)]
    params: Option<Value>,
    id: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

struct McpHttpHandler;

impl McpHttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        let method = request.method();
        let headers = Fields::new();
        let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Origin".to_string(), &[b"*".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Methods".to_string(), &[b"GET, POST, OPTIONS".to_vec()]);
        let _ = headers.set(&"Access-Control-Allow-Headers".to_string(), &[b"Content-Type".to_vec()]);

        if matches!(method, Method::Options) {
            Self::send_response(response_out, 204, headers, b"");
            return;
        }
        if !matches!(method, Method::Post) {
            Self::send_response(response_out, 405, headers, b"Method Not Allowed");
            return;
        }

        let body = Self::read_body(&request);
        let response = match serde_json::from_slice::<JsonRpcRequest>(&body) {
            Ok(req) => Self::handle_jsonrpc(req),
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError { code: -32700, message: format!("Parse error: {}", e) }),
                id: Value::Null,
            },
        };
        let response_body = serde_json::to_vec(&response).unwrap_or_default();
        Self::send_response(response_out, 200, headers, &response_body);
    }

    fn handle_jsonrpc(req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);
        match req.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": { "name": "wasmmcp-image-resize-http", "version": "0.1.0" },
                    "capabilities": { "tools": {} }
                })),
                error: None,
                id,
            },
            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: Some(json!({ "tools": Self::get_tool_list() })),
                error: None,
                id,
            },
            "tools/call" => {
                let params = req.params.unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_args = params.get("arguments").cloned().unwrap_or(json!({}));
                match Self::call_tool(tool_name, tool_args) {
                    Ok(result) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({ "content": [{ "type": "text", "text": result }] })),
                        error: None, id,
                    },
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        result: Some(json!({ "content": [{ "type": "text", "text": e }], "isError": true })),
                        error: None, id,
                    },
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(JsonRpcError { code: -32601, message: format!("Method not found: {}", req.method) }),
                id,
            },
        }
    }

    fn get_tool_list() -> Vec<Value> {
        vec![
            json!({
                "name": "get_image_info",
                "description": "Get detailed information about an image file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": { "type": "string", "description": "Path to the image file" }
                    },
                    "required": ["image_path"]
                }
            }),
            json!({
                "name": "resize_image",
                "description": "Resize an image and return the result as base64-encoded data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": { "type": "string", "description": "Path to the input image" },
                        "width": { "type": "integer", "description": "Target width" },
                        "height": { "type": "integer", "description": "Target height" },
                        "max_size": { "type": "integer", "description": "Maximum dimension" },
                        "quality": { "type": "integer", "description": "Output quality (1-100)" },
                        "output_format": { "type": "string", "description": "JPEG, PNG, or WEBP" }
                    },
                    "required": ["image_path"]
                }
            }),
            json!({
                "name": "scan_directory",
                "description": "Scan a directory for image files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": { "type": "string", "description": "Directory path to scan" },
                        "extensions": { "type": "array", "items": { "type": "string" } },
                        "recursive": { "type": "boolean", "description": "Scan subdirectories" },
                        "include_info": { "type": "boolean", "description": "Include detailed info" }
                    },
                    "required": ["directory"]
                }
            }),
            json!({
                "name": "compute_image_hash",
                "description": "Compute perceptual hash of an image for duplicate detection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": { "type": "string", "description": "Path to the image" },
                        "hash_type": { "type": "string", "description": "phash, dhash, or ahash" }
                    },
                    "required": ["image_path"]
                }
            }),
            json!({
                "name": "compare_hashes",
                "description": "Compare image hashes to find duplicates",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hashes": { "type": "array", "description": "List of hash results" },
                        "threshold": { "type": "integer", "description": "Max difference for duplicate" }
                    },
                    "required": ["hashes"]
                }
            }),
            json!({
                "name": "batch_resize",
                "description": "Resize multiple images at once",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_paths": { "type": "array", "items": { "type": "string" } },
                        "max_size": { "type": "integer", "description": "Maximum dimension" },
                        "quality": { "type": "integer" },
                        "output_format": { "type": "string" }
                    },
                    "required": ["image_paths"]
                }
            }),
        ]
    }

    fn call_tool(name: &str, args: Value) -> Result<String, String> {
        match name {
            "get_image_info" => {
                let path = args.get("image_path").and_then(|v| v.as_str()).ok_or("image_path required")?;
                Self::get_image_info(path)
            }
            "resize_image" => {
                let path = args.get("image_path").and_then(|v| v.as_str()).ok_or("image_path required")?;
                let width = args.get("width").and_then(|v| v.as_u64()).map(|v| v as u32);
                let height = args.get("height").and_then(|v| v.as_u64()).map(|v| v as u32);
                let max_size = args.get("max_size").and_then(|v| v.as_u64()).map(|v| v as u32);
                let quality = args.get("quality").and_then(|v| v.as_u64()).unwrap_or(85) as u8;
                let format = args.get("output_format").and_then(|v| v.as_str()).unwrap_or("JPEG");
                Self::resize_image(path, width, height, max_size, quality, format)
            }
            "scan_directory" => {
                let dir = args.get("directory").and_then(|v| v.as_str()).ok_or("directory required")?;
                let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(true);
                let include_info = args.get("include_info").and_then(|v| v.as_bool()).unwrap_or(false);
                let extensions: Option<Vec<String>> = args.get("extensions")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());
                Self::scan_directory(dir, extensions, recursive, include_info)
            }
            "compute_image_hash" => {
                let path = args.get("image_path").and_then(|v| v.as_str()).ok_or("image_path required")?;
                let hash_type = args.get("hash_type").and_then(|v| v.as_str()).unwrap_or("phash");
                Self::compute_image_hash(path, hash_type)
            }
            "compare_hashes" => {
                let hashes = args.get("hashes").and_then(|v| v.as_array()).ok_or("hashes required")?;
                let threshold = args.get("threshold").and_then(|v| v.as_u64()).unwrap_or(5) as u32;
                Self::compare_hashes(hashes, threshold)
            }
            "batch_resize" => {
                let paths: Vec<String> = args.get("image_paths")
                    .and_then(|v| v.as_array())
                    .ok_or("image_paths required")?
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let max_size = args.get("max_size").and_then(|v| v.as_u64()).unwrap_or(150) as u32;
                let quality = args.get("quality").and_then(|v| v.as_u64()).unwrap_or(75) as u8;
                let format = args.get("output_format").and_then(|v| v.as_str()).unwrap_or("JPEG");
                Self::batch_resize(&paths, max_size, quality, format)
            }
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }

    // Image processing helpers
    fn get_image_format(path: &str) -> Option<ImageFormat> {
        let ext = Path::new(path).extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            "gif" => Some(ImageFormat::Gif),
            "webp" => Some(ImageFormat::WebP),
            _ => None,
        }
    }

    fn output_format(s: &str) -> ImageFormat {
        match s.to_uppercase().as_str() {
            "PNG" => ImageFormat::Png,
            "WEBP" => ImageFormat::WebP,
            _ => ImageFormat::Jpeg,
        }
    }

    fn compute_ahash(img: &image::DynamicImage) -> String {
        let small = img.resize_exact(8, 8, FilterType::Lanczos3);
        let gray = small.to_luma8();
        let pixels: Vec<u8> = gray.pixels().map(|p| p.0[0]).collect();
        let avg: u64 = pixels.iter().map(|&p| p as u64).sum::<u64>() / pixels.len() as u64;
        let mut hash: u64 = 0;
        for (i, &pixel) in pixels.iter().enumerate() {
            if pixel as u64 >= avg { hash |= 1 << i; }
        }
        format!("{:016x}", hash)
    }

    fn compute_dhash(img: &image::DynamicImage) -> String {
        let small = img.resize_exact(9, 8, FilterType::Lanczos3);
        let gray = small.to_luma8();
        let mut hash: u64 = 0;
        let mut bit = 0;
        for y in 0..8 {
            for x in 0..8 {
                if gray.get_pixel(x, y).0[0] > gray.get_pixel(x + 1, y).0[0] {
                    hash |= 1 << bit;
                }
                bit += 1;
            }
        }
        format!("{:016x}", hash)
    }

    fn compute_phash(img: &image::DynamicImage) -> String {
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
            if val > avg { hash |= 1 << i; }
        }
        format!("{:016x}", hash)
    }

    fn hash_distance(h1: &str, h2: &str) -> u32 {
        let v1 = u64::from_str_radix(h1, 16).unwrap_or(0);
        let v2 = u64::from_str_radix(h2, 16).unwrap_or(0);
        (v1 ^ v2).count_ones()
    }

    // Tool implementations
    fn get_image_info(path: &str) -> Result<String, String> {
        let meta = fs::metadata(path).map_err(|e| format!("Cannot access: {}", e))?;
        let img = image::open(path).map_err(|e| format!("Cannot open: {}", e))?;
        let (w, h) = img.dimensions();
        let fmt = Self::get_image_format(path).map(|f| format!("{:?}", f)).unwrap_or("Unknown".into());
        Ok(json!({
            "path": path, "format": fmt, "mode": format!("{:?}", img.color()),
            "width": w, "height": h, "size_bytes": meta.len(),
            "aspect_ratio": if h > 0 { (w as f64 / h as f64 * 100.0).round() / 100.0 } else { 0.0 }
        }).to_string())
    }

    fn resize_image(path: &str, width: Option<u32>, height: Option<u32>, max_size: Option<u32>, quality: u8, output_format: &str) -> Result<String, String> {
        let orig_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let img = image::open(path).map_err(|e| format!("Cannot open: {}", e))?;
        let (ow, oh) = img.dimensions();

        let (nw, nh) = if let Some(ms) = max_size {
            let r = (ms as f64 / ow.max(oh) as f64).min(1.0);
            ((ow as f64 * r) as u32, (oh as f64 * r) as u32)
        } else if let (Some(w), Some(h)) = (width, height) { (w, h) }
        else if let Some(w) = width { (w, (oh as f64 * w as f64 / ow as f64) as u32) }
        else if let Some(h) = height { ((ow as f64 * h as f64 / oh as f64) as u32, h) }
        else { return Err("No resize parameters".into()); };

        let resized = img.resize(nw, nh, FilterType::Lanczos3);
        let fmt = Self::output_format(output_format);
        let out = if fmt == ImageFormat::Jpeg { image::DynamicImage::ImageRgb8(resized.to_rgb8()) } else { resized };

        let mut buf = Cursor::new(Vec::new());
        if fmt == ImageFormat::Jpeg {
            out.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality))
                .map_err(|e| format!("Encode error: {}", e))?;
        } else {
            out.write_to(&mut buf, fmt).map_err(|e| format!("Encode error: {}", e))?;
        }

        let out_bytes = buf.get_ref().len() as u64;
        Ok(json!({
            "success": true, "path": path, "original_size": [ow, oh], "new_size": [nw, nh],
            "original_bytes": orig_bytes, "output_bytes": out_bytes,
            "reduction_ratio": if orig_bytes > 0 { (out_bytes as f64 / orig_bytes as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
            "format": output_format, "data_base64": BASE64.encode(buf.get_ref())
        }).to_string())
    }

    fn scan_directory(dir: &str, extensions: Option<Vec<String>>, recursive: bool, include_info: bool) -> Result<String, String> {
        let exts: Vec<String> = extensions.unwrap_or_else(|| vec![".jpg",".jpeg",".png",".gif",".webp"].iter().map(|s| s.to_string()).collect());
        let dir_path = Path::new(dir);
        if !dir_path.exists() { return Err(format!("Directory not found: {}", dir)); }

        let mut paths: Vec<String> = Vec::new();
        let mut total: u64 = 0;

        fn scan(d: &Path, exts: &[String], rec: bool, paths: &mut Vec<String>, total: &mut u64) {
            if let Ok(entries) = fs::read_dir(d) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension() {
                            let e = format!(".{}", ext.to_string_lossy().to_lowercase());
                            if exts.iter().any(|x| x.to_lowercase() == e) {
                                *total += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                                paths.push(p.to_string_lossy().to_string());
                            }
                        }
                    } else if p.is_dir() && rec {
                        scan(&p, exts, rec, paths, total);
                    }
                }
            }
        }
        scan(dir_path, &exts, recursive, &mut paths, &mut total);

        let mut result = json!({
            "directory": dir, "image_count": paths.len(), "total_size_bytes": total,
            "total_size_mb": (total as f64 / 1048576.0 * 100.0).round() / 100.0, "image_paths": paths
        });

        if include_info {
            let infos: Vec<Value> = paths.iter().filter_map(|p| {
                image::open(p).ok().map(|img| {
                    let (w,h) = img.dimensions();
                    json!({"path": p, "width": w, "height": h, "size_bytes": fs::metadata(p).map(|m|m.len()).unwrap_or(0)})
                })
            }).collect();
            result["images"] = json!(infos);
        }
        Ok(result.to_string())
    }

    fn compute_image_hash(path: &str, hash_type: &str) -> Result<String, String> {
        match image::open(path) {
            Ok(img) => {
                let hash = match hash_type.to_lowercase().as_str() {
                    "ahash" => Self::compute_ahash(&img),
                    "dhash" => Self::compute_dhash(&img),
                    _ => Self::compute_phash(&img),
                };
                Ok(json!({"path": path, "hash": hash, "hash_type": hash_type}).to_string())
            }
            Err(e) => Ok(json!({"path": path, "error": format!("Cannot open: {}", e)}).to_string())
        }
    }

    fn compare_hashes(hashes: &[Value], threshold: u32) -> Result<String, String> {
        let valid: Vec<(&str, &str)> = hashes.iter()
            .filter_map(|h| {
                let path = h.get("path")?.as_str()?;
                let hash = h.get("hash")?.as_str()?;
                if h.get("error").is_some() { None } else { Some((path, hash)) }
            }).collect();

        if valid.len() < 2 {
            let errors: Vec<&Value> = hashes.iter().filter(|h| h.get("error").is_some()).collect();
            return Ok(json!({"total_compared": valid.len(), "duplicate_groups": [], "unique_count": valid.len(), "errors": errors}).to_string());
        }

        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut processed: HashSet<&str> = HashSet::new();

        for i in 0..valid.len() {
            let (p1, h1) = valid[i];
            if processed.contains(p1) { continue; }
            let mut group = vec![p1.to_string()];
            for j in (i+1)..valid.len() {
                let (p2, h2) = valid[j];
                if processed.contains(p2) { continue; }
                if Self::hash_distance(h1, h2) <= threshold {
                    group.push(p2.to_string());
                    processed.insert(p2);
                }
            }
            if group.len() > 1 { processed.insert(p1); groups.push(group); }
        }

        let dups: HashSet<&str> = groups.iter().flat_map(|g| g.iter().map(|s| s.as_str())).collect();
        let unique: Vec<String> = valid.iter().filter(|(p,_)| !dups.contains(*p)).map(|(p,_)| p.to_string()).collect();
        let errors: Vec<&Value> = hashes.iter().filter(|h| h.get("error").is_some()).collect();

        Ok(json!({
            "total_compared": valid.len(), "duplicate_groups": groups, "duplicate_group_count": groups.len(),
            "unique_paths": unique, "unique_count": unique.len(), "threshold": threshold, "errors": errors
        }).to_string())
    }

    fn batch_resize(paths: &[String], max_size: u32, quality: u8, output_format: &str) -> Result<String, String> {
        let fmt = Self::output_format(output_format);
        let mut results = Vec::new();
        let (mut total_in, mut total_out, mut ok, mut fail) = (0u64, 0u64, 0, 0);

        for path in paths {
            let orig = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            total_in += orig;

            match image::open(path) {
                Ok(img) => {
                    let (ow, oh) = img.dimensions();
                    let r = (max_size as f64 / ow.max(oh) as f64).min(1.0);
                    let (nw, nh) = ((ow as f64 * r) as u32, (oh as f64 * r) as u32);
                    let resized = img.resize(nw, nh, FilterType::Lanczos3);
                    let out = if fmt == ImageFormat::Jpeg { image::DynamicImage::ImageRgb8(resized.to_rgb8()) } else { resized };

                    let mut buf = Cursor::new(Vec::new());
                    let enc = if fmt == ImageFormat::Jpeg {
                        out.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality))
                    } else { out.write_to(&mut buf, fmt) };

                    match enc {
                        Ok(_) => {
                            let ob = buf.get_ref().len() as u64;
                            total_out += ob; ok += 1;
                            results.push(json!({"path": path, "success": true, "original_bytes": orig, "output_bytes": ob, "new_size": [nw,nh]}));
                        }
                        Err(e) => { fail += 1; results.push(json!({"path": path, "success": false, "error": format!("{}", e)})); }
                    }
                }
                Err(e) => { fail += 1; results.push(json!({"path": path, "success": false, "error": format!("{}", e)})); }
            }
        }

        Ok(json!({
            "total_images": paths.len(), "successful": ok, "failed": fail,
            "total_input_bytes": total_in, "total_output_bytes": total_out,
            "overall_reduction": if total_in > 0 { (total_out as f64 / total_in as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
            "results": results
        }).to_string())
    }

    fn read_body(request: &IncomingRequest) -> Vec<u8> {
        let mut body = Vec::new();
        if let Some(incoming) = request.consume().ok() {
            if let Ok(stream) = incoming.stream() {
                loop {
                    match stream.blocking_read(4096) {
                        Ok(chunk) if !chunk.is_empty() => body.extend_from_slice(&chunk),
                        _ => break,
                    }
                }
            }
        }
        body
    }

    fn send_response(out: ResponseOutparam, status: u16, headers: Fields, body: &[u8]) {
        let resp = OutgoingResponse::new(headers);
        let _ = resp.set_status_code(status);
        let outgoing_body = resp.body().unwrap();
        ResponseOutparam::set(out, Ok(resp));
        if !body.is_empty() {
            if let Ok(stream) = outgoing_body.write() { let _ = stream.blocking_write_and_flush(body); }
        }
        let _ = OutgoingBody::finish(outgoing_body, None);
    }
}

struct HttpHandler;
impl wasi::exports::http::incoming_handler::Guest for HttpHandler {
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) {
        McpHttpHandler::handle(request, response_out);
    }
}
wasi::http::proxy::export!(HttpHandler);
