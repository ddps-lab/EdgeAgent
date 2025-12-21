//! WasmMCP Procedural Macros
//!
//! Provides attribute macros for building MCP servers in WASM:
//! - `#[mcp_tool]` - Define a tool function with automatic schema generation
//! - `#[wasmmcp_main]` - Generate WASI entry point boilerplate
//! - `#[wasmmcp_tool]` - Define a tool that implements the Tool trait
//! - `export_cli!` - Export McpServer for WASI CLI
//! - `export_http!` - Export McpServer for WASI HTTP

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemFn, FnArg, Pat, Lit, Expr, ExprLit, Meta, MetaNameValue, Ident};

/// Attribute macro for defining MCP tools.
///
/// # Example
/// ```rust,ignore
/// #[mcp_tool(description = "Read a file from the filesystem")]
/// fn read_file(path: String) -> Result<String, String> {
///     std::fs::read_to_string(&path).map_err(|e| e.to_string())
/// }
/// ```
///
/// This generates:
/// - A parameters struct with JSON schema
/// - The tool function wrapped for MCP dispatch
#[proc_macro_attribute]
pub fn mcp_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let attr_meta = parse_macro_input!(attr as Meta);

    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_output = &input.sig.output;

    // Extract description from attributes
    let description = extract_description(&attr_meta);

    // Generate parameter struct name
    let params_struct_name = format_ident!("{}Params", to_pascal_case(&fn_name.to_string()));

    // Extract parameters from function signature
    let params: Vec<_> = input.sig.inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let param_type = &pat_type.ty;
                return Some((param_name.clone(), param_type.clone()));
            }
        }
        None
    }).collect();

    // Generate parameter struct fields
    let struct_fields = params.iter().map(|(name, ty)| {
        quote! {
            pub #name: #ty,
        }
    });

    // Generate parameter destructuring
    let param_names: Vec<_> = params.iter().map(|(name, _)| name).collect();

    // Generate the output
    let expanded = quote! {
        /// Parameters for #fn_name tool
        #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
        pub struct #params_struct_name {
            #(#struct_fields)*
        }

        impl #params_struct_name {
            /// Tool description
            pub const DESCRIPTION: &'static str = #description;
        }

        /// Tool implementation
        #fn_vis fn #fn_name(params: #params_struct_name) #fn_output {
            let #params_struct_name { #(#param_names),* } = params;
            #fn_block
        }
    };

    TokenStream::from(expanded)
}

/// Extract description from attribute meta
fn extract_description(meta: &Meta) -> String {
    match meta {
        Meta::NameValue(MetaNameValue { value: Expr::Lit(ExprLit { lit: Lit::Str(lit_str), .. }), .. }) => {
            lit_str.value()
        }
        Meta::List(list) => {
            // Parse description = "..." from the list
            let tokens = list.tokens.clone();
            let parsed: Result<MetaNameValue, _> = syn::parse2(tokens);
            if let Ok(nv) = parsed {
                if nv.path.is_ident("description") {
                    if let Expr::Lit(ExprLit { lit: Lit::Str(lit_str), .. }) = nv.value {
                        return lit_str.value();
                    }
                }
            }
            "No description provided".to_string()
        }
        _ => "No description provided".to_string(),
    }
}

/// Convert snake_case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

/// Attribute macro for generating WASI CLI entry point.
///
/// # Example
/// ```rust,ignore
/// #[wasmmcp_main]
/// async fn main() {
///     let mcp = WasmMcp::new("my-server");
///     mcp.run().await;
/// }
/// ```
///
/// This generates the appropriate WASI exports for CLI mode.
#[proc_macro_attribute]
pub fn wasmmcp_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_block = &input.block;

    let expanded = quote! {
        struct WasmMcpRunner;

        impl wasi::exports::cli::run::Guest for WasmMcpRunner {
            fn run() -> Result<(), ()> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async move #fn_block);
                Ok(())
            }
        }

        wasi::cli::command::export!(WasmMcpRunner);
    };

    TokenStream::from(expanded)
}

/// Helper macro for the HTTP transport entry point
#[proc_macro_attribute]
pub fn wasmmcp_http(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_block = &input.block;

    let expanded = quote! {
        fn #fn_name() -> impl wasmmcp::server::McpHandler {
            #fn_block
        }

        struct WasmMcpHttpHandler;

        impl wasi::exports::http::incoming_handler::Guest for WasmMcpHttpHandler {
            fn handle(request: wasi::http::types::IncomingRequest, response_out: wasi::http::types::ResponseOutparam) {
                wasmmcp::transport::http::handle_http_request(#fn_name, request, response_out);
            }
        }

        wasi::http::proxy::export!(WasmMcpHttpHandler);
    };

    TokenStream::from(expanded)
}

/// Attribute macro for defining MCP tools that implement the Tool trait.
///
/// This macro generates:
/// - A parameters struct with JSON schema
/// - A tool struct that implements wasmmcp::registry::Tool
/// - A factory function to create the tool
///
/// # Example
/// ```rust,ignore
/// #[wasmmcp_tool(description = "Get the current time in a timezone")]
/// fn get_current_time(timezone: String) -> Result<String, String> {
///     // Implementation
/// }
/// ```
///
/// This generates:
/// - `GetCurrentTimeParams` struct
/// - `GetCurrentTimeTool` struct implementing `Tool`
/// - `get_current_time(params)` function
/// - `get_current_time_tool()` factory function
#[proc_macro_attribute]
pub fn wasmmcp_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let attr_meta = parse_macro_input!(attr as Meta);

    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_output = &input.sig.output;

    // Extract description from attributes
    let description = extract_description(&attr_meta);
    let fn_name_str = fn_name.to_string();

    // Generate names
    let params_struct_name = format_ident!("{}Params", to_pascal_case(&fn_name_str));
    let tool_struct_name = format_ident!("{}Tool", to_pascal_case(&fn_name_str));
    let factory_fn_name = format_ident!("{}_tool", fn_name);

    // Extract parameters from function signature
    let params: Vec<_> = input.sig.inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let param_type = &pat_type.ty;
                return Some((param_name.clone(), param_type.clone()));
            }
        }
        None
    }).collect();

    // Generate parameter struct fields
    let struct_fields = params.iter().map(|(name, ty)| {
        quote! {
            pub #name: #ty,
        }
    });

    // Generate parameter destructuring
    let param_names: Vec<_> = params.iter().map(|(name, _)| name).collect();

    // Generate the output
    let expanded = quote! {
        /// Parameters for #fn_name tool
        #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
        #fn_vis struct #params_struct_name {
            #(#struct_fields)*
        }

        /// Tool implementation struct
        #fn_vis struct #tool_struct_name;

        impl wasmmcp::registry::Tool for #tool_struct_name {
            fn name(&self) -> &'static str {
                #fn_name_str
            }

            fn description(&self) -> &'static str {
                #description
            }

            fn input_schema(&self) -> serde_json::Value {
                let schema = schemars::schema_for!(#params_struct_name);
                serde_json::to_value(schema).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
            }

            fn invoke(&self, args: serde_json::Value) -> Result<serde_json::Value, String> {
                let params: #params_struct_name = serde_json::from_value(args)
                    .map_err(|e| format!("Invalid parameters: {}", e))?;
                let result = #fn_name(params)?;

                // Try to parse as JSON, otherwise return as string
                match serde_json::from_str(&result) {
                    Ok(json) => Ok(json),
                    Err(_) => Ok(serde_json::Value::String(result)),
                }
            }
        }

        /// Tool implementation function
        #fn_vis fn #fn_name(params: #params_struct_name) #fn_output {
            let #params_struct_name { #(#param_names),* } = params;
            #fn_block
        }

        /// Create a new instance of the tool
        #fn_vis fn #factory_fn_name() -> Box<dyn wasmmcp::registry::Tool> {
            Box::new(#tool_struct_name)
        }
    };

    TokenStream::from(expanded)
}

/// Macro to export an McpServer for WASI CLI (stdio transport).
///
/// This macro creates a WASI CLI entry point that runs the McpServer
/// using stdio transport for MCP communication.
///
/// # Example
/// ```rust,ignore
/// use wasmmcp::prelude::*;
///
/// fn create_server() -> McpServer {
///     McpServerBuilder::new("my-server")
///         .tool("greet", "Greet someone", greet)
///         .build()
/// }
///
/// wasmmcp::export_cli!(create_server);
/// ```
#[proc_macro]
pub fn export_cli(input: TokenStream) -> TokenStream {
    let server_fn = parse_macro_input!(input as Ident);

    let expanded = quote! {
        struct WasmMcpCliRunner;

        impl wasi::exports::cli::run::Guest for WasmMcpCliRunner {
            fn run() -> Result<(), ()> {
                use std::io::{BufRead, Write};
                use std::time::Instant;

                // Server creation is part of cold start
                let server = #server_fn();

                // Start timing after server creation (for wasm_total)
                let wasm_start = Instant::now();

                // Simple stdio JSON-RPC loop
                let stdin = std::io::stdin();
                let mut stdout = std::io::stdout();

                for line in stdin.lock().lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => break,
                    };

                    if line.is_empty() {
                        continue;
                    }

                    // Check if this is tools/call (for JSON parse timing)
                    let is_tools_call = line.contains("\"tools/call\"");

                    // Parse JSON-RPC request
                    let json_parse_start = Instant::now();
                    let request: serde_json::Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if is_tools_call {
                        let json_parse_ms = json_parse_start.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("---JSON_PARSE---{:.3}", json_parse_ms);
                    }

                    let method = request.get("method")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let params = request.get("params").cloned();
                    let id = request.get("id").cloned();

                    // Handle the request
                    let result = server.handle_jsonrpc(method, params);

                    // Notifications (no id) don't expect a response per JSON-RPC 2.0 spec
                    let id = match id {
                        Some(id) => id,
                        None => continue,  // Skip response for notifications
                    };

                    let response = result.to_response(id);

                    // Write response
                    let response_str = serde_json::to_string(&response).unwrap_or_default();
                    let _ = writeln!(stdout, "{}", response_str);
                    let _ = stdout.flush();

                    // Output WASM total time after tools/call
                    if method == "tools/call" {
                        let wasm_total_ms = wasm_start.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("---WASM_TOTAL---{:.3}", wasm_total_ms);
                    }
                }

                Ok(())
            }
        }

        wasi::cli::command::export!(WasmMcpCliRunner);
    };

    TokenStream::from(expanded)
}

/// Macro to export an McpServer for WASI HTTP (serverless/proxy).
///
/// This macro creates a WASI HTTP handler that processes MCP requests
/// over HTTP using JSON-RPC.
///
/// # Example
/// ```rust,ignore
/// use wasmmcp::prelude::*;
///
/// fn create_server() -> McpServer {
///     McpServerBuilder::new("my-server")
///         .tool("greet", "Greet someone", greet)
///         .build()
/// }
///
/// wasmmcp::export_http!(create_server);
/// ```
#[proc_macro]
pub fn export_http(input: TokenStream) -> TokenStream {
    let server_fn = parse_macro_input!(input as Ident);

    let expanded = quote! {
        struct WasmMcpHttpHandler;

        impl wasi::exports::http::incoming_handler::Guest for WasmMcpHttpHandler {
            fn handle(request: wasi::http::types::IncomingRequest, response_out: wasi::http::types::ResponseOutparam) {
                use wasi::http::types::{Fields, OutgoingBody, OutgoingResponse, Method};

                let method = request.method();

                // CORS headers
                let headers = Fields::new();
                let _ = headers.set(&"Content-Type".to_string(), &[b"application/json".to_vec()]);
                let _ = headers.set(&"Access-Control-Allow-Origin".to_string(), &[b"*".to_vec()]);
                let _ = headers.set(&"Access-Control-Allow-Methods".to_string(), &[b"GET, POST, OPTIONS".to_vec()]);
                let _ = headers.set(&"Access-Control-Allow-Headers".to_string(), &[b"Content-Type".to_vec()]);

                // Handle CORS preflight
                if matches!(method, Method::Options) {
                    send_response(response_out, 204, headers, b"");
                    return;
                }

                // Only handle POST
                if !matches!(method, Method::Post) {
                    send_response(response_out, 405, headers, b"Method Not Allowed");
                    return;
                }

                // Read request body
                let body = read_body(&request);

                // Parse JSON-RPC request
                let json_request: serde_json::Value = match serde_json::from_slice(&body) {
                    Ok(v) => v,
                    Err(e) => {
                        let error_response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32700,
                                "message": format!("Parse error: {}", e)
                            },
                            "id": serde_json::Value::Null
                        });
                        let response_body = serde_json::to_vec(&error_response).unwrap_or_default();
                        send_response(response_out, 200, headers, &response_body);
                        return;
                    }
                };

                let server = #server_fn();
                let method_str = json_request.get("method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let params = json_request.get("params").cloned();
                let id = json_request.get("id").cloned();

                // Handle the request
                let result = server.handle_jsonrpc(method_str, params);

                // Notifications (no id) don't expect a response per JSON-RPC 2.0 spec
                // Per MCP Streamable HTTP spec, return 202 Accepted with no body
                let id = match id {
                    Some(id) => id,
                    None => {
                        send_response(response_out, 202, headers, b"");
                        return;
                    }
                };

                let response = result.to_response(id);

                // Add timing headers (values set by handle_tools_call during execution)
                let tool_exec_ms = wasmmcp::timing::get_tool_exec_ms();
                let io_ms = wasmmcp::timing::get_io_ms();
                let _ = headers.set(
                    &"X-Tool-Exec-Ms".to_string(),
                    &[format!("{:.3}", tool_exec_ms).into_bytes()],
                );
                let _ = headers.set(
                    &"X-IO-Ms".to_string(),
                    &[format!("{:.3}", io_ms).into_bytes()],
                );

                let response_body = match serde_json::to_vec(&response) {
                    Ok(body) => body,
                    Err(e) => {
                        // Serialization failed - return error response
                        let error_response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32603,
                                "message": format!("Internal error: failed to serialize response: {}", e)
                            },
                            "id": serde_json::Value::Null
                        });
                        serde_json::to_vec(&error_response).unwrap_or_else(|_| b"{}".to_vec())
                    }
                };
                send_response(response_out, 200, headers, &response_body);
            }
        }

        fn read_body(request: &wasi::http::types::IncomingRequest) -> Vec<u8> {
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

        fn send_response(out: wasi::http::types::ResponseOutparam, status: u16, headers: wasi::http::types::Fields, body: &[u8]) {
            use wasi::http::types::{OutgoingResponse, OutgoingBody};

            let resp = OutgoingResponse::new(headers);
            let _ = resp.set_status_code(status);
            let outgoing_body = resp.body().unwrap();
            wasi::http::types::ResponseOutparam::set(out, Ok(resp));

            if !body.is_empty() {
                if let Ok(stream) = outgoing_body.write() {
                    // Write in chunks due to WASI stream 4KB limit
                    // See: https://github.com/bytecodealliance/wasmtime/issues/9653
                    const CHUNK_SIZE: usize = 4096;
                    for chunk in body.chunks(CHUNK_SIZE) {
                        if stream.blocking_write_and_flush(chunk).is_err() {
                            break;
                        }
                    }
                }
            }

            let _ = OutgoingBody::finish(outgoing_body, None);
        }

        wasi::http::proxy::export!(WasmMcpHttpHandler);
    };

    TokenStream::from(expanded)
}
