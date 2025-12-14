//! WasmMCP Procedural Macros
//!
//! Provides attribute macros for building MCP servers in WASM:
//! - `#[mcp_tool]` - Define a tool function with automatic schema generation
//! - `#[wasmmcp_main]` - Generate WASI entry point boilerplate

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemFn, FnArg, Pat, Lit, Expr, ExprLit, Meta, MetaNameValue};

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
