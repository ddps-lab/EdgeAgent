//! WasmMCP Server Builder - FastMCP-like fluent API
//!
//! Provides a simple builder pattern for creating MCP servers with tools.
//! The builder can be used to create both stdio and HTTP servers.

use crate::registry::{Tool, ToolRegistry};
use crate::timing::{
    reset_io_accumulators, get_io_duration, get_disk_io_duration, get_network_io_duration,
    set_tool_exec_ms, set_io_ms, set_disk_io_ms, set_network_io_ms, set_compute_ms,
};
use serde_json::{json, Value};
use schemars::JsonSchema;
use std::time::Instant;

/// Server information
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
}

/// MCP Server Builder
///
/// Use this to create a new MCP server with tools in a fluent style.
///
/// # Example
/// ```ignore
/// let server = McpServerBuilder::new("my-server")
///     .version("1.0.0")
///     .tool("get_time", "Get current time", get_time_fn)
///     .build();
/// ```
pub struct McpServerBuilder {
    info: ServerInfo,
    registry: ToolRegistry,
}

impl McpServerBuilder {
    /// Create a new server builder with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            info: ServerInfo {
                name: name.into(),
                version: "1.0.0".to_string(),
                description: None,
            },
            registry: ToolRegistry::new(),
        }
    }

    /// Set the server version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.info.version = version.into();
        self
    }

    /// Set the server description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.info.description = Some(description.into());
        self
    }

    /// Register a tool using a function
    pub fn tool<P, F>(mut self, name: &'static str, description: &'static str, func: F) -> Self
    where
        P: serde::de::DeserializeOwned + JsonSchema + Send + Sync + 'static,
        F: Fn(P) -> Result<String, String> + Send + Sync + 'static,
    {
        self.registry.register_fn::<P, F>(name, description, func);
        self
    }

    /// Register a boxed tool
    pub fn register_tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.registry.register(tool);
        self
    }

    /// Build the server
    pub fn build(self) -> McpServer {
        McpServer {
            info: self.info,
            registry: self.registry,
        }
    }
}

/// MCP Server with tool registry
///
/// This is a transport-agnostic server that can be used with any transport.
pub struct McpServer {
    info: ServerInfo,
    registry: ToolRegistry,
}

impl McpServer {
    /// Create a new server builder
    pub fn builder(name: impl Into<String>) -> McpServerBuilder {
        McpServerBuilder::new(name)
    }

    /// Get server info
    pub fn info(&self) -> &ServerInfo {
        &self.info
    }

    /// Get the tool registry
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Handle MCP initialize request
    pub fn handle_initialize(&self) -> Value {
        json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.info.name,
                "version": self.info.version
            },
            "capabilities": {
                "tools": {}
            }
        })
    }

    /// Handle MCP tools/list request
    pub fn handle_tools_list(&self) -> Value {
        json!({
            "tools": self.registry.list_mcp()
        })
    }

    /// Handle MCP tools/call request
    pub fn handle_tools_call(&self, name: &str, args: Value) -> Result<Value, String> {
        // Reset all I/O accumulators before tool execution
        reset_io_accumulators();

        // Measure tool execution time
        let start = Instant::now();
        let result = self.registry.call(name, args)?;
        let tool_exec_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Get I/O timing (accumulated during tool execution via measure_io/measure_disk_io/measure_network_io)
        let io_duration = get_io_duration();
        let disk_io_duration = get_disk_io_duration();
        let network_io_duration = get_network_io_duration();

        let io_ms = io_duration.as_secs_f64() * 1000.0;
        let disk_io_ms = disk_io_duration.as_secs_f64() * 1000.0;
        let network_io_ms = network_io_duration.as_secs_f64() * 1000.0;
        let compute_ms = (tool_exec_ms - disk_io_ms - network_io_ms).max(0.0);

        // Store timing values for HTTP transport to read
        set_tool_exec_ms(tool_exec_ms);
        set_io_ms(io_ms);
        set_disk_io_ms(disk_io_ms);
        set_network_io_ms(network_io_ms);
        set_compute_ms(compute_ms);

        // Also output to stderr for CLI mode
        eprintln!("---TOOL_EXEC---{:.3}", tool_exec_ms);
        eprintln!("---IO---{:.3}", io_ms);
        eprintln!("---DISK_IO---{:.3}", disk_io_ms);
        eprintln!("---NETWORK_IO---{:.3}", network_io_ms);
        eprintln!("---COMPUTE---{:.3}", compute_ms);

        // Format as MCP content response
        let text = if result.is_string() {
            result.as_str().unwrap_or("").to_string()
        } else {
            serde_json::to_string(&result).unwrap_or_else(|_| result.to_string())
        };

        // Return MCP response with timing as second content item
        let timing_json = format!(
            r#"{{"_wasm_timing":{{"fn_total_ms":{},"io_ms":{},"disk_io_ms":{},"network_io_ms":{},"compute_ms":{}}}}}"#,
            tool_exec_ms, io_ms, disk_io_ms, network_io_ms, compute_ms
        );

        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "text",
                    "text": timing_json
                }
            ]
        }))
    }

    /// Handle a JSON-RPC request
    pub fn handle_jsonrpc(&self, method: &str, params: Option<Value>) -> JsonRpcResult {
        match method {
            "initialize" => JsonRpcResult::Success(self.handle_initialize()),

            "initialized" => JsonRpcResult::Success(Value::Null),

            "tools/list" => JsonRpcResult::Success(self.handle_tools_list()),

            "tools/call" => {
                let params = params.unwrap_or(Value::Null);
                let name = params.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = params.get("arguments")
                    .cloned()
                    .unwrap_or(json!({}));

                match self.handle_tools_call(name, args) {
                    Ok(result) => JsonRpcResult::Success(result),
                    Err(e) => JsonRpcResult::ToolError(e),
                }
            }

            _ => JsonRpcResult::Error {
                code: -32601,
                message: format!("Method not found: {}", method),
            }
        }
    }
}

/// Result of a JSON-RPC call
pub enum JsonRpcResult {
    Success(Value),
    ToolError(String),
    Error { code: i32, message: String },
}

impl JsonRpcResult {
    /// Convert to JSON-RPC response format
    pub fn to_response(self, id: Value) -> Value {
        match self {
            JsonRpcResult::Success(result) => json!({
                "jsonrpc": "2.0",
                "result": result,
                "id": id
            }),
            JsonRpcResult::ToolError(message) => json!({
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": message
                    }],
                    "isError": true
                },
                "id": id
            }),
            JsonRpcResult::Error { code, message } => json!({
                "jsonrpc": "2.0",
                "error": {
                    "code": code,
                    "message": message
                },
                "id": id
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct GreetParams {
        name: String,
    }

    fn greet(params: GreetParams) -> Result<String, String> {
        Ok(format!("Hello, {}!", params.name))
    }

    #[test]
    fn test_builder() {
        let server = McpServerBuilder::new("test-server")
            .version("1.0.0")
            .description("A test server")
            .tool("greet", "Greet someone", greet)
            .build();

        assert_eq!(server.info().name, "test-server");
        assert_eq!(server.info().version, "1.0.0");
        assert_eq!(server.registry().len(), 1);
    }

    #[test]
    fn test_handle_jsonrpc() {
        let server = McpServerBuilder::new("test")
            .tool("greet", "Greet", greet)
            .build();

        // Test initialize
        let result = server.handle_jsonrpc("initialize", None);
        assert!(matches!(result, JsonRpcResult::Success(_)));

        // Test tools/list
        let result = server.handle_jsonrpc("tools/list", None);
        assert!(matches!(result, JsonRpcResult::Success(_)));

        // Test tools/call
        let params = json!({
            "name": "greet",
            "arguments": { "name": "World" }
        });
        let result = server.handle_jsonrpc("tools/call", Some(params));
        assert!(matches!(result, JsonRpcResult::Success(_)));
    }
}
