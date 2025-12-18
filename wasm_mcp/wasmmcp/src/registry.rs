//! Tool Registry - Transport-agnostic tool management
//!
//! This module provides a `Tool` trait and `ToolRegistry` for managing MCP tools
//! independently of the transport layer. This allows the same tool implementations
//! to be used with both stdio and HTTP transports.

use std::collections::HashMap;
use serde_json::Value;
use schemars::JsonSchema;

/// Recursively fix "items": true to "items": {} in JSON Schema
/// This ensures compatibility with LLMs that require items to be an object
fn fix_array_items_schema(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // If this object has "items": true, replace with "items": {}
            if let Some(items) = map.get_mut("items") {
                if *items == Value::Bool(true) {
                    *items = Value::Object(serde_json::Map::new());
                } else {
                    // Recursively fix nested schemas
                    fix_array_items_schema(items);
                }
            }
            // Recursively process all other fields
            for (_, v) in map.iter_mut() {
                fix_array_items_schema(v);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                fix_array_items_schema(item);
            }
        }
        _ => {}
    }
}

/// Information about a registered tool
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// JSON Schema for tool parameters
    pub input_schema: Value,
}

/// Trait for MCP tools that can be invoked with JSON arguments
///
/// Implement this trait for each tool to enable transport-agnostic invocation.
pub trait Tool: Send + Sync {
    /// Get the tool name
    fn name(&self) -> &'static str;

    /// Get the tool description
    fn description(&self) -> &'static str;

    /// Get the JSON Schema for the tool's input parameters
    fn input_schema(&self) -> Value;

    /// Invoke the tool with JSON arguments and return JSON result
    fn invoke(&self, args: Value) -> Result<Value, String>;
}

/// A function-based tool wrapper
///
/// Wraps a function to implement the `Tool` trait.
pub struct FnTool<P, F>
where
    P: serde::de::DeserializeOwned + JsonSchema,
    F: Fn(P) -> Result<String, String> + Send + Sync,
{
    name: &'static str,
    description: &'static str,
    func: F,
    _phantom: std::marker::PhantomData<P>,
}

impl<P, F> FnTool<P, F>
where
    P: serde::de::DeserializeOwned + JsonSchema,
    F: Fn(P) -> Result<String, String> + Send + Sync,
{
    /// Create a new function-based tool
    pub fn new(name: &'static str, description: &'static str, func: F) -> Self {
        Self {
            name,
            description,
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P, F> Tool for FnTool<P, F>
where
    P: serde::de::DeserializeOwned + JsonSchema + Send + Sync,
    F: Fn(P) -> Result<String, String> + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn input_schema(&self) -> Value {
        let schema = schemars::schema_for!(P);
        let mut value = serde_json::to_value(schema).unwrap_or(Value::Object(serde_json::Map::new()));
        // Fix "items": true to "items": {} for broader LLM compatibility
        fix_array_items_schema(&mut value);
        value
    }

    fn invoke(&self, args: Value) -> Result<Value, String> {
        let params: P = serde_json::from_value(args)
            .map_err(|e| format!("Invalid parameters: {}", e))?;

        let result = (self.func)(params)?;

        // Try to parse as JSON, otherwise return as string
        match serde_json::from_str(&result) {
            Ok(json) => Ok(json),
            Err(_) => Ok(Value::String(result)),
        }
    }
}

/// Registry for managing tools
///
/// Provides a central place to register tools and look them up by name.
/// This is transport-agnostic and can be used with any transport.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Register a function as a tool
    pub fn register_fn<P, F>(&mut self, name: &'static str, description: &'static str, func: F)
    where
        P: serde::de::DeserializeOwned + JsonSchema + Send + Sync + 'static,
        F: Fn(P) -> Result<String, String> + Send + Sync + 'static,
    {
        let tool = FnTool::new(name, description, func);
        self.register(Box::new(tool));
    }

    /// Get a list of all registered tools
    pub fn list(&self) -> Vec<ToolInfo> {
        self.tools
            .values()
            .map(|tool| ToolInfo {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }

    /// Get a list of tools in MCP format
    pub fn list_mcp(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "inputSchema": tool.input_schema()
                })
            })
            .collect()
    }

    /// Call a tool by name
    pub fn call(&self, name: &str, args: Value) -> Result<Value, String> {
        let tool = self.tools.get(name)
            .ok_or_else(|| format!("Unknown tool: {}", name))?;

        tool.invoke(args)
    }

    /// Check if a tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TestParams {
        value: String,
    }

    #[test]
    fn test_fn_tool() {
        let tool = FnTool::new(
            "test_tool",
            "A test tool",
            |params: TestParams| Ok(format!("Got: {}", params.value)),
        );

        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");

        let result = tool.invoke(serde_json::json!({"value": "hello"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry() {
        let mut registry = ToolRegistry::new();

        registry.register_fn(
            "greet",
            "Greet someone",
            |params: TestParams| Ok(format!("Hello, {}!", params.value)),
        );

        assert_eq!(registry.len(), 1);
        assert!(registry.has("greet"));

        let tools = registry.list();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "greet");
    }
}
