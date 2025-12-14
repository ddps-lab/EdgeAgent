//! WasmMcp server implementation
//!
//! The core server struct that manages tool registration and transport handling.

use crate::transport::{Transport, StdioTransport};
use crate::{Error, Result};
use rmcp::{ServerHandler, ServiceExt};

/// Main WasmMCP server struct
///
/// Use `WasmMcp::builder()` to create a new server instance.
pub struct WasmMcp<H: ServerHandler + Clone + Send + Sync + 'static> {
    name: String,
    version: String,
    description: Option<String>,
    handler: H,
}

impl<H: ServerHandler + Clone + Send + Sync + 'static> WasmMcp<H> {
    /// Create a new WasmMcp server with a custom handler
    pub fn new(name: impl Into<String>, handler: H) -> Self {
        Self {
            name: name.into(),
            version: "1.0.0".to_string(),
            description: None,
            handler,
        }
    }

    /// Set the server version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the server description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Run the server with the default transport (stdio for CLI)
    #[cfg(feature = "transport-stdio")]
    pub async fn run(self) -> Result<()> {
        let transport = StdioTransport::new();
        self.run_with_transport(transport).await
    }

    /// Run the server with a specific transport
    pub async fn run_with_transport<T: Transport>(self, transport: T) -> Result<()> {
        let (input, output) = transport.streams();

        let server = self.handler
            .serve((input, output))
            .await
            .map_err(|e| Error::Transport(format!("Failed to start server: {:?}", e)))?;

        server.waiting()
            .await
            .map_err(|e| Error::Transport(format!("Server error: {:?}", e)))?;

        Ok(())
    }

    /// Get the server name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Builder for WasmMcp server
pub struct WasmMcpBuilder {
    name: String,
    version: String,
    description: Option<String>,
}

impl WasmMcpBuilder {
    /// Create a new builder with the given server name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "1.0.0".to_string(),
            description: None,
        }
    }

    /// Set the server version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the server description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Build the server with a custom handler
    pub fn build<H: ServerHandler + Clone + Send + Sync + 'static>(self, handler: H) -> WasmMcp<H> {
        WasmMcp {
            name: self.name,
            version: self.version,
            description: self.description,
            handler,
        }
    }
}

/// Trait for types that can be served by WasmMcp
pub trait McpHandler: ServerHandler + Clone + Send + Sync + 'static {
    /// Create a new instance of the handler
    fn create() -> Self;
}
