//! Time MCP Service - provides time operations and timezone conversion
//! Compatible with Python mcp-server-time

use wasmmcp::timing::{ToolTimer, get_wasm_total_ms};
use rmcp::{
    ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;

use crate::tools;

/// Time MCP Service
#[derive(Debug, Clone)]
pub struct TimeService {
    tool_router: ToolRouter<Self>,
}

impl TimeService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for TimeService {
    fn default() -> Self {
        Self::new()
    }
}

// Parameter structs for MCP tools - matching Python mcp-server-time

#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct GetCurrentTimeParams {
    /// IANA timezone name (required, like Python mcp-server-time)
    #[schemars(description = "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'Etc/UTC' as local timezone if no timezone provided by the user.")]
    pub timezone: String,
}

#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ConvertTimeParams {
    /// Source IANA timezone name
    #[schemars(description = "Source IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'Etc/UTC' as local timezone if no source timezone provided by the user.")]
    pub source_timezone: String,

    /// Time to convert in 24-hour format (HH:MM)
    #[schemars(description = "Time to convert in 24-hour format (HH:MM)")]
    pub time: String,

    /// Target IANA timezone name
    #[schemars(description = "Target IANA timezone name (e.g., 'Asia/Tokyo', 'America/San_Francisco'). Use 'Etc/UTC' as local timezone if no target timezone provided by the user.")]
    pub target_timezone: String,
}

// Tool implementations using rmcp macros - delegates to shared tools module

#[tool_router]
impl TimeService {
    /// Get the current time in a specific timezone
    /// Output format matches Python mcp-server-time
    #[tool(description = "Get the current time in a specific timezone. Returns ISO 8601 formatted time with timezone offset.")]
    fn get_current_time(
        &self,
        Parameters(params): Parameters<GetCurrentTimeParams>,
    ) -> Result<String, String> {
        let timer = ToolTimer::start();
        let result = tools::get_current_time(&params.timezone)?;
        let timing = timer.finish("get_current_time");
        Ok(serde_json::json!({
            "output": result,
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }

    /// Convert time between timezones
    /// Input format: HH:MM (24-hour format), matching Python mcp-server-time
    #[tool(description = "Convert a time from one timezone to another. Input time should be in HH:MM 24-hour format.")]
    fn convert_time(
        &self,
        Parameters(params): Parameters<ConvertTimeParams>,
    ) -> Result<String, String> {
        let timer = ToolTimer::start();
        let result = tools::convert_time(&params.source_timezone, &params.time, &params.target_timezone)?;
        let timing = timer.finish("convert_time");
        Ok(serde_json::json!({
            "output": result,
            "_timing": {
                "wasm_total_ms": get_wasm_total_ms(),
                "fn_total_ms": timing.fn_total_ms,
                "io_ms": timing.io_ms,
                "compute_ms": timing.compute_ms
            }
        }).to_string())
    }
}

#[tool_handler]
impl ServerHandler for TimeService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Time MCP Server - Get current time and convert between timezones. Uses IANA timezone names.".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
