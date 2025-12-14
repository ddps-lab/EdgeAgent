//! Time MCP Service - provides time operations and timezone conversion
//! Compatible with Python mcp-server-time

use chrono::{DateTime, Datelike, NaiveTime, TimeZone, Utc, Weekday};
use chrono_tz::Tz;
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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetCurrentTimeParams {
    /// IANA timezone name (required, like Python mcp-server-time)
    #[schemars(description = "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'Etc/UTC' as local timezone if no timezone provided by the user.")]
    pub timezone: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
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

/// Parse timezone string to Tz
fn parse_timezone(tz_str: &str) -> Result<Tz, String> {
    tz_str.parse::<Tz>().map_err(|_| {
        format!("Invalid timezone: '{}'. Use IANA timezone names like 'America/New_York', 'Asia/Seoul', 'UTC'", tz_str)
    })
}

/// Format datetime with timezone info (ISO 8601)
fn format_datetime<T: TimeZone>(dt: DateTime<T>) -> String
where
    T::Offset: std::fmt::Display
{
    dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

/// Get day of week name
fn day_of_week_name(weekday: Weekday) -> &'static str {
    match weekday {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
}

// Tool implementations using rmcp macros

#[tool_router]
impl TimeService {
    /// Get the current time in a specific timezone
    /// Output format matches Python mcp-server-time
    #[tool(description = "Get the current time in a specific timezone. Returns ISO 8601 formatted time with timezone offset.")]
    fn get_current_time(
        &self,
        Parameters(params): Parameters<GetCurrentTimeParams>,
    ) -> Result<String, String> {
        let tz = parse_timezone(&params.timezone)?;

        let now_utc = Utc::now();
        let now_local = now_utc.with_timezone(&tz);

        // Output format matches Python mcp-server-time:
        // {"timezone": "...", "datetime": "...", "day_of_week": "...", "is_dst": false}
        Ok(serde_json::json!({
            "timezone": params.timezone,
            "datetime": format_datetime(now_local),
            "day_of_week": day_of_week_name(now_local.weekday()),
            "is_dst": false
        }).to_string())
    }

    /// Convert time between timezones
    /// Input format: HH:MM (24-hour format), matching Python mcp-server-time
    #[tool(description = "Convert a time from one timezone to another. Input time should be in HH:MM 24-hour format.")]
    fn convert_time(
        &self,
        Parameters(params): Parameters<ConvertTimeParams>,
    ) -> Result<String, String> {
        let source_tz = parse_timezone(&params.source_timezone)?;
        let target_tz = parse_timezone(&params.target_timezone)?;

        // Parse HH:MM format (matching Python mcp-server-time)
        let naive_time = NaiveTime::parse_from_str(&params.time, "%H:%M")
            .map_err(|_| format!("Invalid time format. Expected HH:MM [24-hour format]"))?;

        // Use today's date with the given time
        let today = Utc::now().with_timezone(&source_tz).date_naive();
        let naive_dt = today.and_time(naive_time);

        // Create datetime in source timezone
        let source_dt = source_tz.from_local_datetime(&naive_dt)
            .single()
            .ok_or_else(|| format!("Ambiguous or invalid time '{}' in timezone '{}'", params.time, params.source_timezone))?;

        // Convert to target timezone
        let target_dt = source_dt.with_timezone(&target_tz);

        // Output format matches Python mcp-server-time
        Ok(serde_json::json!({
            "source": {
                "timezone": params.source_timezone,
                "datetime": format_datetime(source_dt)
            },
            "target": {
                "timezone": params.target_timezone,
                "datetime": format_datetime(target_dt)
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
