//! Shared tool implementations for Time MCP Server
//!
//! These functions contain the core business logic and can be reused
//! by both stdio (service.rs) and HTTP (time-http) transports.

use chrono::{DateTime, Datelike, NaiveTime, TimeZone, Utc, Weekday};
use chrono_tz::Tz;
#[allow(unused_imports)]
use wasmmcp::timing::measure_io;

/// Parse timezone string to Tz
pub fn parse_timezone(tz_str: &str) -> Result<Tz, String> {
    tz_str.parse::<Tz>().map_err(|_| {
        format!("Invalid timezone: '{}'. Use IANA timezone names like 'America/New_York', 'Asia/Seoul', 'UTC'", tz_str)
    })
}

/// Format datetime with timezone info (ISO 8601)
pub fn format_datetime<T: TimeZone>(dt: DateTime<T>) -> String
where
    T::Offset: std::fmt::Display
{
    dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
}

/// Get day of week name
pub fn day_of_week_name(weekday: Weekday) -> &'static str {
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

/// Get the current time in a specific timezone
///
/// # Arguments
/// * `timezone` - IANA timezone name (e.g., 'America/New_York', 'Asia/Seoul', 'UTC')
///
/// # Returns
/// JSON string with timezone, datetime, day_of_week, and is_dst fields
pub fn get_current_time(timezone: &str) -> Result<String, String> {
    let tz = parse_timezone(timezone)?;

    let now_utc = Utc::now();
    let now_local = now_utc.with_timezone(&tz);

    Ok(serde_json::json!({
        "timezone": timezone,
        "datetime": format_datetime(now_local),
        "day_of_week": day_of_week_name(now_local.weekday()),
        "is_dst": false
    }).to_string())
}

/// Convert time between timezones
///
/// # Arguments
/// * `source_timezone` - Source IANA timezone name
/// * `time` - Time in HH:MM 24-hour format
/// * `target_timezone` - Target IANA timezone name
///
/// # Returns
/// JSON string with source and target datetime information
pub fn convert_time(source_timezone: &str, time: &str, target_timezone: &str) -> Result<String, String> {
    let source_tz = parse_timezone(source_timezone)?;
    let target_tz = parse_timezone(target_timezone)?;

    // Parse HH:MM format (matching Python mcp-server-time)
    let naive_time = NaiveTime::parse_from_str(time, "%H:%M")
        .map_err(|_| format!("Invalid time format. Expected HH:MM [24-hour format]"))?;

    // Use today's date with the given time
    let today = Utc::now().with_timezone(&source_tz).date_naive();
    let naive_dt = today.and_time(naive_time);

    // Create datetime in source timezone
    let source_dt = source_tz.from_local_datetime(&naive_dt)
        .single()
        .ok_or_else(|| format!("Ambiguous or invalid time '{}' in timezone '{}'", time, source_timezone))?;

    // Convert to target timezone
    let target_dt = source_dt.with_timezone(&target_tz);

    Ok(serde_json::json!({
        "source": {
            "timezone": source_timezone,
            "datetime": format_datetime(source_dt)
        },
        "target": {
            "timezone": target_timezone,
            "datetime": format_datetime(target_dt)
        }
    }).to_string())
}
