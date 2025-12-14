//! Sequential Thinking MCP Service
//!
//! Provides a tool for dynamic and reflective problem-solving through
//! a structured thinking process. Supports revision, branching, and
//! dynamic adjustment of thinking steps.
//!
//! Note: This is a stateless implementation. Each thought step is processed
//! independently, with the client maintaining the thinking history.

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
use serde_json::json;

/// Sequential Thinking MCP Service
#[derive(Debug, Clone)]
pub struct SequentialThinkingService {
    tool_router: ToolRouter<Self>,
}

impl SequentialThinkingService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for SequentialThinkingService {
    fn default() -> Self {
        Self::new()
    }
}

// Parameter struct for sequential_thinking tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SequentialThinkingParams {
    /// The current thinking step content
    #[schemars(description = "The current thinking step content")]
    pub thought: String,

    /// Whether another thought step is needed
    #[schemars(description = "Whether another thought step is needed after this one")]
    #[serde(rename = "nextThoughtNeeded")]
    pub next_thought_needed: bool,

    /// Current thought number in the sequence
    #[schemars(description = "Current thought number (1-indexed)", range(min = 1))]
    #[serde(rename = "thoughtNumber")]
    pub thought_number: i32,

    /// Estimated total number of thoughts needed
    #[schemars(description = "Estimated total number of thoughts needed", range(min = 1))]
    #[serde(rename = "totalThoughts")]
    pub total_thoughts: i32,

    /// Whether this thought revises previous thinking
    #[schemars(description = "Whether this thought revises previous thinking")]
    #[serde(rename = "isRevision", default)]
    pub is_revision: Option<bool>,

    /// Which thought number is being revised
    #[schemars(description = "Which thought number is being reconsidered (if isRevision is true)")]
    #[serde(rename = "revisesThought")]
    pub revises_thought: Option<i32>,

    /// Thought number to branch from
    #[schemars(description = "Thought number to branch from for alternative reasoning")]
    #[serde(rename = "branchFromThought")]
    pub branch_from_thought: Option<i32>,

    /// Identifier for the branch
    #[schemars(description = "Unique identifier for this reasoning branch")]
    #[serde(rename = "branchId")]
    pub branch_id: Option<String>,

    /// Whether more thoughts are needed beyond the original estimate
    #[schemars(description = "Whether more thoughts are needed beyond the original estimate")]
    #[serde(rename = "needsMoreThoughts", default)]
    pub needs_more_thoughts: Option<bool>,
}

// Tool implementations - Output format matches Node mcp-server-sequential-thinking
#[tool_router]
impl SequentialThinkingService {
    /// Process a sequential thinking step
    /// Tool name and output format match Node mcp-server-sequential-thinking
    /// Note: TypeScript server validates thoughtNumber >= 1 and totalThoughts >= 1 via zod schema
    #[tool(name = "sequentialthinking", description = "A detailed tool for dynamic and reflective problem-solving through structured thinking. Facilitates step-by-step analysis with support for revisions, branching into alternative paths, and dynamic adjustment of the thinking process. Use this for complex problems requiring careful reasoning.")]
    fn sequentialthinking(
        &self,
        Parameters(params): Parameters<SequentialThinkingParams>,
    ) -> Result<String, String> {
        // TypeScript validates thoughtNumber >= 1 and totalThoughts >= 1 via zod schema
        if params.thought_number < 1 {
            return Err("thoughtNumber must be >= 1".to_string());
        }
        if params.total_thoughts < 1 {
            return Err("totalThoughts must be >= 1".to_string());
        }

        // Build branches array (empty for now, matching Node server)
        let branches: Vec<String> = Vec::new();

        // Build response - matches Node mcp-server-sequential-thinking output format
        let response = json!({
            "thoughtNumber": params.thought_number,
            "totalThoughts": params.total_thoughts,
            "nextThoughtNeeded": params.next_thought_needed,
            "branches": branches,
            "thoughtHistoryLength": params.thought_number
        });

        Ok(response.to_string())
    }
}

#[tool_handler]
impl ServerHandler for SequentialThinkingService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Sequential Thinking MCP Server - A tool for dynamic and reflective problem-solving. \
                Use the sequential_thinking tool to break down complex problems into manageable steps, \
                with support for revisions, branching, and dynamic adjustment of the thinking process.".into()
            ),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
