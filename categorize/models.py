"""Data models for MCP Tool Classification."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json


@dataclass
class MCPTool:
    """Represents a single MCP tool."""
    name: str
    title: Optional[str] = None
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        return cls(
            name=data.get("name", ""),
            title=data.get("title"),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema"),
        )


@dataclass
class MCPServer:
    """Represents an MCP server with its tools."""
    server_id: str
    server_name: str
    source: str
    server_url: Optional[str] = None
    tools_count: int = 0
    tools: List[MCPTool] = field(default_factory=list)
    status: str = "success"
    error_message: Optional[str] = None

    # Additional metadata
    connection_types: Optional[List[str]] = None
    connections: Optional[Dict[str, Any]] = None
    remote: Optional[bool] = None
    use_count: Optional[int] = None
    collected_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServer":
        tools = [MCPTool.from_dict(t) for t in data.get("tools", [])]
        return cls(
            server_id=data.get("server_id", ""),
            server_name=data.get("server_name", ""),
            source=data.get("source", ""),
            server_url=data.get("server_url"),
            tools_count=data.get("tools_count", len(tools)),
            tools=tools,
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            connection_types=data.get("connection_types"),
            connections=data.get("connections"),
            remote=data.get("remote"),
            use_count=data.get("use_count"),
            collected_at=data.get("collected_at"),
        )

    def get_tools_summary(self, max_tools: int = 10) -> str:
        """Get a summary of tools for LLM classification."""
        if not self.tools:
            return "No tools available"

        summaries = []
        for tool in self.tools[:max_tools]:
            desc = tool.description[:200] + "..." if len(tool.description) > 200 else tool.description
            summaries.append(f"- {tool.name}: {desc}")

        if len(self.tools) > max_tools:
            summaries.append(f"... and {len(self.tools) - max_tools} more tools")

        return "\n".join(summaries)

    def get_all_descriptions(self) -> str:
        """Get concatenated descriptions of all tools."""
        descriptions = [self.server_name]
        for tool in self.tools:
            if tool.description:
                descriptions.append(tool.description)
        return " ".join(descriptions).lower()


@dataclass
class ClassificationResult:
    """Result of classifying an MCP server."""
    server_id: str
    server_name: str
    category_major: str
    category_minor: str
    tools_count: int
    confidence: float
    method: str  # "keyword" or "llm_{model_name}"
    matched_keyword: Optional[str] = None
    reasoning: Optional[str] = None
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "server_name": self.server_name,
            "category_major": self.category_major,
            "category_minor": self.category_minor,
            "tools_count": self.tools_count,
            "confidence": self.confidence,
            "method": self.method,
            "matched_keyword": self.matched_keyword,
            "reasoning": self.reasoning,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        return cls(
            server_id=data.get("server_id", ""),
            server_name=data.get("server_name", ""),
            category_major=data.get("category_major", ""),
            category_minor=data.get("category_minor", ""),
            tools_count=data.get("tools_count", 0),
            confidence=data.get("confidence", 0.0),
            method=data.get("method", ""),
            matched_keyword=data.get("matched_keyword"),
            reasoning=data.get("reasoning"),
            source=data.get("source", ""),
        )


def load_servers_from_json(filepath: str) -> List[MCPServer]:
    """Load MCP servers from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    servers = []
    for server_data in data.get("servers", []):
        servers.append(MCPServer.from_dict(server_data))

    return servers


def save_results_to_csv(results: List[ClassificationResult], filepath: str) -> None:
    """Save classification results to CSV file."""
    import csv

    fieldnames = [
        "server_id", "server_name", "category_major", "category_minor",
        "tools_count", "confidence", "method", "matched_keyword", "source"
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            # Remove reasoning for CSV (too long)
            row.pop("reasoning", None)
            writer.writerow(row)


def save_results_to_json(results: List[ClassificationResult], filepath: str) -> None:
    """Save classification results to JSON file."""
    data = {
        "total_count": len(results),
        "results": [r.to_dict() for r in results]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results_from_csv(filepath: str) -> List[ClassificationResult]:
    """Load classification results from CSV file."""
    import csv

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['tools_count'] = int(row.get('tools_count', 0))
            row['confidence'] = float(row.get('confidence', 0.0))
            results.append(ClassificationResult.from_dict(row))

    return results
