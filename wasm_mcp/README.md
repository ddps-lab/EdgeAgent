# WasmMCP - Rust MCP Servers in WASM

A framework for building MCP (Model Context Protocol) servers in Rust, compiled to WASM.
Supports both Stdio (local) and HTTP (serverless) transports from a single crate.

## Overview

Write MCP servers once in Rust, compile to WASM, and run anywhere - locally via `wasmtime run`
or remotely via `wasmtime serve` / Knative / serverless platforms.

### Key Features

- **Pure WASM**: No Python/Node.js dependency - just WASM binaries
- **Unified Codebase**: Single crate with feature flags for CLI/HTTP builds
- **Lightweight**: ~300KB-1.3MB binaries (depends on server complexity)
- **9 MCP Servers**: time, filesystem, git, fetch, summarize, log-parser, etc.
- **FastMCP-like DX**: Declarative tool registration with `McpServer::builder()`

## Quick Start

### Prerequisites

```bash
# Install Rust WASI target
rustup target add wasm32-wasip2

# Install Wasmtime
curl https://wasmtime.dev/install.sh -sSf | bash
```

### Build All Servers

```bash
./scripts/build_all.sh
```

This builds all 9 servers in both CLI and HTTP variants:
- `mcp_server_*_cli.wasm` - for `wasmtime run` (stdio)
- `mcp_server_*_http.wasm` - for `wasmtime serve` (HTTP)

### Run Examples

```bash
# Stdio mode (local development)
wasmtime run ./target/wasm32-wasip2/release/mcp_server_time_cli.wasm

# HTTP mode (serverless)
wasmtime serve --addr 127.0.0.1:8000 \
  ./target/wasm32-wasip2/release/mcp_server_time_http.wasm
```

## Project Structure

```
wasm_mcp/
├── wasmmcp/                     # Core framework
│   ├── src/
│   │   ├── lib.rs               # Public API
│   │   ├── server.rs            # McpServer builder
│   │   ├── transport/           # Stdio/HTTP transports
│   │   └── protocol/            # JSON-RPC handling
│
├── wasmmcp-macros/              # export_cli!, export_http! macros
│
├── servers/                     # MCP server implementations
│   ├── time/                    # Time tools (timezone conversion)
│   ├── filesystem/              # 14 file system tools
│   ├── git/                     # Git repository tools
│   ├── fetch/                   # URL fetching (wasi:http)
│   ├── summarize/               # LLM-based summarization
│   ├── log-parser/              # Log parsing with regex
│   ├── data-aggregate/          # Data aggregation tools
│   ├── sequential-thinking/     # Structured thinking tool
│   └── image-resize/            # Image processing tools
│
├── scripts/
│   └── build_all.sh             # Build all servers
│
└── tests/
    └── compare_*.py             # Comparison tests vs native implementations
```

## Writing MCP Servers

### Minimal Example

```rust
use wasmmcp::schemars::JsonSchema;
use wasmmcp::serde::Deserialize;
use wasmmcp::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GreetParams {
    /// Name to greet
    pub name: String,
}

pub fn create_server() -> McpServer {
    McpServer::builder("my-server")
        .version("1.0.0")
        .tool::<GreetParams, _>(
            "greet",
            "Say hello to someone",
            |params| Ok(format!("Hello, {}!", params.name))
        )
        .build()
}

// Export for wasmtime run
#[cfg(feature = "cli-export")]
wasmmcp::export_cli!(create_server);

// Export for wasmtime serve
#[cfg(feature = "http-export")]
wasmmcp::export_http!(create_server);
```

### Cargo.toml

```toml
[package]
name = "mcp-server-myserver"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasmmcp = { path = "../../wasmmcp" }
wasi = "0.14"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
schemars = "1"

[features]
default = ["cli-export"]
cli-export = []
http-export = []
```

### Building

```bash
# CLI version
cargo build --target wasm32-wasip2 --release \
  -p mcp-server-myserver --features cli-export --no-default-features

# HTTP version
cargo build --target wasm32-wasip2 --release \
  -p mcp-server-myserver --features http-export --no-default-features
```

## Available Servers

| Server | Tools | Description |
|--------|-------|-------------|
| time | 2 | Timezone conversion (get_current_time, convert_time) |
| filesystem | 14 | File operations (read/write/edit/search/tree) |
| git | 12 | Git repository read operations |
| fetch | 1 | URL fetching with HTML-to-markdown conversion |
| summarize | 3 | LLM-powered text summarization (OpenAI/Upstage) |
| log-parser | 5 | Log parsing, filtering, statistics |
| data-aggregate | 5 | Data aggregation and deduplication |
| sequential-thinking | 1 | Structured problem-solving |
| image-resize | 6 | Image info, resize, hash, batch operations |

## Architecture

### Unified CLI/HTTP Build

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Logic (tools)                  │
├─────────────────────────────────────────────────────────────┤
│                    wasmmcp framework                         │
│                 (McpServer, ToolDef, Protocol)               │
├──────────────────────────┬──────────────────────────────────┤
│   export_cli!            │   export_http!                    │
│   (wasi:cli/run)         │   (wasi:http/incoming-handler)    │
├──────────────────────────┼──────────────────────────────────┤
│   wasmtime run           │   wasmtime serve / Knative        │
└──────────────────────────┴──────────────────────────────────┘
```

### Feature Flags

- `cli-export`: Exports `wasi:cli/run` for stdio communication
- `http-export`: Exports `wasi:http/incoming-handler` for HTTP requests

Both features share the same `create_server()` function, eliminating code duplication.

## LangChain Integration

### Stdio Connection

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_config = {
    "wasmmcp": {
        "transport": "stdio",
        "command": "wasmtime",
        "args": ["run", "--dir=/tmp", "mcp_server_filesystem_cli.wasm"],
    }
}

client = MultiServerMCPClient(mcp_config)
async with client.session("wasmmcp") as session:
    tools = await load_mcp_tools(session)
```

### HTTP Connection

```python
mcp_config = {
    "wasmmcp_http": {
        "transport": "streamable_http",
        "url": "http://localhost:8000",
    }
}
```

## Serverless Deployment

### Knative with SpinKube

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: wasmmcp-time
spec:
  template:
    spec:
      runtimeClassName: wasmtime-spin
      containers:
        - image: ghcr.io/example/wasmmcp-time:latest
```

### Supported Platforms

- Knative + containerd (runwasi)
- Spin / Fermyon Cloud
- Cloudflare Workers
- AWS Lambda (custom runtime)

## Testing

```bash
# Run comparison test (vs reference implementations)
python tests/compare_time.py
python tests/compare_filesystem.py
python tests/compare_git.py

# Build and test all
./scripts/build_all.sh
for test in tests/compare_*.py; do python "$test"; done
```

## Binary Sizes

| Server | CLI | HTTP | Difference |
|--------|-----|------|------------|
| sequential-thinking | 264KB | 276KB | +12KB |
| summarize | 292KB | 300KB | +8KB |
| data-aggregate | 376KB | 388KB | +12KB |
| git | 388KB | 404KB | +16KB |
| filesystem | 436KB | 448KB | +12KB |
| fetch | 528KB | 540KB | +12KB |
| time | 1.2MB | 1.2MB | ~0 |
| log-parser | 1.3MB | 1.3MB | ~0 |
| image-resize | 1.2MB | 1.2MB | ~0 |

## Dependencies

```toml
[workspace.dependencies]
tokio = { version = "1", default-features = false }
rmcp = { version = "0.10", default-features = false, features = ["server", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
schemars = "1"
wasi = "0.14"
```

## License

MIT
