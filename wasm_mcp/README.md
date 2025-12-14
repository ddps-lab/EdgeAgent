# WasmMCP - Rust MCP 서버를 WASM으로

WASM 바이너리로 컴파일되는 MCP(Model Context Protocol) 서버 프레임워크.
Stdio와 HTTP 두 가지 전송 방식을 지원하여 로컬 개발과 서버리스 배포 모두 가능.

## 개요

이 프로젝트는 Rust로 MCP 서버를 작성하고, WASM(WebAssembly)으로 컴파일하여
다양한 환경에서 실행할 수 있게 해주는 프레임워크입니다.

### 주요 특징

- **순수 WASM**: Python/Node.js 없이 WASM 바이너리만으로 실행
- **두 가지 전송 방식**:
  - Stdio: 로컬 개발용 (`wasmtime run`)
  - HTTP: 서버리스 배포용 (`wasmtime serve`, Knative)
- **경량 바이너리**: Stdio 792KB, HTTP 287KB
- **MCP 프로토콜 완전 지원**: 14개 파일시스템 도구 제공
- **LangChain 호환**: `langchain-mcp-adapters`와 통합 가능

## 프로젝트 구조

```
wasm_mcp/
├── Cargo.toml                      # 워크스페이스 설정
├── servers/
│   ├── filesystem/                 # Stdio 버전 (wasi:cli/run)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # WASI CLI 진입점
│   │       └── service.rs          # 14개 MCP 도구 구현
│   │
│   └── filesystem-http/            # HTTP 버전 (wasi:http/proxy)
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs              # HTTP 핸들러 + MCP 도구
│
├── wasmmcp/                        # 프레임워크 코어
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                  # 공개 API
│       ├── server.rs               # WasmMcp 구조체
│       ├── transport/
│       │   ├── mod.rs              # Transport 트레이트
│       │   ├── stdio.rs            # WASI stdio 전송
│       │   └── http.rs             # WASI HTTP 전송
│       └── protocol/
│           ├── mod.rs
│           └── jsonrpc.rs          # JSON-RPC 2.0 처리
│
├── wasmmcp-macros/                 # 프로시저 매크로
│   └── src/lib.rs
│
├── shared/                         # 공유 유틸리티
│
└── tests/                          # 테스트 스크립트
    ├── test_langchain_integration.py   # Stdio 테스트
    └── test_http_langchain.py          # HTTP 테스트
```

## 빌드 방법

### 사전 요구사항

```bash
# Rust WASI 타겟 설치
rustup target add wasm32-wasip2

# Wasmtime 설치
curl https://wasmtime.dev/install.sh -sSf | bash
```

### Stdio 버전 빌드

```bash
cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem
```

빌드 결과: `target/wasm32-wasip2/release/mcp_server_filesystem.wasm` (792KB)

### HTTP 버전 빌드

```bash
cargo build --target wasm32-wasip2 --release -p mcp-server-filesystem-http
```

빌드 결과: `target/wasm32-wasip2/release/mcp_server_filesystem_http.wasm` (287KB)

## 실행 방법

### Stdio 모드 (로컬 개발)

```bash
# 직접 실행
wasmtime run --dir=/tmp ./target/wasm32-wasip2/release/mcp_server_filesystem.wasm

# JSON-RPC 테스트
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
  wasmtime run --dir=/tmp ./target/wasm32-wasip2/release/mcp_server_filesystem.wasm
```

### HTTP 모드 (서버리스)

```bash
# HTTP 서버 시작
wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir=/tmp \
  ./target/wasm32-wasip2/release/mcp_server_filesystem_http.wasm

# curl로 테스트
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

## LangChain 통합

### Stdio 연결

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_config = {
    "wasmmcp": {
        "transport": "stdio",
        "command": "wasmtime",
        "args": ["run", "--dir=/tmp", "mcp_server_filesystem.wasm"],
    }
}

client = MultiServerMCPClient(mcp_config)
async with client.session("wasmmcp") as session:
    tools = await load_mcp_tools(session)
    # 14개 도구 사용 가능
```

### HTTP 연결 (Streamable HTTP)

```python
mcp_config = {
    "wasmmcp_http": {
        "transport": "streamable_http",
        "url": "http://localhost:8000",
    }
}

client = MultiServerMCPClient(mcp_config)
async with client.session("wasmmcp_http") as session:
    tools = await load_mcp_tools(session)
    # 원격 MCP 서버 도구 사용
```

## 제공 도구 (14개)

| 도구명 | 설명 |
|--------|------|
| `read_file` | 파일 읽기 (deprecated) |
| `read_text_file` | 텍스트 파일 읽기 (head/tail 지원) |
| `read_media_file` | 미디어 파일을 Base64로 읽기 |
| `read_multiple_files` | 여러 파일 동시 읽기 |
| `write_file` | 파일 쓰기 |
| `edit_file` | 파일 편집 (search/replace) |
| `create_directory` | 디렉토리 생성 |
| `list_directory` | 디렉토리 목록 |
| `list_directory_with_sizes` | 크기 포함 디렉토리 목록 |
| `directory_tree` | 디렉토리 트리 (JSON) |
| `move_file` | 파일 이동/이름변경 |
| `search_files` | 파일 검색 (glob 패턴) |
| `get_file_info` | 파일 메타데이터 |
| `list_allowed_directories` | 허용된 디렉토리 목록 |

## 아키텍처

### 전송 계층 분리

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP 서버 로직 (도구들)                      │
├─────────────────────────────────────────────────────────────┤
│                    wasmmcp 프레임워크                         │
├──────────────────────────┬──────────────────────────────────┤
│   Stdio Transport        │   HTTP Transport                 │
│   (wasi:cli/run)         │   (wasi:http/proxy)              │
├──────────────────────────┼──────────────────────────────────┤
│   wasmtime run           │   wasmtime serve / Knative       │
└──────────────────────────┴──────────────────────────────────┘
```

### 두 가지 WASM 월드

1. **wasi:cli/run** (Stdio)
   - `wasi::cli::command::export!` 매크로 사용
   - stdin/stdout으로 JSON-RPC 통신
   - 로컬 MCP 클라이언트와 직접 연결

2. **wasi:http/proxy** (HTTP)
   - `wasi::http::proxy::export!` 매크로 사용
   - HTTP POST로 JSON-RPC 수신
   - 서버리스 환경에서 원격 접속 가능

## 서버리스 배포

### Knative 배포 예시

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: wasmmcp-filesystem
spec:
  template:
    spec:
      containers:
        - image: ghcr.io/example/wasmmcp-filesystem:latest
          # WASM 런타임이 포함된 이미지
```

### 지원 가능한 플랫폼

- Knative + containerd (runwasi)
- Cloudflare Workers (wrangler)
- AWS Lambda (custom runtime)
- Spin / Fermyon Cloud

## 테스트

```bash
# Stdio 테스트
python tests/test_langchain_integration.py

# HTTP 테스트 (서버 먼저 실행 필요)
wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir=/tmp \
  ./target/wasm32-wasip2/release/mcp_server_filesystem_http.wasm &

python tests/test_http_langchain.py
```

## 의존성

```toml
[workspace.dependencies]
tokio = { version = "1", default-features = false, features = ["rt", "io-util", "sync", "macros", "time"] }
rmcp = { version = "0.10", default-features = false, features = ["server", "macros", "base64"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
wasi = "0.14"
```

## 향후 계획

- [ ] `#[mcp_tool]` 매크로로 도구 정의 단순화
- [ ] SSE(Server-Sent Events) 지원
- [ ] 더 많은 MCP 서버 예제 (git, sqlite 등)
- [ ] Spin/Fermyon 배포 가이드
