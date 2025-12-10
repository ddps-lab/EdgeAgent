# EdgeAgent: LangChain MCP Scheduling Middleware

LangChain에서 MCP Tool 호출 시 동적으로 실행 위치(DEVICE/EDGE/CLOUD)를 선택하는 스케줄링 middleware.

## 주요 기능

- **ProxyTool Pattern**: LLM은 tool만 보고, Scheduler가 실행 위치를 결정
- **Constraint-based Routing**: `requires_cloud_api`, `privacy_sensitive` 등 제약 기반 자동 라우팅
- **Args-based Dynamic Routing**: path, key 패턴으로 동적 위치 결정
- **LangChain Integration**: LangChain/LangGraph agent와 seamless 통합
- **Sub-Agent Orchestration**: Location별 Sub-Agent로 분산 실행 지원

## 프로젝트 구조

```
edgeagent/
├── edgeagent/              # Core package
│   ├── types.py            # Location, Runtime types
│   ├── profiles.py         # 4D Tool Profile
│   ├── registry.py         # Multi-endpoint registry
│   ├── scheduler.py        # BaseScheduler, StaticScheduler
│   ├── proxy_tool.py       # LocationAwareProxyTool
│   ├── middleware.py       # EdgeAgentMCPClient
│   ├── planner.py          # Tool sequence planning by location
│   ├── subagent.py         # Sub-Agent HTTP server
│   └── orchestrator.py     # Sub-Agent orchestration
├── config/
│   ├── tools.yaml          # Tool configurations
│   └── subagent_test.yaml  # Sub-Agent test configuration
├── scripts/
│   └── mock_mcp_server.py  # FastMCP mock servers
├── examples/
│   ├── 01_multi_location_routing.py
│   ├── 02_routing_simple.py
│   ├── 03_proxy_tool_structure.py
│   └── 04_constraint_routing.py
└── tests/
```

## 설치

### Python Dependencies

```bash
pip install -r requirements.txt
```

### MCP Servers (Global Installation Required)

MCP 서버들은 **반드시 전역으로 설치**해야 합니다:

```bash
# MCP filesystem server (필수)
npm install -g @modelcontextprotocol/server-filesystem

# Node.js 18+ 필요
node --version  # v18.0.0 이상 확인
```

**참고**: MCP 서버는 stdio transport로 subprocess로 실행되므로 전역 설치가 필요합니다.

## 환경 변수

`.env` 파일 생성:

```
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### 기본 예제

```bash
# Multi-location routing 테스트
python examples/01_multi_location_routing.py

# Constraint routing 테스트 (slack, credentials, compute)
python examples/04_constraint_routing.py
```

### 코드 예시

```python
from edgeagent import EdgeAgentMCPClient

async with EdgeAgentMCPClient("config/tools.yaml") as client:
    tools = await client.get_tools()

    # LLM은 "read_file"만 봄 (location suffix 없음)
    # Scheduler가 args 기반으로 DEVICE/EDGE/CLOUD 결정
    read_file = next(t for t in tools if t.name == "read_file")

    # path에 따라 자동 routing
    result = await read_file.ainvoke({"path": "/tmp/edgeagent_device/test.txt"})
```

## Constraint-based Routing

| Constraint | 동작 |
|------------|------|
| `requires_cloud_api: true` | 무조건 CLOUD |
| `privacy_sensitive: true` | CLOUD 제외 |
| `requires_gpu: true` | EDGE 또는 CLOUD |

## 테스트

```bash
pytest tests/ -v
```
