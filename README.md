# EdgeAgent: LangChain MCP Scheduling Middleware

LangChain에서 MCP Tool 호출 시 동적으로 실행 위치(DEVICE/EDGE/CLOUD)를 선택할 수 있는 middleware 구현.

## 목표

Model Context Protocol (MCP) tools를 Edge-Cloud continuum에서 최적 위치에 배치하여 실행하는 스케줄링 middleware 개발.

## 주요 기능

- **Multi-endpoint Registry**: 동일 tool을 여러 위치(DEVICE/EDGE/CLOUD)에 배포 및 관리
- **Static Scheduler**: Tool-location 매핑 기반 스케줄링
- **LangChain Integration**: LangChain/LangGraph agent와 seamless 통합
- **Location-aware Routing**: Tool 실행 시 적절한 endpoint로 자동 라우팅

## 프로젝트 구조

```
edgeagent/
├── edgeagent/           # 메인 패키지
│   ├── types.py         # Type definitions
│   ├── profiles.py      # 4D Tool Profile
│   ├── registry.py      # Multi-endpoint registry
│   ├── scheduler.py     # Static scheduler
│   └── middleware.py    # EdgeAgentMCPClient
├── config/              # 설정 파일
│   └── tools.yaml
├── examples/            # 예제 및 검증 코드
│   ├── 00_verify_langchain_basic.py
│   ├── 01_verify_mcp_adapter.py
│   ├── 02_verify_multi_location.py
│   └── 03_middleware_routing.py
└── tests/               # Unit tests
```

## 설치

```bash
# Python dependencies
pip install -r requirements.txt

# MCP filesystem server (for testing)
npm install -g @modelcontextprotocol/server-filesystem
```

## 환경 변수

`.env` 파일을 생성하고 OpenAI API key를 설정:

```
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### Phase 0: 기초 검증

```bash
# 1. 순수 LangChain agent 테스트
python examples/00_verify_langchain_basic.py

# 2. MCP adapter 통합 테스트
python examples/01_verify_mcp_adapter.py

# 3. Multi-location 시뮬레이션
python examples/02_verify_multi_location.py
```

### Phase 4: Middleware 사용

```bash
# Middleware를 통한 tool routing 테스트
python examples/03_middleware_routing.py
```

## 개발 로드맵

- [x] Phase 0.2: 기초 LangChain agent 검증 ✅
- [x] Phase 0.3: MCP adapter 통합 ✅
- [~] Phase 0.4: Multi-location 시뮬레이션 (세션 관리 이슈로 스킵)
- [x] Phase 1: 타입 및 Profile 정의 ✅
- [x] Phase 2: Registry & Static Scheduler ✅
- [x] Phase 3: Middleware 구현 ✅
- [ ] Phase 4: End-to-end 테스트 🎯 **← 현재 단계**

## 관련 연구

이 프로젝트는 다음 연구의 구현 프로토타입입니다:
- **EdgeAgent Research Plan v2.1**: Locality-Aware Serverless Execution of MCP Tools in the Edge-Cloud Continuum
- Target: IEEE/ACM CCGrid 2026

## License

MIT
