# Network I/O 분리 작업 요약

## 목표
Disk I/O와 Network I/O 타이밍을 분리하여 측정

## 수정된 파일들

### 핵심 라이브러리

#### 1. `wasmmcp/src/timing.rs`
- `DISK_IO_ACCUMULATOR`, `NETWORK_IO_ACCUMULATOR` 분리
- `measure_disk_io()`, `measure_network_io()` 함수 추가
- `set_tool_exec_ms()`, `set_disk_io_ms()`, `set_network_io_ms()` 추가
- `get_tool_exec_ms()`, `get_disk_io_ms()`, `get_network_io_ms()` 추가
- `ToolTiming` 구조체에 `disk_io_ms`, `network_io_ms` 필드 추가

#### 2. `wasmmcp/src/builder.rs`
- `handle_tools_call()`에서 tool_exec 시간 측정
- `set_tool_exec_ms()`, `set_disk_io_ms()`, `set_network_io_ms()` 호출
- stderr에 `---TOOL_EXEC---`, `---DISK_IO---`, `---NETWORK_IO---` 출력

#### 3. `wasmmcp-macros/src/lib.rs`
- HTTP 헤더 추가: `X-Tool-Exec-Ms`, `X-Disk-IO-Ms`, `X-Network-IO-Ms`

### 서버 tools.rs 수정

#### Disk I/O 서버들 (`measure_disk_io` 사용)
- `servers/filesystem/src/tools.rs`
- `servers/git/src/tools.rs`
- `servers/image-resize/src/tools.rs`

#### Network I/O 서버들 (`measure_network_io` 사용)
- `servers/summarize/src/tools.rs`
- `servers/fetch/src/tools.rs`

#### 순수 계산 서버들 (타이밍 출력 형식만 변경)
- `servers/log-parser/src/tools.rs`
- `servers/time/src/tools.rs`
- `servers/data-aggregate/src/tools.rs`
- `servers/sequential-thinking/src/tools.rs`

## 변경 내용 상세

### timing.rs 주요 변경
```rust
thread_local! {
    static DISK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    static NETWORK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    static TOOL_EXEC_MS: RefCell<f64> = RefCell::new(0.0);
    static DISK_IO_MS: RefCell<f64> = RefCell::new(0.0);
    static NETWORK_IO_MS: RefCell<f64> = RefCell::new(0.0);
}

pub fn measure_disk_io<F, T>(f: F) -> T { ... }
pub fn measure_network_io<F, T>(f: F) -> T { ... }

pub struct ToolTiming {
    pub tool_name: String,
    pub fn_total_ms: f64,
    pub disk_io_ms: f64,
    pub network_io_ms: f64,
    pub compute_ms: f64,
}
```

### builder.rs 주요 변경
```rust
pub fn handle_tools_call(&self, name: &str, args: Value) -> Result<Value, String> {
    reset_io_accumulators();

    let start = Instant::now();
    let result = self.registry.call(name, args)?;
    let tool_exec_ms = start.elapsed().as_secs_f64() * 1000.0;

    let disk_io_ms = get_disk_io_duration().as_secs_f64() * 1000.0;
    let network_io_ms = get_network_io_duration().as_secs_f64() * 1000.0;

    set_tool_exec_ms(tool_exec_ms);
    set_disk_io_ms(disk_io_ms);
    set_network_io_ms(network_io_ms);

    eprintln!("---TOOL_EXEC---{:.3}", tool_exec_ms);
    eprintln!("---DISK_IO---{:.3}", disk_io_ms);
    eprintln!("---NETWORK_IO---{:.3}", network_io_ms);
    ...
}
```

### tools.rs 타이밍 출력 형식
```rust
// 이전
"timing": {
    "wasm_total_ms": get_wasm_total_ms(),
    "fn_total_ms": timing.fn_total_ms,
    "io_ms": timing.io_ms,
    "compute_ms": timing.compute_ms
}

// 이후
"timing": {
    "wasm_total_ms": get_wasm_total_ms(),
    "fn_total_ms": timing.fn_total_ms,
    "disk_io_ms": timing.disk_io_ms,
    "network_io_ms": timing.network_io_ms,
    "compute_ms": timing.compute_ms
}
```

## 예상 결과
- `read_file` 같은 디스크 작업: `disk_io_ms`에 시간 기록, `network_io_ms = 0`
- `fetch` 같은 네트워크 작업: `network_io_ms`에 시간 기록, `disk_io_ms = 0`
- `compute_ms = fn_total_ms - disk_io_ms - network_io_ms`

## 빌드 에러 수정 완료
- `data-aggregate/src/tools.rs` 181번, 277번 줄 수정
- `fetch/src/tools.rs` 367번 줄 수정
