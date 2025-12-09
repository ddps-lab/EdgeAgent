#!/usr/bin/env python3
"""
Generate Test Log Data for Scenario 2

Creates log files of various sizes and formats for testing:
- Small (100 lines): Quick tests
- Medium (1,000 lines): Normal experiments
- Large (10,000 lines): Scale tests
- XLarge (100,000 lines): Stress tests

Log formats supported:
- Python logging format
- Apache Combined format
- Syslog format
"""

import random
from datetime import datetime, timedelta
from pathlib import Path


# Sample data for log generation
LOGGERS = [
    "app.auth", "app.api", "app.db", "app.cache",
    "worker.task", "worker.queue", "scheduler",
    "middleware.request", "middleware.response"
]

LEVELS = {
    "debug": 0.30,    # 30%
    "info": 0.40,     # 40%
    "warning": 0.15,  # 15%
    "error": 0.12,    # 12%
    "critical": 0.03, # 3%
}

MESSAGES = {
    "debug": [
        "Processing request with params: {}",
        "Cache lookup for key: user_{}",
        "Database query executed in {} ms",
        "Memory usage: {} MB",
        "Connection pool status: {} active",
    ],
    "info": [
        "Request processed successfully",
        "User {} logged in",
        "Task {} completed",
        "Cache hit rate: {}%",
        "API response time: {} ms",
        "Session created for user {}",
        "Background job started: {}",
    ],
    "warning": [
        "Slow query detected: {} ms",
        "Cache miss for frequently accessed key",
        "Rate limit approaching for user {}",
        "Memory usage above threshold: {}%",
        "Connection pool running low",
        "Retry attempt {} for operation",
    ],
    "error": [
        "Database connection failed: timeout",
        "Authentication failed for user {}",
        "API request failed with status {}",
        "Task execution failed: {}",
        "Cache write error",
        "Invalid request payload",
    ],
    "critical": [
        "Database connection pool exhausted",
        "Out of memory error",
        "Service unavailable: {}",
        "Unhandled exception in main loop",
        "Critical security violation detected",
    ],
}


def weighted_choice(weights: dict) -> str:
    """Choose a key based on weights"""
    total = sum(weights.values())
    r = random.uniform(0, total)
    cumulative = 0
    for key, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return key
    return list(weights.keys())[-1]


def generate_python_log_line(timestamp: datetime) -> str:
    """Generate a Python logging format line"""
    level = weighted_choice(LEVELS)
    logger = random.choice(LOGGERS)
    message_template = random.choice(MESSAGES[level])

    # Fill in placeholders with random values
    placeholders = message_template.count("{}")
    values = []
    for _ in range(placeholders):
        values.append(random.choice([
            str(random.randint(1, 1000)),
            f"user_{random.randint(1, 100)}",
            f"task_{random.randint(1, 50)}",
            str(random.randint(10, 500)),
        ]))

    message = message_template.format(*values) if values else message_template

    return f"{timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {logger} - {level.upper()} - {message}"


def generate_apache_log_line(timestamp: datetime) -> str:
    """Generate an Apache Combined Log format line"""
    ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
    methods = ["GET", "POST", "PUT", "DELETE"]
    paths = [
        "/api/users", "/api/products", "/api/orders",
        "/static/js/app.js", "/static/css/style.css",
        "/health", "/metrics", "/login", "/logout"
    ]
    statuses = [200, 200, 200, 200, 201, 204, 301, 400, 401, 403, 404, 500, 502, 503]
    status_weights = {
        200: 0.70, 201: 0.05, 204: 0.05, 301: 0.02,
        400: 0.05, 401: 0.03, 403: 0.02, 404: 0.04,
        500: 0.02, 502: 0.01, 503: 0.01
    }

    method = random.choice(methods)
    path = random.choice(paths)
    status = weighted_choice(status_weights)
    size = random.randint(100, 50000)
    referrer = random.choice(["-", "https://example.com", "https://google.com"])
    agent = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36",
        "curl/7.88.1",
        "Python-urllib/3.10",
    ])

    time_str = timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{ip} - - [{time_str}] "{method} {path} HTTP/1.1" {status} {size} "{referrer}" "{agent}"'


def generate_syslog_line(timestamp: datetime) -> str:
    """Generate a syslog format line"""
    hosts = ["web-01", "web-02", "db-01", "cache-01", "worker-01"]
    processes = ["sshd", "nginx", "postgresql", "redis", "systemd"]
    level = weighted_choice(LEVELS)

    host = random.choice(hosts)
    process = random.choice(processes)
    pid = random.randint(1000, 65535)
    message_template = random.choice(MESSAGES[level])

    # Fill placeholders
    placeholders = message_template.count("{}")
    values = [str(random.randint(1, 1000)) for _ in range(placeholders)]
    message = message_template.format(*values) if values else message_template

    time_str = timestamp.strftime("%b %d %H:%M:%S")
    return f"{time_str} {host} {process}[{pid}]: {message}"


def generate_log_file(
    output_path: Path,
    num_lines: int,
    format_type: str = "python",
    start_time: datetime = None,
) -> dict:
    """Generate a log file with specified number of lines"""

    if start_time is None:
        start_time = datetime.now() - timedelta(hours=1)

    generators = {
        "python": generate_python_log_line,
        "apache": generate_apache_log_line,
        "syslog": generate_syslog_line,
    }

    generator = generators.get(format_type, generate_python_log_line)

    lines = []
    current_time = start_time
    level_counts = {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}

    for i in range(num_lines):
        line = generator(current_time)
        lines.append(line)

        # Count levels (approximate for Python format)
        for level in level_counts:
            if level.upper() in line:
                level_counts[level] += 1
                break

        # Advance time (random 1-10 seconds)
        current_time += timedelta(seconds=random.uniform(0.1, 5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    output_path.write_text(content)

    return {
        "path": str(output_path),
        "lines": num_lines,
        "format": format_type,
        "size_bytes": len(content),
        "level_counts": level_counts,
    }


def main():
    """Generate test log files"""
    base_dir = Path(__file__).parent.parent / "data" / "scenario2" / "loghub_samples"

    print("Generating test log files...")
    print()

    configs = [
        # (name, lines, format)
        ("small_python.log", 100, "python"),
        ("medium_python.log", 1000, "python"),
        ("large_python.log", 10000, "python"),
        ("xlarge_python.log", 100000, "python"),

        ("small_apache.log", 100, "apache"),
        ("medium_apache.log", 1000, "apache"),

        ("small_syslog.log", 100, "syslog"),
        ("medium_syslog.log", 1000, "syslog"),
    ]

    results = []
    for name, lines, fmt in configs:
        output_path = base_dir / name
        result = generate_log_file(output_path, lines, fmt)
        results.append(result)
        print(f"  {name}: {lines:,} lines, {result['size_bytes']:,} bytes")

    # Create symlinks for default files
    default_log = base_dir.parent / "server.log"
    if not default_log.exists() or default_log.is_symlink():
        if default_log.exists():
            default_log.unlink()
        # Copy medium python log as default
        medium_log = base_dir / "medium_python.log"
        default_log.write_text(medium_log.read_text())
        print(f"\n  Default log updated: {default_log}")

    print()
    print(f"Generated {len(results)} log files in {base_dir}")
    print()

    # Summary table
    print("Summary:")
    print("-" * 60)
    print(f"{'File':<25} {'Lines':>10} {'Size':>15} {'Format':<10}")
    print("-" * 60)
    for r in results:
        name = Path(r["path"]).name
        print(f"{name:<25} {r['lines']:>10,} {r['size_bytes']:>12,} B {r['format']:<10}")
    print("-" * 60)


if __name__ == "__main__":
    main()
