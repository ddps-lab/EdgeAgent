"""
Real MCP Server Integration Tests

Tests for independent, primitive MCP tools.
Each tool is self-contained - tests verify individual tool behavior.

Design:
- log_parser: parse_logs → filter_entries → compute_log_statistics (LLM orchestrates)
- image_resize: scan_directory → compute_image_hash → compare_hashes (LLM orchestrates)

Usage:
    pytest tests/test_scenarios/test_real_integration.py -v -s
"""

import json
import os
from pathlib import Path

import pytest

# Skip if mcp not available
pytest.importorskip("mcp")


class TestLogParserServer:
    """Test log_parser MCP server - independent tools"""

    @pytest.fixture
    def sample_apache_log(self) -> str:
        """Sample Apache Combined log"""
        return """192.168.1.1 - - [09/Dec/2024:10:15:23 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
192.168.1.2 - - [09/Dec/2024:10:15:24 +0000] "POST /api/login HTTP/1.1" 401 89 "-" "Mozilla/5.0"
192.168.1.1 - - [09/Dec/2024:10:15:25 +0000] "GET /api/products HTTP/1.1" 500 256 "-" "Mozilla/5.0"
192.168.1.3 - - [09/Dec/2024:10:15:26 +0000] "GET /static/style.css HTTP/1.1" 200 4567 "-" "Mozilla/5.0"
192.168.1.2 - - [09/Dec/2024:10:15:27 +0000] "POST /api/login HTTP/1.1" 200 512 "-" "Mozilla/5.0"
10.0.0.1 - - [09/Dec/2024:10:15:28 +0000] "GET /api/admin HTTP/1.1" 403 128 "-" "curl/7.68.0"
192.168.1.1 - - [09/Dec/2024:10:15:29 +0000] "DELETE /api/users/123 HTTP/1.1" 500 89 "-" "Mozilla/5.0"
"""

    @pytest.fixture
    def sample_python_log(self) -> str:
        """Sample Python log"""
        return """2024-12-09 10:15:23,123 - api.server - ERROR - Database connection failed: timeout after 30s
2024-12-09 10:15:24,456 - api.server - INFO - Retrying database connection...
2024-12-09 10:15:25,789 - api.server - ERROR - Database connection failed again
2024-12-09 10:15:26,012 - auth.handler - WARNING - Too many failed login attempts from 192.168.1.100
2024-12-09 10:15:27,345 - system.monitor - CRITICAL - Memory usage exceeded 95%
2024-12-09 10:15:28,678 - payment.processor - ERROR - Transaction failed: insufficient funds
2024-12-09 10:15:29,901 - api.server - INFO - Database connection restored
2024-12-09 10:15:30,234 - api.handler - ERROR - Request timeout for /api/v1/users after 60s
"""

    def test_parse_logs(self, sample_apache_log):
        """Test parse_logs - the first step"""
        from servers.log_parser_server import parse_logs

        result = parse_logs.fn(sample_apache_log, "auto")

        assert result["format_detected"] == "apache_combined"
        assert result["parsed_count"] == 7
        assert result["error_count"] == 0
        assert len(result["entries"]) == 7
        assert result["entries"][0]["ip"] == "192.168.1.1"
        assert "_level" in result["entries"][0]

    def test_parse_python_log(self, sample_python_log):
        """Test parsing Python log format"""
        from servers.log_parser_server import parse_logs

        result = parse_logs.fn(sample_python_log, "auto")

        assert result["format_detected"] == "python"
        assert result["parsed_count"] == 8
        # Each entry should have _level extracted
        for entry in result["entries"]:
            assert "_level" in entry

    def test_filter_entries(self, sample_python_log):
        """Test filter_entries - takes parsed entries as input"""
        from servers.log_parser_server import parse_logs, filter_entries

        # Step 1: Parse (LLM would do this)
        parsed = parse_logs.fn(sample_python_log, "auto")

        # Step 2: Filter (LLM passes parsed entries)
        filtered = filter_entries.fn(parsed["entries"], min_level="error")

        assert filtered["original_count"] == 8
        assert filtered["filtered_count"] < 8
        # Only error and critical should be included
        for entry in filtered["entries"]:
            assert entry["_level"] in ["error", "critical"]

    def test_compute_log_statistics(self, sample_apache_log):
        """Test compute_log_statistics - takes entries as input"""
        from servers.log_parser_server import parse_logs, compute_log_statistics

        # Step 1: Parse
        parsed = parse_logs.fn(sample_apache_log, "auto")

        # Step 2: Compute stats
        stats = compute_log_statistics.fn(parsed["entries"])

        assert stats["entry_count"] == 7
        assert "by_level" in stats
        assert "by_status" in stats
        assert "top_ips" in stats

    def test_search_entries(self, sample_python_log):
        """Test search_entries - find specific patterns"""
        from servers.log_parser_server import parse_logs, search_entries

        # Step 1: Parse
        parsed = parse_logs.fn(sample_python_log, "auto")

        # Step 2: Search for "Database"
        search_result = search_entries.fn(parsed["entries"], pattern="Database")

        assert search_result["match_count"] > 0
        assert search_result["total_entries"] == 8

    def test_full_workflow(self, sample_python_log):
        """Test full LLM-orchestrated workflow: parse → filter → stats"""
        from servers.log_parser_server import (
            parse_logs,
            filter_entries,
            compute_log_statistics,
        )

        # LLM Step 1: Parse logs
        parsed = parse_logs.fn(sample_python_log, "auto")
        assert parsed["parsed_count"] == 8

        # LLM Step 2: Filter to errors only
        filtered = filter_entries.fn(parsed["entries"], min_level="error")
        assert filtered["filtered_count"] < parsed["parsed_count"]

        # LLM Step 3: Compute statistics
        stats = compute_log_statistics.fn(filtered["entries"])
        assert stats["entry_count"] == filtered["filtered_count"]

        print(f"\nWorkflow: {parsed['parsed_count']} → {filtered['filtered_count']} entries")
        print(f"Levels: {stats['by_level']}")


class TestDataAggregateServer:
    """Test data_aggregate MCP server"""

    @pytest.fixture
    def sample_items(self) -> list[dict]:
        """Sample data items"""
        return [
            {"category": "A", "value": 100, "status": "active"},
            {"category": "A", "value": 150, "status": "active"},
            {"category": "B", "value": 200, "status": "inactive"},
            {"category": "B", "value": 250, "status": "active"},
            {"category": "C", "value": 300, "status": "active"},
        ]

    def test_aggregate_list(self, sample_items):
        """Test list aggregation"""
        from servers.data_aggregate_server import aggregate_list

        result = aggregate_list.fn(
            sample_items,
            group_by="category",
            count_field="status",
            sum_fields=["value"],
        )

        assert result["total_count"] == 5
        assert result["group_count"] == 3
        assert "A" in result["groups"]
        assert result["groups"]["A"] == 2

    def test_merge_summaries(self):
        """Test merging summaries"""
        from servers.data_aggregate_server import merge_summaries

        summaries = [
            {"count": 10, "total": 100},
            {"count": 20, "total": 200},
            {"count": 30, "total": 300},
        ]

        result = merge_summaries.fn(summaries)

        assert result["merged_count"] == 3
        assert result["count_total"] == 60
        assert result["total_total"] == 600

    def test_deduplicate(self):
        """Test deduplication"""
        from servers.data_aggregate_server import deduplicate

        items = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
            {"id": "1", "name": "Alice Updated"},
            {"id": "3", "name": "Charlie"},
        ]

        result = deduplicate.fn(items, key_fields=["id"], keep="first")

        assert result["original_count"] == 4
        assert result["unique_count"] == 3


class TestImageResizeServer:
    """Test image_resize MCP server - independent tools"""

    @pytest.fixture
    def sample_image_path(self, tmp_path) -> Path:
        """Create a sample test image"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (800, 600), color="red")
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path, "JPEG")
        return img_path

    @pytest.fixture
    def sample_images_dir(self, tmp_path) -> Path:
        """Create directory with multiple test images"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        for i in range(5):
            img = Image.new("RGB", (200, 200), color=(i * 50, 100, 150))
            img.save(tmp_path / f"image_{i}.jpg", "JPEG")

        # Create duplicates
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        img.save(tmp_path / "dup1.jpg", "JPEG")
        img.save(tmp_path / "dup2.jpg", "JPEG")

        return tmp_path

    def test_get_image_info(self, sample_image_path):
        """Test get_image_info - first step for single image"""
        from servers.image_resize_server import get_image_info

        result = get_image_info.fn(str(sample_image_path))

        assert result["width"] == 800
        assert result["height"] == 600
        assert result["format"] == "JPEG"
        assert result["size_bytes"] > 0

    def test_resize_image(self, sample_image_path):
        """Test resize_image - independent resize"""
        from servers.image_resize_server import resize_image

        result = resize_image.fn(
            str(sample_image_path),
            width=400,
            height=300,
            quality=80,
        )

        assert result["success"] is True
        assert result["new_size"] == (400, 300)
        assert result["output_bytes"] < result["original_bytes"]
        assert "data_base64" in result

    def test_scan_directory(self, sample_images_dir):
        """Test scan_directory - first step for batch ops"""
        from servers.image_resize_server import scan_directory

        result = scan_directory.fn(str(sample_images_dir))

        assert result["image_count"] == 7  # 5 + 2 duplicates
        assert len(result["image_paths"]) == 7
        assert result["total_size_bytes"] > 0

    def test_compute_and_compare_hashes(self, sample_images_dir):
        """Test hash computation and comparison workflow"""
        from servers.image_resize_server import (
            scan_directory,
            compute_image_hash,
            compare_hashes,
        )

        # Step 1: Scan directory
        scan_result = scan_directory.fn(str(sample_images_dir))
        paths = scan_result["image_paths"]

        # Step 2: Compute hashes for each image
        hashes = []
        for path in paths:
            hash_result = compute_image_hash.fn(path)
            hashes.append(hash_result)

        assert len(hashes) == 7

        # Step 3: Compare hashes to find duplicates
        comparison = compare_hashes.fn(hashes, threshold=5)

        assert comparison["total_compared"] == 7
        assert comparison["duplicate_group_count"] >= 1  # dup1 and dup2

    def test_batch_resize(self, sample_images_dir):
        """Test batch_resize for thumbnails"""
        from servers.image_resize_server import scan_directory, batch_resize

        # Step 1: Scan
        scan_result = scan_directory.fn(str(sample_images_dir))

        # Step 2: Batch resize
        resize_result = batch_resize.fn(
            scan_result["image_paths"][:3],
            max_size=50,
        )

        assert resize_result["successful"] == 3
        assert resize_result["overall_reduction"] < 1.0  # Size reduced


class TestSummarizeServer:
    """Test summarize MCP server"""

    @pytest.fixture
    def long_text(self) -> str:
        """Sample long text for summarization"""
        return """
        Artificial Intelligence (AI) has rapidly evolved over the past decade,
        transforming various industries and aspects of daily life. Machine learning,
        a subset of AI, enables systems to learn and improve from experience without
        being explicitly programmed.
        """

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_summarize_text(self, long_text):
        """Test text summarization"""
        os.environ["SUMMARIZE_PROVIDER"] = "openai"
        from servers.summarize_server import summarize_text

        result = summarize_text.fn(long_text, max_length=50, style="concise")

        assert result is not None
        assert len(result) > 0
        assert len(result) < len(long_text)

    def test_get_provider_info(self):
        """Test getting provider info"""
        from servers.summarize_server import get_provider_info

        result = get_provider_info.fn()

        assert "provider" in result
        assert "available_styles" in result


class TestEndToEndWorkflows:
    """Test LLM-orchestrated workflows with independent tools"""

    @pytest.fixture
    def log_file(self, tmp_path) -> Path:
        """Create test log file"""
        log_content = """2024-12-09 10:00:00,000 - app - ERROR - Connection failed
2024-12-09 10:00:01,000 - app - WARNING - Retry attempt 1
2024-12-09 10:00:02,000 - app - ERROR - Connection failed again
2024-12-09 10:00:03,000 - app - INFO - Connection restored
2024-12-09 10:00:04,000 - app - CRITICAL - System overload detected
"""
        log_file = tmp_path / "test.log"
        log_file.write_text(log_content)
        return log_file

    @pytest.fixture
    def image_dir(self, tmp_path) -> Path:
        """Create test images"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        for i in range(5):
            img = Image.new("RGB", (200, 200), color=(i * 50, 100, 150))
            img.save(tmp_path / f"image_{i}.jpg", "JPEG")
        return tmp_path

    def test_log_analysis_workflow(self, log_file):
        """
        Simulated LLM Workflow:
        1. Read log file (filesystem tool)
        2. parse_logs() → entries
        3. filter_entries(entries, min_level="error") → errors
        4. compute_log_statistics(errors) → stats
        5. aggregate() → final summary
        """
        from servers.log_parser_server import (
            parse_logs,
            filter_entries,
            compute_log_statistics,
        )
        from servers.data_aggregate_server import aggregate_list

        # 1. Read log (would be filesystem MCP)
        log_content = log_file.read_text()

        # 2. Parse logs
        parsed = parse_logs.fn(log_content, "auto")
        print(f"\n[Step 2] Parsed {parsed['parsed_count']} entries")

        # 3. Filter errors
        errors = filter_entries.fn(parsed["entries"], min_level="warning")
        print(f"[Step 3] Filtered to {errors['filtered_count']} warnings+errors")

        # 4. Compute statistics
        stats = compute_log_statistics.fn(errors["entries"])
        print(f"[Step 4] Stats: {stats['by_level']}")

        # 5. Aggregate
        aggregated = aggregate_list.fn(errors["entries"], group_by="_level")
        print(f"[Step 5] Aggregated {aggregated['group_count']} groups")

        # Verify data reduction
        input_size = len(log_content)
        output_size = len(json.dumps(aggregated))
        print(f"\nData reduction: {input_size} → {output_size} bytes")

        assert aggregated["total_count"] > 0

    def test_image_processing_workflow(self, image_dir):
        """
        Simulated LLM Workflow:
        1. scan_directory() → image_paths
        2. For each path: compute_image_hash() → hashes
        3. compare_hashes(hashes) → duplicates
        4. batch_resize(unique_paths) → thumbnails
        """
        from servers.image_resize_server import (
            scan_directory,
            compute_image_hash,
            compare_hashes,
            batch_resize,
        )

        # 1. Scan directory
        scan = scan_directory.fn(str(image_dir))
        print(f"\n[Step 1] Found {scan['image_count']} images")

        # 2. Compute hashes
        hashes = []
        for path in scan["image_paths"]:
            h = compute_image_hash.fn(path)
            hashes.append(h)
        print(f"[Step 2] Computed {len(hashes)} hashes")

        # 3. Find duplicates
        comparison = compare_hashes.fn(hashes)
        print(f"[Step 3] Found {comparison['duplicate_group_count']} duplicate groups")

        # 4. Create thumbnails for unique images (or first 3 if all are duplicates)
        paths_to_resize = comparison["unique_paths"][:3] if comparison["unique_paths"] else scan["image_paths"][:3]
        thumbnails = batch_resize.fn(paths_to_resize, max_size=50)
        print(f"[Step 4] Created {thumbnails['successful']} thumbnails")

        assert thumbnails["successful"] > 0


class TestMCPProtocolIntegration:
    """Test actual MCP protocol communication"""

    @pytest.mark.asyncio
    async def test_log_parser_via_mcp(self):
        """Test log_parser server via MCP stdio protocol"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command="python",
            args=["servers/log_parser_server.py"],
            env={"MCP_TRANSPORT": "stdio"},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]

                # Verify independent tools exist
                assert "parse_logs" in tool_names
                assert "filter_entries" in tool_names
                assert "compute_log_statistics" in tool_names
                assert "search_entries" in tool_names

                # Test parse_logs
                result = await session.call_tool(
                    "parse_logs",
                    arguments={
                        "log_content": "2024-12-09 10:00:00,000 - app - ERROR - Test",
                        "format_type": "auto",
                    },
                )

                result_data = json.loads(result.content[0].text)
                assert result_data["parsed_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
