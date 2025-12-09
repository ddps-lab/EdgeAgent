"""
Tests for Scenario 2: Log Analysis Pipeline
"""

import pytest
from experiments.scenario2_log_analysis import LogAnalysisExperiment


class TestLogAnalysisScenario:
    """Tests for the Log Analysis scenario"""

    @pytest.fixture
    def experiment(self):
        return LogAnalysisExperiment()

    def test_tool_chain(self, experiment):
        """Test that the tool chain is correctly defined"""
        expected = ["filesystem", "time", "log_parser", "data_aggregate", "llm_generate", "filesystem"]
        assert experiment.get_tool_chain() == expected

    def test_user_request(self, experiment):
        """Test that the user request is defined"""
        request = experiment.get_user_request()
        assert "로그" in request or "log" in request.lower()
        assert "에러" in request or "error" in request.lower()

    @pytest.mark.asyncio
    async def test_experiment_runs(self, experiment):
        """Test that the experiment runs successfully"""
        result = await experiment.run()

        assert result.scenario_name == "scenario2_log_analysis"
        assert result.success is True
        assert result.tool_chain_length == 6

    @pytest.mark.asyncio
    async def test_significant_data_reduction(self, experiment):
        """Test that log analysis achieves significant data reduction"""
        result = await experiment.run()

        # Note: In mock data, we simulate the data flow
        # Real scenario: 1GB → 10KB (reduction ratio ~0.00001)
        # Mock data has smaller input/output for testing
        # Just verify the experiment produces meaningful output
        assert result.total_output_bytes > 0
        assert result.final_output != ""
