"""
Tests for Scenario 1: Code Review Pipeline
"""

import pytest
from experiments.scenario1_code_review import CodeReviewExperiment


class TestCodeReviewScenario:
    """Tests for the Code Review scenario"""

    @pytest.fixture
    def experiment(self):
        return CodeReviewExperiment()

    def test_tool_chain(self, experiment):
        """Test that the tool chain is correctly defined"""
        expected = ["filesystem", "git", "git", "fetch", "sequentialthinking", "llm_generate", "filesystem"]
        assert experiment.get_tool_chain() == expected

    def test_user_request(self, experiment):
        """Test that the user request is defined"""
        request = experiment.get_user_request()
        assert "Git" in request or "git" in request
        assert "리뷰" in request or "review" in request.lower()

    @pytest.mark.asyncio
    async def test_experiment_runs(self, experiment):
        """Test that the experiment runs successfully"""
        result = await experiment.run()

        assert result.scenario_name == "scenario1_code_review"
        assert result.success is True
        assert result.tool_chain_length == 7
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_data_reduction(self, experiment):
        """Test that data reduction occurs"""
        result = await experiment.run()

        # Some tool calls should show reduction
        assert result.total_input_bytes > 0
        assert result.total_output_bytes > 0

    @pytest.mark.asyncio
    async def test_result_serialization(self, experiment, tmp_path):
        """Test that results can be saved to JSON"""
        result = await experiment.run()

        output_path = result.save(tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".json"
