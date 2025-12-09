"""
Tests for Scenario 3: Research Assistant Pipeline
"""

import pytest
from experiments.scenario3_research_assistant import ResearchAssistantExperiment


class TestResearchAssistantScenario:
    """Tests for the Research Assistant scenario"""

    @pytest.fixture
    def experiment(self):
        return ResearchAssistantExperiment()

    def test_tool_chain(self, experiment):
        """Test that the tool chain is correctly defined"""
        # Note: fetch and summarize are called 5 times each
        chain = experiment.get_tool_chain()
        assert chain[0] == "brave-search"
        assert "fetch" in chain
        assert "summarize" in chain
        assert "data_aggregate" in chain
        assert "sequentialthinking" in chain
        assert "llm_generate" in chain
        assert chain[-1] == "filesystem"

    def test_user_request(self, experiment):
        """Test that the user request is defined"""
        request = experiment.get_user_request()
        assert "논문" in request or "paper" in request.lower()
        assert "요약" in request or "summarize" in request.lower()

    @pytest.mark.asyncio
    async def test_experiment_runs(self, experiment):
        """Test that the experiment runs successfully"""
        result = await experiment.run()

        assert result.scenario_name == "scenario3_research_assistant"
        assert result.success is True
        # 1 search + 5 fetch + 5 summarize + 1 aggregate + 1 thinking + 1 llm + 1 write = 15
        assert result.tool_chain_length >= 7

    @pytest.mark.asyncio
    async def test_multiple_fetch_calls(self, experiment):
        """Test that multiple papers are fetched"""
        result = await experiment.run()

        fetch_calls = [tc for tc in result.tool_calls if tc.tool_name == "fetch"]
        assert len(fetch_calls) == 5  # 5 papers

    @pytest.mark.asyncio
    async def test_multiple_summarize_calls(self, experiment):
        """Test that multiple papers are summarized"""
        result = await experiment.run()

        summarize_calls = [tc for tc in result.tool_calls if tc.tool_name == "summarize"]
        assert len(summarize_calls) == 5  # 5 papers
