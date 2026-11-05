import json

import pytest

from src.agents.research_agent import ResearchAgent
from src.llm.base import BaseLLMClient, LLMResponse
from src.llm.output_parser import StructuredOutputError


class FakeResearchLLMClient(BaseLLMClient):
    """Test client that returns a controlled structured response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt

        return LLMResponse(
            content=self.response,
            model="fake-research-llm",
            is_mock=True,
        )


@pytest.fixture
def valid_research_response():
    return {
        "agent_name": "ResearchAgent",
        "status": "completed",
        "topic": "Enterprise AI adoption",
        "summary": "Enterprise AI adoption is increasing.",
        "findings": [
            {
                "finding": "Organizations are adopting AI tools.",
                "evidence": "This is a simulated finding for testing.",
                "confidence_score": 0.85,
            }
        ],
        "missing_information": [],
    }


def test_research_agent_returns_structured_result(valid_research_response):
    client = FakeResearchLLMClient(json.dumps(valid_research_response))
    agent = ResearchAgent(client=client)

    result = agent.run("Enterprise AI adoption")

    assert result["agent_name"] == "ResearchAgent"
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["output"]["topic"] == "Enterprise AI adoption"
    assert len(result["output"]["findings"]) == 1


def test_research_agent_uses_versioned_prompt(valid_research_response):
    client = FakeResearchLLMClient(json.dumps(valid_research_response))
    agent = ResearchAgent(client=client)

    agent.run("Enterprise AI adoption")

    assert "You are an enterprise research analyst." in client.last_prompt
    assert "Enterprise AI adoption" in client.last_prompt
    assert "Identify missing information." in client.last_prompt


def test_research_agent_rejects_empty_topic():
    client = FakeResearchLLMClient("{}")
    agent = ResearchAgent(client=client)

    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        agent.run("   ")


def test_research_agent_rejects_invalid_json():
    client = FakeResearchLLMClient("This is not valid JSON.")
    agent = ResearchAgent(client=client)

    with pytest.raises(StructuredOutputError):
        agent.run("Enterprise AI adoption")


def test_research_agent_rejects_invalid_schema():
    invalid_response = {
        "agent_name": "ResearchAgent",
        "status": "completed",
        "topic": "Enterprise AI adoption",
        "summary": "A summary.",
        "findings": [
            {
                "finding": "A finding.",
                "evidence": "Test evidence.",
                "confidence_score": 1.5,
            }
        ],
        "missing_information": [],
    }

    client = FakeResearchLLMClient(json.dumps(invalid_response))
    agent = ResearchAgent(client=client)

    with pytest.raises(StructuredOutputError):
        agent.run("Enterprise AI adoption")


def test_research_agent_supports_needs_review(valid_research_response):
    valid_research_response["status"] = "needs_review"
    valid_research_response["missing_information"] = [
        "Independent source verification is required."
    ]

    client = FakeResearchLLMClient(json.dumps(valid_research_response))
    agent = ResearchAgent(client=client)

    result = agent.run("Enterprise AI adoption")

    assert result["status"] == "needs_review"
    assert len(result["output"]["missing_information"]) == 1


def test_research_agent_uses_default_structured_mock_client():
    agent = ResearchAgent()

    result = agent.run("Enterprise AI adoption")

    assert result["agent_name"] == "ResearchAgent"
    assert result["status"] == "completed"
    assert result["output"]["topic"] == "Enterprise AI adoption"
    assert result["output"]["findings"]
    assert "synthetic demonstration evidence" in (
        result["output"]["findings"][0]["evidence"]
    )
    assert result["error"] is None