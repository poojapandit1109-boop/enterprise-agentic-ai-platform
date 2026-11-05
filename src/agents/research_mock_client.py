import json

from src.llm.base import BaseLLMClient, LLMResponse


class ResearchMockLLMClient(BaseLLMClient):
    """Return deterministic structured research output without API calls."""

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        marker = "Research the following topic carefully:\n\n"

        if marker not in prompt:
            raise ValueError("Research topic could not be extracted from prompt.")

        topic = prompt.split(marker, 1)[1].split(
            "\n\nInstructions:", 1
        )[0].strip()

        if not topic:
            raise ValueError("Research topic cannot be empty.")

        response = {
            "agent_name": "ResearchAgent",
            "status": "completed",
            "topic": topic,
            "summary": (
                f"This is a simulated research summary for: {topic}"
            ),
            "findings": [
                {
                    "finding": (
                        f"A simulated research finding about {topic}."
                    ),
                    "evidence": (
                        "This is synthetic demonstration evidence, "
                        "not independently verified information."
                    ),
                    "confidence_score": 0.85,
                }
            ],
            "missing_information": [
                "Independent source verification is not performed "
                "by this mock client."
            ],
        }

        return LLMResponse(
            content=json.dumps(response),
            model="research-mock-llm",
            is_mock=True,
        )