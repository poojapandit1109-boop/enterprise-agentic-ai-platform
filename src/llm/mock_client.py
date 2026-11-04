
from src.llm.base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Simulated LLM client for development and testing."""

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        return LLMResponse(
            content=(
                "This is a simulated LLM response. "
                "The mock client received your prompt successfully. "
                "No external API request was made."
            ),
            model="mock-llm",
            is_mock=True,
        )
