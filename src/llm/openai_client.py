
from openai import OpenAI

from src.config import get_settings
from src.llm.base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of the common LLM interface."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. "
                "Configure it in the project .env file."
            )

        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=30.0,
            max_retries=2,
        )

        self.model = "gpt-4.1-mini"

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=500,
        )

        return LLMResponse(
            content=response.output_text,
            model=self.model,
            is_mock=False,
        )
