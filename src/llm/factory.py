
from functools import lru_cache

from src.config import get_settings
from src.llm.base import BaseLLMClient
from src.llm.mock_client import MockLLMClient


@lru_cache
def get_llm_client() -> BaseLLMClient:
    """
    Return the configured LLM provider.

    Mock mode makes no external API requests.
    """

    settings = get_settings()

    provider = settings.llm_provider.strip().lower()

    if provider == "mock":
        return MockLLMClient()

    if provider == "openai":
        from src.llm.openai_client import OpenAIClient

        return OpenAIClient()

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
