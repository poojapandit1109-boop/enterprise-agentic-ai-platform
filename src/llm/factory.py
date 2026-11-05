
from functools import lru_cache

from src.config import get_settings
from src.llm.base import BaseLLMClient
from src.llm.mock_client import MockLLMClient
from src.llm.reliability import ReliableLLMClient, RetryPolicy


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


@lru_cache
def get_reliable_llm_client() -> ReliableLLMClient:
    """
    Return the configured LLM provider wrapped with reliability features.

    Adds retry handling, exponential backoff, fallback support
    and request metrics without changing the original client factory.
    """

    return ReliableLLMClient(
        client=get_llm_client(),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=8.0,
        ),
    )
