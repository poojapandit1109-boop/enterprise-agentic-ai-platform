
from functools import lru_cache

from openai import OpenAI

from src.config import get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    """
    Create and return a reusable OpenAI client.

    The client is cached so the application does not
    repeatedly initialize new OpenAI connections.
    """

    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Configure it in the project .env file."
        )

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=30.0,
        max_retries=2,
    )
