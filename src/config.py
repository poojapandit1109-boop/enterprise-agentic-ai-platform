
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    app_name: str = "Enterprise Agentic AI Platform"
    app_env: str = "development"
    debug: bool = True

    # LLM provider: mock or openai
    llm_provider: str = "mock"

    # OpenAI configuration
    openai_api_key: str = Field(default="")

    # Database configuration
    database_url: str
    redis_url: str

    # Observability configuration
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
