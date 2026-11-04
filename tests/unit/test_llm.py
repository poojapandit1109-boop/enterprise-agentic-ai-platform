
import pytest

from src.llm.base import BaseLLMClient, LLMResponse
from src.llm.factory import get_llm_client
from src.llm.mock_client import MockLLMClient
from src.llm.openai_client import OpenAIClient


def test_mock_client_implements_base_interface():
    client = MockLLMClient()

    assert isinstance(client, BaseLLMClient)


def test_mock_client_returns_standard_response():
    client = MockLLMClient()

    response = client.generate("Analyze revenue decline")

    assert isinstance(response, LLMResponse)
    assert response.model == "mock-llm"
    assert response.is_mock is True
    assert response.content


def test_mock_client_rejects_empty_prompt():
    client = MockLLMClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("")


def test_mock_client_rejects_whitespace_prompt():
    client = MockLLMClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("   ")


def test_openai_client_implements_base_interface():
    assert issubclass(OpenAIClient, BaseLLMClient)


def test_factory_returns_mock_client():
    get_llm_client.cache_clear()

    client = get_llm_client()

    assert isinstance(client, MockLLMClient)
    assert client.generate("Test prompt").is_mock is True

    get_llm_client.cache_clear()
