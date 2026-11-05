
import pytest

from src.llm.base import BaseLLMClient, LLMResponse
from src.llm.reliability import (
    LLMRetriesExhaustedError,
    ReliableLLMClient,
    RetryPolicy,
    RetryableLLMError,
    estimate_tokens,
)


class SuccessfulMockClient(BaseLLMClient):
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            content="Mock response generated successfully.",
            model="mock-llm",
            is_mock=True,
        )


class FailingMockClient(BaseLLMClient):
    def generate(self, prompt: str) -> LLMResponse:
        raise RetryableLLMError("Temporary service failure.")


class IntermittentMockClient(BaseLLMClient):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1

        if self.calls < 3:
            raise RetryableLLMError("Temporary failure.")

        return LLMResponse(
            content="Recovered successfully.",
            model="mock-llm",
            is_mock=True,
        )


def test_successful_request_without_retry():
    client = ReliableLLMClient(
        SuccessfulMockClient(),
        sleep_fn=lambda _: None,
    )

    response = client.generate("Analyze revenue.")

    assert response.is_mock is True
    assert response.model == "mock-llm"
    assert client.last_metrics.attempts == 1
    assert client.last_metrics.fallback_used is False


def test_temporary_failure_then_success():
    mock = IntermittentMockClient()

    client = ReliableLLMClient(
        mock,
        retry_policy=RetryPolicy(max_attempts=3),
        sleep_fn=lambda _: None,
    )

    response = client.generate("Analyze revenue.")

    assert response.content == "Recovered successfully."
    assert mock.calls == 3
    assert client.last_metrics.attempts == 3


def test_exponential_backoff_delays():
    delays = []
    mock = FailingMockClient()

    client = ReliableLLMClient(
        mock,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=1,
            backoff_multiplier=2,
            max_delay_seconds=8,
        ),
        sleep_fn=delays.append,
    )

    with pytest.raises(LLMRetriesExhaustedError):
        client.generate("Analyze revenue.")

    assert delays == [1, 2]


def test_retries_exhausted_without_fallback():
    client = ReliableLLMClient(
        FailingMockClient(),
        retry_policy=RetryPolicy(max_attempts=3),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(LLMRetriesExhaustedError):
        client.generate("Analyze revenue.")

    assert client.last_metrics.attempts == 3
    assert client.last_metrics.fallback_used is False


def test_fallback_response_after_retries():
    client = ReliableLLMClient(
        FailingMockClient(),
        retry_policy=RetryPolicy(max_attempts=2),
        fallback_response="Fallback response.",
        sleep_fn=lambda _: None,
    )

    response = client.generate("Analyze revenue.")

    assert response.content == "Fallback response."
    assert response.model == "fallback"
    assert response.is_mock is True
    assert client.last_metrics.fallback_used is True
    assert client.last_metrics.attempts == 2


def test_token_usage_metrics():
    client = ReliableLLMClient(
        SuccessfulMockClient(),
        sleep_fn=lambda _: None,
    )

    prompt = "Analyze quarterly revenue."
    response = client.generate(prompt)

    assert client.last_metrics.estimated_input_tokens == (
        estimate_tokens(prompt)
    )

    assert client.last_metrics.estimated_output_tokens == (
        estimate_tokens(response.content)
    )


def test_execution_time_is_recorded():
    client = ReliableLLMClient(
        SuccessfulMockClient(),
        sleep_fn=lambda _: None,
    )

    client.generate("Analyze revenue.")

    assert client.last_metrics.elapsed_seconds >= 0


def test_empty_prompt_is_rejected():
    client = ReliableLLMClient(
        SuccessfulMockClient(),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("")


def test_invalid_retry_attempts_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy(max_attempts=0)


def test_invalid_backoff_multiplier_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy(backoff_multiplier=0.5)


def test_timeout_is_retried():
    class TimeoutMockClient(BaseLLMClient):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> LLMResponse:
            self.calls += 1

            if self.calls == 1:
                raise TimeoutError("Request timed out.")

            return LLMResponse(
                content="Recovered after timeout.",
                model="mock-llm",
                is_mock=True,
            )

    mock = TimeoutMockClient()

    client = ReliableLLMClient(
        mock,
        retry_policy=RetryPolicy(max_attempts=3),
        sleep_fn=lambda _: None,
    )

    response = client.generate("Analyze revenue.")

    assert response.content == "Recovered after timeout."
    assert mock.calls == 2
    assert client.last_metrics.attempts == 2


def test_unexpected_errors_are_not_retried():
    class UnexpectedErrorClient(BaseLLMClient):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> LLMResponse:
            self.calls += 1
            raise ValueError("Unexpected error.")

    mock = UnexpectedErrorClient()

    client = ReliableLLMClient(
        mock,
        retry_policy=RetryPolicy(max_attempts=3),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(ValueError, match="Unexpected error"):
        client.generate("Analyze revenue.")

    assert mock.calls == 1
