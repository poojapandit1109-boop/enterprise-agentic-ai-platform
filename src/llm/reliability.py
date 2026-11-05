
import time
from dataclasses import dataclass

from src.llm.base import BaseLLMClient, LLMResponse


class RetryableLLMError(Exception):
    """Raised when an LLM request fails temporarily."""


class LLMRetriesExhaustedError(RuntimeError):
    """Raised when all configured retry attempts fail."""


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 8.0

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        if self.initial_delay_seconds < 0:
            raise ValueError("Initial delay cannot be negative.")

        if self.backoff_multiplier < 1:
            raise ValueError("Backoff multiplier must be at least 1.")

        if self.max_delay_seconds < 0:
            raise ValueError("Maximum delay cannot be negative.")


@dataclass
class LLMCallMetrics:
    """Basic metrics for an LLM request."""

    attempts: int = 0
    elapsed_seconds: float = 0.0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    fallback_used: bool = False


def estimate_tokens(text: str) -> int:
    """Estimate tokens using approximately four characters per token.

    This is an approximation, not provider-reported token usage.
    """

    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)


class ReliableLLMClient(BaseLLMClient):
    """Add retries, fallback behavior and basic metrics to an LLM client."""

    def __init__(
        self,
        client: BaseLLMClient,
        retry_policy: RetryPolicy | None = None,
        fallback_response: str | None = None,
        sleep_fn=time.sleep,
    ):
        self.client = client
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_response = fallback_response
        self.sleep_fn = sleep_fn
        self.last_metrics = LLMCallMetrics()

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        started_at = time.monotonic()
        attempts = 0

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            attempts = attempt

            try:
                response = self.client.generate(prompt)

                self.last_metrics = LLMCallMetrics(
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started_at,
                    estimated_input_tokens=estimate_tokens(prompt),
                    estimated_output_tokens=estimate_tokens(
                        response.content
                    ),
                    fallback_used=False,
                )

                return response

            except (
                RetryableLLMError,
                TimeoutError,
                ConnectionError,
            ) as exc:

                if attempt == self.retry_policy.max_attempts:
                    break

                delay = min(
                    self.retry_policy.initial_delay_seconds
                    * (
                        self.retry_policy.backoff_multiplier
                        ** (attempt - 1)
                    ),
                    self.retry_policy.max_delay_seconds,
                )

                self.sleep_fn(delay)

            except Exception:
                self.last_metrics = LLMCallMetrics(
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started_at,
                )
                raise

        elapsed = time.monotonic() - started_at

        if self.fallback_response is not None:
            fallback = LLMResponse(
                content=self.fallback_response,
                model="fallback",
                is_mock=True,
            )

            self.last_metrics = LLMCallMetrics(
                attempts=attempts,
                elapsed_seconds=elapsed,
                estimated_input_tokens=estimate_tokens(prompt),
                estimated_output_tokens=estimate_tokens(
                    fallback.content
                ),
                fallback_used=True,
            )

            return fallback

        self.last_metrics = LLMCallMetrics(
            attempts=attempts,
            elapsed_seconds=elapsed,
        )

        raise LLMRetriesExhaustedError(
            f"LLM request failed after {attempts} attempts."
        )
