
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standard response returned by every LLM client."""

    content: str
    model: str
    is_mock: bool


class BaseLLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from a prompt."""
        raise NotImplementedError
