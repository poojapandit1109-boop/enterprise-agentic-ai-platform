
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be parsed or validated."""


def parse_structured_output(
    response: str,
    schema: type[T],
) -> T:
    """Parse JSON text and validate it against a Pydantic schema."""

    if not isinstance(response, str) or not response.strip():
        raise StructuredOutputError("LLM response cannot be empty.")

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"Invalid JSON response: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise StructuredOutputError(
            "Structured output must be a JSON object."
        )

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"Structured output validation failed: {exc}"
        ) from exc
