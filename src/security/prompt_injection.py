
"""Heuristic prompt-injection inspection for untrusted agent input."""

import html
import re
from dataclasses import dataclass


class UnsafeInputError(ValueError):
    """Raised when untrusted input violates configured safety limits."""


@dataclass(frozen=True)
class InputInspectionResult:
    """Result of inspecting untrusted input."""

    is_suspicious: bool
    matched_patterns: tuple[str, ...]
    original_length: int


INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "instruction_override",
        r"\bignore\b.{0,40}\b(previous|prior|above|all)\b.{0,30}\b(instructions?|rules?)\b",
    ),
    (
        "prompt_extraction",
        r"\b(reveal|show|print|expose|repeat|disclose)\b.{0,50}\b(system|developer)\b.{0,30}\b(prompt|message|instructions?)\b",
    ),
    (
        "secret_extraction",
        r"\b(reveal|show|print|expose|extract|disclose)\b.{0,50}\b(api[\s_-]?keys?|passwords?|credentials?|secrets?|tokens?)\b",
    ),
    (
        "security_override",
        r"\b(disable|bypass|override|ignore)\b.{0,40}\b(security|safety|permission|access control)\b",
    ),
    (
        "role_manipulation",
        r"\b(you are now|act as|pretend to be)\b.{0,50}\b(system|administrator|developer|root)\b",
    ),
    (
        "unauthorized_tool_execution",
        r"\b(execute|invoke|call|run)\b.{0,50}\b(delete|drop|truncate|write|admin)\b.{0,30}\b(tool|query|command|operation)\b",
    ),
)


DEFAULT_MAX_INPUT_LENGTH = 12_000


def inspect_untrusted_input(
    text: str,
    max_length: int = DEFAULT_MAX_INPUT_LENGTH,
) -> InputInspectionResult:
    """Inspect untrusted text for common prompt-injection indicators."""

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise TypeError("max_length must be an integer.")

    if max_length < 1:
        raise ValueError("max_length must be positive.")

    if len(text) > max_length:
        raise UnsafeInputError(
            f"Input exceeds the maximum allowed length of {max_length}."
        )

    normalized_text = re.sub(r"\s+", " ", text).casefold()

    matched_patterns = tuple(
        pattern_name
        for pattern_name, pattern in INJECTION_PATTERNS
        if re.search(pattern, normalized_text)
    )

    return InputInspectionResult(
        is_suspicious=bool(matched_patterns),
        matched_patterns=matched_patterns,
        original_length=len(text),
    )


def wrap_untrusted_input(text: str) -> str:
    """Delimit and escape untrusted content before prompt construction."""

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    escaped_text = html.escape(text, quote=True)

    return (
        "<UNTRUSTED_INPUT>\n"
        f"{escaped_text}\n"
        "</UNTRUSTED_INPUT>"
    )
