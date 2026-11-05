
from pydantic import BaseModel

from src.llm.base import BaseLLMClient
from src.llm.output_parser import parse_structured_output


def generate_structured_output(
    client: BaseLLMClient,
    prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    """Generate an LLM response and validate it against a schema."""

    response = client.generate(prompt)

    return parse_structured_output(
        response.content,
        schema,
    )
