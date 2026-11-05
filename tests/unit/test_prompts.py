
import pytest

from src.llm.prompts import (
    PROMPT_REGISTRY,
    PromptTemplate,
    get_prompt,
)


def test_prompt_registry_contains_all_agents():
    assert "research_agent" in PROMPT_REGISTRY
    assert "sql_analyst" in PROMPT_REGISTRY
    assert "python_analyst" in PROMPT_REGISTRY


def test_prompt_versions_are_defined():
    for prompt in PROMPT_REGISTRY.values():
        assert prompt.version == "1.0.0"


def test_research_prompt_renders_successfully():
    prompt = get_prompt("research_agent")

    result = prompt.render(
        topic="Reasons for quarterly revenue decline"
    )

    assert "quarterly revenue decline" in result
    assert "enterprise research analyst" in result.lower()


def test_sql_prompt_renders_successfully():
    prompt = get_prompt("sql_analyst")

    result = prompt.render(
        business_question="Find total quarterly revenue",
        database_schema="sales(id, revenue, quarter)",
    )

    assert "Find total quarterly revenue" in result
    assert "sales(id, revenue, quarter)" in result


def test_python_prompt_renders_successfully():
    prompt = get_prompt("python_analyst")

    result = prompt.render(
        objective="Identify revenue trends",
        data_description="Quarterly revenue dataset",
    )

    assert "Identify revenue trends" in result
    assert "Quarterly revenue dataset" in result


def test_missing_variable_is_rejected():
    prompt = get_prompt("research_agent")

    with pytest.raises(ValueError, match="Missing required"):
        prompt.render()


def test_empty_variable_is_rejected():
    prompt = get_prompt("research_agent")

    with pytest.raises(ValueError, match="non-empty string"):
        prompt.render(topic="   ")


def test_non_string_variable_is_rejected():
    prompt = get_prompt("research_agent")

    with pytest.raises(ValueError, match="non-empty string"):
        prompt.render(topic=123)


def test_unexpected_variable_is_rejected():
    prompt = get_prompt("research_agent")

    with pytest.raises(ValueError, match="Unexpected prompt variables"):
        prompt.render(
            topic="Revenue analysis",
            extra="unexpected",
        )


def test_unknown_prompt_is_rejected():
    with pytest.raises(ValueError, match="Prompt not found"):
        get_prompt("unknown_agent")


def test_empty_template_is_rejected():
    prompt = PromptTemplate(
        name="empty_test",
        version="1.0.0",
        template="   ",
        required_variables=(),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        prompt.render()


def test_prompt_metadata_is_immutable():
    prompt = get_prompt("research_agent")

    with pytest.raises(AttributeError):
        prompt.version = "2.0.0"
