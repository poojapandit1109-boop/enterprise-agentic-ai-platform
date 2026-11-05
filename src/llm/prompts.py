
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Reusable, version-controlled prompt template."""

    name: str
    version: str
    template: str
    required_variables: tuple[str, ...]

    def render(self, **kwargs: str) -> str:
        """Validate variables and render the prompt."""

        if not self.template.strip():
            raise ValueError("Prompt template cannot be empty.")

        missing = set(self.required_variables) - set(kwargs)
        if missing:
            raise ValueError(
                f"Missing required prompt variables: {sorted(missing)}"
            )

        extra = set(kwargs) - set(self.required_variables)
        if extra:
            raise ValueError(
                f"Unexpected prompt variables: {sorted(extra)}"
            )

        for variable, value in kwargs.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Prompt variable '{variable}' must be a non-empty string."
                )

        return self.template.format(**kwargs)


# Versioned prompt definitions for platform agents.

RESEARCH_PROMPT = PromptTemplate(
    name="research_agent",
    version="1.0.0",
    template=(
        "You are an enterprise research analyst.\n"
        "Research the following topic carefully:\n\n"
        "{topic}\n\n"
        "Instructions:\n"
        "- Use only the information provided to you.\n"
        "- Separate verified facts from assumptions.\n"
        "- Identify missing information.\n"
        "- Present findings in clear bullet points."
    ),
    required_variables=("topic",),
)


SQL_ANALYST_PROMPT = PromptTemplate(
    name="sql_analyst",
    version="1.0.0",
    template=(
        "You are an enterprise SQL analyst.\n"
        "Business question: {business_question}\n\n"
        "Available database schema:\n"
        "{database_schema}\n\n"
        "Instructions:\n"
        "- Generate a read-only SQL query.\n"
        "- Use only tables and columns in the supplied schema.\n"
        "- Never generate INSERT, UPDATE, DELETE, or DROP statements.\n"
        "- Explain what the query is intended to retrieve."
    ),
    required_variables=("business_question", "database_schema"),
)


PYTHON_ANALYST_PROMPT = PromptTemplate(
    name="python_analyst",
    version="1.0.0",
    template=(
        "You are an enterprise Python data analyst.\n"
        "Analysis objective: {objective}\n\n"
        "Available data description:\n"
        "{data_description}\n\n"
        "Instructions:\n"
        "- Explain the analysis approach.\n"
        "- Use appropriate statistical methods.\n"
        "- Identify assumptions and limitations.\n"
        "- Present results clearly."
    ),
    required_variables=("objective", "data_description"),
)


PROMPT_REGISTRY = {
    "research_agent": RESEARCH_PROMPT,
    "sql_analyst": SQL_ANALYST_PROMPT,
    "python_analyst": PYTHON_ANALYST_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Retrieve a registered prompt by name."""

    try:
        return PROMPT_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Prompt not found: {name}") from None
