
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructuredOutput(BaseModel):
    """Base configuration shared by agent response schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ResearchFinding(StructuredOutput):
    """A single finding produced by the Research Agent."""

    finding: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ResearchOutput(StructuredOutput):
    """Structured response from the Research Agent."""

    agent_name: Literal["ResearchAgent"] = "ResearchAgent"
    status: Literal["completed", "needs_review"]
    topic: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    findings: list[ResearchFinding] = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)


class SQLAnalysisOutput(StructuredOutput):
    """Structured response from the SQL Analyst Agent."""

    agent_name: Literal["SQLAnalystAgent"] = "SQLAnalystAgent"
    status: Literal["completed", "needs_review"]
    business_question: str = Field(min_length=1)
    sql_query: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    tables_used: list[str] = Field(min_length=1)
    is_read_only: Literal[True]

    @field_validator("sql_query")
    @classmethod
    def validate_query_start(cls, value: str) -> str:
        normalized = value.strip().upper()

        if not (
            normalized.startswith("SELECT ")
            or normalized.startswith("SELECT\n")
            or normalized.startswith("WITH ")
            or normalized.startswith("WITH\n")
        ):
            raise ValueError("Only SELECT or WITH queries are accepted.")

        return value


class PythonAnalysisOutput(StructuredOutput):
    """Structured response from the Python Analyst Agent."""

    agent_name: Literal["PythonAnalystAgent"] = "PythonAnalystAgent"
    status: Literal["completed", "needs_review"]
    objective: str = Field(min_length=1)
    analysis_summary: str = Field(min_length=1)
    key_metrics: dict[str, float]
    insights: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
