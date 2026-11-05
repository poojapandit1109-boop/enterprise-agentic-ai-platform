
from typing import Any

from src.agents.python_data_analyzer import analyze_dataframe
from src.agents.python_mock_client import PythonMockLLMClient
from src.config import get_settings
from src.llm.base import BaseLLMClient
from src.llm.factory import get_reliable_llm_client
from src.llm.prompts import get_prompt
from src.llm.structured_client import generate_structured_output
from src.llm.structured_outputs import PythonAnalysisOutput
from src.orchestration.state import AgentResult


class PythonAnalystAgent:
    """Analyze structured datasets using Pandas and an LLM."""

    def __init__(self, client: BaseLLMClient | None = None):
        if client is not None:
            self.client = client
        elif get_settings().llm_provider.strip().lower() == "mock":
            self.client = PythonMockLLMClient()
        else:
            self.client = get_reliable_llm_client()

        self.prompt_template = get_prompt("python_analyst")

    def run(
        self,
        objective: str,
        records: list[dict[str, Any]],
    ) -> AgentResult:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Analysis objective cannot be empty.")

        if not isinstance(records, list) or not records:
            raise ValueError("Analysis dataset cannot be empty.")

        analysis = analyze_dataframe(records)

        data_description = self._build_data_description(analysis)

        prompt = self.prompt_template.render(
            objective=objective,
            data_description=data_description,
        )

        result = generate_structured_output(
            client=self.client,
            prompt=prompt,
            schema=PythonAnalysisOutput,
        )

        if not isinstance(result, PythonAnalysisOutput):
            raise TypeError("Invalid Python Analyst output.")

        if result.objective != objective:
            raise ValueError("LLM output objective does not match the request.")

        if result.key_metrics != analysis["key_metrics"]:
            raise ValueError(
                "LLM output metrics do not match Pandas calculations."
            )

        return {
            "agent_name": "PythonAnalystAgent",
            "status": result.status,
            "output": result.model_dump(),
            "error": None,
        }

    @staticmethod
    def _build_data_description(analysis: dict[str, Any]) -> str:
        return __import__("json").dumps(
            {
                "row_count": analysis["row_count"],
                "column_count": analysis["column_count"],
                "numeric_columns": analysis["numeric_columns"],
                "key_metrics": analysis["key_metrics"],
                "insights": analysis["insights"],
                "limitations": [
                    "Analysis is based only on the supplied dataset.",
                    "Correlation does not establish causation.",
                    "No independent data verification was performed.",
                ],
            }
        )
