
import json
import re

from src.llm.base import BaseLLMClient, LLMResponse


class PythonMockLLMClient(BaseLLMClient):
    """Generate deterministic structured Python Analyst responses."""

    def generate(self, prompt: str) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        objective_match = re.search(
            r"Analysis objective:\s*(.*?)\n\nAvailable data description:",
            prompt,
            re.DOTALL,
        )

        data_match = re.search(
            r"Available data description:\s*(.*?)\n\nInstructions:",
            prompt,
            re.DOTALL,
        )

        if not objective_match or not data_match:
            raise ValueError("Prompt does not contain required sections.")

        objective = objective_match.group(1).strip()

        try:
            analysis_data = json.loads(data_match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Data description must contain valid JSON."
            ) from exc

        if not isinstance(analysis_data, dict):
            raise TypeError("Data description must be a JSON object.")

        row_count = analysis_data.get("row_count")
        column_count = analysis_data.get("column_count")
        key_metrics = analysis_data.get("key_metrics")
        insights = analysis_data.get("insights")
        limitations = analysis_data.get("limitations")

        if not isinstance(key_metrics, dict):
            raise TypeError("Analysis metrics must be a dictionary.")

        if not isinstance(insights, list) or not insights:
            raise ValueError("Analysis insights must be a non-empty list.")

        if not isinstance(limitations, list):
            raise TypeError("Limitations must be a list.")

        response = {
            "agent_name": "PythonAnalystAgent",
            "status": "completed",
            "objective": objective,
            "analysis_summary": (
                f"Completed a simulated analytical interpretation of "
                f"{row_count} rows across {column_count} columns. "
                "Numerical metrics were calculated using Pandas."
            ),
            "key_metrics": key_metrics,
            "insights": insights,
            "limitations": limitations,
        }

        return LLMResponse(
            content=json.dumps(response),
            model="python-mock-llm",
            is_mock=True,
        )
