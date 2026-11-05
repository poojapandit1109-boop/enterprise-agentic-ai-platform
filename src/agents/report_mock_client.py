
import json

from src.llm.base import BaseLLMClient, LLMResponse


class ReportMockLLMClient(BaseLLMClient):
    """Generate deterministic executive reports without API calls."""

    def generate(self, prompt: str) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        marker = "Report context:\n"

        if marker not in prompt:
            raise ValueError("Report context could not be extracted.")

        context_text = prompt.split(marker, 1)[1].strip()

        try:
            context = json.loads(context_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Report context must contain valid JSON."
            ) from exc

        if not isinstance(context, dict):
            raise TypeError("Report context must be a JSON object.")

        verification_status = context.get("verification_status")

        if verification_status not in {"passed", "needs_review"}:
            raise ValueError("Unsupported verification status.")

        findings = context.get("findings")
        limitations = context.get("limitations", [])

        if not isinstance(findings, list) or not findings:
            raise ValueError("At least one finding is required.")

        if not isinstance(limitations, list):
            raise TypeError("Limitations must be a list.")

        if not all(isinstance(item, dict) for item in findings):
            raise TypeError("Each finding must be a JSON object.")

        if not all(isinstance(item, str) for item in limitations):
            raise TypeError("Each limitation must be a string.")

        report_status = (
            "completed"
            if verification_status == "passed"
            else "needs_review"
        )

        disclaimer = (
            "This report was generated from independently verified "
            "agent outputs."
            if verification_status == "passed"
            else (
                "WARNING: This report contains findings requiring human "
                "review. Do not treat these findings as fully verified "
                "or use them as the sole basis for business decisions."
            )
        )

        response = {
            "agent_name": "ReportGeneratorAgent",
            "status": report_status,
            "report_title": (
                f"Executive Analysis Report: "
                f"{context.get('user_task', 'Business Analysis')}"
            ),
            "executive_summary": (
                "This executive report consolidates the available "
                "specialist-agent findings and their verification status."
            ),
            "key_findings": findings,
            "recommendations": [
                (
                    "Review the findings with the relevant business "
                    "stakeholders before taking action."
                ),
                (
                    "Address the documented limitations and collect "
                    "additional evidence where necessary."
                ),
            ],
            "limitations": limitations,
            "verification_status": verification_status,
            "disclaimer": disclaimer,
        }

        return LLMResponse(
            content=json.dumps(response),
            model="report-mock-llm",
            is_mock=True,
        )
