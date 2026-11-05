
import json
import math
from typing import Any

from src.agents.report_mock_client import ReportMockLLMClient
from src.llm.base import BaseLLMClient
from src.llm.structured_outputs import (
    ExecutiveReportOutput,
    ReportFinding,
)


class ReportGeneratorAgent:
    """Generate an executive report from independently verified output."""

    def __init__(self, client: BaseLLMClient | None = None):
        self.client = client or ReportMockLLMClient()

    def run(
        self,
        user_task: str,
        specialist_result: dict[str, Any],
        verification_result: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(user_task, str) or not user_task.strip():
            raise ValueError("User task cannot be empty.")

        if not isinstance(specialist_result, dict):
            raise TypeError("Specialist result must be a dictionary.")

        if not isinstance(verification_result, dict):
            raise TypeError("Verification result must be a dictionary.")

        specialist_name = specialist_result.get("agent_name")
        specialist_output = specialist_result.get("output")
        verification_output = verification_result.get("output")

        if specialist_name not in {
            "ResearchAgent",
            "SQLAnalystAgent",
            "PythonAnalystAgent",
        }:
            raise ValueError("Unsupported specialist agent.")

        if not isinstance(specialist_output, dict):
            raise TypeError("Specialist output must be a dictionary.")

        if not isinstance(verification_output, dict):
            raise TypeError("Verification output must be a dictionary.")

        if verification_result.get("agent_name") != "VerificationAgent":
            raise ValueError("A valid VerificationAgent result is required.")

        if verification_output.get("agent_name") != "VerificationAgent":
            raise ValueError("Verification output identity is invalid.")

        if verification_output.get("verified_agent") != specialist_name:
            raise ValueError(
                "Verification result does not match the specialist agent."
            )

        verification_status = verification_output.get("status")

        if verification_status == "rejected":
            raise ValueError(
                "Report generation blocked: specialist output was rejected."
            )

        if verification_status not in {"passed", "needs_review"}:
            raise ValueError("Unsupported verification status.")

        if verification_result.get("status") != {
            "passed": "completed",
            "needs_review": "needs_review",
        }[verification_status]:
            raise ValueError(
                "Verification result status is inconsistent."
            )

        if specialist_result.get("status") != "completed":
            raise ValueError(
                "Report generation requires a completed specialist result."
            )

        if specialist_output.get("agent_name") != specialist_name:
            raise ValueError("Specialist output identity is invalid.")

        findings, limitations = self._build_report_context(
            specialist_name,
            specialist_output,
        )

        verification_warnings = verification_output.get("warnings", [])

        if not isinstance(verification_warnings, list) or not all(
            isinstance(item, str) and item.strip()
            for item in verification_warnings
        ):
            raise ValueError("Verification warnings must be valid strings.")

        limitations.extend(verification_warnings)

        context = {
            "user_task": user_task.strip(),
            "verification_status": verification_status,
            "findings": [
                finding.model_dump()
                for finding in findings
            ],
            "limitations": list(dict.fromkeys(limitations)),
        }

        prompt = (
            "Generate an executive report using only the supplied "
            "specialist findings. Do not invent facts or claim that "
            "unexecuted SQL queries were executed. Preserve the supplied "
            "verification status and disclose all limitations.\n\n"
            "Report context:\n"
            f"{json.dumps(context, allow_nan=False)}"
        )

        response = self.client.generate(prompt)

        if not response.content:
            raise ValueError("Report client returned an empty response.")

        try:
            report_data = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Report client returned invalid JSON."
            ) from exc

        report = ExecutiveReportOutput.model_validate(report_data)

        if report.verification_status != verification_status:
            raise ValueError(
                "Generated report changed the verification status."
            )

        if report.status != (
            "completed"
            if verification_status == "passed"
            else "needs_review"
        ):
            raise ValueError(
                "Generated report has an inconsistent report status."
            )

        return {
            "agent_name": "ReportGeneratorAgent",
            "status": report.status,
            "output": report.model_dump(),
            "error": None,
        }

    @staticmethod
    def _build_report_context(
        specialist_name: str,
        output: dict[str, Any],
    ) -> tuple[list[ReportFinding], list[str]]:
        findings: list[ReportFinding] = []
        limitations: list[str] = []

        if specialist_name == "ResearchAgent":
            research_findings = output.get("findings")

            if not isinstance(research_findings, list) or not research_findings:
                raise ValueError("Research output contains no findings.")

            for index, item in enumerate(research_findings, start=1):
                if not isinstance(item, dict):
                    raise TypeError("Research finding must be an object.")

                finding = item.get("finding")
                evidence = item.get("evidence")
                confidence = item.get("confidence_score")

                if (
                    not isinstance(finding, str)
                    or not finding.strip()
                    or not isinstance(evidence, str)
                    or not evidence.strip()
                    or not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                ):
                    raise ValueError("Research finding is invalid.")

                findings.append(
                    ReportFinding(
                        title=f"Research finding {index}",
                        detail=(
                            f"{finding} Evidence: {evidence} "
                            f"Confidence score: {confidence:.2f}."
                        ),
                        source_agent=specialist_name,
                    )
                )

            missing_information = output.get("missing_information", [])

            if not isinstance(missing_information, list) or not all(
                isinstance(item, str) for item in missing_information
            ):
                raise ValueError("Research missing information is invalid.")

            limitations.extend(missing_information)

            limitations.append(
                "Research findings are based on the supplied agent output; "
                "independent external source verification is not established."
            )

        elif specialist_name == "SQLAnalystAgent":
            query = output.get("sql_query")
            explanation = output.get("explanation")
            tables = output.get("tables_used")

            if (
                not isinstance(query, str)
                or not query.strip()
                or not isinstance(explanation, str)
                or not explanation.strip()
                or not isinstance(tables, list)
                or not all(isinstance(table, str) for table in tables)
                or output.get("is_read_only") is not True
            ):
                raise ValueError("SQL analysis output is invalid.")

            findings.append(
                ReportFinding(
                    title="SQL analysis",
                    detail=(
                        f"{explanation} Query: {query} "
                        f"Tables referenced: {', '.join(tables)}."
                    ),
                    source_agent=specialist_name,
                )
            )

            limitations.append(
                "The SQL query was validated but not executed. "
                "No database result or revenue figure is claimed."
            )

        elif specialist_name == "PythonAnalystAgent":
            metrics = output.get("key_metrics")
            insights = output.get("insights")
            reported_limitations = output.get("limitations", [])

            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Python analysis metrics are invalid.")

            for metric_name, value in metrics.items():
                if (
                    not isinstance(metric_name, str)
                    or not metric_name.strip()
                    or not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError("Python analysis metric is invalid.")

                findings.append(
                    ReportFinding(
                        title=f"Metric: {metric_name}",
                        detail=f"Calculated metric value: {value}.",
                        source_agent=specialist_name,
                    )
                )

            if not isinstance(insights, list) or not insights or not all(
                isinstance(item, str) and item.strip()
                for item in insights
            ):
                raise ValueError("Python analysis insights are invalid.")

            for index, insight in enumerate(insights, start=1):
                findings.append(
                    ReportFinding(
                        title=f"Python insight {index}",
                        detail=insight,
                        source_agent=specialist_name,
                    )
                )

            if not isinstance(reported_limitations, list) or not all(
                isinstance(item, str)
                for item in reported_limitations
            ):
                raise ValueError("Python analysis limitations are invalid.")

            limitations.extend(reported_limitations)

        if not findings:
            raise ValueError("No reportable findings were produced.")

        return findings, limitations
