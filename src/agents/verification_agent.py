
import math
from typing import Any

import sqlglot
from sqlglot import exp

from src.llm.structured_outputs import (
    VerificationCheck,
    VerificationOutput,
)

SUPPORTED_AGENTS = {
    "ResearchAgent",
    "SQLAnalystAgent",
    "PythonAnalystAgent",
}


class VerificationAgent:
    """Independently validate specialist-agent results."""

    def run(self, agent_result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(agent_result, dict):
            raise TypeError("Agent result must be a dictionary.")

        agent_name = agent_result.get("agent_name")

        if agent_name not in SUPPORTED_AGENTS:
            raise ValueError("Unsupported agent name for verification.")

        checks: list[VerificationCheck] = []

        self._check_agent_identity(agent_result, agent_name, checks)
        self._check_agent_status(agent_result, checks)

        output = agent_result.get("output")

        if not isinstance(output, dict):
            self._add_check(
                checks,
                name="output_structure",
                passed=False,
                severity="critical",
                details="Agent output must be a dictionary.",
            )
        else:
            if agent_name == "ResearchAgent":
                self._verify_research(output, checks)

            elif agent_name == "SQLAnalystAgent":
                self._verify_sql(output, checks)

            elif agent_name == "PythonAnalystAgent":
                self._verify_python(output, checks)

        failed_critical = any(
            not check.passed and check.severity == "critical"
            for check in checks
        )

        failed_warnings = any(
            not check.passed and check.severity == "warning"
            for check in checks
        )

        if failed_critical:
            verification_status = "rejected"
        elif failed_warnings:
            verification_status = "needs_review"
        else:
            verification_status = "passed"

        warnings = [
            check.details
            for check in checks
            if not check.passed and check.severity == "warning"
        ]

        summary = (
            f"{agent_name} verification {verification_status}. "
            f"{sum(check.passed for check in checks)} of "
            f"{len(checks)} checks passed."
        )

        verification = VerificationOutput(
            status=verification_status,
            verified_agent=agent_name,
            checks=checks,
            summary=summary,
            warnings=warnings,
        )

        agent_status = {
            "passed": "completed",
            "rejected": "failed",
            "needs_review": "needs_review",
        }[verification_status]

        return {
            "agent_name": "VerificationAgent",
            "status": agent_status,
            "output": verification.model_dump(),
            "error": None,
        }

    @staticmethod
    def _add_check(
        checks: list[VerificationCheck],
        name: str,
        passed: bool,
        severity: str,
        details: str,
    ) -> None:
        checks.append(
            VerificationCheck(
                name=name,
                passed=passed,
                severity=severity,
                details=details,
            )
        )

    def _check_agent_identity(
        self,
        result: dict[str, Any],
        agent_name: str,
        checks: list[VerificationCheck],
    ) -> None:
        output = result.get("output")

        output_identity = (
            output.get("agent_name")
            if isinstance(output, dict)
            else None
        )

        self._add_check(
            checks,
            name="agent_identity",
            passed=output_identity == agent_name,
            severity="critical",
            details=(
                "Agent identity matches its output."
                if output_identity == agent_name
                else "Agent identity does not match its output."
            ),
        )

    def _check_agent_status(
        self,
        result: dict[str, Any],
        checks: list[VerificationCheck],
    ) -> None:
        status = result.get("status")

        valid_status = status in {
            "completed",
            "failed",
            "needs_review",
        }

        self._add_check(
            checks,
            name="agent_status",
            passed=valid_status and status == "completed",
            severity="critical" if not valid_status or status == "failed" else "warning",
            details=(
                "Specialist agent completed successfully."
                if status == "completed"
                else f"Specialist agent status requires attention: {status}."
            ),
        )

        if result.get("error") is not None:
            self._add_check(
                checks,
                name="agent_error",
                passed=False,
                severity="critical",
                details=f"Specialist agent reported an error: {result['error']}",
            )

    def _verify_research(
        self,
        output: dict[str, Any],
        checks: list[VerificationCheck],
    ) -> None:
        required_fields = [
            "topic",
            "summary",
            "findings",
            "missing_information",
        ]

        fields_present = all(
            field in output for field in required_fields
        )

        self._add_check(
            checks,
            name="research_required_fields",
            passed=fields_present,
            severity="critical",
            details=(
                "Required research fields are present."
                if fields_present
                else "Research output is missing required fields."
            ),
        )

        findings = output.get("findings")

        valid_findings = (
            isinstance(findings, list)
            and len(findings) > 0
            and all(
                isinstance(item, dict)
                and isinstance(item.get("finding"), str)
                and bool(item["finding"].strip())
                and isinstance(item.get("evidence"), str)
                and bool(item["evidence"].strip())
                and isinstance(item.get("confidence_score"), (int, float))
                and not isinstance(item.get("confidence_score"), bool)
                and math.isfinite(item["confidence_score"])
                and 0.0 <= item["confidence_score"] <= 1.0
                for item in findings
            )
        )

        self._add_check(
            checks,
            name="research_findings",
            passed=valid_findings,
            severity="critical",
            details=(
                "Research findings contain evidence and valid confidence scores."
                if valid_findings
                else "Research findings are empty or contain invalid evidence or confidence scores."
            ),
        )

        missing_information = output.get("missing_information", [])

        has_missing_information = (
            isinstance(missing_information, list)
            and len(missing_information) > 0
        )

        valid_missing_information = isinstance(
            missing_information,
            list,
        )

        if not valid_missing_information:
            self._add_check(
                checks,
                name="research_missing_information",
                passed=False,
                severity="critical",
                details="Research missing_information must be a list.",
            )
        elif has_missing_information:
            self._add_check(
                checks,
                name="research_completeness",
                passed=False,
                severity="warning",
                details=(
                    "Research identified unresolved missing information."
                ),
            )
        else:
            self._add_check(
                checks,
                name="research_completeness",
                passed=True,
                severity="warning",
                details="No missing research information was reported.",
            )

    def _verify_sql(
        self,
        output: dict[str, Any],
        checks: list[VerificationCheck],
    ) -> None:
        required_fields = [
            "business_question",
            "sql_query",
            "explanation",
            "tables_used",
            "is_read_only",
        ]

        fields_present = all(
            field in output for field in required_fields
        )

        self._add_check(
            checks,
            name="sql_required_fields",
            passed=fields_present,
            severity="critical",
            details=(
                "Required SQL analysis fields are present."
                if fields_present
                else "SQL analysis is missing required fields."
            ),
        )

        is_read_only = output.get("is_read_only") is True

        self._add_check(
            checks,
            name="sql_read_only",
            passed=is_read_only,
            severity="critical",
            details=(
                "SQL output declares a read-only query."
                if is_read_only
                else "SQL output does not declare a read-only query."
            ),
        )

        query = output.get("sql_query")

        if not isinstance(query, str) or not query.strip():
            self._add_check(
                checks,
                name="sql_syntax",
                passed=False,
                severity="critical",
                details="SQL query is empty or invalid.",
            )
            return

        try:
            statements = sqlglot.parse(query)

            valid_statement = (
                len(statements) == 1
                and isinstance(
                    statements[0],
                    (exp.Select, exp.Union),
                )
            )

            if not valid_statement:
                raise ValueError(
                    "Only one SELECT or UNION query is accepted."
                )

            statement = statements[0]

            cte_names = {
                cte.alias_or_name.lower()
                for cte in statement.find_all(exp.CTE)
            }

            referenced_tables = {
                table.name.lower()
                for table in statement.find_all(exp.Table)
                if table.name.lower() not in cte_names
            }

            self._add_check(
                checks,
                name="sql_syntax",
                passed=True,
                severity="critical",
                details="SQL query parsed as one read-only SELECT statement.",
            )

        except (sqlglot.errors.ParseError, ValueError) as exc:
            self._add_check(
                checks,
                name="sql_syntax",
                passed=False,
                severity="critical",
                details=f"SQL query failed independent validation: {exc}",
            )
            return

        declared_tables = output.get("tables_used")

        valid_tables = (
            isinstance(declared_tables, list)
            and all(
                isinstance(table, str) and bool(table.strip())
                for table in declared_tables
            )
        )

        if not valid_tables:
            self._add_check(
                checks,
                name="sql_table_consistency",
                passed=False,
                severity="critical",
                details="Declared SQL tables are missing or invalid.",
            )
            return

        normalized_declared_tables = {
            table.strip().lower()
            for table in declared_tables
        }

        self._add_check(
            checks,
            name="sql_table_consistency",
            passed=normalized_declared_tables == referenced_tables,
            severity="critical",
            details=(
                "Declared tables match the tables referenced in SQL."
                if normalized_declared_tables == referenced_tables
                else (
                    "Declared tables do not match SQL references. "
                    f"Declared: {sorted(normalized_declared_tables)}; "
                    f"Referenced: {sorted(referenced_tables)}."
                )
            ),
        )

    def _verify_python(
        self,
        output: dict[str, Any],
        checks: list[VerificationCheck],
    ) -> None:
        required_fields = [
            "objective",
            "analysis_summary",
            "key_metrics",
            "insights",
            "limitations",
        ]

        fields_present = all(
            field in output for field in required_fields
        )

        self._add_check(
            checks,
            name="python_required_fields",
            passed=fields_present,
            severity="critical",
            details=(
                "Required Python analysis fields are present."
                if fields_present
                else "Python analysis is missing required fields."
            ),
        )

        metrics = output.get("key_metrics")

        valid_metrics = (
            isinstance(metrics, dict)
            and len(metrics) > 0
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in metrics.values()
            )
        )

        self._add_check(
            checks,
            name="python_metrics",
            passed=valid_metrics,
            severity="critical",
            details=(
                "Python metrics contain valid finite numeric values."
                if valid_metrics
                else "Python metrics are empty or contain invalid numeric values."
            ),
        )

        insights = output.get("insights")

        valid_insights = (
            isinstance(insights, list)
            and len(insights) > 0
            and all(
                isinstance(insight, str) and bool(insight.strip())
                for insight in insights
            )
        )

        self._add_check(
            checks,
            name="python_insights",
            passed=valid_insights,
            severity="critical",
            details=(
                "Python analysis contains valid insights."
                if valid_insights
                else "Python analysis insights are empty or invalid."
            ),
        )

        limitations = output.get("limitations", [])

        if not isinstance(limitations, list):
            self._add_check(
                checks,
                name="python_limitations",
                passed=False,
                severity="critical",
                details="Python limitations must be a list.",
            )
        elif limitations:
            self._add_check(
                checks,
                name="python_limitations",
                passed=False,
                severity="warning",
                details=(
                    "Python analysis reported limitations requiring review."
                ),
            )
        else:
            self._add_check(
                checks,
                name="python_limitations",
                passed=True,
                severity="warning",
                details="No Python analysis limitations were reported.",
            )
