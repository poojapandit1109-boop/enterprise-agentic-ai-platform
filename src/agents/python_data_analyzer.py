
from __future__ import annotations

import math
from typing import Any

import pandas as pd


class DataAnalysisError(ValueError):
    """Raised when a dataset cannot be analyzed."""


def analyze_dataframe(
    records: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Calculate descriptive statistics and correlations."""

    if isinstance(records, pd.DataFrame):
        dataframe = records.copy()
    else:
        if not isinstance(records, list) or not records:
            raise DataAnalysisError("Analysis dataset cannot be empty.")

        if not all(isinstance(record, dict) for record in records):
            raise DataAnalysisError(
                "Every dataset record must be a dictionary."
            )

        dataframe = pd.DataFrame.from_records(records)

    if dataframe.empty:
        raise DataAnalysisError("Analysis dataset cannot be empty.")

    if not dataframe.columns.is_unique:
        raise DataAnalysisError("Dataset columns must be unique.")

    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(dataframe[column])
    ]

    missing_values = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum())

    key_metrics: dict[str, float] = {
        "rows_analyzed": float(len(dataframe)),
        "columns_analyzed": float(len(dataframe.columns)),
        "missing_values": float(missing_values),
        "duplicate_rows": float(duplicate_rows),
    }

    insights: list[str] = [
        (
            f"The dataset contains {len(dataframe)} rows "
            f"and {len(dataframe.columns)} columns."
        )
    ]

    for column in numeric_columns:
        series = dataframe[column].dropna()

        if series.empty:
            continue

        statistics = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std(ddof=0)),
        }

        for metric_name, value in statistics.items():
            if not math.isfinite(value):
                raise DataAnalysisError(
                    f"Non-finite statistic detected for column {column}."
                )

            key_metrics[f"{metric_name}__{column}"] = value

        insights.append(
            f"{column}: mean={statistics['mean']:.2f}, "
            f"median={statistics['median']:.2f}, "
            f"minimum={statistics['min']:.2f}, "
            f"maximum={statistics['max']:.2f}."
        )

    if missing_values:
        insights.append(f"The dataset contains {missing_values} missing values.")
    else:
        insights.append("No missing values were detected.")

    if duplicate_rows:
        insights.append(f"The dataset contains {duplicate_rows} duplicate rows.")
    else:
        insights.append("No duplicate rows were detected.")

    if len(numeric_columns) >= 2:
        correlations = dataframe[numeric_columns].corr()

        for index, column_a in enumerate(numeric_columns):
            for column_b in numeric_columns[index + 1:]:
                correlation = correlations.loc[column_a, column_b]

                if pd.notna(correlation) and math.isfinite(float(correlation)):
                    key_metrics[
                        f"correlation__{column_a}__{column_b}"
                    ] = float(correlation)

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "numeric_columns": numeric_columns,
        "key_metrics": key_metrics,
        "insights": insights,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }
