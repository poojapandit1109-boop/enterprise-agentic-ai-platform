
import pandas as pd
import pytest

from src.agents.python_data_analyzer import (
    DataAnalysisError,
    analyze_dataframe,
)


def test_analyzes_numeric_columns():
    records = [
        {"spend": 100, "churned": 0},
        {"spend": 200, "churned": 1},
        {"spend": 300, "churned": 0},
        {"spend": 400, "churned": 1},
    ]

    result = analyze_dataframe(records)

    assert result["row_count"] == 4
    assert result["column_count"] == 2
    assert result["key_metrics"]["mean__spend"] == 250.0
    assert result["key_metrics"]["median__spend"] == 250.0
    assert result["key_metrics"]["min__spend"] == 100.0
    assert result["key_metrics"]["max__spend"] == 400.0
    assert result["key_metrics"]["mean__churned"] == 0.5


def test_calculates_population_standard_deviation():
    records = [
        {"value": 10},
        {"value": 20},
        {"value": 30},
    ]

    result = analyze_dataframe(records)

    assert result["key_metrics"]["std__value"] == pytest.approx(
        8.1649658093
    )


def test_detects_missing_and_duplicate_rows():
    records = [
        {"value": 10},
        {"value": None},
        {"value": 10},
    ]

    result = analyze_dataframe(records)

    assert result["missing_values"] == 1
    assert result["duplicate_rows"] == 1
    assert result["key_metrics"]["missing_values"] == 1.0
    assert result["key_metrics"]["duplicate_rows"] == 1.0


def test_calculates_pairwise_correlation():
    records = [
        {"x": 1, "y": 2},
        {"x": 2, "y": 4},
        {"x": 3, "y": 6},
    ]

    result = analyze_dataframe(records)

    assert result["key_metrics"]["correlation__x__y"] == pytest.approx(
        1.0
    )


def test_accepts_pandas_dataframe():
    dataframe = pd.DataFrame(
        [
            {"spend": 100},
            {"spend": 200},
            {"spend": 300},
        ]
    )

    result = analyze_dataframe(dataframe)

    assert result["row_count"] == 3
    assert result["key_metrics"]["mean__spend"] == 200.0


def test_rejects_empty_records():
    with pytest.raises(DataAnalysisError):
        analyze_dataframe([])


def test_rejects_records_with_invalid_items():
    with pytest.raises(DataAnalysisError):
        analyze_dataframe([{"value": 10}, "invalid record"])


def test_rejects_empty_dataframe():
    with pytest.raises(DataAnalysisError):
        analyze_dataframe(pd.DataFrame())


def test_rejects_duplicate_dataframe_columns():
    dataframe = pd.DataFrame(
        [[10, 20]],
        columns=["value", "value"],
    )

    with pytest.raises(DataAnalysisError):
        analyze_dataframe(dataframe)
