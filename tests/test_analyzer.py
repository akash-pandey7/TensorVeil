import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyzer import analyze_data


def test_analyze_data_detects_object_columns():
    data = pd.DataFrame({
        "name": ["alice", "bob", "charlie"],
        "age": [25, 30, 35],
    })
    categorical_columns = analyze_data(data)
    assert "name" in categorical_columns


def test_analyze_data_detects_low_cardinality_numeric():
    # survived has 2 unique values out of 100 rows → clearly categorical
    data = pd.DataFrame({
        "survived": [0, 1] * 50,
        "fare": [round(i * 1.5, 2) for i in range(100)],  # 100 unique values
    })
    categorical_columns = analyze_data(data)
    assert "survived" in categorical_columns
    assert "fare" not in categorical_columns  # high cardinality → not categorical


def test_analyze_data_ignores_high_cardinality_numeric():
    data = pd.DataFrame({
        "passenger_id": list(range(100)),  # 100 unique out of 100 → not categorical
        "pclass": [1, 2, 3, 1] * 25,      # 3 unique out of 100 → categorical
    })
    categorical_columns = analyze_data(data)
    assert "passenger_id" not in categorical_columns
    assert "pclass" in categorical_columns


def test_analyze_data_empty_dataframe():
    data = pd.DataFrame()
    categorical_columns = analyze_data(data)
    assert categorical_columns == []
