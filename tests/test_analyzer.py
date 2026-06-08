import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyzer import analyze_data


def test_analyze_data_detects_object_columns():
    data = pd.DataFrame(
        {
            "name": ["alice", "bob", "charlie"],
            "age": [25, 30, 35],
        }
    )

    categorical_columns = analyze_data(data)

    assert "name" in categorical_columns


def test_analyze_data_detects_low_cardinality_numeric_columns():
    data = pd.DataFrame(
        {
            "survived": [0, 1, 0, 1],
            "fare": [7.25, 71.83, 8.05, 53.1],
        }
    )

    categorical_columns = analyze_data(data)

    assert "survived" in categorical_columns
    assert "fare" in categorical_columns
