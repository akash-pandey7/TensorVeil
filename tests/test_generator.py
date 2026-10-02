import numpy as np
import pandas as pd
import pytest
from src.generator import TensorVeilGenerator

@pytest.mark.slow
def test_generator_returns_row_count():
    data = pd.DataFrame({
        "age" : [25, 30, 35, 40, 45] * 20,
        "survived": [0, 1, 0, 1, 0] * 20,
        "fare" : [7.25, 71.83, 8.05, 53.1, 8.46] * 20,
    })
    categorical_column = ["survived"]
    gen = TensorVeilGenerator(epochs=2)
    gen.train(data, categorical_column)
    
    result = gen.generate(50)
    assert len(result) == 50

@pytest.mark.slow
def test_generator_returns_correct_columns():
    data = pd.DataFrame({
        "age" : [25, 30, 35, 40, 45] * 20,
        "survived": [0, 1, 0, 1, 0] * 20,
        "fare" : [7.25, 71.83, 8.05, 53.1, 8.46] * 20,
    })
    categorical_column = ["survived"]
    gen = TensorVeilGenerator(epochs=2)
    gen.train(data, categorical_column)

    result = gen.generate(50)
    assert list(result.columns) == ["age", "survived", "fare"]

def test_infer_decimal_precision_matches_each_columns_real_precision():
    assert TensorVeilGenerator._infer_decimal_precision(pd.Series([20.0, 25.0, 30.0, 45.0])) == 0
    assert TensorVeilGenerator._infer_decimal_precision(pd.Series([19.99, 5.50, 100.00])) == 2
    assert TensorVeilGenerator._infer_decimal_precision(pd.Series([1.2345, 0.8765, 1.1000])) == 4
    # Values needing more precision than the cap must clamp, not run away.
    assert TensorVeilGenerator._infer_decimal_precision(pd.Series([1.123456789, 2.987654321]), max_decimals=6) == 6
    # An empty/all-NaN column has no precision to infer from — fall back to 2.
    assert TensorVeilGenerator._infer_decimal_precision(pd.Series([np.nan, np.nan])) == 2

@pytest.mark.slow
def test_generate_rounds_each_column_to_its_own_precision_not_a_blanket_two():
    data = pd.DataFrame({
        "age": [25, 30, 35, 40, 45] * 20,        # integer-valued
        "survived": [0, 1, 0, 1, 0] * 20,
        "rate": [1.2345, 0.8765, 1.1, 0.9999, 1.05] * 20,  # needs 4 decimals
    })
    gen = TensorVeilGenerator(epochs=2)
    gen.train(data, categorical_columns=["survived"])

    assert gen._column_decimals["age"] == 0
    assert gen._column_decimals["rate"] == 4

    result = gen.generate(50)
    assert (result["age"] == result["age"].round(0)).all()
    assert (result["rate"] == result["rate"].round(4)).all()