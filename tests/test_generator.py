import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generator import TensorVeilGenerator


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