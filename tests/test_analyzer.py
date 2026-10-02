import pandas as pd
from src.analyzer import analyze_data

def test_analyze_data_detects_object_columns():
    data = pd.DataFrame({
        "name": ["alice", "bob", "charlie"],
        "age": [25, 30, 35],
    })
    categorical_columns = analyze_data(data)
    assert "name" in categorical_columns

def test_analyze_data_detects_string_columns_regardless_of_backing_dtype():
    legacy_object = pd.DataFrame({"name": pd.array(["alice", "bob", "charlie"], dtype="object")})
    assert "name" in analyze_data(legacy_object)

    native_str = pd.DataFrame({"name": pd.array(["alice", "bob", "charlie"], dtype="str")})
    assert "name" in analyze_data(native_str)

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

def test_analyze_data_zero_rows_with_numeric_dtype_does_not_crash():
    data = pd.DataFrame({"age": pd.array([], dtype="int64"), "name": pd.array([], dtype="object")})
    categorical_columns = analyze_data(data)
    assert categorical_columns == []

def test_analyze_data_detects_low_cardinality_numeric_on_a_small_dataset():
    data = pd.DataFrame({
        "category_code": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 15),  # 10 unique / 150 rows
        "continuous_measurement": [round(i * 0.37, 3) for i in range(150)],  # 150 unique
    })
    categorical_columns = analyze_data(data)
    assert "category_code" in categorical_columns
    assert "continuous_measurement" not in categorical_columns

def test_analyze_data_still_excludes_near_unique_numeric_on_a_small_dataset():
    data = pd.DataFrame({
        "measurement": [round(i * 1.1, 2) for i in range(10)],  # 10 unique / 10 rows
    })
    categorical_columns = analyze_data(data)
    assert "measurement" not in categorical_columns