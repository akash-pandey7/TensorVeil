import pandas as pd

def analyze_data(df):
    if len(df) == 0:
        return []
    categorical_columns = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            categorical_columns.append(col)
        elif df[col].nunique() < 20 and df[col].nunique() < len(df) * 0.5:
            categorical_columns.append(col)
    return categorical_columns