def analyze_data(df):
    categorical_column = []
    for col in df.columns:
        if df[col].dtype == "object":
            categorical_column.append(col)
        elif df[col].nunique() < 20:
            categorical_column.append(col)
    return categorical_column