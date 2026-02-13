def analyze_data(df):
    categorical_columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            categorical_columns.append(col)
        elif df[col].nunique() < 20:
            categorical_columns.append(col)
    return categorical_columns