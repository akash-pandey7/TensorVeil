import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def calculate_ks_statistic(real_data, synthetic_data):
    """
    Calculate the Kolmogorov-Smirnov statistic between real and synthetic data.

    Parameters:
    - real_data: A 1D array-like of real data.
    - synthetic_data: A 1D array-like of synthetic data.

    Returns:
    - ks_statistic: The KS statistic value.
    - p_value: The p-value associated with the KS test.
    """
    result = st.ks_2samp(real_data, synthetic_data)
    return float(result.statistic), float(result.pvalue)

def _as_dataframes(real_data, synthetic_data):
    real = pd.DataFrame(real_data).reset_index(drop=True)
    synthetic = pd.DataFrame(synthetic_data).reset_index(drop=True)
    if list(real.columns) != list(synthetic.columns):
        raise ValueError("real_data and synthetic_data must have the same columns")
    if real.empty or synthetic.empty:
        raise ValueError("real_data and synthetic_data must not be empty")
    return real, synthetic

def _total_variation_distance(real_values, synthetic_values):
    real_counts = pd.Series(real_values).value_counts(normalize=True)
    synthetic_counts = pd.Series(synthetic_values).value_counts(normalize=True)
    categories = real_counts.index.union(synthetic_counts.index)
    return float(0.5 * (real_counts.reindex(categories, fill_value=0) - synthetic_counts.reindex(categories, fill_value=0)).abs().sum())

def calculate_statistical_similarity(real_data, synthetic_data):
    """Compare each column with KS (numeric) and total variation distance."""
    real, synthetic = _as_dataframes(real_data, synthetic_data)
    columns = {}
    for column in real.columns:
        if pd.api.types.is_numeric_dtype(real[column]) and pd.api.types.is_numeric_dtype(synthetic[column]):
            ks_statistic, p_value = calculate_ks_statistic(real[column], synthetic[column])
            bins = np.histogram_bin_edges(pd.concat([real[column], synthetic[column]]), bins=10)
            tvd = _total_variation_distance(
                pd.cut(real[column], bins=bins, include_lowest=True),
                pd.cut(synthetic[column], bins=bins, include_lowest=True),
            )
            columns[column] = {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "tvd": tvd,
                "similarity": float(1 - np.mean([ks_statistic, tvd])),
            }
        else:
            tvd = _total_variation_distance(real[column], synthetic[column])
            columns[column] = {"tvd": tvd, "similarity": float(1 - tvd)}
    similarities = [result["similarity"] for result in columns.values()]
    return {"columns": columns, "mean_similarity": float(np.mean(similarities))}

def compare_correlations(real_data, synthetic_data, columns=None):
    """Compare Pearson correlation matrices for shared numeric columns."""
    real, synthetic = _as_dataframes(real_data, synthetic_data)
    selected = columns or list(real.select_dtypes(include=np.number).columns)
    if not selected:
        raise ValueError("at least one numeric column is required")
    real_corr = real[selected].corr()
    synthetic_corr = synthetic[selected].corr()
    difference = (real_corr - synthetic_corr).abs()
    # A column's correlation with itself is always 1 in both matrices, so the
    # diagonal is always exactly 0 difference. Averaging it in alongside the
    # real off-diagonal pairs artificially deflates the score — more so the
    # fewer numeric columns there are (1/3 of a 3-column matrix is diagonal).
    # Compare each pair once, upper triangle only, diagonal excluded.
    if len(selected) > 1:
        upper_triangle = np.triu(np.ones(difference.shape, dtype=bool), k=1)
        mean_absolute_difference = float(difference.to_numpy()[upper_triangle].mean())
    else:
        # A single numeric column has no pairwise correlation to compare.
        mean_absolute_difference = 0.0
    return {
        "real": real_corr.to_dict(),
        "synthetic": synthetic_corr.to_dict(),
        "absolute_difference": difference.to_dict(),
        "mean_absolute_difference": mean_absolute_difference,
    }

def evaluate_tstr_trtr(real_data, synthetic_data, target_column, task="classification", random_state=42):
    """Evaluate train-on-synthetic/test-on-real against train/test-real baselines."""
    real, synthetic = _as_dataframes(real_data, synthetic_data)
    if target_column not in real.columns:
        raise ValueError(f"unknown target column: {target_column}")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    if task == "classification":
        try:
            real_train, real_test = train_test_split(real, test_size=0.2, stratify=real[target_column], random_state=random_state)
        except ValueError:
            # stratify requires every class to have at least 2 members (one
            # for train, one for test). A rare class with only 1 row raises
            # here — fall back to an unstratified split rather than crashing
            # the whole metrics pipeline over one underrepresented class.
            real_train, real_test = train_test_split(real, test_size=0.2, random_state=random_state)
    else:
        real_train, real_test = train_test_split(real, test_size=0.2, random_state=random_state)

    features = [column for column in real.columns if column != target_column]
    numeric = real[features].select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("at least one feature column is required")
    estimator = (RandomForestClassifier(n_estimators=100, random_state=random_state)
                if task == "classification" else
                RandomForestRegressor(n_estimators=100, random_state=random_state))

    def score(train):
        model = Pipeline([("features", ColumnTransformer(transformers)), ("model", estimator)])
        model.fit(train[features], train[target_column])
        predictions = model.predict(real_test[features])
        if task == "classification":
            return {"accuracy": float(accuracy_score(real_test[target_column], predictions)),
                    "f1_weighted": float(f1_score(real_test[target_column], predictions, average="weighted"))}
        return {"r2": float(r2_score(real_test[target_column], predictions)),
                "mae": float(mean_absolute_error(real_test[target_column], predictions))}

    return {"tstr": score(synthetic), "trtr": score(real_train), "task": task, "target_column": target_column}

def calculate_dcr(real_data, synthetic_data):
    """Calculate the distance-to-closest-record for every synthetic row,
    plus a real-to-real baseline so the number means something on its own.

    DCR alone is in standardized, one-hot-expanded Euclidean units — it isn't
    comparable across datasets or column sets, and a value like "2.3" gives
    no signal about whether synthetic rows are suspiciously close to real
    ones. Comparing it against how close real rows naturally sit to each
    other (excluding self-matches) turns it into an interpretable ratio: if
    synthetic DCR is close to the real-to-real baseline, synthetic rows are
    about as far from real data as real rows are from each other — no
    memorization signal. If it's much smaller, synthetic rows are landing
    suspiciously close to specific real records.
    """
    real, synthetic = _as_dataframes(real_data, synthetic_data)
    combined = pd.concat([real, synthetic], ignore_index=True)
    encoded = pd.get_dummies(combined, dummy_na=True)
    scaled = StandardScaler().fit_transform(encoded)
    real_values = scaled[:len(real)]
    synthetic_values = scaled[len(real):]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real_values)
    synthetic_distances = nn.kneighbors(synthetic_values, return_distance=True)[0]
    synthetic_closest = synthetic_distances.min(axis=1)

    real_to_real_median = None
    ratio_to_real_baseline = None
    if len(real) > 1:
        nn_real = NearestNeighbors(n_neighbors=2)
        nn_real.fit(real_values)
        real_distances = nn_real.kneighbors(real_values, return_distance=True)[0]
        # Column 0 is each point matched to itself (distance ~0); column 1 is
        # the nearest *other* real row — the actual real-to-real baseline.
        real_to_real_closest = real_distances[:, 1]
        real_to_real_median = float(np.median(real_to_real_closest))
        if real_to_real_median > 0:
            ratio_to_real_baseline = float(np.median(synthetic_closest) / real_to_real_median)

    return {
        "distances": synthetic_closest.tolist(),
        "minimum": float(synthetic_closest.min()),
        "mean": float(synthetic_closest.mean()),
        "median": float(np.median(synthetic_closest)),
        "percentile_5": float(np.percentile(synthetic_closest, 5)),
        "real_to_real_median": real_to_real_median,
        "ratio_to_real_baseline": ratio_to_real_baseline,
    }

def aggregate_metrics(real_data, synthetic_data, target_column: None, task="classification"):
    """Run all available metrics and return one dictionary."""
    target_column = target_column if target_column in real_data.columns else None
    statistical = calculate_statistical_similarity(real_data, synthetic_data)
    ks = {column: {"ks_statistic": column_result["ks_statistic"], "p_value": column_result["p_value"]}
        for column, column_result in statistical["columns"].items() if "ks_statistic" in column_result}
    numeric_columns = list(real_data.select_dtypes(include=np.number).columns)
    correlation = compare_correlations(real_data, synthetic_data) if numeric_columns else None
    result = {"statistical_similarity": statistical, "ks_test": ks, "correlation": correlation, "dcr": calculate_dcr(real_data, synthetic_data)}
    if target_column is not None:
        result["utility"] = evaluate_tstr_trtr(real_data, synthetic_data, target_column, task)
    return result

# Short aliases for callers that prefer metric names as verbs.
calculate_correlation_comparison = compare_correlations
calculate_utility = evaluate_tstr_trtr