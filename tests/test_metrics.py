from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    aggregate_metrics,
    calculate_dcr,
    calculate_statistical_similarity,
    compare_correlations,
    evaluate_tstr_trtr,
)


def sample_data():
    real = pd.DataFrame({
        "age": [20, 22, 25, 28, 30, 33, 36, 40, 44, 48, 52, 55, 58, 60, 63, 65, 68, 70, 73, 75],
        "income": [20, 22, 25, 30, 35, 38, 42, 45, 48, 52, 55, 58, 60, 63, 65, 68, 70, 73, 76, 80],
        "group": ["a", "a", "b", "b", "c", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b"],
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    synthetic = real.copy()
    synthetic["age"] = synthetic["age"] + 1
    synthetic["income"] = synthetic["income"] - 1
    synthetic["group"] = synthetic["group"].sample(frac=1, random_state=1).values
    return real, synthetic

def noisy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })
    synthetic = real.copy()
    synthetic["feature_a"] = synthetic["feature_a"] + rng.normal(scale=0.01, size=n)
    synthetic["feature_b"] = synthetic["feature_b"] + rng.normal(scale=0.01, size=n)
    return real, synthetic

def categorical_only_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({
        "group": rng.choice(["a", "b", "c"], size=n),
        "label": rng.choice(["x", "y"], size=n),
    })
    synthetic = real.copy()
    synthetic["group"] = synthetic["group"].sample(frac=1, random_state=seed + 1).values
    return real, synthetic

def test_statistical_similarity_contains_ks_and_tvd():
    real, synthetic = sample_data()
    result = calculate_statistical_similarity(real, synthetic)
    assert 0 <= result["mean_similarity"] <= 1
    assert {"ks_statistic", "p_value", "tvd", "similarity"} <= set(result["columns"]["age"])
    assert {"tvd", "similarity"} <= set(result["columns"]["group"])

def test_correlation_comparison_returns_difference():
    real, synthetic = sample_data()
    result = compare_correlations(real, synthetic)
    assert set(result["real"]) == {"age", "income", "target"}
    assert set(result["synthetic"]) == {"age", "income", "target"}
    assert result["mean_absolute_difference"] >= 0

def test_correlation_comparison_excludes_diagonal_from_mean():
    # Regression test: the diagonal of an absolute-difference correlation
    # matrix is always exactly 0 (a column always correlates perfectly with
    # itself in both real and synthetic data). The old implementation
    # averaged the full matrix including that diagonal, which understates
    # the true mean pairwise difference — worse the fewer numeric columns
    # there are (1/3 of a 3-column matrix is diagonal).
    rng = np.random.default_rng(0)
    n = 60
    real = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })
    synthetic = real.copy()
    synthetic["feature_a"] = synthetic["feature_a"] + rng.normal(scale=0.5, size=n)
    synthetic["feature_b"] = synthetic["feature_b"] + rng.normal(scale=0.5, size=n)

    result = compare_correlations(real, synthetic)
    difference = pd.DataFrame(result["absolute_difference"])
    off_diagonal_mask = ~np.eye(len(difference), dtype=bool)
    expected = float(difference.to_numpy()[off_diagonal_mask].mean())
    full_matrix_mean = float(difference.to_numpy().mean())

    assert result["mean_absolute_difference"] == pytest.approx(expected)
    # The two must differ here — if they're equal, the diagonal bug is back.
    assert result["mean_absolute_difference"] > full_matrix_mean

def test_correlation_comparison_single_column_has_no_pairwise_difference():
    # A single numeric column has no pair to compare against, so this must
    # return 0.0 rather than crashing on an empty-array mean (NaN/RuntimeWarning).
    real, synthetic = sample_data()
    result = compare_correlations(real[["age"]], synthetic[["age"]])
    assert result["mean_absolute_difference"] == 0.0

def test_tstr_trtr_returns_classification_scores():
    real, synthetic = sample_data()
    result = evaluate_tstr_trtr(real, synthetic, "target")
    assert set(result) == {"tstr", "trtr", "task", "target_column"}
    assert {"accuracy", "f1_weighted"} == set(result["tstr"])

def test_tstr_trtr_handles_rare_classes_without_crashing():
    # Regression test: sklearn's stratify requires every class to have at
    # least 2 members (one for train, one for test). A classification target
    # with a class that appears only once used to raise ValueError straight
    # out of train_test_split, crashing the whole metrics pipeline over one
    # underrepresented class rather than falling back to an unstratified
    # split for that case.
    real = pd.DataFrame({
        "feature": list(range(50)),
        "label": ["common"] * 48 + ["rare1", "rare2"],
    })
    synthetic = real.copy()
    result = evaluate_tstr_trtr(real, synthetic, "label", task="classification")
    assert set(result) == {"tstr", "trtr", "task", "target_column"}

def test_trtr_does_not_memorize_training_data():
    # Regression test for the fixed train/test split bug: TRTR must be
    # evaluated on a real held-out split, not on the same rows it trained
    # on. With a pure-noise target, a model scored on its own training
    # data would memorize it and land near 1.0 accuracy; scored on a
    # genuine held-out split, accuracy should sit close to chance (0.5).
    real, synthetic = noisy_data()
    result = evaluate_tstr_trtr(real, synthetic, "target")
    trtr_accuracy = result["trtr"]["accuracy"]
    assert trtr_accuracy < 0.85, (
        f"TRTR accuracy of {trtr_accuracy:.2f} on a noise target suggests "
        "it is being evaluated on its own training data rather than a "
        "held-out split."
    )

def test_dcr_returns_one_distance_per_synthetic_row():
    real, synthetic = sample_data()
    result = calculate_dcr(real, synthetic)
    assert len(result["distances"]) == len(synthetic)
    assert result["minimum"] <= result["median"] <= max(result["distances"])
    assert "percentile_5" in result

def test_dcr_ratio_flags_memorized_synthetic_data():
    # Regression test: DCR alone has no reference point (it's in
    # standardized, one-hot-expanded units, not comparable across datasets).
    # Comparing it to a real-to-real baseline turns it into an interpretable
    # ratio. Exact copies of real data should score near-zero.
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    memorized = real.copy()
    result = calculate_dcr(real, memorized)
    assert result["real_to_real_median"] is not None
    assert result["ratio_to_real_baseline"] == pytest.approx(0.0, abs=1e-6)

def test_dcr_ratio_near_one_for_genuinely_novel_synthetic_data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    novel = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    result = calculate_dcr(real, novel)
    # Independent draws from the same distribution should land close to
    # the real-to-real baseline — not exactly 1.0, but in that neighborhood.
    assert 0.7 < result["ratio_to_real_baseline"] < 1.5

def test_dcr_baseline_is_none_with_a_single_real_row():
    # A single real row has no "other" real row to measure distance
    # against, so the baseline (and ratio) must degrade to None rather
    # than raising or producing a bogus NearestNeighbors result.
    real, synthetic = sample_data()
    result = calculate_dcr(real.iloc[:1], synthetic.iloc[:1])
    assert result["real_to_real_median"] is None
    assert result["ratio_to_real_baseline"] is None

def test_aggregate_metrics_returns_all_requested_sections():
    real, synthetic = sample_data()
    result = aggregate_metrics(real, synthetic, cast(Any, "target"))
    assert set(result) == {"statistical_similarity", "ks_test", "correlation", "utility", "dcr"}

def test_aggregate_metrics_without_target_skips_utility():
    real, synthetic = sample_data()
    result = aggregate_metrics(real, synthetic, target_column=None)
    assert set(result) == {"statistical_similarity", "ks_test", "correlation", "dcr"}
    assert "utility" not in result

def test_aggregate_metrics_all_categorical_skips_correlation_without_crashing():
    # Regression test: compare_correlations used to raise ValueError when
    # there were no numeric columns, and aggregate_metrics called it
    # unconditionally, crashing the whole metrics pipeline on any
    # all-categorical dataset.
    real, synthetic = categorical_only_data()
    result = aggregate_metrics(real, synthetic, cast(Any, "label"), task="classification")
    assert result["correlation"] is None
    assert set(result) == {"statistical_similarity", "ks_test", "correlation", "utility", "dcr"}