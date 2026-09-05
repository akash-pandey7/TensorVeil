import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

def test_tstr_trtr_returns_classification_scores():
    real, synthetic = sample_data()
    result = evaluate_tstr_trtr(real, synthetic, "target")
    assert set(result) == {"tstr", "trtr", "task", "target_column"}
    assert {"accuracy", "f1_weighted"} == set(result["tstr"])

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

def test_aggregate_metrics_returns_all_requested_sections():
    real, synthetic = sample_data()
    result = aggregate_metrics(real, synthetic, cast(Any, "target"))
    assert set(result) == {"statistical_similarity", "ks_test", "correlation", "utility", "dcr"}

def test_aggregate_metrics_without_target_skips_utility():
    real, synthetic = sample_data()
    result = aggregate_metrics(real, synthetic, target_column=None)
    assert set(result) == {"statistical_similarity", "ks_test", "correlation", "dcr"}
    assert "utility" not in result