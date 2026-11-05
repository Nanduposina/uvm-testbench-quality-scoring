import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pipeline
from pipeline import LABELS, METRICS, generate, run


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full_pipeline")
    report = run(out)
    return out, report


@pytest.mark.parametrize("seed", [0, 42, 12345, 2**32 - 1])
def test_generated_metric_invariants(seed):
    data = generate(seed=seed)
    assert len(data) == 550
    assert len(METRICS) == 24 and len(set(METRICS)) == 24
    assert data.run_id.is_unique
    assert np.isfinite(data[METRICS]).all().all()
    assert (data[METRICS] >= 0).all().all()
    for metric in (
        "stimulus_reuse_ratio", "corner_case_ratio", "randomization_success",
        "assertion_pass_ratio", "coverage_initial", "coverage_final", "coverage_growth", "idle_fraction",
    ):
        assert data[metric].between(0, 1).all()
    counts = [m for m in METRICS if m not in {
        "payload_entropy", "stimulus_reuse_ratio", "corner_case_ratio", "randomization_success",
        "assertion_pass_ratio", "coverage_initial", "coverage_final", "coverage_growth",
        "idle_fraction", "transactions_per_ms", "response_latency_us", "runtime_ms",
    }]
    assert (data[counts] == np.floor(data[counts])).all().all()
    assert (data.assertion_failures <= data.assertion_checks).all()
    np.testing.assert_allclose(
        data.assertion_pass_ratio, 1 - data.assertion_failures / data.assertion_checks
    )
    np.testing.assert_allclose(data.coverage_final - data.coverage_initial, data.coverage_growth)
    np.testing.assert_allclose(data.coverage_bins_hit / 400, data.coverage_final)
    assert (2 * data.warning_bursts <= data.warning_count).all()
    assert (data.unique_addresses <= data.transaction_count).all()
    assert (data.component_pairs <= data.component_messages).all()
    assert (data.runtime_ms > 0).all()
    np.testing.assert_allclose(data.transactions_per_ms, data.transaction_count / data.runtime_ms)
    assert data.payload_entropy.between(0, 8).all()
    assert data.reference_score_0_100.between(0, 100).all()
    expected = np.select(
        [data.reference_score_0_100 < 42, data.reference_score_0_100 < 69],
        ["low", "medium"], default="high",
    )
    assert (data.reference_category == expected).all()


def test_reproducible_generation_is_independent_of_global_rng():
    first = generate(40, 42)
    np.random.seed(123)
    pd.testing.assert_frame_equal(first, generate(40, 42))
    assert not first.equals(generate(40, 43))
    assert len(generate(30)) == 30


@pytest.mark.parametrize("n", [0, 29, -10, 30.5, True])
def test_invalid_run_count(n):
    with pytest.raises(ValueError, match="At least 30"):
        generate(n)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="Seed must"):
        generate(seed=seed)


@pytest.mark.parametrize("score,label", [
    (0, "low"), (np.nextafter(42.0, 0), "low"), (42, "medium"),
    (np.nextafter(69.0, 0), "medium"), (69, "high"), (100, "high"),
])
def test_reference_threshold_boundaries(score, label):
    assert pipeline.reference_category(score) == label


def test_reference_formula_and_clipping():
    metrics = dict(coverage_growth=0.35, corner_case_ratio=0.35, idle_fraction=0.5,
                   randomization_success=0.5, component_pairs=11, assertion_checks=425)
    assert pipeline.reference_score(metrics) == pytest.approx(50)
    metrics.update(coverage_growth=0, corner_case_ratio=0, idle_fraction=1,
                   randomization_success=0, component_pairs=0, assertion_checks=0)
    assert pipeline.reference_score(metrics) == 0
    # Saturation is per term, including coverage growth above its nominal scale.
    metrics.update(coverage_growth=1, corner_case_ratio=1, idle_fraction=0,
                   randomization_success=1, component_pairs=100, assertion_checks=2000)
    assert pipeline.reference_score(metrics) == pytest.approx(100)
    metrics.update(coverage_growth=0.35, corner_case_ratio=0, idle_fraction=1,
                   randomization_success=0, component_pairs=0, assertion_checks=0)
    assert pipeline.reference_score(metrics) == pytest.approx(14)


@pytest.mark.parametrize("order", [
    ["high", "low", "medium"], ["low", "medium", "high"], ["medium", "high", "low"],
])
def test_probability_score_alignment_and_argmax_category(order):
    class StubModel:
        classes_ = np.array(order)

        def predict_proba(self, features):
            # Include a case where the expected score crosses a rubric threshold:
            # low is the most probable class, but score 42 is not a low rubric score.
            by_label = {"low": [1, 0, 0, 0.4], "medium": [0, 1, 0, 0.36], "high": [0, 0, 1, 0.24]}
            return np.array([by_label[label] for label in self.classes_]).T

        def predict(self, features):
            return self.classes_[self.predict_proba(features).argmax(axis=1)]

    data = generate(30).iloc[:4]
    result = pipeline.assess(StubModel(), data)
    np.testing.assert_allclose(result.quality_score_0_100, [0, 50, 100, 42])
    np.testing.assert_allclose(result.confidence, [1, 1, 1, 0.4])
    assert result.predicted_category.tolist() == ["low", "medium", "high", "low"]


def test_training_and_importance_are_isolated_from_holdout(monkeypatch, tmp_path):
    fitted = []
    predictions = []
    importance_calls = []
    real_importance = pipeline.permutation_importance

    class RecordingForest(pipeline.RandomForestClassifier):
        def fit(self, features, target, **kwargs):
            assert list(features.columns) == METRICS
            assert target.name == "reference_category"
            fitted.append(set(features.index))
            return super().fit(features, target, **kwargs)

        def predict(self, features):
            assert list(features.columns) == METRICS
            assert not fitted[0].intersection(features.index)
            predictions.append(set(features.index))
            return super().predict(features)

        def predict_proba(self, features):
            # RandomForest.predict also delegates to predict_proba.
            if isinstance(features, pd.DataFrame):
                assert list(features.columns) == METRICS
                assert not fitted[0].intersection(features.index)
            return super().predict_proba(features)

    def record_importance(model, features, target, **kwargs):
        assert list(features.columns) == METRICS
        assert not fitted[0].intersection(features.index)
        assert kwargs["scoring"] == "accuracy"
        importance_calls.append(set(features.index))
        return real_importance(model, features, target, **kwargs)

    monkeypatch.setattr(pipeline, "RandomForestClassifier", RecordingForest)
    monkeypatch.setattr(pipeline, "permutation_importance", record_importance)
    report = run(tmp_path, 60)
    data = pd.read_csv(tmp_path / "synthetic_executions.csv")
    test_ids = set(data.loc[data.split == "test", "run_id"])
    assert len(fitted) == 1
    assert fitted[0] | test_ids == set(range(60))
    assert fitted[0].isdisjoint(test_ids)
    assert all(ids == test_ids for ids in predictions + importance_calls)
    assert report["training_runs"] == 48 and report["test_runs"] == 12
    assert set(METRICS).isdisjoint({"run_id", "reference_score_0_100", "reference_category", "split"})


def test_outputs_reconcile_with_predictions(full_run):
    out, report = full_run
    data = pd.read_csv(out / "synthetic_executions.csv", float_precision="round_trip")
    results = pd.read_csv(out / "quality_assessments.csv", float_precision="round_trip")
    saved_report = json.loads((out / "evaluation.json").read_text(encoding="utf8"))
    assert report == saved_report
    assert report["runs"] == len(data) == 550
    assert report["metrics"] == len(METRICS)
    assert report["training_runs"] == 440
    assert report["test_runs"] == len(results) == 110
    assert results.run_id.is_unique
    test = data.loc[data.split == "test"].set_index("run_id")
    assert set(results.run_id) == set(test.index)
    assert (results.reference_category.to_numpy() == test.loc[results.run_id].reference_category.to_numpy()).all()
    matrix = confusion_matrix(results.reference_category, results.predicted_category, labels=LABELS)
    np.testing.assert_array_equal(report["confusion_matrix"], matrix)
    assert matrix.sum() == len(results)
    correct = int((results.reference_category == results.predicted_category).sum())
    assert report["correct_predictions"] == correct
    assert report["agreement_accuracy"] == correct / len(results)
    assert report["agreement_accuracy"] == accuracy_score(results.reference_category, results.predicted_category)
    assert report["target_met"] == (correct / len(results) >= 0.90)
    independently_computed = classification_report(
        results.reference_category, results.predicted_category, labels=LABELS,
        output_dict=True, zero_division=0,
    )
    assert report["classification_report"] == independently_computed
    assert report["macro_f1"] == pytest.approx(independently_computed["macro avg"]["f1-score"])
    assert report["balanced_accuracy"] == pytest.approx(np.mean([
        independently_computed[label]["recall"] for label in LABELS
    ]))
    for split in ("train", "test"):
        subset = data.loc[data.split == split]
        assert set(subset.reference_category) == set(LABELS)
        assert report["category_counts"][split] == subset.reference_category.value_counts().to_dict()
    assert report["category_counts"]["all"] == data.reference_category.value_counts().to_dict()
    majority = data.loc[data.split == "train"].reference_category.value_counts().idxmax()
    assert report["majority_baseline"]["training_majority_category"] == majority
    assert report["majority_baseline"]["test_accuracy"] == (results.reference_category == majority).mean()
    assert report["synthetic_executions_sha256"] == hashlib.sha256((out / "synthetic_executions.csv").read_bytes()).hexdigest()
    probabilities = results[[f"probability_{label}" for label in LABELS]].to_numpy()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(results.confidence, probabilities.max(axis=1))
    np.testing.assert_allclose(results.quality_score_0_100, np.round(probabilities @ [0, 50, 100], 2))
    assert results.quality_score_0_100.between(0, 100).all()
    # A predicted category must achieve the maximum probability (ties are valid).
    for row in results.itertuples():
        assert getattr(row, f"probability_{row.predicted_category}") == row.confidence


def test_statistics_and_importance_outputs(full_run):
    out, report = full_run
    data = pd.read_csv(out / "synthetic_executions.csv", float_precision="round_trip")
    statistics = pd.read_csv(out / "metric_statistics.csv", index_col=0)
    pd.testing.assert_frame_equal(statistics, data[METRICS].describe().T, check_dtype=False)
    grouped = pd.read_csv(out / "metric_statistics_by_category.csv", index_col=[0, 1])
    assert len(grouped) == 3 * len(METRICS)
    for label in LABELS:
        expected = data.loc[data.reference_category == label, METRICS].describe().T
        expected.index.name = "metric"
        pd.testing.assert_frame_equal(grouped.loc[label], expected, check_dtype=False)
    correlations = pd.read_csv(out / "metric_correlations.csv", index_col=0)
    pd.testing.assert_frame_equal(correlations, data[METRICS].corr(method="spearman"))
    importance = pd.read_csv(out / "feature_importance.csv")
    assert set(importance.metric) == set(METRICS) and len(importance) == len(METRICS)
    assert np.isfinite(importance[["importance_mean", "importance_std"]]).all().all()
    assert (importance.importance_std >= 0).all()
    assert importance.importance_mean.is_monotonic_decreasing
    assert report["permutation_importance"]["evaluation_split"] == "test"
    assert report["permutation_importance"]["repeats"] == 10


def test_serialized_outputs_use_lf_for_portable_fingerprints(full_run):
    out, _ = full_run
    for path in out.iterdir():
        assert b"\r\n" not in path.read_bytes(), path.name


def test_complete_pipeline_reproduces_every_output(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run(first, 60, 43)
    run(second, 60, 43)
    assert {p.name for p in first.iterdir()} == {p.name for p in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_saved_dataset_reproduces_model_and_permutation_importance(full_run):
    out, report = full_run
    data = pd.read_csv(out / "synthetic_executions.csv", float_precision="round_trip")
    train, test = pipeline.train_test_split(
        data, test_size=report["split"]["test_fraction"], stratify=data.reference_category,
        random_state=report["split"]["seed"],
    )
    assert (train.split == "train").all() and (test.split == "test").all()
    test = test.sort_values("run_id")
    model = pipeline.RandomForestClassifier(**report["model_parameters"])
    model.fit(train[METRICS], train.reference_category)
    saved = pd.read_csv(out / "quality_assessments.csv", float_precision="round_trip")
    np.testing.assert_array_equal(saved.predicted_category, model.predict(test[METRICS]))
    probabilities = model.predict_proba(test[METRICS])
    for i, label in enumerate(model.classes_):
        np.testing.assert_allclose(saved[f"probability_{label}"], probabilities[:, i], atol=1e-15)
    recomputed = pipeline.permutation_importance(
        model, test[METRICS], test.reference_category, scoring="accuracy",
        n_repeats=report["permutation_importance"]["repeats"], random_state=report["seed"], n_jobs=1,
    )
    importance = pd.read_csv(out / "feature_importance.csv").set_index("metric").loc[METRICS]
    np.testing.assert_allclose(importance.importance_mean, recomputed.importances_mean, atol=1e-15)
    np.testing.assert_allclose(importance.importance_std, recomputed.importances_std, atol=1e-15)


def test_insufficient_class_support_fails_before_writing(monkeypatch, tmp_path):
    data = generate(30)
    data["reference_category"] = "medium"
    data.loc[0, "reference_category"] = "high"
    monkeypatch.setattr(pipeline, "generate", lambda *args: data)
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="at least two runs per reference category"):
        run(out, 30)
    assert not out.exists()


def test_rare_category_missing_from_holdout_is_rejected(monkeypatch, tmp_path):
    data = generate(30)
    data["reference_category"] = ["low"] * 2 + ["medium"] * 26 + ["high"] * 2
    monkeypatch.setattr(pipeline, "generate", lambda *args: data)
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="did not retain all three reference categories"):
        run(out, 30)
    assert not out.exists()


@pytest.mark.parametrize("correct,total,expected", [
    (0, 10, [0, 0.2775327998628892]),
    (5, 10, [0.236593090512564, 0.763406909487436]),
    (10, 10, [0.7224672001371108, 1]),
])
def test_wilson_interval_known_values(correct, total, expected):
    np.testing.assert_allclose(pipeline.wilson_interval(correct, total), expected, atol=1e-15)


def test_cli_validates_inputs_without_creating_output(tmp_path):
    out = tmp_path / "invalid"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "pipeline.py"), "--runs", "29", "--output", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "At least 30 runs" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()
