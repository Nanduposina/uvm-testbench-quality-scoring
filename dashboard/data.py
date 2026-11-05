"""Load and reconcile the existing pipeline artifacts for the dashboard."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.pipeline import LABELS, METRICS, SCORE_WEIGHTS, reference_category

FILES = (
    "synthetic_executions.csv", "quality_assessments.csv", "feature_importance.csv",
    "metric_statistics.csv", "metric_statistics_by_category.csv", "metric_correlations.csv",
    "evaluation.json",
)
FRACTIONS = [
    "stimulus_reuse_ratio", "corner_case_ratio", "randomization_success", "assertion_pass_ratio",
    "coverage_initial", "coverage_final", "coverage_growth", "idle_fraction",
]
COUNTS = [name for name in METRICS if name not in FRACTIONS + [
    "payload_entropy", "transactions_per_ms", "response_latency_us", "runtime_ms",
]]


class OutputDataError(ValueError):
    """Missing, malformed, stale or inconsistent pipeline output."""


@dataclass
class Artifacts:
    executions: pd.DataFrame
    assessments: pd.DataFrame
    importance: pd.DataFrame
    statistics: pd.DataFrame
    category_statistics: pd.DataFrame
    correlations: pd.DataFrame
    report: dict

    @property
    def combined(self):
        return self.executions.merge(
            self.assessments.drop(columns="reference_category"), on="run_id", how="left", validate="one_to_one"
        )


def require(condition, message):
    if not condition:
        raise OutputDataError(message)


def numeric(frame, columns, filename):
    require(set(columns).issubset(frame.columns), f"{filename}: required columns are missing.")
    require(all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns),
            f"{filename}: metric values must be numeric.")
    require(np.isfinite(frame[columns].to_numpy(dtype=float)).all(),
            f"{filename}: numeric values must be finite and non-missing.")


def rubric_evidence(row, report):
    """Reconstruct the reference rubric only; this is not a forest explanation."""
    rows = []
    for metric, term in report["reference_rubric"]["terms"].items():
        value = float(row[metric])
        normalized = value / term["scale"]
        if term.get("invert", False):
            normalized = 1 - normalized
        normalized = float(np.clip(normalized, 0, 1))
        rows.append({"Metric": metric, "Observed value": value, "Normalized term": normalized,
                     "Weight": term["weight"], "Reference score points": 100 * normalized * term["weight"]})
    return pd.DataFrame(rows)


def load_outputs(directory):
    directory = Path(directory)
    missing = [name for name in FILES if not (directory / name).is_file()]
    require(not missing, "Missing pipeline outputs: " + ", ".join(missing))
    try:
        read = lambda name, **kwargs: pd.read_csv(directory / name, float_precision="round_trip", **kwargs)
        artifacts = Artifacts(
            read(FILES[0]), read(FILES[1]), read(FILES[2]), read(FILES[3], index_col=0),
            read(FILES[4], index_col=[0, 1]), read(FILES[5], index_col=0),
            json.loads((directory / FILES[6]).read_text(encoding="utf8")),
        )
        validate_outputs(artifacts, directory)
        return artifacts
    except OutputDataError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, AssertionError) as error:
        raise OutputDataError(f"Malformed or inconsistent pipeline outputs: {error}") from error


def validate_outputs(artifacts, directory):
    data, assessments, importance, report = (
        artifacts.executions, artifacts.assessments, artifacts.importance, artifacts.report
    )
    require(isinstance(report, dict), "evaluation.json: expected a JSON object.")
    for name in ("seed", "runs", "metrics", "training_runs", "test_runs", "correct_predictions", "coverage_total_bins"):
        require(isinstance(report[name], int) and not isinstance(report[name], bool) and report[name] >= 0,
                f"evaluation.json: {name} must be a nonnegative integer.")
    require(report["coverage_total_bins"] > 0 and report["test_runs"] > 0, "Invalid coverage universe or empty holdout.")
    for name in ("agreement_accuracy", "balanced_accuracy", "macro_f1", "target_agreement"):
        require(isinstance(report[name], (int, float)) and not isinstance(report[name], bool)
                and np.isfinite(report[name]) and 0 <= report[name] <= 1,
                f"evaluation.json: {name} must be a finite fraction.")
    interval = report["agreement_accuracy_95pct_wilson"]
    require(isinstance(interval, list) and len(interval) == 2
            and all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 1 for v in interval)
            and interval[0] <= report["agreement_accuracy"] <= interval[1], "Invalid accuracy interval.")
    for name in ("split", "model_parameters", "environment"):
        require(isinstance(report[name], dict), f"evaluation.json: {name} must be an object.")
    permutation = report["permutation_importance"]
    require(isinstance(permutation, dict) and isinstance(permutation["repeats"], int)
            and permutation["repeats"] > 0 and permutation["scoring"] == "accuracy"
            and permutation["evaluation_split"] == "test", "Invalid permutation importance metadata.")
    require(report["dataset"] == "synthetic" and report["categories"] == LABELS,
            "evaluation.json: expected synthetic data and low/medium/high categories.")
    require(report["feature_columns"] == METRICS and report["metrics"] == len(METRICS),
            "evaluation.json: feature definitions do not match this project.")
    numeric(data, ["run_id", "reference_score_0_100"] + METRICS, FILES[0])
    require({"reference_category", "split"}.issubset(data.columns), f"{FILES[0]}: missing labels or split.")
    require(len(data) == report["runs"] and data.run_id.is_unique and len(data) > 0,
            f"{FILES[0]}: run count mismatch or duplicate IDs.")
    require((data.run_id >= 0).all() and (data.run_id % 1 == 0).all(), "Execution IDs must be nonnegative integers.")
    require(data.split.isin(["train", "test"]).all() and data.reference_category.isin(LABELS).all(),
            f"{FILES[0]}: invalid split or reference categories.")
    require((data[METRICS] >= 0).all().all() and data[FRACTIONS].le(1).all().all(),
            f"{FILES[0]}: metrics are outside their valid ranges.")
    require((data[COUNTS] % 1 == 0).all().all() and data.payload_entropy.le(8).all(),
            f"{FILES[0]}: invalid event counts or entropy.")
    require((data.assertion_checks > 0).all() and (data.runtime_ms > 0).all(), "Checks and runtime must be positive.")
    require((data.assertion_failures <= data.assertion_checks).all()
            and (2 * data.warning_bursts <= data.warning_count).all(), "Inconsistent assertion or warning event counts.")
    require(np.allclose(data.assertion_pass_ratio, 1 - data.assertion_failures / data.assertion_checks)
            and np.allclose(data.coverage_growth, data.coverage_final - data.coverage_initial)
            and np.allclose(data.coverage_final, data.coverage_bins_hit / report["coverage_total_bins"])
            and np.allclose(data.transactions_per_ms, data.transaction_count / data.runtime_ms),
            "Derived runtime metrics disagree with the underlying counts.")
    require(data.reference_score_0_100.between(0, 100).all()
            and (data.reference_category == data.reference_score_0_100.map(reference_category)).all(),
            "Reference scores and categories disagree.")
    require(hashlib.sha256((directory / FILES[0]).read_bytes()).hexdigest() == report["synthetic_executions_sha256"],
            "Dataset fingerprint differs from evaluation.json; regenerate the outputs together.")
    terms = report["reference_rubric"]["terms"]
    require(bool(terms) and set(terms).issubset(METRICS)
            and all(t["scale"] > 0 and 0 <= t["weight"] <= 1 for t in terms.values())
            and np.isclose(sum(t["weight"] for t in terms.values()), 1), "Invalid reference rubric.")
    rubric_scores = pd.Series(0.0, index=data.index)
    for name, term in terms.items():
        values = data[name] / term["scale"]
        if term.get("invert", False):
            values = 1 - values
        rubric_scores += 100 * values.clip(0, 1) * term["weight"]
    require(np.allclose(rubric_scores, data.reference_score_0_100), "Exported reference scores disagree with the rubric.")

    probabilities = [f"probability_{label}" for label in LABELS]
    numeric(assessments, ["run_id", "confidence", "quality_score_0_100"] + probabilities, FILES[1])
    require({"reference_category", "predicted_category"}.issubset(assessments.columns), "Missing assessment categories.")
    test = data.loc[data.split == "test"].set_index("run_id")
    require(assessments.run_id.is_unique and set(assessments.run_id) == set(test.index),
            "Assessments must contain exactly the held-out execution IDs, without training runs.")
    require(len(test) == report["test_runs"] and len(data) - len(test) == report["training_runs"], "Split sizes disagree.")
    require((assessments.reference_category.to_numpy() == test.loc[assessments.run_id].reference_category.to_numpy()).all()
            and assessments.predicted_category.isin(LABELS).all(), "Assessment labels disagree with the dataset.")
    p = assessments[probabilities].to_numpy()
    require(((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(axis=1), 1), "Invalid class probabilities.")
    require(np.allclose(assessments.confidence, p.max(axis=1)), "Confidence does not match the class probabilities.")
    selected = np.array([LABELS.index(label) for label in assessments.predicted_category])
    require(np.allclose(p[np.arange(len(p)), selected], p.max(axis=1)), "Predicted category is not a maximum-probability class.")
    require(report["score_weights"] == SCORE_WEIGHTS
            and np.allclose(assessments.quality_score_0_100, np.round(p @ [0, 50, 100], 2), atol=1e-9),
            "Model scores do not match the exported class probabilities.")
    truth, predicted = assessments.reference_category, assessments.predicted_category
    correct = int((truth == predicted).sum())
    require(correct == report["correct_predictions"] and np.isclose(correct / len(test), report["agreement_accuracy"]),
            "Agreement accuracy disagrees with individual predictions.")
    require(np.isclose(report["balanced_accuracy"], balanced_accuracy_score(truth, predicted))
            and np.isclose(report["macro_f1"], f1_score(truth, predicted, labels=LABELS, average="macro"))
            and np.array_equal(report["confusion_matrix"], confusion_matrix(truth, predicted, labels=LABELS)),
            "Evaluation metrics or confusion matrix disagree with individual predictions.")
    expected_report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
    require(report["classification_report"] == expected_report, "Classification report disagrees with predictions.")
    require(report["target_met"] == (report["agreement_accuracy"] >= report["target_agreement"]), "Target status is inconsistent.")
    for group, subset in [("all", data), ("train", data[data.split == "train"]), ("test", data[data.split == "test"])]:
        require(report["category_counts"][group] == subset.reference_category.value_counts().to_dict(), "Category counts disagree.")
    training_majority = data[data.split == "train"].reference_category.value_counts().reindex(LABELS).idxmax()
    require(report["majority_baseline"]["training_majority_category"] == training_majority
            and np.isclose(report["majority_baseline"]["test_accuracy"], (truth == training_majority).mean()),
            "Training-majority baseline is inconsistent.")

    numeric(importance, ["importance_mean", "importance_std"], FILES[2])
    require(set(importance.metric) == set(METRICS) and importance.metric.is_unique
            and (importance.importance_std >= 0).all(), "Invalid feature importance metrics or deviations.")
    pd.testing.assert_frame_equal(artifacts.statistics, data[METRICS].describe().T, check_dtype=False)
    for label in LABELS:
        expected = data.loc[data.reference_category == label, METRICS].describe().T
        expected.index.name = "metric"
        pd.testing.assert_frame_equal(artifacts.category_statistics.loc[label], expected, check_dtype=False)
    pd.testing.assert_frame_equal(artifacts.correlations, data[METRICS].corr(method="spearman"), check_dtype=False)


def filter_executions(frame, categories, id_range, basis="reference_category"):
    require(basis in {"reference_category", "predicted_category"}, "Invalid category filter basis.")
    return frame.loc[frame[basis].isin(categories) & frame.run_id.between(*id_range)].copy()
