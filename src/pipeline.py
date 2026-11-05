"""Synthetic run-level UVM metric summaries; no RTL simulator is invoked."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

METRICS = [
    "stimulus_types", "unique_addresses", "payload_entropy", "stimulus_reuse_ratio",
    "corner_case_ratio", "randomization_success", "assertion_checks",
    "assertion_pass_ratio", "assertion_failures", "coverage_initial", "coverage_final",
    "coverage_growth", "coverage_bins_hit", "warning_count", "warning_bursts",
    "error_count", "idle_fraction", "transaction_count", "transactions_per_ms",
    "component_pairs", "component_messages", "response_latency_us", "protocol_retries",
    "runtime_ms",
]
LABELS = ["low", "medium", "high"]
SCORE_WEIGHTS = {"low": 0.0, "medium": 50.0, "high": 100.0}
COVERAGE_TOTAL_BINS = 400
PERMUTATION_REPEATS = 10
TARGET_AGREEMENT = 0.90
# Each term is clipped to [0, 1] before applying its weight.
RUBRIC = {
    "coverage_growth": {"weight": 0.28, "scale": 0.7},
    "corner_case_ratio": {"weight": 0.17, "scale": 0.7},
    "idle_fraction": {"weight": 0.17, "scale": 1.0, "invert": True},
    "randomization_success": {"weight": 0.12, "scale": 1.0},
    "component_pairs": {"weight": 0.13, "scale": 22.0},
    "assertion_checks": {"weight": 0.13, "scale": 850.0},
}
DEPENDENCIES = (
    "numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl",
    "narwhals", "cloudpickle", "python-dateutil", "pytz", "tzdata", "six",
    "pytest", "iniconfig", "packaging", "pluggy", "pygments", "colorama",
)


def reference_score(metrics):
    """Return the synthetic rubric score, not a human quality assessment."""
    score = 0.0
    for metric, term in RUBRIC.items():
        value = metrics[metric] / term["scale"]
        if term.get("invert", False):
            value = 1.0 - value
        score += term["weight"] * float(np.clip(value, 0.0, 1.0))
    return 100.0 * score


def reference_category(score):
    return "low" if score < 42.0 else "medium" if score < 69.0 else "high"


def generate(n=550, seed=42):
    """Generate independent synthetic summaries with consistent derived metrics."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 30:
        raise ValueError("At least 30 runs required (integer --runs)")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Shared latent quality intentionally correlates many features. It is not
        # exported or passed to the classifier; labels use only the rubric below.
        latent = float(rng.beta(2, 2))
        runtime = float(rng.uniform(100, 1000))
        transactions = int(rng.integers(80, 300) + latent * 900)
        initial_bins = int(rng.integers(20, 121))
        initial = initial_bins / COVERAGE_TOTAL_BINS
        raw_growth = float(np.clip(0.05 + 0.65 * latent + rng.normal(0, 0.04), 0, 1 - initial))
        final_bins = int(round((initial + raw_growth) * COVERAGE_TOTAL_BINS))
        final = final_bins / COVERAGE_TOTAL_BINS
        warnings = int(rng.poisson(3 + 17 * (1 - latent)))
        row = dict(
            run_id=i,
            stimulus_types=int(2 + latent * 16 + rng.integers(0, 3)),
            unique_addresses=int(10 + latent * 500 + rng.integers(0, 20)),
            payload_entropy=float(np.clip(1 + latent * 6 + rng.normal(0, 0.15), 0, 8)),
            stimulus_reuse_ratio=float(np.clip(0.85 - 0.7 * latent + rng.normal(0, 0.04), 0, 1)),
            corner_case_ratio=float(np.clip(0.05 + 0.6 * latent + rng.normal(0, 0.04), 0, 1)),
            randomization_success=float(np.clip(0.8 + 0.2 * latent + rng.normal(0, 0.015), 0, 1)),
            assertion_checks=int(20 + latent * 800 + rng.integers(0, 25)),
        )
        pass_probability = float(np.clip(0.85 + 0.15 * latent + rng.normal(0, 0.01), 0, 1))
        failures = int(rng.binomial(row["assertion_checks"], 1 - pass_probability))
        row.update(
            assertion_pass_ratio=1 - failures / row["assertion_checks"],
            assertion_failures=failures,
            coverage_initial=initial,
            coverage_final=final,
            coverage_growth=final - initial,
            coverage_bins_hit=final_bins,
            warning_count=warnings,
            # Disjoint groups containing at least two warnings each.
            warning_bursts=int(min(rng.poisson(1 + 5 * (1 - latent)), warnings // 2)),
            error_count=int(rng.poisson(2 * (1 - latent))),
            idle_fraction=float(np.clip(0.75 - 0.65 * latent + rng.normal(0, 0.04), 0, 1)),
            transaction_count=transactions,
            transactions_per_ms=transactions / runtime,
            component_pairs=int(1 + latent * 20 + rng.integers(0, 3)),
            component_messages=int(10 + latent * 500 + rng.integers(0, 20)),
            response_latency_us=float(max(0.1, 30 - 25 * latent + rng.normal(0, 2))),
            protocol_retries=int(rng.poisson(2 + 12 * (1 - latent))),
            runtime_ms=runtime,
        )
        row["reference_score_0_100"] = reference_score(row)
        row["reference_category"] = reference_category(row["reference_score_0_100"])
        rows.append(row)
    return pd.DataFrame(rows)


def assess(model, test):
    """Assess held-out runs, aligning probability columns by class name."""
    probabilities = model.predict_proba(test[METRICS])
    result = test[["run_id", "reference_category"]].copy()
    result["predicted_category"] = model.predict(test[METRICS])
    result["confidence"] = probabilities.max(axis=1)
    for label in LABELS:
        result[f"probability_{label}"] = probabilities[:, list(model.classes_).index(label)]
    result["quality_score_0_100"] = np.round(
        sum(result[f"probability_{label}"] * SCORE_WEIGHTS[label] for label in LABELS), 2
    )
    return result


def wilson_interval(correct, total):
    """95% Wilson binomial interval; does not model dataset or model uncertainty."""
    z = NormalDist().inv_cdf(0.975)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, float(center - half_width)), min(1.0, float(center + half_width))]


def write_csv(frame, path, *, index=False):
    # Retain double precision and consistent line endings for output round trips.
    frame.to_csv(path, index=index, float_format="%.17g", lineterminator="\n")


def run(output="outputs", n=550, seed=42):
    data = generate(n, seed)
    counts = data.reference_category.value_counts().reindex(LABELS, fill_value=0)
    if (counts < 2).any():
        raise ValueError(
            "Stratified evaluation needs at least two runs per reference category; "
            "increase --runs or use another --seed"
        )
    split_seed = (int(seed) + 1) % 2**32
    train, test = train_test_split(
        data, test_size=0.2, stratify=data.reference_category, random_state=split_seed
    )
    if any(set(part.reference_category) != set(LABELS) for part in (train, test)):
        raise ValueError(
            "Stratified split did not retain all three reference categories in both subsets; "
            "increase --runs or use another --seed"
        )
    test = test.sort_values("run_id")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=int(seed), n_jobs=1
    )
    # Explicit allowlist excludes IDs, split metadata, reference scores and labels.
    # Fixed model parameters: held-out data is never used to fit or select a model.
    model.fit(train[METRICS], train.reference_category)
    assessments = assess(model, test)
    predictions = assessments.predicted_category
    accuracy = float(accuracy_score(test.reference_category, predictions))
    correct = int((test.reference_category == predictions).sum())
    train_counts = train.reference_category.value_counts().reindex(LABELS, fill_value=0)
    majority_label = train_counts.idxmax()
    importance = permutation_importance(
        model, test[METRICS], test.reference_category, scoring="accuracy",
        n_repeats=PERMUTATION_REPEATS, random_state=int(seed), n_jobs=1,
    )
    data["split"] = np.where(data.run_id.isin(test.run_id), "test", "train")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(data, out / "synthetic_executions.csv")
    write_csv(assessments, out / "quality_assessments.csv")
    write_csv(data[METRICS].describe().T, out / "metric_statistics.csv", index=True)
    by_category = pd.concat(
        {label: data.loc[data.reference_category == label, METRICS].describe().T for label in LABELS},
        names=["reference_category", "metric"],
    )
    write_csv(by_category, out / "metric_statistics_by_category.csv", index=True)
    write_csv(data[METRICS].corr(method="spearman"), out / "metric_correlations.csv", index=True)
    write_csv(
        pd.DataFrame({
            "metric": METRICS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False, kind="stable"),
        out / "feature_importance.csv",
    )
    report = {
        "dataset": "synthetic",
        "runs": int(n),
        "metrics": len(METRICS),
        "seed": int(seed),
        "training_runs": len(train),
        "test_runs": len(test),
        "split": {"method": "stratified run-level holdout", "test_fraction": 0.2, "seed": split_seed},
        "category_counts": {
            "all": {label: int(counts[label]) for label in LABELS},
            "train": {label: int(train_counts[label]) for label in LABELS},
            "test": {label: int((test.reference_category == label).sum()) for label in LABELS},
        },
        "correct_predictions": correct,
        "agreement_accuracy": accuracy,
        "agreement_accuracy_95pct_wilson": wilson_interval(correct, len(test)),
        "balanced_accuracy": float(balanced_accuracy_score(test.reference_category, predictions)),
        "macro_f1": float(f1_score(test.reference_category, predictions, labels=LABELS, average="macro")),
        "majority_baseline": {
            "training_majority_category": majority_label,
            "test_accuracy": float((test.reference_category == majority_label).mean()),
        },
        "target_agreement": TARGET_AGREEMENT,
        "target_met": accuracy >= TARGET_AGREEMENT,
        "target_definition": "Held-out point accuracy >= target; not a confidence-bound claim.",
        "categories": LABELS,
        "confusion_matrix_axes": {"rows": "reference_category", "columns": "predicted_category"},
        "confusion_matrix": confusion_matrix(test.reference_category, predictions, labels=LABELS).tolist(),
        "classification_report": classification_report(
            test.reference_category, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "model_parameters": model.get_params(),
        "feature_columns": METRICS,
        "permutation_importance": {
            "evaluation_split": "test", "scoring": "accuracy", "repeats": PERMUTATION_REPEATS,
            "std_definition": "Population standard deviation across permutations, not a confidence interval.",
        },
        "reference_rubric": {
            "terms": RUBRIC, "clip_each_term": [0, 1], "score_multiplier": 100,
            "category_thresholds": {"low": "score < 42", "medium": "42 <= score < 69", "high": "score >= 69"},
        },
        "score_weights": SCORE_WEIGHTS,
        "coverage_total_bins": COVERAGE_TOTAL_BINS,
        "synthetic_executions_sha256": hashlib.sha256((out / "synthetic_executions.csv").read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(), "system": platform.system(),
            "dependencies": {name: version(name) for name in DEPENDENCIES},
        },
        "limitations": (
            "Synthetic run summaries, not RTL simulation output or expert assessments. "
            "Agreement measures recovery of the generated rubric on the same generator distribution. "
            "Features share a latent quality factor; correlated features can mask permutation importance. "
            "Importance is exploratory and is not used for model or feature selection. "
            "Confidence is uncalibrated maximum class probability. The expected score is ordinal, "
            "not a calibrated industrial quality measure or a category threshold rule. "
            "The Wilson interval describes held-out binomial sampling uncertainty only."
        ),
    }
    (out / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
    print(f"{n} synthetic executions; {len(METRICS)} metrics; "
          f"test agreement {correct}/{len(test)} ({accuracy:.2%}); target met: {report['target_met']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=550)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    try:
        run(args.output, args.runs, args.seed)
    except ValueError as error:
        parser.error(str(error))
