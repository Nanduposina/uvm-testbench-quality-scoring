"""Dashboard tests use the checked-in pipeline artifacts and Streamlit AppTest."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.data import FILES, OutputDataError, filter_executions, load_outputs, rubric_evidence
from src.pipeline import LABELS, METRICS, RandomForestClassifier


@pytest.fixture(scope="module")
def bundle():
    return load_outputs(ROOT / "outputs")


@pytest.fixture
def copied_outputs(tmp_path):
    target = tmp_path / "outputs"
    shutil.copytree(ROOT / "outputs", target)
    return target


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("UVM_OUTPUT_DIR", str(ROOT / "outputs"))

    def forbid_training(*args, **kwargs):
        raise AssertionError("Dashboard must not train a model")

    monkeypatch.setattr(RandomForestClassifier, "fit", forbid_training)
    return AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()


def metrics(app):
    return {metric.label: metric.value for metric in app.metric}


def test_loader_reconciles_real_outputs(bundle):
    assert len(bundle.executions) == bundle.report["runs"] >= 500
    assert len(METRICS) == bundle.report["metrics"] >= 20
    assert len(bundle.combined) == len(bundle.executions)
    assert bundle.combined.loc[bundle.combined.split == "train", "quality_score_0_100"].isna().all()
    assert len(bundle.assessments) == bundle.report["test_runs"]


@pytest.mark.parametrize("filename", FILES)
def test_missing_output_is_understandable(copied_outputs, filename):
    (copied_outputs / filename).unlink()
    with pytest.raises(OutputDataError, match="Missing pipeline outputs") as caught:
        load_outputs(copied_outputs)
    assert filename in str(caught.value)


@pytest.mark.parametrize("contents", ["{broken", "[]", "{}", "null"])
def test_malformed_evaluation_is_understandable(copied_outputs, contents):
    (copied_outputs / "evaluation.json").write_text(contents, encoding="utf8")
    with pytest.raises(OutputDataError):
        load_outputs(copied_outputs)


@pytest.mark.parametrize("field,value", [
    ("agreement_accuracy_95pct_wilson", ["bad", 1]), ("environment", []),
    ("model_parameters", None), ("permutation_importance", {}), ("seed", "bad"),
    ("target_agreement", float("nan")),
])
def test_malformed_page_metadata_is_rejected(copied_outputs, field, value):
    path = copied_outputs / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf8"))
    report[field] = value
    path.write_text(json.dumps(report), encoding="utf8")
    with pytest.raises(OutputDataError):
        load_outputs(copied_outputs)


@pytest.mark.parametrize("problem", ["probability", "confidence", "score", "matrix", "duplicate", "column", "negative", "statistics", "empty"])
def test_inconsistent_artifacts_are_rejected(copied_outputs, problem):
    filename = "quality_assessments.csv"
    if problem == "matrix":
        path = copied_outputs / "evaluation.json"
        report = json.loads(path.read_text(encoding="utf8"))
        report["confusion_matrix"][0][0] += 1
        path.write_text(json.dumps(report), encoding="utf8")
    else:
        if problem in {"duplicate", "negative"}:
            filename = "synthetic_executions.csv"
        if problem == "statistics":
            filename = "metric_statistics.csv"
        path = copied_outputs / filename
        data = pd.read_csv(path, float_precision="round_trip")
        if problem == "probability":
            data.loc[0, "probability_high"] = 2
        elif problem == "confidence":
            data.loc[0, "confidence"] = 0
        elif problem == "score":
            data.loc[0, "quality_score_0_100"] = -1
        elif problem == "duplicate":
            data.loc[1, "run_id"] = data.loc[0, "run_id"]
        elif problem == "column":
            data = data.drop(columns="confidence")
        elif problem == "negative":
            data.loc[0, "warning_count"] = -1
        elif problem == "statistics":
            data.loc[0, "mean"] = -500
        if problem == "empty":
            path.write_text("", encoding="utf8")
        else:
            data.to_csv(path, index=False)
    with pytest.raises(OutputDataError):
        load_outputs(copied_outputs)


def test_filters_preserve_source_rows_and_holdout_semantics(bundle):
    frame = bundle.combined
    result = filter_executions(frame, ["high"], (100, 250))
    expected = frame.loc[(frame.reference_category == "high") & frame.run_id.between(100, 250)]
    pd.testing.assert_frame_equal(result, expected)
    predicted = filter_executions(frame, LABELS, (0, int(frame.run_id.max())), "predicted_category")
    assert set(predicted.run_id) == set(bundle.assessments.run_id)
    assert predicted.split.eq("test").all()
    assert filter_executions(frame, [], (0, 10)).empty


def test_individual_evidence_reconstructs_saved_reference_scores(bundle):
    for label in LABELS:
        row = bundle.executions.loc[bundle.executions.reference_category == label].iloc[0]
        evidence = rubric_evidence(row, bundle.report)
        assert evidence["Reference score points"].sum() == pytest.approx(row.reference_score_0_100)
        assert evidence["Normalized term"].between(0, 1).all()
        for item in evidence.to_dict("records"):
            assert item["Observed value"] == row[item["Metric"]]


@pytest.mark.parametrize("page,charts", [
    ("Overview", 3), ("Quality analysis", 4), ("Behavioural metrics", 12),
    ("Feature importance", 1), ("Individual execution", 1), ("Model evaluation", 1),
])
def test_all_dashboard_sections_load_without_training(app, page, charts):
    app.sidebar.radio(key="page").set_value(page).run()
    assert not app.exception and not app.error
    assert len(app.get("plotly_chart")) == charts
    for chart in app.get("plotly_chart"):
        figure = json.loads(chart.proto.spec)
        assert figure["data"], page


def test_overview_kpis_and_filters_match_outputs(app, bundle):
    values = metrics(app)
    assert values["Total synthetic executions"] == str(bundle.report["runs"])
    assert values["Behavioural metrics"] == str(bundle.report["metrics"])
    assert values["Overall held-out agreement"] == f"{bundle.report['agreement_accuracy']:.2%}"
    assert values["Mean selected held-out score"] == f"{bundle.assessments.quality_score_0_100.mean():.2f}"
    app.sidebar.multiselect(key="categories").set_value(["high"])
    app.sidebar.slider(key="id_range").set_range(100, 250).run()
    expected = filter_executions(bundle.combined, ["high"], (100, 250))
    assert metrics(app)["Executions in selection"] == str(len(expected))
    assert metrics(app)["Mean selected held-out score"] == f"{expected.quality_score_0_100.mean():.2f}"
    assert not app.exception and not app.error
    app.sidebar.multiselect(key="categories").set_value(LABELS)
    app.sidebar.slider(key="id_range").set_range(0, int(bundle.executions.run_id.max()))
    app.sidebar.selectbox(key="category_basis").select("Predicted category · held-out only").run()
    assert metrics(app)["Executions in selection"] == str(bundle.report["test_runs"])


def test_individual_assessment_matches_export_and_training_is_not_scored(app, bundle):
    app.sidebar.radio(key="page").set_value("Individual execution").run()
    saved = bundle.assessments.iloc[5]
    app.selectbox(key="execution_id").select(int(saved.run_id)).run()
    values = metrics(app)
    assert values["Predicted category"] == saved.predicted_category.title()
    assert values["Model quality score"] == f"{saved.quality_score_0_100:.2f}"
    assert values["Reference category"] == saved.reference_category.title()
    assert values["Prediction confidence"] == f"{saved.confidence:.2%}"
    observed = next(table.value for table in app.dataframe if "Observed value" in table.value and "Metric" in table.value and len(table.value) == len(METRICS))
    source = bundle.executions.set_index("run_id").loc[saved.run_id]
    np.testing.assert_allclose(observed["Observed value"], source[METRICS].astype(float))
    app.checkbox(key="heldout_only").uncheck().run()
    training = bundle.executions.loc[bundle.executions.split == "train"].iloc[0]
    app.selectbox(key="execution_id").select(int(training.run_id)).run()
    assert metrics(app)["Predicted category"] == "Not assessed"
    assert metrics(app)["Model quality score"] == "N/A"
    assert not app.get("plotly_chart") and not app.exception


def test_empty_filter_is_understandable(app):
    app.sidebar.multiselect(key="categories").set_value([]).run()
    assert not app.exception and not app.error and not app.get("plotly_chart")
    assert any("No executions match" in item.value for item in app.info)


@pytest.mark.parametrize("malformed", [False, True])
def test_dashboard_shows_actionable_error(monkeypatch, copied_outputs, malformed):
    if malformed:
        (copied_outputs / "evaluation.json").write_text("broken JSON", encoding="utf8")
    else:
        (copied_outputs / "quality_assessments.csv").unlink()
    monkeypatch.setenv("UVM_OUTPUT_DIR", str(copied_outputs))
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception and app.error
    assert "Cannot load pipeline outputs" in app.error[0].value
    assert "python src/pipeline.py --output outputs" in app.code[0].value


def test_dashboard_does_not_modify_artifacts(app):
    before = {name: hashlib.sha256((ROOT / "outputs" / name).read_bytes()).hexdigest() for name in FILES}
    for page in ["Quality analysis", "Behavioural metrics", "Feature importance", "Individual execution", "Model evaluation"]:
        app.sidebar.radio(key="page").set_value(page).run()
        assert not app.exception
    after = {name: hashlib.sha256((ROOT / "outputs" / name).read_bytes()).hexdigest() for name in FILES}
    assert before == after
