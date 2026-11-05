"""Streamlit exploration of saved UVM quality outputs; never fits a model."""

import os
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import FILES, OutputDataError, filter_executions, load_outputs, rubric_evidence
from src.pipeline import LABELS, METRICS

PAGES = ["Overview", "Quality analysis", "Behavioural metrics", "Feature importance",
         "Individual execution", "Model evaluation"]
COLORS = {"low": "#d74f68", "medium": "#d4921c", "high": "#079b80"}
NAMES = {m: m.replace("_", " ").capitalize() for m in METRICS}
NAMES.update(run_id="Execution ID", reference_category="Reference category",
             predicted_category="Predicted category", quality_score_0_100="Model quality score (0–100)",
             transactions_per_ms="Transactions / ms", runtime_ms="Runtime (ms)",
             response_latency_us="Response latency (µs)", payload_entropy="Payload entropy (bits)")


@st.cache_data(show_spinner=False)
def read_artifacts(directory, signature):
    # The file metadata signature invalidates the cache after outputs regenerate.
    return load_outputs(directory)


def plot(figure, height=390):
    figure.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=48, b=25),
        font=dict(family="Arial, sans-serif", color="#33465e", size=12),
        legend_title_text="", legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def scatter(frame, x, y, title, **kwargs):
    return px.scatter(frame, x=x, y=y, color="reference_category", color_discrete_map=COLORS,
                      category_orders={"reference_category": LABELS}, hover_data=["run_id"],
                      labels=NAMES, title=title, opacity=0.8, **kwargs)


def distribution(scored):
    plot(px.histogram(scored, x="quality_score_0_100", color="predicted_category", nbins=20,
                      color_discrete_map=COLORS, category_orders={"predicted_category": LABELS},
                      labels=NAMES, title="Held-out model score distribution", barmode="stack"))


def category_distribution(selected, basis):
    counts = selected[basis].value_counts().reindex(LABELS, fill_value=0).rename_axis("Category").reset_index(name="Executions")
    plot(px.bar(counts, x="Category", y="Executions", color="Category", color_discrete_map=COLORS,
                text="Executions", title=f"{'Reference' if basis == 'reference_category' else 'Predicted'} quality categories"))


def overview(bundle, selected, scored, basis):
    report = bundle.report
    columns = st.columns(4)
    columns[0].metric("Total synthetic executions", f"{report['runs']:,}")
    columns[1].metric("Behavioural metrics", report["metrics"])
    columns[2].metric("Overall held-out agreement", f"{report['agreement_accuracy']:.2%}")
    columns[3].metric("Mean selected held-out score", f"{scored.quality_score_0_100.mean():.2f}" if len(scored) else "N/A")
    columns = st.columns(4)
    for column, label in zip(columns[:3], ["high", "medium", "low"]):
        column.metric(f"{label.title()} · {'reference' if basis == 'reference_category' else 'predicted'}", int((selected[basis] == label).sum()))
    columns[3].metric("Executions in selection", len(selected))
    st.caption(f"Corpus and overall agreement cards are global. Category counts follow filters; scores cover {len(scored)} selected held-out runs only.")
    left, right = st.columns(2)
    with left:
        if len(scored):
            distribution(scored)
        else:
            st.info("No held-out scores in this selection.")
    with right:
        category_distribution(selected, basis)
    plot(scatter(selected, "coverage_growth", "idle_fraction", "Coverage growth and idle activity",
                 size="transaction_count", size_max=17))


def quality_analysis(selected, scored, basis):
    left, right = st.columns(2)
    with left:
        if len(scored):
            distribution(scored)
        else:
            st.info("No held-out scores in this selection.")
    with right:
        category_distribution(selected, basis)
    if len(scored):
        ordered = scored.sort_values("run_id")
        figure = px.line(ordered, x="run_id", y="quality_score_0_100", markers=True, labels=NAMES,
                         hover_data=["reference_category", "predicted_category", "confidence"],
                         title="Held-out quality scores across execution IDs", color_discrete_sequence=["#2563eb"])
        figure.update_yaxes(range=[0, 100])
        plot(figure)
        st.caption("Execution ID order is not a time series or evidence of improvement over time. Training runs have no exported model score.")
    st.subheader("Productive and low-value behaviour proxies")
    st.caption("Compare high/low synthetic reference groups as illustrative proxies. No human productivity or verification effectiveness was measured.")
    comparison = selected.loc[selected.reference_category.isin(["low", "high"])]
    if comparison.empty:
        st.info("Select low or high reference executions to view this comparison.")
        return
    measures = ["coverage_growth", "corner_case_ratio", "idle_fraction", "warning_count", "transactions_per_ms", "component_pairs"]
    long = comparison.melt(id_vars=["run_id", "reference_category"], value_vars=measures, var_name="Metric", value_name="Value")
    figure = px.box(long, x="reference_category", y="Value", color="reference_category", facet_col="Metric", facet_col_wrap=3,
                     color_discrete_map=COLORS, category_orders={"reference_category": LABELS}, labels=NAMES)
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.for_each_annotation(lambda annotation: annotation.update(text=NAMES[annotation.text.split("=")[-1]]))
    plot(figure, 640)


def behavioural_metrics(bundle, selected):
    stimulus, assertions, coverage, warnings, activity, components, statistics = st.tabs(
        ["Stimulus", "Assertions", "Coverage", "Warnings", "Idle & traffic", "Components", "Statistics"]
    )
    with stimulus:
        plot(scatter(selected, "unique_addresses", "payload_entropy", "Stimulus diversity", size="stimulus_types", size_max=17))
        long = selected.melt(id_vars=["reference_category"], value_vars=["stimulus_reuse_ratio", "corner_case_ratio", "randomization_success"], var_name="Measure", value_name="Fraction")
        plot(px.box(long, x="Measure", y="Fraction", color="reference_category", color_discrete_map=COLORS, title="Stimulus fractions"))
    with assertions:
        plot(scatter(selected, "assertion_checks", "assertion_failures", "Assertion checks and failures"))
        plot(px.box(selected, x="reference_category", y="assertion_pass_ratio", color="reference_category", color_discrete_map=COLORS, labels=NAMES, title="Observed assertion pass ratio"))
    with coverage:
        figure = scatter(selected, "coverage_initial", "coverage_final", "Initial and final coverage", size="coverage_bins_hit", size_max=17)
        figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#aab7c8", dash="dot"))
        plot(figure)
        plot(px.box(selected, x="reference_category", y="coverage_growth", color="reference_category", color_discrete_map=COLORS, labels=NAMES, title="Coverage growth (fraction)"))
    with warnings:
        plot(scatter(selected, "warning_count", "warning_bursts", "Warnings and disjoint warning bursts"))
        plot(scatter(selected, "protocol_retries", "error_count", "Protocol retries and errors"))
    with activity:
        plot(scatter(selected, "idle_fraction", "transactions_per_ms", "Idle fraction and transaction frequency", size="transaction_count", size_max=17))
        plot(scatter(selected, "runtime_ms", "response_latency_us", "Runtime and response latency"))
    with components:
        plot(scatter(selected, "component_pairs", "component_messages", "Component interaction"))
    with statistics:
        st.caption("Saved statistics describe the full corpus and are unaffected by sidebar filters. Comparisons are descriptive, not significance tests.")
        st.dataframe(bundle.statistics.rename_axis("Metric").reset_index(), hide_index=True, width="stretch")
        measures = st.multiselect("Correlation metrics", METRICS, default=list(bundle.report["reference_rubric"]["terms"]), key="correlation_metrics")
        if measures:
            plot(px.imshow(bundle.correlations.loc[measures, measures], zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="Global Spearman correlations"), 560)


def feature_importance(bundle):
    st.info("Global held-out permutation importance is fixed for the saved evaluation. Sidebar execution filters do not recompute it.")
    importance = bundle.importance.sort_values("importance_mean").copy()
    importance["Accuracy drop (percentage points)"] = importance.importance_mean * 100
    importance["Permutation standard deviation"] = importance.importance_std * 100
    plot(px.bar(importance, x="Accuracy drop (percentage points)", y="metric", orientation="h",
                error_x="Permutation standard deviation", color_discrete_sequence=["#2563eb"],
                title="Global behavioural feature importance", labels={"metric": "Behavioural metric"}), 850)
    st.caption(f"{bundle.report['permutation_importance']['repeats']} permutations per metric. Error bars are population standard deviations, not confidence intervals. Negative importance is retained.")
    st.warning("Importance describes aggregate prediction sensitivity, not causation or an individual execution's explanation. Correlated and derived metrics can mask one another's importance.")
    st.dataframe(bundle.importance, hide_index=True, width="stretch")


def individual_execution(bundle, selected):
    heldout_only = st.checkbox("Only held-out assessments", value=True, key="heldout_only")
    candidates = selected.loc[selected.split == "test"] if heldout_only else selected
    if candidates.empty:
        st.info("No eligible execution in this selection. Adjust the category or ID filters.")
        return
    execution_id = st.selectbox("Simulated execution ID", sorted(candidates.run_id.astype(int)), key="execution_id")
    row = candidates.loc[candidates.run_id == execution_id].iloc[0]
    assessed = pd.notna(row.predicted_category)
    columns = st.columns(4)
    columns[0].metric("Predicted category", row.predicted_category.title() if assessed else "Not assessed")
    columns[1].metric("Model quality score", f"{row.quality_score_0_100:.2f}" if assessed else "N/A")
    columns[2].metric("Reference category", row.reference_category.title())
    columns[3].metric("Prediction confidence", f"{row.confidence:.2%}" if assessed else "N/A")
    st.caption(f"Execution {execution_id} · {row.split} partition · synthetic reference score {row.reference_score_0_100:.4f}/100")
    if assessed:
        probabilities = pd.DataFrame({"Category": LABELS, "Probability": [row[f"probability_{label}"] for label in LABELS]})
        plot(px.bar(probabilities, x="Category", y="Probability", color="Category", color_discrete_map=COLORS, range_y=[0, 1], title="Exported class probabilities"), 300)
        st.write(f"Score calculation: 50 × P(medium) + 100 × P(high) = {row.quality_score_0_100:.2f} (rounded). The category is the maximum-probability class; confidence is uncalibrated.")
    else:
        st.info("This training execution has no exported held-out prediction. Its metrics and synthetic reference rubric remain available below.")
    st.subheader("Observed behavioural evidence")
    evidence = pd.DataFrame({"Metric": METRICS, "Observed value": [row[m] for m in METRICS]})
    for label in ["low", "high"]:
        evidence[f"{label.title()} reference median (full corpus)"] = [bundle.category_statistics.loc[(label, m), "50%"] for m in METRICS]
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption("Reference medians provide descriptive context; they do not explain the forest's individual decision.")
    st.subheader("Traceable reference rubric")
    st.dataframe(rubric_evidence(row, bundle.report), hide_index=True, width="stretch")
    st.caption("This reconstructs the synthetic reference score, not a local model attribution. No original UVM log or tree-level explanation is exported.")
    st.code(f"synthetic_executions.csv: run_id={execution_id}\nquality_assessments.csv: {'run_id=' + str(execution_id) if assessed else 'no row (training partition)'}\nevaluation.json: rubric, score weights, split and environment\nDataset SHA-256: {bundle.report['synthetic_executions_sha256']}", language="text")


def model_evaluation(bundle):
    report = bundle.report
    st.info(f"Global evaluation on all {report['test_runs']} held-out executions; unaffected by sidebar filters. Reference categories come from the synthetic rubric, not independent experts.")
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Agreement accuracy", "Balanced accuracy", "Macro F1", "Agreement target"],
                                     [f"{report['agreement_accuracy']:.2%}", f"{report['balanced_accuracy']:.2%}", f"{report['macro_f1']:.4f}", f"{report['target_agreement']:.0%}"]):
        column.metric(label, value)
    message = f"Target {'met' if report['target_met'] else 'not met'}: {report['correct_predictions']}/{report['test_runs']} correct held-out predictions. Target comparison uses point accuracy."
    (st.success if report["target_met"] else st.warning)(message)
    lower, upper = report["agreement_accuracy_95pct_wilson"]
    st.caption(f"95% Wilson interval: {lower:.2%}–{upper:.2%}. Binomial sampling approximation only; excludes generator/model uncertainty. Training-majority baseline: {report['majority_baseline']['test_accuracy']:.2%}.")
    left, right = st.columns([1, 1.3])
    with left:
        plot(px.imshow(report["confusion_matrix"], x=LABELS, y=LABELS, text_auto=True,
                        labels=dict(x="Predicted category", y="Reference category", color="Executions"),
                        color_continuous_scale="Blues", title="Held-out confusion matrix"), 430)
    with right:
        st.subheader("Classification report")
        classification = pd.DataFrame({label: report["classification_report"][label] for label in LABELS + ["macro avg", "weighted avg"]}).T
        st.dataframe(classification.rename_axis("Category").reset_index(), hide_index=True, width="stretch")
    with st.expander("Evaluation provenance"):
        st.json({key: report[key] for key in ["seed", "split", "category_counts", "model_parameters", "environment"]})


def main():
    st.set_page_config(page_title="UVM Quality Analytics", page_icon="◈", layout="wide")
    st.markdown("""<style>
      .block-container {max-width:1480px;padding-top:2.2rem;padding-bottom:2rem}
      [data-testid="stSidebar"] {border-right:1px solid #dfe6ef}
      [data-testid="stMetric"] {background:#fff;border:1px solid #e1e7f0;border-radius:12px;padding:18px;margin-bottom:12px}
      [data-testid="stMetricValue"] {color:#143762;font-weight:650}
      .eyebrow {color:#2563eb;font-size:12px;font-weight:700;letter-spacing:2px;margin-bottom:8px}
      .tag {display:inline-block;background:#e6f3f0;color:#087b67;border-radius:16px;padding:4px 12px;font-size:12px;font-weight:600}
      h1 {letter-spacing:-1px} h3 {letter-spacing:-.3px}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("## ◈ UVM Quality\nEngineering analytics")
    st.sidebar.caption("PERSONAL PROJECT · SYNTHETIC DATA")
    page = st.sidebar.radio("Navigate", PAGES, key="page")
    directory = Path(os.environ.get("UVM_OUTPUT_DIR", str(ROOT / "outputs"))).resolve()
    try:
        signature = tuple((name, (directory / name).stat().st_mtime_ns, (directory / name).stat().st_size) for name in FILES if (directory / name).is_file())
        bundle = read_artifacts(str(directory), signature)
    except (OutputDataError, OSError) as error:
        st.error(f"Cannot load pipeline outputs. {error}")
        st.code("python src/pipeline.py --output outputs", language="bash")
        st.caption("Generate a complete, consistent output set. UVM_OUTPUT_DIR can select an alternative local output directory.")
        st.stop()
    st.sidebar.divider()
    basis_label = st.sidebar.selectbox("Category basis", ["Reference category · all runs", "Predicted category · held-out only"], key="category_basis")
    basis = "reference_category" if basis_label.startswith("Reference") else "predicted_category"
    categories = st.sidebar.multiselect("Quality categories", LABELS, default=LABELS, key="categories")
    lower, upper = int(bundle.executions.run_id.min()), int(bundle.executions.run_id.max())
    id_range = st.sidebar.slider("Execution ID range", lower, upper, (lower, upper), key="id_range")
    selected = filter_executions(bundle.combined, categories, id_range, basis)
    scored = selected.dropna(subset=["quality_score_0_100"])
    st.sidebar.caption(f"{len(selected)} executions selected · {len(scored)} held-out scores")
    st.sidebar.caption("Evaluation and feature importance remain global. The app only reads saved outputs; it never trains a model.")
    st.markdown('<div class="eyebrow">VERIFICATION / RUNTIME ANALYTICS</div>', unsafe_allow_html=True)
    st.title(page)
    st.markdown('<span class="tag">Synthetic rubric-defined reference categories</span>', unsafe_allow_html=True)
    st.caption("Explore runtime behaviour, trace model scores, and compare reference categories. Agreement is not independent expert validation.")
    if selected.empty and page not in {"Feature importance", "Model evaluation"}:
        st.info("No executions match these filters. Select a category or widen the execution ID range.")
        return
    if page == "Overview":
        overview(bundle, selected, scored, basis)
    elif page == "Quality analysis":
        quality_analysis(selected, scored, basis)
    elif page == "Behavioural metrics":
        behavioural_metrics(bundle, selected)
    elif page == "Feature importance":
        feature_importance(bundle)
    elif page == "Individual execution":
        individual_execution(bundle, selected)
    else:
        model_evaluation(bundle)
    st.divider()
    st.caption(f"Saved dataset · seed {bundle.report['seed']} · {bundle.report['training_runs']} training / {bundle.report['test_runs']} held-out · {len(selected)} selected executions")


if __name__ == "__main__":
    main()
