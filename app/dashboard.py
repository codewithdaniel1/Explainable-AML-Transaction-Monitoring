"""Local review dashboard for synthetic AML study outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from aml_monitoring.data import IBM_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results" / "study_full_history_10pct"


@st.cache_data
def read_predictions(path: str, model: str, period: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        selected = chunk.loc[(chunk["model"] == model) & (chunk["period"] == period)]
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True).sort_values(
        "score", ascending=False, kind="stable"
    ).reset_index(drop=True)


@st.cache_data
def read_json(path: str):
    return json.loads(Path(path).read_text())


@st.cache_data
def read_sender_history(csv_path: str, from_bank: str, from_account: str,
                        current_time: str) -> pd.DataFrame:
    """Scan the synthetic CSV in chunks for a selected sender's earlier week."""
    now = pd.Timestamp(current_time)
    earliest = now - pd.Timedelta(days=7)
    parts = []
    for chunk in pd.read_csv(csv_path, header=0, names=IBM_COLUMNS, chunksize=100_000,
                             dtype={name: str for name in ("from_bank", "from_account",
                                                               "to_bank", "to_account")}):
        sender = (chunk["from_bank"] == from_bank) & (chunk["from_account"] == from_account)
        if sender.any():
            selected = chunk.loc[sender].copy()
            selected["timestamp"] = pd.to_datetime(selected["timestamp"])
            selected = selected.loc[(selected["timestamp"] >= earliest) &
                                    (selected["timestamp"] < now)]
            if not selected.empty:
                parts.append(selected)
    return (pd.concat(parts, ignore_index=True).sort_values("timestamp") if parts
            else pd.DataFrame(columns=IBM_COLUMNS))


st.set_page_config(page_title="Synthetic AML review", layout="wide")
st.title("Synthetic AML transaction review")
st.caption("Local class-project dashboard. Labels are synthetic; scores are not calibrated probabilities.")

study_dir = Path(st.sidebar.text_input("Study results folder", str(DEFAULT_STUDY))).expanduser()
required = ["study_manifest.json", "metrics.json", "predictions.csv", "explanations.json"]
missing = [name for name in required if not (study_dir / name).exists()]
if missing:
    st.info("Study outputs are missing. Run the complete-history study command in README.md first. "
            f"Missing: {', '.join(missing)}")
    st.stop()

manifest = read_json(str(study_dir / "study_manifest.json"))
metrics = read_json(str(study_dir / "metrics.json"))
explanations = read_json(str(study_dir / "explanations.json"))
selected_model = manifest["selected_by_validation_average_precision"]
model = st.sidebar.selectbox("Specification", list(metrics),
                             index=list(metrics).index(selected_model))
period = st.sidebar.selectbox("Evaluation period", ["test", "validation", "late_stress"])
predictions = read_predictions(str(study_dir / "predictions.csv"), model, period)
max_capacity = min(1000, len(predictions))
capacity = st.sidebar.slider("Review capacity", min_value=1, max_value=max_capacity,
                             value=min(100, max_capacity))
chosen_metrics = metrics[model][period]
top = predictions.head(capacity).copy()
top.insert(0, "alert_rank", range(1, len(top) + 1))
found = int(top["label"].sum())

if not manifest.get("full_history", False):
    st.warning("This run uses incomplete sampled account history. See the study report for limits.")
if period == "late_stress":
    st.warning("The late period has a sharp volume and label-rate shift. Its metrics are a stress check.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows in period", f"{len(predictions):,}")
c2.metric("Positive labels", f"{int(predictions['label'].sum()):,}")
c3.metric("Average precision", f"{chosen_metrics['pr_auc']:.4f}")
c4.metric(f"Labels in top {capacity}", f"{found}")
st.caption(f"Selected by validation AP: {selected_model}. Alert precision at this capacity: "
           f"{found / capacity:.1%}; recall: {found / max(int(predictions['label'].sum()), 1):.1%}.")

with st.expander("Compare all specifications", expanded=True):
    comparison = pd.DataFrame([
        {"model": name, "validation_ap": value["validation"]["pr_auc"],
         "primary_test_ap": value["test"]["pr_auc"],
         "test_labels_top_100": value["test"]["positives_found"]}
        for name, value in metrics.items()
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")

replay_daily = study_dir / "replay_daily_monitoring.csv"
if replay_daily.exists():
    with st.expander("Saved-score alert replay and daily monitoring"):
        replay = pd.read_csv(replay_daily)
        st.caption("Cutoff selected from validation scores only. This is a replay of saved scores, "
                   "not live scoring. The late period is a separate stress check.")
        st.dataframe(replay, hide_index=True, width="stretch")
        st.bar_chart(replay.set_index("date")["alerts"])

st.subheader("Ranked transactions")
st.dataframe(top[["alert_rank", "source_row", "timestamp", "payment_currency",
                  "label", "score"]], hide_index=True, width="stretch")

if (study_dir / "case_context.csv").exists():
    context = pd.read_csv(study_dir / "case_context.csv", dtype=str).set_index("source_row")
    case_id = st.selectbox("Inspect a top alert", top["source_row"].astype(str).tolist())
    if case_id in context.index:
        st.write(context.loc[[case_id]].T.rename(columns={case_id: "Synthetic transaction"}))
        raw_path = Path(manifest["source"])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        if raw_path.exists() and st.button("Load sender's prior 7 days"):
            case = context.loc[case_id]
            sender_history = read_sender_history(
                str(raw_path), case["from_bank"], case["from_account"], case["timestamp"]
            )
            st.write(f"{len(sender_history)} earlier synthetic transactions")
            st.dataframe(sender_history[["timestamp", "to_bank", "to_account",
                                         "amount_paid", "payment_currency", "payment_format"]],
                         hide_index=True, width="stretch")
            if not sender_history.empty:
                counterparties = sender_history.groupby("to_account").size().sort_values(
                    ascending=False).head(10)
                st.bar_chart(counterparties)
                edges = (sender_history.groupby(["to_bank", "to_account"]).size()
                         .sort_values(ascending=False).head(10))
                sender_id = f"{case['from_bank']}:{case['from_account']}"
                lines = ["digraph G {", "rankdir=LR;", "node [shape=box];",
                         f"{json.dumps(sender_id)} [label={json.dumps('Sender ' + sender_id)}];"]
                for (bank, account), count in edges.items():
                    receiver_id = f"{bank}:{account}"
                    lines.append(f"{json.dumps(sender_id)} -> {json.dumps(receiver_id)} "
                                 f"[label={json.dumps(str(count) + ' txns')}];")
                lines.append("}")
                st.graphviz_chart("\n".join(lines), width="stretch")
    else:
        st.info("Case context was exported for the selected model's original top 100 alerts only.")
    notes_path = study_dir / "case_notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    current = notes.get(case_id, {})
    with st.form("case_review"):
        disposition = st.selectbox("Review disposition", ["unreviewed", "needs review", "dismissed"],
                                   index=["unreviewed", "needs review", "dismissed"].index(
                                       current.get("disposition", "unreviewed")))
        note = st.text_area("Synthetic case note", value=current.get("note", ""))
        if st.form_submit_button("Save local note"):
            notes[case_id] = {"disposition": disposition, "note": note}
            notes_path.write_text(json.dumps(notes, indent=2) + "\n")
            st.success("Saved under the ignored study results folder.")

st.subheader("Model explanation")
detail = explanations.get(model)
if detail:
    terms = detail.get("top_importances", detail.get("top_coefficients", detail.get("top_terms", [])))
    st.dataframe(pd.DataFrame(terms), hide_index=True, width="stretch")
    local = detail.get("local_tree_shap_log_odds")
    if local:
        st.caption("Native Tree SHAP contributions are in log-odds units for five top-scored "
                   "primary-test transactions. They explain model behavior, not causation.")
        local_ids = [str(row["source_row"]) for row in local]
        local_id = st.selectbox("Local explanation source row", local_ids)
        item = next(row for row in local if str(row["source_row"]) == local_id)
        st.write(f"Base value: {item['base_value']:.3f} log odds")
        st.dataframe(pd.DataFrame(item["top_terms"]), hide_index=True, width="stretch")
else:
    st.info("The illustrative rules do not have fitted model explanations.")

figure = ROOT / "docs" / "figures" / "sampled_pilot_performance.png"
if figure.exists():
    st.image(str(figure), caption="Published 10% sampled pilot with complete prior history")
st.caption("See docs/STUDY_RESULTS.md and docs/MODEL_CARD.md for methods, limits, and intended use.")
