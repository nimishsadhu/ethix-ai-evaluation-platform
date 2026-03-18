"""
streamlit_app.py
----------------
ETHIX — AI Evaluation and Monitoring Platform

Main entry point.  Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os

# Make sure the project src folder is importable regardless of where you run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import streamlit as st

from src.data_loader    import load_dataset, get_basic_info, detect_column_types, validate_target
from src.preprocessing  import full_preprocessing_pipeline
from src.eda            import perform_eda
from src.model          import train_model, evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.fairness       import calculate_fairness_metrics, plot_group_positive_rates, plot_fairness_dashboard, plot_bias_mitigation_comparison
from src.ethical_score  import calculate_ethical_score, plot_ethical_score_gauge, plot_score_components
from src.predict        import predict_applicant, predict_batch


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ETHIX – AI Ethics Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark theme with accent colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #3a3a5c;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px 8px 0 0;
        color: #aaaaaa;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #16213e;
        color: #4C72B0 !important;
        border-bottom: 2px solid #4C72B0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0ff;
        padding: 8px 0 4px 0;
        border-bottom: 2px solid #4C72B0;
        margin-bottom: 16px;
    }

    /* Status badges */
    .badge-fair     { color: #2ecc71; font-weight: 700; }
    .badge-unfair   { color: #e74c3c; font-weight: 700; }

    /* Info box */
    .info-box {
        background: #1a1a2e;
        border-left: 4px solid #4C72B0;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    /* Ethical score card */
    .score-card {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        border: 1px solid #3a3a5c;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/nolan/96/scales.png",
        width=64,
    )
    st.title("ETHIX ⚖️")
    st.caption("AI Evaluation & Monitoring Platform")
    st.divider()

    # --- Dataset upload ---
    st.subheader("📂  Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload any tabular binary-classification dataset.",
    )

    # Load a sample dataset for quick demos
    use_sample = st.checkbox("Use sample dataset (Adult Income)", value=False)

    if use_sample and uploaded_file is None:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                uploaded_file = f  # pass file object directly

    st.divider()

    # Placeholders — filled once a file is loaded
    target_col    = None
    sensitive_col = None
    run_analysis  = False

    if uploaded_file is not None:
        try:
            df_raw = load_dataset(uploaded_file)

            st.subheader("🎯  Column Selection")

            target_col = st.selectbox(
                "Target column (what to predict)",
                options=df_raw.columns.tolist(),
                index=len(df_raw.columns) - 1,
            )
            sensitive_col = st.selectbox(
                "Sensitive attribute (for fairness)",
                options=df_raw.columns.tolist(),
                index=min(6, len(df_raw.columns) - 1),
            )

            st.divider()
            run_analysis = st.button(
                "🚀  Run Analysis",
                type="primary",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Could not read file: {e}")
            uploaded_file = None


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

# --- Landing page (no file uploaded yet) ---
if uploaded_file is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center'>
            <h1 style='font-size:3rem; color:#4C72B0;'>⚖️ ETHIX</h1>
            <h3 style='color:#aaaaaa; font-weight:400;'>AI Evaluation & Monitoring Platform</h3>
            <br>
            <p style='color:#cccccc; font-size:1.05rem; line-height:1.7;'>
            ETHIX helps you evaluate machine learning models not just on accuracy,
            but on <strong>fairness</strong>, <strong>bias</strong>, and
            <strong>ethical responsibility</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("📊  **EDA**\nAutomatic exploratory data analysis")
        with c2:
            st.info("⚖️  **Fairness**\nSPD, DIR, EOD metrics")
        with c3:
            st.info("🛡️  **Mitigation**\nBias reduction experiment")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("← Upload a CSV file in the sidebar to get started, or tick the sample dataset checkbox.")
    st.stop()


# ---------------------------------------------------------------------------
# File is loaded — show basic info immediately
# ---------------------------------------------------------------------------
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("<h2 style='color:#e0e0ff; margin-bottom:4px;'>⚖️ ETHIX Dashboard</h2>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<span style='background:#4C72B0;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;'>LIVE</span>",
        unsafe_allow_html=True,
    )

st.divider()

# --- Quick stats row ---
info = get_basic_info(df_raw)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",           f"{info['rows']:,}")
c2.metric("Columns",        info["columns"])
c3.metric("Missing Cells",  f"{info['missing_cells']:,}")
c4.metric("Duplicate Rows", f"{info['duplicate_rows']:,}")
c5.metric("Memory (MB)",    info["memory_mb"])

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs([
    "📋 Dataset",
    "🔍 EDA",
    "🤖 Model",
    "📊 Performance",
    "⚖️ Fairness",
    "🛡️ Mitigation",
    "🏆 Ethical Score",
    "🔮 Predict",
])

tab_dataset, tab_eda, tab_model, tab_perf, tab_fair, tab_miti, tab_score, tab_predict = tabs


# ===================================================================
# TAB 1 — DATASET
# ===================================================================
with tab_dataset:
    st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)

    st.dataframe(df_raw.head(100), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>Column Types (auto-detected)</div>", unsafe_allow_html=True)
        col_types = detect_column_types(df_raw)

        type_df = []
        for ctype, cols in col_types.items():
            for c in cols:
                type_df.append({"Column": c, "Detected Type": ctype.capitalize()})
        if type_df:
            st.dataframe(pd.DataFrame(type_df), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.describe(include="all").T, use_container_width=True)

    if col_types.get("identifier"):
        st.warning(
            f"⚠️  Possible identifier column(s) detected: "
            f"**{', '.join(col_types['identifier'])}** — these will be dropped before training."
        )


# ===================================================================
# TAB 2 — EDA
# ===================================================================
with tab_eda:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    if target_col:
        is_valid, msg = validate_target(df_raw, target_col)
        if not is_valid:
            st.error(f"❌  {msg}")
        else:
            col_types_eda = detect_column_types(df_raw)
            eda_figs = perform_eda(df_raw, target_col, col_types_eda)

            # Missing values
            st.markdown("**Missing Values**")
            st.pyplot(eda_figs["missing"], use_container_width=False)
            st.markdown("<br>", unsafe_allow_html=True)

            # Target distribution
            st.markdown("**Target Distribution**")
            st.pyplot(eda_figs["target_dist"], use_container_width=False)
            st.markdown("<br>", unsafe_allow_html=True)

            # Categorical vs target
            if eda_figs["cat_plots"]:
                st.markdown("**Categorical Features vs Target**")
                cat_cols_chunks = [eda_figs["cat_plots"][i:i+2] for i in range(0, len(eda_figs["cat_plots"]), 2)]
                for pair in cat_cols_chunks:
                    cols = st.columns(len(pair))
                    for col_ui, (col_name, fig) in zip(cols, pair):
                        with col_ui:
                            st.pyplot(fig, use_container_width=False)

            st.markdown("<br>", unsafe_allow_html=True)

            # Numerical vs target
            if eda_figs["num_plots"]:
                st.markdown("**Numerical Features vs Target**")
                num_chunks = [eda_figs["num_plots"][i:i+2] for i in range(0, len(eda_figs["num_plots"]), 2)]
                for pair in num_chunks:
                    cols = st.columns(len(pair))
                    for col_ui, (col_name, fig) in zip(cols, pair):
                        with col_ui:
                            st.pyplot(fig, use_container_width=False)

            st.markdown("<br>", unsafe_allow_html=True)

            # Correlation heatmap
            st.markdown("**Correlation Heatmap**")
            st.pyplot(eda_figs["heatmap"], use_container_width=False)
    else:
        st.info("Please select a target column in the sidebar.")


# ===================================================================
# TAB 3 — MODEL TRAINING
# ===================================================================
with tab_model:
    st.markdown("<div class='section-header'>Model Training</div>", unsafe_allow_html=True)

    if not run_analysis:
        st.info("👈  Configure your columns in the sidebar and click **Run Analysis** to train the model.")
    elif target_col is None or sensitive_col is None:
        st.error("Please select both a target column and a sensitive attribute.")
    else:
        is_valid, msg = validate_target(df_raw, target_col)
        if not is_valid:
            st.error(f"❌  {msg}")
        else:
            # Store results in session state so other tabs can read them
            if "model_results" not in st.session_state:
                with st.spinner("Running full pipeline…"):
                    col_types = detect_column_types(df_raw)

                    # Step 1 — Preprocess
                    prep = full_preprocessing_pipeline(
                        df_raw, target_col, sensitive_col, col_types, drop_sensitive=False
                    )

                    # Step 2 — Train
                    model = train_model(prep["X_train"], prep["y_train"])

                    # Step 3 — Evaluate
                    eval_results = evaluate_model(model, prep["X_test"], prep["y_test"])

                    # Step 4 — Fairness (needs sensitive column in test set)
                    if prep["sensitive_test"] is not None:
                        fairness_results = calculate_fairness_metrics(
                            eval_results["y_test_enc"],
                            eval_results["y_pred"],
                            prep["X_test"],
                            prep["sensitive_test"],
                        )
                    else:
                        fairness_results = None

                    # Step 5 — Ethical score
                    if fairness_results:
                        eth_score = calculate_ethical_score(
                            eval_results["accuracy"], fairness_results
                        )
                    else:
                        eth_score = None

                    # Step 6 — Bias mitigation (drop sensitive attribute)
                    prep_miti = full_preprocessing_pipeline(
                        df_raw, target_col, sensitive_col, col_types, drop_sensitive=True
                    )
                    model_miti    = train_model(prep_miti["X_train"], prep_miti["y_train"])
                    eval_miti     = evaluate_model(model_miti, prep_miti["X_test"], prep_miti["y_test"])

                    # For mitigation fairness we need sensitive values from the original test split
                    if prep["sensitive_test"] is not None:
                        # Align index lengths (mitigated model may have slightly different test split)
                        s_test = prep["sensitive_test"]
                        y_true_miti = prep_miti["y_test"].reset_index(drop=True)
                        # Use the shorter one
                        min_len = min(len(s_test), len(y_true_miti), len(eval_miti["y_pred"]))
                        fairness_miti = calculate_fairness_metrics(
                            eval_miti["y_test_enc"][:min_len],
                            eval_miti["y_pred"][:min_len],
                            prep_miti["X_test"].iloc[:min_len],
                            s_test.iloc[:min_len],
                        )
                    else:
                        fairness_miti = None

                    if fairness_miti:
                        eth_score_miti = calculate_ethical_score(
                            eval_miti["accuracy"], fairness_miti
                        )
                    else:
                        eth_score_miti = None

                    # Build target_labels map: {0: "Approved", 1: "Rejected"} etc.
                    # Use the raw df values sorted so index matches LabelEncoder order
                    raw_target_vals = sorted(df_raw[target_col].dropna().astype(str).str.strip().unique())
                    target_labels = {i: v for i, v in enumerate(raw_target_vals)}

                    # Compute group approval rates for ethical flagging
                    group_approval_rates = {}
                    if prep["sensitive_test"] is not None:
                        s_train = prep["X_train"][sensitive_col]
                        y_train_pred = model.predict(prep["X_train"])
                        for g_enc in s_train.unique():
                            mask  = (s_train == g_enc).values
                            rate  = float(y_train_pred[mask].mean())
                            try:
                                from sklearn.preprocessing import LabelEncoder
                                g_name = str(g_enc)
                                if hasattr(prep["artifacts"]["encoders"].get(sensitive_col, None), "inverse_transform"):
                                    g_name = prep["artifacts"]["encoders"][sensitive_col].inverse_transform([int(g_enc)])[0]
                            except Exception:
                                g_name = str(g_enc)
                            group_approval_rates[g_name] = round(rate, 4)

                    # feature cols for mitigated model
                    feature_cols_miti = [c for c in prep_miti["X_train"].columns]

                    # Cache everything
                    st.session_state["model_results"] = {
                        "prep":               prep,
                        "model":              model,
                        "eval":               eval_results,
                        "fairness":           fairness_results,
                        "ethical_score":      eth_score,
                        "prep_miti":          prep_miti,
                        "model_miti":         model_miti,
                        "eval_miti":          eval_miti,
                        "fairness_miti":      fairness_miti,
                        "eth_score_miti":     eth_score_miti,
                        "col_types":          col_types,
                        "group_approval_rates": group_approval_rates,
                        "feature_cols_miti":  feature_cols_miti,
                        "target_labels":      target_labels,
                    }
                st.success("✅  Analysis complete!  Explore the tabs above.")

            # Show training summary
            res = st.session_state["model_results"]

            st.markdown("""
            <div class='info-box'>
            <strong>Model:</strong> Logistic Regression &nbsp;|&nbsp;
            <strong>Solver:</strong> lbfgs &nbsp;|&nbsp;
            <strong>Max iterations:</strong> 1000 &nbsp;|&nbsp;
            <strong>Train/Test split:</strong> 80% / 20%
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Training set shape**")
                st.write(f"X_train: `{res['prep']['X_train'].shape}`")
                st.write(f"y_train: `{res['prep']['y_train'].shape}`")
            with col_b:
                st.markdown("**Test set shape**")
                st.write(f"X_test: `{res['prep']['X_test'].shape}`")
                st.write(f"y_test: `{res['prep']['y_test'].shape}`")

            st.markdown("<br>", unsafe_allow_html=True)

            # Feature importance
            feat_fig = plot_feature_importance(
                res["model"], res["prep"]["X_train"].columns.tolist()
            )
            if feat_fig:
                st.markdown("**Feature Importance (Coefficient Magnitude)**")
                st.pyplot(feat_fig, use_container_width=False)


# ===================================================================
# TAB 4 — PERFORMANCE
# ===================================================================
with tab_perf:
    st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.info("Run the analysis first (Model Training tab).")
    else:
        res  = st.session_state["model_results"]
        evl  = res["eval"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{evl['accuracy']:.4f}",  delta=None)
        c2.metric("Precision", f"{evl['precision']:.4f}", delta=None)
        c3.metric("Recall",    f"{evl['recall']:.4f}",    delta=None)
        c4.metric("F1 Score",  f"{evl['f1_score']:.4f}",  delta=None)

        if evl["roc_auc"] != "N/A":
            st.metric("ROC-AUC", f"{evl['roc_auc']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            class_names = [str(c) for c in sorted(res["prep"]["y_test"].unique())]
            cm_fig = plot_confusion_matrix(evl["confusion_matrix"], class_names)
            st.pyplot(cm_fig, use_container_width=False)

        with col_roc:
            if evl["y_pred_prob"] is not None:
                st.markdown("**ROC Curve**")
                roc_fig = plot_roc_curve(evl["y_test_enc"], evl["y_pred_prob"])
                st.pyplot(roc_fig, use_container_width=False)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Classification Report**")
        st.code(evl["classification_report"], language="text")


# ===================================================================
# TAB 5 — FAIRNESS
# ===================================================================
with tab_fair:
    st.markdown("<div class='section-header'>Fairness Analysis</div>", unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.info("Run the analysis first.")
    elif st.session_state["model_results"]["fairness"] is None:
        st.warning(
            "Fairness metrics could not be calculated. "
            "Please check that your sensitive attribute column is valid."
        )
    else:
        res     = st.session_state["model_results"]
        fairness= res["fairness"]

        binarize_note = fairness.get("binarize_note")
        if binarize_note:
            st.warning(
                f"⚠️  **'{sensitive_col}'** has many unique values. "
                f"Auto-grouped into 2 groups: {binarize_note}"
            )

        st.markdown("<div class='info-box'>"
            f"Sensitive attribute: <strong>{sensitive_col}</strong><br>"
            f"Privileged group: <strong>{fairness['spd']['privileged_label']}</strong> &nbsp;|&nbsp; "
            f"Unprivileged group: <strong>{fairness['spd']['unprivileged_label']}</strong>"
            "</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric summary table
        rows = []
        for key in ["spd", "dir", "eod"]:
            m = fairness[key]
            rows.append({
                "Metric":        m["metric"],
                "Value":         m["value"],
                "Ideal":         m["ideal"],
                "Fair Range":    m["fair_range"],
                "Verdict":       "✅ Fair" if m["is_fair"] else "❌ Unfair",
            })
        ind = fairness["individual"]
        rows.append({
            "Metric":     "Individual Fairness Score",
            "Value":      ind["score"],
            "Ideal":      1.0,
            "Fair Range": "≥ 0.8",
            "Verdict":    "✅ Fair" if ind["score"] >= 0.8 else "❌ Unfair",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Group positive rate comparison
        st.markdown("**Group Positive Prediction Rates**")
        gpr_fig = plot_group_positive_rates(fairness["spd"], fairness["dir"])
        st.pyplot(gpr_fig, use_container_width=False)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dashboard grid
        st.markdown("**Fairness Metrics Dashboard**")
        dash_fig = plot_fairness_dashboard(fairness)
        st.pyplot(dash_fig, use_container_width=False)

        # Individual fairness details
        with st.expander("Individual Fairness Details"):
            st.write(f"- Similar pairs checked: **{ind['total_pairs_checked']}**")
            st.write(f"- Similar pairs found (distance < 0.3): **{ind['total_similar_pairs']}**")
            st.write(f"- Inconsistent predictions among similar pairs: **{ind['inconsistent_pairs']}**")
            st.write(f"- Individual Fairness Score: **{ind['score']}**")


# ===================================================================
# TAB 6 — BIAS MITIGATION
# ===================================================================
with tab_miti:
    st.markdown("<div class='section-header'>Bias Mitigation Experiment</div>", unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.info("Run the analysis first.")
    else:
        res         = st.session_state["model_results"]
        evl_orig    = res["eval"]
        evl_miti    = res["eval_miti"]
        fair_orig   = res["fairness"]
        fair_miti   = res["fairness_miti"]

        acc_delta = evl_miti["accuracy"] - evl_orig["accuracy"]

        # explain what mitigation does and what the result means
        if abs(acc_delta) < 0.001:
            mitigation_insight = (
                "✅ <strong>Accuracy is stable after removing the sensitive attribute.</strong> "
                "This tells us education was not contributing meaningful predictive power — "
                "the model was already making decisions based on financial features like "
                "CIBIL score and income. Removing it is safe and reduces bias risk."
            )
        elif acc_delta > 0:
            mitigation_insight = (
                f"📈 <strong>Accuracy improved by {abs(acc_delta):.4f} after mitigation.</strong> "
                "Removing the sensitive attribute actually helped — it was adding noise, not signal."
            )
        else:
            mitigation_insight = (
                f"⚠️ <strong>Accuracy dropped by {abs(acc_delta):.4f} after mitigation.</strong> "
                "There is a fairness-accuracy trade-off here. The sensitive attribute was carrying "
                "predictive signal. Review the fairness metrics below to decide if this trade-off is worth it."
            )

        st.markdown(f"""
        <div class='info-box'>
        <strong>Strategy:</strong> Train a second model with the sensitive attribute
        (<em>education</em>) removed from the feature set and compare both models.<br><br>
        {mitigation_insight}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Accuracy comparison metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Accuracy",  f"{evl_orig['accuracy']:.4f}")
        c2.metric("Mitigated Accuracy", f"{evl_miti['accuracy']:.4f}", delta=f"{acc_delta:+.4f}")
        c3.metric(
            "Accuracy Change",
            "✅ Stable" if abs(acc_delta) <= 0.001
            else ("📈 Improved" if acc_delta > 0.001 else f"📉 Dropped {abs(acc_delta):.4f}"),
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if fair_orig and fair_miti:
            comp_fig = plot_bias_mitigation_comparison(
                fair_orig, fair_miti,
                evl_orig["accuracy"], evl_miti["accuracy"],
            )
            st.pyplot(comp_fig, use_container_width=False)

            st.markdown("<br>", unsafe_allow_html=True)

            # Side-by-side fairness comparison
            st.markdown("**Fairness Metric Comparison**")
            cmp_rows = []
            for key in ["spd", "dir", "eod"]:
                o = fair_orig[key]
                m = fair_miti[key]
                cmp_rows.append({
                    "Metric":          o["metric"],
                    "Original Value":  o["value"],
                    "Mitigated Value": m["value"],
                    "Original Fair":   "✅" if o["is_fair"] else "❌",
                    "Mitigated Fair":  "✅" if m["is_fair"] else "❌",
                })
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Fairness comparison not available — check that the sensitive attribute has 2 unique values.")


# ===================================================================
# TAB 7 — ETHICAL SCORE
# ===================================================================
with tab_score:
    st.markdown("<div class='section-header'>Ethical Score</div>", unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.info("Run the analysis first.")
    elif st.session_state["model_results"]["ethical_score"] is None:
        st.warning("Ethical score could not be computed — fairness metrics unavailable.")
    else:
        res = st.session_state["model_results"]
        eth = res["ethical_score"]
        eth_miti = res["eth_score_miti"]

        # --- Original model score ---
        st.markdown("### Original Model")
        col_gauge, col_info = st.columns([1, 1])

        with col_gauge:
            gauge_fig = plot_ethical_score_gauge(eth)
            st.pyplot(gauge_fig, use_container_width=False)

        with col_info:
            st.markdown(f"""
            <div class='score-card'>
                <h2 style='color:{eth['colour']};font-size:2.5rem;'>{eth['ethical_score']:.3f}</h2>
                <h3 style='color:{eth['colour']};'>{eth['grade']}</h3>
                <hr style='border-color:#3a3a5c;'>
                <p style='color:#aaaaaa;font-size:0.85rem;'>
                Ethical Score = 0.35 × Accuracy<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                + 0.25 × SPD score<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                + 0.25 × DIR score<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                + 0.15 × EOD score
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            comp_data = eth["components"]
            for label, key in [
                ("Accuracy Score",      "accuracy_score"),
                ("SPD Score",           "spd_score"),
                ("DIR Score",           "dir_score"),
                ("EOD Score",           "eod_score"),
            ]:
                val = comp_data[key]
                bar_color = "#4C72B0" if val >= 0.7 else ("#f39c12" if val >= 0.4 else "#e74c3c")
                st.markdown(
                    f"**{label}**: `{val:.3f}` "
                    f"<span style='color:{bar_color};'>{'█' * int(val * 20)}</span>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        comp_fig = plot_score_components(eth)
        st.pyplot(comp_fig, use_container_width=False)

        # --- Mitigated model score comparison ---
        if eth_miti:
            st.divider()
            st.markdown("### Mitigated Model Comparison")

            c1, c2, c3 = st.columns(3)
            delta_score = eth_miti["ethical_score"] - eth["ethical_score"]
            c1.metric("Original Ethical Score",  f"{eth['ethical_score']:.3f}",  delta=eth["grade"])
            c2.metric("Mitigated Ethical Score", f"{eth_miti['ethical_score']:.3f}", delta=f"{delta_score:+.3f}")
            c3.metric(
                "Verdict",
                "Mitigated model is more ethical ✅" if delta_score > 0.01
                else ("Comparable ≈" if abs(delta_score) <= 0.01 else "Original model scored higher"),
            )

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("**Original**")
                st.pyplot(plot_ethical_score_gauge(eth), use_container_width=False)
            with col_g2:
                st.markdown("**Mitigated**")
                st.pyplot(plot_ethical_score_gauge(eth_miti), use_container_width=False)

        # Score interpretation guide
        with st.expander("📖  Score Interpretation Guide"):
            st.markdown("""
            | Score Range | Grade      | Meaning |
            |-------------|------------|---------|
            | 0.85 – 1.00 | Excellent  | Model is both accurate and fair across groups |
            | 0.70 – 0.84 | Good       | Strong performance with minor fairness concerns |
            | 0.55 – 0.69 | Fair       | Acceptable but fairness issues should be investigated |
            | 0.40 – 0.54 | Poor       | Significant bias or accuracy problems present |
            | 0.00 – 0.39 | Critical   | Model should not be deployed without mitigation |
            """)


# ===================================================================
# TAB 8 — PREDICT ON NEW DATA
# ===================================================================
with tab_predict:
    st.markdown("<div class='section-header'>🔮 Predict on New / Unseen Data</div>", unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.info("👈  Run the analysis first (Model Training tab) — the model needs to be trained before predicting.")
    else:
        res = st.session_state["model_results"]

        st.markdown("""
        <div class='info-box'>
        The model has been trained. Enter details of a new applicant below and click
        <strong>Predict</strong> to see the outcome. The result includes a prediction,
        confidence score, and an <strong>ethical flag</strong> that checks whether
        the sensitive attribute is influencing the decision.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── collect feature columns and metadata ──────────────────────────
        feature_cols      = list(res["prep"]["X_train"].columns)
        feature_cols_miti = res.get("feature_cols_miti", [c for c in feature_cols if c != sensitive_col])
        encoders          = res["prep"]["artifacts"]["encoders"]
        scaler            = res["prep"]["artifacts"]["scaler"]
        cols_to_scale     = res["prep"]["artifacts"]["scaled_cols"]
        group_ap_rates    = res.get("group_approval_rates", {})

        # figure out column types for building the form
        col_types_info = res["col_types"]
        num_cols_form  = [c for c in col_types_info.get("numerical",   []) if c in feature_cols and c != target_col]
        cat_cols_form  = [c for c in col_types_info.get("categorical", []) if c in feature_cols and c != target_col]

        # ── SINGLE APPLICANT FORM ─────────────────────────────────────────
        st.markdown("### Single Applicant Prediction")

        with st.form("predict_form"):
            st.markdown("**Fill in the applicant details:**")
            st.markdown("<br>", unsafe_allow_html=True)

            applicant_input = {}

            # numerical fields — two per row
            if num_cols_form:
                st.markdown("**Numerical Features**")
                num_pairs = [num_cols_form[i:i+2] for i in range(0, len(num_cols_form), 2)]
                for pair in num_pairs:
                    cols_ui = st.columns(len(pair))
                    for col_ui, col_name in zip(cols_ui, pair):
                        with col_ui:
                            # sensible default: median of training data
                            try:
                                default_val = float(
                                    res["prep"]["processed_df"][col_name].median()
                                )
                            except Exception:
                                default_val = 0.0
                            applicant_input[col_name] = st.number_input(
                                col_name.replace("_", " ").title(),
                                value=default_val,
                                key=f"pred_{col_name}",
                            )

            st.markdown("<br>", unsafe_allow_html=True)

            # categorical fields — dropdowns using known values from training
            if cat_cols_form:
                st.markdown("**Categorical Features**")
                cat_pairs = [cat_cols_form[i:i+2] for i in range(0, len(cat_cols_form), 2)]
                for pair in cat_pairs:
                    cols_ui = st.columns(len(pair))
                    for col_ui, col_name in zip(cols_ui, pair):
                        with col_ui:
                            if col_name in encoders:
                                options = [str(c).strip() for c in encoders[col_name].classes_.tolist()]
                            else:
                                options = list(df_raw[col_name].dropna().unique()) if col_name in df_raw.columns else ["Unknown"]
                            applicant_input[col_name] = st.selectbox(
                                col_name.replace("_", " ").title(),
                                options=options,
                                key=f"pred_cat_{col_name}",
                            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮  Predict", type="primary", use_container_width=True)

        # ── SHOW RESULT ───────────────────────────────────────────────────
        if submitted:
            try:
                result = predict_applicant(
                    applicant       = applicant_input,
                    model           = res["model"],
                    model_miti      = res["model_miti"],
                    feature_cols    = feature_cols,
                    feature_cols_miti = feature_cols_miti,
                    encoders        = encoders,
                    scaler          = scaler,
                    cols_to_scale   = cols_to_scale,
                    cat_cols        = cat_cols_form,
                    sensitive_col   = sensitive_col,
                    target_col      = target_col,
                    group_approval_rates = group_ap_rates,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Result")

                # colour based on prediction
                pred_label = result["prediction"]
                pred_lower = str(pred_label).lower()
                if any(w in pred_lower for w in ["approved", "yes", "1", "true", "accept"]):
                    pred_color = "#2ecc71"
                    pred_icon  = "✅"
                else:
                    pred_color = "#e74c3c"
                    pred_icon  = "❌"

                # flag colour
                flag_colors = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}
                flag_color  = flag_colors.get(result["flag_level"], "#aaaaaa")

                # big result card
                col_pred, col_conf, col_flag = st.columns(3)

                with col_pred:
                    st.markdown(f"""
                    <div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:12px;
                                padding:20px;text-align:center;'>
                        <div style='font-size:0.85rem;color:#aaaaaa;margin-bottom:6px;'>PREDICTION</div>
                        <div style='font-size:2rem;font-weight:700;color:{pred_color};'>{pred_icon} {pred_label}</div>
                        <div style='font-size:0.8rem;color:#aaaaaa;margin-top:6px;'>
                            Group: <strong>{result['group']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_conf:
                    bar_width = int(result["confidence"] * 100)
                    st.markdown(f"""
                    <div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:12px;
                                padding:20px;text-align:center;'>
                        <div style='font-size:0.85rem;color:#aaaaaa;margin-bottom:6px;'>CONFIDENCE</div>
                        <div style='font-size:2rem;font-weight:700;color:#4C72B0;'>{result['confidence_pct']}</div>
                        <div style='background:#2a2a4e;border-radius:4px;height:8px;margin-top:10px;'>
                            <div style='background:#4C72B0;width:{bar_width}%;height:8px;border-radius:4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_flag:
                    st.markdown(f"""
                    <div style='background:#1a1a2e;border:1px solid {flag_color};border-radius:12px;
                                padding:20px;text-align:center;'>
                        <div style='font-size:0.85rem;color:#aaaaaa;margin-bottom:6px;'>ETHICAL FLAG</div>
                        <div style='font-size:1.6rem;font-weight:700;color:{flag_color};'>
                            {result['flag_level']}
                        </div>
                        <div style='font-size:0.75rem;color:{flag_color};margin-top:4px;'>
                            Fair model: {result['prediction_miti']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # recommendation banner
                st.markdown(f"""
                <div style='background:#1a1a2e;border-left:4px solid {flag_color};
                            border-radius:4px;padding:14px 18px;font-size:1rem;'>
                    {result['recommendation']}
                </div>
                """, unsafe_allow_html=True)

                # ethical flags detail
                if result["ethical_flags"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Ethical Flag Details:**")
                    for flag in result["ethical_flags"]:
                        st.warning(flag)
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success("✅  No ethical concerns detected for this applicant.")

                # probability breakdown
                with st.expander("📊  Probability Breakdown"):
                    try:
                        class_names = encoders[target_col].classes_.tolist()
                    except Exception:
                        class_names = ["Class 0", "Class 1"]

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    probs  = [result["prob_class_0"], result["prob_class_1"]]
                    colors = ["#DD8452", "#4C72B0"]
                    bars   = ax.barh(class_names, probs, color=colors,
                                     edgecolor="white", linewidth=0.6)
                    for bar, val in zip(bars, probs):
                        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                f"{val:.1%}", va="center", fontsize=10,
                                color="#e0e0e0", fontweight="bold")
                    ax.set_xlim(0, 1.15)
                    ax.set_xlabel("Probability", color="#e0e0e0", fontsize=9)
                    ax.set_title("Class Probabilities", color="#e0e0e0", fontsize=10, pad=8)
                    ax.tick_params(colors="#e0e0e0")
                    fig.patch.set_facecolor("#1a1a2e")
                    ax.set_facecolor("#16213e")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#3a3a5c")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # ── BATCH PREDICTION ──────────────────────────────────────────────
        st.divider()
        st.markdown("### Batch Prediction — Upload Multiple Applicants")

        st.markdown("""
        <div class='info-box'>
        Upload a CSV with multiple new applicants (same columns as the training data,
        without the target column). The model will predict all of them at once and
        return a results table with ethical flags.
        </div>
        """, unsafe_allow_html=True)

        batch_file = st.file_uploader(
            "Upload new applicants CSV",
            type=["csv"],
            key="batch_upload",
        )

        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                batch_df.columns = (
                    batch_df.columns.str.strip().str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace(r"[^\w]", "_", regex=True)
                )

                # drop target and id columns if present
                drop_cols = [target_col, "loan_id", "id"]
                batch_df.drop(columns=[c for c in drop_cols if c in batch_df.columns],
                              errors="ignore", inplace=True)

                st.markdown(f"**Uploaded:** {len(batch_df)} applicants — preview:")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀  Run Batch Prediction", type="primary"):
                    with st.spinner("Predicting..."):
                        batch_results = predict_batch(
                            applicants_df     = batch_df,
                            model             = res["model"],
                            model_miti        = res["model_miti"],
                            feature_cols      = feature_cols,
                            feature_cols_miti = feature_cols_miti,
                            encoders          = encoders,
                            scaler            = scaler,
                            cols_to_scale     = cols_to_scale,
                            cat_cols          = cat_cols_form,
                            sensitive_col     = sensitive_col,
                            target_col        = target_col,
                            group_approval_rates = group_ap_rates,
                            label_encoder     = res["eval"].get("label_encoder"),
                            target_labels     = res.get("target_labels"),
                        )

                    # ── summary stats ────────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    total     = len(batch_results)
                    approved  = (batch_results["Prediction"].astype(str).str.lower().str.contains(
                                    "approv|yes|accept|1", na=False)).sum()
                    red_flags = (batch_results["Bias Flag"] == "RED").sum()
                    yellow    = (batch_results["Bias Flag"] == "YELLOW").sum()

                    c1.metric("Total Applicants", total)
                    c2.metric("✅ Approved",       int(approved))
                    c3.metric("🔴 Bias Detected",  int(red_flags))
                    c4.metric("🟡 Needs Review",   int(yellow))

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── highlight biased rows ─────────────────────────────
                    def highlight_row(row):
                        flag = row.get("Bias Flag", "GREEN")
                        if flag == "RED":
                            return ["background-color:#3a1a1a"] * len(row)
                        elif flag == "YELLOW":
                            return ["background-color:#3a2e1a"] * len(row)
                        return [""] * len(row)

                    def highlight_flag_cell(val):
                        colors_map = {
                            "GREEN":  "background-color:#1a3a2a;color:#2ecc71;font-weight:bold",
                            "YELLOW": "background-color:#3a2e1a;color:#f39c12;font-weight:bold",
                            "RED":    "background-color:#3a1a1a;color:#e74c3c;font-weight:bold",
                        }
                        return colors_map.get(val, "")

                    def highlight_pred_cell(val):
                        v = str(val).lower()
                        if "approv" in v or "yes" in v or "accept" in v:
                            return "color:#2ecc71;font-weight:bold"
                        elif "reject" in v or "no" in v or "deny" in v:
                            return "color:#e74c3c;font-weight:bold"
                        return ""

                    styled = (
                        batch_results.style
                        .apply(highlight_row, axis=1)
                        .applymap(highlight_flag_cell,  subset=["Bias Flag"])
                        .applymap(highlight_pred_cell,  subset=["Prediction", "Fair Model"])
                    )

                    st.markdown("**Full Results — all applicant data with predictions and ethical flags:**")
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # ── highlight who is biased ───────────────────────────
                    biased = batch_results[batch_results["Bias Flag"] == "RED"]
                    if len(biased) > 0:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.error(f"🚨  **{len(biased)} applicant(s) show potential bias** — the model's decision changes when the sensitive attribute is removed.")
                        for _, br in biased.iterrows():
                            st.markdown(
                                f"- **Applicant #{int(br['Applicant #'])}** "
                                f"({br.get(sensitive_col, '')}): "
                                f"Original → **{br['Prediction']}** | "
                                f"Without {sensitive_col} → **{br['Fair Model']}** | "
                                f"Reason: {br['Bias Reason']}"
                            )

                    review = batch_results[batch_results["Bias Flag"] == "YELLOW"]
                    if len(review) > 0:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning(f"🟡  **{len(review)} applicant(s) flagged for human review** — low confidence or disadvantaged group concern.")
                        for _, rr in review.iterrows():
                            st.markdown(
                                f"- **Applicant #{int(rr['Applicant #'])}** "
                                f"({rr.get(sensitive_col, '')}): "
                                f"Confidence {rr['Confidence']} | "
                                f"Reason: {rr['Bias Reason']}"
                            )

                    # ── download as CSV ───────────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    csv_data = batch_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️  Download Full Results as CSV",
                        data=csv_data,
                        file_name="ethix_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
