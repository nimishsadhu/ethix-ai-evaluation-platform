"""
fairness.py
-----------
Fairness and bias measurement.

We implement three classic group-fairness metrics:

  1. Statistical Parity Difference (SPD)
     Ideal value = 0   (both groups predicted positive at the same rate)
     Concerning   < -0.1 or > 0.1

  2. Disparate Impact Ratio (DIR)
     Ideal value = 1.0  (equal positive rates)
     The 80% rule says DIR should be >= 0.8 to be considered fair.

  3. Equal Opportunity Difference (EOD)
     Compares True Positive Rates across groups.
     Ideal value = 0

We also include a simple individual-fairness check that measures
how often pairs of "similar" people get different predictions.

IMPORTANT: If the sensitive attribute has more than 2 unique values
(e.g. a numeric column like cibil_score), we automatically split it
into two groups using the median as the cut-off point.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

TEXT_COLOR = "#FAFAFA"


# ---------------------------------------------------------------------------
# Core metric calculations
# ---------------------------------------------------------------------------

def _binarize_sensitive(sensitive_values: pd.Series) -> pd.Series:
    """
    Make sure the sensitive attribute has exactly 2 groups.

    - If it already has 2 unique values  -> return as-is
    - If it is numeric with >2 values    -> split at median (below vs above)
    - If it is categorical with >2 values-> keep top-2 most frequent categories
    """
    sensitive_values = sensitive_values.reset_index(drop=True)
    n_unique = sensitive_values.nunique()

    if n_unique == 2:
        # Perfect — nothing to do
        return sensitive_values, None

    if n_unique < 2:
        raise ValueError("Sensitive attribute has fewer than 2 unique values — cannot compute fairness.")

    # More than 2 values
    if pd.api.types.is_numeric_dtype(sensitive_values):
        # Split at the median: "below median" vs "above median"
        median_val = sensitive_values.median()
        binarized  = sensitive_values.apply(
            lambda x: f"<= {median_val:.0f}" if x <= median_val else f"> {median_val:.0f}"
        )
        note = f"Numeric column split at median ({median_val:.0f})"
    else:
        # Keep only the 2 most common categories
        top2      = sensitive_values.value_counts().head(2).index.tolist()
        mask      = sensitive_values.isin(top2)
        binarized = sensitive_values.copy()
        binarized[~mask] = top2[1]   # assign rare categories to the second group
        note = f"Categorical column — kept top-2 groups: {top2}"

    return binarized, note


def _get_group_masks(sensitive_values: pd.Series):
    """
    Split (already binarized) sensitive attribute into two group masks.
    Majority group  = 'privileged'
    Minority group  = 'unprivileged'
    """
    counts           = sensitive_values.value_counts()
    privileged_val   = counts.index[0]
    unprivileged_val = counts.index[1]

    mask_priv   = sensitive_values == privileged_val
    mask_unpriv = sensitive_values == unprivileged_val

    return mask_priv, mask_unpriv, privileged_val, unprivileged_val


def statistical_parity_difference(
    y_pred: np.ndarray, sensitive_values: pd.Series
) -> dict:
    """
    SPD = P(Y_hat=1 | unprivileged) - P(Y_hat=1 | privileged)

    Positive SPD -> unprivileged group gets more positive predictions.
    Negative SPD -> unprivileged group gets fewer positive predictions.
    """
    sensitive_values, _ = _binarize_sensitive(sensitive_values)
    mask_priv, mask_unpriv, priv_val, unpriv_val = _get_group_masks(sensitive_values)

    y_pred_arr  = np.array(y_pred, dtype=float)
    mp           = np.array(mask_priv)
    mup          = np.array(mask_unpriv)

    rate_priv   = float(y_pred_arr[mp].mean())  if mp.sum()  > 0 else 0.0
    rate_unpriv = float(y_pred_arr[mup].mean()) if mup.sum() > 0 else 0.0

    spd = rate_unpriv - rate_priv

    return {
        "metric":            "Statistical Parity Difference (SPD)",
        "value":             round(spd, 4),
        "rate_privileged":   round(rate_priv,   4),
        "rate_unprivileged": round(rate_unpriv, 4),
        "privileged_label":  str(priv_val),
        "unprivileged_label":str(unpriv_val),
        "ideal":             0,
        "fair_range":        "Between -0.1 and 0.1",
        "is_fair":           abs(spd) <= 0.1,
    }


def disparate_impact_ratio(
    y_pred: np.ndarray, sensitive_values: pd.Series
) -> dict:
    """
    DIR = P(Y_hat=1 | unprivileged) / P(Y_hat=1 | privileged)

    A DIR below 0.8 is typically flagged as discriminatory (the '80% rule').
    """
    sensitive_values, _ = _binarize_sensitive(sensitive_values)
    mask_priv, mask_unpriv, priv_val, unpriv_val = _get_group_masks(sensitive_values)

    y_pred_arr  = np.array(y_pred, dtype=float)
    mp           = np.array(mask_priv)
    mup          = np.array(mask_unpriv)

    rate_priv   = float(y_pred_arr[mp].mean())  if mp.sum()  > 0 else 1e-9
    rate_unpriv = float(y_pred_arr[mup].mean()) if mup.sum() > 0 else 0.0

    dir_value = float(rate_unpriv / rate_priv) if rate_priv > 0 else 0.0

    return {
        "metric":             "Disparate Impact Ratio (DIR)",
        "value":              round(dir_value, 4),
        "rate_privileged":    round(rate_priv,   4),
        "rate_unprivileged":  round(rate_unpriv, 4),
        "privileged_label":   str(priv_val),
        "unprivileged_label": str(unpriv_val),
        "ideal":              1.0,
        "fair_range":         ">= 0.8  (80% rule)",
        "is_fair":            dir_value >= 0.8,
    }


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_values: pd.Series,
) -> dict:
    """
    EOD = TPR(unprivileged) - TPR(privileged)

    Measures whether the model catches positive cases equally well
    across both groups.
    """
    sensitive_values, _ = _binarize_sensitive(sensitive_values)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask_priv, mask_unpriv, priv_val, unpriv_val = _get_group_masks(sensitive_values)

    def tpr(mask):
        mask_arr  = np.array(mask)
        positives = (y_true[mask_arr] == 1)
        if positives.sum() == 0:
            return 0.0
        return float((y_pred[mask_arr][positives] == 1).sum() / positives.sum())

    tpr_priv   = tpr(mask_priv)
    tpr_unpriv = tpr(mask_unpriv)
    eod        = tpr_unpriv - tpr_priv

    return {
        "metric":             "Equal Opportunity Difference (EOD)",
        "value":              round(eod, 4),
        "tpr_privileged":     round(tpr_priv,   4),
        "tpr_unprivileged":   round(tpr_unpriv, 4),
        "privileged_label":   str(priv_val),
        "unprivileged_label": str(unpriv_val),
        "ideal":              0,
        "fair_range":         "Between -0.1 and 0.1",
        "is_fair":            abs(eod) <= 0.1,
    }


def individual_fairness_score(
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    n_pairs: int = 500,
    similarity_threshold: float = 0.3,
) -> dict:
    """
    Rough individual-fairness estimate.

    Randomly sample pairs of individuals, measure Euclidean distance,
    and check whether similar pairs (distance < threshold) get the
    same prediction.
    """
    n = len(X_test)
    if n < 2:
        return {"score": 1.0, "inconsistent_pairs": 0, "total_similar_pairs": 0, "total_pairs_checked": 0}

    X_arr = X_test.values.astype(float)
    rng   = np.random.default_rng(42)

    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    same  = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n

    dists         = np.linalg.norm(X_arr[idx_a] - X_arr[idx_b], axis=1)
    similar_mask  = dists < similarity_threshold
    y_pred_arr    = np.array(y_pred)
    inconsistent  = int((y_pred_arr[idx_a][similar_mask] != y_pred_arr[idx_b][similar_mask]).sum())
    total_similar = int(similar_mask.sum())

    score = 1.0 - (inconsistent / total_similar) if total_similar > 0 else 1.0

    return {
        "score":               round(float(score), 4),
        "inconsistent_pairs":  inconsistent,
        "total_similar_pairs": total_similar,
        "total_pairs_checked": n_pairs,
    }


def calculate_fairness_metrics(
    y_true, y_pred, X_test, sensitive_values
) -> dict:
    """
    Master function: compute all four fairness metrics.

    Automatically handles:
    - String labels (Approved/Rejected, <=50K/>50K etc)
    - Numeric sensitive attributes (split at median)
    - Categorical sensitive attributes with >2 values (keep top 2)
    """
    from sklearn.preprocessing import LabelEncoder

    # Encode predictions and true labels to 0/1 integers
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if y_pred.dtype == object or y_pred.dtype.kind in ("U", "S"):
        le = LabelEncoder()
        combined = np.concatenate([y_true.ravel(), y_pred.ravel()])
        le.fit(combined)
        y_pred = le.transform(y_pred.ravel())
        y_true = le.transform(y_true.ravel())
    else:
        y_pred = y_pred.astype(float)
        y_true = y_true.astype(float)

    # Binarize sensitive attribute (handles numeric columns with many values)
    sensitive_series = pd.Series(sensitive_values).reset_index(drop=True)
    sensitive_bin, binarize_note = _binarize_sensitive(sensitive_series)

    spd  = statistical_parity_difference(y_pred, sensitive_bin)
    dir_ = disparate_impact_ratio(y_pred, sensitive_bin)
    eod  = equal_opportunity_difference(y_true, y_pred, sensitive_bin)
    ind  = individual_fairness_score(X_test, y_pred)

    return {
        "spd":            spd,
        "dir":            dir_,
        "eod":            eod,
        "individual":     ind,
        "binarize_note":  binarize_note,   # shown in UI if column was auto-grouped
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _dark_ax(fig, ax):
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")


def plot_group_positive_rates(spd_result: dict, dir_result: dict) -> plt.Figure:
    """
    Side-by-side bar chart comparing positive prediction rates
    for the privileged vs. unprivileged group.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Privileged", "Unprivileged"]
    colors = ["#4C72B0", "#DD8452"]

    for ax, result, title in zip(
        axes,
        [spd_result, dir_result],
        ["Positive Prediction Rates (SPD)", "Positive Prediction Rates (DIR)"],
    ):
        rates = [result["rate_privileged"], result["rate_unprivileged"]]
        bars  = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

        for bar, val in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=10, color=TEXT_COLOR, fontweight="bold",
            )

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylim(0, min(1.2, max(rates) * 1.4 + 0.05))
        ax.set_ylabel("Positive Rate", fontsize=9)
        _dark_ax(fig, ax)

    fig.tight_layout()
    return fig


def plot_fairness_dashboard(metrics: dict) -> plt.Figure:
    """
    A 2x2 grid showing all four fairness metric values with
    colour-coded fair/unfair indicators.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Fairness Metrics Dashboard", fontsize=14, color=TEXT_COLOR, y=1.01)
    fig.patch.set_facecolor("#1a1a2e")

    ifs_score = metrics["individual"]["score"]

    metric_configs = [
        (axes[0, 0], metrics["spd"],  "SPD", (-0.1, 0.1)),
        (axes[0, 1], metrics["dir"],  "DIR", (0.8,  1.2)),
        (axes[1, 0], metrics["eod"],  "EOD", (-0.1, 0.1)),
        (axes[1, 1], None,            "IFS", (0.8,  1.0)),
    ]

    full_names = {
        "SPD": "Statistical Parity Diff.",
        "DIR": "Disparate Impact Ratio",
        "EOD": "Equal Opportunity Diff.",
        "IFS": "Individual Fairness Score",
    }

    for ax, result, label, fair_range in metric_configs:
        if result is None:
            value   = ifs_score
            is_fair = ifs_score >= 0.8
        else:
            value   = result["value"]
            is_fair = result["is_fair"]

        color   = "#55A868" if is_fair else "#C44E52"
        verdict = "FAIR ✓"  if is_fair else "UNFAIR ✗"

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 0.65, f"{value:.3f}", ha="center", va="center",
                fontsize=28, color=color, fontweight="bold",
                transform=ax.transAxes)

        ax.text(0.5, 0.90, full_names[label], ha="center", va="center",
                fontsize=10, color=TEXT_COLOR, fontweight="bold",
                transform=ax.transAxes)

        ax.text(0.5, 0.40,
                f"Fair range: {fair_range[0]} to {fair_range[1]}",
                ha="center", va="center", fontsize=8, color="#aaaaaa",
                transform=ax.transAxes)

        ax.text(0.5, 0.20, verdict, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "33", edgecolor=color))

        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a5c")

    fig.tight_layout()
    return fig


def plot_bias_mitigation_comparison(
    original_metrics: dict,
    mitigated_metrics: dict,
    original_accuracy: float,
    mitigated_accuracy: float,
) -> plt.Figure:
    """
    Bar chart comparing original vs mitigated model on accuracy and fairness.
    Accuracy axis is zoomed to show meaningful differences clearly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # --- Left: accuracy comparison ---
    ax = axes[0]
    labels = ["Original\n(with education)", "Mitigated\n(without education)"]
    values = [original_accuracy, mitigated_accuracy]
    colors = ["#4C72B0", "#55A868"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.45)

    # annotate each bar with exact value and change
    delta = mitigated_accuracy - original_accuracy
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0015,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, color=TEXT_COLOR, fontweight="bold")

    # delta annotation between bars
    arrow_x = 1
    if abs(delta) < 0.0005:
        verdict = "No change\neducation was not\ninfluencing accuracy"
        verdict_color = "#f39c12"
    elif delta > 0:
        verdict = f"Accuracy improved\nby {abs(delta):.4f}"
        verdict_color = "#2ecc71"
    else:
        verdict = f"Accuracy dropped\nby {abs(delta):.4f}"
        verdict_color = "#e74c3c"

    ax.text(0.5, 0.08, verdict,
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=9, color=verdict_color, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                      edgecolor=verdict_color, alpha=0.9))

    # zoom Y axis to make difference visible — start from 85% of min value
    y_min = min(values) * 0.985
    y_max = max(values) * 1.02
    ax.set_ylim(y_min, y_max + 0.01)
    ax.set_title("Accuracy: Original vs Mitigated", fontsize=11, pad=8)
    ax.set_ylabel("Accuracy", fontsize=9)
    _dark_ax(fig, ax)

    # --- Right: fairness metrics comparison ---
    ax = axes[1]
    metric_labels = ["SPD (abs)", "DIR", "EOD (abs)"]
    orig_vals = [
        abs(original_metrics["spd"]["value"]),
        original_metrics["dir"]["value"],
        abs(original_metrics["eod"]["value"]),
    ]
    miti_vals = [
        abs(mitigated_metrics["spd"]["value"]),
        mitigated_metrics["dir"]["value"],
        abs(mitigated_metrics["eod"]["value"]),
    ]

    x     = np.arange(len(metric_labels))
    width = 0.32

    b1 = ax.bar(x - width/2, orig_vals, width, label="Original",
                color="#4C72B0", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, miti_vals, width, label="Mitigated",
                color="#55A868", edgecolor="white", linewidth=0.5)

    # value labels on bars
    for bar, val in zip(b1, orig_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                color=TEXT_COLOR, fontweight="bold")
    for bar, val in zip(b2, miti_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                color=TEXT_COLOR, fontweight="bold")

    # fair range reference lines
    ax.axhline(y=0.1,  color="#e74c3c", linestyle="--", linewidth=0.8,
               alpha=0.6, label="SPD/EOD fair limit (0.1)")
    ax.axhline(y=0.8,  color="#f39c12", linestyle="--", linewidth=0.8,
               alpha=0.6, label="DIR fair threshold (0.8)")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9, color=TEXT_COLOR)
    ax.set_title("Fairness Metrics: Original vs Mitigated", fontsize=11, pad=8)
    ax.set_ylabel("Metric Value", fontsize=9)

    legend = ax.legend(fontsize=7, loc="upper right")
    legend.get_frame().set_facecolor("#1a1a2e")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _dark_ax(fig, ax)
    fig.suptitle("Bias Mitigation — Removing the Sensitive Attribute",
                 fontsize=12, color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    return fig
