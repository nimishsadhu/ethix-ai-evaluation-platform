"""
ethical_score.py
----------------
Computes a single composite Ethical Score for the model.

The formula combines:
  - Model accuracy           (weight: 0.35)  — performance matters
  - SPD fairness component   (weight: 0.25)  — statistical parity
  - DIR fairness component   (weight: 0.25)  — disparate impact
  - EOD fairness component   (weight: 0.15)  — equal opportunity

Each fairness metric is converted into a 0–1 "goodness" score before
being combined, so the final Ethical Score is always in [0, 1].

Score interpretation
  ≥ 0.85   → Excellent   (green)
  ≥ 0.70   → Good        (light green)
  ≥ 0.55   → Fair        (yellow)
  ≥ 0.40   → Poor        (orange)
  < 0.40   → Critical    (red)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

TEXT_COLOR = "#FAFAFA"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _spd_to_score(spd_value: float) -> float:
    """
    SPD in [-1, 1]; ideal = 0.
    Score = 1 when SPD == 0, falls linearly to 0 at ±0.5.
    """
    return max(0.0, 1.0 - abs(spd_value) / 0.5)


def _dir_to_score(dir_value: float) -> float:
    """
    DIR ≥ 0; ideal = 1.
    Penalty kicks in below 0.8, reward above 1.0 is capped.
    """
    if dir_value >= 1.0:
        # Small bonus for super-fair models, max score
        return min(1.0, 0.8 + dir_value * 0.2)
    elif dir_value >= 0.8:
        # Linearly between 0.6 and 1.0
        return 0.6 + (dir_value - 0.8) / 0.2 * 0.4
    else:
        # Below 0.8: linearly from 0.6 down to 0
        return max(0.0, dir_value / 0.8 * 0.6)


def _eod_to_score(eod_value: float) -> float:
    """
    EOD in [-1, 1]; ideal = 0.
    Same shape as SPD score.
    """
    return max(0.0, 1.0 - abs(eod_value) / 0.5)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_ethical_score(
    accuracy: float,
    fairness_metrics: dict,
    weights: dict = None,
) -> dict:
    """
    Combine performance and fairness into a single Ethical Score.

    Parameters
    ----------
    accuracy        : float in [0, 1]
    fairness_metrics: dict returned by fairness.calculate_fairness_metrics()
    weights         : optional dict overriding default component weights

    Returns a dict with individual component scores and the final score.
    """
    if weights is None:
        weights = {
            "accuracy": 0.35,
            "spd":      0.25,
            "dir":      0.25,
            "eod":      0.15,
        }

    spd_value = fairness_metrics["spd"]["value"]
    dir_value = fairness_metrics["dir"]["value"]
    eod_value = fairness_metrics["eod"]["value"]

    spd_score = _spd_to_score(spd_value)
    dir_score = _dir_to_score(dir_value)
    eod_score = _eod_to_score(eod_value)

    # Accuracy is already in [0, 1]
    ethical_score = (
        weights["accuracy"] * accuracy
        + weights["spd"]      * spd_score
        + weights["dir"]      * dir_score
        + weights["eod"]      * eod_score
    )
    ethical_score = round(float(ethical_score), 4)

    # Interpret the score
    if   ethical_score >= 0.85:
        grade, colour = "Excellent", "#2ecc71"
    elif ethical_score >= 0.70:
        grade, colour = "Good",      "#27ae60"
    elif ethical_score >= 0.55:
        grade, colour = "Fair",      "#f39c12"
    elif ethical_score >= 0.40:
        grade, colour = "Poor",      "#e67e22"
    else:
        grade, colour = "Critical",  "#e74c3c"

    return {
        "ethical_score": ethical_score,
        "grade":         grade,
        "colour":        colour,
        "components": {
            "accuracy_score": round(accuracy,    4),
            "spd_score":      round(spd_score,   4),
            "dir_score":      round(dir_score,   4),
            "eod_score":      round(eod_score,   4),
        },
        "weights": weights,
        "raw_metrics": {
            "spd": spd_value,
            "dir": dir_value,
            "eod": eod_value,
        },
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_ethical_score_gauge(score_result: dict) -> plt.Figure:
    """
    A large gauge / speedometer visual for the Ethical Score.
    """
    score  = score_result["ethical_score"]
    grade  = score_result["grade"]
    colour = score_result["colour"]

    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), lw=22, color="#2a2a4e", solid_capstyle="round")

    # Coloured arc (progress)
    progress_theta = np.linspace(np.pi, np.pi - score * np.pi, 300)
    ax.plot(np.cos(progress_theta), np.sin(progress_theta), lw=22, color=colour,
            solid_capstyle="round", alpha=0.9)

    # Zone markers
    zone_angles = [0, 0.40 * np.pi, 0.55 * np.pi, 0.70 * np.pi, 0.85 * np.pi, np.pi]
    zone_colors = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#2ecc71"]
    zone_labels = ["0.0", "0.40", "0.55", "0.70", "0.85", "1.0"]
    for angle, label in zip(zone_angles, zone_labels):
        a = np.pi - angle
        ax.text(1.18 * np.cos(a), 1.18 * np.sin(a), label,
                ha="center", va="center", fontsize=7.5, color="#aaaaaa")

    # Centre score text
    ax.text(0, 0.20, f"{score:.3f}",
            ha="center", va="center",
            fontsize=40, fontweight="bold", color=colour)
    ax.text(0, -0.15, grade,
            ha="center", va="center",
            fontsize=16, color=colour, fontweight="bold")
    ax.text(0, -0.38, "Ethical Score",
            ha="center", va="center",
            fontsize=11, color=TEXT_COLOR, alpha=0.7)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.55, 1.4)
    fig.tight_layout()
    return fig


def plot_score_components(score_result: dict) -> plt.Figure:
    """
    Horizontal bar chart showing the weighted contribution of each
    component to the final Ethical Score.
    """
    components = score_result["components"]
    weights    = score_result["weights"]

    labels = [
        "Accuracy",
        "Stat. Parity\n(SPD)",
        "Disparate Impact\n(DIR)",
        "Equal Opportunity\n(EOD)",
    ]
    raw_scores   = [
        components["accuracy_score"],
        components["spd_score"],
        components["dir_score"],
        components["eod_score"],
    ]
    weight_keys  = ["accuracy", "spd", "dir", "eod"]
    contributions= [raw_scores[i] * weights[k] for i, k in enumerate(weight_keys)]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: raw component scores
    ax = axes[0]
    bars = ax.barh(labels, raw_scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, raw_scores):
        ax.text(min(val + 0.02, 1.12), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_COLOR)
    ax.set_xlim(0, 1.2)
    ax.set_title("Component Scores (0 – 1)", fontsize=10, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("Score", fontsize=9, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_facecolor("#16213e")
    fig.patch.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")

    # Right: weighted contributions
    ax = axes[1]
    bars = ax.barh(labels, contributions, color=colors, edgecolor="white", linewidth=0.5)
    total = sum(contributions)
    for bar, val in zip(bars, contributions):
        ax.text(min(val + 0.005, total * 1.12), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_COLOR)
    ax.set_xlim(0, total * 1.25)
    ax.set_title("Weighted Contributions to Ethical Score", fontsize=10, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("Contribution", fontsize=9, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")

    # Vertical line showing total score
    ax.axvline(total, color=score_result["colour"], linewidth=2, linestyle="--", alpha=0.8,
               label=f"Total: {score_result['ethical_score']:.3f}")
    legend = ax.legend(fontsize=9)
    legend.get_frame().set_facecolor("#1a1a2e")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    return fig
