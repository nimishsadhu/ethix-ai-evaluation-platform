"""
eda.py
------
Exploratory Data Analysis helpers.
Every function returns a Matplotlib Figure so Streamlit can render it
with st.pyplot() — no side effects, easy to test.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- shared colour palette ---
PALETTE      = "Set2"
ACCENT_COLOR = "#4C72B0"
BG_COLOR     = "#0E1117"      # Streamlit dark background
TEXT_COLOR   = "#FAFAFA"

def _apply_dark_style(fig, axes_list):
    """Apply a clean dark-mode style to all axes in the figure."""
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes_list:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a5c")


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """
    Simple bar chart showing how many rows belong to each target class.
    Imbalanced classes are immediately visible here.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    value_counts = df[target_col].value_counts()
    colors = ["#4C72B0", "#DD8452"]

    bars = ax.bar(
        [str(v) for v in value_counts.index],
        value_counts.values,
        color=colors[:len(value_counts)],
        edgecolor="white",
        linewidth=0.8,
    )

    # Label each bar with count and percentage
    total = len(df)
    for bar, count in zip(bars, value_counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR,
        )

    ax.set_title(f"Target Distribution — {target_col}", fontsize=12, pad=10)
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.margins(y=0.15)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


def plot_categorical_vs_target(
    df: pd.DataFrame, cat_col: str, target_col: str
) -> plt.Figure:
    """
    Grouped bar chart: for each category in cat_col, show how the
    target classes are distributed. Useful for spotting group-level bias.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Limit to top 10 categories to keep the chart readable
    top_cats = df[cat_col].value_counts().head(10).index
    subset = df[df[cat_col].isin(top_cats)]

    sns.countplot(
        data=subset,
        x=cat_col,
        hue=str(target_col),
        palette=PALETTE,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(f"{cat_col}  ×  {target_col}", fontsize=11, pad=8)
    ax.set_xlabel(cat_col, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    legend = ax.get_legend()
    if legend:
        legend.set_title(target_col)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        legend.get_title().set_color(TEXT_COLOR)
        legend.get_frame().set_facecolor("#1a1a2e")

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


def plot_numerical_vs_target(
    df: pd.DataFrame, num_col: str, target_col: str
) -> plt.Figure:
    """
    Side-by-side boxplots — one per class — so we can see whether a
    numerical feature has different distributions across target classes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    classes = sorted(df[target_col].unique())
    data_by_class = [
        df.loc[df[target_col] == cls, num_col].dropna().values
        for cls in classes
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    bp = ax.boxplot(
        data_by_class,
        labels=[str(c) for c in classes],
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=1.8),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"{num_col}  by  {target_col}", fontsize=11, pad=8)
    ax.set_xlabel("Class", fontsize=9)
    ax.set_ylabel(num_col, fontsize=9)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: list) -> plt.Figure:
    """
    Correlation matrix for all numerical columns.
    Annotated so you can read the values directly.
    """
    valid_cols = [c for c in numerical_cols if c in df.columns]
    if len(valid_cols) < 2:
        # Nothing useful to show with fewer than 2 columns
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Not enough numerical columns for a heatmap",
                ha="center", va="center", color=TEXT_COLOR)
        _apply_dark_style(fig, [ax])
        return fig

    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(max(6, len(valid_cols)), max(5, len(valid_cols) - 1)))

    mask = np.triu(np.ones_like(corr, dtype=bool))   # only lower triangle
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="#3a3a5c",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )

    ax.set_title("Correlation Heatmap — Numerical Features", fontsize=11, pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart showing missing-value percentage per column.
    Only includes columns that actually have missing data.
    """
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "✓  No missing values detected",
                ha="center", va="center", color="#55A868", fontsize=12)
        ax.axis("off")
        _apply_dark_style(fig, [ax])
        return fig

    fig, ax = plt.subplots(figsize=(7, max(3, len(missing) * 0.4)))

    bars = ax.barh(missing.index, missing.values, color="#DD8452", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, missing.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color=TEXT_COLOR)

    ax.set_xlabel("Missing (%)", fontsize=9)
    ax.set_title("Missing Values per Column", fontsize=11, pad=8)
    ax.set_xlim(0, max(missing.values) * 1.15)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


def perform_eda(
    df: pd.DataFrame,
    target_col: str,
    column_types: dict,
    max_cat_cols: int = 4,
    max_num_cols: int = 4,
) -> dict:
    """
    Master EDA function.  Generates all the figures and returns them in
    a dictionary so the Streamlit layer can place them wherever it wants.
    """
    cat_cols = [c for c in column_types.get("categorical", []) if c != target_col]
    num_cols = [c for c in column_types.get("numerical",   []) if c != target_col]

    figures = {}

    figures["target_dist"] = plot_target_distribution(df, target_col)
    figures["missing"]     = plot_missing_values(df)

    # Categorical plots (limit to avoid overwhelming the user)
    figures["cat_plots"] = []
    for col in cat_cols[:max_cat_cols]:
        fig = plot_categorical_vs_target(df, col, target_col)
        figures["cat_plots"].append((col, fig))

    # Numerical plots
    figures["num_plots"] = []
    for col in num_cols[:max_num_cols]:
        fig = plot_numerical_vs_target(df, col, target_col)
        figures["num_plots"].append((col, fig))

    # Correlation heatmap
    figures["heatmap"] = plot_correlation_heatmap(df, num_cols)

    return figures
