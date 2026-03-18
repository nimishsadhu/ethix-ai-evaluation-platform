"""
model.py
--------
Model training and evaluation.
Handles string labels like 'Approved'/'Rejected' or '<=50K'/'>50K'
by encoding them to 0/1 before computing any metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

TEXT_COLOR = "#FAFAFA"


def _encode_labels(y):
    """
    Convert any label array to integer 0/1.
    - If already 0/1 integers  -> return as-is, encoder=None
    - If strings or other types -> use LabelEncoder
    Returns (y_encoded_as_int, label_encoder_or_None)
    """
    y = np.array(y)

    # Already numeric with only 0 and 1 values
    if y.dtype.kind in ("i", "u", "f"):
        unique = set(y.ravel().tolist())
        if unique.issubset({0, 1, 0.0, 1.0}):
            return y.astype(int), None

    # String labels or numeric with values other than 0/1
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


def train_model(X_train, y_train, max_iter: int = 1000) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    Encodes string labels to 0/1 so sklearn never sees raw strings.
    """
    y_enc, _ = _encode_labels(y_train)
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_enc)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute a full set of classification metrics.
    Works with string labels (Approved/Rejected etc) and integer labels (0/1).
    Always stores y_test_enc (0/1 integers) for downstream use.
    """
    # Encode test labels to 0/1
    y_enc, le = _encode_labels(y_test)

    y_pred      = model.predict(X_test)
    y_pred_prob = None

    # ROC-AUC needs probability scores
    try:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        roc_auc     = roc_auc_score(y_enc, y_pred_prob)
    except Exception:
        roc_auc = None

    results = {
        "accuracy":  round(accuracy_score(y_enc, y_pred), 4),
        "precision": round(precision_score(y_enc, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_enc, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_enc, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc":   round(roc_auc, 4) if roc_auc is not None else "N/A",

        # confusion matrix and report use the encoded (0/1) labels
        "confusion_matrix":      confusion_matrix(y_enc, y_pred),
        "classification_report": classification_report(y_enc, y_pred, zero_division=0),

        "y_pred":        y_pred,    # 0/1 integer predictions
        "y_pred_prob":   y_pred_prob,
        "y_test_enc":    y_enc,     # 0/1 encoded ground truth — use this for ROC etc.
        "label_encoder": le,
    }

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None) -> plt.Figure:
    """Heatmap of the confusion matrix."""
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#3a3a5c",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_xlabel("Predicted", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("Actual",    fontsize=10, color=TEXT_COLOR)
    ax.set_title("Confusion Matrix", fontsize=12, pad=10, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")

    fig.tight_layout()
    return fig


def plot_roc_curve(y_test_enc, y_pred_prob) -> plt.Figure:
    """
    ROC curve.
    y_test_enc must already be 0/1 integers — pass result['y_test_enc'] from evaluate_model.
    """
    fpr, tpr, _ = roc_curve(y_test_enc, y_pred_prob)
    auc = roc_auc_score(y_test_enc, y_pred_prob)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#555555", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#4C72B0")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("True Positive Rate",  fontsize=10, color=TEXT_COLOR)
    ax.set_title("ROC Curve", fontsize=12, pad=10, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    legend = ax.legend(loc="lower right", fontsize=9)
    legend.get_frame().set_facecolor("#1a1a2e")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")

    fig.tight_layout()
    return fig


def plot_feature_importance(model: LogisticRegression, feature_names: list) -> plt.Figure:
    """Absolute coefficient magnitudes as a proxy for feature importance."""
    if not hasattr(model, "coef_"):
        return None

    coefs      = np.abs(model.coef_[0])
    sorted_idx = np.argsort(coefs)[-15:]

    fig, ax = plt.subplots(figsize=(7, max(4, len(sorted_idx) * 0.35)))

    ax.barh(
        [feature_names[i] for i in sorted_idx],
        coefs[sorted_idx],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("|Coefficient|", fontsize=9, color=TEXT_COLOR)
    ax.set_title("Feature Importance (Logistic Regression)", fontsize=11, pad=8, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5c")

    fig.tight_layout()
    return fig
