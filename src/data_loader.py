"""
data_loader.py
--------------
Handles everything related to reading a CSV file and doing the first
round of inspection — column types, potential ID columns, and basic
shape information.
"""

import pandas as pd
import numpy as np


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV from a Streamlit UploadedFile object.
    Also cleans column names right away so the rest of the code
    never has to worry about spaces or mixed capitalisation.
    """
    df = pd.read_csv(uploaded_file)

    # Clean column names: lowercase + replace spaces with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Return a small summary dictionary that the UI can display
    without having to re-compute things multiple times.
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Split columns into three categories:
      - numerical   : int or float dtype
      - categorical : object or low-cardinality int
      - identifier  : unique value count == number of rows  (likely an ID column)

    Returns a dict with keys 'numerical', 'categorical', 'identifier'.
    """
    numerical = []
    categorical = []
    identifier = []

    for col in df.columns:
        n_unique = df[col].nunique()
        n_rows   = len(df)

        # A column whose every value is unique is almost certainly an ID
        if n_unique == n_rows:
            identifier.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Only treat as categorical if truly binary (0/1 flags)
            # Count/ordinal columns like no_of_dependents (0-5) or
            # loan_term (2-20) should be numerical, not categorical
            if n_unique <= 2 and df[col].dtype in [np.int64, np.int32, np.int16]:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "identifier": identifier,
    }


def validate_target(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    """
    Make sure the chosen target column is suitable for binary classification.
    Returns (is_valid: bool, message: str).
    """
    if target_col not in df.columns:
        return False, f"Column '{target_col}' not found in the dataset."

    n_classes = df[target_col].nunique()

    if n_classes < 2:
        return False, "Target column has only one unique value — nothing to classify."

    if n_classes > 2:
        return (
            False,
            f"Target column has {n_classes} unique values. "
            "ETHIX currently supports binary classification only (exactly 2 classes).",
        )

    return True, "Target column looks good — binary classification detected."
