"""
preprocessing.py
----------------
Everything that touches the data before it reaches the model:
  - dropping identifier columns
  - imputing missing values
  - encoding categoricals
  - scaling numerics
  - train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_dataset(
    df: pd.DataFrame,
    identifier_cols: list,
    target_col: str,
) -> pd.DataFrame:
    """
    Drop identifier columns (they add no predictive value) and
    remove rows where the target is missing.
    """
    cols_to_drop = [c for c in identifier_cols if c in df.columns and c != target_col]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.dropna(subset=[target_col])
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame:
    """
    Simple but sensible imputation strategy:
      - Numerical  → fill with the median (robust to outliers)
      - Categorical → fill with the mode (most frequent category)
    """
    df = df.copy()

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    return df


def encode_and_scale(
    df: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
    target_col: str,
    sensitive_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode every categorical column with LabelEncoder and
    scale every numerical column with StandardScaler.

    We keep a dictionary of fitted encoders so the UI can
    show original class names where needed.

    Returns (transformed_df, artifacts_dict).
    """
    df = df.copy()
    artifacts = {"encoders": {}, "scaler": None, "scaled_cols": []}

    # --- encode categoricals ---
    for col in categorical_cols:
        if col not in df.columns:
            continue
        # strip whitespace first so ' Graduate' and 'Graduate' are the same value
        df[col] = df[col].astype(str).str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        artifacts["encoders"][col] = le

    # --- scale numericals (but NOT the target or sensitive attribute) ---
    cols_to_scale = [
        c for c in numerical_cols
        if c in df.columns and c != target_col and c != sensitive_col
    ]
    if cols_to_scale:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        artifacts["scaler"] = scaler
        artifacts["scaled_cols"] = cols_to_scale

    return df, artifacts


def split_data(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    drop_sensitive: bool = False,
) -> tuple:
    """
    Split into X/y train and test sets.

    If drop_sensitive is True we remove the sensitive column from X —
    this is used for the bias-mitigation experiment.

    Returns (X_train, X_test, y_train, y_test, sensitive_test).
    """
    feature_cols = [c for c in df.columns if c != target_col]

    # Hold on to the sensitive column values for fairness evaluation
    sensitive_test_values = None

    if drop_sensitive and sensitive_col in feature_cols:
        feature_cols = [c for c in feature_cols if c != sensitive_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # If we kept the sensitive column, pull it out of the test split
    if not drop_sensitive and sensitive_col in X_test.columns:
        sensitive_test_values = X_test[sensitive_col].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, sensitive_test_values


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    column_types: dict,
    drop_sensitive: bool = False,
) -> dict:
    """
    Convenience wrapper that runs the entire preprocessing sequence
    and returns a single results dictionary.
    """
    identifier_cols = column_types.get("identifier", [])
    numerical_cols  = [c for c in column_types.get("numerical", [])  if c != target_col]
    categorical_cols= [c for c in column_types.get("categorical", []) if c != target_col]

    # Step 1 – drop junk columns
    df = clean_dataset(df, identifier_cols, target_col)

    # Step 2 – fill gaps
    df = handle_missing_values(df, numerical_cols, categorical_cols)

    # Step 3 – encode + scale
    df, artifacts = encode_and_scale(df, numerical_cols, categorical_cols, target_col, sensitive_col)

    # Step 4 – split
    X_train, X_test, y_train, y_test, sensitive_test = split_data(
        df, target_col, sensitive_col, drop_sensitive=drop_sensitive
    )

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "sensitive_test": sensitive_test,
        "artifacts": artifacts,
        "processed_df": df,
    }
