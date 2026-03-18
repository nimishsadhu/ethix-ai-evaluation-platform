"""
predict.py
----------
Handles prediction on brand new unseen applicant data.

Takes raw input (single dict or small DataFrame), applies the same
preprocessing pipeline used during training, runs the model, and
returns a prediction with an ethical flag.

The ethical flag checks three things:
  1. Is the model's confidence low? (model is unsure)
  2. Does removing the sensitive attribute change the prediction? (proxy bias)
  3. Is the applicant from the historically disadvantaged group?
"""

import numpy as np
import pandas as pd


def preprocess_single(
    applicant: dict,
    feature_cols: list,
    encoders: dict,
    scaler,
    cols_to_scale: list,
    cat_cols: list,
) -> pd.DataFrame:
    """
    Apply the exact same transformations used during training
    to a single new applicant record.

    Parameters
    ----------
    applicant     : raw dict of feature values
    feature_cols  : ordered list of feature column names the model expects
    encoders      : dict of {col_name: fitted LabelEncoder}
    scaler        : fitted StandardScaler
    cols_to_scale : list of numerical column names that were scaled
    cat_cols      : list of categorical column names

    Returns
    -------
    Single-row DataFrame ready to pass into model.predict()
    """
    row = pd.DataFrame([applicant])

    # strip whitespace from string columns
    for col in row.select_dtypes(include="object").columns:
        row[col] = row[col].str.strip()

    # encode categoricals using the fitted encoders
    for col in cat_cols:
        if col not in row.columns:
            raise ValueError(f"Missing column in input: '{col}'")
        if col in encoders:
            try:
                # strip whitespace to match how training data was cleaned
                row[col] = row[col].astype(str).str.strip()
                row[col] = encoders[col].transform(row[col])
            except ValueError:
                known = encoders[col].classes_.tolist()
                raise ValueError(
                    f"Unknown value '{row[col].values[0]}' for column '{col}'. "
                    f"Expected one of: {known}"
                )

    # make sure all feature columns are present, fill missing with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    # keep only the columns the model was trained on, in the right order
    row = row[feature_cols].copy()

    # convert everything to numeric
    for col in row.columns:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    # scale numerical columns
    if scaler is not None and cols_to_scale:
        valid_scale = [c for c in cols_to_scale if c in row.columns]
        row[valid_scale] = scaler.transform(row[valid_scale])

    return row


def predict_applicant(
    applicant: dict,
    model,
    model_miti,
    feature_cols: list,
    feature_cols_miti: list,
    encoders: dict,
    scaler,
    cols_to_scale: list,
    cat_cols: list,
    sensitive_col: str,
    target_col: str,
    group_approval_rates: dict,
    confidence_threshold: float = 0.65,
) -> dict:
    """
    Predict loan approval for a single unseen applicant.
    Returns prediction + confidence + a detailed ethical flag.

    Parameters
    ----------
    applicant            : raw dict of applicant details
    model                : original trained model (with sensitive attribute)
    model_miti           : mitigated model (without sensitive attribute)
    feature_cols         : feature columns for original model
    feature_cols_miti    : feature columns for mitigated model
    encoders             : fitted LabelEncoder dict
    scaler               : fitted StandardScaler
    cols_to_scale        : numerical columns that were scaled
    cat_cols             : categorical column names
    sensitive_col        : name of sensitive attribute column
    target_col           : name of target column
    group_approval_rates : dict of {group_name: approval_rate} from training data
    confidence_threshold : below this probability → flag as low confidence

    Returns
    -------
    dict with keys: prediction, confidence, group, ethical_flags, recommendation
    """

    # ── preprocess for original model ────────────────────────────────────
    row_orig = preprocess_single(
        applicant, feature_cols, encoders, scaler, cols_to_scale, cat_cols
    )

    # ── preprocess for mitigated model (no sensitive col) ─────────────────
    row_miti = preprocess_single(
        applicant, feature_cols_miti, encoders, scaler, cols_to_scale, cat_cols
    )

    # ── original model prediction ─────────────────────────────────────────
    pred_enc  = int(model.predict(row_orig)[0])
    prob      = model.predict_proba(row_orig)[0]
    confidence= float(max(prob))

    # decode prediction back to original label
    # target_col is NOT in encoders (it was encoded by model._encode_labels, not preprocessing)
    # so we store the integer and let predict_batch decode using target_labels
    prediction      = pred_enc       # int 0 or 1
    pred_miti_enc   = int(model_miti.predict(row_miti)[0])
    prediction_miti = pred_miti_enc  # int 0 or 1

    # ── get applicant's group ─────────────────────────────────────────────
    raw_group = str(applicant.get(sensitive_col, "Unknown")).strip()
    group     = raw_group

    # ── ethical flag logic ────────────────────────────────────────────────
    ethical_flags  = []
    flag_level     = "GREEN"   # GREEN / YELLOW / RED

    # Flag 1 — low confidence
    if confidence < confidence_threshold:
        ethical_flags.append(
            f"⚠️  Low confidence ({confidence:.1%}) — model is unsure about this decision."
        )
        flag_level = "YELLOW"

    # Flag 2 — prediction changes when sensitive attribute is removed
    if prediction != prediction_miti:
        ethical_flags.append(
            f"🚨  Sensitive attribute '{sensitive_col}' is influencing this prediction. "
            f"Original model: {prediction} | Fair model (without {sensitive_col}): {prediction_miti}. "
            f"This applicant may be experiencing bias."
        )
        flag_level = "RED"

    # Flag 3 — disadvantaged group getting rejected
    # find the group with the lowest approval rate from training
    if group_approval_rates:
        min_rate_group = min(group_approval_rates, key=group_approval_rates.get)
        if group == min_rate_group and str(prediction).lower() in ["rejected", "0", "no", "1"]:
            ethical_flags.append(
                f"⚠️  Applicant belongs to '{group}' — the historically "
                f"lower-approval group (training approval rate: "
                f"{group_approval_rates.get(group, 'N/A'):.1%}). "
                f"Rejection flagged for human review."
            )
            if flag_level == "GREEN":
                flag_level = "YELLOW"

    # Final recommendation
    if flag_level == "RED":
        recommendation = "🔴  REFER TO HUMAN REVIEWER — potential bias detected."
    elif flag_level == "YELLOW":
        recommendation = "🟡  REVIEW RECOMMENDED — low confidence or group-level concern."
    else:
        recommendation = "🟢  PROCEED — no ethical concerns detected."

    return {
        "prediction":       prediction,
        "confidence":       round(confidence, 4),
        "confidence_pct":   f"{confidence:.1%}",
        "prediction_miti":  prediction_miti,
        "group":            group,
        "flag_level":       flag_level,
        "ethical_flags":    ethical_flags,
        "recommendation":   recommendation,
        "prob_class_0":     round(float(prob[0]), 4),
        "prob_class_1":     round(float(prob[1]), 4),
    }


def predict_batch(
    applicants_df: pd.DataFrame,
    model,
    model_miti,
    feature_cols: list,
    feature_cols_miti: list,
    encoders: dict,
    scaler,
    cols_to_scale: list,
    cat_cols: list,
    sensitive_col: str,
    target_col: str,
    group_approval_rates: dict,
    confidence_threshold: float = 0.65,
    label_encoder=None,
    target_labels: dict = None,
) -> pd.DataFrame:
    """
    Run predict_applicant() on every row of a DataFrame.
    Returns a summary DataFrame with one row per applicant.
    """
    def decode_label(val):
        # try label_encoder first (from model.py _encode_labels)
        if label_encoder is not None:
            try:
                return label_encoder.inverse_transform([int(float(str(val)))])[0]
            except Exception:
                pass
        # try target encoder in encoders dict
        if target_col in encoders:
            try:
                return encoders[target_col].inverse_transform([int(float(str(val)))])[0]
            except Exception:
                pass
        # fallback: use target_labels mapping if provided
        if target_labels:
            try:
                return target_labels.get(int(float(str(val))), str(val))
            except Exception:
                pass
        return str(val)

    results = []
    for i, row in applicants_df.iterrows():
        applicant = row.to_dict()
        result = predict_applicant(
            applicant, model, model_miti,
            feature_cols, feature_cols_miti,
            encoders, scaler, cols_to_scale, cat_cols,
            sensitive_col, target_col, group_approval_rates,
            confidence_threshold,
        )

        # decode 0/1 back to human-readable labels
        pred_label = decode_label(result["prediction"])
        miti_label = decode_label(result["prediction_miti"])

        # build row: applicant number + all original columns + result columns
        out = {"Applicant #": i + 1}
        for col in applicants_df.columns:
            out[col] = applicant.get(col, "")
        out["Prediction"]     = pred_label
        out["Fair Model"]     = miti_label
        out["Confidence"]     = result["confidence_pct"]
        out["Bias Flag"]      = result["flag_level"]
        out["Bias Reason"]    = "; ".join(result["ethical_flags"]) if result["ethical_flags"] else "None"
        out["Recommendation"] = (result["recommendation"]
                                 .replace("🔴  ", "").replace("🟡  ", "").replace("🟢  ", ""))
        results.append(out)

    return pd.DataFrame(results)
