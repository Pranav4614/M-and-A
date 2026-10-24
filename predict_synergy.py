#!/usr/bin/env python3
"""Guarded inference for the M&A synergy model.

Predictions are unavailable until the strict real-data training gate passes and
model artifacts exist. Unknown payment-method categories are rejected; low-support
categories and numeric extrapolation are explicitly flagged in the JSON result.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_model_dataset import ROOT, standardize_payment_method


def make_input_frame(acquirer_revenue_usd_m: float, target_revenue_usd_m: float,
                     payment_method: str) -> pd.DataFrame:
    """Validate raw user input and create the same feature columns used in training."""
    for name, value in [("acquirer_revenue_usd_m", acquirer_revenue_usd_m),
                        ("target_revenue_usd_m", target_revenue_usd_m)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number in USD millions")
    method = standardize_payment_method(payment_method)
    if method not in {"cash", "stock", "mixed"}:
        raise ValueError("payment_method must be cash, stock, or mixed")
    return pd.DataFrame([{
        "feature_log_acquirer_revenue": math.log(float(acquirer_revenue_usd_m)),
        "feature_log_target_revenue": math.log(float(target_revenue_usd_m)),
        "payment_method_model": method,
        "acquirer_revenue_usd_m": float(acquirer_revenue_usd_m),
        "target_revenue_usd_m": float(target_revenue_usd_m),
    }])


def _metadata(model_dir: Path, key: str) -> dict[str, Any] | None:
    p = model_dir / f"{key}_model_metadata.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _predict_target(row: pd.DataFrame, model_dir: Path, key: str,
                    metadata: dict[str, Any]) -> float:
    from train_synergy_model import FEATURES
    spec = metadata.get("selected_spec", {})
    if spec.get("family") == "baseline":
        value = float(metadata["baseline_median_value"])
    else:
        artifact = model_dir / f"selected_synergy_{key}_model.joblib"
        if not artifact.exists():
            raise FileNotFoundError(f"Missing fitted model artifact: {artifact}")
        import joblib
        model = joblib.load(artifact)
        feature_cols = FEATURES[spec["feature_set"]]
        value = float(np.asarray(model.predict(row[feature_cols]), dtype=float)[0])
        if spec.get("target_transform") == "log1p":
            value = float(np.expm1(value))
    return max(0.0, value)


def predict_one(acquirer_revenue_usd_m: float, target_revenue_usd_m: float,
                payment_method: str, model_dir: Path | None = None,
                evaluation_path: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir or ROOT / "models"
    evaluation_path = evaluation_path or ROOT / "reports" / "model_evaluation_result.json"
    try:
        row = make_input_frame(acquirer_revenue_usd_m, target_revenue_usd_m, payment_method)
    except ValueError as exc:
        return {"status": "INVALID_INPUT", "error": str(exc)}

    ratio_meta = _metadata(model_dir, "ratio")
    dollar_meta = _metadata(model_dir, "dollar")
    if ratio_meta is None or dollar_meta is None:
        return {
            "status": "MODEL_NOT_AVAILABLE",
            "reason": "No fitted model metadata is present. The project training gate currently blocks fitting until the verified real-data sample and temporal validation design are sufficient.",
            "predictions": None,
        }
    if not evaluation_path.exists():
        return {"status": "MODEL_NOT_AVAILABLE", "reason": "The latest model evaluation record is missing; refusing to use an unverified or stale artifact.", "predictions": None}
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    eval_manifest = evaluation.get("run_manifest", {})
    ratio_manifest = ratio_meta.get("run_manifest", {})
    dollar_manifest = dollar_meta.get("run_manifest", {})
    if evaluation.get("status") != "EVALUATED_EXPLORATORY" or not evaluation.get("model_fit_attempted"):
        return {"status": "MODEL_NOT_AVAILABLE", "reason": "The latest training evaluation is blocked or not complete; refusing to issue predictions from a stale artifact.", "predictions": None}
    if eval_manifest.get("source_data_attested_real") is not True or eval_manifest.get("data_is_synthetic") is not False:
        return {"status": "MODEL_NOT_AVAILABLE", "reason": "The latest evaluation does not positively attest source-reviewed real research data; synthetic or unattested model artifacts cannot be used for project predictions.", "predictions": None}
    expected_hash = eval_manifest.get("input_sha256")
    expected_script_hash = eval_manifest.get("training_script_sha256")
    if (not expected_hash or ratio_manifest.get("input_sha256") != expected_hash or dollar_manifest.get("input_sha256") != expected_hash
            or not expected_script_hash or ratio_manifest.get("training_script_sha256") != expected_script_hash
            or dollar_manifest.get("training_script_sha256") != expected_script_hash):
        return {"status": "MODEL_NOT_AVAILABLE", "reason": "Model artifact provenance does not match the latest evaluation record; retrain before prediction.", "predictions": None}

    ratio_counts = ratio_meta.get("development_payment_method_counts", {})
    method = str(row.loc[0, "payment_method_model"])
    method_n = int(ratio_counts.get(method, 0))
    if method_n == 0:
        return {
            "status": "UNSUPPORTED_PAYMENT_METHOD_CATEGORY",
            "reason": f"The payment category '{method}' has no observations in model development data; no numeric estimate will be issued for this category.",
            "supported_payment_method_counts": ratio_counts,
            "predictions": None,
        }

    ratio_pct = _predict_target(row, model_dir, "ratio", ratio_meta)
    direct_dollars = _predict_target(row, model_dir, "dollar", dollar_meta)
    ratio_dollars = ratio_pct / 100.0 * float(target_revenue_usd_m)

    ranges = ratio_meta.get("development_feature_ranges", {})
    extrapolations = []
    for col in ["feature_log_acquirer_revenue", "feature_log_target_revenue"]:
        bounds = ranges.get(col)
        if bounds and (float(row.loc[0, col]) < bounds["min"] or float(row.loc[0, col]) > bounds["max"]):
            extrapolations.append(col)

    min_cat_n = int(ratio_meta.get("minimum_rows_per_payment_method_for_interpretation", 10))
    low_support = method_n < min_cat_n
    if extrapolations:
        status = "EXTRAPOLATION_WARNING"
    elif low_support:
        status = "LOW_SUPPORT_ESTIMATE"
    else:
        status = "POINT_ESTIMATE_WITHIN_OBSERVED_SUPPORT"

    return {
        "status": status,
        "model_version": "credibility_v3",
        "prediction_scope": "Management-announced annual pre-tax cost-synergy claim, conditional on the selected deal cohort disclosing a quantified point estimate; not realized synergy.",
        "inputs": {"acquirer_revenue_usd_m": float(acquirer_revenue_usd_m),
                   "target_revenue_usd_m": float(target_revenue_usd_m), "payment_method": method},
        "predictions": {
            "normalized_annual_cost_synergy_pct_target_revenue": ratio_pct,
            "ratio_prediction_converted_to_usd_m": ratio_dollars,
            "direct_dollar_model_usd_m": direct_dollars,
            "difference_between_dollar_estimates_usd_m": abs(ratio_dollars - direct_dollars),
        },
        "support": {
            "payment_method_observations_in_development": method_n,
            "minimum_payment_category_n_for_stronger_interpretation": min_cat_n,
            "payment_method_low_support": low_support,
            "numeric_features_outside_development_range": extrapolations,
        },
        "caveats": [
            "Prediction accuracy must be judged from the untouched future-period test result, not fit or training scores.",
            "A model estimate does not validate whether management will realize the synergy claim.",
            "Difference between ratio-converted dollars and direct-dollar prediction is a model-consistency diagnostic, not a confidence interval.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acquirer-revenue-usd-m", type=float, required=True)
    ap.add_argument("--target-revenue-usd-m", type=float, required=True)
    ap.add_argument("--payment-method", choices=["cash", "stock", "mixed"], required=True)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--evaluation-path", type=Path, default=ROOT / "reports" / "model_evaluation_result.json")
    args = ap.parse_args()
    result = predict_one(args.acquirer_revenue_usd_m, args.target_revenue_usd_m, args.payment_method, args.model_dir, args.evaluation_path)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] not in {"INVALID_INPUT", "MODEL_NOT_AVAILABLE", "UNSUPPORTED_PAYMENT_METHOD_CATEGORY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
