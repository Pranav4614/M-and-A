#!/usr/bin/env python3
"""Leakage-resistant, year-grouped model selection for announced M&A cost synergy claims.

No fitting occurs until both the minimum sample gate and temporal-design checks pass.
The final future-period test set is never used for hyperparameter selection.
"""
from __future__ import annotations
import argparse, json, math, hashlib, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from build_model_dataset import DEFAULT_INPUT, ROOT, prepare_dataset

MIN_N = 100                       # governance floor, not a guarantee of adequate power
MIN_DEVELOPMENT_N = 60
MIN_CALIBRATION_N = 10
MIN_TEST_N = 20
MIN_CV_TRAIN_N = 30
MIN_CV_VALIDATION_N = 5
MIN_CV_FOLDS = 3
MIN_CAL_N_FOR_INTERVAL = 20       # intervals are suppressed below this point
TARGET = "target_recomputed_pct"
DOLLAR_TARGET = "announced_annual_cost_synergy_usd_m"
FEATURES = {
    "full": ["feature_log_acquirer_revenue", "feature_log_target_revenue", "payment_method_model"],
    "target_only": ["feature_log_target_revenue", "payment_method_model"],
}
NUMERICS = {
    "full": ["feature_log_acquirer_revenue", "feature_log_target_revenue"],
    "target_only": ["feature_log_target_revenue"],
}
CATEGORICALS = ["payment_method_model"]


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["announcement_date"] = pd.to_datetime(x["announcement_date"], errors="coerce")
    x["announcement_year"] = x["announcement_date"].dt.year
    return x.sort_values(["announcement_date", "deal_id"], kind="stable").reset_index(drop=True)


def grouped_future_holdouts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reserve whole announcement years for calibration and final test; never split a year."""
    x = _sort_df(df)
    if x.empty or x["announcement_year"].isna().any():
        return {"development": x.iloc[0:0].copy(), "calibration": x.iloc[0:0].copy(), "test": x.iloc[0:0].copy()}
    years = sorted(x["announcement_year"].astype(int).unique())
    # Do not invent train/test roles for a pilot sample. No holdout assignment is meaningful below the sample/year gate.
    if len(x) < MIN_N or len(years) < 8:
        empty = x.iloc[0:0].copy()
        return {"development": empty.copy(), "calibration": empty.copy(), "test": empty.copy()}
    test_years: list[int] = []
    test_target = max(MIN_TEST_N, math.ceil(0.20 * len(x)))
    test_rows = 0
    for year in reversed(years):
        test_years.insert(0, year)
        test_rows += int((x["announcement_year"] == year).sum())
        if test_rows >= test_target:
            break
    remaining_years = [y for y in years if y not in test_years]
    cal_years: list[int] = []
    cal_target = max(MIN_CALIBRATION_N, math.ceil(0.10 * len(x)))
    cal_rows = 0
    # Leave at least two years for development if possible.
    for year in reversed(remaining_years):
        if len([y for y in remaining_years if y not in cal_years]) <= 2:
            break
        cal_years.insert(0, year)
        cal_rows += int((x["announcement_year"] == year).sum())
        if cal_rows >= cal_target:
            break
    dev_years = [y for y in years if y not in set(test_years + cal_years)]
    dev = x.loc[x["announcement_year"].isin(dev_years)].copy()
    cal = x.loc[x["announcement_year"].isin(cal_years)].copy()
    test = x.loc[x["announcement_year"].isin(test_years)].copy()
    return {"development": dev, "calibration": cal, "test": test}


def grouped_rolling_folds(development: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Create expanding-window folds on whole years with multiple historical windows.

    The development years are partitioned into an initial training block followed by
    three non-overlapping validation blocks. Validation blocks are kept to whole years.
    """
    x = _sort_df(development)
    years = np.array(sorted(x["announcement_year"].dropna().astype(int).unique()))
    if len(years) < MIN_CV_FOLDS + 3:
        return []
    val_years_per_fold = max(1, len(years) // (MIN_CV_FOLDS + 3))
    initial_train_year_count = len(years) - MIN_CV_FOLDS * val_years_per_fold
    if initial_train_year_count < 3:
        return []
    folds = []
    for fold_no in range(1, MIN_CV_FOLDS + 1):
        train_end = initial_train_year_count + (fold_no - 1) * val_years_per_fold
        val_end = train_end + val_years_per_fold
        train_years = set(years[:train_end])
        val_year_set = set(years[train_end:val_end])
        train_idx = np.flatnonzero(x["announcement_year"].isin(train_years).to_numpy())
        val_idx = np.flatnonzero(x["announcement_year"].isin(val_year_set).to_numpy())
        meta = {
            "fold": fold_no,
            "train_n": int(len(train_idx)),
            "validation_n": int(len(val_idx)),
            "train_year_min": int(min(train_years)), "train_year_max": int(max(train_years)),
            "validation_year_min": int(min(val_year_set)), "validation_year_max": int(max(val_year_set)),
        }
        folds.append((train_idx, val_idx, meta))
    return folds


def gate_status(ready: pd.DataFrame, input_n: int) -> dict[str, Any]:
    parts = grouped_future_holdouts(ready)
    sizes = {k: int(len(v)) for k, v in parts.items()}
    years = {k: sorted(parts[k]["announcement_year"].dropna().astype(int).unique().tolist()) for k in parts}
    folds = grouped_rolling_folds(parts["development"])
    fold_metadata = [m for _, _, m in folds]
    probs: list[str] = []
    n = int(len(ready))
    if n < MIN_N: probs.append(f"Only {n} US-listed, strict-label, full-feature-ready rows; governance floor is {MIN_N}.")
    if sizes["development"] < MIN_DEVELOPMENT_N: probs.append(f"Year-grouped development set has {sizes['development']} rows; minimum is {MIN_DEVELOPMENT_N}.")
    if sizes["calibration"] < MIN_CALIBRATION_N: probs.append(f"Year-grouped calibration set has {sizes['calibration']} rows; minimum is {MIN_CALIBRATION_N}.")
    if sizes["test"] < MIN_TEST_N: probs.append(f"Year-grouped final future test has {sizes['test']} rows; minimum is {MIN_TEST_N}.")
    if len(folds) < MIN_CV_FOLDS: probs.append(f"Only {len(folds)} valid whole-year expanding-window folds; minimum is {MIN_CV_FOLDS}.")
    for m in fold_metadata:
        if m["train_n"] < MIN_CV_TRAIN_N:
            probs.append(f"CV fold {m['fold']} has {m['train_n']} training rows; minimum is {MIN_CV_TRAIN_N}.")
        if m["validation_n"] < MIN_CV_VALIDATION_N:
            probs.append(f"CV fold {m['fold']} has {m['validation_n']} validation rows; minimum is {MIN_CV_VALIDATION_N}.")
    if len(ready) and ready["announcement_year"].nunique() < 8:
        probs.append("Fewer than 8 distinct announcement years in model-ready rows; future-period validation is too fragile.")
    payment_counts = ready["payment_method_model"].value_counts().to_dict() if len(ready) and "payment_method_model" in ready else {}
    payment_support_warnings = []
    for category in ["cash", "stock", "mixed"]:
        count = int(payment_counts.get(category, 0))
        if count == 0:
            payment_support_warnings.append(f"No approved rows in the current model-ready cohort for payment category '{category}'; do not issue empirically supported predictions for it.")
        elif count < 10:
            payment_support_warnings.append(f"Only {count} approved rows in the current model-ready cohort for payment category '{category}'; category effects are not stable enough for strong interpretation.")
    return {
        "status": "READY_TO_FIT_EXPLORATORY" if not probs else "BLOCKED_BEFORE_FIT",
        "gate_pass": not probs,
        "eligible_n": n, "input_n": int(input_n), "partition_sizes": sizes,
        "partition_years": years, "rolling_cv_folds": fold_metadata, "problems": probs,
        "payment_method_category_counts": {str(k): int(v) for k, v in payment_counts.items()},
        "payment_method_support_warnings": payment_support_warnings,
        "minimum_rows_per_payment_method_for_interpretation": 10,
        "minimum_n_is_governance_floor_not_statistical_guarantee": True,
        "prediction_task": "Estimate the announced annual pre-tax cost-synergy point claim normalized by target revenue, conditional on qualifying deals with quantified claims.",
        "not_the_task": "Does not estimate realized synergy savings, causal merger effects, or whether the management claim will come true.",
    }


def _make_pipeline(family: str, feature_set: str, params: dict[str, Any]):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
    num = NUMERICS[feature_set]
    prep = ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICALS),
    ])
    if family == "ridge":
        estimator = Ridge(alpha=float(params["alpha"]))
    elif family == "huber":
        estimator = HuberRegressor(alpha=float(params["alpha"]), epsilon=float(params["epsilon"]), max_iter=2000)
    elif family == "quantile":
        # Median quantile regression directly targets the conditional median, aligning with MAE selection.
        estimator = QuantileRegressor(quantile=0.5, alpha=float(params["alpha"]), solver="highs", fit_intercept=True)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return Pipeline([("prep", prep), ("model", estimator)])


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "median_baseline", "family": "baseline", "feature_set": "none", "target_transform": "none", "params": {}, "complexity": 0}]
    for feat in ["target_only", "full"]:
        for trans in ["raw", "log1p"]:
            for alpha in [1.0, 10.0, 100.0]:
                specs.append({"name": f"ridge_{feat}_{trans}_a{alpha:g}", "family": "ridge", "feature_set": feat, "target_transform": trans, "params": {"alpha": alpha}, "complexity": 1 if feat == "target_only" else 2})
    # Median quantile regression aligns its loss with the primary selection metric (MAE).
    # A short regularization grid limits selection overfit on modest datasets.
    for feat in ["target_only", "full"]:
        for trans in ["raw", "log1p"]:
            for alpha in [0.1, 1.0]:
                specs.append({"name": f"quantile_median_{feat}_{trans}_a{alpha:g}", "family": "quantile", "feature_set": feat, "target_transform": trans, "params": {"alpha": alpha}, "complexity": 1 if feat == "target_only" else 2})
    # One robust-loss setting pair; do not explode the hyperparameter search on modest datasets.
    for feat in ["target_only", "full"]:
        for trans in ["raw", "log1p"]:
            for eps in [1.35, 1.75]:
                specs.append({"name": f"huber_{feat}_{trans}_e{eps:g}", "family": "huber", "feature_set": feat, "target_transform": trans, "params": {"alpha": 0.01, "epsilon": eps}, "complexity": 2 if feat == "target_only" else 3})
    return specs


def _fit_and_predict(spec: dict[str, Any], train: pd.DataFrame, y_train: np.ndarray, pred: pd.DataFrame) -> np.ndarray:
    if spec["family"] == "baseline":
        return np.repeat(float(np.median(y_train)), len(pred))
    feats = FEATURES[spec["feature_set"]]
    y_fit = np.log1p(y_train) if spec["target_transform"] == "log1p" else y_train
    estimator = _make_pipeline(spec["family"], spec["feature_set"], spec["params"])
    estimator.fit(train[feats], y_fit)
    pred_fit = estimator.predict(pred[feats])
    if spec["target_transform"] == "log1p":
        pred_fit = np.expm1(pred_fit)
    return np.maximum(0.0, np.asarray(pred_fit, dtype=float))


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def evaluate_candidate_cv(spec: dict[str, Any], df: pd.DataFrame, folds, target_col: str) -> list[dict[str, Any]]:
    x = _sort_df(df)
    y = x[target_col].to_numpy(dtype=float)
    scores = []
    for tr_idx, va_idx, meta in folds:
        pred = _fit_and_predict(spec, x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx])
        row = {**meta, "candidate": spec["name"], "mae": _mae(y[va_idx], pred)}
        baseline = float(np.median(y[tr_idx]))
        row["median_baseline_mae"] = _mae(y[va_idx], np.repeat(baseline, len(va_idx)))
        scores.append(row)
    return scores


def select_one_standard_error(cv_scores: pd.DataFrame, specs: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, float]:
    summary = cv_scores.groupby("candidate", as_index=False).agg(
        cv_mean_mae=("mae", "mean"), cv_sd_mae=("mae", "std"), cv_folds=("mae", "count")
    )
    summary["cv_se_mae"] = summary["cv_sd_mae"].fillna(0) / np.sqrt(summary["cv_folds"].clip(lower=1))
    best_row = summary.loc[summary["cv_mean_mae"].idxmin()]
    threshold = float(best_row["cv_mean_mae"] + best_row["cv_se_mae"])
    by_name = {s["name"]: s for s in specs}
    summary["complexity"] = summary["candidate"].map(lambda n: by_name[n]["complexity"])
    summary["within_one_se_of_best"] = summary["cv_mean_mae"] <= threshold + 1e-12
    admissible = summary.loc[summary["within_one_se_of_best"]].sort_values(["complexity", "cv_mean_mae", "candidate"])
    chosen_name = str(admissible.iloc[0]["candidate"])
    selected = dict(by_name[chosen_name])
    summary["selected_by_one_se_rule"] = summary["candidate"].eq(chosen_name)
    baseline_scores = cv_scores.loc[cv_scores["candidate"] == "median_baseline"].set_index("fold")["mae"]
    baseline_mean = float(baseline_scores.mean())
    summary["median_baseline_cv_mean_mae"] = baseline_mean
    summary["cv_skill_vs_median_baseline"] = 1.0 - summary["cv_mean_mae"] / baseline_mean if baseline_mean > 0 else np.nan
    folds_beaten = {}
    for name, sub in cv_scores.groupby("candidate"):
        aligned = sub.set_index("fold")["mae"]
        common = aligned.index.intersection(baseline_scores.index)
        folds_beaten[name] = int((aligned.loc[common] < baseline_scores.loc[common]).sum())
    summary["folds_beating_median_baseline"] = summary["candidate"].map(folds_beaten)
    return selected, summary.sort_values(["cv_mean_mae", "complexity"]).reset_index(drop=True), threshold


def metrics_for_test(y: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray, unit_suffix: str) -> dict[str, Any]:
    err = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    abs_err = np.abs(err)
    baseline_mae = _mae(y, baseline_pred)
    mae = _mae(y, pred)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    from sklearn.metrics import r2_score
    return {
        f"n_test": int(len(y)),
        f"mae_{unit_suffix}": mae,
        f"median_absolute_error_{unit_suffix}": float(np.median(abs_err)),
        f"rmse_{unit_suffix}": rmse,
        f"mean_error_actual_minus_prediction_{unit_suffix}": float(np.mean(err)),
        f"p90_absolute_error_{unit_suffix}": float(np.quantile(abs_err, 0.90)),
        "r2_descriptive_only": float(r2_score(y, pred)) if len(y) >= 2 else None,
        f"median_baseline_mae_{unit_suffix}": baseline_mae,
        "mae_skill_vs_median_baseline": (1.0 - mae / baseline_mae) if baseline_mae > 0 else None,
        "improves_on_median_baseline": bool(mae < baseline_mae),
        "note": f"Errors are {unit_suffix.replace('_', ' ')}; R² is secondary and unstable on small test samples."
    }


def _fit_candidate(spec: dict[str, Any], development: pd.DataFrame, target_col: str):
    y_dev = development[target_col].to_numpy(dtype=float)
    if spec["family"] == "baseline":
        return None, float(np.median(y_dev))
    feats = FEATURES[spec["feature_set"]]
    y_fit = np.log1p(y_dev) if spec["target_transform"] == "log1p" else y_dev
    estimator = _make_pipeline(spec["family"], spec["feature_set"], spec["params"])
    estimator.fit(development[feats], y_fit)
    return estimator, float(np.median(y_dev))


def _predict_candidate(spec: dict[str, Any], model, baseline_value: float, frame: pd.DataFrame) -> np.ndarray:
    if spec["family"] == "baseline":
        return np.repeat(baseline_value, len(frame))
    pred = model.predict(frame[FEATURES[spec["feature_set"]]])
    if spec["target_transform"] == "log1p":
        pred = np.expm1(pred)
    return np.maximum(0.0, np.asarray(pred, dtype=float))


def _interval_from_calibration(y_cal: np.ndarray, pred_cal: np.ndarray, n_min: int = MIN_CAL_N_FOR_INTERVAL) -> dict[str, Any]:
    if len(y_cal) < n_min:
        return {"status": "SUPPRESSED_INSUFFICIENT_CALIBRATION_N", "calibration_n": int(len(y_cal)), "minimum_calibration_n": n_min}
    resid = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float))
    rank_1based = int(math.ceil((len(resid) + 1) * 0.90))
    if rank_1based > len(resid):
        return {"status": "SUPPRESSED_FINITE_SAMPLE_QUANTILE_UNAVAILABLE", "calibration_n": int(len(y_cal))}
    return {"status": "EMPIRICAL_SPLIT_CONFORMAL_90_PERCENT_NOMINAL", "calibration_n": int(len(y_cal)),
            "quantile": float(np.sort(resid)[rank_1based - 1]),
            "caveat": "Nominal split-conformal coverage relies on exchangeability; temporal distribution shift may invalidate it."}


def _out_of_range_support(development: pd.DataFrame, evaluation: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    numeric_used = [f for f in selected_features if f in ["feature_log_acquirer_revenue", "feature_log_target_revenue"]]
    out = {"numeric_feature_ranges": {}, "test_rows_with_any_numeric_extrapolation": 0, "categorical_unseen_values": {}}
    outside_any = np.zeros(len(evaluation), dtype=bool)
    for feat in numeric_used:
        lo, hi = float(development[feat].min()), float(development[feat].max())
        mask = (evaluation[feat].to_numpy(dtype=float) < lo) | (evaluation[feat].to_numpy(dtype=float) > hi)
        out["numeric_feature_ranges"][feat] = {
            "development_min": lo, "development_max": hi,
            "evaluation_n_below_or_above_range": int(mask.sum()), "evaluation_n": int(len(evaluation)),
        }
        outside_any |= mask
    out["test_rows_with_any_numeric_extrapolation"] = int(outside_any.sum())
    if "payment_method_model" in selected_features:
        known = set(development["payment_method_model"].dropna().astype(str))
        unseen = ~evaluation["payment_method_model"].astype(str).isin(known).to_numpy()
        out["categorical_unseen_values"] = {
            "development_categories": sorted(known),
            "evaluation_rows_with_unseen_payment_method": int(unseen.sum()),
            "evaluation_n": int(len(evaluation)),
        }
    return out


def _run_target_experiment(target_col: str, key: str, metric_suffix: str, parts: dict[str, pd.DataFrame],
                           folds, specs, output_dir: Path, model_dir: Path,
                           run_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    development, calibration, test = parts["development"], parts["calibration"], parts["test"]
    cv_rows = []
    for spec in specs:
        for row in evaluate_candidate_cv(spec, development, folds, target_col):
            row["target_key"] = key
            cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    selected, comparison, one_se_threshold = select_one_standard_error(cv, specs)
    cv.to_csv(output_dir / f"{key}_rolling_cv_fold_scores.csv", index=False)
    comparison.to_csv(output_dir / f"{key}_candidate_comparison.csv", index=False)

    model, baseline_value = _fit_candidate(selected, development, target_col)
    y_cal = calibration[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)
    cal_pred = _predict_candidate(selected, model, baseline_value, calibration)
    test_pred = _predict_candidate(selected, model, baseline_value, test)
    interval = _interval_from_calibration(y_cal, cal_pred)
    baseline_pred = np.repeat(baseline_value, len(test))
    metrics = metrics_for_test(y_test, test_pred, baseline_pred, metric_suffix)
    selected_cv_mae = float(cv.loc[cv["candidate"] == selected["name"], "mae"].mean())
    baseline_cv_mae = float(cv.loc[cv["candidate"] == "median_baseline", "mae"].mean())
    selected_features = FEATURES.get(selected["feature_set"], [])
    extrapolation = _out_of_range_support(development, test, selected_features) if selected_features else {
        "numeric_feature_ranges": {}, "test_rows_with_any_numeric_extrapolation": 0,
        "categorical_unseen_values": {}, "note": "Selected estimator is the median baseline and uses no predictors."}
    (output_dir / f"{key}_test_extrapolation_diagnostics.json").write_text(json.dumps(extrapolation, indent=2), encoding="utf-8")

    metadata = {
        "model_version": "model_credibility_v3",
        "selected_candidate": selected["name"], "selected_spec": selected,
        "baseline_median_value": baseline_value,
        "run_manifest": run_manifest,
        "target_column": target_col,
        "target": ("annual announced pre-tax cost synergy point claim / target revenue * 100" if key == "ratio" else "annual announced pre-tax cost synergy point claim in USD millions"),
        "target_unit": metric_suffix, "features": selected_features,
        "development_year_range": [int(development["announcement_year"].min()), int(development["announcement_year"].max())],
        "development_n": int(len(development)), "calibration_n": int(len(calibration)), "final_test_n": int(len(test)),
        "development_payment_method_counts": {str(k): int(v) for k, v in development["payment_method_model"].value_counts().to_dict().items()},
        "minimum_rows_per_payment_method_for_interpretation": 10,
        "payment_method_support_note": "Predictions for unobserved categories are unsupported; categories with fewer than 10 development rows should be marked low-support.",
        "development_feature_ranges": {c: {"min": float(development[c].min()), "max": float(development[c].max())} for c in ["feature_log_acquirer_revenue", "feature_log_target_revenue"] if c in development},
        "selected_cv_mean_mae": selected_cv_mae, "median_baseline_cv_mean_mae": baseline_cv_mae,
        "cv_skill_vs_median_baseline": (1.0 - selected_cv_mae / baseline_cv_mae) if baseline_cv_mae > 0 else None,
        "one_standard_error_threshold": one_se_threshold, "calibration_interval": interval,
        "feature_support_diagnostics": extrapolation,
        "not_a_model_for": ["realized synergies", "causal synergy estimation", "acquisition-premium prediction"],
    }
    (model_dir / f"{key}_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if model is not None:
        import joblib
        joblib.dump(model, model_dir / f"selected_synergy_{key}_model.joblib")
    return {"key": key, "selected": selected, "model": model, "baseline_value": baseline_value,
            "calibration_pred": cal_pred, "test_pred": test_pred, "interval": interval,
            "metrics": metrics, "selected_cv_mae": selected_cv_mae, "baseline_cv_mae": baseline_cv_mae,
            "cv_skill": metadata["cv_skill_vs_median_baseline"], "comparison": comparison, "extrapolation": extrapolation}


def train_model(df: pd.DataFrame, output_dir: Path, model_dir: Path, gate: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    parts = grouped_future_holdouts(df)
    dev = parts["development"]
    folds = grouped_rolling_folds(dev)
    specs = candidate_specs()
    ratio_result = _run_target_experiment(TARGET, "ratio", "percentage_points", parts, folds, specs, output_dir, model_dir, gate.get("run_manifest"))
    dollar_result = _run_target_experiment(DOLLAR_TARGET, "dollar", "usd_millions", parts, folds, specs, output_dir, model_dir, gate.get("run_manifest"))

    test = parts["test"]
    actual_dollars = test[DOLLAR_TARGET].to_numpy(dtype=float)
    ratio_to_dollar_pred = ratio_result["test_pred"] / 100.0 * test["target_revenue_usd_m"].to_numpy(dtype=float)
    dollar_baseline_pred = np.repeat(dollar_result["baseline_value"], len(test))
    ratio_as_dollar_metrics = metrics_for_test(actual_dollars, ratio_to_dollar_pred, dollar_baseline_pred, "usd_millions")
    combined = test[[c for c in ["deal_id", "announcement_date", "announcement_year", "acquirer", "target", "target_revenue_usd_m", TARGET, DOLLAR_TARGET] if c in test.columns]].copy()
    combined["ratio_model_prediction_pct"] = ratio_result["test_pred"]
    combined["ratio_model_predicted_synergy_usd_m"] = ratio_to_dollar_pred
    combined["ratio_model_residual_pp"] = combined[TARGET] - combined["ratio_model_prediction_pct"]
    combined["ratio_model_dollar_residual_usd_m"] = combined[DOLLAR_TARGET] - combined["ratio_model_predicted_synergy_usd_m"]
    combined["direct_dollar_model_prediction_usd_m"] = dollar_result["test_pred"]
    combined["direct_dollar_model_residual_usd_m"] = combined[DOLLAR_TARGET] - combined["direct_dollar_model_prediction_usd_m"]
    combined["median_baseline_dollar_prediction_usd_m"] = dollar_baseline_pred
    if ratio_result["interval"]["status"] == "EMPIRICAL_SPLIT_CONFORMAL_90_PERCENT_NOMINAL":
        q = ratio_result["interval"]["quantile"]
        combined["ratio_lower_90_pct"] = np.maximum(0.0, ratio_result["test_pred"] - q)
        combined["ratio_upper_90_pct"] = ratio_result["test_pred"] + q
        combined["ratio_lower_90_dollar_usd_m"] = combined["ratio_lower_90_pct"] / 100.0 * combined["target_revenue_usd_m"]
        combined["ratio_upper_90_dollar_usd_m"] = combined["ratio_upper_90_pct"] / 100.0 * combined["target_revenue_usd_m"]
    if dollar_result["interval"]["status"] == "EMPIRICAL_SPLIT_CONFORMAL_90_PERCENT_NOMINAL":
        qd = dollar_result["interval"]["quantile"]
        combined["direct_dollar_lower_90_usd_m"] = np.maximum(0.0, dollar_result["test_pred"] - qd)
        combined["direct_dollar_upper_90_usd_m"] = dollar_result["test_pred"] + qd
    combined.to_csv(output_dir / "final_future_test_predictions_both_targets.csv", index=False, date_format="%Y-%m-%d")

    out = {**gate,
           "status": "EVALUATED_EXPLORATORY", "model_fit_attempted": True,
           "primary_ratio_model": {"selected_candidate": ratio_result["selected"]["name"], "cv_mae_pp": ratio_result["selected_cv_mae"],
                                   "median_baseline_cv_mae_pp": ratio_result["baseline_cv_mae"], "cv_skill_vs_baseline": ratio_result["cv_skill"],
                                   "test_metrics": ratio_result["metrics"], "interval": ratio_result["interval"],
                                   "feature_support_diagnostics": ratio_result["extrapolation"]},
           "secondary_dollar_model": {"selected_candidate": dollar_result["selected"]["name"], "cv_mae_usd_m": dollar_result["selected_cv_mae"],
                                      "median_baseline_cv_mae_usd_m": dollar_result["baseline_cv_mae"], "cv_skill_vs_baseline": dollar_result["cv_skill"],
                                      "test_metrics": dollar_result["metrics"], "interval": dollar_result["interval"],
                                      "feature_support_diagnostics": dollar_result["extrapolation"]},
           "ratio_model_converted_to_dollar_comparison": ratio_as_dollar_metrics,
           "credibility_policy": "Report both targets; do not claim superiority unless the final future-period test beats the median baseline. A model with negative test skill is not an accuracy improvement.",
           "note": "Final future test years were not used for feature/model selection. A single future window remains a noisy estimate. The outcome is announced claims conditional on quantified disclosure, not realized synergy or causal effect."}
    (output_dir / "model_evaluation_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def make_run_manifest(input_path: Path, source_data_attested_real: bool) -> dict[str, Any]:
    """Capture provenance and whether the operator/source path attests to real research data."""
    import sklearn
    input_path = input_path.resolve()
    script_path = Path(__file__).resolve()
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "input_file_name": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "training_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "source_data_attested_real": bool(source_data_attested_real),
        "data_is_synthetic": False if source_data_attested_real else None,
        "project_version": "model_credibility_v3",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--attest-real-source-data", action="store_true",
                    help="Explicitly attest that a custom input CSV contains only source-reviewed real research observations.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    is_canonical_curated_seed = args.input.resolve() == DEFAULT_INPUT.resolve()
    source_attested_real = bool(args.attest_real_source_data or is_canonical_curated_seed)
    run_manifest = make_run_manifest(args.input, source_attested_real)
    audit, prep_summary = prepare_dataset(source)
    if audit.empty:
        ready = audit.copy()
    else:
        ready = audit.loc[audit["model_feature_ready"]].copy()
    gate = gate_status(ready, len(source))
    gate["run_manifest"] = run_manifest
    gate["preflight_status"] = prep_summary.get("status")
    gate["preflight_scope_n"] = prep_summary.get("scope_eligible_rows")
    gate["preflight_label_evidence_n"] = prep_summary.get("label_evidence_ready_rows")
    gate["preflight_feature_blocker_counts"] = prep_summary.get("blocker_counts", {})
    gate["preflight_payment_method_support_warnings"] = prep_summary.get("payment_method_support_warnings", [])
    gate["primary_predictors"] = FEATURES["full"]
    gate["target_only_variant_available_for_ablation"] = True
    gate["premium_excluded_from_predictors"] = True
    gate["subsequent_outcome_excluded_from_predictors"] = True
    gate["year_used_for_grouped_splits_not_as_predictor"] = True
    gate["model_fit_attempted"] = False
    gate["test_metric_values"] = None
    if not run_manifest["source_data_attested_real"]:
        gate["problems"].append("Input is not the canonical curated seed and has not been explicitly attested as source-reviewed real research data; use --attest-real-source-data only after reviewing source evidence.")
        gate["gate_pass"] = False
        gate["status"] = "BLOCKED_BEFORE_FIT"
    if not gate["gate_pass"]:
        gate["methodology"] = "No fit: strict data and temporal-validation gates remain in force."
        (args.output_dir / "model_evaluation_result.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        report = [
            "# Model accuracy and credibility audit v2", "", "**Status: BLOCKED BEFORE FIT.**", "",
            f"Rows in input seed: **{len(source)}**", f"Rows passing strict full-feature model checks: **{len(ready)}**",
            f"Rows with label evidence and cohort eligibility but feature caveats: **{int((audit['strict_label_candidate'] & ~audit['model_feature_ready']).sum()) if not audit.empty else 0}**", "",
            "## Hard blockers", *[f"- {p}" for p in gate["problems"]], "",
            "## Cohort and feature decisions", *[f"- `{k}`: {v}" for k, v in prep_summary.get("blocker_counts", {}).items()], "",
            "## Model improvements implemented", 
            "- Explicit US-exchange listing-at-announcement gate; US listing after completion does not qualify a historical announcement.",
            "- Source evidence separated into cohort, target-label, and input-feature checks.",
            "- Revenue source links must resolve to SEC filings; revenue publication dates must be strictly before announcement day; same-day order is treated as unverified.",
            "- Synergy and both revenue inputs must be explicitly normalized to USD millions; non-USD rows require documented currency conversion before admission.",
            "- The numeric synergy amount must appear in the annual-cost source quote and independently in the quote supporting the pre-tax basis.",
            "- Duplicate deal events are blocked pending adjudication to avoid pseudo-replication.",
            "- Model candidate set is deliberately restrained: median baseline, Ridge, median Quantile Regression (MAE-aligned), and Huber robust regression across raw/log1p targets and a target-only ablation.",
            "- Expanding-window cross-validation groups by announcement year; no random shuffle and no same-year train/test split.",
            "- One-standard-error selection prefers the simplest candidate when performance differences are not clear.",
            "- Last whole announcement years are reserved for calibration and untouched future testing.",
            "- 90% intervals are suppressed unless calibration n is at least 20.",
            "- Later completion/termination outcomes and acquisition premium cannot enter predictors.",
            "- Payment-method category support is explicitly reported; unobserved categories cannot be presented as validated predictions and categories with fewer than 10 observations are low-support.",
            "- The strict label validator checks that the pre-tax evidence quote itself contains the synergy claim amount, not merely the words 'pre-tax'.",
            "- Reproducibility manifest records Python/library versions, input SHA-256, and training-script SHA-256.",
            "- The prediction CLI refuses output unless the latest evaluation is successful and both artifacts match the evaluation input hash, preventing stale-model use.", "",
            "## Interpretation rule", "The target measures management-announced annual pre-tax cost-synergy point claims conditional on firms disclosing quantified claims. It does not measure realized savings or prove the claim is economically credible.", "",
            "No estimator was fitted; therefore no accuracy numbers are reported. The pipeline and gate have been tested with unit tests and a synthetic mechanics-only fixture; that fixture is not used as model data."
        ]
        (args.output_dir / "model_credibility_audit.md").write_text("\n".join(report), encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return 2
    result = train_model(ready, args.output_dir, args.model_dir, gate)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
