import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_model_dataset import prepare_dataset, standardize_payment_method
from train_synergy_model import grouped_future_holdouts, grouped_rolling_folds, gate_status, candidate_specs, _fit_and_predict, select_one_standard_error
from predict_synergy import make_input_frame, predict_one

SEED = ROOT / "data" / "processed" / "strict_verified_seed_observations.csv"


def load_seed():
    return pd.read_csv(SEED)


def test_target_recomputation_matches_source_normalized_target():
    audit, summary = prepare_dataset(load_seed())
    assert not audit.empty
    assert audit["target_abs_difference_pp"].dropna().max() < 0.02


def test_us_exchange_listing_is_checked_at_announcement_not_after_close():
    audit, _ = prepare_dataset(load_seed())
    amcor = audit.loc[audit.deal_id == "AMCOR_BEMIS"].iloc[0]
    assert not bool(amcor.scope_eligible)
    assert "both_parties_not_verified_us_exchange_listed_at_announcement" in amcor.scope_flags
    assert not bool(amcor.model_feature_ready)


def test_primary_feature_ready_rows_exclude_amcor_and_include_chevron_after_documented_harmonization():
    audit, summary = prepare_dataset(load_seed())
    ready = audit.loc[audit.model_feature_ready]
    assert set(ready.deal_id) == {"EATON_COOPER_COST_COMPONENT", "ENSCO_ROWAN", "CHEVRON_NOBLE", "STERIS_CANTEL", "LAM_KLA"}
    assert "AMCOR_BEMIS" not in set(ready.deal_id)   # fails US-listing scope at announcement (ASX-listed)
    assert summary["primary_full_feature_ready_rows"] == 5
    assert summary["target_only_feature_ready_diagnostic_rows"] == 5


def test_tax_basis_requires_explicit_tax_quote_and_source_url():
    src = load_seed()
    src.loc[src.deal_id == "LAM_KLA", "tax_basis_source_quote"] = "cost synergies expected"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "LAM_KLA"].iloc[0]
    assert "tax_basis_quote_does_not_explicitly_say_pre_tax" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_strict_tax_basis_requires_sec_filing_source_type():
    src = load_seed()
    src.loc[src.deal_id == "STERIS_CANTEL", "tax_basis_source_type"] = "OFFICIAL_COMPANY_PRESS_RELEASE"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "STERIS_CANTEL"].iloc[0]
    assert "tax_basis_evidence_not_from_sec_filing" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_revenue_filing_must_precede_announcement_date():
    src = load_seed()
    src.loc[0, "target_revenue_filing_date"] = src.loc[0, "announcement_date"]
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == src.loc[0, "deal_id"]].iloc[0]
    assert "target_revenue_not_proven_public_before_announcement" in row.feature_readiness_flags
    assert not bool(row.model_feature_ready)


def test_non_strict_tax_basis_is_blocked():
    src = load_seed()
    src.loc[0, "tax_basis_source_quote"] = "estimated savings after integration"
    src.loc[0, "tax_basis"] = "unclear"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == src.loc[0, "deal_id"]].iloc[0]
    assert "pre_tax_basis_not_explicit" in row.label_evidence_flags


def test_lower_bound_and_range_claims_are_blocked():
    src = load_seed()
    src.loc[0, "range_or_lower_bound"] = "AT_LEAST"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == src.loc[0, "deal_id"]].iloc[0]
    assert "claim_not_a_single_point_estimate" in row.label_evidence_flags


def test_missing_source_url_is_blocked():
    src = load_seed()
    src.loc[0, "tax_basis_source_url"] = ""
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == src.loc[0, "deal_id"]].iloc[0]
    assert "missing_tax_basis_source_url" in row.label_evidence_flags


def test_duplicate_transaction_is_blocked_to_avoid_pseudoreplication():
    src = load_seed()
    extra = src.iloc[[0]].copy()
    extra.loc[:, "deal_id"] = "DUPLICATE_EATON"
    src = pd.concat([src, extra], ignore_index=True)
    audit, _ = prepare_dataset(src)
    rows = audit.loc[audit.acquirer == src.loc[0, "acquirer"]]
    assert rows.duplicate_transaction_event.all()
    assert rows.scope_flags.str.contains("duplicate_transaction_event_requires_adjudication").all()


def test_payment_method_normalization():
    assert standardize_payment_method("all stock") == "stock"
    assert standardize_payment_method("mixed cash and stock") == "mixed"
    assert standardize_payment_method("cash consideration") == "cash"
    assert standardize_payment_method("unknown") is None


def test_no_split_is_assigned_to_tiny_pilot_data():
    audit, _ = prepare_dataset(load_seed())
    ready = audit.loc[audit.model_feature_ready]
    parts = grouped_future_holdouts(ready)
    assert all(len(parts[k]) == 0 for k in ("development", "calibration", "test"))
    gate = gate_status(ready, len(load_seed()))
    assert gate["gate_pass"] is False
    assert gate["eligible_n"] == 5
    assert gate["status"] == "BLOCKED_BEFORE_FIT"


def test_future_test_and_rolling_folds_keep_years_intact():
    rows = []
    for year in range(2005, 2023):
        for j in range(7):
            rows.append({
                "deal_id": f"{year}_{j}", "announcement_date": f"{year}-06-{j+1:02d}",
                "announcement_year": year, "target_recomputed_pct": 4.0 + (year - 2005) * 0.15 + 0.05*j,
                "feature_log_acquirer_revenue": 8.0 + 0.1*j,
                "feature_log_target_revenue": 7.0 + 0.07*j,
                "payment_method_model": ["stock", "mixed", "cash"][j % 3], "model_feature_ready": True,
            })
    df = pd.DataFrame(rows)
    parts = grouped_future_holdouts(df)
    assert len(parts["development"]) >= 60
    assert len(parts["calibration"]) >= 10
    assert len(parts["test"]) >= 20
    year_sets = {k:set(v.announcement_year.astype(int)) for k,v in parts.items()}
    assert year_sets["development"].isdisjoint(year_sets["calibration"])
    assert year_sets["development"].isdisjoint(year_sets["test"])
    assert year_sets["calibration"].isdisjoint(year_sets["test"])
    folds = grouped_rolling_folds(parts["development"])
    assert len(folds) == 3
    for tr, va, meta in folds:
        train_years = set(pd.to_datetime(parts["development"].iloc[tr].announcement_date).dt.year.astype(int))
        val_years = set(pd.to_datetime(parts["development"].iloc[va].announcement_date).dt.year.astype(int))
        assert train_years.isdisjoint(val_years)
        assert max(train_years) < min(val_years)
        assert meta["train_n"] >= 30
        assert meta["validation_n"] >= 5


def test_one_standard_error_selection_favours_baseline_when_differences_are_unclear():
    scores = pd.DataFrame([
        {"candidate":"median_baseline", "mae":2.0, "fold":1},
        {"candidate":"median_baseline", "mae":2.1, "fold":2},
        {"candidate":"median_baseline", "mae":1.9, "fold":3},
        {"candidate":"ridge_target_only_raw_a1", "mae":1.9, "fold":1},
        {"candidate":"ridge_target_only_raw_a1", "mae":2.0, "fold":2},
        {"candidate":"ridge_target_only_raw_a1", "mae":2.0, "fold":3},
    ])
    specs = [
        {"name":"median_baseline", "family":"baseline", "feature_set":"none", "target_transform":"none", "params":{}, "complexity":0},
        {"name":"ridge_target_only_raw_a1", "family":"ridge", "feature_set":"target_only", "target_transform":"raw", "params":{"alpha":1.0}, "complexity":1},
    ]
    selected, summary, threshold = select_one_standard_error(scores, specs)
    assert selected["name"] == "median_baseline"


def test_candidate_prediction_plumbing_on_ephemeral_mechanics_fixture_only():
    # This tiny generated fixture validates code mechanics only; it is never written to project data or used for research metrics.
    rng = np.random.default_rng(10)
    train = pd.DataFrame({
        "feature_log_target_revenue": rng.normal(8, 0.5, 35),
        "feature_log_acquirer_revenue": rng.normal(9, 0.6, 35),
        "payment_method_model": ["stock", "mixed", "cash"] * 11 + ["stock", "mixed"],
        "target_recomputed_pct": rng.uniform(2, 15, 35),
    })
    val = pd.DataFrame({"feature_log_target_revenue":[8.2,7.9], "feature_log_acquirer_revenue":[9.1,8.7], "payment_method_model":["cash","new_unknown"]})
    spec = next(s for s in candidate_specs() if s["family"] == "ridge" and s["feature_set"] == "full" and s["target_transform"] == "log1p")
    pred = _fit_and_predict(spec, train, train.target_recomputed_pct.to_numpy(), val)
    assert len(pred) == 2
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()


def test_payment_method_must_have_source_quote_support_and_announcement_date():
    src = load_seed()
    src.loc[src.deal_id == "EATON_COOPER_COST_COMPONENT", "payment_method_source_quote"] = "The companies expect strong synergies."
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "EATON_COOPER_COST_COMPONENT"].iloc[0]
    assert "payment_method_source_quote_does_not_support_category" in row.feature_readiness_flags
    assert not bool(row.model_feature_ready)


def test_claim_quote_must_match_numeric_label_and_annual_cost_definition():
    src = load_seed()
    src.loc[src.deal_id == "ENSCO_ROWAN", "source_quote"] = "$250 million of annual pre-tax expense synergies identified."
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "ENSCO_ROWAN"].iloc[0]
    assert "claim_source_quote_does_not_match_numeric_label" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_sec_tax_basis_source_requires_sec_domain_not_only_type_flag():
    src = load_seed()
    src.loc[src.deal_id == "LAM_KLA", "tax_basis_source_url"] = "https://example.com/fake-sec-filing"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "LAM_KLA"].iloc[0]
    assert "tax_basis_evidence_not_from_sec_filing" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_reverse_party_order_duplicate_is_flagged():
    src = load_seed()
    extra = src.loc[src.deal_id == "EATON_COOPER_COST_COMPONENT"].copy()
    extra.loc[:, "deal_id"] = "DUPLICATE_REVERSED"
    extra.loc[:, "acquirer"] = src.loc[src.deal_id == "EATON_COOPER_COST_COMPONENT", "target"].iloc[0]
    extra.loc[:, "target"] = src.loc[src.deal_id == "EATON_COOPER_COST_COMPONENT", "acquirer"].iloc[0]
    src = pd.concat([src, extra], ignore_index=True)
    audit, _ = prepare_dataset(src)
    assert audit.loc[audit.deal_id.isin(["EATON_COOPER_COST_COMPONENT", "DUPLICATE_REVERSED"]), "duplicate_transaction_event"].all()


def test_revenue_source_urls_must_be_sec_filings():
    src = load_seed()
    src.loc[src.deal_id == "LAM_KLA", "target_revenue_source_url"] = "https://example.com/company-revenue"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "LAM_KLA"].iloc[0]
    assert "revenue_source_not_sec_filing:target_revenue_source_url" in row.feature_readiness_flags
    assert not bool(row.model_feature_ready)


def test_pre_tax_quote_must_tie_tax_basis_to_the_numeric_claim():
    src = load_seed()
    src.loc[src.deal_id == "STERIS_CANTEL", "tax_basis_source_quote"] = "Annualized pre-tax cost synergies of $900 million."
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "STERIS_CANTEL"].iloc[0]
    assert "tax_basis_quote_does_not_tie_pre_tax_basis_to_claim_amount" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_prediction_input_builder_validates_units_and_log_features():
    row = make_input_frame(10000.0, 2000.0, "mixed")
    assert np.isclose(row.loc[0, "feature_log_acquirer_revenue"], np.log(10000.0))
    assert np.isclose(row.loc[0, "feature_log_target_revenue"], np.log(2000.0))
    assert row.loc[0, "payment_method_model"] == "mixed"
    import pytest
    with pytest.raises(ValueError):
        make_input_frame(-1.0, 2000.0, "mixed")


def test_prediction_is_refused_when_no_real_trained_artifacts_exist(tmp_path):
    result = predict_one(10000.0, 2000.0, "mixed", tmp_path)
    assert result["status"] == "MODEL_NOT_AVAILABLE"
    assert result["predictions"] is None


def test_non_usd_synergy_label_is_blocked_until_converted_and_documented():
    src = load_seed()
    src.loc[src.deal_id == "LAM_KLA", "synergy_currency"] = "EUR"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "LAM_KLA"].iloc[0]
    assert "synergy_currency_not_usd_or_not_verified" in row.label_evidence_flags
    assert not bool(row.strict_label_candidate)


def test_non_usd_revenue_input_is_blocked():
    src = load_seed()
    src.loc[src.deal_id == "LAM_KLA", "target_revenue_currency"] = "EUR"
    audit, _ = prepare_dataset(src)
    row = audit.loc[audit.deal_id == "LAM_KLA"].iloc[0]
    assert "target_revenue_currency_not_usd_or_not_verified" in row.feature_readiness_flags
    assert not bool(row.model_feature_ready)


def test_gate_exposes_payment_method_class_imbalance():
    audit, _ = prepare_dataset(load_seed())
    ready = audit.loc[audit.model_feature_ready].copy()
    gate = gate_status(ready, len(load_seed()))
    assert gate["payment_method_category_counts"] == {"mixed": 3, "stock": 2}
    assert any("cash" in warning for warning in gate["payment_method_support_warnings"])


def test_predictor_refuses_stale_artifacts_when_latest_gate_is_blocked(tmp_path):
    import json
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ratio_model_metadata.json").write_text(json.dumps({"run_manifest": {"input_sha256": "abc"}}))
    (model_dir / "dollar_model_metadata.json").write_text(json.dumps({"run_manifest": {"input_sha256": "abc"}}))
    evaluation = tmp_path / "model_evaluation_result.json"
    evaluation.write_text(json.dumps({"status": "BLOCKED_BEFORE_FIT", "model_fit_attempted": False, "run_manifest": {"input_sha256": "abc"}}))
    result = predict_one(10000.0, 2000.0, "mixed", model_dir, evaluation)
    assert result["status"] == "MODEL_NOT_AVAILABLE"
    assert "stale artifact" in result["reason"]


def test_predictor_rejects_synthetic_mechanics_artifacts(tmp_path):
    import json
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    manifest = {"input_sha256": "abc", "training_script_sha256": "def", "data_is_synthetic": True, "source_data_attested_real": False}
    for key in ["ratio", "dollar"]:
        (model_dir / f"{key}_model_metadata.json").write_text(json.dumps({"run_manifest": manifest}))
    evaluation = tmp_path / "model_evaluation_result.json"
    evaluation.write_text(json.dumps({"status": "EVALUATED_EXPLORATORY", "model_fit_attempted": True, "run_manifest": manifest}))
    result = predict_one(10000.0, 2000.0, "mixed", model_dir, evaluation)
    assert result["status"] == "MODEL_NOT_AVAILABLE"
    assert "unattested" in result["reason"] or "synthetic" in result["reason"]


def test_predictors_exclude_premium_and_post_announcement_or_outcome_fields():
    from train_synergy_model import FEATURES
    forbidden = {"reported_premium_percent", "acquisition_premium", "completion_status", "termination_status",
                 "actual_synergies", "realized_synergies", "completion_date", "announcement_year", "target_recomputed_pct",
                 "announced_annual_cost_synergy_usd_m", "transaction_value_usd_m"}
    used = {col for cols in FEATURES.values() for col in cols}
    assert used.isdisjoint(forbidden)
    assert "feature_log_acquirer_revenue" in FEATURES["full"]
    assert "feature_log_target_revenue" in FEATURES["full"]
    assert "payment_method_model" in FEATURES["full"]


def test_quantile_regression_candidate_is_mae_aligned_and_predicts_nonnegative():
    rng = np.random.default_rng(44)
    train = pd.DataFrame({
        "feature_log_target_revenue": rng.normal(8.0, 0.4, 36),
        "feature_log_acquirer_revenue": rng.normal(9.0, 0.5, 36),
        "payment_method_model": ["mixed", "stock", "cash"] * 12,
    })
    y = 6.0 + 0.7 * (train["feature_log_target_revenue"].to_numpy() - 8.0) + rng.normal(0, 0.3, 36)
    pred_frame = pd.DataFrame({
        "feature_log_target_revenue": [8.2, 7.9],
        "feature_log_acquirer_revenue": [9.2, 8.5],
        "payment_method_model": ["mixed", "cash"],
    })
    spec = next(s for s in candidate_specs() if s["family"] == "quantile" and s["feature_set"] == "full" and s["target_transform"] == "raw")
    pred = _fit_and_predict(spec, train, y, pred_frame)
    assert len(pred) == 2
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()


# ---- regression tests (defects found in independent validation) -------------------------------
import tempfile
import train_synergy_model as train
import numpy as np
from build_model_dataset import (quote_mentions_usd_million_amount as _qamt, synergy_type_is_cost_only as _cost_only,
                                 standardize_payment_method as _std_pay, payment_quote_matches_category as _pay_quote)


def test_per_share_cash_price_is_not_classified_as_mixed():
    assert _std_pay("all cash ($62 per share)") == "cash"
    assert _std_pay("all cash (165p / 340 yen per share)") == "cash"
    assert _std_pay("all stock (0.6966 SLB shares per Smith share)") == "stock"
    assert _std_pay("cash and stock ($32.00 + 0.5 Lam share)") == "mixed"


def test_unverified_payment_label_never_becomes_a_category():
    assert _std_pay("to verify") is None
    assert _std_pay("to verify (share exchange plus cash distributions)") is None


def test_payment_quote_support_handles_standard_sec_phrasing():
    assert _pay_quote("all cash", "each share will be converted into the right to receive $40.75 in cash per share")
    assert _pay_quote("all stock", "converted into 0.7 Acquirer shares, with cash in lieu of fractional shares")
    assert not _pay_quote("all cash", "0.7 shares of Acquirer stock per Target share")


def test_amount_matcher_ignores_bare_numbers():
    assert not _qamt("expects $200 million of savings and employs 375 people", 375)
    assert not _qamt("closing in Q3 2012; synergy of $260 million", 2012)
    assert _qamt("$260m cost-out", 260) and _qamt("$1.0B of savings", 1000) and _qamt("US$ 1.5 billion", 1500)


def test_cost_only_type_requires_affirmative_evidence():
    for bad in ["earnings synergies (not cost-only)", "mixed: cost synergies and tax savings combined",
                "cost and sales synergies", "mixed: $1.0B operating cost savings + $1.5B reduction in exploration expenditure"]:
        assert not _cost_only(bad), bad
    for good in ["cost (operating)", "cost component of a mixed $135M total", "cost-only component of a larger pre-tax estimate"]:
        assert _cost_only(good), good


def test_partial_announcement_dates_are_scope_blocked():
    from build_model_dataset import prepare_dataset
    import pandas as pd
    seed = pd.read_csv(ROOT / "data" / "processed" / "strict_verified_seed_observations.csv")
    seed.loc[0, "announcement_date"] = "2012-05"
    audit, _ = prepare_dataset(seed)
    assert "announcement_date_not_day_precision" in audit.loc[0, "scope_flags"]


def _synthetic(n, years, signal, seed):
    r = np.random.default_rng(seed); yrs = np.arange(2005, 2005 + years)
    y = np.concatenate([yrs, r.choice(yrs, n - years)])
    ta = np.exp(r.normal(6.5, 1.0, n)); aq = np.exp(r.normal(8.5, 1.2, n)); pm = r.choice(["cash", "stock", "mixed"], n)
    pct = np.clip(8 - 0.9 * np.log(ta) + r.normal(0, 0.6, n), 0.3, None) if signal else np.clip(r.lognormal(1.0, 0.6, n), 0.3, None)
    return pd.DataFrame({"deal_id": [f"S{i}" for i in range(n)], "announcement_date": pd.to_datetime([f"{v}-06-15" for v in y]),
        "announcement_year": y, "acquirer": "A", "target": "T", "target_revenue_usd_m": ta, "acquirer_revenue_usd_m": aq,
        "feature_log_target_revenue": np.log(ta), "feature_log_acquirer_revenue": np.log(aq), "payment_method_model": pm,
        "target_recomputed_pct": pct, "announced_annual_cost_synergy_usd_m": pct / 100 * ta})


def test_machinery_claims_no_skill_on_pure_noise_and_detects_planted_signal():
    """MECHANICS ONLY: synthetic data, temp dir, never mixed with project data."""
    from pathlib import Path
    import warnings; warnings.filterwarnings("ignore")
    for signal in (False, True):
        df = _synthetic(220, 14, signal, seed=3); gate = train.gate_status(df, len(df)); assert gate["gate_pass"]
        with tempfile.TemporaryDirectory() as d:
            m = train.train_model(df, Path(d) / "o", Path(d) / "m", gate)["primary_ratio_model"]["test_metrics"]
        assert m["improves_on_median_baseline"] is signal


def test_compact_ledger_never_labels_a_lower_bound_as_a_point():
    from make_compact_ledger import compact
    m = pd.read_csv(ROOT / "data" / "processed" / "MA_Synergy_Ledger_MASTER.csv")
    c = compact(m).set_index("deal_id")
    assert len(c) == len(m) and c.loc["MDT_COV", "kind"] == "L" and c.loc["MDT_COV", "usd_m"] == ">=850"


def test_documented_revenue_definition_resolution_clears_review_flag_but_undocumented_yes_does_not():
    seed = load_seed(); i = seed.index[seed.deal_id == "CHEVRON_NOBLE"][0]
    audit, _ = prepare_dataset(seed); assert "revenue_definition_review_required" not in str(audit.loc[i, "blocking_flags"])
    seed.loc[i, "revenue_definition_review_note"] = None            # YES without documentation must not clear the flag
    audit, _ = prepare_dataset(seed); assert "revenue_definition_review_required" in str(audit.loc[i, "blocking_flags"])


def test_benchmark_loo_never_sees_the_held_out_deal_and_screening_scales_linearly():
    from benchmark_estimator import loo, screening_range
    df = pd.read_csv(ROOT / "data" / "processed" / "model_features_primary.csv"); df = df[df["model_feature_ready"] == True].reset_index(drop=True)
    base, _ = loo(df)
    tweaked = df.copy(); tweaked.loc[0, "target_recomputed_pct"] = 99.0          # change ONLY deal 0's own value
    again, _ = loo(tweaked)
    assert abs(base.loc[0, "median_pred"] - again.loc[0, "median_pred"]) < 1e-12   # its own LOO prediction must not move
    a, b = screening_range(df, 1000), screening_range(df, 2000)
    assert abs(b["median_synergy_usd_m"] - 2 * a["median_synergy_usd_m"]) < 0.2 and a["n"] == len(df)


def test_curated_ledger_readiness_matches_pipeline_admitted_rows():
    """The curated ledger and the pipeline must agree on which deals are admitted (they drifted apart once)."""
    cur = pd.read_csv(ROOT / "data" / "processed" / "MA_Synergy_Ledger_Curated.csv")
    reg = pd.read_csv(ROOT / "data" / "processed" / "MA_Synergy_Data_Issue_Register.csv")
    feats = pd.read_csv(ROOT / "data" / "processed" / "model_features_primary.csv")
    admitted = set(feats["source_ledger_deal_id"])
    assert set(cur.loc[cur.training_readiness_clean == "YES", "deal_id"]) == admitted
    assert set(reg.loc[reg.training_readiness_clean == "YES", "deal_id"]) == admitted
