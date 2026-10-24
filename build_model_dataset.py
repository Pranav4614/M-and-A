#!/usr/bin/env python3
"""Strict, source-auditable dataset preflight for announced M&A cost-synergy claims.

The pipeline separates (1) label evidence, (2) cohort eligibility, and (3) predictor readiness.
It never fits a model, imputes label values, or treats later transaction outcomes as predictors.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _portable(path: Path) -> str:
    """Path relative to the repo root (never leak a local absolute path into committed outputs)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name
DEFAULT_INPUT = ROOT / "data" / "processed" / "strict_verified_seed_observations.csv"
DEFAULT_OUT = ROOT / "data" / "processed"

REQUIRED = [
    "deal_id", "announcement_date", "acquirer", "target", "payment_method",
    "acquirer_revenue_usd_m", "target_revenue_usd_m", "announced_annual_cost_synergy_usd_m",
    "synergy_currency", "acquirer_revenue_currency", "target_revenue_currency",
    "normalized_cost_synergy_pct_target_revenue", "strict_label_approved_for_pilot", "tax_basis",
    "tax_basis_source_url", "tax_basis_source_type", "tax_basis_source_quote", "tax_basis_source_publication_date", "claim_source_publication_date",
    "range_or_lower_bound", "synergy_period", "synergy_type",
    "acquirer_revenue_available_by_announcement", "target_revenue_available_by_announcement",
    "payment_method_available_by_announcement", "acquirer_revenue_filing_date", "target_revenue_filing_date",
    "public_public_nonfinancial_screen", "both_parties_us_exchange_listed_at_announcement",
    "us_listing_evidence_url", "acquirer_revenue_source_url", "target_revenue_source_url",
    "synergy_source_url", "payment_method_source_url", "payment_method_source_quote",
    "payment_method_source_publication_date", "payment_method_source_type", "source_quote",
    "revenue_definition_harmonized", "qualifier", "range_or_lower_bound",
]
NUMERIC = ["acquirer_revenue_usd_m", "target_revenue_usd_m", "announced_annual_cost_synergy_usd_m"]
START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2022-12-31")


def norm(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def yes(v: Any) -> bool:
    return norm(v) in {"y", "yes", "true", "1", "pass"}


def pass_status(v: Any) -> bool:
    return norm(v) == "pass"


def explicit_pretax(v: Any) -> bool:
    return bool(re.search(r"pre\s*[- ]?tax|before\s+tax|pretax", norm(v)))


def has_url(v: Any) -> bool:
    return norm(v).startswith(("https://", "http://"))


def is_sec_url(v: Any) -> bool:
    if not has_url(v):
        return False
    host = (urlparse(str(v).strip()).hostname or "").lower()
    return host == "sec.gov" or host.endswith(".sec.gov")


def quote_mentions_usd_million_amount(quote: Any, amount_usd_m: float) -> bool:
    """Check that a source quote contains the claimed amount in $Xm / $Xbn / $Xk form.

    A number only counts as a money amount if it carries a currency marker ($, US$, USD) or an
    explicit unit (million/mm/m, billion/bn/b, thousand/k). Bare numbers ("375 employees", "2012")
    no longer match, which previously allowed false ties between quote and label.
    """
    if pd.isna(amount_usd_m) or not np.isfinite(float(amount_usd_m)):
        return False
    text = norm(quote).replace(",", "")
    if not text or text == "nan":
        return False
    pattern = re.compile(
        r"(?P<cur>us\$|usd\s*|\$)?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>billion|bn|million|mm|m|thousand|k|b)?(?![a-z0-9])",
        re.I,
    )
    amounts = []
    for m in pattern.finditer(text):
        unit = (m.group("unit") or "").lower()
        if not m.group("cur") and not unit:
            continue                      # bare number: not evidence of a money amount
        if unit == "b" and not m.group("cur"):
            continue                      # lone 'b' only counts with a currency marker
        value = float(m.group("num"))
        if unit in {"billion", "bn", "b"}:
            value *= 1000.0
        elif unit in {"thousand", "k"}:
            value *= 0.001
        amounts.append(value)
    tolerance = max(0.05, abs(float(amount_usd_m)) * 0.005)
    return any(abs(v - float(amount_usd_m)) <= tolerance for v in amounts)


_PER_SHARE = re.compile(r"\bper\s+(?:[a-z0-9&.\-]+\s+){0,3}shares?\b")
_CASH_IN_LIEU = re.compile(r"\bcash\s+in\s+lieu(?:\s+of\s+(?:any\s+)?(?:fractional\s+)?(?:shares?|interests?))?")
# Acquirer-share consideration evidence: all-stock wording or an exchange ratio like "0.7966 ... shares".
_STOCK_CONSIDERATION = re.compile(
    r"\ball[- ](?:stock|share|equity)\b|\bstock[- ]for[- ]stock\b|\bexchange ratio\b|\bstock merger\b"
    r"|\b\d+(?:\.\d+)?\s+(?:[a-z0-9&.\-]+\s+){0,3}(?:shares|ads|adss)\b"
)


def payment_quote_matches_category(method: Any, quote: Any) -> bool:
    """Does the source quote support the coded payment category?

    'per share' (a price quoted per TARGET share) and 'cash in lieu of fractional shares' are not
    evidence of stock consideration; treating them as such mislabeled many cash and stock deals.
    """
    category = standardize_payment_method(method)
    q = norm(quote)
    if not q or category is None:
        return False
    q_clean = _CASH_IN_LIEU.sub(" ", _PER_SHARE.sub(" ", q))
    mentions_cash = bool(re.search(r"\bcash\b", q_clean))
    loose_stock = bool(re.search(r"\b(stock|share|shares|ordinary shares|common stock|equity)\b", q_clean))
    strict_stock = bool(_STOCK_CONSIDERATION.search(q_clean))
    if category == "mixed":
        return mentions_cash and loose_stock
    if category == "cash":
        return mentions_cash and not strict_stock
    if category == "stock":
        return strict_stock and not mentions_cash
    return False


def standardize_payment_method(v: Any) -> str | None:
    s = norm(v).replace("-", " ").replace("/", " ")
    if s in {"", "nan", "none", "unknown", "n a", "not disclosed", "undisclosed"}:
        return None
    if s.startswith("to verify"):
        return None                      # unverified labels must never become a model category
    s = _PER_SHARE.sub(" ", s)           # "$62 per share" is a cash price, not share consideration
    if "mixed" in s or ("cash" in s and ("stock" in s or "share" in s)):
        return "mixed"
    if "stock" in s or "share" in s or "equity" in s:
        return "stock"
    if "cash" in s:
        return "cash"
    return None


def synergy_type_is_cost_only(v: Any) -> bool:
    """True only if the type text affirms a cost-only claim. Substring 'cost' alone is not enough
    ('not cost-only', 'mixed: cost synergies and tax savings', 'cost and sales synergies' all contain it)."""
    t = norm(v)
    if "cost" not in t:
        return False
    if re.search(r"not\s+cost[- ]only|unclear|mixed/unclear", t):
        return False
    if re.match(r"cost[- ]only|cost component", t):   # explicit separable cost line inside a larger claim
        return True
    return not re.search(
        r"\bmixed\b|tax\s+(?:benefit|saving|synerg)|revenue synerg|sales synerg|earnings synerg|financ(?:ing|ial) synerg", t)


def _point_claim_status(row: pd.Series) -> bool:
    """Allow one approximate point estimate; block ranges and lower bounds."""
    if norm(row.get("range_or_lower_bound", "")) != "no":
        return False
    amount = pd.to_numeric(row.get("announced_annual_cost_synergy_usd_m"), errors="coerce")
    low = pd.to_numeric(row.get("synergy_low_usd_m"), errors="coerce") if "synergy_low_usd_m" in row else np.nan
    high = pd.to_numeric(row.get("synergy_high_usd_m"), errors="coerce") if "synergy_high_usd_m" in row else np.nan
    if pd.isna(amount) or amount <= 0:
        return False
    # If low/high columns are present, they must not express a non-point range.
    if pd.notna(low) and pd.notna(high) and abs(float(low) - float(high)) > 1e-9:
        return False
    return True


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return pd.DataFrame(), {
            "status": "SCHEMA_BLOCKED", "input_rows": int(len(df)), "missing_columns": missing,
            "model_feature_ready_rows": 0, "note": "Required source/audit columns are missing; no rows were admitted."
        }
    x = df.copy()
    x["announcement_date_day_precision"] = x["announcement_date"].astype(str).str.strip().str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$")
    x["announcement_date"] = pd.to_datetime(x["announcement_date"], errors="coerce", format="mixed")
    for c in ["acquirer_revenue_filing_date", "target_revenue_filing_date", "tax_basis_source_publication_date", "claim_source_publication_date", "payment_method_source_publication_date"]:
        x[c] = pd.to_datetime(x[c], errors="coerce", format="mixed")
    for c in NUMERIC + ["normalized_cost_synergy_pct_target_revenue"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x["announcement_year"] = x["announcement_date"].dt.year
    x["target_recomputed_pct"] = 100.0 * x["announced_annual_cost_synergy_usd_m"] / x["target_revenue_usd_m"]
    x.loc[~np.isfinite(x["target_recomputed_pct"]), "target_recomputed_pct"] = np.nan
    x["target_abs_difference_pp"] = (
        x["target_recomputed_pct"] - x["normalized_cost_synergy_pct_target_revenue"]
    ).abs()
    x["feature_log_acquirer_revenue"] = np.log(x["acquirer_revenue_usd_m"].where(x["acquirer_revenue_usd_m"] > 0))
    x["feature_log_target_revenue"] = np.log(x["target_revenue_usd_m"].where(x["target_revenue_usd_m"] > 0))
    x["target_to_acquirer_revenue_ratio"] = x["target_revenue_usd_m"] / x["acquirer_revenue_usd_m"]
    x["payment_method_model"] = x["payment_method"].map(standardize_payment_method)

    # Duplicate event detection prevents two differently defined labels for the same deal from acting like independent observations.
    # Make party order irrelevant so the same transaction entered as Acquirer/Target or Target/Acquirer is flagged.
    party_pair = x.apply(lambda r: "|".join(sorted([norm(r["acquirer"]), norm(r["target"])])), axis=1)
    event_key = x["announcement_date"].dt.strftime("%Y-%m-%d").fillna("") + "|" + party_pair
    duplicated_event = event_key.ne("||") & event_key.duplicated(keep=False)
    x["duplicate_transaction_event"] = duplicated_event

    scope_flags: list[list[str]] = []
    label_flags: list[list[str]] = []
    feature_flags: list[list[str]] = []
    for _, r in x.iterrows():
        sf, lf, ff = [], [], []
        # Scope: announced within project years, public-to-public/non-financial, and both parties US-exchange-listed at announcement.
        if not bool(r["announcement_date_day_precision"]): sf.append("announcement_date_not_day_precision")
        if pd.isna(r["announcement_date"]): sf.append("missing_or_invalid_announcement_date")
        elif not START_DATE <= r["announcement_date"] <= END_DATE: sf.append("outside_approved_announcement_window")
        if not pass_status(r["public_public_nonfinancial_screen"]): sf.append("public_public_nonfinancial_screen_not_passed")
        if not yes(r["both_parties_us_exchange_listed_at_announcement"]): sf.append("both_parties_not_verified_us_exchange_listed_at_announcement")
        if not has_url(r["us_listing_evidence_url"]): sf.append("missing_us_listing_evidence_url")
        if bool(r["duplicate_transaction_event"]): sf.append("duplicate_transaction_event_requires_adjudication")

        # Label evidence: only explicit pre-tax, cost-only, annual, single-point claim rows are primary labels.
        if not yes(r["strict_label_approved_for_pilot"]): lf.append("strict_label_not_approved")
        if not explicit_pretax(r["tax_basis"]): lf.append("pre_tax_basis_not_explicit")
        if not has_url(r["tax_basis_source_url"]): lf.append("missing_tax_basis_source_url")
        if not is_sec_url(r["tax_basis_source_url"]) or not norm(r["tax_basis_source_type"]).startswith("sec_"):
            lf.append("tax_basis_evidence_not_from_sec_filing")
        if not re.search(r"pre\s*[- ]?tax|before\s+tax|pretax", norm(r["tax_basis_source_quote"])):
            lf.append("tax_basis_quote_does_not_explicitly_say_pre_tax")
        if not quote_mentions_usd_million_amount(r["tax_basis_source_quote"], r["announced_annual_cost_synergy_usd_m"]):
            lf.append("tax_basis_quote_does_not_tie_pre_tax_basis_to_claim_amount")
        if pd.isna(r["tax_basis_source_publication_date"]): lf.append("tax_basis_source_date_missing")
        if pd.isna(r["claim_source_publication_date"]) or pd.isna(r["announcement_date"]) or r["claim_source_publication_date"] > r["announcement_date"]:
            lf.append("synergy_claim_source_not_verified_by_announcement_date")
        if not _point_claim_status(r): lf.append("claim_not_a_single_point_estimate")
        if not synergy_type_is_cost_only(r["synergy_type"]):
            lf.append("synergy_type_not_confirmed_cost_only")
        if not re.search(r"annual|per annum|annualized|run.rate", norm(r["synergy_period"])):
            lf.append("synergy_period_not_annual")
        if any(pd.isna(r[c]) or float(r[c]) <= 0 for c in NUMERIC): lf.append("missing_or_nonpositive_label_or_revenue")
        if norm(r["synergy_currency"]) != "usd": lf.append("synergy_currency_not_usd_or_not_verified")
        if not has_url(r["synergy_source_url"]): lf.append("missing_synergy_source_url")
        if not str(r["source_quote"]).strip() or norm(r["source_quote"]) in {"nan", "none"}:
            lf.append("missing_source_quote")
        else:
            quote = norm(r["source_quote"])
            if not quote_mentions_usd_million_amount(r["source_quote"], r["announced_annual_cost_synergy_usd_m"]):
                lf.append("claim_source_quote_does_not_match_numeric_label")
            if not re.search(r"annual|per annum|annualized|run.rate", quote):
                lf.append("claim_source_quote_does_not_support_annual_period")
            if not re.search(r"cost|cost.out|expense", quote):
                lf.append("claim_source_quote_does_not_support_cost_only_label")
        if not is_sec_url(r["synergy_source_url"]):
            lf.append("synergy_claim_source_not_sec_filing_or_exhibit")
        if pd.isna(r["target_abs_difference_pp"]) or r["target_abs_difference_pp"] > 0.02:
            lf.append("normalized_target_mismatch_over_0_02pp")

        # Feature readiness: predictors must be known by announcement and have evidence links.
        if pd.isna(r["acquirer_revenue_usd_m"]) or r["acquirer_revenue_usd_m"] <= 0: ff.append("acquirer_revenue_missing_or_nonpositive")
        if pd.isna(r["target_revenue_usd_m"]) or r["target_revenue_usd_m"] <= 0: ff.append("target_revenue_missing_or_nonpositive")
        if norm(r["acquirer_revenue_currency"]) != "usd": ff.append("acquirer_revenue_currency_not_usd_or_not_verified")
        if norm(r["target_revenue_currency"]) != "usd": ff.append("target_revenue_currency_not_usd_or_not_verified")
        if not yes(r["acquirer_revenue_available_by_announcement"]): ff.append("acquirer_revenue_availability_not_verified")
        if not yes(r["target_revenue_available_by_announcement"]): ff.append("target_revenue_availability_not_verified")
        # Strictly before announcement, avoiding ambiguous same-day filing/announcement ordering.
        if pd.isna(r["announcement_date"]) or pd.isna(r["acquirer_revenue_filing_date"]):
            ff.append("acquirer_revenue_filing_date_missing")
        elif r["acquirer_revenue_filing_date"] >= r["announcement_date"]:
            ff.append("acquirer_revenue_not_proven_public_before_announcement")
        if pd.isna(r["announcement_date"]) or pd.isna(r["target_revenue_filing_date"]):
            ff.append("target_revenue_filing_date_missing")
        elif r["target_revenue_filing_date"] >= r["announcement_date"]:
            ff.append("target_revenue_not_proven_public_before_announcement")
        for c in ["acquirer_revenue_source_url", "target_revenue_source_url", "payment_method_source_url"]:
            if not has_url(r[c]): ff.append(f"missing_source_url:{c}")
        # This primary dataset is designed around filing-verifiable financial inputs.
        for c in ["acquirer_revenue_source_url", "target_revenue_source_url"]:
            if has_url(r[c]) and not is_sec_url(r[c]): ff.append(f"revenue_source_not_sec_filing:{c}")
        for c in ["acquirer_revenue_basis", "target_revenue_basis", "acquirer_revenue_fy", "target_revenue_fy"]:
            if norm(r.get(c, "")) in {"", "nan", "none"}: ff.append(f"revenue_basis_or_fiscal_year_missing:{c}")
        if not yes(r["payment_method_available_by_announcement"]): ff.append("payment_method_not_verified_available_by_announcement")
        if pd.isna(r["payment_method_model"]): ff.append("payment_method_missing_or_unclassifiable")
        payment_quote = norm(r["payment_method_source_quote"])
        if not payment_quote or payment_quote in {"nan", "none"}:
            ff.append("payment_method_source_quote_missing")
        elif not payment_quote_matches_category(r["payment_method"], payment_quote):
            ff.append("payment_method_source_quote_does_not_support_category")
        if not is_sec_url(r["payment_method_source_url"]) or not norm(r["payment_method_source_type"]).startswith("sec_"):
            ff.append("payment_method_source_not_sec_filing_or_exhibit")
        payment_pub_date = r["payment_method_source_publication_date"]
        if pd.isna(payment_pub_date) or pd.isna(r["announcement_date"]) or payment_pub_date > r["announcement_date"]:
            ff.append("payment_method_source_not_published_by_announcement")
        # An explicit YES is authoritative only when a documented review note exists; stale free-text keywords
        # in data_quality_notes must not override a documented resolution (and must not re-trigger on the word 'harmonized').
        documented_yes = yes(r["revenue_definition_harmonized"]) and norm(r.get("revenue_definition_review_note", "")) not in {"", "nan"}
        if not documented_yes: ff.append("revenue_definition_review_required")
        notes = norm(r.get("data_quality_notes", ""))
        if not documented_yes and any(term in notes for term in ["definition mismatch", "should be harmonized", "revenue line definition", "harmonize"]):
            if "revenue_definition_review_required" not in ff: ff.append("revenue_definition_review_required")
        scope_flags.append(sf); label_flags.append(lf); feature_flags.append(ff)

    x["scope_flags"] = [";".join(z) for z in scope_flags]
    x["label_evidence_flags"] = [";".join(z) for z in label_flags]
    x["feature_readiness_flags"] = [";".join(z) for z in feature_flags]
    x["scope_eligible"] = [len(z) == 0 for z in scope_flags]
    x["label_evidence_ready"] = [len(z) == 0 for z in label_flags]
    x["strict_label_candidate"] = x["scope_eligible"] & x["label_evidence_ready"]
    x["model_feature_ready"] = x["strict_label_candidate"] & x["feature_readiness_flags"].eq("")
    # Target-only variant is a diagnostic ablation, not the primary specification.
    x["target_only_feature_ready"] = x["strict_label_candidate"] & ~x["feature_readiness_flags"].str.contains(
        "target_revenue_missing|target_revenue_not_proven|target_revenue_availability|payment_method|missing_source_url:target_revenue|missing_source_url:payment_method",
        regex=True
    )
    x["blocking_flags"] = x.apply(
        lambda r: ";".join(filter(None, [r["scope_flags"], r["label_evidence_flags"], r["feature_readiness_flags"]])), axis=1
    )

    counts: dict[str, int] = {}
    for col in ["scope_flags", "label_evidence_flags", "feature_readiness_flags"]:
        for cell in x[col].fillna(""):
            for flag in filter(None, cell.split(";")):
                counts[flag] = counts.get(flag, 0) + 1
    ready = x.loc[x["model_feature_ready"]]
    target_only_ready = x.loc[x["target_only_feature_ready"]]
    categories = ready["payment_method_model"].value_counts().to_dict() if len(ready) else {}
    expected_payment_categories = {"cash", "stock", "mixed"}
    payment_support_warnings = []
    for category in sorted(expected_payment_categories):
        count = int(categories.get(category, 0))
        if count == 0:
            payment_support_warnings.append(f"Payment method '{category}' has no approved training observations; do not issue empirically supported predictions for this category.")
        elif count < 10:
            payment_support_warnings.append(f"Payment method '{category}' has only {count} approved observations; category effects are exploratory and unstable.")
    summary = {
        "status": "PREFLIGHT_COMPLETE_NO_FIT",
        "input_rows": int(len(x)),
        "scope_eligible_rows": int(x["scope_eligible"].sum()),
        "label_evidence_ready_rows": int(x["label_evidence_ready"].sum()),
        "strict_label_candidates_in_scope": int(x["strict_label_candidate"].sum()),
        "strict_label_candidate_deal_ids": x.loc[x["strict_label_candidate"], "deal_id"].astype(str).tolist(),
        "primary_full_feature_ready_rows": int(x["model_feature_ready"].sum()),
        "primary_full_feature_ready_deal_ids": x.loc[x["model_feature_ready"], "deal_id"].astype(str).tolist(),
        "target_only_feature_ready_diagnostic_rows": int(x["target_only_feature_ready"].sum()),
        "target_only_feature_ready_diagnostic_deal_ids": x.loc[x["target_only_feature_ready"], "deal_id"].astype(str).tolist(),
        "scope_excluded_deal_ids": x.loc[~x["scope_eligible"], "deal_id"].astype(str).tolist(),
        "blocked_rows": int((~x["model_feature_ready"]).sum()),
        "approved_announcement_window": "2005-01-01 through 2022-12-31 inclusive",
        "primary_target": "100 * announced annual pre-tax cost-synergy point claim / target revenue",
        "currency_policy": "All synergy and revenue values used in the model must be sourced/reconciled to USD millions; non-USD values are blocked until documented conversion is performed.",
        "target_unit": "percentage points of target revenue (e.g., 5 means 5% of target revenue)",
        "primary_predictors": ["log(acquirer revenue)", "log(target revenue)", "payment method known at announcement"],
        "excluded_from_predictors": ["acquisition premium", "completion/termination outcome", "post-announcement financial data", "synergy realization outcome", "listing/eligibility decision fields"],
        "payment_method_counts_in_primary_ready_rows": {str(k): int(v) for k, v in categories.items()},
        "payment_method_categories_not_observed": sorted(set(["cash", "stock", "mixed"]) - set(categories)),
        "payment_method_categories_with_fewer_than_10_rows": sorted([str(k) for k, v in categories.items() if int(v) < 10]),
        "payment_method_effect_warning": "Payment-method contrasts cannot be trusted until each intended category has substantial support; unseen categories cannot be treated as empirically validated.",
        "payment_method_support_warnings": payment_support_warnings,
        "payment_method_category_minimum_for_interpretation": 10,
        "payment_method_support_is_not_a_causal_effect_test": True,
        "target_recomputation_max_abs_difference_pp": float(x["target_abs_difference_pp"].max()) if len(x) and x["target_abs_difference_pp"].notna().any() else None,
        "blocker_counts": counts,
        "model_fit_performed": False,
        "warning": "A source-reviewed announced-claim label is not evidence of realized synergies. The model estimates announced claims conditional on disclosure, not realized savings or causal synergy effects.",
    }
    return x, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(args.input)
    audit, summary = prepare_dataset(src)
    if not audit.empty:
        audit.to_csv(args.output_dir / "model_dataset_row_audit.csv", index=False, date_format="%Y-%m-%d")
        audit.loc[audit["model_feature_ready"]].to_csv(args.output_dir / "model_features_primary.csv", index=False, date_format="%Y-%m-%d")
        audit.loc[audit["target_only_feature_ready"]].to_csv(args.output_dir / "model_features_target_only_diagnostic.csv", index=False, date_format="%Y-%m-%d")
    summary["input_path"] = _portable(Path(args.input))
    summary["outputs"] = {
        "row_audit_csv": _portable(args.output_dir / "model_dataset_row_audit.csv"),
        "primary_features_csv": _portable(args.output_dir / "model_features_primary.csv"),
        "target_only_diagnostic_csv": _portable(args.output_dir / "model_features_target_only_diagnostic.csv"),
        "summary_json": _portable(args.output_dir / "model_dataset_preflight.json"),
    }
    (args.output_dir / "model_dataset_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") != "SCHEMA_BLOCKED" else 2

if __name__ == "__main__":
    raise SystemExit(main())
