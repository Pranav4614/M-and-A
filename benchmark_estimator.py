#!/usr/bin/env python3
"""Descriptive benchmark for announced cost synergies as % of TARGET revenue (small-n, NOT the gated model).

Valid at the data size we have (n = admitted rows in model_features_primary.csv). It makes no claim of predictive
skill beyond what the leave-one-out check below shows, and it never conditions on payment method or size
(n is far too small to support that).

    python scripts/benchmark_estimator.py                              # write reports/benchmark_result.json
    python scripts/benchmark_estimator.py --target-revenue-usd-m 3000  # screening range in $M for a target
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_features_primary.csv"
OUT = ROOT / "reports" / "benchmark_result.json"
LEVEL = 0.80


def t_interval(x, level=LEVEL):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return (float("nan"), float("nan"))
    half = stats.t.ppf((1 + level) / 2, n - 1) * x.std(ddof=1) * np.sqrt(1 + 1 / n)
    return (max(0.0, x.mean() - half), x.mean() + half)


def bootstrap_median_ci(x, level=0.90, reps=20000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.median(r.choice(x, size=(reps, len(x)), replace=True), axis=1)
    return tuple(np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2]))


def loo(df):
    """Predict each deal's pct from the OTHER deals; compare estimators and interval coverage."""
    pct = df["target_recomputed_pct"].to_numpy(float); rev = df["target_revenue_usd_m"].to_numpy(float)
    syn = df["announced_annual_cost_synergy_usd_m"].to_numpy(float); rows = []
    for i in range(len(pct)):
        o = np.delete(pct, i); lo, hi = t_interval(o)
        rows.append(dict(deal_id=df["deal_id"].iloc[i], actual_pct=pct[i], median_pred=float(np.median(o)), mean_pred=float(o.mean()),
                         lo=lo, hi=hi, covered=bool(lo <= pct[i] <= hi),
                         usd_ape_median=abs(np.median(o) / 100 * rev[i] - syn[i]) / syn[i],
                         usd_ape_const_dollar=abs(np.median(np.delete(syn, i)) - syn[i]) / syn[i]))  # ignores target size
    r = pd.DataFrame(rows); base = np.abs(pct - np.median(pct))  # in-sample reference only
    return r, dict(
        loo_mae_pp_median_rule=float(np.abs(r.actual_pct - r.median_pred).mean()),
        loo_mae_pp_mean_rule=float(np.abs(r.actual_pct - r.mean_pred).mean()),
        loo_median_abs_pct_error_usd=float(r.usd_ape_median.median()),
        loo_median_abs_pct_error_usd_ignoring_size=float(r.usd_ape_const_dollar.median()),
        scaling_wins=int((r.usd_ape_median < r.usd_ape_const_dollar).sum()), scaling_mean_ape=float(r.usd_ape_median.mean()),
        ignoring_size_mean_ape=float(r.usd_ape_const_dollar.mean()),
        scaling_verdict=("favours scaling" if (r.usd_ape_median < r.usd_ape_const_dollar).sum() >= int(np.ceil(0.8 * len(r))) else "inconclusive"),
        loo_interval_coverage=float(r.covered.mean()), coverage_note="with n=5 coverage moves in steps of 20%; treat as uninformative", nominal_coverage=LEVEL,
        spread_reference_mad_pp=float(np.median(np.abs(pct - np.median(pct)))))


def summarize(df):
    pct = df["target_recomputed_pct"].to_numpy(float); lr, lm = loo(df)
    return dict(
        note="Descriptive small-n benchmark. NOT a validated forecasting model; do not condition on payment method or size.",
        n=int(len(pct)), years=sorted(int(y) for y in df["announcement_year"].unique()), deal_ids=df["deal_id"].tolist(),
        pct_of_target_revenue=dict(min=float(pct.min()), median=float(np.median(pct)), mean=float(pct.mean()), max=float(pct.max()),
                                   sd=float(pct.std(ddof=1)), median_bootstrap_90ci=[float(v) for v in bootstrap_median_ci(pct)],
                                   predictive_interval_80=[float(v) for v in t_interval(pct)]),
        leave_one_out=lm, leave_one_out_rows=lr.round(3).to_dict("records"))


def screening_range(df, target_revenue_usd_m):
    pct = df["target_recomputed_pct"].to_numpy(float); lo, hi = t_interval(pct); R = float(target_revenue_usd_m)
    return dict(target_revenue_usd_m=R, median_synergy_usd_m=round(np.median(pct) / 100 * R, 1),
                observed_range_usd_m=[round(pct.min() / 100 * R, 1), round(pct.max() / 100 * R, 1)],
                predictive_interval_80_usd_m=[round(lo / 100 * R, 1), round(hi / 100 * R, 1)], n=len(pct),
                warning=f"Based on only {len(pct)} deals; intervals are wide by necessity. Screening aid only, not a forecast.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target-revenue-usd-m", type=float); a = ap.parse_args()
    df = pd.read_csv(DATA); df = df[df["model_feature_ready"] == True] if "model_feature_ready" in df else df
    if len(df) < 3: print("Too few admitted rows for a benchmark."); return 2
    if a.target_revenue_usd_m is not None:
        if a.target_revenue_usd_m <= 0: print("target revenue must be > 0"); return 2
        print(json.dumps(screening_range(df, a.target_revenue_usd_m), indent=2)); return 0
    res = summarize(df); OUT.write_text(json.dumps(res, indent=2)); p = res["pct_of_target_revenue"]; l = res["leave_one_out"]
    print(f"n={res['n']} years={res['years']}\nsynergy % of target revenue: median {p['median']:.2f} (90% bootstrap CI {p['median_bootstrap_90ci'][0]:.2f}-{p['median_bootstrap_90ci'][1]:.2f}), "
          f"range {p['min']:.2f}-{p['max']:.2f}, 80% predictive {p['predictive_interval_80'][0]:.2f}-{p['predictive_interval_80'][1]:.2f}")
    print(f"LOO MAE: median-rule {l['loo_mae_pp_median_rule']:.2f}pp, mean-rule {l['loo_mae_pp_mean_rule']:.2f}pp | LOO median $ error {l['loo_median_abs_pct_error_usd']:.0%}"
          f"\nscaling by target revenue vs ignoring size: wins {l['scaling_wins']}/{res['n']}, mean $ error {l['scaling_mean_ape']:.0%} vs {l['ignoring_size_mean_ape']:.0%} -> {l['scaling_verdict']}\n80% interval LOO coverage {l['loo_interval_coverage']:.0%} (nominal 80%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
