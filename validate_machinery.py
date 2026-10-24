#!/usr/bin/env python3
"""Operating characteristics of the training/selection machinery on SYNTHETIC data (mechanics only).

Answers: how often does the pipeline claim skill when there is none (false positive), and how often does it
find skill when a real relationship exists (power), at different signal strengths and sample sizes?
Says NOTHING about real M&A accuracy. Output: reports/machinery_validation.json
    python scripts/validate_machinery.py [--reps 24] [--workers 4]
"""
import argparse, json, sys, tempfile, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [("noise", 100, 12, 0.0), ("noise", 200, 14, 0.0), ("weak", 200, 14, 0.15), ("moderate", 200, 14, 0.4),
             ("strong", 200, 14, 0.9), ("moderate", 100, 12, 0.4), ("moderate", 400, 14, 0.4)]


def make(n, years, beta, seed, noise_sd=0.9):
    r = np.random.default_rng(seed); yrs = np.arange(2005, 2005 + years); y = np.concatenate([yrs, r.choice(yrs, n - years)])
    ta = np.exp(r.normal(6.5, 1.0, n)); aq = np.exp(r.normal(8.5, 1.2, n)); pm = r.choice(["cash", "stock", "mixed"], n)
    sig = -beta * np.log(ta); eps = r.normal(0, noise_sd, n)
    pct = np.clip(3.0 + sig - sig.mean() + eps, 0.3, None)
    r2 = float(np.var(sig) / (np.var(sig) + noise_sd ** 2))
    return pd.DataFrame({"deal_id": [f"S{i}" for i in range(n)], "announcement_date": pd.to_datetime([f"{v}-06-15" for v in y]),
        "announcement_year": y, "acquirer": "A", "target": "T", "target_revenue_usd_m": ta, "acquirer_revenue_usd_m": aq,
        "feature_log_target_revenue": np.log(ta), "feature_log_acquirer_revenue": np.log(aq), "payment_method_model": pm,
        "target_recomputed_pct": pct, "announced_annual_cost_synergy_usd_m": pct / 100 * ta}), r2


def one(args):
    name, n, years, beta, seed = args
    warnings.filterwarnings("ignore"); import train_synergy_model as t
    df, r2 = make(n, years, beta, seed); g = t.gate_status(df, len(df))
    if not g["gate_pass"]: return dict(name=name, n=n, beta=beta, gate=False, r2=r2)
    with tempfile.TemporaryDirectory() as d:
        o = t.train_model(df, Path(d) / "o", Path(d) / "m", g)["primary_ratio_model"]
    m = o["test_metrics"]
    return dict(name=name, n=n, beta=beta, gate=True, r2=r2, beats=bool(m["improves_on_median_baseline"]), skill=float(m["mae_skill_vs_median_baseline"]),
                selected=o["selected_candidate"], mae=float(m["mae_percentage_points"]), base_mae=float(m["median_baseline_mae_percentage_points"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=24); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    jobs = [(nm, n, y, b, 1000 * i + s) for i, (nm, n, y, b) in enumerate(SCENARIOS) for s in range(a.reps)]
    with ProcessPoolExecutor(a.workers) as ex: res = list(ex.map(one, jobs))
    df = pd.DataFrame(res); out = []
    for (nm, n, b), g in df.groupby(["name", "n", "beta"], sort=False):
        ok = g[g.gate]; row = dict(scenario=nm, n_rows=n, true_signal_r2=round(float(g.r2.mean()), 3), runs=len(g), gate_pass_rate=round(float(g.gate.mean()), 2))
        if len(ok):
            row.update(beats_baseline_rate=round(float(ok.beats.mean()), 2), median_skill=round(float(ok.skill.median()), 3),
                       median_test_mae_pp=round(float(ok.mae.median()), 3), median_baseline_mae_pp=round(float(ok.base_mae.median()), 3),
                       baseline_selected_rate=round(float((ok.selected == "median_baseline").mean()), 2))
        out.append(row)
    p = ROOT / "reports" / "machinery_validation.json"; p.write_text(json.dumps({"note": "SYNTHETIC mechanics only; not evidence about real M&A synergy accuracy.", "results": out}, indent=2))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
