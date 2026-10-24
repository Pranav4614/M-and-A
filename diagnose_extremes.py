#!/usr/bin/env python3
"""Create a cautious descriptive diagnostics report for extreme synergy ratios.

The script never deletes observations. With fewer than 10 label-usable rows it reports
summary statistics only and does not classify any deal as an outlier.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from build_model_dataset import DEFAULT_INPUT, ROOT, prepare_dataset

MIN_N_FOR_FLAGS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=ROOT/"reports")
    ap.add_argument("--processed-dir", type=Path, default=ROOT/"data"/"processed")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(args.input)
    audit, prep = prepare_dataset(src)
    if audit.empty:
        summary = {"status":"SCHEMA_BLOCKED", "n":0, "outlier_flags_created":False}
        (args.output_dir/"extreme_deal_diagnostics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2)); return 2
    rows = audit.loc[audit["strict_label_candidate"]].copy()
    y = rows["target_recomputed_pct"].dropna()
    n = int(y.size)
    q1 = float(y.quantile(.25)) if n else None
    med = float(y.median()) if n else None
    q3 = float(y.quantile(.75)) if n else None
    iqr = q3-q1 if n and q1 is not None and q3 is not None else None
    mad = float(np.median(np.abs(y.to_numpy()-med))) if n else None
    rows["iqr_outlier_candidate"] = False
    rows["robust_z_mad"] = np.nan
    rows["robust_z_outlier_candidate"] = False
    if n >= MIN_N_FOR_FLAGS and iqr is not None:
        rows["iqr_outlier_candidate"] = (rows["target_recomputed_pct"] < q1-1.5*iqr) | (rows["target_recomputed_pct"] > q3+1.5*iqr)
        if mad and mad > 0:
            rows["robust_z_mad"] = 0.6745*(rows["target_recomputed_pct"]-med)/mad
            rows["robust_z_outlier_candidate"] = rows["robust_z_mad"].abs() > 3.5
        flag_status = "DESCRIPTIVE_AND_CANDIDATE_FLAGS"
    else:
        flag_status = "INSUFFICIENT_N_FOR_OUTLIER_CLASSIFICATION"
    rows["outlier_decision"] = np.where(rows["iqr_outlier_candidate"] | rows["robust_z_outlier_candidate"],
                                        "REVIEW_ONLY_NOT_AUTO_EXCLUDED", "NO_AUTOMATIC_FLAG")
    output_cols = [c for c in ["deal_id","acquirer","target","announcement_date","target_recomputed_pct","payment_method_model","data_quality_notes","iqr_outlier_candidate","robust_z_mad","robust_z_outlier_candidate","outlier_decision"] if c in rows.columns]
    rows[output_cols].to_csv(args.processed_dir/"extreme_deal_review.csv",index=False,date_format="%Y-%m-%d")
    summary = {"status":flag_status,"n_label_usable_rows":n,"minimum_n_for_outlier_classification":MIN_N_FOR_FLAGS,
               "target_unit":"percentage points of target revenue","min":float(y.min()) if n else None,
               "q1":q1,"median":med,"q3":q3,"max":float(y.max()) if n else None,"iqr":iqr,"mad":mad,
               "iqr_outlier_candidates":int(rows["iqr_outlier_candidate"].sum()),
               "robust_z_outlier_candidates":int(rows["robust_z_outlier_candidate"].sum()),
               "policy":"Extreme observations are review candidates only and are never removed automatically.",
               "feature_note":"A row with a usable synergy label but unresolved revenue feature definitions may appear in label distribution diagnostics, while remaining blocked from primary feature training.",
               "preflight_blockers":prep.get("blocker_counts",{})}
    (args.output_dir/"extreme_deal_diagnostics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
