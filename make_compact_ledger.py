#!/usr/bin/env python3
"""Token-lean, LOSSLESS-BY-REFERENCE view of MA_Synergy_Ledger_MASTER.csv.

MASTER stays the single source of truth. The compact view keeps only decision-relevant fields in
normalised form (using the same parsers the model pipeline uses, so it can't drift) and drops long
free text/URLs. Anything dropped is retrievable per deal:  --show DEAL_ID  prints the full MASTER row.

    python scripts/make_compact_ledger.py                 # writes data/processed/MA_Synergy_Ledger_COMPACT.csv
    python scripts/make_compact_ledger.py --show SHELL_BG
Columns: deal_id,date,acq,tgt,usd_m,kind,pay,type,pretax,status,rule1,note
  kind: P point | R range | L lower-bound (usd_m shown as >=X) | E low==high no point | N none
  pay: cash/stock/mixed/? | type: cost/mixed/unclear/none | pretax: Y/N/? | note: 1 = MASTER has eligibility_flags/notes text
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_model_dataset import standardize_payment_method, synergy_type_is_cost_only, explicit_pretax, norm

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "MA_Synergy_Ledger_MASTER.csv"
OUT = ROOT / "data" / "processed" / "MA_Synergy_Ledger_COMPACT.csv"


def _fmt(x):
    return "" if pd.isna(x) else f"{x:g}"


def compact(m: pd.DataFrame) -> pd.DataFrame:
    pt, lo, hi = (pd.to_numeric(m[c], errors="coerce") for c in ["synergy_point_usd_m", "synergy_low_usd_m", "synergy_high_usd_m"])
    # Lower-bound wording beats a populated point column: MASTER stores e.g. MDT_COV 'at least $850M' in the point field.
    lower = m["status"].eq("LOWER_BOUND_ONLY") | m["qualifier"].fillna("").str.contains(r"at least|lower bound|in excess", case=False)
    kind = np.select([lower, pt.notna(), lo.notna() & hi.notna() & (lo != hi), lo.notna() & (lo == hi)],
                     ["L", "P", "R", "E"], "N")
    def _usd(p, l, h, k):
        if k == "N": return ""
        if k == "L": return ">=" + _fmt(l if pd.notna(l) else p)
        if k == "P": return _fmt(p)
        return f"{_fmt(l)}-{_fmt(h)}" if k == "R" else _fmt(l)
    usd = [_usd(p, l, h, k) for p, l, h, k in zip(pt, lo, hi, kind)]
    ptype = m["type"].map(lambda v: "none" if pd.isna(v) else ("unclear" if "unclear" in norm(v) or norm(v) == "unspecified"
                           else "cost" if synergy_type_is_cost_only(v) else "mixed"))
    pretax = [("Y" if norm(p).startswith("y") else "N" if norm(p).startswith("n") else "?") for p in m["pretax_explicit"]]
    note = (m["eligibility_flags"].notna() | m["notes"].notna()).astype(int)
    return pd.DataFrame({"deal_id": m["deal_id"], "date": m["announcement_date"].astype(str), "acq": m["acquirer"], "tgt": m["target"],
                         "usd_m": usd, "kind": kind, "pay": m["payment_method"].map(standardize_payment_method).fillna("?"),
                         "type": ptype, "pretax": pretax, "status": m["status"], "rule1": m["rule1_status"], "note": note})


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--show"); ap.add_argument("--master", default=str(MASTER))
    a = ap.parse_args(); m = pd.read_csv(a.master)
    if a.show:
        r = m.loc[m["deal_id"] == a.show]
        print(r.iloc[0].dropna().to_string() if len(r) else f"{a.show}: not found"); return 0 if len(r) else 1
    c = compact(m); c.to_csv(OUT, index=False)
    print(f"{len(c)} rows -> {OUT}  ({MASTER.stat().st_size} B master -> {OUT.stat().st_size} B compact)"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
