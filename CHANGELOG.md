# Changelog

## v3
- Independent validation; reproduced v2 outputs exactly.
- Fixed data-gate defects: "per share" cash prices mislabeled as stock/mixed (cash observations in ledger 7 -> 16); payment-quote validator
  rejecting standard SEC phrasing; substring cost-only filter admitting non-cost-only types; bare numbers matching as money amounts;
  "to verify" payment labels receiving categories; partial dates silently coerced.
- Admitted Chevron–Noble after documented revenue-definition alignment (4 -> 5 admitted rows). Explicit `revenue_definition_harmonized = YES`
  now requires a documented review note.
- Added `benchmark_estimator.py` (descriptive benchmark + leave-one-out), `validate_machinery.py` (false-positive rate and power on synthetic data),
  `make_compact_ledger.py` (token-lean ledger view).
- Tests 30 -> 41, including regression tests that fail on the v2 code.
- Repo cleanup: removed superseded v2 reports, a stale Excel snapshot and byte-identical duplicate outputs; dropped version suffixes from file names; curated tables now match the pipeline (Amcor–Bemis excluded, Chevron–Noble admitted); portable paths in generated files; CI; license.

## v2
Credibility-gated pipeline; 6 source-reviewed seed candidates, 4 admitted; no model fitted.
