# Independent validation & v3 changes

**Verdict:** the project's central claim holds. Fitting is correctly blocked (4 usable rows vs 100 floor), no accuracy
numbers are invented, and the training machinery is sound. But the data-admission rules had 5 defects that would have
distorted the cohort as it grows. All are fixed in v3 with regression tests; your real-data outcome is unchanged (6 candidates / 4 ready).

## Verified (reproduced, not assumed)
- 30/30 original tests pass; full pipeline reproduces shipped outputs exactly (differences: paths, timestamps, library versions). Runs on pandas 3.0.2 and 2.2.3.
- Training machinery on synthetic data (temp dir, discarded): pure noise -> median baseline selected 8/8, 0.0 skill; planted signal -> positive out-of-time skill 8/8. No holdout leakage into selection.
- Prediction CLI correctly refuses (MODEL_NOT_AVAILABLE).

## Defects fixed (severity order)
| # | Defect | Evidence on your MASTER ledger | Fix |
|---|--------|-------------------------------|-----|
| 1 | `"per share"` counted as stock -> cash deals coded "mixed" | 9 of 16 all-cash deals mislabeled; cash obs **7 -> 16**, mixed 41 -> 31 | strip per-share price phrases |
| 2 | Quote validator rejected standard SEC phrasing | "$X in cash per share" fails cash; "cash in lieu of fractional shares" fails stock | strict acquirer-share evidence, ignore in-lieu cash |
| 3 | Cost-only filter = substring "cost" | 10 non-cost-only types passed, incl. "earnings synergies (not cost-only)", "cost synergies and tax savings" | affirmative classifier; 70 -> 62 pass |
| 4 | Amount matcher accepted bare numbers | "employs 375 people" ties to a $375M label; "Q3 2012" to 2012 | require $/unit |
| 5 | "to verify" payment labels got categories | 2 rows | return None |
| 6 | Partial dates ("2006-01") silently coerced to day 1 | 8 rows; weakens the filed-before-announcement leakage test | scope flag `announcement_date_not_day_precision` |

## Findings NOT changed (your decision)
1. **Gate minimum is optimistic.** "100 rows / 8 years" passes ~0-3% of the time at 8 years; ~97% needs >=150 rows over >=12 years (because 6 dev years must remain after test+calibration years). Plan data collection to that, not 100.
2. **Data supply.** MASTER funnel: 88 -> 51 point (non-lower-bound) -> 38 cost-only -> 21 pre-tax explicit -> 15 in 2005-2022 -> 13 payment known; 7 years, max 3/yr. This is an upper bound before SEC-source and revenue checks. Largest leaks: pre-tax basis not explicit (17 rows) and the 2022 window cap (6 rows). Revisit `END_DATE` if 2023+ deals are acceptable.
3. **MASTER data issue:** `MDT_COV` stores an "at least $850M" lower bound in `synergy_point_usd_m` (`AVAGO_BROADCOM` is ambiguous). The seed guard protects the model, but any filter on that column alone would admit a lower bound as a point label.
4. **Ratio-target coupling.** target = synergy / target_revenue while log(target_revenue) is a predictor, which mechanically induces correlation. Once n allows, also test `log(synergy) ~ log(target_rev) + log(acq_rev)` (slope on log target revenue = 1 means pure proportionality).
5. Duplicate detection only catches exact name matches ("Chiron Corp." vs "Chiron Corporation" are not flagged).

## Token efficiency (lossless)
- `scripts/make_compact_ledger.py`: MASTER (~10.3k tok) -> COMPACT (~2.7k tok), **74% smaller per session**; uses the same parsers as the model so it cannot drift; `--show DEAL_ID` pulls the full MASTER row on demand. MASTER remains the source of truth.
- Removed 3 byte-identical report duplicates + the superseded `README_base.md` (~2.3k tok); nothing referenced them.
- Working rule: read COMPACT first, `--show` only the deals under review; read `model_dataset_preflight.json` instead of re-running scripts.
(Token figures are chars/4 estimates.)

## Machinery operating characteristics (synthetic; `scripts/validate_machinery.py`, 16 runs/scenario)
Mechanics only. This says how trustworthy the *method* is, not how accurate it would be on real deals.
| True signal (R² of revenue->synergy%) | Rows | Gate passes | Claims skill over baseline | Median skill |
|---|---|---|---|---|
| none | 100 | 56% | 0 of 9 passing runs | 0.0 |
| none | 200 | 100% | 1 of 16 (6% false positive) | 0.0 |
| weak (~3%) | 200 | 100% | 0% | 0.0 |
| moderate (~16%) | 100 | 44% | 0% | 0.0 |
| moderate (~16%) | 200 | 100% | 50% | 1.2% |
| moderate (~16%) | 400 | 100% | 62% | 7.0% |
| strong (~50%) | 200 | 100% | 88% | 26.4% |

Reading: the method is conservative. It rarely invents skill (<=6%), but if the real revenue-to-synergy link is only
moderate, even 200 rows detects it about half the time. A "median baseline selected" outcome on real data is therefore a
legitimate result, not a failure. At 100 rows the gate itself often cannot be satisfied.

## Completion pass (within the existing 88-deal ledger)
**Data ceiling:** only 88 deals exist and the project's own queue offers ~6 more possible admissions after eligibility, so the
100-row regression gate can never be met from this ledger. The project is therefore completed as a *benchmark*, with the
gated model kept for if more data ever arrives.

**Changes**
- `CHEVRON_NOBLE` admitted (4 -> **5 ready rows**, years 2012-2021). Acquirer revenue aligned to Chevron FY2019 Total Revenues and Other Income
  $146,516M (10-K filed 2020-02-21; verified); Noble FY2019 Total revenues $4,438M (10-K filed 2020-02-12; verified). Label (target revenue only) is unchanged at 6.76%.
- Logic fix: an explicit `revenue_definition_harmonized = YES` now counts only with a documented review note; stale keyword hits in free text no longer override it (and no longer re-trigger on the word "harmonized").
- `AMCOR_BEMIS` stays excluded: it fails the US-exchange-listing scope at announcement (the project's own flag; Amcor was ASX-listed then, not re-verified here), in addition to non-SEC sources. It could form a separate non-US-listed-acquirer cohort.
- Correction to earlier text: the seed has 6 rows, 5 in scope, 4 were ready (not "6 candidates").
- New `scripts/benchmark_estimator.py` (descriptive, small-n, never conditions on payment method or size) with leave-one-out validation.

**Benchmark result (n=5: Eaton-Cooper cost component, Lam-KLA, Ensco-Rowan, Chevron-Noble, STERIS-Cantel)**
| Quantity | Value |
|---|---|
| Announced cost synergies, % of target revenue | median **8.88%**, range 4.81-11.69% |
| 90% bootstrap CI for the median | 4.81-11.69% (as wide as the data) |
| 80% predictive interval (t, n=5) | 3.82-13.37% |
| Leave-one-out MAE (median rule) | 3.0 percentage points |
| Leave-one-out median $ error | 33% |
| Scaling by target revenue vs ignoring size | wins 3 of 5; mean $ error 43% vs 55%; **inconclusive** |
| Leave-one-out interval coverage | 80% (but moves in 20% steps at n=5; uninformative) |
Example: a $3,000M-revenue target gives a median $266M, observed range $144-351M, 80% interval $115-401M.

**What this does and does not show.** It supports one claim: across 5 verified public non-financial deals, announced cost
synergies cluster around 5-12% of target revenue. It does not show that payment method, size, or any feature predicts synergies,
and it is not a validated forecaster. The prediction CLI remains disabled by design.

**Still open (needs outside data, not more work on this ledger):** each additional admitted row needs filing-sourced revenue
for both parties, an SEC-sourced pre-tax statement, and a listing check. Highest-value remaining: RHD-Dex Media, AstraZeneca-Alexion.
