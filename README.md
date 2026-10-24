# M&A Cost-Synergy Benchmark & Credibility-Gated Model

Estimate the annual **pre-tax cost synergies** management announced for US-exchange-listed, non-financial, public-to-public
acquisitions, as a share of **target revenue**, with a pipeline that refuses to report accuracy it cannot support.

> **Status: small-data benchmark, not a validated forecaster.** Only **5** source-verified deals are admitted (years 2012–2021).
> The regression pipeline is gated at 100 rows and is correctly blocked. What the data supports today is a descriptive
> benchmark: announced cost synergies are a **median 8.9% of target revenue (range 4.8–11.7%, 80% predictive interval 3.8–13.4%)**.
> See [Results](#results) and [Limitations](#limitations).

## What is in this repo

| Path | Purpose |
|---|---|
| `data/processed/MA_Synergy_Ledger_MASTER.csv` | 88-deal source ledger (announced synergy claims, status, source URLs) |
| `data/processed/MA_Synergy_Ledger_Curated.csv` | Curated version with eligibility, tax-basis and readiness flags |
| `data/processed/strict_verified_seed_observations.csv` | Source-reviewed seed rows with revenue and evidence quotes |
| `data/processed/model_features_primary.csv` | The admitted rows (the only rows any estimate uses) |
| `scripts/build_model_dataset.py` | Applies scope, label-evidence and feature gates; writes audit tables |
| `scripts/benchmark_estimator.py` | Descriptive benchmark with leave-one-out check (works at n=5) |
| `scripts/train_synergy_model.py` | Credibility-gated regression pipeline (blocks before fitting until data suffices) |
| `scripts/predict_synergy.py` | Prediction CLI; refuses unless a gate-passing model exists |
| `scripts/validate_machinery.py` | Synthetic-data test of false-positive rate and power of the training machinery |
| `scripts/make_compact_ledger.py` | Compact, token-lean view of the MASTER ledger (`--show DEAL_ID` for a full row) |
| `docs/` | Design ([model_architecture.md](docs/model_architecture.md)), validation log ([validation.md](docs/validation.md)), data queue notes |
| `reports/` | Latest generated outputs (benchmark, gate record, audit, machinery validation) |
| `tests/` | 41 tests (data gates, regression tests for past defects, machinery validation) |

## Quick start

Python 3.10+.

```bash
pip install -r requirements.txt
pytest -q                                   # 41 tests
python scripts/build_model_dataset.py       # rebuild audit tables and feature sets
python scripts/benchmark_estimator.py       # descriptive benchmark -> reports/benchmark_result.json
python scripts/benchmark_estimator.py --target-revenue-usd-m 3000   # screening range for a target
python scripts/train_synergy_model.py       # exits 2 while gated: expected, nothing is fitted
```

A nonzero exit from `train_synergy_model.py` is expected while the gate is blocked. It means no model was fitted and no
performance metrics were invented. If you use a different input CSV, source-review it first and pass
`--attest-real-source-data`; never set that flag for synthetic fixtures.

## Results

Admitted deals: Eaton–Cooper (cost component), Lam–KLA, Ensco–Rowan, Chevron–Noble, STERIS–Cantel.

| Quantity | Value |
|---|---|
| Announced cost synergies, % of target revenue | median **8.88%**, range 4.81–11.69% |
| 90% bootstrap CI for the median | 4.81–11.69% |
| 80% predictive interval (t, n=5) | 3.82–13.37% |
| Leave-one-out MAE (median rule) | 3.0 percentage points |
| Leave-one-out median dollar error | 33% |
| Scaling by target revenue vs ignoring size | wins 3 of 5 deals; **inconclusive** |

These support one claim only: across 5 verified deals, announced cost synergies cluster around 5–12% of target revenue.
They do **not** show that payment method, size or any other feature predicts synergies.

**Reliability of the modeling machinery** (synthetic data, `reports/machinery_validation.json`): on pure noise the pipeline
claimed skill in at most 6% of runs; with a moderate true signal (R² ≈ 16%) it detected it about half the time at 200 rows and
62% at 400 rows. It is conservative: it rarely invents skill, and "median baseline wins" is a legitimate outcome on real data.
This says nothing about real M&A accuracy.

## Target and predictors

Primary target: `100 * annual_pre_tax_cost_synergies_usd_m / target_revenue_usd_m` (percentage points). Secondary: dollar
synergies. Only exact point claims are labels; ranges and lower bounds are never converted to points.
Predictors (when a model is eventually fitted): `log(acquirer revenue)`, `log(target revenue)`, payment method
(`cash`/`stock`/`mixed`, verified by a contemporaneous quote). Premium, outcomes, post-announcement data and eligibility
fields are excluded; announcement year is used only for chronological grouping.

## Evidence rules (what "admitted" means)

- Both parties US-exchange-listed at announcement; non-financial; public-to-public.
- Synergy claim and pre-tax statement each quote the amount; the tax quote explicitly says pre-tax; SEC filing/exhibit sources.
- Both revenue figures were filed before the announcement, in USD millions, with comparable revenue definitions (documented).
- Payment method supported by a contemporaneous source quote.

## Validation policy

Baseline: median predictor-free model. Candidates: Ridge, Huber, median quantile regression (raw and `log1p`, full vs
target-only). Selection by expanding-window whole-year validation with a one-standard-error simplicity rule; separate
calibration years and an untouched future test; accuracy is claimed only if the future test beats the baseline. Gate:
≥100 rows, ≥60 development / ≥10 calibration rows, whole-year folds. In simulation the practical requirement is about
**150 rows over ≥12 years** (see [docs/validation.md](docs/validation.md)). Extreme ratios are reviewed, never auto-deleted.

## Limitations

- **n = 5.** Intervals are wide by necessity; the benchmark is a screening aid, not a forecast.
- **Management-announced claims only**, conditional on disclosure. This is not a realized-savings or causal model, and disclosure is non-random.
- Revenue definitions differ slightly between companies (e.g. Chevron's total revenues include equity income); documented per row.
- The cohort is US-listed non-financial deals; it does not generalize to financials, cross-listed or private-target deals.
- Not investment advice.

## Reproducibility

Each training attempt records a UTC timestamp, library versions, and SHA-256 of the input file and script. Source URLs for every
claim and revenue figure are in the data files. Generated files in `data/processed/` and `reports/` are reproducible from the scripts.

## Contributing data

Each new admitted deal needs: filing-sourced revenue for both parties (filed before announcement), an SEC-sourced pre-tax
statement, a listing check, and a payment-method quote. Add it to `strict_verified_seed_observations.csv`, run
`build_model_dataset.py` and `pytest -q`, and check the row's `blocking_flags` in `model_dataset_row_audit.csv`.

## License

Code: MIT (see [LICENSE](LICENSE)). Data compiled from public company filings and announcements; verify against the cited sources before reuse.
