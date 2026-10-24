# Model accuracy and credibility audit v2

**Status: BLOCKED BEFORE FIT.**

Rows in input seed: **6**
Rows passing strict full-feature model checks: **5**
Rows with label evidence and cohort eligibility but feature caveats: **0**

## Hard blockers
- Only 5 US-listed, strict-label, full-feature-ready rows; governance floor is 100.
- Year-grouped development set has 0 rows; minimum is 60.
- Year-grouped calibration set has 0 rows; minimum is 10.
- Year-grouped final future test has 0 rows; minimum is 20.
- Only 0 valid whole-year expanding-window folds; minimum is 3.
- Fewer than 8 distinct announcement years in model-ready rows; future-period validation is too fragile.

## Cohort and feature decisions
- `both_parties_not_verified_us_exchange_listed_at_announcement`: 1
- `tax_basis_evidence_not_from_sec_filing`: 1
- `synergy_claim_source_not_sec_filing_or_exhibit`: 1
- `revenue_source_not_sec_filing:acquirer_revenue_source_url`: 1
- `payment_method_source_quote_does_not_support_category`: 1
- `payment_method_source_not_sec_filing_or_exhibit`: 1

## Model improvements implemented
- Explicit US-exchange listing-at-announcement gate; US listing after completion does not qualify a historical announcement.
- Source evidence separated into cohort, target-label, and input-feature checks.
- Revenue source links must resolve to SEC filings; revenue publication dates must be strictly before announcement day; same-day order is treated as unverified.
- Synergy and both revenue inputs must be explicitly normalized to USD millions; non-USD rows require documented currency conversion before admission.
- The numeric synergy amount must appear in the annual-cost source quote and independently in the quote supporting the pre-tax basis.
- Duplicate deal events are blocked pending adjudication to avoid pseudo-replication.
- Model candidate set is deliberately restrained: median baseline, Ridge, median Quantile Regression (MAE-aligned), and Huber robust regression across raw/log1p targets and a target-only ablation.
- Expanding-window cross-validation groups by announcement year; no random shuffle and no same-year train/test split.
- One-standard-error selection prefers the simplest candidate when performance differences are not clear.
- Last whole announcement years are reserved for calibration and untouched future testing.
- 90% intervals are suppressed unless calibration n is at least 20.
- Later completion/termination outcomes and acquisition premium cannot enter predictors.
- Payment-method category support is explicitly reported; unobserved categories cannot be presented as validated predictions and categories with fewer than 10 observations are low-support.
- The strict label validator checks that the pre-tax evidence quote itself contains the synergy claim amount, not merely the words 'pre-tax'.
- Reproducibility manifest records Python/library versions, input SHA-256, and training-script SHA-256.
- The prediction CLI refuses output unless the latest evaluation is successful and both artifacts match the evaluation input hash, preventing stale-model use.

## Interpretation rule
The target measures management-announced annual pre-tax cost-synergy point claims conditional on firms disclosing quantified claims. It does not measure realized savings or prove the claim is economically credible.

No estimator was fitted; therefore no accuracy numbers are reported. The pipeline and gate have been tested with unit tests and a synthetic mechanics-only fixture; that fixture is not used as model data.