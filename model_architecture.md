# M&A Synergy Model Architecture v2

## 1. Prediction question
For a qualifying US-exchange-listed, non-financial public-to-public deal announced from 2005 through 2022, estimate the announced annual pre-tax cost-synergy point claim, conditional on a quantified point claim being publicly disclosed.

## 2. Outcome variables
Let `S` be the annual pre-tax cost-synergy point claim in USD millions and `R_t` be target revenue in USD millions from a filing available before the announcement.

- Primary normalized outcome: `Y_ratio = 100 × S / R_t`, measured in percentage points of target revenue.
- Secondary dollar outcome: `Y_dollar = S`, measured in USD millions.

Both outcomes are computed from the approved source values. The stored ratio is checked against the recomputed ratio (tolerance 0.02 percentage points). No ranges or lower bounds are turned into point estimates.

## 3. Feature vector
The primary feature set is `X = [ln(R_a), ln(R_t), payment_method]`, where `R_a` is acquirer revenue filed before announcement and payment method is one of cash/stock/mixed. A target-revenue-only ablation is evaluated to test whether the additional acquirer revenue feature materially improves generalization. Announcement year defines the validation order but is not a predictive feature.

Acquisition premium is excluded from X because premium is a separate outcome for the downstream underwriting analysis and is susceptible to leakage/circular interpretation. Completion/termination and realized synergy outcomes are also excluded.

## 4. Data admission gates
A row is admitted to the primary fit only if all of these hold:
1. Announcement date is in the project window and the deal passes public-to-public/non-financial and both-US-exchange-listed-at-announcement checks.
2. Source-backed exact point claim; cost-only, annual/annualized/run-rate, explicit pre-tax claim; no range/lower-bound semantics.
3. SEC filing/exhibit evidence; claim quote matches the numeric amount; tax-basis quote explicitly says pre-tax and independently contains the claim amount.
4. Positive revenue and synergy values in USD millions; revenue-source URLs are SEC filings; revenue statement publication dates precede the announcement date.
5. Payment method is known by announcement, supported by source quote and SEC filing/exhibit citation.
6. Revenue definition is explicitly reviewed/harmonized and transaction isn't an unresolved duplicate.

Failure flags are retained in a row audit; a row isn't silently repaired into a passing label.

## 5. Candidate regressors and selection
- Median baseline, trained only on the training partition.
- Ridge linear regression with small alpha grid `{1, 10, 100}`.
- Median Quantile Regression with a short alpha grid `{0.1, 1.0}`; its pinball objective targets the conditional median and is aligned with MAE selection.
- Huber robust regression with two loss thresholds `{1.35, 1.75}` and fixed conservative regularization.
- Raw and `log1p` outcome scales.
- Full and target-only predictor ablations.

All preprocessing is inside a sklearn pipeline so scale fitting and one-hot category learning happen on the training fold only. Predictions are constrained to non-negative values. The one-standard-error rule is a simplicity heuristic; it is not treated as an inferential confidence interval.

## 6. Time-aware validation
- Keep all observations from a given announcement year in a single temporal partition.
- Reserve the latest whole years for a future-period test; reserve the immediately preceding whole years for calibration; use only earlier years for rolling expanding-window model selection.
- Require at least 100 approved observations (governance floor), at least 8 distinct years, development n≥60, calibration n≥10, test n≥20, and at least 3 valid rolling folds with fold training n≥30 and validation n≥5.
- The calibration interval is suppressed below 20 calibration observations. Even above that it is labelled nominal split-conformal and carries a temporal-distribution-shift caveat.
- Do not report fit-quality or training-set metrics as out-of-sample accuracy.

## 7. Accuracy metrics
Report test-set MAE, median absolute error, RMSE, mean signed error (actual minus predicted), 90th-percentile absolute error, descriptive R², and skill relative to the training-median baseline. Compare ratio-derived dollar estimates to direct dollar estimates as a consistency diagnostic, not as a confidence interval. If a fitted model fails to beat the baseline in the untouched future test, that is not an accuracy improvement.

## 8. Prediction restrictions
The inference CLI checks model artifacts against the latest successful evaluation and identical input SHA-256. It refuses predictions for payment categories absent from the development data. It flags categories with fewer than 10 development examples and inputs outside the development feature ranges.

## 9. Interpretation limits
This is predictive modeling of management-announced cost-synergy claims among deals that disclose qualifying point estimates. It does not predict realized synergies, determine causality, establish deal-specific truth, or by itself state that an acquisition premium is justified. The premium analysis requires its own EV/equity-value reconciliation, phased after-tax cash flows, integration costs, discounting, and sensitivity analysis.

## 10. Current status
The live strict seed contains six source-reviewed candidates, of which five pass primary full-feature checks (Amcor–Bemis is excluded on US-listing scope). Payment method support is 3 mixed, 2 stock, 0 cash. Therefore no estimator is fitted, no cross-validated candidate ranking is meaningful, and no accuracy score is reported. Continue source collection and review before fitting.
