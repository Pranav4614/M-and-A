# Data expansion queue v2 — screening leads, not labels

The current strict seed does not have enough observations for any meaningful accuracy estimate. The queue therefore prioritizes additional source review using the existing 88-row discovery ledger. Every row in `model_data_expansion_queue.csv` remains `HOLD_NOT_TRAINING_ROW`; none is automatically imported into the strict training seed.

## P1 — source-backed point-claim candidates that may widen year / sector coverage

- **AstraZeneca–Alexion (2020-12-12; $500m discovery claim):** the ledger identifies an explicit pre-tax cost-synergy point claim in a SEC source, but financial inputs and eligibility/source checks are incomplete. Resolve US-exchange listing at announcement, announcement-time revenue filings, currency, payment-method quote and exact tax/claim wording before admission.
- **R.H. Donnelley–Dex Media (2005-10-03; ~$50m discovery claim):** older candidate that could widen temporal coverage if both parties' public/listed status and source-supported pre-announcement financials are confirmed. Verify payment terms, financial-statement publication dates and revenue line items.

## P2 — candidates with material unresolved issues

- **Alcatel–Lucent (2006-04-02; approximately $1.7bn / EUR 1.4bn in the discovery ledger):** eligibility and announcement-time financials are missing. Any EUR-to-USD conversion must be supported by an explicit date/rate policy and reported as a conversion, not as a source-native label.
- **Quad/Graphics–Worldcolor (2010-01-26; ~$225m):** public-to-public, control-acquisition and listing eligibility require adjudication, including target restructuring history, before any revenue collection.
- **Kraton–Arizona Chemical (2015-09-28; ~$65m):** likely private-target issue. Verify public/listed status first; exclude if target was private.
- **Parker Hannifin–Exotic Metals (2019-07-29; ~$13m):** likely private-target issue. Verify listed status first; exclude if target was private.

## Hold / probable scope exclusions

- **Vantiv–Worldpay (2017-08-09; ~$200m):** payment-processing/financial-services scope review required before proceeding.
- **Aon–Willis Towers Watson (2020-03-09; $800m):** insurance brokerage / professional financial-services scope review required; hold out of the non-financial cohort unless the approved scope rule says otherwise.
- **U.S. Bancorp–MUFG Union Bank (2021-09-21; $900m):** financial-sector candidate; the current ledger already marks it `EXCLUDE_SECTOR`.

## Admission checklist for every queued deal

1. Exact announcement date and announcement claim source are verified.
2. A qualifying annual cost-only exact point is present; ranges and lower bounds stay out of the point-label model.
3. The SEC filing/exhibit tax-basis quote explicitly says pre-tax and contains the same claim amount.
4. Both companies meet public-to-public, non-financial, and US-exchange-listed-at-announcement scope rules.
5. Acquirer and target revenue are taken from SEC filings published strictly before announcement; currency and top-line definition are documented.
6. Payment method is supported by a contemporaneous SEC source quote; exact category is derived from that evidence, not from a guessed label.
7. Duplicate events, post-announcement data, and outcome leakage are ruled out.

The 100-row governance floor must not be used as pressure to admit unresolved rows. A clean smaller dataset is preferable to a larger contaminated target table.
