# Take-Home Problem Statements — Detailed Briefs

Each is written the way an actual take-home packet would read: background, data provided, task, required deliverables, and evaluation criteria.

---
---

[↩️ Back to README.md](./README.md)

---
---

## Take-Home 1: Securities-Lending Fee & Rebate-Rate Forecasting

**Background:** The Equities & Delta One financing desk prices new securities loans off a forward-looking specialness fee. Mispricing either loses the loan to a competitor (fee too high) or gives away spread (fee too low).

**Data provided:** `sec_lending_panel.csv` — daily panel, ~2,000 tickers × 2 years: utilization %, lendable supply, days-to-cover proxy, dividend/index-event calendar flags, GC rate, historic fee (bps).

**Task:** Build a model that forecasts fee at t+1, t+3, and t+5 business days.

**Required outputs:**
1. A trained model (or ensemble) with a documented feature list and rationale for each feature's inclusion.
2. Out-of-sample backtest results using a **walk-forward** split (not k-fold) — report weighted MAPE (notional-weighted), directional hit-rate on moves >25bps, and pinball loss at P10/P90.
3. A written explanation (≤1 page) of *why* you chose your model class(es), including at least one rejected alternative and why.
4. A short section on production deployment: retraining cadence, monitoring metric(s), and a fallback plan if the model degrades.
5. A named output table: predicted fee, 80% interval, and a flag for "≥90% utilization regime" per name per date.

---

## Take-Home 2: Client Margin & Haircut Optimization

**Background:** Prime margin schedules are currently a static rules-based grid (haircut by asset class/rating/liquidity bucket). This over-margins low-risk collateral and under-margins in correlation-breakdown stress.

**Data provided:** `collateral_universe.csv` (per-asset: volatility, ADV, market correlation, rating, asset class) and `client_portfolios.csv` (sample portfolios with positions) plus historical stress scenario returns.

**Task:** Produce a data-driven haircut per asset and a portfolio-level required Initial Margin, calibrated to a 99% 5-day shortfall coverage target.

**Required outputs:**
1. A per-asset haircut model with reported coefficients/feature importances — must be explainable enough for a model-risk reviewer to sign off (state which features you excluded and why, e.g. collinearity).
2. A portfolio-level IM figure per sample client portfolio, with a breakdown showing the "linear per-asset" contribution vs. any "correlation/concentration add-on."
3. Backtest evidence: Kupiec POF and Christoffersen independence test results on daily VaR breach counts, plus a stress-scenario replay (at minimum GFC 2008 and COVID March 2020) against current sample portfolios.
4. A comparison table: your model's total required margin vs. the existing static grid's, across the sample portfolios, with commentary on where/why they diverge.
5. Governance section: floor/cap logic, and what triggers a manual override.

---

## Take-Home 3: Cross-Asset Funding-Spread Anomaly Detection

**Background:** The financing desk needs early warning when repo, sec-lending, cross-currency basis, or credit basis dislocate — before it shows up as realized cost.

**Data provided:** `funding_spreads.csv` (daily: GC-SOFR spread, sec-lending fee index, cross-ccy basis, CDX-cash basis) and `desk_commentary.txt` (sample overnight notes, some describing genuine stress events, some benign).

**Task:** Build a system that scores each day's spread complex for anomalousness and fuses that with a classification of the accompanying commentary.

**Required outputs:**
1. A statistical anomaly detector (your choice of method) with a documented threshold-setting methodology — not an arbitrary cutoff.
2. A text classifier (rule-based, fine-tuned, or prompted LLM — your choice, but justify it) that labels commentary into stress categories, with a small labeled validation set and reported precision/recall.
3. A fused alert score combining both signals, with the fusion weight/logic explained.
4. A ranked list of the top 10 most anomalous days in the provided history, each with: the driving spread(s), the statistical score, the text-derived signal, and a one-paragraph rationale a trader could act on in <30 seconds.
5. A false-positive/false-negative cost discussion — state explicitly which error type you optimized against and why.

---

## Take-Home 4: Prime Balance & Utilization Forecasting (Deep Learning)

**Background:** Treasury needs a forward balance forecast for RWA and LCR planning, including the known month-end/quarter-end "window dressing" balance reduction pattern.

**Data provided:** `prime_balances.csv` — daily aggregate and top-20-client financing balances and utilization ratio, 2+ years, with a calendar flag column (quarter-end window, holiday, etc.).

**Task:** Forecast balances 1–20 business days ahead, explicitly accounting for the calendar effect, and produce an uncertainty band, not just a point forecast.

**Required outputs:**
1. At minimum three baseline models (e.g., seasonal-naive, SARIMAX with calendar dummy, tuned gradient-boosted trees) **and** one deep sequence model (RNN/LSTM/GRU) — all evaluated on the same walk-forward split.
2. Evidence that the deep learning model outperforms the best baseline — if it doesn't, say so and recommend the baseline instead (this is being tested).
3. P10/P50/P90 forecast bands (quantile or equivalent probabilistic output), not single-point forecasts.
4. An explicit visualization/table showing forecast behavior around a quarter-end window, demonstrating the calendar effect is captured.
5. A short deployment note: expected retrain frequency, feature/calendar pipeline dependencies, and serving latency expectations.

---

## Take-Home 5: RAG Financing-Desk Copilot (GenAI)

**Background:** Traders currently search across policy PDFs, desk notes, and model outputs manually. The desk wants a copilot that answers grounded questions with citations and refuses to guess.

**Data provided:** `policy_docs/` (sample haircut policy, financing rate policy), `desk_notes/` (sample dated commentary), and mock API access to the outputs of Take-Homes 1–4.

**Task:** Build a retrieval-augmented question-answering system over this corpus.

**Required outputs:**
1. A working retrieval pipeline (your choice of sparse/dense/hybrid) with a reported recall@k on a self-constructed set of ≥15 test queries with labeled relevant documents.
2. A generation component that answers only from retrieved context, cites sources per claim, and returns an explicit "insufficient evidence" response when appropriate — demonstrate this refusal behavior on at least 2 constructed out-of-scope queries.
3. A written comparison of prompt-engineering-only vs. fine-tuning approaches for this use case, with a recommendation and justification (this maps directly to the JD's "evaluate trade-offs" requirement).
4. An evaluation harness: faithfulness/groundedness score methodology, plus results on ≥10 test queries.
5. A one-paragraph infrastructure note: what would run on-prem vs. via external API, and why, given data sensitivity.
