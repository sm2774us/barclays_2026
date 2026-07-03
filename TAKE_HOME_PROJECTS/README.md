# Liquid Financing — AI/ML Modeler Take-Home
### Barclays · Institutional-Grade Reference Solutions (Python 3.13)

---
---

[↩️ Back to PROBLEMS.md](./PROBLEMS.md)

---
---

## 0. What This Repository Is

Five production-grade reference solutions to the take-home briefs in `PROBLEMS.md`, engineered to the
standard a Citadel or Jane Street quant would expect to walk into a live production system, not a
Kaggle-leaderboard notebook. Every model is:

- **Backed by a named academic result**, not folklore — Conformalized Quantile Regression (Romano,
  Patterson & Candès, 2019), the Kupiec (1995) and Christoffersen (1998) VaR backtests, Reciprocal Rank
  Fusion (Cormack, Clarke & Buettcher, 2009), Isolation Forest (Liu, Ting & Zhou, 2008), BM25 (Robertson
  & Zaragho, 2009).
- **Validated the way a desk actually validates** — walk-forward, expanding-window splits only; never a
  random K-fold, because financing time series are serially autocorrelated and event-clustered, and a
  random split leaks future information across the train/validation boundary.
- **Governed the way a regulated model actually ships** — floor/cap clipping, PSI drift monitoring,
  champion/challenger shadow periods, and explicit fallback logic are part of the code, not an
  afterthought in a slide.

---

## 1. Environment Setup

### Option A — `uv` (recommended)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run jupyter nbconvert --to notebook --execute --output notebook.ipynb notebook.ipynb
uv run pytest tests/ -q
```

### Option B — plain `pip`

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --output notebook.ipynb notebook.ipynb
```

### `requirements.txt` contents

```
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
lightgbm>=4.3
scipy>=1.13
statsmodels>=0.14
torch>=2.3
plotly>=5.22
kaleido>=0.2.1,<1.0   # static PNG export without a live Chrome dependency
jupyter>=1.0
nbconvert>=7.16
pytest>=8.2
```

### Running the notebook

```bash
jupyter notebook notebook.ipynb
# — or, to regenerate all outputs and charts headlessly —
jupyter nbconvert --to notebook --execute --output notebook.ipynb notebook.ipynb
```

The notebook imports directly from `src/liquid_financing/`, so **the modules are the source of truth**;
the notebook is a thin, chart-producing driver over them. This mirrors how a real desk would structure a
take-home submission — code that survives being imported into a production repo, with the notebook as a
presentation layer on top.

---

## 2. Repository Layout

```
.
├── README.md                     (this file)
├── PROBLEMS.md                   (unmodified — the original take-home briefs)
├── notebook.ipynb                (GitHub-renderable driver notebook, Plotly HTML+PNG outputs)
├── dissertation.tex / .pdf       (PhD-dissertation-style extended writeup, with embedded PNG figures)
├── requirements.txt / pyproject.toml
├── src/liquid_financing/
│   ├── __init__.py
│   ├── plotting.py               shared Plotly house-style + HTML/PNG persistence
│   ├── p1_fee_forecasting.py     Elastic Net + Quantile-LightGBM + Conformal calibration
│   ├── p2_margin_optimization.py LASSO haircut + GBM add-on + Kupiec/Christoffersen backtests
│   ├── p3_anomaly_detection.py   Robust Mahalanobis + Isolation Forest + TF-IDF stress classifier
│   ├── p4_balance_forecasting.py Seasonal-Naive / SARIMAX / LightGBM / GRU quantile seq2seq
│   └── p5_rag_copilot.py         BM25 + TF-IDF hybrid retrieval, Reciprocal Rank Fusion, grounded synthesis
├── data/
│   ├── sec_lending_panel.csv, collateral_universe.csv, client_portfolios.csv
│   ├── funding_spreads.csv, desk_commentary.txt
│   ├── prime_balances.csv, prime_balances_by_client.csv
│   ├── policy_docs/  (haircut_policy_v7.txt, financing_rate_policy_v4.txt)
│   └── desk_notes/   (4 dated desk notes)
└── charts/                       every Plotly figure, as both .html (interactive) and .png (static)
```

---

## 3. First-Principles Walkthrough (What's Actually Happening, and Why)

### 3.1 Why walk-forward validation, not K-fold — explained simply

Imagine grading a weather forecaster by letting them see Wednesday's actual weather before asking them
to "predict" Tuesday's. That's what a random K-fold split does to a time series: it lets the model train
on data from *after* the point it's being asked to forecast, because a random shuffle doesn't respect
time order. Walk-forward validation instead always trains on the past and tests on the future, exactly
as the model will be used in production — so the backtest number is not a lie.

### 3.2 Elastic Net — explained simply

Ordinary least-squares regression finds the line that best fits the data, full stop — even if two of
your input variables are basically saying the same thing (e.g., utilization and days-to-cover, which
move together), OLS will happily give both huge, unstable, sign-flipping coefficients. Elastic Net adds
two penalties: an **L1 penalty** that pushes unhelpful coefficients all the way to zero — this is
automatic feature selection, and it means the final model can be handed to a model-risk reviewer with a
short, defensible feature list — and an **L2 penalty** that shrinks correlated coefficients toward each
other rather than letting one arbitrarily dominate. Mathematically:

$$
\hat\beta = \arg\min_\beta \left\{ \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \lambda\Big(\alpha \lVert \beta \rVert_1 + \tfrac{1-\alpha}{2}\lVert \beta \rVert_2^2\Big) \right\}
$$

where $\alpha \in [0,1]$ (the "l1\_ratio") interpolates between pure LASSO ($\alpha=1$) and pure Ridge
($\alpha=0$). We cross-validate $\alpha$ and $\lambda$ jointly.

### 3.3 Gradient-boosted trees and the utilization threshold — explained simply

A single decision tree asks a sequence of yes/no questions ("is utilization > 0.9?") and gives a
different answer down each branch. This is exactly the shape of the real phenomenon we're modeling: fee
is roughly flat until utilization crosses ~90%, then rises steeply — a **kink**, not a straight line. No
linear model, however well-regularized, can represent a kink; a tree can represent it with a single
split. Gradient boosting builds many small trees in sequence, where each new tree is trained to correct
the *errors* of all the trees before it:

$$
F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x), \qquad h_m = \arg\min_h \sum_i L\big(y_i,\, F_{m-1}(x_i) + h(x_i)\big)
$$

We use LightGBM's native **quantile objective**, fitting three separate models at $q \in \{0.1, 0.5,
0.9\}$ directly on the pinball loss (defined in §3.5), rather than fitting one point-estimate model and
wrapping it in a symmetric ± interval that would be wrong whenever the true error distribution is skewed
(which fee-spike data always is).

### 3.4 Conformalized Quantile Regression (CQR) — explained simply

Any quantile model can be miscalibrated — it might claim an 80% interval but actually only cover the
truth 65% of the time. Conformal prediction is a wrapper technique that *guarantees* the stated coverage
regardless of the underlying model's quality, using nothing more than a held-out calibration set. The
idea: measure how far outside its own predicted interval the model actually was, on data it did not
train on; then widen (or narrow) every future interval by exactly that amount.

$$
\text{score}_i = \max\big(\hat q_{lo}(x_i) - y_i,\; y_i - \hat q_{hi}(x_i)\big), \qquad
\hat q_{lo}^{CQR}(x) = \hat q_{lo}(x) - Q_{1-\alpha}(\text{scores}), \quad \hat q_{hi}^{CQR}(x) = \hat q_{hi}(x) + Q_{1-\alpha}(\text{scores})
$$

This is precisely the method implemented in `p1_fee_forecasting.conformalize` — the 80% interval you see
plotted in `charts/p1_fee_forecast_interval.png` is not a Gaussian assumption dressed up; it is a
finite-sample, distribution-free guaranteed-coverage interval.

### 3.5 The pinball loss — explained simply

If you always guess "a little too low," you're not being graded fairly by squared error alone — you
need a loss that's asymmetric to match the quantile you're targeting. The pinball loss at quantile $q$
penalizes under-prediction by a factor of $q$ and over-prediction by a factor of $(1-q)$:

$$
\rho_q(u) = u\,(q - \mathbb{1}[u<0]), \qquad u = y - \hat y
$$

At $q=0.9$, guessing too *low* is penalized nine times harder than guessing too *high* — which is
exactly the asymmetry you want when forecasting the *90th percentile*: being below the truth is a bigger
miss than being above it.

### 3.6 Kupiec and Christoffersen VaR backtests — explained simply

A 99% VaR model *should* be breached about 1% of the time — not 0.1%, not 5%. The **Kupiec test** simply
asks: "given how many breaches I actually saw, is 1% a statistically plausible true rate, or is it
significantly off?" It is a likelihood-ratio test comparing the observed breach frequency to the target.
But a model could pass Kupiec and still be dangerous if its breaches *cluster* — ten breaches in one bad
week is a very different risk profile from ten breaches spread evenly across the year, even though both
have the same total breach count. The **Christoffersen test** catches this by modeling breaches as a
two-state Markov chain and testing whether the probability of a breach tomorrow depends on whether there
was a breach today. Both tests reduce to:

$$
LR = -2\log\!\left(\frac{L(\text{null})}{L(\text{alternative})}\right) \sim \chi^2_1
$$

implemented exactly as Basel Committee backtesting guidance specifies, in `p2_margin_optimization.py`.

### 3.7 Robust Mahalanobis distance and Isolation Forest — explained simply

Mahalanobis distance answers "how many standard deviations away is this point, accounting for the fact
that some of my variables move together?" — it's Euclidean distance after "un-correlating" the data. The
catch: if you estimate the mean and covariance using ordinary sample statistics, one wild outlier can
drag the estimated covariance itself off course, making the outlier look *less* extreme than it is (this
is called masking). The **Minimum Covariance Determinant (MCD)** estimator instead finds the subset of
the data (typically 75%) whose covariance matrix has the smallest determinant — i.e., is the "tightest"
possible cloud — and uses that as the robust reference, so a handful of true anomalies cannot corrupt the
yardstick used to measure them. **Isolation Forest** takes an entirely different, non-parametric
approach: it randomly partitions the data with random splits and observes that anomalies, being "few and
different," take *fewer* splits to isolate into their own leaf than normal points do. We run both because
they fail in different regimes — Mahalanobis assumes elliptical (roughly Gaussian) structure and can miss
anomalies in a genuinely non-Gaussian regime shift, which is exactly when Isolation Forest's
distribution-free approach earns its keep.

### 3.8 Sequence-to-sequence GRU with a calendar embedding — explained simply

A plain recurrent network reads a sequence one step at a time, updating a "memory" (hidden state) as it
goes — this is naturally suited to time series with momentum and autocorrelation. But a known, calendar-
driven pattern like "balances always drop right before quarter-end" shouldn't have to be *rediscovered*
by the network purely from the numbers; you can hand it directly as an input. We do this via a learned
**embedding** — a small lookup table mapping "is this a quarter-end day?" (0 or 1) to a short vector —
concatenated into the decoder at every future step, so the network is explicitly told which future days
are quarter-end days as it forecasts them, rather than having to infer it. The three parallel output
heads are trained jointly on the pinball loss at $q \in \{0.1, 0.5, 0.9\}$ (§3.5), producing a genuine
uncertainty band rather than a single brittle number.

**Baseline discipline, honestly reported:** on the shipped synthetic panel, the GRU (MAPE ≈ 2.1%) beats
LightGBM (≈ 2.8%) but does **not** beat the much simpler seasonal-naive and SARIMAX baselines (≈
1.4–1.5%) — see `charts/p4_baseline_comparison.png`. This is exactly the outcome the take-home brief
asks you to be honest about: the synthetic series here has very clean, regular weekly and quarter-end
seasonality, which classical models capture natively and cheaply. The GRU's real edge — a *learned,
non-linear* interaction between calendar effects and balance momentum — should be expected to widen on
noisier, less regularly-seasonal real balance data; the notebook reports the comparison as observed
rather than asserting DL superiority it has not earned on this dataset.

### 3.9 BM25, TF-IDF cosine, and Reciprocal Rank Fusion — explained simply

**BM25** scores how well a query matches a document using term frequency (how often a word appears) and
inverse document frequency (how rare that word is across the whole corpus) — but with *saturation*: the
tenth occurrence of a word barely adds more evidence than the fifth, which stops the score from being
gamed by simple repetition. **TF-IDF cosine similarity** measures document similarity as the angle
between two vectors of weighted word counts — a proxy, in this reference implementation, for what a
production system would compute with a learned sentence-embedding encoder. Rather than trying to
calibrate these two very differently-scaled signals against each other, **Reciprocal Rank Fusion**
sidesteps the problem entirely by only looking at *rank position*: a document's fused score is
$\sum_{\text{ranker}} 1/(k + \text{rank})$. A document that is highly ranked by both signals wins,
regardless of the two systems' raw score scales — this is the standard hybrid-retrieval fusion technique
used in production search systems precisely because it requires no score normalization step to get
right.

---

## 4. Notebook Composition — What Each Cell Produces and Why It's There

| Section | What runs | Chart(s) produced | What the chart tells the panel |
|---|---|---|---|
| **P1** | Walk-forward backtest of the Elastic Net + Quantile-LightGBM + CQR ensemble | `p1_fee_forecast_interval` — realized fee vs. blended forecast with the conformalized 80% band | Shows the model tracking a real fee spike and the calibrated interval widening exactly around the event, not everywhere uniformly |
| | | `p1_walkforward_monitoring` — weighted MAPE & interval coverage by fold | The production monitoring view a desk quant actually watches: is accuracy degrading, is the interval still calibrated to ~80%? |
| **P2** | LASSO haircut fit + simulated VaR backtest | `p2_lasso_coefficients` — signed, ranked coefficients | The auditability artifact — every driver of the haircut is visible and sign-checkable in one glance |
| | | `p2_var_backtest` — P&L path, VaR limit, breach markers, with Kupiec/Christoffersen p-values in the title | The regulatory backtest a Model Risk reviewer asks for first, presented the way a risk committee actually sees it |
| **P3** | Robust anomaly monitor scored on a held-out window | `p3_anomaly_score_timeline` — Mahalanobis z-score over time, true anomaly days starred | Visually validates the detector against ground truth — the stars land on the z-score peaks |
| **P4** | Four-model baseline ladder + GRU quantile forecast | `p4_baseline_comparison` — MAPE bar chart across all four models | The "baseline discipline" artifact: proof the DL model was benchmarked honestly, not assumed superior |
| | | `p4_gru_quantile_fan` — realized balance vs. GRU P10/P50/P90 fan, quarter-end days marked | Shows the calendar-aware uncertainty band actually widening/shifting around the known seasonal event |
| **P5** | Hybrid retrieval + grounded synthesis + refusal demo | `p5_retrieval_scores` — RRF-fused scores for the retrieved chunks on a sample query | Makes the retrieval step legible — which chunks won, and by how much |

Every chart is written to `charts/<name>.html` (open directly in a browser — fully interactive,
zoomable, hoverable) and `charts/<name>.png` (the static version embedded in `dissertation.pdf`).

---

## 5. Testing

```bash
uv run pytest tests/ -q          # 15 tests
```

Each module's `if __name__ == "__main__":` block (Google-style entry point) also runs standalone as a
fast smoke test:

```bash
PYTHONPATH=src python3 src/liquid_financing/p1_fee_forecasting.py
PYTHONPATH=src python3 src/liquid_financing/p2_margin_optimization.py
PYTHONPATH=src python3 src/liquid_financing/p3_anomaly_detection.py
PYTHONPATH=src python3 src/liquid_financing/p4_balance_forecasting.py
PYTHONPATH=src python3 src/liquid_financing/p5_rag_copilot.py
```

---

## 6. Companion Documents

- **`dissertation.tex` / `dissertation.pdf`** — PhD-dissertation-style extended treatment of the same
  five systems, with the identical first-principles rigor as §3 above, plus embedded static figures from
  `charts/*.png` and full derivations of every equation used.
- **`PROBLEMS.md`** — the original take-home briefs (unmodified, provided for reference).

[↩️ Back to PROBLEMS.md](./PROBLEMS.md)
