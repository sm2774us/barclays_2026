# Liquid Financing — AI/ML Modeler Take-Home
### Barclays · Institutional-Grade Reference Solutions (Python 3.13)

---
---

[↩️ Back to README.md](../README.md)

---
---

## Table of Contents

- [0. What This Repository Is](#0-what-this-repository-is)
- [1. Environment Setup](#1-environment-setup)
  - [Option A — `uv` (recommended)](#option-a--uv-recommended)
  - [Option B — plain `pip`](#option-b--plain-pip)
  - [`requirements.txt` contents](#requirementstxt-contents)
  - [Running the notebook](#running-the-notebook)
- [2. Repository Layout](#2-repository-layout)
- [3. First-Principles Walkthrough (What's Actually Happening, and Why)](#3-first-principles-walkthrough-whats-actually-happening-and-why)
  - [3.1 Why walk-forward validation, not K-fold — explained simply](#31-why-walk-forward-validation-not-k-fold--explained-simply)
  - [3.2 Elastic Net — explained simply](#32-elastic-net--explained-simply)
  - [3.3 Gradient-boosted trees and the utilization threshold — explained simply](#33-gradient-boosted-trees-and-the-utilization-threshold--explained-simply)
  - [3.4 Conformalized Quantile Regression (CQR) — explained simply](#34-conformalized-quantile-regression-cqr--explained-simply)  
  - [3.5 The pinball loss — explained simply](#35-the-pinball-loss--explained-simply)
  - [3.6 Kupiec and Christoffersen VaR backtests — explained simply](#36-kupiec-and-christoffersen-var-backtests--explained-simply)
  - [3.7 Robust Mahalanobis distance and Isolation Forest — explained simply](#37-robust-mahalanobis-distance-and-isolation-forest--explained-simply)
  - [3.8 Sequence-to-sequence GRU with a calendar embedding — explained simply](#38-sequence-to-sequence-gru-with-a-calendar-embedding--explained-simply)
  - [3.9 BM25, TF-IDF cosine, and Reciprocal Rank Fusion — explained simply](#39-bm25-tf-idf-cosine-and-reciprocal-rank-fusion--explained-simply)
- [4. Per-Project Architecture Diagrams and Extended Derivations (Delta Deep-Dive)](#4-per-project-architecture-diagrams-and-extended-derivations-delta-deep-dive)
  - [4.1. P1 — Securities-Lending Fee Forecasting: Full System Architecture](#41-p1--securities-lending-fee-forecasting-full-system-architecture)
  - [4.2. P2 — Client Margin & Haircut Optimization: Full System Architecture](#42-p2--client-margin--haircut-optimization-full-system-architecture)
  - [4.3. P3 — Cross-Asset Funding-Spread Anomaly Detection: Full System Architecture](#43-p3--cross-asset-funding-spread-anomaly-detection-full-system-architecture)
  - [4.4. P4 — Prime Balance Forecasting: Full System Architecture](#44-p4--prime-balance-forecasting-full-system-architecture)
  - [4.5. P5 — RAG Financing-Desk Copilot: Full System Architecture](#45-p5--rag-financing-desk-copilot-full-system-architecture)
  - [4.6. Cross-Project Production Standards (Referenced Throughout §4 and PRESENTATION_TALKING_POINTS.md)](#46-cross-project-production-standards-referenced-throughout-4-and-presentation_talking_pointsmd)
- [5 Mapping Every Project Back to the JD, Line by Line](#5-mapping-every-project-back-to-the-jd-line-by-line)
- [7 Companion Documents](#7-companion-documents)


[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

---

## 3. First-Principles Walkthrough (What's Actually Happening, and Why)

### 3.1. Why walk-forward validation, not K-fold — explained simply

Imagine grading a weather forecaster by letting them see Wednesday's actual weather before asking them
to "predict" Tuesday's. That's what a random K-fold split does to a time series: it lets the model train
on data from *after* the point it's being asked to forecast, because a random shuffle doesn't respect
time order. Walk-forward validation instead always trains on the past and tests on the future, exactly
as the model will be used in production — so the backtest number is not a lie.

### 3.2. Elastic Net — explained simply

Ordinary least-squares regression finds the line that best fits the data, full stop — even if two of
your input variables are basically saying the same thing (e.g., utilization and days-to-cover, which
move together), OLS will happily give both huge, unstable, sign-flipping coefficients. Elastic Net adds
two penalties: an **L1 penalty** that pushes unhelpful coefficients all the way to zero — this is
automatic feature selection, and it means the final model can be handed to a model-risk reviewer with a
short, defensible feature list — and an **L2 penalty** that shrinks correlated coefficients toward each
other rather than letting one arbitrarily dominate. Mathematically:

$$
\hat\beta = \arg\min_\beta \lbrace \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \lambda\Big(\alpha \lVert \beta \rVert_1 + \tfrac{1-\alpha}{2}\lVert \beta \rVert_2^2\Big) \rbrace
$$

where $\alpha \in [0,1]$ (the "l1\_ratio") interpolates between pure LASSO ($\alpha=1$) and pure Ridge
($\alpha=0$). We cross-validate $\alpha$ and $\lambda$ jointly.

### 3.3. Gradient-boosted trees and the utilization threshold — explained simply

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

### 3.4. Conformalized Quantile Regression (CQR) — explained simply

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

### 3.5. The pinball loss — explained simply

If you always guess "a little too low," you're not being graded fairly by squared error alone — you
need a loss that's asymmetric to match the quantile you're targeting. The pinball loss at quantile $q$
penalizes under-prediction by a factor of $q$ and over-prediction by a factor of $(1-q)$:

$$
\rho_q(u) = u\,(q - \mathbb{1}[u<0]), \qquad u = y - \hat y
$$

At $q=0.9$, guessing too *low* is penalized nine times harder than guessing too *high* — which is
exactly the asymmetry you want when forecasting the *90th percentile*: being below the truth is a bigger
miss than being above it.

### 3.6. Kupiec and Christoffersen VaR backtests — explained simply

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

### 3.7. Robust Mahalanobis distance and Isolation Forest — explained simply

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

### 3.8. Sequence-to-sequence GRU with a calendar embedding — explained simply

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

### 3.9. BM25, TF-IDF cosine, and Reciprocal Rank Fusion — explained simply

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

[🔝 Back to Top](#table-of-contents)

---

## 4. Per-Project Architecture Diagrams and Extended Derivations (Delta Deep-Dive)

The sections above explain each *technique*; this section explains each *system* end-to-end — full
data-flow architecture, every equation the codebase actually executes (not just the headline one), and
the specific engineering decisions a Model Risk reviewer or QR panelist will drill into.

### 4.1. P1 — Securities-Lending Fee Forecasting: Full System Architecture

```
  RAW PANEL                    FEATURE ENGINEERING                   THREE-TIER MODEL STACK
  (sec_lending_panel.csv)      (make_features)                       (fit_and_evaluate_fold)
┌────────────────────┐        ┌─────────────────────────┐         ┌─────────────────────────────────┐
│ date, name_id,     │        │ fee_lag1, fee_lag5,     │         │  TIER 1: ElasticNetCV           │
│ utilization,       │───────▶│ fee_ewm10 (EWMA span=10)│────────▶│  (auditable linear baseline)    │
│ days_to_cover,     │        │ fee_mom_5d = lag1-lag5  │         │        │                        │
│ dividend_flag,     │        │ util_x_dtc (interaction)│         │        ▼                        │
│ gc_spread,         │        │ util_squared_over_90    │         │  TIER 2: LightGBM × 3 quantiles │
│ fee_bps            │        │  = max(util-0.9,0)^2    │         │  (q=0.1, 0.5, 0.9 — captures    │
└────────────────────┘        └─────────────────────────┘         │  the >90% utilization kink)     │
                                                                  │        │                        │
                                                                  │        ▼                        │
                                                                  │  TIER 3: NNLS blend weights     │
                                                                  │  (fit on CALIB fold only)       │
                                                                  └────────┬────────────────────────┘
                                                                           ▼      
                                                          ┌──────────────────────────────────────┐
                                                          │  TIER 4: Split-Conformal (CQR)       │
                                                          │  widens [P10,P90] using CALIB-fold   │
                                                          │  conformity scores → guaranteed      │
                                                          │  marginal 80% coverage               │
                                                          └────────────────┬─────────────────────┘
                                                                           ▼
                                                    y_pred_blend, y_pred_p10, y_pred_p90
                                                    + weighted_MAPE, hit_rate, pinball loss,
                                                      coverage_80  (FeeForecastResult)
```

**Why three data splits per fold (train / calib / test), not two.** A subtlety worth stating explicitly
to the panel: conformal calibration *requires* a calibration set that is disjoint from *both* the
training set (or its conformity scores are optimistic — the model always fits its own training residuals
too well) and the test set (or you're calibrating on the answer key). `walk_forward_backtest` explicitly
carves out a middle `calib` slice between `train` and `test` for exactly this reason — collapsing this to
a two-way split is the single most common conformal-prediction implementation bug, and volunteering that
you've built it correctly is a strong signal.

**The NNLS blend, derived.** The blend weights solve:

$$
\hat w = \arg\min_{w \geq 0} \left\lVert \begin{pmatrix}\hat y^{EN}_{\text{calib}} & \hat y^{GBM}_{\text{calib}}\end{pmatrix} w - y_{\text{calib}} \right\rVert_2^2
$$

**Line-by-line:** this is ordinary least squares with a **non-negativity constraint** on the weights —
solved via the Lawson-Hanson active-set algorithm (`scipy.optimize.nnls`), not a closed-form normal
equation, because the constraint makes the feasible region a cone rather than all of $\mathbb{R}^2$. The
non-negativity constraint is a **deliberate model-risk choice**: it guarantees the blended forecast can
never be interpreted as "the linear model says X but we're subtracting some of it based on what the tree
model thinks" — a negative blend weight would be uninterpretable to a reviewer and is structurally
forbidden here, at the (small) cost of not being the unconstrained-OLS-optimal blend. After solving, we
renormalize $w \leftarrow w / \sum w$ so the blend is a genuine convex combination (weights sum to 1) —
this guarantees the blended forecast can never exceed the envelope of the two tier forecasts, a further
interpretability guarantee.

**Say it out loud, if pushed on "why not just always use the GBM, since it captures the kink":** *"The
Elastic Net tier isn't there to compete on raw accuracy — it's there because a model-risk reviewer can
verify every one of its ~10 coefficients by hand in about five minutes, and NNLS blending means that
whenever the linear tier is genuinely informative on a calibration fold (e.g., for the ~85% of names
trading well below the utilization kink, where the relationship really is close to linear), the blend
weight discovers that and up-weights it automatically — the two tiers aren't in competition, the blend
is a data-driven statement of which tier to trust and by how much, per fold."*

### 4.2. P2 — Client Margin & Haircut Optimization: Full System Architecture

```
  PER-ASSET LAYER                              PORTFOLIO LAYER                      GOVERNANCE LAYER
┌─────────────────────────┐                ┌────────────────────────────────┐    ┌───────────────────────────┐
│ collateral_universe     │                │ portfolio_features             │    │ required_initial_margin() │
│  [vol, adv, corr_mkt,   │                │ (concentration, correlation)   │    │                           │
│   rating] → log-linear  │                │        +                       │    │  im = linear_im           │
│   LASSO:                │───┐            │ shortfall_residual             │    │     + correlation_addon   │
│  log(haircut) ~         │   │            │ (HistSim shortfall NOT         │    │                           │
│  log_vol+log_adv+       │   │            │  explained by summed per-asset │    │  im_clipped = clip(im,    │
│  corr_mkt+rating        │   │            │  haircuts)                     │    │    rules_floor,           │
└─────────────────────────┘   │            └───────────────┬────────────────┘    │    1.2 * var99_cap)       │
             │                │                            │                     │                           │
             ▼                │                            ▼                     │  ← ML can only TIGHEN     │
      per-asset haircut       │                 GradientBoostingRegressor        │    risk within a pre-     │
      → linear_im = haircut × │                 (Huber loss, robust to fat       │    approved band, never   │
        |qty| × price         │                  shortfall tails)                │    silently drift outside │
                              └────────────────────────────┬─────────────────────│    it                     │
                                                           ▼                     └───────────────────────────┘
                                              BACKTEST: Kupiec POF +
                                              Christoffersen independence +
                                              Basel traffic-light zone
```

**Why the correlation add-on is fit on the *residual*, not as a second full model.** This is the single
most senior-sounding technical point in P2 — worth stating explicitly, unprompted. If you instead fit a
second GBM to predict shortfall directly (not the residual), you lose the guarantee that the linear
layer's contribution is separable and auditable — the GBM could silently "re-explain" the same variance
the LASSO layer already captured, making the two-layer decomposition meaningless for a reviewer trying to
understand "how much of this client's margin is coming from the auditable linear layer vs. the black-box
add-on." Fitting on the residual **forces** an orthogonal decomposition by construction:

$$
\text{Shortfall}_{\text{total}} = \underbrace{\sum_{i} h_i \cdot |q_i| \cdot p_i}_{\text{linear\_im (fully auditable)}} + \underbrace{g_\theta(\text{portfolio features})}_{\text{correlation\_addon (GBM on residual)}}
$$

**Kupiec test, worked with actual numbers from the codebase's demo.** With `target_rate=0.01` and a
250-day window, if `n_breaches=3`: $\hat p = 3/250 = 0.012$.

$$
LR = -2\left[(250-3)\ln(0.99) + 3\ln(0.01) - \big((247)\ln(0.988)+3\ln(0.012)\big)\right]
$$

Computing: $247\ln(0.99)=-2.483$, $3\ln(0.01)=-13.816$, sum $=-16.30$ (null log-likelihood). Alternative:
$247\ln(0.988)=-2.981$, $3\ln(0.012)=-13.32$, sum$=-16.30$ (nearly identical — as expected, since 1.2% is
very close to the 1% target). $LR \approx 0.06$, $p\text{-value} \approx 0.81$ — **fails to reject**,
meaning 3 breaches out of 250 is entirely consistent with a well-calibrated 99% VaR model. **Say it out
loud:** *"3 breaches against a 1% target on 250 days is well within statistical noise — I'd only start
worrying around 6-7+ breaches, which is exactly the Basel amber-zone boundary the traffic-light function
implements."*

### 4.3. P3 — Cross-Asset Funding-Spread Anomaly Detection: Full System Architecture

```
  STATISTICAL PATH                                          TEXTUAL PATH
┌──────────────────────────────┐                    ┌─────────────────────────────┐
│ funding_spreads.csv          │                    │ desk_commentary.txt         │
│ [gc_sofr_spread,             │                    │  (labeled stress/benign)    │
│  sec_lending_fee_index,      │                    └──────────────┬──────────────┘
│  xccy_basis_3m,              │                                   ▼
│  cdx_cash_basis]             │                    TfidfVectorizer(1,2-grams)
└──────────────┬───────────────┘                    → LogisticRegression(balanced)
               ▼                                                   │
     MinCovDet(support_fraction=0.75)                              ▼
     → robust location/covariance                          text_stress_prob ∈ [0,1]
     → mahalanobis_z = sqrt(d_MCD(x))                              │
               │                                                   │
               ▼                                                   │
     IsolationForest(contamination=0.02)                           │
     → isolation_score = -avg_path_length                          │
               │                                                   │
               └─────────────────┬─────────────────────────────────┘
                                 ▼
                       fuse_scores(): alpha·sigmoid(z - 3.0) + (1-alpha)·text_stress_prob
                                 ▼
                       neyman_pearson_threshold(): minimize cost_fn·P(miss) + cost_fp·P(false_alarm)
                                 ▼
                       Ranked AlertRecord list, per-day driving_spread attribution
```

**Isolation Forest's anomaly score, derived — the part most candidates only gesture at.** For a point
$x$, average its path length $h(x)$ across an ensemble of $t$ random isolation trees, then normalize
against the *expected* path length under random splitting of $n$ points:

$$
s(x,n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \quad H(k)=\sum_{j=1}^{k}\frac1j \text{ (harmonic number)}
$$

**Line-by-line:** $c(n)$ is the expected path length of an **unsuccessful search in a Binary Search
Tree** built on $n$ random points — this is the exact same quantity that appears in BST average-case
analysis, repurposed here as the "normal" reference path length. If $\mathbb{E}[h(x)] \to c(n)$ (average
isolation difficulty), $s\to 0.5$; if $\mathbb{E}[h(x)]\to 0$ (isolated almost immediately — very few
splits needed), $s\to 1$ (maximally anomalous); if $\mathbb{E}[h(x)]\to n-1$ (never isolated — needs
essentially every point split individually), $s\to 0$. The codebase uses `-score_samples`, which
sklearn defines so that **more negative raw scores mean more anomalous** — negating flips this to the
intuitive "higher = more anomalous" convention used throughout `AlertRecord`.

**Neyman-Pearson threshold, derived from the actual code.** The codebase doesn't use a fixed 0.5
threshold or maximize F1 — it directly minimizes:

$$
\text{Cost}(\tau) = c_{FN}\cdot n_{\text{pos}}\cdot(1-\text{Recall}(\tau)) + c_{FP}\cdot \text{FP}(\tau)
$$

where $FP(\tau)$ is recovered algebraically from the precision-recall curve via
$FP = TP\cdot(1/\text{Precision} - 1)$ (since $\text{Precision}=TP/(TP+FP)$). **Say it out loud:** *"With
$c_{FN}=5, c_{FP}=1$ in the reference implementation, missing a genuine funding-stress event is treated
as five times costlier than a false alarm — a ratio I'd calibrate with the desk directly (e.g., against
the realized carry cost of an un-hedged dislocation vs. the analyst-hours cost of chasing a false alarm),
not leave as an arbitrary default; the threshold-selection *methodology* being explicit and re-calibratable
is exactly what Take-Home 3's brief asks for ('not an arbitrary cutoff')."*

### 4.4. P4 — Prime Balance Forecasting: Full System Architecture

```
  ENCODER                                    DECODER (unrolled horizon steps)
┌───────────────────────┐         ┌─────────────────────────────────────────────────────┐
│ balance history       │         │  for t in range(horizon):                           │
│ (60-day window,       │         │      calendar_emb = Embedding(calendar_code[t])     │
│  standardized)        │────────▶│      h = GRUCell([h, calendar_emb], h)              │
│         │             │         │      p10 = Linear_p10(h)                            │
│         ▼             │         │      p50 = Linear_p50(h)                            │
│  nn.GRU(n_layers=1)   │         │      p90 = Linear_p90(h)                            │
│  → h_n (final hidden) │         └─────────────────────────────┬───────────────────────┘
└───────────────────────┘                                       ▼
                                                Loss = Σ_q pinball_loss(y, pred_q, q)  for q∈{.1,.5,.9}
                                                             (all 3 heads trained JOINTLY, single backward pass)
```

**Why calendar codes are injected at the *decoder*, not concatenated into the encoder input.** A subtle,
easy-to-miss design point worth volunteering: the calendar effect the model needs to condition on is
**known in advance for the forecast horizon** (we know today which future days are quarter-end) but the
encoder only ever sees *historical* data. Concatenating calendar flags into the encoder's input sequence
would be correct but insufficient — the encoder's fixed final hidden state $h_n$ has no mechanism to
"know" which *future* days are quarter-end. Injecting the embedding at each **decoder** step instead
directly conditions every single forecasted step on its own known calendar label — this is precisely
because we have *perfect future information* about the calendar (unlike the balance itself), and the
architecture should exploit that asymmetry rather than forcing the model to infer it indirectly.

**GRU update equations, restated for what this specific architecture computes at each decoder step**
(the concatenated input here is $[h_{t-1}, \text{calendar\_emb}_t]$, fed through a `GRUCell`, which
internally computes exactly the $z_t, r_t, \tilde h_t$ gates from Q24 of the interview prep document —
worth being able to write out cold if asked "what does GRUCell actually compute internally"):

$$
z_t = \sigma\big(W_z[h_{t-1}, c_t] + b_z\big), \quad r_t = \sigma\big(W_r[h_{t-1}, c_t]+b_r\big), \quad \tilde h_t = \tanh\big(W[r_t\odot h_{t-1}, c_t]+b\big), \quad h_t = (1-z_t)h_{t-1} + z_t\tilde h_t
$$

where $c_t$ = calendar embedding at decoder step $t$. **The honest baseline-discipline result** (already
reported in §3.8) is the single most important talking point for this project: the GRU does **not** win
on this synthetic panel, and the module says so — this is a deliberate test of judgment, not a bug to
apologize for.

### 4.5. P5 — RAG Financing-Desk Copilot: Full System Architecture

```
  INDEXING (offline)                QUERY TIME                                     GENERATION
┌──────────────────────────┐      ┌───────────────────────────────────────┐      ┌────────────────────────────────┐
│ Document corpus          │      │ query                                 │      │  build_prompt(query, retrieved)│
│ (policy_docs/,           │      │   │                                   │      │  → SYSTEM_PROMPT +             │
│  desk_notes/)            │      │   ├──▶ BM25.score(query)              │      │     [doc_id:source] context +  │
│         │                │      │   │      (sparse, exact-match)        │      │     QUESTION                   │
│         ▼                │      │   │           │                       │      │              │                 │
│  BM25 index              │      │   │           ▼                       │      │              ▼                 │
│  (term freq, doc freq,   │      │   │      bm25_rank                    │      │  generate_grounded_answer():   │
│   avg doc length)        │      │   │                                   │      │   IF max_dense_similarity      │
│         +                │      │   └──▶ TfidfVectorizer.transform      │      │      < 0.12 (floor)            │
│  TfidfVectorizer.fit     │      │        → cosine_similarity            │      │      → INSUFFICIENT_EVIDENCE   │
│  (dense-retrieval proxy) │      │        → dense_rank                   │      │   ELSE                         │
└──────────────────────────┘      │                    │                  │      │      → synthesize + cite       │
                                  │                    ▼                  │      │        [doc_id] per claim      │
                                  │         RRF: Σ 1/(60+rank) per ranker │      └────────────────────────────────┘
                                  └───────────────────────────────────────┘
```

**The refusal gate, derived — why raw cosine similarity and not the RRF score.** RRF scores are
**rank-based**, not magnitude-based (§3.9) — on a 3-document toy corpus, even a spurious single-term
match will receive a non-trivial RRF score simply by virtue of being *some* document's best available
rank. This means RRF alone cannot distinguish "genuinely relevant, strong match" from "best of a bad lot"
— exactly the distinction the refusal behavior needs. `max_dense_similarity` instead directly reports the
raw cosine similarity (bounded and 0 for genuinely no lexical overlap), giving a magnitude-meaningful
signal for the floor check:

$$
\text{is\_refusal} = \mathbb{1}\big[\max_i \cos(\vec q, \vec d_i) < \text{floor}\big], \qquad \text{floor}=0.12 \text{ (tunable per corpus)}
$$

**Say it out loud, anticipating the obvious follow-up "how did you pick 0.12":** *"This threshold should
be calibrated empirically against a labeled set of in-scope vs. out-of-scope queries, exactly the same
way the Neyman-Pearson threshold in P3 is calibrated against labeled data rather than asserted — 0.12 is
a reasonable default for a small, topically-narrow corpus like this reference implementation's 3-document
demo, but I would not ship it unchanged against the real policy-doc corpus without first running the
recall@k and refusal-precision evaluation described in the take-home brief's required deliverable #4."*

**Faithfulness proxy vs. true LLM-as-judge faithfulness — an honest limitation, stated explicitly.** The
shipped `faithfulness_proxy` (Jaccard token overlap between answer and retrieved context) is a
**necessary-but-not-sufficient** sanity check: an answer can have high token overlap with its context and
still misstate a number (e.g., citing the right clause but transposing "8.5%" to "5.8%" — every token
still overlaps). **Say it out loud:** *"This proxy is intentionally deterministic and model-free so it
can run in CI on every commit without an API dependency or cost — it is a lower bound, not a replacement,
for a live LLM-as-judge faithfulness score (per Q6 of the technical interview prep — decomposing the
answer into atomic claims and checking each against context), which is the production-grade evaluation
this module is designed to be swapped in for without changing the retrieval or refusal contract at all."*

### 4.6. Cross-Project Production Standards (Referenced Throughout §4 and PRESENTATION_TALKING_POINTS.md)

```
STANDARD                     P1        P2          P3        P4                 P5
──────────────────────       ───────   ───────     ───────   ───────            ───────
Walk-forward validation      ✓         ✓           ✓         ✓                  n/a (retrieval eval = recall@k)
Auditable baseline tier      ✓ (EN)    ✓ (LASSO)   n/a       ✓ (naive/SARIMAX)  n/a
Uncertainty quantification   ✓ (CQR)   n/a         n/a       ✓ (quantile GRU)   ✓ (confidence tier)
Explicit cost-asymmetry      ✓ (wMAPE  n/a         ✓ (N-P    n/a                ✓ (refusal =
handling                     floor)                thresh.)                     cost of false answer)
Governance clipping/refusal  n/a       ✓ (floor/   n/a       n/a                ✓ (INSUFFICIENT_
                                        cap)                                    EVIDENCE)
Regulatory backtest           n/a      ✓ (Kupiec/  n/a       n/a                n/a
                                       Christ.)
Baseline-beats-DL discipline n/a       n/a         n/a       ✓ (honestly        n/a
                                                             reported)
```

This table is the single artifact that answers "how does this actually ship" (§3, Notebook Composition)
across all five projects simultaneously — every column has at least one concrete, code-backed answer,
not an aspiration.

[🔝 Back to Top](#table-of-contents)

---

## 5. Mapping Every Project Back to the JD, Line by Line

```
JD REQUIREMENT (verbatim from AI _ ML Modeler - Liquid Finance.docx)          PROJECT(S) THAT DEMONSTRATE IT
─────────────────────────────────────────────────────────────────────────    ──────────────────────────────
"Regression (OLS, LASSO, Elastic Net)"                                        P1 (Elastic Net), P2 (LASSO)
"Tree Based Models"                                                           P1 (LightGBM quantile), P2 (GBM
                                                                              add-on), P3 (Isolation Forest)
"Time Series – Analysis, Forecasting, Pricing Modeling"                       P1 (fee forecasting), P4 (SARIMAX,
                                                                              seasonal-naive, LightGBM, GRU)
"Deep Learning (DL) methods: Neural Networks, MLP's, RNN, LSTM, GRU"          P4 (GRU seq2seq encoder-decoder)
"Strong Gen AI expertise: fine-tuning, prompt engineering, retrieval-         P5 (hybrid RAG, grounded
 augmented generation, model evaluation, and research"                        generation contract, recall@k eval)
"Independently take on business problem statements ... end-to-end"            All 5 — each maps to a distinct
                                                                              P&L/risk lever, not a generic demo
"Evaluate and articulate trade-offs between modeling techniques"              P1 (EN vs. GBM), P4 (baseline
                                                                              discipline), P5 (fine-tune vs. RAG)
"configure inference infrastructure (BMC servers)"                            P5 (on-prem/API routing note)
"Write clean, production-grade Python code ... released into ...              All 5 — Google-style docstrings,
 production codebase"                                                         dataclasses, typed signatures, tests
"Collaborate with the IT engineering team ... deployment and infrastructure"  Cross-Project Production Standards
                                                                              table (§4.6.) + governance logic
```

[🔝 Back to Top](#table-of-contents)

---

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

[🔝 Back to Top](#table-of-contents)

---

## 6. Testing

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

[🔝 Back to Top](#table-of-contents)

---

## 7. Companion Documents

- **`dissertation.tex` / `dissertation.pdf`** — PhD-dissertation-style extended treatment of the same
  five systems, with the identical first-principles rigor as §3 above, plus embedded static figures from
  `charts/*.png` and full derivations of every equation used.
- **`PROBLEMS.md`** — the original take-home briefs (unmodified, provided for reference).

[🔝 Back to Top](#table-of-contents)

---
