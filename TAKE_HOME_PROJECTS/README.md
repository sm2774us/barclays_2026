# Barclays — AI/ML Modeler, Liquid Financing
### Take-Home Research Portfolio · 5 Production-Grade ML/GenAI Case Studies
#### Cross-Asset Financing: Equities & Delta One · Rate & Credit Financing · FX · Risk · Futures & Prime Derivatives

> **Delivery philosophy:** Liquid Financing is a *funding-cost and balance-sheet* business first, a modeling business second. Every project below opens with the P&L or risk lever it moves, states the modeling approach with math as evidence, and closes with a production deployment path (data contracts, retraining cadence, monitoring, model risk sign-off) — because a VP-level AI/ML Modeler here is expected to ship into the firm's production codebase, not hand off a notebook.

---
---

[↩️ Back to README.md](../README.md)

---
---

## ⏱️ Interview Session Budget (60 minutes, 5 QR/Traders)

```
PROJECT   TOPIC                                   TIME     PANEL LENS
────────  ──────────────────────────────────────  ───────  ──────────────────────────────────
P1        Sec-Lending Fee & Rebate Forecasting     10 min   Time-series / financing desk P&L
P2        Client Margin & Haircut Optimization     10 min   Credit risk + regression, model risk
P3        Funding-Spread Anomaly Detection          8 min   Unsupervised + NLP, market surveillance
P4        Prime Balance Forecasting (DL)           10 min   RNN/LSTM/GRU, capacity planning
P5        RAG Financing-Desk Copilot (GenAI)       12 min   LLM fine-tuning, prompt eng., infra
Q&A / Roadmap tie-back                              10 min   Business alignment, 50+ backlog fit
```

> **Priority rule:** The JD calls out Gen AI (fine-tuning, RAG, eval), regression (OLS/LASSO/Elastic Net), tree models, time-series/pricing, and DL (MLP/RNN/LSTM/GRU) *by name*. Each project below is deliberately mapped to at least one of those bullet points so no requirement in the JD is left uncovered.

---

## Table of Contents

- [P1 · Securities-Lending Fee & Rebate-Rate Forecasting](#p1--securities-lending-fee--rebate-rate-forecasting)
- [P2 · Client Margin & Haircut Optimization](#p2--client-margin--haircut-optimization)
- [P3 · Cross-Asset Funding-Spread Anomaly Detection](#p3--cross-asset-funding-spread-anomaly-detection)
- [P4 · Prime Balance & Utilization Forecasting (Deep Learning)](#p4--prime-balance--utilization-forecasting-deep-learning)
- [P5 · RAG Financing-Desk Copilot (GenAI / LLM)](#p5--rag-financing-desk-copilot-genai--llm)
- [Cross-Project Production Standards](#cross-project-production-standards)
- [Quick-Reference Equation Sheet](#quick-reference-equation-sheet)

[🔝 Back to Top](#table-of-contents)

---
---

# P1 · Securities-Lending Fee & Rebate-Rate Forecasting

**Open with the business framing (15 seconds):**
> "Liquid Financing earns the spread between what we pay to borrow a security (or the rebate we pay on cash collateral) and what we charge the client. That spread is not static — it moves with utilization, days-to-cover on the short side, and corporate-action calendars (dividends, index rebalances). If I can forecast the *fee curve* a few days ahead, the desk can pre-position inventory and price new loans correctly instead of reactively re-pricing after a squeeze."

## Problem Statement

Forecast the **daily specialness fee** $f_{i,t}$ (annualized bps over the general-collateral rate) for a universe of ~2,000 hard-to-borrow (HTB) equities, 1–5 business days ahead, to feed the pre-trade pricing engine for Equities & Delta One financing.

## Data & Features

```
FEATURE GROUP            EXAMPLES
────────────────────     ──────────────────────────────────────────────────
Utilization               Loan qty / Lendable qty, %-of-float on loan
Availability              Lendable supply, # of active lenders
Short-interest proxies    FINRA-style short interest, days-to-cover
Corporate actions         T-2 to T+2 dividend record-date flag, index add/drop
Rate context               GC rate, SOFR, term structure of rebate
Market microstructure     Borrow-request rejection rate, recall notices/day
```

## Modeling Approach — Three Tiers (per JD: OLS/LASSO/Elastic Net, Tree, Time-Series)

**Tier 1 — Elastic Net baseline (interpretable, model-risk-friendly):**

$$
\hat{f}_{i,t+h} = \beta_0 + \sum_{k=1}^{K}\beta_k x_{k,i,t} + \lambda_1\lVert\beta\rVert_1 + \lambda_2\lVert\beta\rVert_2^2
$$

**Say it out loud:** *"Elastic Net gives us a defensible, auditable linear benchmark — L1 does feature selection across our ~40 utilization/microstructure features so Model Risk can see exactly which drivers survive, L2 handles the multicollinearity between utilization and days-to-cover."*

**Tier 2 — Gradient-boosted trees (LightGBM) for non-linear regime capture:**

$$
\hat{f}^{GBM}_{i,t+h} = \sum_{m=1}^{M} \gamma_m \, h_m(x_{i,t}), \qquad h_m = \arg\min_h \sum_i L\!\left(f_{i,t+h}, F_{m-1}(x_i) + h(x_i)\right)
$$

Trees capture the **threshold effect** in specialness: fees are near-flat until utilization crosses ~90%, then convex. A linear model cannot express this kink; a shallow tree ensemble does natively.

**Tier 3 — Time-series overlay (state-space) for the fee's own autocorrelation and event decay:**

$$
f_{i,t} = \mu_i + \phi (f_{i,t-1}-\mu_i) + \sum_{j} \kappa_j \, \mathbb{1}[\text{event}_{j,t}]\, e^{-\eta_j (t-t_j)} + \varepsilon_t
$$

An Ornstein–Uhlenbeck-style mean reversion term plus exponentially-decaying event kernels for dividend record dates and index rebalances, which are the two most predictable specialness spikes on the desk.

```
FEE DYNAMICS AROUND A DIVIDEND RECORD DATE (STYLIZED)

  Fee (bps)
  120 │                        ▄▄█▓▓
   90 │                     ▄▓█████▓▓▄
   60 │                  ▄▓████████████▓▄
   30 │▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▓██████████████████▓▄▄▁▁▁▁▁▁
    0 └──────────────┬───────┬───────┬─────────────
                    T-3     T-1(rec) T+2
              Utilization    Peak    Decay (η≈0.6/day)
              climbs         squeeze  as shorts unwind
```

## Ensembling & Blend

$$
\hat{f}_{i,t+h} = w_1 \hat{f}^{EN} + w_2 \hat{f}^{GBM} + w_3 \hat{f}^{TS}, \qquad \sum w = 1
$$

Weights fit via constrained least squares on a rolling out-of-sample window (walk-forward, never a random K-fold — fee data is autocorrelated and event-clustered).

```python
import lightgbm as lgb
from sklearn.linear_model import ElasticNetCV
import numpy as np

def train_fee_ensemble(X_train, y_train, X_val, y_val):
    """
    Three-tier fee forecasting ensemble for HTB securities lending.
    X_train columns: utilization, days_to_cover, dividend_flag, gc_spread, ...
    """
    # Tier 1: Elastic Net (auditable baseline)
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1], cv=5, max_iter=5000)
    en.fit(X_train, y_train)

    # Tier 2: LightGBM (non-linear utilization threshold effects)
    gbm = lgb.LGBMRegressor(
        n_estimators=600, num_leaves=31, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, objective="regression_l1"
    )
    gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)])

    preds = np.column_stack([en.predict(X_val), gbm.predict(X_val)])
    # Non-negative least squares blend weight (constrained, sums to 1 downstream)
    from scipy.optimize import nnls
    w, _ = nnls(preds, y_val)
    w = w / w.sum()
    return en, gbm, w

def forecast(en, gbm, w, X_new):
    p = np.column_stack([en.predict(X_new), gbm.predict(X_new)])
    return p @ w
```

## Evaluation

| Metric | Why it matters for financing desk |
|---|---|
| Weighted MAPE (weighted by loan notional) | $ impact, not per-name accuracy |
| Directional hit-rate on fee *changes* > 25bps | Desk cares about repricing triggers, not absolute level |
| Pinball loss at 10th/90th quantile | Tail risk on squeeze days — feeds VaR-style inventory buffer |
| PSI (population stability index) on features, monthly | Drift monitor for Model Risk |

**Deployment:** batch daily at 5:30am ET into the pre-trade pricing cache; GBM + EN retrained weekly, blend weights refit monthly; automatic fallback to EN-only if GBM feature PSI > 0.25.

[🔝 Back to Top](#table-of-contents)

---
---

# P2 · Client Margin & Haircut Optimization

**Open with the business framing:**
> "Margin is capital efficiency for the client and credit protection for the bank — every bps of over-margining is client attrition risk, every bps of under-margining is unhedged tail loss. This is a regression + tree-model problem with a credit-risk objective function, and it needs to be explainable enough to survive a Model Risk review and a client dispute."

## Problem Statement

Given a client's collateral basket (cross-asset: equities, FI, FX forwards, futures) predict the **required initial margin (IM)** haircut per asset such that predicted 99% 5-day VaR of the *liquidation* shortfall is covered, while minimizing over-collateralization versus the current SPAN/rules-based haircut grid.

## Modeling Approach

**Step 1 — OLS/LASSO baseline on log-volatility and liquidity features:**

$$
\log h_i = \beta_0 + \beta_1 \log\sigma_i + \beta_2 \log(\text{ADV}_i) + \beta_3 \, \text{Corr}_{i,\text{mkt}} + \beta_4\,\text{Rating}_i + \varepsilon_i, \quad \hat\beta = \arg\min_\beta \lVert y - X\beta\rVert_2^2 + \lambda\lVert\beta\rVert_1
$$

LASSO shrinks the ~60 candidate liquidity/vol/correlation features to the ~12 that survive Model Risk challenge — critical because every surviving coefficient must have a plausible economic sign story in the model documentation.

**Step 2 — Random Forest / Gradient Boosted Trees for the *portfolio-level* concentration and correlation-breakdown effect that a linear per-asset haircut cannot express:**

$$
\text{IM}_{portfolio} = \underbrace{\sum_i h_i \cdot |q_i| \cdot P_i}_{\text{linear component}} + \underbrace{g_{\theta}(\mathbf{q}, \boldsymbol{\Sigma}, \text{concentration})}_{\text{tree-learned correlation-breakdown add-on}}
$$

where $g_\theta$ is a GBM trained to predict the *historical simulation shortfall residual* not explained by the linear component — this is the piece that captures "correlations go to 1 in a stress move," which is exactly the failure mode 2008/2020 punished rules-based grids for.

```
MARGIN MODEL ARCHITECTURE

  Client Portfolio
        │
        ▼
  ┌─────────────────┐     ┌──────────────────────┐
  │ Per-Asset Linear │     │ Portfolio GBM         │
  │ Haircut (LASSO)  │ ──▶ │ Concentration/Corr Add-on │
  └─────────────────┘     └──────────────────────┘
        │                          │
        └───────────┬──────────────┘
                     ▼
          Required IM  (floor: rules-based grid,
                         cap: 99% 5-day historical-sim VaR × 1.2)
```

## Constrained Optimization Layer

The ML output is never used raw — it is clipped by a governance floor/cap, and the *objective* the model is trained against is:

$$
\min_h \; \mathbb{E}\big[\text{Capital Cost}(h)\big] \quad \text{s.t.} \quad \Pr\big(\text{Shortfall}_{5d} > h \cdot V\big) \le 1\%
$$

```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def fit_margin_model(X_asset, y_haircut, X_portfolio, y_shortfall_residual):
    # Linear, auditable per-asset haircut
    lasso = LassoCV(cv=5, n_alphas=100).fit(X_asset, np.log(y_haircut))

    # Tree-based portfolio correlation-breakdown add-on
    gbm = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.02,
        loss="huber", alpha=0.9  # robust to fat-tailed shortfall residuals
    ).fit(X_portfolio, y_shortfall_residual)

    return lasso, gbm

def required_im(lasso, gbm, X_asset, qty, price, X_portfolio, rules_floor, var99_cap):
    per_asset_haircut = np.exp(lasso.predict(X_asset))
    linear_im = np.sum(per_asset_haircut * np.abs(qty) * price)
    addon = max(gbm.predict(X_portfolio)[0], 0)
    im = linear_im + addon
    return float(np.clip(im, rules_floor, var99_cap))
```

## Backtesting Discipline

- **Kupiec POF + Christoffersen independence tests** on daily VaR breach counts — this is what Model Risk will ask for first.
- Walk-forward, expanding window, re-estimated quarterly; frozen champion/challenger comparison for 2 full quarters before promotion.
- Stress overlay: 2008 GFC, Mar-2020 COVID, 2023 regional-bank scenario replay on current book.

[🔝 Back to Top](#table-of-contents)

---
---

# P3 · Cross-Asset Funding-Spread Anomaly Detection

**Open with the business framing:**
> "The financing desk's biggest tail risk isn't a bad forecast, it's *not noticing* a funding-spread dislocation until it's already cost money. This project fuses a statistical anomaly detector on cross-asset basis/spread time series with an NLP layer that reads overnight repo-market commentary, so the model can distinguish 'this is noise' from 'this is the start of a squeeze.'"

## Problem Statement

Flag anomalous divergence across the repo/rebate/CDS-basis/cross-currency-basis complex that Liquid Financing prices off, in real time, ranked by likely P&L impact, with a natural-language rationale a trader can act on in under 30 seconds.

## Statistical Layer — Robust Multivariate Anomaly Score

$$
z_t = \left(\mathbf{x}_t - \boldsymbol{\mu}_t\right)^{\top} \boldsymbol{\Sigma}_t^{-1} \left(\mathbf{x}_t - \boldsymbol{\mu}_t\right)
$$

Mahalanobis distance on an EWMA-estimated $(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$ across the spread vector $\mathbf{x}_t$ = {GC-SOFR spread, sec-lending fee index, cross-ccy basis, CDX-cash basis}, robustified with a Minimum Covariance Determinant estimator so a single dislocated series doesn't blow up the whole covariance matrix.

An **Isolation Forest** runs in parallel as a non-parametric cross-check (handles regime-dependent non-Gaussianity better than Mahalanobis alone):

$$
s(x) = 2^{-\frac{E[h(x)]}{c(n)}}
$$

where $E[h(x)]$ is average path length to isolate $x$ across the forest — scores near 1 are anomalies.

## NLP Layer — Repo Market Commentary Classification (Gen AI per JD)

Overnight desk notes / news wires are embedded and classified into {benign, funding-stress, idiosyncratic, macro-driven} using a fine-tuned small transformer (or prompt-engineered LLM call as the lower-cost tier), then fused with the statistical anomaly score:

$$
\text{Alert Score}_t = \alpha \cdot \text{sigmoid}(z_t - z^{*}) + (1-\alpha)\cdot p(\text{stress}\mid \text{text}_t)
$$

```
ANOMALY-TO-ALERT PIPELINE

  Spread Time Series ──▶ EWMA Mahalanobis ──┐
                                             ├──▶ Alert Score ──▶ Ranked
  Overnight Commentary ──▶ LLM Classifier ───┘                  Trader Queue
       (RAG over desk           (fine-tuned or
        precedent notes)         prompted, JSON-schema output)
```

```python
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

class FundingSpreadMonitor:
    def __init__(self, halflife=10):
        self.halflife = halflife
        self.mcd = MinCovDet(support_fraction=0.75)
        self.iforest = IsolationForest(n_estimators=300, contamination=0.02, random_state=7)

    def fit(self, X_hist: np.ndarray):
        self.mcd.fit(X_hist)
        self.iforest.fit(X_hist)
        return self

    def score(self, x_t: np.ndarray) -> dict:
        maha = self.mcd.mahalanobis(x_t.reshape(1, -1))[0]
        iso = -self.iforest.score_samples(x_t.reshape(1, -1))[0]  # higher = more anomalous
        z = maha ** 0.5
        return {"mahalanobis_z": float(z), "isolation_score": float(iso)}

def fuse_with_text_signal(stat_score: dict, p_stress_text: float, alpha: float = 0.6) -> float:
    stat_sig = 1 / (1 + np.exp(-(stat_score["mahalanobis_z"] - 3.0)))  # sigmoid centered at z*=3
    return float(alpha * stat_sig + (1 - alpha) * p_stress_text)
```

## Human-in-the-Loop Design

Every alert above threshold routes to a trader queue with (a) the driving spread series chart, (b) the top-3 SHAP-attributed features from the statistical model, (c) the LLM's one-paragraph rationale with citations to the source commentary — **never** an auto-executed action. Alert precision/recall tracked weekly against trader-labeled outcomes; threshold retuned via a Neyman-Pearson-style cost ratio (false-negative cost ≫ false-positive cost on this desk).

[🔝 Back to Top](#table-of-contents)

---
---

# P4 · Prime Balance & Utilization Forecasting (Deep Learning)

**Open with the business framing:**
> "Balance sheet is the scarcest resource in Liquid Financing — every quarter-end, Treasury wants a forecast of client balances and utilization so the desk isn't caught flat-footed on RWA/LCR optimization. This is a genuinely sequential problem: balances have daily seasonality, month-end/quarter-end window-dressing effects, and multi-week momentum — which is exactly the regime where RNN/LSTM/GRU beat both ARIMA and gradient-boosted trees on a rolling basis, per the JD's DL requirement."

## Problem Statement

Forecast daily aggregate client financing balances and utilization ratio, 1–20 business days ahead, by asset class and by top-20 client concentration, with explicit quarter-end spike/reversal modeling for Treasury's RWA and LCR planning.

## Architecture

$$
h_t = \text{GRU}(x_t, h_{t-1}), \qquad \hat{y}_{t+1:t+H} = W_o \, h_T + b_o
$$

Sequence-to-sequence GRU encoder-decoder (GRU chosen over LSTM as primary due to fewer parameters / faster retrain on a daily cadence, with an LSTM variant kept as a challenger given its stronger long-memory gating for the quarter-end effect):

```
ENCODER-DECODER FORECAST ARCHITECTURE

  x_{t-60}...x_t  ──▶ [GRU Encoder, 2 layers, 64 units] ──▶ context vector h_T
                                                                │
                          quarter-end / holiday / dividend      │
                          calendar embeddings  ─────────────────┤
                                                                 ▼
                                            [GRU Decoder, 2 layers] ──▶ ŷ_{t+1..t+20}
                                                                 │
                                                   quantile heads (P10/P50/P90)
```

**Quarter-end effect modeled explicitly** via a learned calendar embedding concatenated at every decoder step — because window-dressing balance reductions are a *known, calendar-driven* pattern the model should not have to rediscover purely from raw magnitude, and Model Risk expects an explicit, inspectable treatment of a known seasonal effect rather than a black-box hope that the network finds it.

$$
\mathcal{L} = \sum_{q\in\{0.1,0.5,0.9\}} \sum_{h=1}^{H} \rho_q\big(y_{t+h} - \hat{y}_{t+h}^{(q)}\big), \quad \rho_q(u)=u\cdot(q-\mathbb{1}[u<0])
$$

Pinball (quantile) loss across three quantile heads gives Treasury a P10/P50/P90 balance forecast band, not a false-precision point estimate.

```python
import torch
import torch.nn as nn

class BalanceForecastGRU(nn.Module):
    """
    Seq2seq GRU with calendar-embedding conditioning and quantile heads,
    for prime balance / utilization forecasting.
    """
    def __init__(self, n_features, hidden=64, n_layers=2, calendar_dim=8, horizon=20):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, n_layers, batch_first=True)
        self.calendar_emb = nn.Embedding(10, calendar_dim)  # e.g. day-type: normal/qtr-end/holiday...
        self.decoder_cell = nn.GRUCell(hidden + calendar_dim, hidden)
        self.horizon = horizon
        self.heads = nn.ModuleDict({
            q: nn.Linear(hidden, 1) for q in ["p10", "p50", "p90"]
        })

    def forward(self, x_hist, calendar_codes_future):
        _, h_n = self.encoder(x_hist)          # h_n: (n_layers, B, hidden)
        h = h_n[-1]                             # (B, hidden)
        outs = {"p10": [], "p50": [], "p90": []}
        for t in range(self.horizon):
            cal = self.calendar_emb(calendar_codes_future[:, t])
            h = self.decoder_cell(torch.cat([h, cal], dim=-1), h)
            for q, head in self.heads.items():
                outs[q].append(head(h))
        return {q: torch.cat(v, dim=1) for q, v in outs.items()}

def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return torch.mean(torch.max(q * diff, (q - 1) * diff))
```

## Baselines the DL model must beat (walk-forward, not shuffled CV)

| Model | Role |
|---|---|
| Seasonal-Naive (t-5 / t-20) | Sanity floor |
| SARIMAX with quarter-end dummy | Classical time-series benchmark |
| LightGBM on lagged features + calendar dummies | Strong tabular benchmark — DL must beat this to justify production complexity |
| LSTM (single-layer, no calendar embedding) | Ablation — proves the calendar-embedding design choice earns its keep |
| **GRU + calendar embedding + quantile heads (proposed)** | Production candidate |

**Say it out loud in the room:** *"I don't reach for an LSTM/GRU by default — I make it beat a well-tuned LightGBM on walk-forward MAPE and pinball loss first, on this exact series, before I'd argue for the added production complexity of a PyTorch serving path over a tree model artifact."* This is the answer that survives a QR panel — DL as *justified*, not *assumed*.

[🔝 Back to Top](#table-of-contents)

---
---

# P5 · RAG Financing-Desk Copilot (GenAI / LLM)

**Open with the business framing:**
> "The JD explicitly calls out fine-tuning, prompt engineering, RAG, model evaluation, and BMC/inference infrastructure. This project is the direct answer: a retrieval-augmented copilot that lets a financing trader ask 'why did the XYZ borrow fee spike Tuesday' or 'what's our current haircut policy for BBB financials collateral' and get a grounded, cited answer sourced from desk notes, policy docs, and the P1–P4 model outputs — not a hallucinated one."

## Problem Statement

Build a retrieval-augmented generation system over (a) internal financing policy/haircut-grid documents, (b) daily desk commentary, (c) structured outputs from the fee/margin/anomaly/balance models above, that answers trader queries with citations and refuses ungrounded questions.

## Architecture

```
                     ┌─────────────────────────────┐
   Trader Query ───▶ │  Query Router (intent class) │
                     └──────────────┬──────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   Structured-data tool      Document retriever          Model-output tool
   (SQL over fee/margin/     (hybrid BM25 + dense         (calls P1-P4 model
    balance warehouse)        embedding, reranked)         inference endpoints)
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                      ▼
                          LLM synthesis (grounded,
                          JSON-schema constrained,
                          citation-required)
                                      │
                                      ▼
                          Answer + citations + confidence
                          + "insufficient evidence" fallback
```

## Retrieval — Hybrid Dense + Sparse with Reranking

$$
\text{score}(q,d) = \lambda \cdot \text{BM25}(q,d) + (1-\lambda) \cdot \cos\!\big(E(q), E(d)\big)
$$

followed by a cross-encoder reranker on the top-50 candidates; $\lambda \approx 0.4$ tuned on a held-out set of 200 trader-authored queries with human-labeled relevant documents (recall@10 as the retrieval KPI, not just top-1 accuracy, because financing policy answers often need 2–3 supporting clauses).

## Fine-Tuning vs. Prompting Decision (explicit trade-off, per JD "evaluate and articulate trade-offs")

```
DIMENSION                PROMPT-ENGINEERED (base model + RAG)   FINE-TUNED (LoRA on desk corpus)
────────────────────     ──────────────────────────────────    ──────────────────────────────
Time to deploy            Days                                    Weeks (data curation + eval)
Domain jargon fluency      Good with few-shot examples            Best — learns "GC", "specialness",
                                                                    "recall", "cash-driven" natively
Cost per query             Higher (longer context, few-shot)      Lower (shorter prompts)
Governance / auditability  Easier — prompt + retrieved docs        Harder — need eval harness +
                            are the full audit trail                weight diff documentation
Update cadence              Instant (swap retrieved docs)          Retrain cycle required

RECOMMENDATION: Start prompt-engineered + RAG for month 1-2 to establish eval harness
and gather production query logs; graduate the intent-classification and jargon-normalization
sub-tasks to a small fine-tuned (LoRA) model once ≥5k labeled queries exist, keep the
final-answer synthesis on the frontier base model with RAG for auditability.
```

## Evaluation Harness (this is what separates a toy RAG demo from production)

$$
\text{Faithfulness} = \frac{\#\{\text{claims in answer supported by retrieved context}\}}{\#\{\text{claims in answer}\}}
$$

| Metric | Method |
|---|---|
| Retrieval Recall@10 | Labeled query/doc relevance set |
| Faithfulness / groundedness | LLM-as-judge + spot-audited by desk SMEs, monthly |
| Answer correctness | Human eval against a 100-query golden set, refreshed quarterly |
| Refusal precision | % of out-of-scope queries correctly declined vs. hallucinated |
| Latency (p50/p95) | SLA target: p95 < 4s for a trader-facing tool during market hours |

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float

def hybrid_retrieve(query: str, bm25_index, dense_index, lam: float = 0.4, k: int = 50) -> List[RetrievedChunk]:
    bm25_hits = bm25_index.search(query, top_k=k)
    dense_hits = dense_index.search(query, top_k=k)
    fused = {}
    for h in bm25_hits:
        fused[h.doc_id] = lam * h.score
    for h in dense_hits:
        fused[h.doc_id] = fused.get(h.doc_id, 0.0) + (1 - lam) * h.score
    ranked_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [RetrievedChunk(text=bm25_index.get_text(i), source=bm25_index.get_source(i), score=fused[i])
            for i in ranked_ids]

SYSTEM_PROMPT = """You are a Liquid Financing desk copilot. Answer ONLY using the provided
context blocks. Every factual claim must carry a [source_id] citation. If the context does
not contain enough information to answer confidently, respond exactly with:
"INSUFFICIENT_EVIDENCE: <what is missing>". Never invent fee levels, haircuts, or policy terms."""

def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks[:8])
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nJSON_SCHEMA: " \
           '{"answer": str, "citations": [str], "confidence": "high|medium|low"}'
```

## Inference Infrastructure (per JD: "configure inference infrastructure (BMC servers)")

- On-prem/BMC-hosted open-weight base model for anything touching non-public desk data (data residency + Model Risk requirement); external frontier API only for public-market-context queries with no client-identifying data.
- Request routing layer enforces PII/entitlement filtering *before* the prompt is assembled, not after generation.
- Autoscaled inference pool sized off P95 trading-hours query volume; async batch mode for EOD report generation to keep interactive latency SLAs protected.

[🔝 Back to Top](#table-of-contents)

---
---

# Cross-Project Production Standards

```
STAGE                  STANDARD APPLIED ACROSS ALL 5 PROJECTS
──────────────────     ───────────────────────────────────────────────────────
Validation split        Walk-forward / expanding window ONLY — never random
                         K-fold on time series (leakage across autocorrelated obs)
Champion/Challenger      Minimum 2 full quarters shadow-mode before promotion
Model Risk pack          Purpose, conceptual soundness, developmental evidence,
                         outcomes analysis (backtests), ongoing monitoring plan
Monitoring               Feature PSI, prediction drift, backtest breach counts,
                         auto-fallback to prior champion on threshold breach
Explainability           SHAP for tree models; attention/attribution + faithfulness
                         eval for the RAG copilot; coefficient sign review for linear
Code / release           Feature store parity between research & production, unit
                         tests on feature pipelines, canary rollout on 5% of traffic
```

[🔝 Back to Top](#table-of-contents)

---
---

# Quick-Reference Equation Sheet

```
══════════════════════════════════════════════════════════════════════════════
P1 — FEE FORECASTING
══════════════════════════════════════════════════════════════════════════════
Elastic Net:      β̂ = argmin ||y - Xβ||² + λ1||β||₁ + λ2||β||₂²
GBM stage update: h_m = argmin_h Σ L(y, F_{m-1}(x) + h(x))
Event-decay OU:   f_t = μ + φ(f_{t-1}-μ) + Σ κ_j·1[event]·e^{-η_j(t-t_j)} + ε_t

══════════════════════════════════════════════════════════════════════════════
P2 — MARGIN / HAIRCUT
══════════════════════════════════════════════════════════════════════════════
Log-linear haircut:   log h_i = β0 + β1 logσ_i + β2 log(ADV_i) + β3 Corr_i + ...
Portfolio IM:          IM = Σ h_i|q_i|P_i + g_θ(q, Σ, concentration)
Constraint:             Pr(Shortfall_5d > h·V) ≤ 1%

══════════════════════════════════════════════════════════════════════════════
P3 — ANOMALY DETECTION
══════════════════════════════════════════════════════════════════════════════
Mahalanobis:      z_t = (x_t - μ_t)ᵀ Σ_t⁻¹ (x_t - μ_t)
Isolation Forest:  s(x) = 2^(-E[h(x)]/c(n))
Fused alert:       Alert_t = α·sigmoid(z_t - z*) + (1-α)·p(stress|text_t)

══════════════════════════════════════════════════════════════════════════════
P4 — BALANCE FORECASTING (DL)
══════════════════════════════════════════════════════════════════════════════
GRU state:         h_t = GRU(x_t, h_{t-1})
Pinball loss:      L = Σ_q Σ_h ρ_q(y_{t+h} - ŷ_{t+h}^(q)),  ρ_q(u)=u(q-1[u<0])

══════════════════════════════════════════════════════════════════════════════
P5 — RAG COPILOT (GenAI)
══════════════════════════════════════════════════════════════════════════════
Hybrid retrieval:  score(q,d) = λ·BM25(q,d) + (1-λ)·cos(E(q),E(d))
Faithfulness:      #{claims supported by context} / #{claims in answer}
══════════════════════════════════════════════════════════════════════════════
```

[🔝 Back to Top](#table-of-contents)

---

*Last updated: July 2026 · Take-Home Research Portfolio — AI/ML Modeler, Liquid Financing (Barclays)*

[↩️ Back to README.md](../README.md)
