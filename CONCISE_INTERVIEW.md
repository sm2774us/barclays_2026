# Barclays — AI/ML Modeler, Liquid Financing — MD Screening Playbook
### Interviewer: Rishi D. | MD, Global Markets — Electronification, eTrading & AI/ML | Prime Services, Financing & Delta One
#### 20 Questions × 30-Minute Format · 5 Behavioral + 10 Technical + 5 Domain

> **Delivery philosophy:** Every answer follows *Feynman intuition first, rigorous math/ML second, production code third*. Rishi runs electronification and AI/ML for Prime, Financing & Delta One — he has built teams that ship, not teams that publish. He will discount any answer that stays academic and reward any answer that lands on "here is exactly how this changes P&L, risk, or desk workflow." Never name a technique you can't derive or code from scratch in the follow-up.

---
---

## ⏱️ 30-Minute Question Budget

```
SECTION       QUESTIONS   TIME BUDGET     RISHI'S LENS
────────────  ──────────  ──────────────  ──────────────────────────────────────────
BEHAVIORAL    B1 – B5      8 min          Can this person ship in a bank, not a lab?
TECHNICAL     T1 – T10    15 min          Depth on demand — ML, DL, GenAI, LoRA/RAG/Agents
DOMAIN        D1 – D5      7 min          Do they actually understand Financing/Delta One?
```

> **Priority rule:** In a 30-minute MD screen, expect 6-8 questions actually asked live, chosen adaptively based on your first two answers. Front-load strength: nail B1, T1, T5/T6/T7 (GenAI trio), and D1. Everything else is a depth-probe he'll pull on if you signal mastery.

---

## Table of Contents

### 🧭 BEHAVIORAL
- [B1 · Model Ideation to Production, End-to-End](#b1--model-ideation-to-production-end-to-end)
- [B2 · Justifying ML vs DL vs GenAI Trade-offs to Business Stakeholders](#b2--justifying-ml-vs-dl-vs-genai-trade-offs-to-business-stakeholders)
- [B3 · Prioritizing Within a 50+ Item Roadmap](#b3--prioritizing-within-a-50-item-roadmap)
- [B4 · Partnering With a 10-15 Person Engineering Team](#b4--partnering-with-a-10-15-person-engineering-team)
- [B5 · A Production Model Failure and Its Aftermath](#b5--a-production-model-failure-and-its-aftermath)

### 🧠 TECHNICAL
- [T1 · Time Series Forecasting for Pricing — Classical vs Deep Learning](#t1--time-series-forecasting-for-pricing--classical-vs-deep-learning)
- [T2 · Regression Family — OLS, LASSO, Elastic Net for Financing Rates](#t2--regression-family--ols-lasso-elastic-net-for-financing-rates)
- [T3 · Tree-Based Models vs Neural Nets — Explainability Under Model Risk](#t3--tree-based-models-vs-neural-nets--explainability-under-model-risk)
- [T4 · RNN, LSTM, GRU — Mechanics and When Recurrence Still Wins](#t4--rnn-lstm-gru--mechanics-and-when-recurrence-still-wins)
- [T5 · Fine-Tuning LLMs — Full FT vs LoRA vs QLoRA](#t5--fine-tuning-llms--full-ft-vs-lora-vs-qlora)
- [T6 · RAG Architecture for Financing Desk Knowledge](#t6--rag-architecture-for-financing-desk-knowledge)
- [T7 · Agentic AI — ReAct, Tool Use, Multi-Agent Orchestration](#t7--agentic-ai--react-tool-use-multi-agent-orchestration)
- [T8 · Model Evaluation — Classical Metrics vs LLM Evaluation](#t8--model-evaluation--classical-metrics-vs-llm-evaluation)
- [T9 · Prompt Engineering and Inference Optimization on BMC Servers](#t9--prompt-engineering-and-inference-optimization-on-bmc-servers)
- [T10 · Deep Learning Fundamentals — Backprop, Optimizers, Regularization](#t10--deep-learning-fundamentals--backprop-optimizers-regularization)

### 🏦 DOMAIN
- [D1 · Liquid Financing, Delta One & Prime — Where AI/ML Actually Adds Value](#d1--liquid-financing-delta-one--prime--where-aiml-actually-adds-value)
- [D2 · Repo Rate Forecasting — GC vs Specials](#d2--repo-rate-forecasting--gc-vs-specials)
- [D3 · ML for Margin, Haircut Optimization & Counterparty Risk](#d3--ml-for-margin-haircut-optimization--counterparty-risk)
- [D4 · State of Research — Foundation Models for Markets & Time Series](#d4--state-of-research--foundation-models-for-markets--time-series)
- [D5 · Model Governance — SR 11-7 Meets LLMs](#d5--model-governance--sr-11-7-meets-llms)

- **[Quick-Reference Equation & Architecture Sheet](#quick-reference-equation--architecture-sheet)**

[🔝 Back to Top](#table-of-contents)

---
---

# 🧭 BEHAVIORAL

---

## B1 · Model Ideation to Production, End-to-End

**Open with the intuition (15 seconds):**
> "A model that lives in a notebook is a research artifact. A model that lives in the firm's production codebase, with monitoring and a rollback plan, is a business capability. My job is closing that gap — and closing it fast, because the JD says the roadmap is already built with 10 prioritized problem statements. Nobody needs another proof of concept."

**Structured answer (STAR, compressed):**
> "At my last seat, financing desk asked for a model to flag anomalous securities-lending fee quotes before they hit the book. **Situation:** manual review caught maybe 60% of mispriced quotes, always after the fact. **Task:** build and ship a real-time flag, no analyst bottleneck. **Action:** I started with the simplest defensible baseline — a gradient-boosted classifier on quote-vs-peer-basket z-scores, not a neural net, because false positives cost trader trust and I needed something explainable on day one. I validated against 18 months of fee history, worked with engineering to wrap it as a low-latency scoring service, and instrumented it with drift monitors on the input distributions. **Result:** it caught 91% of anomalies with fewer than 3% false-positive alerts, and it's still running. The lesson I carry into this Barclays role: ship the simplest model that clears the risk bar first, then earn the right to add complexity — LSTM, GNN, whatever the residual error demands."

**Why this lands with an MD who ships:** it shows (1) restraint — not reaching for deep learning by default, (2) production discipline — monitoring, rollback, drift, (3) a quantifiable result, (4) a generalizable principle he can picture applied to *his* 10 prioritized problem statements.

[🔝 Back to Top](#table-of-contents)

---
---

## B2 · Justifying ML vs DL vs GenAI Trade-offs to Business Stakeholders

**Open with the intuition:**
> "Traders don't care about my architecture — they care about latency, explainability, and whether it's right more often than they are. My job is translating 'why gradient boosting beats a transformer here' into 'why this won't blow up your P&L and won't take 200ms when you need an answer in 5.'"

**Structured answer:**
> "I was asked whether we should use an LLM to summarize overnight financing-cost commentary for the morning trader huddle, versus a templated NLG system. I laid out three axes for the business: **accuracy risk** — an LLM can hallucinate a number, a templated system cannot; **latency** — sub-second matters at 6:45am, LLM inference doesn't guarantee that without careful engineering; **maintenance cost** — templates need a rules engineer, LLMs need prompt/eval iteration. I recommended a hybrid: deterministic templates pull hard numbers (rates, spreads, volumes) from the data warehouse, and an LLM only handles the *prose synthesis* layer, constrained by RAG grounding so it can't invent a rate that isn't in the retrieved context. That let the desk get natural-language color without exposing the firm to hallucinated numbers in a document traders act on. The meta-skill is knowing GenAI is a language-and-reasoning layer, not a numerical-guarantee layer — and structuring the system so each component only does what it's provably good at."

**Decision framework I'd show on a whiteboard:**

```
                     LOW STAKES / HIGH VARIANCE OK        HIGH STAKES / NEEDS GUARANTEE
                    ┌───────────────────────────┐        ┌───────────────────────────┐
STRUCTURED,         │  Gradient Boosting /      │        │  OLS / Elastic Net —      │
TABULAR DATA        │  Random Forest            │        │  interpretable, auditable │
                    └───────────────────────────┘        └───────────────────────────┘

SEQUENTIAL /        ┌───────────────────────────┐        ┌───────────────────────────┐
TIME-DEPENDENT      │  LSTM / Temporal Fusion   │        │  ARIMA-GARCH / state-space│
                    │  Transformer              │        │  with confidence bands    │
                    └───────────────────────────┘        └───────────────────────────┘

UNSTRUCTURED /      ┌──────────────────────────────┐        ┌───────────────────────────┐
LANGUAGE            │  Fine-tuned / prompted LLM   │        │  LLM + RAG grounding +    │
                    │  for drafting, classification│        │  deterministic guardrails │
                    └──────────────────────────────┘        └───────────────────────────┘
```

[🔝 Back to Top](#table-of-contents)

---
---

## B3 · Prioritizing Within a 50+ Item Roadmap

**Open with the intuition:**
> "Fifty problem statements is a portfolio, not a to-do list. I run it like one: expected P&L or risk-reduction impact on one axis, engineering/data cost on the other, and I make the frontier visible so business stakeholders argue about priorities with numbers, not politics."

**Structured answer:**
> "In a prior role I inherited a backlog of 30+ candidate ML use cases across three trading desks. I built a simple 2x2 — impact estimate (bps saved, hours of analyst time recovered, or risk events prevented) against implementation cost (data readiness, model complexity, integration surface) — and scored every item with the relevant desk head in a 30-minute working session. That collapsed 30 items to 4 that made the quarter. The two things I insist on: first, a scoring rubric everyone can see, so 'my project' politics can't dominate; second, revisiting the ranking every 6-8 weeks because data availability and business priorities both shift. For 50 identified statements with 10 already prioritized here, I'd want to understand the scoring Rishi's team already used, validate it against what's shipped versus stalled, and slot in as someone who executes the top of that list fast rather than re-litigating it."

[🔝 Back to Top](#table-of-contents)

---
---

## B4 · Partnering With a 10-15 Person Engineering Team

**Open with the intuition:**
> "A model is maybe 20% of the system that goes to production — data pipelines, serving infra, monitoring, and rollback are the other 80%, and that's engineering's world, not mine. My job is to hand them something they can actually build around: a clear contract, not a notebook."

**Structured answer:**
> "The pattern that's worked for me: I define the model as a service contract first — inputs, outputs, latency SLA, retraining cadence — before I even finalize the architecture, so engineering can build the surrounding infrastructure in parallel instead of waiting on me. I write my training code to Google's Python style guide from day one, because code that looks like a research script doesn't survive a code review from a production engineering team, and that friction kills timelines. I also push hard for a shared staging environment where I can validate the deployed model bit-for-bit matches my training-time output — model drift introduced by a serialization or feature-computation mismatch between training and serving is one of the most common and most preventable failure modes in bank ML, and it's purely a collaboration problem, not a modeling problem."

[🔝 Back to Top](#table-of-contents)

---
---

## B5 · A Production Model Failure and Its Aftermath

**Open with the intuition:**
> "Every model fails eventually — the difference between a good and bad modeler is whether the failure is caught by your monitoring or by a client complaint. I want to talk about a failure I caught, and what changed afterward, because that's the model-risk maturity a bank actually wants to hear."

**Structured answer:**
> "A time-series forecasting model I owned for short-term funding cost had a silent regime-break during a liquidity event — the training distribution simply didn't contain a scenario like it, and the point forecast stayed confidently wrong for two days before my drift monitor on residual autocorrelation flagged it. **What I did:** froze the model to a fallback heuristic (last-observed-value plus a widened band) within an hour of the alert, ran a root-cause with the desk, and rebuilt the loss function to include a regime-classification gate — the model now checks whether current market conditions resemble its training distribution before trusting its own point estimate, and defers to a conservative interval otherwise. **What changed structurally:** I pushed for out-of-distribution detection to be a standing requirement for any model I ship, not an afterthought — measured via Mahalanobis distance on the input feature vector against the training covariance. That's the discipline I'd bring to Liquid Financing: assume the tail event is coming, build the model to know when it doesn't know."

[🔝 Back to Top](#table-of-contents)

---
---

# 🧠 TECHNICAL

---

## T1 · Time Series Forecasting for Pricing — Classical vs Deep Learning

**Open with the intuition (15 seconds):**
> "Classical time-series models assume the past has a stable statistical structure — a mean it reverts to, a volatility process that clusters. Deep learning models assume nothing about structure and instead learn it from data, at the cost of needing much more of it. For financing rates and pricing, I choose based on how much history I actually have and how nonlinear the driving factors are."

**The classical baseline:**

$$\text{ARIMA}(p,d,q): \quad \left(1 - \sum_{i=1}^{p}\phi_i L^i\right)(1-L)^d y_t = \left(1 + \sum_{j=1}^{q}\theta_j L^j\right)\varepsilon_t$$

$$\text{GARCH}(1,1): \quad \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Say it out loud:** *"ARIMA captures the autocorrelation in the level and differences of a series. GARCH separately models the fact that volatility clusters — calm periods and turbulent periods each persist. For financing spreads, I typically model the level with an ARIMA-X that includes exogenous drivers like collateral scarcity and repo GC, and model the residual volatility with GARCH so my confidence bands widen appropriately around known volatile periods like quarter-end and triparty settlement windows."*

---

**GC** - **G**eneral **C**ollateral

In the financial world, **GC** stands for **G**eneral **C**ollateral. In a repurchase agreement (repo) market, it refers to a broad basket of highly liquid, interchangeable securities (like U.S. Treasuries) used as collateral for short-term cash loans.

Here are the key details about **GC**: 

- **Interchangeable Assets:** The lender of cash is essentially indifferent to which specific security they receive, as long as it falls within the agreed-upon GC basket (e.g., any standard Treasury bill or note).
- **Cash-Driven:** Because the securities act as near-perfect substitutes, GC repo rates are driven by the overall supply and demand for cash rather than the demand for a specific, hard-to-find bond.
- **GCF Repo:** You might also hear this referred to as General Collateral Finance (GCF), a streamlined version where the specific collateral is not designated until the end of the trading day, vastly improving market efficiency.

Understanding **GC** is essential because the **GC repo rate** serves as a *vital benchmark for the near-risk-free cost of borrowing cash in the financial system*.

---

**The deep learning case — when it earns its complexity:**

```
DECISION TREE — PRICING/RATE FORECASTING MODEL CHOICE

  How much clean history do I have?
  ├── < 2 years / sparse, single-asset series
  │     → ARIMA-GARCH-X. DL will overfit; classical model is auditable and fast.
  │
  ├── 2-5 years, multiple correlated series (cross-asset financing book)
  │     → Gradient boosting on engineered lag/rolling features, OR
  │       LSTM/GRU if nonlinear cross-series interaction dominates.
  │
  └── 5+ years, many correlated series, need multi-horizon forecasts
        → Temporal Fusion Transformer / N-BEATS — attention over static
          (counterparty, asset class) and time-varying covariates,
          native multi-horizon quantile output for risk bands.
```

**Production-grade code — a regime-aware ensemble forecaster:**

```python
"""Regime-aware financing spread forecaster.

Combines a classical ARIMA-GARCH baseline with an LSTM residual
corrector, gated by an out-of-distribution check on the input
feature vector. Falls back to the classical model when the market
regime is outside the LSTM's training distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import torch
from torch import nn


class RegimeGatedForecaster:
    """Forecasts a financing spread with a classical/DL ensemble.

    Attributes:
        arima_order: The (p, d, q) order for the ARIMA mean model.
        ood_threshold: Mahalanobis distance threshold beyond which the
            LSTM correction is disabled and only the classical
            forecast is trusted.
    """

    def __init__(
        self,
        arima_order: tuple[int, int, int] = (2, 1, 2),
        ood_threshold: float = 3.5,
    ) -> None:
        self.arima_order = arima_order
        self.ood_threshold = ood_threshold
        self._lstm: nn.Module | None = None
        self._train_mean: np.ndarray | None = None
        self._train_cov_inv: np.ndarray | None = None

    def fit(self, series: np.ndarray, features: np.ndarray) -> None:
        """Fits the classical baseline and the LSTM residual model.

        Args:
            series: 1D array of the target financing spread, in bps.
            features: 2D array (n_obs, n_features) of exogenous
                drivers (e.g. collateral scarcity score, GC rate,
                quarter-end indicator).
        """
        self._arima_res = ARIMA(series, order=self.arima_order).fit()
        garch = arch_model(self._arima_res.resid, vol="GARCH", p=1, q=1)
        self._garch_res = garch.fit(disp="off")

        residuals = self._arima_res.resid
        self._lstm = self._build_lstm(n_features=features.shape[1])
        self._train_lstm(self._lstm, features, residuals)

        self._train_mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False) + 1e-6 * np.eye(features.shape[1])
        self._train_cov_inv = np.linalg.inv(cov)

    def predict(
        self, latest_features: np.ndarray, horizon: int = 1
    ) -> dict[str, float]:
        """Produces a point forecast and a confidence band.

        Args:
            latest_features: 1D array of the most recent feature
                vector used for the OOD check and LSTM correction.
            horizon: Forecast horizon in periods.

        Returns:
            A dict with 'point_forecast_bps', 'lower_bps', 'upper_bps',
            and 'regime_flag' ('IN_DISTRIBUTION' or 'OUT_OF_DISTRIBUTION').
        """
        base_forecast = self._arima_res.forecast(steps=horizon)[-1]
        vol_forecast = np.sqrt(
            self._garch_res.forecast(horizon=horizon).variance.values[-1, -1]
        )

        ood_distance = distance.mahalanobis(
            latest_features, self._train_mean, self._train_cov_inv
        )
        regime_flag = (
            "OUT_OF_DISTRIBUTION"
            if ood_distance > self.ood_threshold
            else "IN_DISTRIBUTION"
        )

        correction = 0.0
        if regime_flag == "IN_DISTRIBUTION" and self._lstm is not None:
            with torch.no_grad():
                x = torch.tensor(latest_features, dtype=torch.float32)
                correction = self._lstm(x.unsqueeze(0)).item()

        point_forecast = base_forecast + correction
        band_width = vol_forecast * (2.5 if regime_flag == "OUT_OF_DISTRIBUTION" else 1.65)

        return {
            "point_forecast_bps": float(point_forecast),
            "lower_bps": float(point_forecast - band_width),
            "upper_bps": float(point_forecast + band_width),
            "regime_flag": regime_flag,
        }

    @staticmethod
    def _build_lstm(n_features: int, hidden_size: int = 32) -> nn.Module:
        return nn.Sequential(
            nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True),
        )

    def _train_lstm(
        self, model: nn.Module, features: np.ndarray, residuals: np.ndarray
    ) -> None:
        # Training loop omitted for brevity — standard Adam + MSE loss
        # on (features -> ARIMA residual) pairs with early stopping on
        # a held-out validation window.
        pass
```

**Answer for Rishi:** *"For pricing in a financing book, I default to the classical ARIMA-GARCH-X baseline because it's auditable and I can explain every coefficient to a risk committee. I only add an LSTM or TFT layer as a residual corrector — on top of, not instead of, the classical model — and I gate it with an out-of-distribution check so the model 'knows when it doesn't know' and falls back to the interpretable baseline exactly when it matters most: in a liquidity event, which is precisely when a black-box DL model is most likely to be wrong and least likely to say so."*

[🔝 Back to Top](#table-of-contents)

---
---

## T2 · Regression Family — OLS, LASSO, Elastic Net for Financing Rates

**Open with the intuition:**
> "OLS asks every feature to justify its coefficient with no penalty for being greedy. In financing data, features are highly collinear — GC rate, term, collateral quality, and counterparty credit spread all move together — so OLS coefficients become unstable and flip sign from month to month. LASSO and Elastic Net add a penalty that forces the model to be disciplined about which features it actually trusts."

**The math:**

$$\hat{\beta}_{\text{OLS}} = \arg\min_\beta \; \|y - X\beta\|_2^2$$

$$\hat{\beta}_{\text{LASSO}} = \arg\min_\beta \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

$$\hat{\beta}_{\text{Elastic Net}} = \arg\min_\beta \; \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

**Say it out loud:** *"LASSO's L1 penalty pushes weak coefficients exactly to zero — it does feature selection for you. That's powerful but dangerous with correlated features: LASSO will arbitrarily pick one of two correlated financing-rate drivers and zero out the other, which hurts interpretability when a risk committee asks 'why does the model care about repo GC but not collateral scarcity, when they're basically the same signal.' Elastic Net's L2 term fixes that — it tends to keep correlated features together, shrinking their coefficients as a group rather than arbitrarily choosing a winner. For a financing-rate model with 20-30 collinear macro and collateral features, Elastic Net is almost always my default over pure LASSO."*

```
BIAS-VARIANCE TRADEOFF ACROSS THE REGRESSION FAMILY

  MODEL           BIAS       VARIANCE    HANDLES COLLINEARITY   INTERPRETABILITY
  ─────────────   ────────   ─────────   ─────────────────────  ─────────────────
  OLS             Lowest     Highest     Poor — unstable coefs  High but unstable
  Ridge (L2)      Medium     Medium      Good — shrinks group   High
  LASSO (L1)      Medium     Lower       Poor — picks 1 of many  High but arbitrary
  Elastic Net     Medium     Lowest      Good — grouped shrink  High and stable
```

**Code — cross-validated Elastic Net for repo spread modeling:**

```python
"""Elastic Net model for overnight repo spread prediction."""

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def build_repo_spread_model(l1_ratio_grid: list[float] | None = None) -> Pipeline:
    """Builds a standardized, cross-validated Elastic Net pipeline.

    Args:
        l1_ratio_grid: Candidate L1/L2 mixing ratios to search over.
            Defaults to a grid spanning ridge-like to lasso-like.

    Returns:
        An unfitted sklearn Pipeline ready for `.fit(X, y)`.
    """
    l1_ratio_grid = l1_ratio_grid or [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    model = ElasticNetCV(
        l1_ratio=l1_ratio_grid,
        n_alphas=100,
        cv=5,
        max_iter=10_000,
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("elastic_net", model)])
```

**Answer for Rishi:** *"I'd reach for OLS only as a diagnostic — to see raw signal before regularization — and go straight to Elastic Net for anything that ships, because financing-rate drivers are collinear by construction and I need stable, explainable coefficients a risk committee can sign off on quarter over quarter."*

[🔝 Back to Top](#table-of-contents)

---
---

## T3 · Tree-Based Models vs Neural Nets — Explainability Under Model Risk

**Open with the intuition:**
> "On tabular financial data with a few hundred to a few thousand features, gradient-boosted trees usually beat neural nets on accuracy *and* give me a native path to explainability via SHAP. Neural nets earn their place when the data has spatial or sequential structure trees can't exploit — images, text, time-ordered sequences — or when the dataset is large enough that representation learning beats hand-engineered features."

**The math of a gradient-boosted ensemble:**

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x), \qquad h_m = \arg\min_h \sum_i L\!\left(y_i, F_{m-1}(x_i) + h(x_i)\right)$$

**Say it out loud:** *"Each new tree is fit to the negative gradient of the loss with respect to the current ensemble's predictions — it's literally gradient descent, but in function space instead of parameter space. Every tree is a weak learner correcting the errors of everything before it, and the learning rate η controls how aggressively we trust each correction."*

**SHAP for model-risk-grade explainability:**

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\Big[f(S \cup \{i\}) - f(S)\Big]$$

**Say it out loud:** *"SHAP values are Shapley values from cooperative game theory applied to features — each feature's contribution is its average marginal impact across every possible ordering of features being 'added' to the prediction. That's exactly what a model validation team wants: a mathematically fair, additive decomposition of 'why did the model output this number,' feature by feature, for every single prediction — not just a global feature-importance ranking."*

```
WHEN TREES BEAT NETS (AND VICE VERSA) — TABULAR FINANCING DATA

  DATA CHARACTERISTIC              PREFER              WHY
  ────────────────────────────    ─────────────────    ──────────────────────────────
  < 100k rows, tabular,           XGBoost / LightGBM    Trees handle mixed types,
  mixed categorical+numeric                             missing values, need no scaling
  
  Need row-level explainability   XGBoost + SHAP        Native TreeSHAP is exact and
  for model validation                                   fast (polynomial, not exp.)
  
  Sequential/time-ordered,        LSTM / TFT /          Trees ignore order unless you
  long lookback dependencies      Transformer            hand-engineer every lag
  
  Very large data, want           Neural net             Representation learning beats
  automatic feature learning                              manual feature engineering
  
  Need calibrated uncertainty     Quantile GBM or        Both can do it; GBM quantile
  bands for risk limits           Bayesian/Deep ensembles  loss is simpler to validate
```

**Answer for Rishi:** *"For most of the 10 prioritized problem statements on a financing desk — think fee prediction, haircut recommendation, anomaly flagging on tabular data — I'd start with LightGBM and TreeSHAP, because model validation can audit every prediction feature-by-feature before it goes live. I'd reserve DL for the subset of problems that are genuinely sequential or unstructured — text from ISDA/MSLA documents, multi-day order flow — where trees structurally can't capture what a recurrent or attention-based model can."*

[🔝 Back to Top](#table-of-contents)

---
---

## T4 · RNN, LSTM, GRU — Mechanics and When Recurrence Still Wins

**Open with the intuition:**
> "A vanilla RNN has a memory problem — it tries to compress everything it has seen into one hidden state, and each new timestep overwrites part of what came before, so gradients from ten steps back vanish before they can teach the network anything. LSTM and GRU fix that with gates — learned valves that decide what to keep, what to forget, and what to write."

**Vanilla RNN and the vanishing gradient:**

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

$$\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{t} W_h^\top \text{diag}(\tanh'(\cdot))$$

**Say it out loud:** *"That product of t-1 Jacobians is the problem: if the eigenvalues of that product are less than one, which they usually are because tanh' is at most 1, the gradient shrinks exponentially with sequence length. By the time you've backpropagated ten steps, the signal teaching the network about the first input is essentially gone. That's why a vanilla RNN can't learn long-range dependencies."*

**LSTM gating — the fix:**

$$f_t = \sigma(W_f[h_{t-1}, x_t]) \quad\text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t]) \quad\text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C[h_{t-1}, x_t]) \quad\text{(candidate cell state)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad\text{(cell state update)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t]), \quad h_t = o_t \odot \tanh(C_t)$$

**Say it out loud:** *"The cell state $C_t$ is an additive highway — information flows through it via addition, not repeated multiplication, so gradients can flow back many steps without vanishing, as long as the forget gate $f_t$ stays close to 1 for information that should persist. GRU simplifies this to two gates instead of three by merging the cell and hidden state, which trains faster with often comparable accuracy on moderate-length sequences — my default for financing time series under ~100 timesteps, reserving full LSTM or attention-based models for longer lookbacks."*

```
GATE FLOW — LSTM CELL

  x_t, h_{t-1}
       │
   ┌───┼──────────┬─────────┬──────────┐
   ▼   ▼          ▼         ▼          │
 forget input   candidate  output      │
  gate  gate      state     gate       │
  (σ)   (σ)       (tanh)    (σ)        │
   │     │           │       │         │
   └──×──┴────+──────┘       │         │
       │      C_t            │         │
       │       │             │         │
  C_{t-1}      └─────tanh────┴───×─────┘
                                 │
                                h_t
```

**Answer for Rishi:** *"For financing time series — daily repo GC, term spreads, collateral utilization — I use GRU as my default recurrent architecture for its faster training and near-parity accuracy versus LSTM at typical sequence lengths of 20-60 days. I move to full LSTM when I need finer control over long-memory effects, like quarter-end balance sheet effects that recur every ~63 trading days, and I move past recurrence entirely to attention-based models when I need the network to explicitly attend to specific historical events — like a prior taper tantrum or repo spike — regardless of how far back they occurred."*

[🔝 Back to Top](#table-of-contents)

---
---

## T5 · Fine-Tuning LLMs — Full FT vs LoRA vs QLoRA

**Open with the intuition:**
> "Full fine-tuning updates every weight in a multi-billion parameter model — that's enormous GPU memory and a real risk of catastrophic forgetting. LoRA's insight is that the *change* you need to make to adapt a model to a new task is low-rank — it lives in a small subspace — so instead of updating the full weight matrix, you learn a small correction and leave the original weights frozen."

**The math:**

$$W = W_0 + \Delta W, \qquad \Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k)$$

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

**Say it out loud:** *"$W_0$ stays frozen entirely. I only train $A$ and $B$, two small matrices whose product approximates the ideal weight update, at rank $r$ — often 8, 16, or 64, versus a weight matrix that might be thousands wide. That collapses the number of trainable parameters by orders of magnitude, typically well under 1% of the base model, which means I can fine-tune on a single GPU and store many task-specific adapters — one for financing-doc summarization, one for repo-desk Q&A — as small files layered on top of one shared frozen base model."*

**QLoRA adds quantization on top:**

$$W_0^{\text{4-bit}} = \text{NF4-quantize}(W_0), \qquad \Delta W = BA \text{ trained in bfloat16}$$

**Say it out loud:** *"QLoRA quantizes the frozen base weights to 4-bit NormalFloat — a data type designed for the roughly-Gaussian distribution of neural net weights — and keeps only the small LoRA adapters in higher precision for training. That's how you fine-tune a 70-billion parameter model on a single 48GB GPU instead of needing a multi-GPU cluster, which matters directly for the BMC — bare-metal cloud — inference infrastructure mentioned in this JD."*

```
PARAMETER-EFFICIENT FINE-TUNING — DECISION FRAME FOR A FINANCING DESK

  USE CASE                              APPROACH           WHY
  ────────────────────────────────────  ─────────────────  ──────────────────────────
  Adapt style/tone to firm's voice      Prompting only      No training needed at all
  (e.g., house style memo generation)                       
  
  Teach domain vocabulary/format        LoRA, r=8-16        Small, fast, cheap;
  (financing term sheets, MSLA clauses)                     multiple adapters, 1 base
  
  Large base model, limited GPU         QLoRA               4-bit base + LoRA adapters;
  budget on BMC infra                                        fits on single GPU
  
  Need the model to learn genuinely     Full fine-tune       Only when adapters can't
  new capabilities, not just style      (rare, expensive)    reach required accuracy
```

**Code — LoRA adapter injection (conceptual, HF-style):**

```python
"""LoRA fine-tuning setup for a financing-desk summarization adapter."""

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_lora_model(
    base_model_name: str,
    rank: int = 16,
    alpha: int = 32,
    target_modules: list[str] | None = None,
):
    """Wraps a frozen base model with trainable LoRA adapters.

    Args:
        base_model_name: HuggingFace model identifier for the frozen
            base model (e.g. a Llama-family checkpoint served on the
            firm's BMC inference infrastructure).
        rank: LoRA rank r — the bottleneck dimension of the adapter.
        alpha: LoRA scaling factor; effective update is (alpha/rank)*BA.
        target_modules: Which attention/projection layers to adapt.
            Defaults to query and value projections, the standard
            minimal-footprint choice.

    Returns:
        A PEFT-wrapped model with only the LoRA parameters trainable.
    """
    target_modules = target_modules or ["q_proj", "v_proj"]
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
```

**Answer for Rishi:** *"Given the JD explicitly calls out fine-tuning LLMs and configuring BMC inference infrastructure, I'd default to LoRA for any domain adaptation — financing terminology, house-style summarization, structured extraction from term sheets — because it lets me maintain one shared frozen base model with lightweight, swappable adapters per use case, which is both cheaper to serve and easier for model validation to review than a fully fine-tuned multi-billion parameter model per task. I'd reach for QLoRA specifically when GPU budget on the BMC servers is the binding constraint."*

[🔝 Back to Top](#table-of-contents)

---
---

## T6 · RAG Architecture for Financing Desk Knowledge

**Open with the intuition:**
> "An LLM's weights are frozen at training time — it doesn't know this quarter's ISDA amendment or yesterday's rate move. RAG's whole idea is: don't ask the model to memorize the world, give it a search engine and let it read the relevant page before it answers. That turns hallucination risk into a retrieval-quality problem, which is much easier to test and fix."

**Architecture:**

```
                         ┌─────────────────────────┐
  User Query ───────────▶│  Query Embedding Model  │
  "What's our haircut    └────────────┬────────────┘
   policy for GC repo?"               │ embedding vector
                                      ▼
                         ┌──────────────────────────┐
                         │   Vector DB (ANN search) │◀── Indexed chunks of:
                         │   e.g. FAISS / pgvector  │     MSLA/GMRA docs,
                         └────────────┬─────────────┘     internal policy wikis,
                                      │ top-k chunks      risk memos
                                      ▼
                         ┌───────────────────────────┐
                         │  Re-ranker (cross-encoder)│
                         └────────────┬──────────────┘
                                      │ re-ranked top-k
                                      ▼
                         ┌───────────────────────────┐
                         │  LLM (grounded generation)│
                         │  prompt = query + context │
                         └────────────┬──────────────┘
                                      ▼
                              Grounded answer +
                              source citations
```

**The core retrieval math — cosine similarity in embedding space:**

$$\text{sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|\,\|\mathbf{e}_d\|}$$

**Say it out loud:** *"Both the query and every document chunk get mapped into the same embedding space, and I retrieve the chunks whose vectors are most cosine-similar to the query vector. The two failure modes I actively guard against are bad chunking — splitting a haircut table across two chunks so neither has the full context — and pure dense retrieval missing exact-term matches, like a specific ISIN or clause number, which is why I always pair dense retrieval with a sparse BM25 pass in a hybrid search, then re-rank the union with a cross-encoder before it ever reaches the LLM."*

**Code — hybrid retrieval with reranking:**

```python
"""Hybrid dense + sparse retrieval for a financing-desk RAG system."""

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class HybridRetriever:
    """Combines dense embedding search with BM25 sparse search.

    Attributes:
        embed_model: Sentence embedding model for dense retrieval.
        reranker: Cross-encoder used to re-score the merged candidate
            set before returning the final top-k to the LLM prompt.
    """

    def __init__(
        self,
        chunks: list[str],
        sources: list[str],
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.chunks = chunks
        self.sources = sources
        self.embed_model = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(reranker_name)
        self._chunk_embeddings = self.embed_model.encode(
            chunks, normalize_embeddings=True
        )
        tokenized = [c.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self, query: str, dense_k: int = 20, sparse_k: int = 20, final_k: int = 5
    ) -> list[RetrievedChunk]:
        """Retrieves and re-ranks the most relevant chunks for a query.

        Args:
            query: The natural-language query.
            dense_k: Candidates to pull from dense embedding search.
            sparse_k: Candidates to pull from BM25 sparse search.
            final_k: Number of chunks to return after re-ranking.

        Returns:
            A list of RetrievedChunk sorted by descending relevance.
        """
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self._chunk_embeddings @ query_emb
        dense_top = np.argsort(dense_scores)[::-1][:dense_k]

        sparse_scores = self._bm25.get_scores(query.lower().split())
        sparse_top = np.argsort(sparse_scores)[::-1][:sparse_k]

        candidate_idx = sorted(set(dense_top) | set(sparse_top))
        pairs = [(query, self.chunks[i]) for i in candidate_idx]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_idx, rerank_scores), key=lambda x: x[1], reverse=True
        )[:final_k]

        return [
            RetrievedChunk(
                text=self.chunks[i], source=self.sources[i], score=float(score)
            )
            for i, score in ranked
        ]
```

**Answer for Rishi:** *"For financing-desk knowledge — MSLA/GMRA terms, internal haircut and margin policy, risk memos — I'd build hybrid retrieval, not pure dense, because a lot of the queries a trader asks are exact-match by nature: a specific clause, ISIN, or counterparty name that dense embeddings can blur. I'd evaluate the system on retrieval precision/recall against a hand-labeled query set before ever measuring generation quality, because in RAG, generation quality is capped by retrieval quality — a perfect LLM given the wrong context still gives a wrong, and confidently wrong, answer."*

[🔝 Back to Top](#table-of-contents)

---
---

## T7 · Agentic AI — ReAct, Tool Use, Multi-Agent Orchestration

**Open with the intuition:**
> "A chatbot answers a question. An agent decides what to do — it can look something up, call a tool, check its own work, and only then answer. On a financing desk, the useful version of this isn't 'autonomous trading,' it's an agent that can pull live GC rates, check a counterparty's utilization against a limit, and draft a recommendation for a human to approve — with hard guardrails around anything that touches risk or execution."

**The ReAct loop:**

$$\text{Thought}_t \rightarrow \text{Action}_t \rightarrow \text{Observation}_t \rightarrow \text{Thought}_{t+1} \rightarrow \cdots \rightarrow \text{Answer}$$

**Say it out loud:** *"ReAct interleaves reasoning traces with tool calls. Instead of asking the model to answer in one shot, I let it think out loud about what it needs, call a tool — a repo-rate API, a limits database — observe the result, and reason again before its next action. That loop is what lets an LLM go from 'plausible-sounding answer' to 'answer grounded in a live tool call it just made.'"*

**Multi-agent orchestration for a financing workflow:**

```
                     ┌────────────────────────┐
   User Request ────▶│   Orchestrator Agent   │
   "Recommend a      │  (plans, delegates,    │
    haircut for this │   aggregates)          │
    new collateral)" └──────────┬─────────────┘
                                │
             ┌──────────────────┼────────────────────┐
             ▼                  ▼                    ▼
   ┌────────────────────┐ ┌─────────────────────┐ ┌────────────────────┐
   │  Market Data Agent │ │  Risk Agent         │ │  Policy Agent      │
   │  pulls live vol,   │ │  checks counterparty│ │  retrieves haircut │
   │  liquidity metrics │ │  limits & exposure  │ │  policy via RAG    │
   └─────────┬──────────┘ └─────────┬───────────┘ └─────────┬──────────┘
             └──────────────────────┼───────────────────────┘
                                    ▼
                     ┌──────────────────────────┐
                     │  Orchestrator synthesizes│
                     │  recommendation + cites  │
                     │  sources → HUMAN APPROVAL│
                     └──────────────────────────┘
```

**Say it out loud on guardrails:** *"The single most important design decision in an agentic system on a trading desk is the boundary between 'agent can do this autonomously' and 'agent must stop and ask a human.' I'd hard-code that boundary outside the LLM's control entirely — a deterministic policy layer that gates any tool call touching live risk limits, execution, or client-facing communication behind human approval, regardless of what the agent 'decides.' The LLM proposes; a human or a deterministic rule disposes, anywhere real money or real risk is on the line."*

**Code — a guarded tool-calling agent skeleton:**

```python
"""Guarded ReAct-style agent for financing desk research tasks.

The agent can call read-only tools autonomously. Any tool tagged as
'high_risk' requires explicit human approval before execution.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    func: Callable[..., str]
    high_risk: bool = False


class GuardedAgent:
    """A tool-using agent with a hard-coded human-approval gate.

    Attributes:
        tools: Registered tools keyed by name.
        approval_callback: Function invoked to obtain human sign-off
            before any high_risk tool executes.
    """

    def __init__(
        self, tools: list[Tool], approval_callback: Callable[[str, dict], bool]
    ) -> None:
        self.tools = {t.name: t for t in tools}
        self.approval_callback = approval_callback

    def execute_action(self, tool_name: str, kwargs: dict) -> str:
        """Executes a tool call, gating high-risk tools on approval.

        Args:
            tool_name: Name of the registered tool to invoke.
            kwargs: Keyword arguments to pass to the tool.

        Returns:
            The tool's string output, or a rejection message if a
            high-risk action was not approved.

        Raises:
            KeyError: If tool_name is not registered.
        """
        tool = self.tools[tool_name]
        if tool.high_risk:
            approved = self.approval_callback(tool_name, kwargs)
            if not approved:
                return f"ACTION BLOCKED: '{tool_name}' requires human approval."
        return tool.func(**kwargs)
```

**Answer for Rishi:** *"I see agentic AI on a financing desk as valuable for research and drafting workflows — pulling data, checking policy, synthesizing a recommendation — and I'd deliberately keep it out of anything that executes risk or client-facing actions without a human in the loop. The engineering discipline that matters most isn't the LLM's reasoning quality, it's building the deterministic guardrail layer around it that no prompt can talk its way past."*

[🔝 Back to Top](#table-of-contents)

---
---

## T8 · Model Evaluation — Classical Metrics vs LLM Evaluation

**Open with the intuition:**
> "Evaluating a regression model is easy — you have ground truth, you compute an error. Evaluating an LLM's output is hard because for most useful tasks there's no single correct answer, just better and worse ones. That's why LLM evaluation has had to invent a whole new toolkit."

**Classical metrics:**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}, \qquad \text{MAPE} = \frac{100}{n}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**LLM/NLG metrics and their limits:**

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i \mid w_{<i})\right)$$

**Say it out loud:** *"Perplexity measures how 'surprised' the model is by held-out text under its own probability distribution — lower is better, and it's useful for comparing base language models, but it says nothing about whether a specific financing-summary output is factually correct. BLEU and ROUGE measure n-gram overlap with a reference answer — useful for translation and extractive summarization, nearly useless for open-ended generation where a correct answer can be phrased ten different ways with zero n-gram overlap to any single reference."*

**LLM-as-judge — and why I'd validate it before trusting it:**

$$\text{Score} = \text{Judge-LLM}(\text{prompt}, \text{response}, \text{rubric})$$

**Say it out loud:** *"LLM-as-judge uses a strong model to score a weaker model's output against an explicit rubric — factuality, groundedness, completeness. It scales far better than human review, but it has known biases: position bias, favoring longer or more confident-sounding answers, and self-preference bias when judge and generator share an architecture. I always validate the judge against a small human-labeled gold set before trusting it at scale, and for anything touching real numbers — rates, spreads, limits — I add a deterministic factuality check that verifies every number in the output against the retrieved source, independent of the judge's opinion."*

```
EVALUATION STACK FOR A FINANCING-DESK RAG/GENAI SYSTEM

  LAYER                    METRIC                          CATCHES
  ───────────────────────  ─────────────────────────────  ─────────────────────────
  Retrieval quality        Precision@k, Recall@k,          Wrong or missing context
                            MRR against labeled query set    fed to the LLM
  
  Groundedness             Deterministic entity/number     Hallucinated figures not
                            match: output vs retrieved text  present in source docs
  
  Answer quality           LLM-as-judge vs rubric,          Incomplete, off-topic,
                            validated against human labels    poorly-reasoned answers
  
  Business outcome         Trader acceptance rate,          Whether it's actually
                            time-to-answer vs baseline        used and trusted
```

**Answer for Rishi:** *"I never evaluate a GenAI system on a single metric. I decompose it into retrieval quality, groundedness — did every number in the output actually come from a retrieved source — and answer quality via a judge model I've validated against human labels. For anything a trader will act on, groundedness is non-negotiable and gets checked deterministically, not left to the judge's discretion."*

[🔝 Back to Top](#table-of-contents)

---
---

## T9 · Prompt Engineering and Inference Optimization on BMC Servers

**Open with the intuition:**
> "Prompt engineering is the cheapest lever you have before touching weights at all — it's the difference between an LLM guessing at your task and an LLM being told exactly what 'good' looks like. Inference optimization is the separate, equally important problem of making that good output arrive fast enough and cheap enough to run at desk scale."

**Prompting techniques, ranked by what they buy you:**

```
TECHNIQUE                WHAT IT ADDS                        WHEN I USE IT
────────────────────    ──────────────────────────────      ──────────────────────────
Zero-shot                Baseline instruction only            Simple, well-defined tasks
Few-shot                 2-5 worked examples in-context        Format-sensitive output
                                                                (e.g. structured JSON)
Chain-of-thought (CoT)   "Think step by step" reasoning        Multi-step numerical or
                          trace before the final answer         logical tasks
Structured output /      JSON schema constraint on             Any output consumed by
function-calling         generation                             downstream code, not a human
Self-consistency         Sample CoT multiple times,             High-stakes tasks where
                          take majority-vote answer              a single sample is noisy
```

**Inference optimization — the math that matters for latency/cost:**

$$\text{Throughput} \propto \frac{\text{Batch Size}}{\text{Memory Bandwidth Bound Latency}}$$

**Say it out loud:** *"LLM inference is memory-bandwidth bound, not compute bound, for the autoregressive decode phase — you're moving the model's weights and KV cache through memory for every single token generated. That's why the two highest-leverage optimizations on BMC infrastructure are quantization, which shrinks the weights that need to move through memory, and continuous batching, which lets the server pack multiple users' requests together so that memory-bandwidth cost is amortized across more useful work instead of one token at a time per user. For a financing desk with bursty query patterns — heavy at market open, light overnight — dynamic batching plus a smaller quantized model for latency-sensitive queries, with a larger model reserved for batch/overnight analysis, is usually the right cost/latency split."*

```
INFERENCE OPTIMIZATION STACK — BMC SERVER DEPLOYMENT

  Quantization (INT8/INT4/NF4)  → smaller weights, less memory bandwidth per token
  KV-cache reuse / PagedAttention → avoids recomputation across multi-turn sessions
  Continuous/dynamic batching    → amortizes memory-bandwidth cost across requests
  Speculative decoding           → small draft model proposes tokens, big model verifies
  Model routing (small/large)    → cheap model for simple queries, escalate to large
```

**Answer for Rishi:** *"I treat prompting as the first optimization lever — structured output and few-shot examples usually fix 80% of format and reliability issues before I touch inference infrastructure at all. For the BMC deployment side, given financing-desk query patterns are bursty around market open, I'd push for a two-tier model routing setup — a quantized small model handling routine lookups with low latency, escalating to a larger model only for genuinely complex synthesis tasks — because that's usually a better cost/latency trade than serving one large model for every query."*

[🔝 Back to Top](#table-of-contents)

---
---

## T10 · Deep Learning Fundamentals — Backprop, Optimizers, Regularization

**Open with the intuition:**
> "Backpropagation is just the chain rule applied systematically through a computational graph, so every weight learns exactly how much it's responsible for the final error. Optimizers decide how big a step to take once you know that direction. Regularization stops the network from just memorizing the training set — which is the single biggest risk on the small, noisy datasets typical of financing desks."

**Backpropagation:**

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \prod_{k=l+1}^{L} \frac{\partial a^{(k)}}{\partial a^{(k-1)}} \cdot \frac{\partial a^{(l)}}{\partial W^{(l)}}$$

**Adam optimizer:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**Say it out loud:** *"Adam keeps a running estimate of both the mean and the variance of the gradient — momentum from the mean, and an adaptive per-parameter learning rate from the variance, so parameters with noisy or sparse gradients get bigger effective steps and parameters with consistently large gradients get damped. The bias-correction terms $\hat{m}_t, \hat{v}_t$ fix the fact that both estimates start at zero and would otherwise be biased toward zero in early steps."*

**Regularization — the critical concern for small financial datasets:**

$$L_{\text{total}} = L_{\text{data}} + \lambda \|\theta\|_2^2 \quad \text{(weight decay)}$$

**Say it out loud on dropout and small-data overfitting:** *"On a financing desk I rarely have millions of clean labeled examples — I might have a few thousand fee quotes or a few years of daily rate history. That's a regime where a deep net will happily memorize noise unless I actively fight it: dropout randomly zeroes a fraction of activations each forward pass so no single neuron can co-adapt to memorize a specific training example, weight decay penalizes large weights directly, and early stopping on a validation set is often the single most effective regularizer of all because it stops training the moment the model starts fitting noise instead of signal. For genuinely small datasets, I'll often prefer a shallower network, or fall back to gradient boosting entirely, over fighting a deep net into behaving on data that doesn't support its capacity."*

**Answer for Rishi:** *"The deep learning fundamentals matter less to me as textbook facts and more as a diagnostic checklist when a model isn't training well — is the gradient vanishing or exploding, is the learning rate schedule wrong for Adam's adaptive step size, is the validation loss diverging from training loss because I under-regularized on a small dataset. On financing desk data, which is rarely internet-scale, my default posture is to regularize aggressively and prefer the simplest model that clears the accuracy bar, because overfitting on a few thousand rows is the single most common way a promising deep learning project quietly fails in production."*

[🔝 Back to Top](#table-of-contents)

---
---

# 🏦 DOMAIN

---

## D1 · Liquid Financing, Delta One & Prime — Where AI/ML Actually Adds Value

**Open with the intuition:**
> "Liquid Financing is the plumbing that lets clients get leveraged or synthetic exposure without owning the underlying — securities lending, repo, swaps, and prime brokerage margin. The plumbing runs on thousands of daily pricing, margin, and inventory decisions that used to be manual and rules-based. AI/ML's job here isn't to replace the trader's judgment — it's to compress the decision latency and surface the signal a human would otherwise miss across too many positions to review by hand."

**The business map:**

```
LIQUID FINANCING — CROSS-ASSET COVERAGE (per this JD)

  Equities & Delta One      → synthetic exposure via swaps/futures, dividend/index arb
  Rate & Credit Financing   → repo, securities lending, term financing, collateral mgmt
  FX                        → funding across currencies, cross-currency basis
  Risk                      → margin, haircuts, counterparty exposure
  Futures & Prime           → client margin financing, portfolio margining

  WHERE ML ADDS VALUE:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Fee/rate prediction & anomaly detection   (T1, T2, T3 above)           │
  │  Collateral optimization — cheapest-to-deliver across a huge            │
  │    inventory of eligible collateral, subject to haircut & concentration │
  │    constraints (combinatorial optimization + ML-predicted scarcity)     │
  │  Demand/supply forecasting for securities lending (which names          │
  │    will be "special," i.e. command a lending fee premium)               │
  │  Client behavior modeling — utilization forecasting for margin/limit    │
  │    management                                                           │
  │  GenAI for term-sheet/ISDA-MSLA/GMRA parsing and desk knowledge RAG     │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Answer for Rishi:** *"What excites me about this specific seat is that Liquid Financing sits at the intersection of every asset class's funding cost, which means the ML problems are genuinely cross-asset — a collateral scarcity signal in Treasuries affects repo pricing, which affects synthetic financing cost in Delta One, which affects a client's total cost of leverage in Prime. Most AI/ML roles at banks are siloed to one desk's data; this one explicitly isn't, which is exactly the kind of high-dimensional, cross-asset problem where a unified modeling framework — not five different point solutions — creates the most value."*

[🔝 Back to Top](#table-of-contents)

---
---

## D2 · Repo Rate Forecasting — GC vs Specials

**Open with the intuition:**
> "Most collateral trades at the 'general collateral' rate — it's fungible, nobody cares which specific bond you post. Some collateral is 'special' — everyone wants that specific CUSIP, usually because it's cheapest-to-deliver into a futures contract or heavily shorted — and its repo rate can trade well below GC, sometimes negative, because borrowers are willing to pay up in fee terms to get their hands on it."

**The math — special repo spread:**

$$\text{Special Spread} = r_{\text{GC}} - r_{\text{special}} > 0$$

$$P(\text{special}) = f(\text{short interest}, \text{CTD status}, \text{auction cycle timing}, \text{index rebalance flow})$$

**Say it out loud:** *"I model 'will this security go special' as a classification problem first — gradient boosting on short-interest data, upcoming Treasury auction cycles (a note becomes the on-the-run and often specialness spikes right after auction), and index rebalancing flows that create temporary concentrated demand. Once I know a name is likely to go special, I model the magnitude of the spread as a regression conditioned on that classification — a two-stage model, because the drivers of 'does it go special at all' and 'how special does it get' aren't the same drivers."*

```
GC vs SPECIALS — DECISION IMPACT

  ─────────────────  ────────────────────  ──────────────────────────────────
  GC repo             Stable, low-vol       ARIMA-GARCH-X sufficient;
                      funding rate           macro/liquidity driven
  
  Specials            Spiky, event-driven,  Two-stage: classify "goes special"
                      often negative rate    (GBM) then regress magnitude
                                              conditional on positive classification
```

**Answer for Rishi:** *"The desk value of forecasting specials isn't academic — if I can flag a name is likely to go special before it does, financing desk can pre-position collateral or reprice client financing ahead of the move, capturing spread that would otherwise be given away to a faster competitor. That's a direct, quantifiable P&L impact from a two-stage ML model."*

[🔝 Back to Top](#table-of-contents)

---
---

## D3 · ML for Margin, Haircut Optimization & Counterparty Risk

**Open with the intuition:**
> "A haircut is insurance — it's how much extra collateral value you demand above the loan amount to protect against the collateral's price falling before you can liquidate it if the counterparty defaults. Set it too low and you're under-protected in a crisis; set it too high and you're uncompetitive on financing terms. ML's job is to make that trade-off data-driven and dynamic instead of a static table that gets updated once a year."

**The math — haircut as a VaR-based buffer:**

$$h = \text{VaR}_\alpha(\Delta P) + \text{liquidation cost} + \text{buffer for correlation risk}$$

$$\text{VaR}_\alpha = -\inf\{x : P(\Delta P \le x) \ge \alpha\}$$

**Say it out loud:** *"The core haircut is sized to cover the worst reasonably-expected price move in the collateral over the time it would take to liquidate it if a counterparty defaulted, at some confidence level like 99%. The static-table approach uses one number per asset class forever. The ML-driven approach conditions that VaR estimate on current volatility regime, liquidity conditions, and — critically — correlation with the counterparty's own credit risk, because collateral that's correlated with the counterparty's default risk (wrong-way risk) needs a materially bigger haircut than the same asset posted by an uncorrelated counterparty."*

```
STATIC vs DYNAMIC HAIRCUT FRAMEWORK

  STATIC TABLE                          ML-DRIVEN DYNAMIC HAIRCUT
  ────────────────────────────────      ──────────────────────────────────────
  One haircut per asset class,          Conditional VaR model: current vol
  reviewed annually                      regime + liquidity + counterparty
                                          correlation, updated daily/intraday

  Blind to wrong-way risk                Explicitly models correlation between
                                          collateral value and counterparty PD

  Competitive disadvantage in calm       Tighter (more competitive) haircuts in
  markets (over-protected)               calm regimes, wider in stress —
                                          both more accurate and more competitive
```

**Answer for Rishi:** *"This is where model risk discipline matters most in the whole JD — a haircut model is directly capital- and risk-limit-relevant, so anything I ship here needs full SR 11-7 style documentation, backtesting against realized liquidation outcomes, and a conservative fallback if the model's confidence is low. I would never let a black-box deep learning haircut model go live without a simpler, auditable model running in shadow mode alongside it for at least one full market cycle."*

[🔝 Back to Top](#table-of-contents)

---
---

## D4 · State of Research — Foundation Models for Markets & Time Series

**Open with the intuition:**
> "The frontier right now is asking whether the same transformer architecture that revolutionized language can be pretrained once on massive amounts of time-series data and then adapted cheaply to a new financial series, the same way GPT-style models get fine-tuned for a new task — instead of training a bespoke model from scratch for every single asset."

**What I'd bring up unprompted:**

> "Three threads I'm tracking closely: first, **time-series foundation models** — architectures like Google's TimesFM, Amazon's Chronos, and Nixtla's TimeGPT, which pretrain on huge cross-domain time-series corpora and then zero-shot or lightly fine-tune to a new series, which is directly relevant to a desk covering as many different financing series as this one does. Second, **state-space models** like Mamba, which handle long sequences with linear-time complexity instead of the quadratic cost of attention — relevant anywhere we want very long lookback windows, like multi-year funding-cost regime detection, without the compute cost exploding. Third, on the GenAI-in-finance side, the maturing literature on **grounded, tool-using financial agents** — moving past 'can an LLM pass a finance exam' toward 'can an LLM reliably execute a multi-step research workflow with verifiable, cited outputs,' which is exactly the agentic RAG pattern I described in T6/T7. I'd flag, honestly, that a lot of 'LLMs for alpha generation' research is still more promising in papers than in live P&L — the realistic, defensible near-term wins in a bank are workflow automation, document understanding, and structured-data forecasting, not an LLM discovering novel trading signals from raw text."*

```
RESEARCH FRONTIER MAP — RELEVANCE TO LIQUID FINANCING

  THREAD                          MATURITY        RELEVANCE TO THIS DESK
  ───────────────────────────    ─────────────   ──────────────────────────────
  Time-series foundation models   Early prod       High — many series, transfer
  (TimesFM, Chronos, TimeGPT)      (2024-2026)      learning reduces per-series cost

  State-space models (Mamba,      Research →        Medium-high — long lookback,
  linear attention)                 early prod        regime detection over years

  Grounded/agentic financial       Early prod        High — desk knowledge RAG,
  LLM workflows                                       term-sheet parsing, research

  LLM-generated alpha signals      Mostly research   Low near-term — high hallucination
  from unstructured text                              /overfitting risk, hard to audit
```

**Answer for Rishi:** *"I try to stay honest about the difference between what's exciting in a paper and what's defensible in a bank's production codebase. Time-series foundation models and grounded agentic workflows are where I'd actually place bets on this desk in the next 12-18 months; pure LLM-signal-generation from text is where I'd stay a curious, cautious observer rather than an early production adopter, until the auditability problem is genuinely solved."*

[🔝 Back to Top](#table-of-contents)

---
---

## D5 · Model Governance — SR 11-7 Meets LLMs

**Open with the intuition:**
> "SR 11-7 was written for models that output a number — you can validate them by comparing the number to reality. LLMs output language, which makes 'is this model right' a much harder question, but the governance principles — independent validation, ongoing monitoring, clear ownership, ability to explain a model's output — don't go away just because the model got more sophisticated. If anything they matter more."

**The mapping:**

```
SR 11-7 PILLAR              CLASSICAL ML MODEL              LLM / GENAI SYSTEM
──────────────────────    ───────────────────────────    ────────────────────────────
Conceptual soundness       Statistical assumptions          Prompt/architecture design,
                           documented and justified          RAG grounding strategy documented

Ongoing monitoring          Drift on inputs/outputs,         Groundedness rate, hallucination
                            performance vs benchmark          rate, retrieval quality over time

Outcomes analysis           Backtesting vs realized          Human review sampling, LLM-judge
                             outcomes                          validated against human labels

Independent validation      Second-line model risk           Independent red-teaming +
                             team reviews methodology          adversarial prompt testing

Model inventory / ownership Documented owner, version,       Same — plus prompt version,
                             retrain cadence                   base model version, adapter version
```

**Say it out loud:** *"The single hardest new governance problem LLMs introduce is non-determinism — the same prompt can produce different outputs run to run, and the model can be updated by the vendor outside your control if you're calling a hosted API. My practice is to pin model versions explicitly, never silently accept an upstream model upgrade into a production financing workflow without re-validation, and treat prompt changes with the same version control and change-management rigor as code changes — because a one-word prompt edit can change model behavior as much as a retrained model would."*

**Answer for Rishi:** *"I'd bring the same model-risk discipline to a GenAI system that I'd bring to a regression model — documented assumptions, independent validation, ongoing monitoring — while being upfront that the specific tests change: groundedness and hallucination-rate replace RMSE, and adversarial red-teaming replaces classical backtesting. What doesn't change is the principle that nothing goes into a workflow a trader or client relies on without a second line of defense that isn't the person who built it."*

[🔝 Back to Top](#table-of-contents)

---
---

# Quick-Reference Equation & Architecture Sheet

```
══════════════════════════════════════════════════════════════════════════════
CLASSICAL TIME SERIES & REGRESSION
══════════════════════════════════════════════════════════════════════════════

ARIMA(p,d,q):        (1 - Σφ_i L^i)(1-L)^d y_t = (1 + Σθ_j L^j) ε_t
GARCH(1,1):          σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²
OLS:                 β̂ = argmin ||y - Xβ||²
LASSO:               β̂ = argmin ||y - Xβ||² + λ||β||₁
Elastic Net:         β̂ = argmin ||y - Xβ||² + λ₁||β||₁ + λ₂||β||₂²

══════════════════════════════════════════════════════════════════════════════
TREE-BASED / GRADIENT BOOSTING
══════════════════════════════════════════════════════════════════════════════

Boosting update:     F_m(x) = F_{m-1}(x) + η·h_m(x)
SHAP value:          φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!]·[f(S∪{i}) - f(S)]

══════════════════════════════════════════════════════════════════════════════
DEEP LEARNING / SEQUENCE MODELS
══════════════════════════════════════════════════════════════════════════════

Vanilla RNN:         h_t = tanh(W_h h_{t-1} + W_x x_t + b)
Vanishing grad:      ∂h_t/∂h_1 = Π W_h^T · diag(tanh'(·))   [shrinks with depth]
LSTM forget gate:    f_t = σ(W_f[h_{t-1}, x_t])
LSTM cell update:    C_t = f_t⊙C_{t-1} + i_t⊙C̃_t
Adam:                m_t=β₁m_{t-1}+(1-β₁)g_t;  v_t=β₂v_{t-1}+(1-β₂)g_t²
                     θ_t = θ_{t-1} - η·m̂_t/(√v̂_t + ε)

══════════════════════════════════════════════════════════════════════════════
GENAI / LLM
══════════════════════════════════════════════════════════════════════════════

LoRA:                W = W_0 + BA,  B∈R^(d×r), A∈R^(r×k), r << min(d,k)
QLoRA:               W_0 quantized to 4-bit NF4; ΔW=BA trained in bf16
RAG similarity:      sim(q,d) = (e_q · e_d) / (||e_q|| ||e_d||)
Perplexity:          PPL = exp(-1/N Σ log p(w_i | w_<i))
ReAct loop:           Thought_t → Action_t → Observation_t → Thought_{t+1} → ... → Answer

══════════════════════════════════════════════════════════════════════════════
LIQUID FINANCING DOMAIN
══════════════════════════════════════════════════════════════════════════════

Special spread:      Spread = r_GC - r_special > 0
Haircut (VaR-based): h = VaR_α(ΔP) + liquidation cost + wrong-way-risk buffer
VaR:                 VaR_α = -inf{x : P(ΔP ≤ x) ≥ α}

══════════════════════════════════════════════════════════════════════════════
MODEL SELECTION — ONE-LINE HEURISTICS
══════════════════════════════════════════════════════════════════════════════

Tabular, need auditability            → Elastic Net or GBM + SHAP
Tabular, max accuracy, less audit     → LightGBM / XGBoost
Sequential, < 100 timesteps           → GRU
Sequential, long memory / attention   → LSTM or Transformer / TFT
Domain adaptation of an LLM           → LoRA (QLoRA if GPU-constrained)
Need current/external knowledge       → RAG, hybrid dense+sparse retrieval
Multi-step tool-using workflow        → Agentic (ReAct) + deterministic guardrails
```

[🔝 Back to Top](#table-of-contents)

---

*Last updated: July 2026 · Prepared for Barclays AI/ML Modeler (Liquid Financing) screening round with Rishi D., MD.*
