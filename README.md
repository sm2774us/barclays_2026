# Barclays — AI / ML Modeler, Liquid Financing — Technical Interview Playbook
### Rishi Dhingra (MD, Global Markets — Electronification, eTrading & AI/ML | Prime Services, Financing & Delta One) · HLS Trading Co-Founder Panel
#### 30 Questions × 7 Domains · Wednesday, July 8 2026 · 1-Hour Technical Round

> **Delivery philosophy:** Every answer follows *intuition → math derived line-by-line → Feynman restatement → production Python 3.13*. Rishi built an electronic trading/AI franchise from the ground up and co-founded HLS Trading — he will probe whether you can **build**, not just describe. The team has 50+ problem statements queued and 10 prioritized; he wants evidence you can independently scope a business problem, choose the right tool (classical ML vs. DL vs. Gen AI), and ship it into production alongside a 10–15 person engineering team.

---
---

[↩️ Back to README.md](../README.md)

---
---

## ⏱️ Interview Question Budget

```
DOMAIN                          QUESTIONS   WEIGHT   RISHI'S LENS
───────────────────────────────  ─────────  ───────  ──────────────────────────────────────
EXPERIENCE / PRODUCTION ML        Q1–Q3       High    Can you own a problem end-to-end?
GEN AI (Fine-tune/RAG/Eval)       Q4–Q8       High    LLM orchestration for Liquid Financing
REGRESSION                        Q9–Q11      Med     First-principles derivation, not sklearn
TREE-BASED MODELS                 Q12–Q14     Med     XGBoost internals, not just "it works"
TIME SERIES / FORECASTING         Q15–Q19     High    Financing curve is a time-series problem
DEEP LEARNING (NN/MLP/RNN/LSTM)   Q20–Q27     High    Core JD requirement — go deep
SYSTEM DESIGN / STRATEGY          Q28–Q30     High    "Build AI/ML capability from the ground up"
TAKE HOME PROJECTS                P1-P5       High    "Research and Present"
```

> **Priority rule:** Liquid Financing spans Equities & Delta One, Rate/Credit Financing, FX, Risk, Futures & Prime Derivatives. Rishi's own background is electronification/eTrading — expect him to push hardest on **time series forecasting for pricing curves (repo/financing spreads)**, **RNN/LSTM sequence modeling**, and **Gen AI infra** (the team explicitly uses Claude Code). Have Q15–Q27 cold; Q1–Q3 and Q28–Q30 are where you differentiate as someone who ships, not just researches.

---

## Table of Contents

### 🧑‍💻 EXPERIENCE & PRODUCTION PYTHON
- [Q1 · The Full Research-to-Production Lifecycle](#q1--the-full-research-to-production-lifecycle)
- [Q2 · Python Performance, the GIL & Mechanical Sympathy](#q2--python-performance-the-gil--mechanical-sympathy)
- [Q3 · Designing a Production-Grade Signal Engine](#q3--designing-a-production-grade-signal-engine)

### 🤖 GEN AI
- [Q4 · Fine-Tuning vs. Prompt Engineering vs. RAG — and LoRA Math](#q4--fine-tuning-vs-prompt-engineering-vs-rag--and-lora-math)
- [Q5 · RAG Architecture Deep-Dive for Financing Docs](#q5--rag-architecture-deep-dive-for-financing-docs)
- [Q6 · LLM Evaluation — RAGAS, Hallucination Detection, LLM-as-Judge](#q6--llm-evaluation--ragas-hallucination-detection-llm-as-judge)
- [Q7 · Model Selection Matrix — Claude, GPT, Open-Weight](#q7--model-selection-matrix--claude-gpt-open-weight)
- [Q8 · Structured Extraction & Prompt Engineering for Term Sheets](#q8--structured-extraction--prompt-engineering-for-term-sheets)

### 📉 REGRESSION
- [Q9 · OLS — Normal Equations & Gauss-Markov Proof](#q9--ols--normal-equations--gauss-markov-proof)
- [Q10 · Ridge, Lasso & Elastic Net — Derivation and Bias-Variance](#q10--ridge-lasso--elastic-net--derivation-and-bias-variance)
- [Q11 · Heteroskedasticity, Autocorrelation & Newey-West](#q11--heteroskedasticity-autocorrelation--newey-west)

### 🌳 TREE-BASED MODELS
- [Q12 · Decision Trees — Entropy, Gini & Information Gain](#q12--decision-trees--entropy-gini--information-gain)
- [Q13 · Random Forest vs. Gradient Boosting — XGBoost 2nd-Order Taylor Expansion](#q13--random-forest-vs-gradient-boosting--xgboost-2nd-order-taylor-expansion)
- [Q14 · Feature Importance & SHAP Values](#q14--feature-importance--shap-values)

### ⏱️ TIME SERIES & FORECASTING
- [Q15 · Stationarity, ADF Test & ARIMA](#q15--stationarity-adf-test--arima)
- [Q16 · GARCH/EGARCH — Deriving Volatility Forecasts](#q16--garchegarch--deriving-volatility-forecasts)
- [Q17 · Hidden Markov Models — Baum-Welch & Viterbi for Regime Detection](#q17--hidden-markov-models--baum-welch--viterbi-for-regime-detection)
- [Q18 · Walk-Forward Validation & Combinatorial Purged Cross-Validation](#q18--walk-forward-validation--combinatorial-purged-cross-validation)
- [Q19 · Kalman Filter for Dynamic Hedge Ratios / Financing Spread Tracking](#q19--kalman-filter-for-dynamic-hedge-ratios--financing-spread-tracking)

### 🧠 DEEP LEARNING
- [Q20 · MLP — Forward Pass & Backpropagation Derived Line-by-Line](#q20--mlp--forward-pass--backpropagation-derived-line-by-line)
- [Q21 · Activation Functions & the Vanishing Gradient Problem](#q21--activation-functions--the-vanishing-gradient-problem)
- [Q22 · RNNs — Backpropagation Through Time](#q22--rnns--backpropagation-through-time)
- [Q23 · LSTM — Gate Equations Derived from First Principles](#q23--lstm--gate-equations-derived-from-first-principles)
- [Q24 · GRU vs. LSTM — Simplification Trade-offs](#q24--gru-vs-lstm--simplification-trade-offs)
- [Q25 · Regularization — Dropout & BatchNorm Math](#q25--regularization--dropout--batchnorm-math)
- [Q26 · Attention & the Transformer Building Block](#q26--attention--the-transformer-building-block)
- [Q27 · Autoencoders for Dimensionality Reduction & Anomaly Detection](#q27--autoencoders-for-dimensionality-reduction--anomaly-detection)

### 🏗️ SYSTEM DESIGN & STRATEGY
- [Q28 · Design an End-to-End Alpha/Pricing Signal Pipeline](#q28--design-an-end-to-end-alphapricing-signal-pipeline)
- [Q29 · Bayesian Inference for Regime-Adaptive Position Sizing](#q29--bayesian-inference-for-regime-adaptive-position-sizing)
- [Q30 · Building an AI/ML Capability From Zero — the Greenfield Roadmap](#q30--building-an-aiml-capability-from-zero--the-greenfield-roadmap)

### 🏗️ TAKE HOME PROJECTS
- **P1 · Securities-Lending Fee & Rebate-Rate Forecasting**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-1-securities-lending-fee--rebate-rate-forecasting)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#p1--securities-lending-fee--rebate-rate-forecasting)**
- **P2 · Client Margin & Haircut Optimization**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-2-client-margin--haircut-optimization)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#p2--client-margin--haircut-optimization)**
- **P3 · Cross-Asset Funding-Spread Anomaly Detection**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-3-cross-asset-funding-spread-anomaly-detection)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#p3--cross-asset-funding-spread-anomaly-detection)**
- **P4 · Prime Balance & Utilization Forecasting (Deep Learning)**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-4-prime-balance--utilization-forecasting-deep-learning)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#p4--prime-balance--utilization-forecasting-deep-learning)**
- **P5 · RAG Financing-Desk Copilot (GenAI / LLM)**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-5-rag-financing-desk-copilot-genai)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#p5--rag-financing-desk-copilot-genai--llm)**

- **[Quick-Reference Equation Sheet](#quick-reference-equation-sheet)**

[🔝 Back to Top](#table-of-contents)

---
---

# 🧑‍💻 EXPERIENCE & PRODUCTION PYTHON

---

## Q1 · The Full Research-to-Production Lifecycle

**Open with the intuition (15 seconds):**
> "A model that lives in a notebook is a hypothesis, not a capability. My job across BAM, Highbridge, and Millburn has always been the same shape: formulate a testable hypothesis, validate it against a null that's actually hard to beat, and only then does it earn a place in the production risk pipeline. The failure mode I actively guard against is overfitting the backtest — so every signal I ship has been through purged, embargoed cross-validation before I'll defend it to a PM."

### The Lifecycle, as a State Machine

```
 HYPOTHESIS ──▶ DATA ENGINEERING ──▶ FEATURE RESEARCH ──▶ MODEL FIT ──▶ VALIDATION
     ▲                                                                       │
     │                                                                       ▼
 POST-MORTEM ◀── LIVE MONITORING ◀── PROD DEPLOYMENT ◀── PAPER TRADE ◀── OOS TEST
     │                                                                       │
     └───────────────────── feedback loop (research library) ◀───────────────┘
```

**Feynman explanation:** Think of it like a drug trial. You don't approve a drug because it worked once on the training population (in-sample fit). You need a held-out population (out-of-sample), a placebo comparison (a naive benchmark — e.g., a random-walk forecast), and post-market surveillance (live PnL attribution vs. backtest expectation) because the population itself drifts (regime change). A model is never "done" — it's on a lifecycle with a kill-switch.

### Production Checklist (what I actually gate on before flipping a signal live)

```python
"""Production readiness gate for systematic signals.

Enforces institutional pre-deployment checks before a signal is permitted
to route into the live risk pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class GateStatus(Enum):
    """Outcome of a single readiness check."""

    PASS = auto()
    FAIL = auto()
    WARN = auto()


@dataclass(slots=True)
class ReadinessCheck:
    """A single named production-readiness gate.

    Attributes:
        name: Human-readable identifier of the check (e.g., "OOS Sharpe").
        status: Result of evaluating the check.
        detail: Free-text explanation shown in the deployment report.
    """

    name: str
    status: GateStatus
    detail: str


@dataclass(slots=True)
class SignalReadinessReport:
    """Aggregates readiness checks and issues a go/no-go decision.

    Attributes:
        signal_id: Unique identifier of the candidate signal.
        checks: Ordered list of ReadinessCheck results.
    """

    signal_id: str
    checks: list[ReadinessCheck] = field(default_factory=list)

    def add(self, name: str, status: GateStatus, detail: str) -> None:
        """Records a single readiness check result.

        Args:
            name: Name of the check being recorded.
            status: PASS, FAIL, or WARN outcome.
            detail: Explanation of the result for audit trail purposes.
        """
        self.checks.append(ReadinessCheck(name, status, detail))

    def is_deployable(self) -> bool:
        """Determines whether the signal clears all hard gates.

        Returns:
            True if no check has status FAIL, else False.
        """
        return all(c.status is not GateStatus.FAIL for c in self.checks)


def evaluate_signal(
    oos_sharpe: float,
    cpcv_p_value: float,
    capacity_usd: float,
    min_capacity_usd: float = 25_000_000.0,
) -> SignalReadinessReport:
    """Runs the standard institutional pre-deployment gate.

    Args:
        oos_sharpe: Out-of-sample Sharpe ratio on the embargoed test fold.
        cpcv_p_value: p-value from Combinatorial Purged CV overfitting test
            (probability the strategy is a false discovery).
        capacity_usd: Estimated strategy capacity before market impact
            erodes the edge below cost.
        min_capacity_usd: Minimum viable capacity for the business.

    Returns:
        A SignalReadinessReport with PASS/FAIL/WARN per check.
    """
    report = SignalReadinessReport(signal_id="candidate")
    report.add(
        "OOS Sharpe >= 0.5",
        GateStatus.PASS if oos_sharpe >= 0.5 else GateStatus.FAIL,
        f"oos_sharpe={oos_sharpe:.2f}",
    )
    report.add(
        "CPCV p-value < 0.05 (not a false discovery)",
        GateStatus.PASS if cpcv_p_value < 0.05 else GateStatus.FAIL,
        f"p={cpcv_p_value:.3f}",
    )
    report.add(
        "Capacity >= desk minimum",
        GateStatus.PASS if capacity_usd >= min_capacity_usd else GateStatus.WARN,
        f"capacity=${capacity_usd:,.0f}",
    )
    return report


if __name__ == "__main__":
    result = evaluate_signal(oos_sharpe=0.71, cpcv_p_value=0.021, capacity_usd=40_000_000)
    print(f"Deployable: {result.is_deployable()}")
    for check in result.checks:
        print(f"  [{check.status.name:4}] {check.name}: {check.detail}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q2 · Python Performance, the GIL & Mechanical Sympathy

**Open with the intuition:**
> "Python is a coordination language, not a compute language. The trick is to spend 95% of wall-clock time inside vectorized NumPy/Numba kernels that release the GIL, and use Python only for orchestration. 'Mechanical sympathy' means writing code that respects how the CPU actually moves data — contiguous memory, cache-line-friendly access patterns, minimal branching in hot loops."

### Where Time Actually Goes

```
LAYER                  TYPICAL COST            MECHANICAL-SYMPATHY FIX
──────────────────    ──────────────────      ────────────────────────────────────
Pure Python loop        O(n), GIL-bound         Vectorize with NumPy (SIMD, C loop)
Pandas .apply()         Row-by-row, boxed       Vectorized ops / .values / Numba
Random memory access    Cache misses ~100 cyc   Column-major / contiguous np.ascontiguousarray
Python object overhead  28+ bytes/int           NumPy dtype arrays (int32/float32)
GIL-bound multithread   No parallel CPU use     multiprocessing / Numba nogil / Cython
```

### The Numba `nogil` Pattern for a Hot Path

$$\text{EWMA}_t = \lambda \cdot x_t + (1-\lambda)\cdot \text{EWMA}_{t-1}$$

**Say it out loud:** *"This recursion has a true sequential data dependency — you can't vectorize it away entirely — so the right tool is a compiled, GIL-releasing loop, not a Python for-loop."*

```python
"""High-throughput EWMA kernel with mechanical-sympathy design."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, nogil=True)
def ewma_kernel(x: np.ndarray, lam: float) -> np.ndarray:
    """Computes an exponentially weighted moving average in a tight loop.

    Uses a contiguous float64 array and a single sequential pass to
    maximize cache-line reuse and allow LLVM to auto-vectorize the
    surrounding arithmetic; nogil=True permits this to run concurrently
    with other Python threads (e.g., I/O for the next data batch).

    Args:
        x: 1-D contiguous array of input observations.
        lam: Smoothing factor in (0, 1]; higher = more reactive.

    Returns:
        1-D array of the same length containing the EWMA path.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = x[0]
    for t in range(1, n):
        out[t] = lam * x[t] + (1.0 - lam) * out[t - 1]
    return out


@njit(cache=True, parallel=True)
def batch_ewma(matrix: np.ndarray, lam: float) -> np.ndarray:
    """Applies ewma_kernel across independent columns in parallel.

    Args:
        matrix: 2-D array, shape (n_timesteps, n_series), column-contiguous
            series that are independent of one another (embarrassingly
            parallel across the second axis).
        lam: Smoothing factor shared across all series.

    Returns:
        2-D array of the same shape with each column smoothed.
    """
    n_t, n_s = matrix.shape
    result = np.empty_like(matrix)
    for j in prange(n_s):
        result[:, j] = ewma_kernel(np.ascontiguousarray(matrix[:, j]), lam)
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2_000_000, 8)).astype(np.float64)
    smoothed = batch_ewma(data, lam=0.06)
    print(f"Shape: {smoothed.shape}, sample tail: {smoothed[-1, :3]}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q3 · Designing a Production-Grade Signal Engine

**Feynman explanation:** A signal engine is a factory line: raw ticks come in one end, a number between -1 and +1 comes out the other, and every station in between (cleaning, feature computation, model inference, risk scaling) must be independently testable, replaceable, and fast enough to never become the bottleneck. I use the **Strategy pattern** so a PM can swap a Ridge model for an LSTM without touching the pipeline plumbing.

```python
"""Signal engine skeleton: pluggable feature pipeline + model strategy."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


class SignalModel(abc.ABC):
    """Abstract interface every alpha model must implement."""

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Maps a feature matrix to raw (unscaled) signal scores.

        Args:
            features: Array of shape (n_assets, n_features).

        Returns:
            Array of shape (n_assets,) with raw signal scores.
        """
        raise NotImplementedError


@dataclass(slots=True)
class RidgeSignalModel(SignalModel):
    """Ridge-regression-based signal model.

    Attributes:
        weights: Fitted coefficient vector, shape (n_features,).
        intercept: Fitted intercept term.
    """

    weights: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Computes Ridge model scores.

        Args:
            features: Array of shape (n_assets, n_features).

        Returns:
            Array of shape (n_assets,) of linear signal scores.
        """
        return features @ self.weights + self.intercept


class SignalEngine:
    """Orchestrates feature computation, inference, and risk scaling.

    Attributes:
        model: A SignalModel implementation (Ridge, GBM, LSTM, ...).
        vol_target: Annualized volatility target for signal scaling.
    """

    def __init__(self, model: SignalModel, vol_target: float = 0.10) -> None:
        """Initializes the engine with a pluggable model and risk target.

        Args:
            model: Concrete SignalModel used for inference.
            vol_target: Target annualized volatility for position sizing.
        """
        self.model = model
        self.vol_target = vol_target

    def generate(self, features: np.ndarray, realized_vol: np.ndarray) -> np.ndarray:
        """Produces volatility-scaled trading signals.

        Args:
            features: Array of shape (n_assets, n_features).
            realized_vol: Per-asset realized annualized vol, shape (n_assets,).

        Returns:
            Array of shape (n_assets,) of signals scaled to vol_target,
            clipped to [-1, 1].
        """
        raw = self.model.predict(features)
        scaled = raw * (self.vol_target / np.maximum(realized_vol, 1e-6))
        return np.clip(scaled, -1.0, 1.0)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_assets, n_features = 12, 5
    model = RidgeSignalModel(weights=rng.normal(size=n_features) * 0.1, intercept=0.0)
    engine = SignalEngine(model, vol_target=0.10)
    feats = rng.normal(size=(n_assets, n_features))
    vol = rng.uniform(0.08, 0.35, size=n_assets)
    signals = engine.generate(feats, vol)
    print(f"Signals: {np.round(signals, 3)}")
```

[🔝 Back to Top](#table-of-contents)

---
---

# 🤖 GEN AI

---

## Q4 · Fine-Tuning vs. Prompt Engineering vs. RAG — and LoRA Math

**Open with the intuition:**
> "These three sit on a spectrum of *how much you change the model* vs. *how much you change the context*. Prompt engineering changes nothing about the weights — cheapest, fastest, best first move. RAG keeps weights frozen but injects fresh, proprietary, and *citable* information at inference time — this is what you want when facts change daily (financing rates) or must be auditable. Fine-tuning changes the weights — you reach for it only when you need a **behavior** change (format compliance, domain jargon, latency via a smaller distilled model), not a **knowledge** change."

### Decision Framework

```
                    Does the answer need FRESH / PROPRIETARY facts?
                                   │
                    ┌───────yes────┴────no──────────┐
                    ▼                               ▼
                   RAG                  Is it a FORMATTING / STYLE /
        (embeddings + retriever         DOMAIN-BEHAVIOR problem?
         + frozen LLM + citations)                  │
                                        ┌────yes────┴────no─────┐
                                        ▼                       ▼
                              FINE-TUNE (LoRA)          PROMPT ENGINEER
                              (compress a workflow      (few-shot, CoT,
                               into weights, or          structured output,
                               distill to smaller,       system prompt)
                               cheaper model)
```

### LoRA — Low-Rank Adaptation, Derived

Full fine-tuning updates weight matrix $W_0 \in \mathbb{R}^{d\times k}$ directly: $W = W_0 + \Delta W$, with $\Delta W$ having $d\times k$ free parameters — for a 7B-parameter model that's billions of trainable values.

LoRA's hypothesis: the *update* $\Delta W$ needed to adapt a pretrained model to a new task has **low intrinsic rank** $r \ll \min(d,k)$. So constrain:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d\times r}, \quad A \in \mathbb{R}^{r\times k}, \quad r \ll \min(d,k)$$

Forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

**Line-by-line:**
1. $W_0$ stays **frozen** — no gradient computed for it, so no optimizer state (Adam moment estimates) needed for the base weights. This is where the memory savings come from.
2. $A$ is initialized from $\mathcal{N}(0,\sigma^2)$, $B$ initialized to **zero** — so at step 0, $BA = 0$ and $h = W_0 x$, i.e., training starts exactly at the pretrained behavior (no cold-start degradation).
3. Only $A$ and $B$ receive gradients: parameter count drops from $d\times k$ to $r(d+k)$. For $d=k=4096,\ r=8$: $16.7\text{M} \to 65.5\text{K}$ — a **256× reduction**.
4. At inference, $B A$ can be **merged** into $W_0$ (since it's just addition), so there is **zero latency overhead** versus the base model — unlike adapter layers that add sequential compute.

**Feynman explanation:** Imagine the pretrained model already speaks fluent "general English" and you just need it to pick up "financing-desk jargon." You don't need to re-teach it English (don't touch $W_0$) — you need a small correction, and that correction, empirically, lives in a low-dimensional subspace. $B$ and $A$ are two thin "translation lenses" bolted onto the frozen brain; the low rank $r$ is the size of that lens.

```python
"""Minimal LoRA linear layer, PyTorch-native, production-style."""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer augmented with a low-rank trainable update.

    Attributes:
        base: Frozen nn.Linear representing W_0.
        lora_a: Trainable projection down to rank r.
        lora_b: Trainable projection back up to output dim (zero-init).
        scaling: alpha / r scaling factor applied to the LoRA branch.
    """

    def __init__(
        self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16
    ) -> None:
        """Initializes the frozen base layer and trainable low-rank factors.

        Args:
            in_features: Input dimension k.
            out_features: Output dimension d.
            rank: Rank r of the low-rank decomposition, r << min(d, k).
            alpha: LoRA scaling numerator; effective scale is alpha / rank.
        """
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes h = W_0 x + scaling * B A x.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + self.scaling * lora_out


if __name__ == "__main__":
    layer = LoRALinear(in_features=4096, out_features=4096, rank=8, alpha=16)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable:,} | Frozen: {frozen:,} | Ratio: {trainable/frozen:.4%}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q5 · RAG Architecture Deep-Dive for Financing Docs

**Open with the intuition:**
> "For Liquid Financing, the highest-value Gen AI use case isn't a chatbot — it's turning ISDA/MSFTA/repo agreements, financing rate sheets, and internal research notes into a queryable, citation-backed knowledge layer that a trader can trust *because it shows its sources*."

### Architecture

```
  DOCS (ISDA, term sheets,        QUERY: "What's the haircut on
  research notes, rate memos)      GC repo vs. specials for HY collateral?"
         │                                   │
         ▼                                   ▼
 ┌─────────────────┐                 ┌─────────────────┐
 │ Chunk (semantic,│                 │ Embed query     │
 │ 512–1024 tok,   │                 │ (same encoder)  │
 │ 10–15% overlap) │                 └───────┬─────────┘
 └───────┬─────────┘                         │
         ▼                                   ▼
 ┌────────────────┐                 ┌────────────────────┐
 │ Embed chunks    │                │ ANN search (HNSW/  │
 │ (bi-encoder)    │────────────── ▶│ IVF-PQ) top-k=50   │
 └───────┬─────────┘                └────────┬───────────┘
         ▼                                   ▼
 ┌────────────────┐                 ┌──────────────────────┐
 │ Vector store   │                 │ Cross-encoder rerank │
 │ (metadata:     │                 │ → top-k=6            │
 │ doc_id, date,  │                 └────────┬─────────────┘
 │ counterparty)  │                          ▼
 └────────────────┘                 ┌──────────────────────┐
                                    │ LLM synthesis w/     │
                                    │ inline citations     │
                                    └────────┬─────────────┘
                                             ▼
                                     Answer + source spans
                                     (auditable for Compliance)
```

**Why cross-encoder rerank matters (the math):** a bi-encoder scores relevance as $\text{sim}(E_q(q), E_d(d)) = \frac{E_q(q)\cdot E_d(d)}{\lVert E_q(q)\rVert \lVert E_d(d)\rVert}$ — query and doc are embedded **independently**, so the model never sees them jointly; it's fast ($O(1)$ per doc at query time) but loses fine-grained token interaction. A cross-encoder instead scores $f(q \oplus d)$ jointly through a full transformer, capturing token-level interaction (e.g., matching "GC" specifically to "general collateral" rather than any repo term) at the cost of $O(N)$ forward passes — hence: bi-encoder for cheap top-50 recall, cross-encoder for expensive top-6 precision.

```python
"""RAG retrieval-and-rerank pipeline skeleton for financing documents."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DocChunk:
    """A retrievable unit of a source document.

    Attributes:
        chunk_id: Unique identifier.
        text: Raw chunk text.
        embedding: Dense vector representation.
        metadata: Source doc_id, date, counterparty, etc.
    """

    chunk_id: str
    text: str
    embedding: np.ndarray
    metadata: dict[str, str]


class RagRetriever:
    """Two-stage dense retrieval: ANN recall + cross-encoder rerank.

    Attributes:
        chunks: In-memory chunk store (swap for FAISS/HNSW in production).
    """

    def __init__(self, chunks: list[DocChunk]) -> None:
        """Initializes the retriever with a pre-embedded chunk corpus.

        Args:
            chunks: List of DocChunk with precomputed embeddings.
        """
        self.chunks = chunks
        self._matrix = np.stack([c.embedding for c in chunks])
        self._norms = np.linalg.norm(self._matrix, axis=1)

    def recall(self, query_embedding: np.ndarray, top_k: int = 50) -> list[DocChunk]:
        """Stage 1: cosine-similarity ANN recall.

        Args:
            query_embedding: Dense embedding of the user query.
            top_k: Number of candidates to recall.

        Returns:
            Top-k DocChunk candidates ranked by cosine similarity.
        """
        q_norm = np.linalg.norm(query_embedding)
        sims = (self._matrix @ query_embedding) / (self._norms * q_norm + 1e-12)
        top_idx = np.argsort(-sims)[:top_k]
        return [self.chunks[i] for i in top_idx]

    def rerank(
        self, query: str, candidates: list[DocChunk], cross_encoder_score_fn, top_k: int = 6
    ) -> list[DocChunk]:
        """Stage 2: cross-encoder rerank of recalled candidates.

        Args:
            query: Raw query text.
            candidates: Chunks returned by recall().
            cross_encoder_score_fn: Callable(query, chunk_text) -> float.
            top_k: Number of chunks to keep for LLM synthesis.

        Returns:
            Top-k DocChunk reranked by joint relevance score.
        """
        scored = [(c, cross_encoder_score_fn(query, c.text)) for c in candidates]
        scored.sort(key=lambda pair: -pair[1])
        return [c for c, _ in scored[:top_k]]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    corpus = [
        DocChunk(f"c{i}", f"clause {i}", rng.normal(size=384), {"doc_id": f"ISDA_{i%5}"})
        for i in range(500)
    ]
    retriever = RagRetriever(corpus)
    recalled = retriever.recall(rng.normal(size=384), top_k=50)
    fake_ce = lambda q, t: rng.random()
    reranked = retriever.rerank("GC repo haircut HY collateral", recalled, fake_ce, top_k=6)
    print(f"Recalled: {len(recalled)}, Final context chunks: {[c.chunk_id for c in reranked]}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q6 · LLM Evaluation — RAGAS, Hallucination Detection, LLM-as-Judge

**Feynman explanation:** You cannot grade an essay with a ruler. LLM outputs are open-ended, so evaluation needs metrics that separate **faithfulness** (did it stick to the retrieved facts?) from **relevance** (did it answer the actual question?) from **correctness** (is it factually right against ground truth?) — conflating these three is the single biggest mistake I see in Gen AI eval.

### Core RAG Metrics, Defined

$$\text{Faithfulness} = \frac{|\{\text{claims in answer supported by context}\}|}{|\{\text{claims in answer}\}|}$$

$$\text{Context Precision} = \frac{1}{\big|\{\text{relevant chunks retrieved}\}\big|}\sum_{k}\text{Precision@}k \cdot \mathbb{1}[\text{relevant}_k]$$

$$\text{Answer Relevance} = \frac{1}{N}\sum_{i=1}^{N}\cos\big(E(q), E(q_i')\big)$$

where $q_i'$ are LLM-generated "reverse questions" from the answer — **if the answer is truly relevant, an LLM asked to reconstruct the question from it should land close to the real query in embedding space.**

**Say it out loud:** *"Faithfulness decomposes the answer into atomic claims and checks each against retrieved context with a second LLM call — it's a hallucination detector for RAG specifically. Answer relevance flips the problem: generate synthetic questions that the answer would address, and if they don't align with what was actually asked, the model likely answered a different (perhaps easier) question — a common failure mode called 'answer drift'."*

```python
"""Lightweight LLM-as-judge evaluation harness for RAG outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class RagEvalResult:
    """Container for a single evaluated (question, answer) pair.

    Attributes:
        faithfulness: Fraction of answer claims supported by context.
        answer_relevance: Cosine similarity between query and reverse-
            engineered questions from the answer.
        context_precision: Weighted precision of retrieved chunk relevance.
    """

    faithfulness: float
    answer_relevance: float
    context_precision: float

    @property
    def composite(self) -> float:
        """Computes an equal-weighted composite RAG quality score.

        Returns:
            Mean of faithfulness, answer_relevance, and context_precision.
        """
        return float(np.mean([self.faithfulness, self.answer_relevance, self.context_precision]))


def evaluate_rag_output(
    claims_supported: int,
    claims_total: int,
    reverse_question_embeddings: np.ndarray,
    original_query_embedding: np.ndarray,
    chunk_relevance_flags: list[bool],
) -> RagEvalResult:
    """Computes faithfulness, answer relevance, and context precision.

    Args:
        claims_supported: Count of atomic answer claims backed by context.
        claims_total: Total atomic claims extracted from the answer.
        reverse_question_embeddings: Embeddings of N synthetic reverse
            questions generated from the answer, shape (N, d).
        original_query_embedding: Embedding of the real user query, (d,).
        chunk_relevance_flags: Binary relevance judgments for each
            retrieved chunk in rank order (True = relevant).

    Returns:
        A populated RagEvalResult.
    """
    faithfulness = claims_supported / max(claims_total, 1)

    q_norm = original_query_embedding / (np.linalg.norm(original_query_embedding) + 1e-12)
    r_norms = reverse_question_embeddings / (
        np.linalg.norm(reverse_question_embeddings, axis=1, keepdims=True) + 1e-12
    )
    answer_relevance = float(np.mean(r_norms @ q_norm))

    hits, weighted_sum = 0, 0.0
    for k, is_relevant in enumerate(chunk_relevance_flags, start=1):
        if is_relevant:
            hits += 1
            weighted_sum += hits / k
    context_precision = weighted_sum / max(hits, 1) if hits else 0.0

    return RagEvalResult(faithfulness, answer_relevance, context_precision)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    result = evaluate_rag_output(
        claims_supported=8,
        claims_total=9,
        reverse_question_embeddings=rng.normal(size=(5, 64)),
        original_query_embedding=rng.normal(size=64),
        chunk_relevance_flags=[True, True, False, True, False, False],
    )
    print(f"Faithfulness={result.faithfulness:.2f}, Relevance={result.answer_relevance:.2f}, "
          f"CtxPrecision={result.context_precision:.2f}, Composite={result.composite:.2f}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q7 · Model Selection Matrix — Claude, GPT, Open-Weight

**Open with the intuition:**
> "I don't pick a 'favorite model' — I pick the cheapest model that clears the accuracy bar for a given task, and I route by task type. The desk uses Claude Code heavily for agentic engineering work, but a routing/classification microservice with a 50ms SLA has no business calling a frontier model."

```
TASK TYPE                              RECOMMENDED CLASS                WHY
──────────────────────────────────    ────────────────────────────    ─────────────────────────────
Agentic coding / repo-scale            Claude (Sonnet/Opus class)       Best long-horizon tool-use,
refactors, CI-integrated workflows     via Claude Code                  strong instruction-following
                                                                        over multi-file context

Long-document synthesis (ISDA,         Claude (large context)           Long context window +
research notes), RAG generation                                        low hallucination rate on
                                                                        grounded QA

Structured extraction, JSON-strict     Smaller distilled / fine-tuned   Deterministic, cheap, fast;
microservices (trade capture,          open-weight model (Llama/        JSON mode + low temp; frontier
term-sheet parsing) at high QPS        Mistral class) or GPT-4o-mini    model is overkill and too slow

Numerical/quantitative reasoning,      Frontier reasoning model         Chain-of-thought reasoning
multi-step research questions          (Claude Opus / GPT o-series)     models materially outperform
                                                                        base LLMs on multi-step math

On-prem / data-residency constrained   Open-weight (Llama 3, Mistral,   Full control of weights and
workloads (compliance-sensitive)       DeepSeek) self-hosted            inference infra; no data
                                                                        leaves the firm's VPC

Embeddings for RAG                     Open-weight embedding model      Cheap at scale, no need for
                                        (BGE, E5) or provider embedding  frontier reasoning capability
```

**Feynman explanation:** Model selection is a cost-quality-latency Pareto frontier, not a leaderboard. Ask three questions in order: (1) does this need reasoning depth or just pattern extraction? (2) does data residency/compliance force self-hosting? (3) what's my latency SLA? Only after answering those do you pick a specific model family — and you re-evaluate quarterly, because the frontier moves fast (which is why I'd suggest an internal model-eval harness rather than hard-coding a choice).

[🔝 Back to Top](#table-of-contents)

---
---

## Q8 · Structured Extraction & Prompt Engineering for Term Sheets

**Feynman explanation:** For extracting structured fields (haircut %, collateral eligibility, tenor) out of unstructured financing agreements, the prompt itself is the "schema contract." I always (1) give the exact JSON schema, (2) give 2–3 few-shot examples spanning edge cases (missing fields, ambiguous units), (3) force the model to cite the source span for each field so it can be audited, and (4) validate the output programmatically (Pydantic) rather than trusting it blindly.

```python
"""Structured extraction pipeline with schema validation."""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError


class FinancingTerms(BaseModel):
    """Schema for extracted repo/financing agreement terms.

    Attributes:
        haircut_pct: Collateral haircut as a percentage (0-100).
        tenor_days: Financing tenor in calendar days.
        collateral_type: Eligible collateral category string.
        source_span: Verbatim quote supporting the extraction, for audit.
    """

    haircut_pct: float = Field(ge=0.0, le=100.0)
    tenor_days: int = Field(ge=0)
    collateral_type: str
    source_span: str


EXTRACTION_SYSTEM_PROMPT = """You are a financing-terms extraction engine.
Return ONLY valid JSON matching this schema, no prose:
{"haircut_pct": float, "tenor_days": int, "collateral_type": str, "source_span": str}
Rules:
- source_span MUST be a verbatim quote from the document supporting the fields.
- If a field is not present in the document, use null and explain in source_span.
- Never infer numeric values that are not explicitly stated."""


def validate_extraction(raw_json: dict) -> FinancingTerms | None:
    """Validates an LLM's structured extraction against the Pydantic schema.

    Args:
        raw_json: Parsed JSON dict returned by the LLM.

    Returns:
        A validated FinancingTerms instance, or None if validation fails.
    """
    try:
        return FinancingTerms.model_validate(raw_json)
    except ValidationError as exc:
        print(f"Schema validation failed, routing to human review: {exc}")
        return None


if __name__ == "__main__":
    sample_llm_output = {
        "haircut_pct": 8.5,
        "tenor_days": 30,
        "collateral_type": "HY Corporate Bonds",
        "source_span": "the Haircut applicable to High Yield Collateral shall be 8.5%",
    }
    parsed = validate_extraction(sample_llm_output)
    print(parsed)
```

[🔝 Back to Top](#table-of-contents)

---
---

# 📉 REGRESSION

---

## Q9 · OLS — Normal Equations & Gauss-Markov Proof

**Setup:** $y = X\beta + \varepsilon$, $y\in\mathbb{R}^n$, $X\in\mathbb{R}^{n\times k}$, $\varepsilon$ mean-zero.

**Derive the estimator.** Minimize the residual sum of squares:

$$S(\beta) = (y-X\beta)^\top(y-X\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$$

Take the gradient and set to zero:

$$\frac{\partial S}{\partial \beta} = -2X^\top y + 2X^\top X\beta = 0 \quad\Rightarrow \quad X^\top X\,\hat\beta = X^\top y \quad \Rightarrow \quad \hat\beta = (X^\top X)^{-1}X^\top y$$

**Line-by-line:** the first term $-2X^\top y$ is the derivative of the linear cross-term $-2\beta^\top X^\top y$ with respect to $\beta$ (a standard $\partial(a^\top\beta)/\partial\beta = a$ rule, doubled by the scalar out front); the second term comes from $\partial(\beta^\top A \beta)/\partial\beta = 2A\beta$ for symmetric $A = X^\top X$. Setting the gradient to zero gives the **normal equations**; because $S(\beta)$ is convex (Hessian $2X^\top X \succeq 0$), this stationary point is the global minimum, and it's unique whenever $X^\top X$ is invertible (i.e., $X$ has full column rank — no perfect multicollinearity).

**Gauss-Markov proof sketch (OLS is BLUE — Best Linear Unbiased Estimator):**

*Unbiasedness:* $\hat\beta = (X^\top X)^{-1}X^\top(X\beta+\varepsilon) = \beta + (X^\top X)^{-1}X^\top\varepsilon$. Taking expectation with $X$ fixed and $\mathbb{E}[\varepsilon]=0$: $\mathbb{E}[\hat\beta]=\beta$.

*Minimum variance among linear unbiased estimators:* Consider any other linear unbiased estimator $\tilde\beta = Cy$ with $C = (X^\top X)^{-1}X^\top + D$ for some matrix $D$. Unbiasedness of $\tilde\beta$ forces $DX = 0$. Then:

$$\text{Var}(\tilde\beta) = \sigma^2 CC^\top = \sigma^2\Big[(X^\top X)^{-1} + DD^\top\Big]$$

since the cross-terms vanish because $DX=0$. Because $DD^\top \succeq 0$ (positive semi-definite), $\text{Var}(\tilde\beta) \succeq \text{Var}(\hat\beta)$ — OLS achieves the minimum variance, with equality only when $D=0$, i.e., $\tilde\beta = \hat\beta$. $\blacksquare$

**Feynman explanation:** OLS finds the line that minimizes total squared vertical distance to the data — squaring instead of taking absolute value makes the problem smooth and solvable in closed form (calculus). Gauss-Markov is the guarantee that, **as long as your errors are unbiased, homoskedastic, and uncorrelated with each other**, no other unbiased "linear recipe" for combining the data can beat OLS's variance — but the moment those assumptions break (financial time series: heteroskedastic, autocorrelated), OLS is still unbiased but no longer *efficient*, which motivates Q11.

```python
"""OLS estimator with closed-form solution and diagnostic residual plot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(slots=True)
class OlsResult:
    """Holds fitted OLS coefficients and diagnostics.

    Attributes:
        beta: Fitted coefficient vector, shape (k,).
        residuals: Fitted residuals, shape (n,).
        r_squared: Coefficient of determination.
    """

    beta: np.ndarray
    residuals: np.ndarray
    r_squared: float


def fit_ols(X: np.ndarray, y: np.ndarray) -> OlsResult:
    """Fits OLS via the normal equations using a stable QR solve.

    Args:
        X: Design matrix of shape (n, k), first column typically ones
            for the intercept.
        y: Response vector of shape (n,).

    Returns:
        An OlsResult with beta, residuals, and R^2.
    """
    q_mat, r_mat = np.linalg.qr(X)
    beta = np.linalg.solve(r_mat, q_mat.T @ y)
    fitted = X @ beta
    residuals = y - fitted
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot
    return OlsResult(beta=beta, residuals=residuals, r_squared=r_squared)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 500, 3
    X_design = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    true_beta = np.array([1.5, -2.0, 0.8])
    y_obs = X_design @ true_beta + rng.normal(scale=1.0, size=n)

    result = fit_ols(X_design, y_obs)
    print(f"Fitted beta: {np.round(result.beta, 3)}, R^2={result.r_squared:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_design @ result.beta, y=result.residuals,
                              mode="markers", marker=dict(size=5, color="steelblue"),
                              name="Residuals"))
    fig.add_hline(y=0, line_dash="dash", line_color="crimson")
    fig.update_layout(title="OLS Residuals vs. Fitted Values",
                       xaxis_title="Fitted", yaxis_title="Residual",
                       template="plotly_white")
    fig.write_html("/tmp/ols_residuals.html")
    print("Saved plot to /tmp/ols_residuals.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q10 · Ridge, Lasso & Elastic Net — Derivation and Bias-Variance

**Ridge — closed form.** Add an $\ell_2$ penalty:

$$S_{\text{ridge}}(\beta) = (y-X\beta)^\top(y-X\beta) + \lambda\beta^\top\beta$$

$$\frac{\partial S_{\text{ridge}}}{\partial\beta} = -2X^\top y + 2X^\top X\beta + 2\lambda\beta = 0 \quad \Rightarrow \quad \hat\beta_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$$

**Say it out loud:** **"Adding $\lambda I$ to $X^\top X$ before inverting guarantees invertibility even when $X$ is rank-deficient or near-collinear — this is literally 'ridge' because you're adding a ridge along the diagonal — and it shrinks every coefficient toward zero, trading a little bias for a large reduction in variance whenever predictors are correlated (which financial factors always are)."**

**Lasso — no closed form, subgradient condition.** The $\ell_1$ penalty $\lambda\lVert\beta\rVert_1$ is non-differentiable at 0. The KKT/subgradient stationarity condition for coordinate $j$:

$$
-2x_j^\top(y - X\beta) + \lambda\, s_j = 0, \quad s_j \in 
\begin{cases}
\{\text{sign}(\beta_j)\} & \beta_j\neq 0 \\
[-1,1] & \beta_j = 0
\end{cases}
$$

This is why Lasso produces **exact sparsity**: whenever the correlation of a feature with the residual is smaller in magnitude than $\lambda/2$, the optimal $\beta_j$ is driven exactly to 0 (a corner solution of the $\ell_1$ ball), unlike Ridge's smooth, everywhere-differentiable penalty which shrinks but never zeroes out.

**Elastic Net** blends both:

$$S_{\text{EN}}(\beta) = \lVert y-X\beta\rVert_2^2 + \lambda_1\lVert\beta\rVert_1 + \lambda_2\lVert\beta\rVert_2^2$$

solved via **coordinate descent** — cycling through $j=1,\dots,k$, holding all other coefficients fixed, and applying the soft-thresholding operator:

$$\hat\beta_j \leftarrow \frac{S\big(x_j^\top r_{-j},\ \lambda_1/2\big)}{1 + \lambda_2}, \qquad S(z,\gamma) = \text{sign}(z)\max(|z|-\gamma,\,0)$$

where $r_{-j} = y - X_{-j}\beta_{-j}$ is the partial residual excluding feature $j$.

**Feynman explanation — the bias-variance trade:**

$$\text{MSE}(\hat\beta) = \underbrace{\text{Bias}(\hat\beta)^2}_{\text{how wrong on average}} + \underbrace{\text{Var}(\hat\beta)}_{\text{how much it wiggles across samples}}$$

OLS sits at zero bias but can have huge variance when features are correlated (near-singular $X^\top X$ means tiny data perturbations swing $\hat\beta$ wildly). Ridge/Lasso/EN accept a small, known amount of bias to buy a much larger reduction in variance — like a mechanic detuning an engine slightly to make it more reliable across road conditions rather than perfectly tuned for one road.

```python
"""Elastic Net via cyclic coordinate descent, from scratch."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def soft_threshold(z: float, gamma: float) -> float:
    """Applies the soft-thresholding operator used in Lasso/EN updates.

    Args:
        z: Raw coordinate-wise correlation with the partial residual.
        gamma: Threshold, lambda_1 / 2.

    Returns:
        Soft-thresholded value: sign(z) * max(|z| - gamma, 0).
    """
    return float(np.sign(z) * max(abs(z) - gamma, 0.0))


def fit_elastic_net(
    X: np.ndarray, y: np.ndarray, lambda1: float, lambda2: float,
    n_iter: int = 200, tol: float = 1e-8,
) -> np.ndarray:
    """Fits Elastic Net regression via cyclic coordinate descent.

    Args:
        X: Standardized design matrix, shape (n, k) (no intercept column;
            assume y and X are centered).
        y: Centered response vector, shape (n,).
        lambda1: L1 penalty weight.
        lambda2: L2 penalty weight.
        n_iter: Maximum coordinate-descent sweeps.
        tol: Convergence tolerance on max coefficient change per sweep.

    Returns:
        Fitted coefficient vector, shape (k,).
    """
    n, k = X.shape
    beta = np.zeros(k)
    col_sq = (X ** 2).sum(axis=0)

    for _ in range(n_iter):
        beta_old = beta.copy()
        for j in range(k):
            residual_partial = y - X @ beta + X[:, j] * beta[j]
            rho_j = X[:, j] @ residual_partial
            beta[j] = soft_threshold(rho_j, lambda1 / 2) / (col_sq[j] + lambda2)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n, k = 300, 20
    X_raw = rng.normal(size=(n, k))
    X_std = (X_raw - X_raw.mean(0)) / X_raw.std(0)
    true_beta = np.zeros(k)
    true_beta[[0, 3, 7]] = [3.0, -2.0, 1.5]
    y_raw = X_std @ true_beta + rng.normal(scale=0.5, size=n)
    y_c = y_raw - y_raw.mean()

    fitted = fit_elastic_net(X_std, y_c, lambda1=0.5, lambda2=0.1)
    print(f"Nonzero coefficients recovered at indices: {np.nonzero(fitted)[0]}")

    fig = go.Figure(go.Bar(x=[f"b{i}" for i in range(k)], y=fitted, marker_color="teal"))
    fig.update_layout(title="Elastic Net Coefficients (sparsity from L1 term)",
                       template="plotly_white")
    fig.write_html("/tmp/elastic_net_coefs.html")
    print("Saved plot to /tmp/elastic_net_coefs.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q11 · Heteroskedasticity, Autocorrelation & Newey-West

**Feynman explanation:** OLS point estimates $\hat\beta$ stay unbiased even when errors are heteroskedastic or autocorrelated — the *point forecast* is fine. What breaks is the **standard errors**, because the usual formula $\text{Var}(\hat\beta)=\sigma^2(X^\top X)^{-1}$ assumes $\text{Var}(\varepsilon)=\sigma^2 I$. In financial time series, volatility clusters (heteroskedastic) and shocks persist (autocorrelated), so naive t-stats are wrong — usually overstated — and you'll think a factor is significant when it isn't.

**Derivation — the sandwich estimator.** The true variance of $\hat\beta$ is:

$$\text{Var}(\hat\beta) = (X^\top X)^{-1}X^\top \Omega X (X^\top X)^{-1}, \qquad \Omega = \text{Var}(\varepsilon)$$

**Line-by-line:** starting from $\hat\beta - \beta = (X^\top X)^{-1}X^\top\varepsilon$, we get $\text{Var}(\hat\beta) = (X^\top X)^{-1}X^\top\,\text{Var}(\varepsilon)\,X(X^\top X)^{-1}$ by the linear-transformation variance rule $\text{Var}(A\varepsilon)=A\,\Omega\,A^\top$. Under classical (homoskedastic, uncorrelated) assumptions, $\Omega=\sigma^2 I$ and the middle term collapses back to the textbook formula — the "bread-meat-bread" sandwich only matters when $\Omega \neq \sigma^2 I$.

**Newey-West** estimates the "meat" $X^\top\Omega X$ using sample residuals $\hat\varepsilon_t = y_t - x_t^\top\hat\beta$ with a truncated-lag, tapered autocovariance sum:

$$\widehat{X^\top\Omega X} = \sum_{t=1}^n \hat\varepsilon_t^2 x_t x_t^\top + \sum_{l=1}^{L}\left(1-\frac{l}{L+1}\right)\sum_{t=l+1}^{n}\hat\varepsilon_t\hat\varepsilon_{t-l}\big(x_t x_{t-l}^\top + x_{t-l}x_t^\top\big)$$

**Say it out loud:** **"The first term is the standard heteroskedasticity-robust (White) correction — weight each observation's contribution by its own squared residual instead of assuming a common $\sigma^2$. The second term adds cross-products of residuals at lags 1 through $L$, down-weighted linearly (the Bartlett kernel) so more distant lags contribute less — this captures the fact that today's residual is correlated with yesterday's, which OLS otherwise ignores entirely."** The lag truncation **$L$** is typically set via **$L = \lfloor 4(n/100)^{2/9}\rfloor$** (Newey-West's rule of thumb) to guarantee the resulting matrix stays positive semi-definite.

```python
"""Newey-West HAC standard errors for OLS coefficients."""
from __future__ import annotations

import numpy as np


def newey_west_se(X: np.ndarray, residuals: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Computes Newey-West heteroskedasticity- and autocorrelation-
    consistent (HAC) standard errors for OLS coefficients.

    Args:
        X: Design matrix used in the original OLS fit, shape (n, k).
        residuals: OLS residuals, shape (n,).
        max_lag: Bartlett-kernel truncation lag L; defaults to the
            Newey-West rule of thumb floor(4*(n/100)^(2/9)).

    Returns:
        Vector of HAC standard errors, shape (k,).
    """
    n, k = X.shape
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    xtx_inv = np.linalg.inv(X.T @ X)
    scores = X * residuals[:, None]  # (n, k), each row x_t * eps_t

    meat = scores.T @ scores  # lag-0 term
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    sandwich = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.diag(sandwich))


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 1000
    X_design = np.column_stack([np.ones(n), rng.normal(size=n)])
    true_beta = np.array([0.5, 1.2])

    # Simulate AR(1) heteroskedastic errors (volatility clustering).
    eps = np.zeros(n)
    vol = np.ones(n)
    for t in range(1, n):
        vol[t] = 0.9 * vol[t - 1] + 0.1 * abs(eps[t - 1]) + 0.05
        eps[t] = 0.6 * eps[t - 1] + rng.normal(scale=vol[t])

    y_obs = X_design @ true_beta + eps
    beta_hat = np.linalg.lstsq(X_design, y_obs, rcond=None)[0]
    resid = y_obs - X_design @ beta_hat

    naive_se = np.sqrt(np.diag(
        (resid @ resid / (n - X_design.shape[1])) * np.linalg.inv(X_design.T @ X_design)
    ))
    hac_se = newey_west_se(X_design, resid)
    print(f"Naive SE:  {np.round(naive_se, 4)}")
    print(f"HAC SE:    {np.round(hac_se, 4)}  (typically wider under clustering/autocorrelation)")
```

[🔝 Back to Top](#table-of-contents)

---
---

# 🌳 TREE-BASED MODELS

---

## Q12 · Decision Trees — Entropy, Gini & Information Gain

**Definitions.** For a node with class proportions $p_1,\dots,p_C$:

$$\text{Entropy}(p) = -\sum_{c=1}^C p_c \log_2 p_c \qquad\qquad \text{Gini}(p) = \sum_{c=1}^C p_c(1-p_c) = 1-\sum_{c=1}^C p_c^2$$

**Information Gain** of a candidate split $s$ that partitions node $N$ into children $N_L, N_R$:

$$IG(N,s) = \text{Entropy}(N) - \left(\frac{|N_L|}{|N|}\text{Entropy}(N_L) + \frac{|N_R|}{|N|}\text{Entropy}(N_R)\right)$$

**Line-by-line:** entropy measures the average number of bits needed to encode the class label given the current node's distribution — a pure node (all one class) has entropy 0 (no surprise, no bits needed); a 50/50 node has entropy 1 (maximal uncertainty for two classes). Information gain is the **reduction** in that uncertainty after the split, weighted by how large each child is (so a split that only helps a tiny subpopulation contributes little). The greedy tree-growing algorithm picks, at every node, the feature/threshold pair that **maximizes** IG — this is a locally greedy, NP-hard-to-solve-globally heuristic (finding the truly optimal tree is NP-complete), which is exactly why ensembles (Q13) are needed to control variance.

**Why Gini in practice (CART default):** Gini avoids the $\log$ computation (cheaper), and empirically tracks entropy closely — both are concave, both are zero at purity, both maximize at the uniform distribution — so the *chosen splits* rarely differ meaningfully; Gini is preferred purely for computational efficiency at scale.

```python
"""Entropy/Gini-based split evaluation used inside a decision tree grower."""
from __future__ import annotations

import numpy as np


def entropy(labels: np.ndarray) -> float:
    """Computes Shannon entropy of a discrete label array.

    Args:
        labels: 1-D integer array of class labels.

    Returns:
        Entropy in bits.
    """
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Computes information gain of a binary split.

    Args:
        parent: Labels at the parent node before the split.
        left: Labels routed to the left child.
        right: Labels routed to the right child.

    Returns:
        Information gain (bits) of this split relative to the parent.
    """
    n = len(parent)
    weighted_child_entropy = (
        len(left) / n * entropy(left) + len(right) / n * entropy(right)
    )
    return entropy(parent) - weighted_child_entropy


def best_split(feature: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Finds the threshold on a single feature maximizing information gain.

    Args:
        feature: 1-D continuous feature values, shape (n,).
        labels: 1-D class labels aligned with feature, shape (n,).

    Returns:
        Tuple of (best_threshold, best_information_gain).
    """
    order = np.argsort(feature)
    sorted_feat, sorted_labels = feature[order], labels[order]
    candidates = (sorted_feat[:-1] + sorted_feat[1:]) / 2.0

    best_thresh, best_gain = float("nan"), -np.inf
    for thresh in np.unique(candidates):
        mask = feature <= thresh
        if mask.sum() == 0 or mask.sum() == len(feature):
            continue
        gain = information_gain(labels, labels[mask], labels[~mask])
        if gain > best_gain:
            best_thresh, best_gain = thresh, gain
    return best_thresh, best_gain


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    feat = np.concatenate([rng.normal(-1, 0.5, 100), rng.normal(2, 0.5, 100)])
    lbls = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    thresh, gain = best_split(feat, lbls)
    print(f"Best threshold={thresh:.3f}, Information Gain={gain:.3f} bits")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q13 · Random Forest vs. Gradient Boosting — XGBoost 2nd-Order Taylor Expansion

```
DIMENSION            RANDOM FOREST                        GRADIENT BOOSTING (XGBoost)
──────────────────  ────────────────────────────────    ──────────────────────────────────
Ensembling logic     Bagging: parallel trees on           Boosting: sequential trees, each
                      bootstrap samples + feature          fitting the RESIDUAL/gradient of
                      subsampling, averaged                the previous ensemble
Bias/Variance         Reduces VARIANCE (trees are          Reduces BIAS (each stage
                       decorrelated, roughly unbiased       actively corrects prior errors);
                       individually)                        variance controlled via
                                                            shrinkage/regularization
Overfit risk          Lower — averaging is inherently       Higher — needs careful
                       robust                                learning-rate/depth/early-stop
Parallelizable         Fully (trees independent)             Sequential in boosting rounds
                                                            (each tree needs prior tree's
                                                            gradient), parallel within a tree
```

**XGBoost objective, derived via 2nd-order Taylor expansion.** At boosting round $t$, we add a new tree $f_t$ to the ensemble prediction $\hat y_i^{(t-1)}$ to minimize:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n \ell\big(y_i,\ \hat y_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t)$$

Taylor-expand $\ell$ around $\hat y_i^{(t-1)}$ to second order in $f_t(x_i)$:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \Big[\ell(y_i,\hat y_i^{(t-1)}) + g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\Big] + \Omega(f_t)$$

where $g_i = \partial_{\hat y}\ell(y_i,\hat y_i^{(t-1)})$ (gradient) and $h_i = \partial^2_{\hat y}\ell(y_i,\hat y_i^{(t-1)})$ (Hessian).

**Line-by-line:** drop the constant $\ell(y_i,\hat y_i^{(t-1)})$ term (doesn't depend on $f_t$). Write the tree as $f_t(x)=w_{q(x)}$, mapping each input to a leaf weight $w_j$ via structure function $q$, and let $I_j = \{i: q(x_i)=j\}$ be the set of training points landing in leaf $j$. Regularization $\Omega(f_t) = \gamma T + \tfrac12\lambda\sum_j w_j^2$ (T = number of leaves) penalizes tree complexity directly. Substituting and grouping by leaf:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T}\Big[\big(\textstyle\sum_{i\in I_j} g_i\big) w_j + \tfrac12\big(\textstyle\sum_{i\in I_j} h_i + \lambda\big) w_j^2\Big] + \gamma T$$

This is now a **separate quadratic in each $w_j$** — take $\partial/\partial w_j = 0$:

$$w_j^\ast = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda} = -\frac{G_j}{H_j+\lambda}$$

Plugging back in gives the closed-form optimal objective value for a fixed tree structure — this is exactly what XGBoost uses as its **split-quality gain formula**:

$$\text{Gain} = \tfrac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

**Feynman explanation:** Ordinary gradient boosting only uses the gradient (first derivative — "which direction to move"). XGBoost's insight is that the Hessian (second derivative — "how curved the loss is here, i.e., how confident to be about that direction") is nearly free to compute for standard losses (squared error, logloss) and dramatically improves convergence, exactly the way Newton's method beats plain gradient descent. The gain formula is literally "how much does splitting this leaf reduce the loss, penalized by the cost $\gamma$ of adding a new leaf" — this is why XGBoost naturally supports pruning: if Gain $<0$, don't split.

```python
"""Minimal single-tree gradient-boosting split search (XGBoost-style gain)."""
from __future__ import annotations

import numpy as np


def xgboost_split_gain(
    gradients: np.ndarray, hessians: np.ndarray, left_mask: np.ndarray,
    reg_lambda: float = 1.0, reg_gamma: float = 0.0,
) -> float:
    """Computes the exact XGBoost split-quality gain for a candidate split.

    Args:
        gradients: First-order loss gradients g_i for all points in the
            current node, shape (n,).
        hessians: Second-order loss Hessians h_i for all points in the
            current node, shape (n,).
        left_mask: Boolean mask assigning points to the left child.
        reg_lambda: L2 regularization on leaf weights.
        reg_gamma: Minimum-gain complexity penalty per additional leaf.

    Returns:
        The split gain; negative or below reg_gamma implies "do not split."
    """
    g_left, h_left = gradients[left_mask].sum(), hessians[left_mask].sum()
    g_right, h_right = gradients[~left_mask].sum(), hessians[~left_mask].sum()
    g_all, h_all = g_left + g_right, h_left + h_right

    gain = 0.5 * (
        g_left ** 2 / (h_left + reg_lambda)
        + g_right ** 2 / (h_right + reg_lambda)
        - g_all ** 2 / (h_all + reg_lambda)
    ) - reg_gamma
    return float(gain)


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 200
    y_true = rng.normal(size=n)
    y_pred = np.zeros(n)  # squared-error loss: g = pred - y, h = 1
    grad = y_pred - y_true
    hess = np.ones(n)

    feature = rng.normal(size=n)
    mask = feature <= np.median(feature)
    gain = xgboost_split_gain(grad, hess, mask, reg_lambda=1.0, reg_gamma=0.1)
    print(f"Split gain at median threshold: {gain:.4f}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q14 · Feature Importance & SHAP Values

**Feynman explanation:** "Gain" importance (average loss reduction attributed to a feature across all its splits) is fast but biased toward high-cardinality features and doesn't explain *individual predictions*. SHAP fixes this by asking a game-theoretic question: **"if features are players cooperating to produce a prediction, how do we fairly split the credit?"** — the answer is the unique allocation satisfying efficiency, symmetry, dummy, and additivity axioms, given by the **Shapley value**:

$$\phi_i = \sum_{S \subseteq F\setminus\{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{i\}) - v(S)\big)\Big]$$

**Line-by-line:** $F$ is the full feature set, $S$ ranges over every subset **not** containing feature $i$, and $v(S)$ is the model's expected prediction using only features in $S$ (marginalizing out the rest). The bracket is the **marginal contribution** of adding feature $i$ to coalition $S$. The combinatorial weight $\frac{|S|!(|F|-|S|-1)!}{|F|!}$ is the probability of that particular ordering of features arriving in a random permutation — so $\phi_i$ is literally **the average marginal contribution of feature $i$, averaged over every possible order in which features could be "revealed" to the model**. This is exponential to compute exactly ($2^{|F|}$ coalitions), which is why TreeSHAP exploits tree structure to compute it in polynomial time by tracking feature-conditioning paths through the tree exactly.

```python
"""Permutation-based approximate SHAP value estimator (model-agnostic)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def approximate_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimates Shapley values via Monte Carlo permutation sampling.

    Args:
        predict_fn: Callable mapping a feature matrix (m, k) to predictions
            (m,).
        instance: The single instance being explained, shape (k,).
        background: Reference dataset used to marginalize out "missing"
            features, shape (n_bg, k).
        n_permutations: Number of random feature orderings to average over.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Approximate Shapley value per feature, shape (k,).
    """
    rng = rng or np.random.default_rng()
    k = instance.shape[0]
    phi = np.zeros(k)

    for _ in range(n_permutations):
        perm = rng.permutation(k)
        baseline_idx = rng.integers(0, background.shape[0])
        composite = background[baseline_idx].copy()
        prev_pred = float(predict_fn(composite[None, :]))

        for feature_idx in perm:
            composite[feature_idx] = instance[feature_idx]
            new_pred = float(predict_fn(composite[None, :]))
            phi[feature_idx] += new_pred - prev_pred
            prev_pred = new_pred

    return phi / n_permutations


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    true_w = np.array([2.0, -1.0, 0.5, 0.0])
    model = lambda X: X @ true_w
    bg = rng.normal(size=(50, 4))
    x0 = np.array([1.5, 2.0, -0.5, 3.0])

    shap_vals = approximate_shap_values(model, x0, bg, n_permutations=500, rng=rng)
    print(f"Approx SHAP values: {np.round(shap_vals, 3)}")
    print(f"Sum(SHAP) + E[f(bg)] ≈ f(x0)? "
          f"{shap_vals.sum() + model(bg).mean():.3f} vs {model(x0):.3f}")
```

[🔝 Back to Top](#table-of-contents)

---
---

# ⏱️ TIME SERIES & FORECASTING

---

## Q15 · Stationarity, ADF Test & ARIMA

**Feynman explanation:** Most time-series theory assumes the statistical properties (mean, variance, autocorrelation) don't change over time — "stationarity." Financing spreads and rate curves are usually **not** stationary in levels (a repo rate can trend with the Fed funds rate for years), but their **differences** (day-over-day changes) often are — which is exactly the "I" (Integrated) in ARIMA.

**Augmented Dickey-Fuller test.** Test $H_0:$ unit root exists (non-stationary) against $H_1:$ stationary, using the regression:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p}\delta_i \Delta y_{t-i} + \varepsilon_t$$

**Say it out loud:** **"If $\gamma = 0$, then $y_{t-1}$ has no explanatory power for the change $\Delta y_t$ beyond the lagged differences — meaning today's level doesn't pull tomorrow's change back toward anything, which is the definition of a random walk / unit root. If $\gamma < 0$ and statistically significant, the series exhibits mean reversion: a high $y_{t-1}$ predicts a negative $\Delta y_t$, pulling the series back down."** The ADF test statistic is **$t_\gamma = \hat\gamma / SE(\hat\gamma)$**, compared against **non-standard** critical values (not the usual t-distribution, because under **$H_0$** the process is a random walk and the asymptotic distribution is the Dickey-Fuller distribution).

**ARIMA(p,d,q)**: difference $d$ times to induce stationarity, then model as AR(p) + MA(q):

$$\left(1-\sum_{i=1}^p \phi_i L^i\right)(1-L)^d y_t = \left(1+\sum_{j=1}^q \theta_j L^j\right)\varepsilon_t$$

where $L$ is the lag operator ($Ly_t = y_{t-1}$). **Line-by-line:** the left side says "the $d$-times-differenced series, minus a weighted sum of its own past $p$ values, equals..." and the right side says "...a white-noise shock plus a weighted sum of the past $q$ shocks." AR captures persistence (momentum/mean-reversion), MA captures the decay of a one-off shock's influence.

```python
"""ADF test statistic (from scratch) + ARIMA(1,1,1) fit via statsmodels."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA


def adf_statistic(series: np.ndarray, n_lags: int = 1) -> float:
    """Computes the Augmented Dickey-Fuller test statistic from scratch.

    Args:
        series: 1-D time series (levels).
        n_lags: Number of lagged difference terms to include.

    Returns:
        The ADF t-statistic on the gamma (unit-root) coefficient.
    """
    y = series
    dy = np.diff(y)
    n = len(dy) - n_lags

    y_lag1 = y[n_lags:-1]
    cols = [np.ones(n), y_lag1]
    for lag in range(1, n_lags + 1):
        cols.append(dy[n_lags - lag:-lag] if lag > 0 else dy[n_lags:])
    X_design = np.column_stack(cols)
    target = dy[n_lags:]

    beta, *_ = np.linalg.lstsq(X_design, target, rcond=None)
    residuals = target - X_design @ beta
    dof = len(target) - X_design.shape[1]
    sigma2 = (residuals @ residuals) / dof
    cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
    gamma_idx = 1
    return float(beta[gamma_idx] / np.sqrt(cov_beta[gamma_idx, gamma_idx]))


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    n = 500
    # Simulate a mean-reverting financing spread (OU-like) vs. a random walk.
    mean_rev = np.zeros(n)
    for t in range(1, n):
        mean_rev[t] = mean_rev[t - 1] + 0.05 * (0.0 - mean_rev[t - 1]) + rng.normal(scale=0.1)
    random_walk = np.cumsum(rng.normal(scale=0.1, size=n))

    print(f"ADF stat (mean-reverting spread): {adf_statistic(mean_rev):.3f} (more negative = stationary)")
    print(f"ADF stat (random walk):           {adf_statistic(random_walk):.3f}")

    model = ARIMA(mean_rev, order=(1, 1, 1)).fit()
    forecast = model.get_forecast(steps=20)
    mean_fc = forecast.predicted_mean
    ci = forecast.conf_int()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mean_rev, name="Observed Spread", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=np.arange(n, n + 20), y=mean_fc, name="ARIMA(1,1,1) Forecast",
                              line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=np.arange(n, n + 20), y=ci[:, 0], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(n, n + 20), y=ci[:, 1], line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(220,20,60,0.15)", name="95% CI"))
    fig.update_layout(title="ARIMA(1,1,1) Financing Spread Forecast", template="plotly_white")
    fig.write_html("/tmp/arima_forecast.html")
    print("Saved plot to /tmp/arima_forecast.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q16 · GARCH/EGARCH — Deriving Volatility Forecasts

**Motivation:** returns are approximately unpredictable in mean but **volatility clusters** — big moves follow big moves. GARCH(1,1) models the conditional variance:

$$\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2, \qquad \varepsilon_t = \sigma_t z_t,\ z_t\sim N(0,1)$$

**Line-by-line:** $\omega>0$ is a floor (long-run variance component), $\alpha \varepsilon_{t-1}^2$ says "yesterday's squared surprise raises today's variance" (the ARCH term — reactive/shock component), $\beta \sigma_{t-1}^2$ says "yesterday's variance level persists into today" (the GARCH term — memory/persistence component). Unconditional (long-run) variance is found by setting $\sigma_t^2=\sigma_{t-1}^2=\bar\sigma^2$ and $\mathbb{E}[\varepsilon_{t-1}^2]=\bar\sigma^2$:

$$\bar\sigma^2 = \omega + (\alpha+\beta)\bar\sigma^2 \quad \Rightarrow \quad \bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$$

which requires $\alpha+\beta<1$ for stationarity — **the persistence parameter** $\alpha+\beta$ determines the half-life of a volatility shock: $\text{half-life} = \frac{\ln(0.5)}{\ln(\alpha+\beta)}$.

**EGARCH** fixes GARCH's blind spot — it can't distinguish a positive shock from a negative one of the same size, but equities famously exhibit a **leverage effect** (down moves raise vol more than up moves of equal size):

$$\ln(\sigma_t^2) = \omega + \beta\ln(\sigma_{t-1}^2) + \alpha\left(\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}} - \mathbb{E}\left[\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}}\right]\right) + \gamma\frac{\varepsilon_{t-1}}{\sigma_{t-1}}$$

**Say it out loud:** **"Modeling log-variance guarantees positivity automatically — no need to constrain $\omega,\alpha,\beta \geq 0$ like plain GARCH. The $\gamma$ term is the asymmetry: because it multiplies the **signed** standardized shock rather than its absolute value, a negative shock ($\varepsilon_{t-1}<0$) with $\gamma<0$ adds positive log-variance — 'bad news raises vol more than good news of the same magnitude,' exactly the leverage effect."**

```python
"""GARCH(1,1) log-likelihood estimation via scipy MLE, from scratch."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize


def garch11_neg_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Computes the negative Gaussian log-likelihood for GARCH(1,1).

    Args:
        params: [omega, alpha, beta] parameter vector.
        returns: De-meaned return series, shape (n,).

    Returns:
        Negative log-likelihood (to be minimized).
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    log_lik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
    return -float(log_lik)


def fit_garch11(returns: np.ndarray) -> np.ndarray:
    """Fits GARCH(1,1) parameters via constrained MLE.

    Args:
        returns: De-meaned return series, shape (n,).

    Returns:
        Fitted [omega, alpha, beta].
    """
    x0 = np.array([np.var(returns) * 0.05, 0.05, 0.90])
    result = minimize(
        garch11_neg_loglik, x0, args=(returns,), method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )
    return result.x


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    n = 1000
    true_omega, true_alpha, true_beta = 0.02, 0.10, 0.85
    sim_returns, sim_sigma2 = np.zeros(n), np.zeros(n)
    sim_sigma2[0] = true_omega / (1 - true_alpha - true_beta)
    for t in range(1, n):
        sim_sigma2[t] = true_omega + true_alpha * sim_returns[t - 1] ** 2 + true_beta * sim_sigma2[t - 1]
        sim_returns[t] = np.sqrt(sim_sigma2[t]) * rng.normal()

    fitted = fit_garch11(sim_returns)
    print(f"True params:   omega={true_omega}, alpha={true_alpha}, beta={true_beta}")
    print(f"Fitted params: omega={fitted[0]:.4f}, alpha={fitted[1]:.4f}, beta={fitted[2]:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.sqrt(sim_sigma2), name="True Conditional Vol", line=dict(color="black")))
    fig.update_layout(title="GARCH(1,1) Simulated Conditional Volatility", template="plotly_white")
    fig.write_html("/tmp/garch_vol.html")
    print("Saved plot to /tmp/garch_vol.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q17 · Hidden Markov Models — Baum-Welch & Viterbi for Regime Detection

**Feynman explanation:** Markets don't announce "we're in a risk-off regime now" — you only observe the noisy consequence (returns, vol, spreads). An HMM posits a small number of **hidden** states (e.g., calm / stressed financing regime) with Markovian transitions, each emitting observations from a state-specific distribution. Two questions matter: "what's the most likely hidden state path given the data?" (Viterbi) and "how do I *learn* the transition/emission parameters from data?" (Baum-Welch, an EM instance).

**Model:** hidden states $s_t \in \{1,\dots,K\}$, transition matrix $A_{ij}=P(s_t=j\mid s_{t-1}=i)$, emission density $b_j(x_t)=P(x_t\mid s_t=j)$ (Gaussian per regime here), initial distribution $\pi$.

**Forward algorithm** — probability of the observations up to $t$ and being in state $j$ at $t$:

$$\alpha_t(j) = b_j(x_t)\sum_{i=1}^K \alpha_{t-1}(i)A_{ij}, \qquad \alpha_1(j) = \pi_j\, b_j(x_1)$$

**Viterbi** replaces the sum with a max to recover the single most probable state sequence:

$$\delta_t(j) = b_j(x_t)\max_{i}\big[\delta_{t-1}(i)A_{ij}\big], \qquad \psi_t(j)=\arg\max_i\big[\delta_{t-1}(i)A_{ij}\big]$$

**Say it out loud:** **"The $\delta_{t}(j)$ is the probability of the single best path that ends in state $j$ at time $t$ — we build it up recursively: to be in state $j$ optimally at time $t$, you must have arrived from whichever prior state $i$ maximized $\delta_{t-1}(i)A_{ij}$, then multiply by how well state $j$ explains the current observation. The backpointer $\psi_t(j)$ remembers which $i$ won, so a final backward pass from $\arg\max_j \delta_T(j)$ reconstructs the whole optimal regime path."**

**Baum-Welch (EM)** alternates: E-step computes $\gamma_t(j)=P(s_t=j\mid X,\theta)$ and $\xi_t(i,j)=P(s_t=i,s_{t+1}=j\mid X,\theta)$ via forward-backward; M-step re-estimates:

$$\hat A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad \hat\mu_j = \frac{\sum_t \gamma_t(j) x_t}{\sum_t \gamma_t(j)}$$

— i.e., the new transition probability is "expected number of $i\to j$ transitions" over "expected time spent in $i$," and the new emission mean is a $\gamma$-weighted average of observations, exactly analogous to soft-assignment k-means.

```python
"""2-state Gaussian HMM regime detection via hmmlearn."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM


def fit_regime_hmm(returns: np.ndarray, n_states: int = 2, seed: int = 42) -> GaussianHMM:
    """Fits a Gaussian HMM to a return series for regime detection.

    Args:
        returns: 1-D array of asset/spread returns.
        n_states: Number of latent regimes to fit.
        seed: Random seed for EM initialization.

    Returns:
        A fitted hmmlearn GaussianHMM instance.
    """
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=200, random_state=seed)
    model.fit(returns.reshape(-1, 1))
    return model


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n = 600
    calm = rng.normal(0.0002, 0.004, n // 2)
    stressed = rng.normal(-0.0010, 0.018, n // 2)
    series = np.concatenate([calm, stressed[: n // 4], calm[: n // 8], stressed[n // 4 :]])

    hmm = fit_regime_hmm(series, n_states=2)
    states = hmm.predict(series.reshape(-1, 1))
    print(f"Regime means: {hmm.means_.flatten()}, Transition matrix:\n{np.round(hmm.transmat_, 3)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.cumsum(series), mode="lines", name="Cumulative Return",
                              line=dict(color="black")))
    fig.add_trace(go.Scatter(y=np.where(states == 1, np.cumsum(series), np.nan),
                              mode="markers", marker=dict(color="crimson", size=4),
                              name="Detected Stress Regime"))
    fig.update_layout(title="HMM-Detected Financing Regime Shifts", template="plotly_white")
    fig.write_html("/tmp/hmm_regimes.html")
    print("Saved plot to /tmp/hmm_regimes.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q18 · Walk-Forward Validation & Combinatorial Purged Cross-Validation

**Feynman explanation:** Standard k-fold CV shuffles data randomly — but in finance, that leaks the future into the past (a training fold can contain observations *after* a test-fold observation, and adjacent observations share overlapping label windows via rolling features/labels, so information leaks even without literal look-ahead). CPCV (López de Prado) fixes both leakage channels: it (1) **purges** training observations whose label-computation window overlaps the test window, and (2) **embargoes** a further buffer after each test fold to remove serial-correlation leakage.

```
NAIVE K-FOLD (WRONG for time series)          PURGED + EMBARGOED WALK-FORWARD (correct)
──────────────────────────────────           ─────────────────────────────────────────
[ train ][ test ][ train ][ test ]            [ train ][ purge ][ test ][ embargo ][ train ]
   Random fold assignment leaks                Purge: drop training obs whose LABEL
   future info into past training               window overlaps the test window
   folds via shuffling                          Embargo: drop N obs immediately after
                                                 test fold from the next training fold
                                                 (residual serial correlation)
```

**CPCV construction:** partition the timeline into $N$ groups; form $\binom{N}{k}$ combinations choosing $k$ groups as test, evaluating each combination as an independent backtest path — this produces **multiple, non-overlapping "paths"** through the data, letting you build a full distribution of out-of-sample Sharpe ratios rather than a single point estimate, from which you can compute the **Probability of Backtest Overfitting (PBO)**:

$$\text{PBO} = P\big(\text{rank of IS-best config in OOS} > \text{median}\big)$$

**Say it out loud:** *"If the configuration that looked best in-sample is systematically only median-or-worse out-of-sample across the combinatorial paths, that's direct evidence you're picking noise, not signal — PBO quantifies exactly how often that happens."*

```python
"""Purged & embargoed walk-forward split generator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PurgedWalkForwardSplitter:
    """Generates purged, embargoed train/test index splits for time series.

    Attributes:
        n_splits: Number of walk-forward folds.
        label_horizon: Number of forward periods a label depends on
            (used to purge overlapping training observations).
        embargo_pct: Fraction of fold length embargoed after each test
            fold to remove residual serial-correlation leakage.
    """

    n_splits: int
    label_horizon: int
    embargo_pct: float = 0.01

    def split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Produces purged/embargoed (train_idx, test_idx) pairs.

        Args:
            n_samples: Total number of chronologically ordered observations.

        Returns:
            List of (train_idx, test_idx) index arrays, one per fold.
        """
        fold_size = n_samples // (self.n_splits + 1)
        embargo = int(fold_size * self.embargo_pct)
        splits = []

        for fold in range(1, self.n_splits + 1):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n_samples)
            test_idx = np.arange(test_start, test_end)

            purge_start = max(0, test_start - self.label_horizon)
            embargo_end = min(n_samples, test_end + embargo)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_mask[test_end:] = train_mask[test_end:] & (
                np.arange(test_end, n_samples) >= embargo_end
            )
            train_idx = np.where(train_mask[:test_start])[0]
            splits.append((train_idx, test_idx))
        return splits


if __name__ == "__main__":
    splitter = PurgedWalkForwardSplitter(n_splits=5, label_horizon=10, embargo_pct=0.02)
    for i, (train_idx, test_idx) in enumerate(splitter.split(n_samples=1000)):
        print(f"Fold {i}: train=[{train_idx.min()}..{train_idx.max()}] (n={len(train_idx)}), "
              f"test=[{test_idx.min()}..{test_idx.max()}] (n={len(test_idx)})")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q19 · Kalman Filter for Dynamic Hedge Ratios / Financing Spread Tracking

**Feynman explanation:** A static OLS hedge ratio assumes the relationship between two assets never changes — false in financing markets where the repo-vs-GC relationship drifts with collateral scarcity. The Kalman filter treats the hedge ratio itself as a **hidden, slowly-evolving state** and updates it optimally, in closed form, every time a new observation arrives — no re-fitting a whole regression window.

**State-space form:**

$$\text{State equation:}\quad \beta_t = \beta_{t-1} + w_t,\ w_t\sim N(0,Q)$$
$$\text{Observation equation:}\quad y_t = x_t\beta_t + v_t,\ v_t\sim N(0,R)$$

**Predict step** (before seeing $y_t$):

$$\hat\beta_{t|t-1} = \hat\beta_{t-1|t-1}, \qquad P_{t|t-1} = P_{t-1|t-1} + Q$$

**Update step** (after seeing $y_t$) — derived by minimizing posterior variance, i.e., choosing Kalman gain $K_t$ that minimizes $\text{Var}(\beta_t - \hat\beta_{t|t})$:

$$K_t = \frac{P_{t|t-1}\,x_t}{x_t^2 P_{t|t-1} + R}, \qquad \hat\beta_{t|t} = \hat\beta_{t|t-1} + K_t\big(y_t - x_t\hat\beta_{t|t-1}\big), \qquad P_{t|t} = (1-K_t x_t)P_{t|t-1}$$

**Line-by-line:** $K_t$ is a **signal-to-noise ratio** — the numerator $P_{t|t-1}x_t$ is "how much our prior uncertainty in $\beta$ projects onto the observation," the denominator adds the observation noise $R$; when $R$ is large (noisy market data) $K_t\to 0$ and we barely update our belief; when $Q$ is large (fast-moving true hedge ratio) $P$ grows quickly each step and $K_t\to 1$, trusting new data almost fully. The term $y_t - x_t\hat\beta_{t|t-1}$ is the **innovation** — the surprise between observed and predicted — and the update is "prior belief plus gain times surprise," the same Bayesian-update structure that recurs throughout statistics.

```python
"""Kalman filter for a time-varying hedge ratio / financing spread beta."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(slots=True)
class KalmanHedgeRatio:
    """Tracks a time-varying linear hedge ratio via a scalar Kalman filter.

    Attributes:
        process_var: Q, variance of the random-walk state innovation.
        obs_var: R, variance of the observation noise.
        beta: Current filtered hedge-ratio estimate.
        state_var: Current filtered state variance P.
    """

    process_var: float
    obs_var: float
    beta: float = 0.0
    state_var: float = 1.0

    def step(self, x_t: float, y_t: float) -> float:
        """Runs one predict-update Kalman cycle given a new observation pair.

        Args:
            x_t: Independent variable observation at time t (e.g., GC repo
                rate change).
            y_t: Dependent variable observation at time t (e.g., special
                repo rate change).

        Returns:
            The updated filtered hedge-ratio estimate beta_t|t.
        """
        # Predict.
        beta_pred = self.beta
        p_pred = self.state_var + self.process_var

        # Update.
        innovation = y_t - x_t * beta_pred
        s = x_t ** 2 * p_pred + self.obs_var
        kalman_gain = p_pred * x_t / s

        self.beta = beta_pred + kalman_gain * innovation
        self.state_var = (1 - kalman_gain * x_t) * p_pred
        return self.beta


if __name__ == "__main__":
    rng = np.random.default_rng(21)
    n = 400
    true_beta = 0.5 + np.cumsum(rng.normal(scale=0.005, size=n))  # slow drift
    x_series = rng.normal(scale=1.0, size=n)
    y_series = true_beta * x_series + rng.normal(scale=0.3, size=n)

    kf = KalmanHedgeRatio(process_var=0.0002, obs_var=0.3 ** 2, beta=0.0, state_var=1.0)
    filtered_beta = np.array([kf.step(x_series[t], y_series[t]) for t in range(n)])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true_beta, name="True Hedge Ratio", line=dict(color="black", dash="dash")))
    fig.add_trace(go.Scatter(y=filtered_beta, name="Kalman-Filtered Estimate", line=dict(color="crimson")))
    fig.update_layout(title="Kalman-Filtered Dynamic Hedge Ratio", template="plotly_white")
    fig.write_html("/tmp/kalman_hedge_ratio.html")
    print(f"Final filtered beta: {filtered_beta[-1]:.3f} vs true: {true_beta[-1]:.3f}")
    print("Saved plot to /tmp/kalman_hedge_ratio.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

# 🧠 DEEP LEARNING

---

## Q20 · MLP — Forward Pass & Backpropagation Derived Line-by-Line

**Setup:** a single hidden layer, input $x\in\mathbb{R}^d$, hidden weights $W^{(1)}\in\mathbb{R}^{h\times d}$, output weights $W^{(2)}\in\mathbb{R}^{1\times h}$, activation $\sigma$, loss $\mathcal{L}$ (squared error).

$$z^{(1)} = W^{(1)}x + b^{(1)}, \quad a^{(1)}=\sigma(z^{(1)}), \quad z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}, \quad \hat y = z^{(2)}, \quad \mathcal{L}=\tfrac12(\hat y-y)^2$$

**Backprop — the chain rule, one layer at a time.**

Output layer gradient (this is the "error signal" $\delta^{(2)}$):

$$\frac{\partial\mathcal{L}}{\partial z^{(2)}} = (\hat y - y) \equiv \delta^{(2)}$$

Gradient w.r.t. $W^{(2)}$:

$$\frac{\partial\mathcal{L}}{\partial W^{(2)}} = \frac{\partial\mathcal{L}}{\partial z^{(2)}}\cdot\frac{\partial z^{(2)}}{\partial W^{(2)}} = \delta^{(2)}\,(a^{(1)})^\top$$

Backpropagate the error into the hidden layer:

$$\delta^{(1)} = \frac{\partial\mathcal{L}}{\partial z^{(1)}} = \left(\frac{\partial z^{(2)}}{\partial a^{(1)}}\right)^\top \delta^{(2)} \odot \sigma'(z^{(1)}) = (W^{(2)})^\top\delta^{(2)}\odot\sigma'(z^{(1)})$$

$$\frac{\partial\mathcal{L}}{\partial W^{(1)}} = \delta^{(1)}x^\top$$

**Line-by-line:** each $\delta$ is "how much the loss would change per unit change in this layer's pre-activation" — it's computed at the output directly from the loss, then propagated backward by (a) mapping through the transpose of the forward weight matrix (undoing the linear mix) and (b) multiplying element-wise by the local activation derivative $\sigma'(z)$ (undoing the nonlinearity, scaled by how sensitive $\sigma$ is at that point). This is precisely the chain rule applied mechanically layer-by-layer — "backpropagation" is not a separate algorithm from calculus, it's calculus with cached intermediate values (reverse-mode autodiff) so that all layers' gradients are computed in a single backward sweep, $O(\text{one forward pass})$ in cost rather than one pass per parameter.

**Feynman explanation:** Imagine a rumor of "you're too high" (the loss gradient) starting at the output and traveling backward through the company org chart (the network). At each layer, the rumor gets reshaped by "how much did I actually listen to my inputs" ($W^\top$) and "was I even paying attention at this activation level" ($\sigma'$) — if $\sigma'\approx 0$ (saturated neuron), the rumor dies there and that neuron's weights barely update — this is precisely the vanishing gradient problem (Q21).

```python
"""Two-layer MLP with manual forward/backward pass, Google-style docstrings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation.

    Args:
        z: Pre-activation input array.

    Returns:
        Element-wise sigmoid output, same shape as z.
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


@dataclass(slots=True)
class MLPRegressor:
    """A one-hidden-layer MLP trained via manual backpropagation.

    Attributes:
        w1: Input-to-hidden weight matrix, shape (hidden_dim, input_dim).
        b1: Hidden layer bias, shape (hidden_dim,).
        w2: Hidden-to-output weight matrix, shape (1, hidden_dim).
        b2: Output bias, scalar.
        lr: Learning rate for gradient descent.
    """

    input_dim: int
    hidden_dim: int
    lr: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        """Initializes weights with He-scaled random values."""
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(scale=np.sqrt(2 / self.input_dim), size=(self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(scale=np.sqrt(2 / self.hidden_dim), size=(1, self.hidden_dim))
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the forward pass, caching activations for backprop.

        Args:
            x: Input vector, shape (input_dim,).

        Returns:
            Tuple of (hidden_activation, output_prediction).
        """
        z1 = self.w1 @ x + self.b1
        a1 = sigmoid(z1)
        z2 = self.w2 @ a1 + self.b2
        return a1, z2

    def train_step(self, x: np.ndarray, y: float) -> float:
        """Runs one forward + backward + SGD update on a single example.

        Args:
            x: Input vector, shape (input_dim,).
            y: Scalar target value.

        Returns:
            The squared-error loss for this example before the update.
        """
        a1, y_hat = self.forward(x)
        loss = 0.5 * float((y_hat - y) ** 2)

        delta2 = (y_hat - y)  # dL/dz2, shape (1,)
        grad_w2 = np.outer(delta2, a1)
        grad_b2 = delta2

        delta1 = (self.w2.T @ delta2) * (a1 * (1 - a1))  # sigmoid'(z1) = a1(1-a1)
        grad_w1 = np.outer(delta1, x)
        grad_b1 = delta1

        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1
        return loss


if __name__ == "__main__":
    rng = np.random.default_rng(15)
    X = rng.normal(size=(500, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(scale=0.1, size=500)

    net = MLPRegressor(input_dim=3, hidden_dim=16, lr=0.02, seed=1)
    losses = []
    for epoch in range(300):
        epoch_loss = np.mean([net.train_step(X[i], y[i]) for i in range(len(X))])
        losses.append(epoch_loss)

    fig = go.Figure(go.Scatter(y=losses, mode="lines", line=dict(color="darkorange")))
    fig.update_layout(title="MLP Training Loss (Manual Backprop)",
                       xaxis_title="Epoch", yaxis_title="MSE", template="plotly_white")
    fig.write_html("/tmp/mlp_training_loss.html")
    print(f"Final loss: {losses[-1]:.4f}")
    print("Saved plot to /tmp/mlp_training_loss.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q21 · Activation Functions & the Vanishing Gradient Problem

$$\sigma(z)=\frac{1}{1+e^{-z}}, \quad \sigma'(z)=\sigma(z)(1-\sigma(z)) \le 0.25 \qquad \tanh'(z)=1-\tanh^2(z)\le 1 \qquad \text{ReLU}'(z)=\mathbb{1}[z>0]$$

**Derivation of the vanishing gradient in a deep net.** By the chain rule (Q20 generalized to $L$ layers), the gradient at layer 1 involves a **product** of $L$ Jacobians:

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} \propto \prod_{l=2}^{L}\Big[(W^{(l)})^\top \odot \sigma'(z^{(l)})\Big]$$

**Say it out loud:** **"Each factor in that product includes $\sigma'(z^{(l)})$, which for sigmoid is bounded above by $0.25$ — multiply 20 such factors and the gradient magnitude shrinks by roughly $0.25^{20}\approx 10^{-12}$, i.e., early layers essentially stop learning. ReLU's derivative is exactly 1 wherever the unit is active, so it doesn't systematically shrink gradients through depth — at the cost of a new failure mode, 'dying ReLU,' where a unit that goes permanently negative has zero gradient forever."**

```
ACTIVATION   RANGE        MAX |DERIVATIVE|   FAILURE MODE
──────────  ───────────  ────────────────   ─────────────────────────────
Sigmoid      (0, 1)        0.25               Vanishing gradient (deep nets)
Tanh         (-1, 1)       1.0                Vanishing (less severe than sigmoid)
ReLU         [0, ∞)        1.0 (z>0), 0 else  Dying ReLU (permanently zero grad)
LeakyReLU    (-∞, ∞)       1.0 / 0.01α         Mitigates dying ReLU
GELU/SiLU    smooth        ~1.0 near 0        Used in modern transformers
```

**Feynman explanation:** Think of gradient flow as a message passed hand-to-hand down a very long line of people, and each person can only pass along **at most a quarter** of what they heard (sigmoid). After enough people, the message is inaudible. ReLU replaces "pass along a quarter" with "pass along all of it, or nothing at all" — no systematic attenuation, just an occasional dropped message.

[🔝 Back to Top](#table-of-contents)

---
---

## Q22 · RNNs — Backpropagation Through Time

**Recurrence:** $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$, output $\hat y_t = W_{hy}h_t$.

**BPTT derivation.** The total loss is **$\mathcal{L}=\sum_t \mathcal{L}_t$**. Because **$h_t$** depends on **$h_{t-1}$** which depends on **$h_{t-2}$**, etc., the gradient w.r.t. the **shared** weight **$W_{hh}$** must sum contributions through **every** path back through time:

$$\frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t}\frac{\partial \mathcal{L}_t}{\partial h_t}\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)\frac{\partial h_k}{\partial W_{hh}}$$

**Line-by-line:** $\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1-\tanh^2(z_j))\,W_{hh}$ — the same $W_{hh}$ matrix reappears at **every** timestep because weights are shared across time (that's what makes it "recurrent"). So the product term $\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}$ is a product of $(t-k)$ copies of essentially the same matrix (modulated by the tanh-derivative diagonal) — and just like Q21's depth argument, repeated multiplication by a matrix with eigenvalues $<1$ **vanishes exponentially in $t-k$**, while eigenvalues $>1$ **explode exponentially**. This is precisely why vanilla RNNs cannot learn dependencies more than roughly 10–20 steps back — and precisely the motivation for LSTM (Q23).

**Feynman explanation:** BPTT is backprop through an "unrolled" network where every timestep is a layer sharing the same weights — the same vanishing/exploding-gradient logic from Q21 applies, except now "depth" is literally "how many timesteps ago," so a repo curve model trying to remember a shock from 60 days ago faces the exact same mathematical wall a 60-layer MLP does.

```python
"""Vanilla RNN cell with BPTT gradient computed manually for one timestep chain."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VanillaRNNCell:
    """A single-layer vanilla RNN with manually implemented BPTT.

    Attributes:
        w_hh: Hidden-to-hidden recurrent weight matrix.
        w_xh: Input-to-hidden weight matrix.
        b_h: Hidden bias vector.
    """

    hidden_dim: int
    input_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Initializes recurrent and input weight matrices."""
        rng = np.random.default_rng(self.seed)
        scale = 1 / np.sqrt(self.hidden_dim)
        self.w_hh = rng.uniform(-scale, scale, size=(self.hidden_dim, self.hidden_dim))
        self.w_xh = rng.uniform(-scale, scale, size=(self.hidden_dim, self.input_dim))
        self.b_h = np.zeros(self.hidden_dim)

    def forward_sequence(self, X: np.ndarray) -> list[np.ndarray]:
        """Runs the RNN forward across a full input sequence.

        Args:
            X: Input sequence, shape (T, input_dim).

        Returns:
            List of hidden states h_1..h_T, each shape (hidden_dim,).
        """
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t in range(X.shape[0]):
            h = np.tanh(self.w_hh @ h + self.w_xh @ X[t] + self.b_h)
            hidden_states.append(h)
        return hidden_states

    def gradient_norm_through_time(self, hidden_states: list[np.ndarray]) -> np.ndarray:
        """Illustrates vanishing/exploding gradient magnitude vs. lag k.

        Args:
            hidden_states: Output of forward_sequence().

        Returns:
            Array of ||prod of Jacobians|| for each lag back from the
            final timestep, demonstrating exponential decay/growth.
        """
        t_final = len(hidden_states) - 1
        jacobian_product = np.eye(self.hidden_dim)
        norms = [np.linalg.norm(jacobian_product, 2)]
        for j in range(t_final, 0, -1):
            d_tanh = np.diag(1 - hidden_states[j] ** 2)
            step_jacobian = d_tanh @ self.w_hh
            jacobian_product = step_jacobian @ jacobian_product
            norms.append(np.linalg.norm(jacobian_product, 2))
        return np.array(norms)


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    seq = rng.normal(size=(60, 4))
    cell = VanillaRNNCell(hidden_dim=32, input_dim=4, seed=2)
    states = cell.forward_sequence(seq)
    grad_norms = cell.gradient_norm_through_time(states)
    print(f"Gradient norm at lag 5:  {grad_norms[5]:.6f}")
    print(f"Gradient norm at lag 30: {grad_norms[30]:.6e}  (vanishing over long horizons)")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q23 · LSTM — Gate Equations Derived from First Principles

**The core fix:** replace the multiplicative $\tanh$-chain of vanilla RNN with an **additive** cell-state update, so gradients can flow through time via a path with derivative $\approx 1$ (like a residual/skip connection).

$$f_t = \sigma(W_f[h_{t-1},x_t]+b_f) \quad\text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1},x_t]+b_i) \quad\text{(input gate)}$$
$$\tilde C_t = \tanh(W_C[h_{t-1},x_t]+b_C) \quad\text{(candidate cell content)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t \quad\text{(cell state update — the key equation)}$$
$$o_t = \sigma(W_o[h_{t-1},x_t]+b_o), \qquad h_t = o_t \odot \tanh(C_t)$$

**Why this solves vanishing gradients — derive $\partial C_t/\partial C_{t-1}$:**

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + (\text{terms involving } \partial f_t/\partial C_{t-1},\ \partial i_t/\partial C_{t-1},\ \text{etc.})$$

**Line-by-line:** the *dominant* term is simply $f_t$ (the forget gate value itself), because $C_t$ is built by **element-wise addition**, not repeated matrix multiplication through a saturating nonlinearity. If the network learns $f_t\approx 1$ for a feature that needs to be remembered over long horizons, gradients flow backward through the cell state almost undiminished (an additive "gradient highway" — the same trick later generalized into ResNet skip connections). This is the entire reason LSTM can learn dependencies across 100+ timesteps where vanilla RNN fails around 10–20.

**Feynman explanation:** think of the cell state $C_t$ as a conveyor belt running the length of the sequence, with gates as valves that can let information on (input gate), let it drain off (forget gate), or read from the belt without disturbing it (output gate). Because the belt itself doesn't get squashed through a nonlinearity at every step — it's just added to — information (and its gradient) can ride the belt across long distances mostly intact.

```python
"""LSTM cell forward pass implemented from the raw gate equations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Args:
        z: Pre-activation array.

    Returns:
        Element-wise sigmoid output.
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


@dataclass(slots=True)
class LSTMCell:
    """A single LSTM cell implementing the four canonical gates.

    Attributes:
        hidden_dim: Dimensionality of the hidden/cell state.
        input_dim: Dimensionality of the input at each timestep.
    """

    hidden_dim: int
    input_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Initializes gate weight matrices (concatenated [h, x] input)."""
        rng = np.random.default_rng(self.seed)
        concat_dim = self.hidden_dim + self.input_dim
        scale = 1 / np.sqrt(concat_dim)
        shape = (self.hidden_dim, concat_dim)
        self.w_f = rng.uniform(-scale, scale, shape)
        self.w_i = rng.uniform(-scale, scale, shape)
        self.w_c = rng.uniform(-scale, scale, shape)
        self.w_o = rng.uniform(-scale, scale, shape)
        self.b_f = np.ones(self.hidden_dim)  # forget-gate bias init to 1 (remember by default)
        self.b_i = np.zeros(self.hidden_dim)
        self.b_c = np.zeros(self.hidden_dim)
        self.b_o = np.zeros(self.hidden_dim)

    def step(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one LSTM timestep.

        Args:
            x_t: Input vector at time t, shape (input_dim,).
            h_prev: Previous hidden state, shape (hidden_dim,).
            c_prev: Previous cell state, shape (hidden_dim,).

        Returns:
            Tuple of (h_t, c_t), the new hidden and cell states.
        """
        concat = np.concatenate([h_prev, x_t])
        f_t = sigmoid(self.w_f @ concat + self.b_f)
        i_t = sigmoid(self.w_i @ concat + self.b_i)
        c_tilde = np.tanh(self.w_c @ concat + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = sigmoid(self.w_o @ concat + self.b_o)
        h_t = o_t * np.tanh(c_t)
        return h_t, c_t

    def forward_sequence(self, X: np.ndarray) -> np.ndarray:
        """Runs the LSTM across a full sequence.

        Args:
            X: Input sequence, shape (T, input_dim).

        Returns:
            Hidden states across time, shape (T, hidden_dim).
        """
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        outputs = []
        for t in range(X.shape[0]):
            h, c = self.step(X[t], h, c)
            outputs.append(h)
        return np.stack(outputs)


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    seq = rng.normal(size=(100, 6))
    lstm = LSTMCell(hidden_dim=16, input_dim=6, seed=4)
    hidden_path = lstm.forward_sequence(seq)
    print(f"Hidden state path shape: {hidden_path.shape}")
    print(f"Final hidden state: {np.round(hidden_path[-1, :4], 3)}...")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q24 · GRU vs. LSTM — Simplification Trade-offs

$$z_t = \sigma(W_z[h_{t-1},x_t]) \quad\text{(update gate — merges LSTM's forget+input)}$$
$$r_t = \sigma(W_r[h_{t-1},x_t]) \quad\text{(reset gate)}$$
$$\tilde h_t = \tanh(W[r_t\odot h_{t-1}, x_t]) \qquad h_t = (1-z_t)\odot h_{t-1} + z_t\odot \tilde h_t$$

**Say it out loud:** **"GRU collapses LSTM's separate cell state and hidden state into one, and collapses the forget/input gates into a single update gate $z_t$ that directly interpolates between the old hidden state and a new candidate — no separate 'memory conveyor belt,' just a convex combination each step. This costs one fewer gate (3 weight matrices vs. LSTM's 4), roughly 25% fewer parameters, faster to train, and empirically matches LSTM performance on many tasks — but LSTM's separate cell state can be advantageous for very long sequences where you want finer-grained control over what's remembered vs. what's exposed to downstream layers."**

```
DIMENSION           LSTM                              GRU
──────────────────  ────────────────────────────────  ──────────────────────────────
Gates                3 (forget, input, output)          2 (update, reset)
State                Separate cell (C) + hidden (h)      Single hidden state (h)
Params (per layer)   ~4×(d²+dk)                          ~3×(d²+dk)
Training speed       Slower (more params)                ~15-25% faster
Long-range memory    Slightly stronger (separate C)      Slightly weaker on very long seqs
When to prefer        Long sequences, ample compute       Latency-sensitive prod, smaller data
```

**Feynman explanation:** if LSTM is a house with a separate pantry (cell state) and living room (hidden state) connected by controlled doors (gates), GRU merges them into a studio apartment — less to manage, almost as functional, cheaper rent (compute).

[🔝 Back to Top](#table-of-contents)

---
---

## Q25 · Regularization — Dropout & BatchNorm Math

**Dropout.** During training, each unit is independently zeroed with probability $p$:

$$\tilde a_j = \frac{m_j}{1-p}\, a_j, \qquad m_j \sim \text{Bernoulli}(1-p)$$

**Say it out loud:** **"The $1/(1-p)$ scaling is 'inverted dropout' — it rescales the surviving activations at train time so that the expected activation matches what it would be at test time with no dropout at all, meaning inference requires zero changes to the forward pass."** Mechanistically, dropout prevents co-adaptation: a unit can't rely on any specific set of other units always being present, which is mathematically equivalent to training an implicit ensemble of **$2^h$** thinned sub-networks that share weights, averaged at test time.

**Batch Normalization.** For a mini-batch $\mathcal{B}=\{z_1,\dots,z_m\}$ of pre-activations for one unit:

$$\mu_\mathcal{B} = \frac1m\sum_i z_i, \qquad \sigma_\mathcal{B}^2 = \frac1m\sum_i(z_i-\mu_\mathcal{B})^2, \qquad \hat z_i = \frac{z_i-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}, \qquad y_i = \gamma\hat z_i + \beta$$

**Line-by-line:** normalizing to zero-mean/unit-variance keeps activations in the nonlinearity's well-conditioned region (avoiding saturation, which we showed in Q21 kills gradients), stabilizing and accelerating training by reducing sensitivity to weight-initialization scale and internal covariate shift. The learnable $\gamma,\beta$ restore representational capacity — without them, BatchNorm would force every layer's output to have exactly zero mean and unit variance, which can be strictly worse than what the optimal network wants; $\gamma,\beta$ let the network **undo** the normalization if that's optimal, so BatchNorm is a strict generalization of the un-normalized layer.

```python
"""Dropout and BatchNorm implemented from scratch (NumPy, forward-only)."""
from __future__ import annotations

import numpy as np


def inverted_dropout(activations: np.ndarray, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Applies inverted dropout to a layer's activations.

    Args:
        activations: Pre-dropout activations, any shape.
        keep_prob: Probability 1-p of keeping a unit active.
        rng: NumPy random generator for reproducibility.

    Returns:
        Activations with dropout applied and rescaled by 1/keep_prob.
    """
    mask = rng.binomial(1, keep_prob, size=activations.shape)
    return activations * mask / keep_prob


def batch_norm_forward(
    z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Applies batch normalization to a batch of pre-activations.

    Args:
        z: Pre-activation matrix, shape (batch_size, n_units).
        gamma: Learnable per-unit scale, shape (n_units,).
        beta: Learnable per-unit shift, shape (n_units,).
        eps: Numerical stability constant.

    Returns:
        Normalized, scaled, and shifted output, shape (batch_size, n_units).
    """
    mu = z.mean(axis=0)
    var = z.var(axis=0)
    z_hat = (z - mu) / np.sqrt(var + eps)
    return gamma * z_hat + beta


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    batch = rng.normal(loc=5.0, scale=3.0, size=(64, 10))
    gamma, beta = np.ones(10), np.zeros(10)

    normalized = batch_norm_forward(batch, gamma, beta)
    print(f"Pre-BN  mean/std: {batch.mean():.3f} / {batch.std():.3f}")
    print(f"Post-BN mean/std: {normalized.mean():.3f} / {normalized.std():.3f}")

    dropped = inverted_dropout(batch, keep_prob=0.8, rng=rng)
    print(f"Expected activation preserved (train-time scaling): "
          f"pre-mean={batch.mean():.3f}, post-dropout-mean={dropped.mean():.3f}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q26 · Attention & the Transformer Building Block

**Feynman explanation:** RNN/LSTM process sequences step-by-step, so information from position 1 has to survive $T-1$ hops to influence position $T$ — attention instead lets every position look **directly** at every other position in a single step, at the cost of $O(T^2)$ compute. For sequence modeling on financing curves (e.g., attending over the last 60 days of rate observations to price today), attention layers let the model learn *which* historical days matter most without a fixed decay structure.

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Line-by-line:** $Q=XW_Q$ (queries — "what am I looking for"), $K=XW_K$ (keys — "what do I contain"), $V=XW_V$ (values — "what do I offer if selected"). $QK^\top$ computes a similarity score between every query-key pair (a dot product — larger when they point in a similar direction in the learned representation space). Dividing by $\sqrt{d_k}$ counteracts the fact that dot products of high-dimensional random vectors grow in variance with $d_k$ (variance $\propto d_k$ for i.i.d. components), which would otherwise push softmax into a saturated, near-one-hot regime with vanishing gradients (the same saturation problem as Q21, in a different guise). Softmax converts scores to a probability distribution over positions, and the output is a **weighted average of values**, weighted by learned relevance.

**Feynman explanation, again, more simply:** attention is a differentiable, learned "lookup table" — instead of hard-indexing into memory, you compute a soft (probabilistic) blend of everything in memory, weighted by how relevant each item is to the current question.

```python
"""Scaled dot-product self-attention implemented from scratch."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        x: Input array.
        axis: Axis along which to normalize.

    Returns:
        Softmax-normalized array along the given axis.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    X: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes single-head scaled dot-product self-attention.

    Args:
        X: Input sequence, shape (seq_len, d_model).
        w_q: Query projection weights, shape (d_model, d_k).
        w_k: Key projection weights, shape (d_model, d_k).
        w_v: Value projection weights, shape (d_model, d_v).

    Returns:
        Tuple of (output, attention_weights). output has shape
        (seq_len, d_v); attention_weights has shape (seq_len, seq_len).
    """
    Q, K, V = X @ w_q, X @ w_k, X @ w_v
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    seq_len, d_model, d_k = 10, 8, 8
    X = rng.normal(size=(seq_len, d_model))
    w_q = rng.normal(scale=0.1, size=(d_model, d_k))
    w_k = rng.normal(scale=0.1, size=(d_model, d_k))
    w_v = rng.normal(scale=0.1, size=(d_model, d_k))

    out, attn = scaled_dot_product_attention(X, w_q, w_k, w_v)
    print(f"Output shape: {out.shape}")
    print(f"Attention weights row-sums (should be 1.0): {attn.sum(axis=1).round(4)}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q27 · Autoencoders for Dimensionality Reduction & Anomaly Detection

**Setup:** encoder $z = f_\theta(x)$ compresses $x\in\mathbb{R}^d$ to $z\in\mathbb{R}^k,\ k\ll d$; decoder $\hat x = g_\phi(z)$ reconstructs. Objective:

$$\min_{\theta,\phi}\ \mathbb{E}_x\big[\lVert x - g_\phi(f_\theta(x))\rVert_2^2\big]$$

**Relationship to PCA.** **For a linear encoder/decoder with no nonlinearity and squared-error loss, the optimal solution spans the same subspace as the top - $k$ principal components of $X$ (this can be shown via the Eckart–Young theorem: the best rank - $k$ approximation of $X$ in Frobenius norm is exactly its truncated SVD). A nonlinear autoencoder generalizes this to a nonlinear manifold approximation — it can capture curved lower-dimensional structure (e.g., a nonlinear relationship across the financing curve/term structure) that linear PCA cannot.**

**Anomaly detection.** Train the autoencoder only on "normal" regime data; at inference, compute the reconstruction error:

$$e(x) = \lVert x - g_\phi(f_\theta(x))\rVert_2^2$$

**Say it out loud:** *"The autoencoder can only compress and reconstruct patterns similar to what it saw in training — an anomalous financing/collateral pattern it's never seen won't fit the learned low-dimensional manifold, so it reconstructs poorly. Reconstruction error above a threshold (calibrated on a validation set, e.g., 99th percentile of normal-regime error) flags a novel/adverse-selection pattern for review — this is exactly the adverse-selection-flow detection use case relevant to financing desks."*

```python
"""Autoencoder-based anomaly detector for financing microstructure features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(slots=True)
class LinearAutoencoder:
    """A simple linear autoencoder trained via gradient descent (~PCA).

    Attributes:
        w_enc: Encoder weight matrix, shape (latent_dim, input_dim).
        w_dec: Decoder weight matrix, shape (input_dim, latent_dim).
        lr: Learning rate.
    """

    input_dim: int
    latent_dim: int
    lr: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        """Initializes encoder/decoder weights."""
        rng = np.random.default_rng(self.seed)
        scale = 1 / np.sqrt(self.input_dim)
        self.w_enc = rng.normal(scale=scale, size=(self.latent_dim, self.input_dim))
        self.w_dec = rng.normal(scale=scale, size=(self.input_dim, self.latent_dim))

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encodes then decodes a batch of inputs.

        Args:
            X: Input batch, shape (n, input_dim).

        Returns:
            Reconstructed batch, shape (n, input_dim).
        """
        z = X @ self.w_enc.T
        return z @ self.w_dec.T

    def fit(self, X: np.ndarray, n_epochs: int = 200) -> list[float]:
        """Trains via full-batch gradient descent on reconstruction MSE.

        Args:
            X: Training data, shape (n, input_dim), assumed centered.
            n_epochs: Number of full-batch gradient steps.

        Returns:
            List of MSE loss values per epoch.
        """
        losses = []
        n = X.shape[0]
        for _ in range(n_epochs):
            z = X @ self.w_enc.T
            x_hat = z @ self.w_dec.T
            error = x_hat - X
            loss = float(np.mean(np.sum(error ** 2, axis=1)))
            losses.append(loss)

            grad_w_dec = (2 / n) * error.T @ z
            grad_z = (2 / n) * error @ self.w_dec
            grad_w_enc = grad_z.T @ X

            self.w_dec -= self.lr * grad_w_dec
            self.w_enc -= self.lr * grad_w_enc
        return losses

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Computes per-sample squared reconstruction error.

        Args:
            X: Input batch, shape (n, input_dim).

        Returns:
            Per-sample error, shape (n,).
        """
        recon = self.reconstruct(X)
        return np.sum((X - recon) ** 2, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(27)
    normal = rng.multivariate_normal(mean=np.zeros(6), cov=np.eye(6) * 0.5, size=800)
    anomalies = rng.multivariate_normal(mean=np.ones(6) * 4, cov=np.eye(6) * 0.5, size=20)

    ae = LinearAutoencoder(input_dim=6, latent_dim=2, lr=0.02, seed=5)
    losses = ae.fit(normal, n_epochs=300)

    normal_err = ae.reconstruction_error(normal)
    anomaly_err = ae.reconstruction_error(anomalies)
    threshold = float(np.percentile(normal_err, 99))
    print(f"99th-pct normal-regime error (threshold): {threshold:.3f}")
    print(f"Mean anomaly error: {anomaly_err.mean():.3f}  "
          f"({(anomaly_err > threshold).mean()*100:.0f}% flagged)")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=normal_err, name="Normal Regime", opacity=0.6, marker_color="steelblue"))
    fig.add_trace(go.Histogram(x=anomaly_err, name="Anomalous Flow", opacity=0.6, marker_color="crimson"))
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="99th pct threshold")
    fig.update_layout(barmode="overlay", title="Autoencoder Reconstruction Error — Anomaly Detection",
                       template="plotly_white")
    fig.write_html("/tmp/autoencoder_anomaly.html")
    print("Saved plot to /tmp/autoencoder_anomaly.html")
```

[🔝 Back to Top](#table-of-contents)

---
---

# 🏗️ SYSTEM DESIGN & STRATEGY

---

## Q28 · Design an End-to-End Alpha/Pricing Signal Pipeline

```
 ┌───────────────┐   ┌────────────────┐   ┌───────────────────┐   ┌─────────────────┐
 │  DATA LAYER   │──▶│ FEATURE STORE  │──▶│  MODEL SERVING    │──▶│  RISK / EXEC    │
 │ Market data,  │   │ Point-in-time  │   │ Ensemble: Ridge   │   │ Vol-target size,│
 │ repo/financing│   │ correct joins  │   │ (linear factor) + │   │ position limits,│
 │ rates, order  │   │ (NO lookahead  │   │ GBM (nonlinear) + │   │ TCA feedback    │
 │ flow, alt data│   │ bias)          │   │ LSTM (sequence)   │   │                 │
 └───────────────┘   └────────────────┘   └───────────────────┘   └─────────────────┘
        │                    │                     │                     │
        ▼                    ▼                     ▼                     ▼
   Kafka/streaming     Versioned Parquet     Champion/challenger    Live PnL vs.
   + batch ETL          + feature lineage     model registry         backtest PnL
                                                                     (drift alarm)
```

**Feynman explanation:** the hardest part of this system is not any individual model — it's the **feature store's point-in-time correctness**. If a feature computed "as of" day $t$ accidentally includes information only known on day $t+1$ (e.g., a restated economic figure, or an end-of-day close mislabeled as available intraday), every backtest built on it is silently invalid — this is the single most common cause of a strategy that backtests brilliantly and then fails live, and defending against it is an engineering discipline (immutable, timestamped feature snapshots), not a modeling one.

**Model ensembling rationale:** Ridge captures the stable linear factor exposure (interpretable, low variance); GBM captures nonlinear interactions the linear model misses (e.g., regime-conditional effects); LSTM captures genuinely sequential dependency (order-flow momentum/reversal patterns) that neither can. Combine via a **stacked meta-learner** (e.g., a constrained regression on out-of-fold predictions) rather than a naive average, so the ensemble weights adapt to which sub-model is most reliable in which regime.

[🔝 Back to Top](#table-of-contents)

---
---

## Q29 · Bayesian Inference for Regime-Adaptive Position Sizing

**Setup:** treat a macro factor exposure $\beta$ as uncertain with a prior $\beta \sim N(\mu_0, \tau_0^2)$, and update given observed data via Bayes' rule.

$$p(\beta \mid y) \propto p(y\mid\beta)\,p(\beta)$$

For a Gaussian likelihood $y\mid\beta \sim N(X\beta, \sigma^2)$ and Gaussian prior, the posterior is conjugate — also Gaussian — with closed form:

$$\tau_1^2 = \left(\frac{1}{\tau_0^2} + \frac{X^\top X}{\sigma^2}\right)^{-1}, \qquad \mu_1 = \tau_1^2\left(\frac{\mu_0}{\tau_0^2} + \frac{X^\top y}{\sigma^2}\right)$$

**Line-by-line:** the posterior precision (inverse variance) $1/\tau_1^2$ is the **sum** of the prior precision and the data precision — more data (larger $X^\top X$) always shrinks posterior uncertainty, and a tighter prior (smaller $\tau_0^2$, i.e., stronger conviction before seeing data) resists being moved by noisy data. The posterior mean $\mu_1$ is a **precision-weighted average** of the prior mean and the OLS estimate $\hat\beta_{\text{OLS}}=(X^\top X)^{-1}X^\top y$ — exactly the same "prior belief + gain × surprise" structure as the Kalman filter in Q19 (the Kalman filter *is*, in fact, sequential Bayesian updating of a Gaussian state).

**Feynman explanation, tied to the JD's "regime-adaptive sizing":** under **high macro uncertainty** (e.g., ahead of a central bank decision), you widen the prior variance $\tau_0^2$ — new data moves your belief (and hence position size) faster, appropriately, because you have little confidence in the stale pre-event estimate. Under a **stable, well-understood regime**, keep the prior tight — new noisy data shouldn't whipsaw a position that's backed by years of stable evidence. This gives a principled, continuously varying position-sizing rule rather than an arbitrary if/else regime switch.

```python
"""Bayesian linear regression with a Gaussian conjugate prior for adaptive sizing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BayesianLinearRegressor:
    """Bayesian linear regression with a Gaussian conjugate prior.

    Attributes:
        prior_mean: Prior mean vector for beta, shape (k,).
        prior_cov: Prior covariance matrix for beta, shape (k, k).
        noise_var: Assumed observation noise variance, sigma^2.
    """

    prior_mean: np.ndarray
    prior_cov: np.ndarray
    noise_var: float

    def update(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the closed-form Gaussian posterior over beta.

        Args:
            X: Design matrix of new observations, shape (n, k).
            y: Response vector of new observations, shape (n,).

        Returns:
            Tuple of (posterior_mean, posterior_cov).
        """
        prior_precision = np.linalg.inv(self.prior_cov)
        data_precision = (X.T @ X) / self.noise_var

        posterior_cov = np.linalg.inv(prior_precision + data_precision)
        posterior_mean = posterior_cov @ (
            prior_precision @ self.prior_mean + (X.T @ y) / self.noise_var
        )
        return posterior_mean, posterior_cov

    def position_size(self, posterior_mean: np.ndarray, posterior_cov: np.ndarray,
                       kelly_fraction: float = 0.5) -> float:
        """Sizes a position using a fractional-Kelly rule scaled by posterior confidence.

        Args:
            posterior_mean: Posterior mean of the exposure/edge estimate.
            posterior_cov: Posterior covariance matrix.
            kelly_fraction: Fraction of full Kelly to actually deploy
                (risk-management haircut).

        Returns:
            Recommended position size, shrunk toward zero under high
            posterior uncertainty (i.e., during regime transitions).
        """
        edge = float(np.sum(posterior_mean))
        uncertainty = float(np.sqrt(np.trace(posterior_cov)))
        return kelly_fraction * edge / (uncertainty ** 2 + 1e-6)


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    k = 3
    model = BayesianLinearRegressor(
        prior_mean=np.zeros(k), prior_cov=np.eye(k) * 1.0, noise_var=0.25,
    )
    X_new = rng.normal(size=(50, k))
    true_beta = np.array([0.8, -0.3, 0.5])
    y_new = X_new @ true_beta + rng.normal(scale=0.5, size=50)

    post_mean, post_cov = model.update(X_new, y_new)
    size = model.position_size(post_mean, post_cov)
    print(f"Posterior mean: {np.round(post_mean, 3)}")
    print(f"Posterior uncertainty (trace^0.5): {np.sqrt(np.trace(post_cov)):.4f}")
    print(f"Recommended fractional-Kelly position size: {size:.3f}")
```

[🔝 Back to Top](#table-of-contents)

---
---

## Q30 · Building an AI/ML Capability From Zero — the Greenfield Roadmap

**Open with the intuition — this is the question the JD is actually asking underneath every technical question:**
> "The mandate isn't 'be a good modeler' — it's 'build a capability.' That means the deliverable in month one is not a model, it's a **research library architecture** and a **prioritization framework**, because the team already has 50+ problem statements and only 10 prioritized — the bottleneck is execution discipline, not idea generation."

### A 90-Day Framework

```
DAYS 0–30: FOUNDATION                  DAYS 30–60: FIRST SHIPS               DAYS 60–90: SCALE
────────────────────────────          ────────────────────────────         ────────────────────────
• Audit the 50+ problem statements     • Ship 1-2 highest ROI models        • Templatize: turn the first
  against: data availability,            end-to-end (research→prod)          model's pipeline into a
  expected Sharpe/cost impact,         • Establish CPCV/walk-forward          reusable scaffold — this
  engineering lift, business urgency     validation as the DEFAULT           IS "templatizing AI/ML
• Stand up point-in-time-correct        gate, not an afterthought            solutions across business
  feature store (prevents lookahead   • Build the champion/challenger        lines" from the JD
  bias across ALL future models)        model registry so iteration        • Cross-train other business
• Define the model-risk governance      doesn't require re-plumbing          lines (Delta One, FX, Rates)
  process WITH compliance/risk up     • Set up live-vs-backtest PnL          on the same infra
  front — retrofitting governance       drift monitoring                   • Establish the Gen AI stack
  later is 10x more expensive                                               (RAG for research library,
                                                                             Claude Code for engineering
                                                                             velocity) as a force
                                                                             multiplier across ALL 10
                                                                             prioritized problem statements
```

**Prioritization scoring — a simple, defensible framework to present:**

$$\text{Priority Score} = w_1\cdot\hat{\text{Sharpe}} + w_2\cdot\text{Capacity} - w_3\cdot\text{EngLift} - w_4\cdot\text{DataRisk}$$

**Say it out loud:** *"I don't pick problems by gut feel — I score each of the 50 against expected Sharpe contribution, capacity/business impact, engineering lift required, and data-quality risk, calibrate the weights with the desk head, and defend the ranking transparently. This turns 'here's my opinion on what to build' into 'here's a reproducible framework the business can audit and adjust.'"*

**Feynman explanation — why this is the right answer for a founder-led panel:** Rishi co-founded HLS Trading — he has personally lived the 0-to-1 problem. He doesn't want to hear "I'm good at XGBoost." He wants to hear that you understand a greenfield AI function fails from **process debt** (no governance, no reusable infra, no prioritization discipline) far more often than from **modeling skill gaps** — and that you'd spend your first month building the scaffolding that lets the next 49 problem statements ship fast, not just solving problem #1 in isolation.

**Closing line for the interview:**
> "Across BAM, Highbridge, and Millburn I've built the same thing three times in three different wrappers — a research-to-production pipeline with rigorous validation at its core. What excites me about this role is that here I'd be building that pipeline as *the* deliverable, for a whole cross-asset franchise, not just for one desk's PnL."

[🔝 Back to Top](#table-of-contents)

---
---

# Quick-Reference Equation Sheet

```
══════════════════════════════════════════════════════════════════════════════
REGRESSION
══════════════════════════════════════════════════════════════════════════════
OLS:            beta_hat = (X'X)^-1 X'y
Ridge:          beta_hat = (X'X + lambda*I)^-1 X'y
Lasso (KKT):    -2*x_j'(y - X*beta) + lambda*sign(beta_j) = 0
Elastic Net CD: beta_j <- SoftThreshold(x_j' r_-j, lambda1/2) / (||x_j||^2 + lambda2)
Newey-West:     Var(beta) = (X'X)^-1 [sum(e_t^2 x_t x_t') + Bartlett-weighted
                             lagged cross terms] (X'X)^-1

══════════════════════════════════════════════════════════════════════════════
TREE-BASED
══════════════════════════════════════════════════════════════════════════════
Entropy:        H(p) = -sum p_c log2(p_c)
Gini:           G(p) = 1 - sum p_c^2
Info Gain:      IG = H(parent) - weighted_avg(H(children))
XGBoost gain:   Gain = 0.5*[G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam)] - gamma
Leaf weight:    w_j* = -G_j / (H_j + lambda)
Shapley value:  phi_i = sum_S  |S|!(|F|-|S|-1)!/|F|! * [v(S U {i}) - v(S)]

══════════════════════════════════════════════════════════════════════════════
TIME SERIES
══════════════════════════════════════════════════════════════════════════════
ADF regression: dy_t = alpha + beta*t + gamma*y_{t-1} + sum(delta_i * dy_{t-i}) + e_t
GARCH(1,1):     sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
Persistence:    alpha + beta < 1;  half-life = ln(0.5)/ln(alpha+beta)
EGARCH:         ln(sigma_t^2) = omega + beta*ln(sigma_{t-1}^2)
                + alpha*(|e_{t-1}|/s_{t-1} - E[.]) + gamma*(e_{t-1}/s_{t-1})
HMM Viterbi:    delta_t(j) = b_j(x_t) * max_i[delta_{t-1}(i) * A_ij]
Kalman gain:    K_t = P_{t|t-1} x_t / (x_t^2 P_{t|t-1} + R)
Kalman update:  beta_t|t = beta_t|t-1 + K_t*(y_t - x_t*beta_t|t-1)

══════════════════════════════════════════════════════════════════════════════
DEEP LEARNING
══════════════════════════════════════════════════════════════════════════════
MLP backprop:   delta^(1) = (W^(2))' delta^(2) . sigma'(z^(1))
Vanishing grad: dL/dW^(1) ~ prod_{l=2}^{L} [(W^(l))' . sigma'(z^(l))]
RNN BPTT:       dL_t/dW_hh = sum_k (dL_t/dh_t) * prod_{j=k+1}^{t}(dh_j/dh_{j-1}) * dh_k/dW_hh
LSTM cell:      C_t = f_t . C_{t-1} + i_t . C~_t         (additive -> gradient highway)
Attention:      Attention(Q,K,V) = softmax(QK'/sqrt(d_k)) V
BatchNorm:      z_hat = (z - mu_B)/sqrt(var_B + eps);  y = gamma*z_hat + beta
Dropout:        a~_j = (m_j / (1-p)) * a_j,  m_j ~ Bernoulli(1-p)

══════════════════════════════════════════════════════════════════════════════
GEN AI
══════════════════════════════════════════════════════════════════════════════
LoRA:           h = W_0 x + (alpha/r) * B A x,   B in R^(dxr), A in R^(rxk), r << min(d,k)
RAG recall:     sim(q,d) = E_q(q).E_d(d) / (||E_q(q)|| ||E_d(d)||)
Faithfulness:   |supported claims| / |total claims|
Bayesian post.: mu_1 = tau_1^2 * (mu_0/tau_0^2 + X'y/sigma^2),
                tau_1^2 = (1/tau_0^2 + X'X/sigma^2)^-1
══════════════════════════════════════════════════════════════════════════════
```

[🔝 Back to Top](#table-of-contents)

---

*Last updated: July 2026 · Shaikat Majumdar · Barclays AI / ML Modeler — Liquid Financing — Technical Round Prep*

[↩️ Back to README.md](../README.md)
