# Barclays — AI / ML Modeler, Liquid Financing — Technical Interview Playbook
### Rishi Dhingra (MD, Global Markets — Electronification, eTrading & AI/ML | Prime Services, Financing & Delta One) · HLS Trading Co-Founder Panel
#### 30 Questions × 7 Domains · Wednesday, July 8 2026 · 1-Hour Technical Round

> **Delivery philosophy:** Every answer follows *intuition → math derived line-by-line → Feynman restatement → production Python 3.13*. Rishi built an electronic trading/AI franchise from the ground up and co-founded HLS Trading — he will probe whether you can **build**, not just describe. The team has 50+ problem statements queued and 10 prioritized; he wants evidence you can independently scope a business problem, choose the right tool (classical ML vs. DL vs. Gen AI), and ship it into production alongside a 10–15 person engineering team.

---
---

## ⏱️ Interview Question Budget

```
DOMAIN                            QUESTIONS   WEIGHT    RISHI'S LENS
───────────────────────────────   ─────────   ───────   ──────────────────────────────────────
EXPERIENCE / PRODUCTION ML        Q1–Q3       High      Can you own a problem end-to-end?
GEN AI (Fine-tune/RAG/Eval)       Q4–Q8       High      LLM orchestration for Liquid Financing
REGRESSION                        Q9–Q11      Med       First-principles derivation, not sklearn
TREE-BASED MODELS                 Q12–Q14     Med       XGBoost internals, not just "it works"
TIME SERIES / FORECASTING         Q15–Q19     High      Financing curve is a time-series problem
DEEP LEARNING (NN/MLP/RNN/LSTM)   Q20–Q27     High      Core JD requirement — go deep
SYSTEM DESIGN / STRATEGY          Q28–Q30     High      "Build AI/ML capability from the ground up"
```

> **Priority rule:** Liquid Financing spans Equities & Delta One, Rate/Credit Financing, FX, Risk, Futures & Prime Derivatives. Rishi's own background is electronification/eTrading — expect him to push hardest on **time series forecasting for pricing curves (repo/financing spreads)**, **RNN/LSTM sequence modeling**, and **Gen AI infra** (the team explicitly uses Claude Code). Have Q15–Q27 cold; Q1–Q3 and Q28–Q30 are where you differentiate as someone who ships, not just researches.

---

## 📚 Sources Mined for Question Selection

This question set was calibrated against publicly documented interview experiences and technical study guides for this exact role and its closest analogues (buy-side/sell-side AI/ML Modeler, Quant Researcher, and LLM/GenAI Engineer roles), prioritizing exact-role and most-recent material:

- **Barclays AVP – Global Markets AI/ML Modeller (Built In NYC / Barclays careers site, search.jobs.barclays)** — confirms the "building an AI/ML capability from the ground up... start-up mindset... PyTorch/TensorFlow... deploying scalable ML models in production" language verbatim matches this JD, validating the framing of Q1, Q3, and Q30.
- **Barclays ML Engineer / Data Scientist interview guides (InterviewQuery, Glassdoor)** — confirm recurring emphasis on supervised-vs-unsupervised fundamentals, production deployment discussion, and translating technical work for non-technical stakeholders (Q1, Q3).
- **Wall Street Oasis — "Best Resources/Guide For Quant Interviews" forum** — practitioner consensus that buy-side/sell-side quant research interviews "focus more on digging deeper on statistics and machine learning... penalized regression and ensemble methods: LASSO, Ridge, Elastic Net, Gradient Boosted Trees, Random Forest" — directly informed the weighting toward Q9–Q14.
- **WSO Quant Finance Interview Course syllabus** — corroborates GARCH/volatility modeling, stationarity, and neural-network/transformer topics as standard quant technical-round material (Q15–Q19, Q26).
- **LLM/GenAI interview question banks (DataCamp "Top 36 LLM Interview Questions 2026," InterviewBit, GeeksforGeeks Generative AI guide, KalyanKS-NLP LLM Interview Hub, dev.to "LLM Interview Cheat Sheet")** — converged on LoRA/PEFT mechanics, RAG pipeline steps (chunk→embed→retrieve→rerank→generate), fine-tuning-vs-RAG-vs-prompting decision framing, and hallucination mitigation as the highest-frequency Gen AI questions across 2025–2026 postings — directly informing Q4–Q8.
- **CFA Institute — "For LLMs in the Financial Industry: A Practical Guide"** — confirms RAG as the dominant financial-industry LLM adaptation pattern over full fine-tuning, and CoT prompting / RLHF as standard hallucination-mitigation techniques cited in Q6 and Q8.
- **Nomura interview reports (Glassdoor)** — confirm technical rounds for quant/MLE-adjacent roles emphasize applied ML case studies, end-to-end pipeline walkthroughs ("study case E2E ML Pipeline, NLP"), and "explain your last project" depth-probing — directly informing the system-design framing of Q28–Q30.

> Where a platform (e.g., Blind, LeetCode/HackerRank) returned no role-specific technical content beyond generic coding-test logistics for this exact role/title, that platform's material was excluded rather than padded with irrelevant filler.

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
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#41-p1--securities-lending-fee-forecasting-full-system-architecture)**
- **P2 · Client Margin & Haircut Optimization**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-2-client-margin--haircut-optimization)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#42-p2--client-margin--haircut-optimization-full-system-architecture)**
- **P3 · Cross-Asset Funding-Spread Anomaly Detection**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-3-cross-asset-funding-spread-anomaly-detection)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#43-p3--cross-asset-funding-spread-anomaly-detection-full-system-architecture)**
- **P4 · Prime Balance & Utilization Forecasting (Deep Learning)**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-4-prime-balance--utilization-forecasting-deep-learning)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#44-p4--prime-balance-forecasting-full-system-architecture)**
- **P5 · RAG Financing-Desk Copilot (GenAI / LLM)**
  - **[PROBLEM STATEMENT](./TAKE_HOME_PROJECTS/PROBLEMS.md#take-home-5-rag-financing-desk-copilot-genai)**
  - **[SOLUTION](./TAKE_HOME_PROJECTS/README.md#45-p5--rag-financing-desk-copilot-full-system-architecture)**

### 🏗️ Reference Sheet
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

### Why Purging and Embargo Matter — A Worked Numerical Example

Suppose you're building a 20-day-forward financing-spread-widening classifier. A naive train/test split at day 500 looks clean — but the **label** for day 495 is "did the spread widen by day 515?" That label's construction reaches 15 days *past* the split point. If day 515 lands in your test set, you have used test-set information to construct a training-set label — a leak invisible to any code review that only checks index ranges.

```
Day:        490   495   500(SPLIT)   505   510   515
Label(495) uses:              [===================]   ← reaches into "test" region
```

The fix: **purge** any training observation whose label window overlaps the test window (remove days 480–500 from training if the test set starts at 500 and labels have a 20-day horizon), and **embargo** a further buffer after the test set ends before the next training fold begins (because serial correlation in returns/spreads means day 505's *features*, not just its label, still "know something" about day 500's realized outcome).

### The Cost of Getting This Wrong — A Concrete Failure Mode

At Millburn, an early iteration of a macro signal library candidate showed a backtested Sharpe of 1.8 using naive k-fold CV. Under purged walk-forward validation, the same signal's Sharpe fell to 0.3 — not because the model was "bad," but because 15% of its apparent skill was the model exploiting overlapping labels between adjacent folds (essentially, partially seeing its own answer key). This is precisely why the readiness gate in the code below enforces CPCV as a hard requirement, not an optional nice-to-have — I have personally watched a promising signal evaporate under the correct validation protocol, and shipping it live first would have been a costly, avoidable mistake.

### Deeper Feynman Analogy — Research Library as an Immune System

Think of the research library (30+ independent signals built at Millburn) as an immune system, not a warehouse. A warehouse just stores things; an immune system actively tests every new "candidate" (antibody / signal) against a hostile, adversarial environment (purged OOS data, transaction costs, capacity constraints) before allowing it into general circulation (production risk). Signals that pass are "memory cells" — reusable building blocks (e.g., a validated momentum signal becomes a component ensembled into five different macro strategies). Signals that fail are discarded with a **documented reason** (overfitting, capacity, correlation to existing signals) — this documentation is what lets a research org avoid re-testing the same failed idea in three years when someone else proposes it.

### Extended Production Checklist — Model Risk Governance Fields

A single Sharpe/p-value gate (shown below) is necessary but not sufficient for a bank's model risk framework, which typically requires additional fields for audit:

```
FIELD                        WHY IT MATTERS
───────────────────────────  ──────────────────────────────────────────────
Model owner + reviewer        Accountability chain (SR 11-7 / model risk mgmt)
Assumptions & limitations     Explicit statement of when the model breaks
Data lineage                  Which feature store snapshot, which vendor feed
Backtest period & regime      Was validation period representative? (2008?
                               2020 COVID? low-vol 2017?)
Kill-switch criteria           Live drawdown / drift threshold that auto-halts
Champion/challenger cadence   How often is the model re-validated vs. re-fit
```

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

$$
\text{EWMA}_t = \lambda \cdot x_t + (1-\lambda)\cdot \text{EWMA}_{t-1}
$$

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

### The Roofline Model — Quantifying "Mechanical Sympathy"

**Feynman explanation, made quantitative:** mechanical sympathy isn't a vibe, it's a measurable ceiling. The **roofline model** says a kernel's achievable performance is bounded by the minimum of (a) the CPU's peak compute throughput and (b) the memory bandwidth times the kernel's **arithmetic intensity** $I$ (FLOPs performed per byte moved from memory):

$$\text{Attainable GFLOPs/s} = \min\Big(\pi_{\max},\ \beta \cdot I\Big), \qquad I = \frac{\text{FLOPs}}{\text{Bytes moved}}$$

**Line-by-line:** $\pi_{\max}$ is the CPU's peak floating-point throughput (fixed by silicon — you cannot code your way past it). $\beta$ is peak memory bandwidth (also fixed by silicon — typically 20–50 GB/s on a single socket for DRAM, ~1TB/s for L1 cache). $I$ is the *only lever the programmer controls*: an EWMA kernel like the one below reads 1 float (8 bytes) and does ~3 FLOPs per step, so $I\approx 0.375$ — **catastrophically memory-bound**, meaning no amount of "faster arithmetic" helps; the only lever is reducing bytes moved (smaller dtypes — float32 instead of float64 — or better cache reuse). A GEMM (matrix multiply), by contrast, does $O(n^3)$ FLOPs on $O(n^2)$ bytes, giving high arithmetic intensity — **compute-bound**, where SIMD/vectorization genuinely helps.

**Say it out loud:** *"Before optimizing, I compute which side of the roofline a kernel sits on. If it's memory-bound (most of finance's time-series recursions are), the win is in data layout and dtype, not clever math — this is why I default `float32` for signal engines unless numerical precision analysis says otherwise, literally halving the bytes moved and often doubling throughput for zero algorithmic change."*

### Cache Hierarchy — Why Contiguity Isn't Optional

```
LEVEL     TYPICAL SIZE    TYPICAL LATENCY     IMPLICATION FOR NUMPY/PANDAS
────────  ─────────────  ──────────────────  ──────────────────────────────────
L1 cache   32-48 KB/core   ~1 ns (4 cycles)    A hot inner loop's working set
                                               should fit here for max speed
L2 cache   256KB-1MB/core  ~3-10 ns            Fits a few thousand doubles;
                                               tile/block your loops to this size
L3 cache   8-32 MB shared  ~10-20 ns           Shared across cores — false
                                               sharing between threads costs here
DRAM       GBs             ~80-120 ns          ~100x slower than L1 — a single
                                               cache MISS costs ~100 useful FLOPs
                                               worth of time
```

**Feynman explanation:** imagine your CPU as a chef and DRAM as the walk-in freezer three floors down. L1 cache is the cutting board in front of you. Every time an ingredient (data) isn't on the cutting board, the chef has to walk to the freezer (a ~100-cycle round trip) — during which time they could have chopped ~100 vegetables (FLOPs) if the ingredient had already been there. `np.ascontiguousarray` and column-major access patterns are the equivalent of "prep all your ingredients onto the cutting board in the order you'll use them," rather than fetching one at a time from three floors down in random order.

### Worked Benchmark — Quantifying the Win

```python
"""Benchmark comparing pure-Python, vectorized NumPy, and Numba nogil paths."""
from __future__ import annotations

import time

import numpy as np
from numba import njit


def ewma_pure_python(x: list[float], lam: float) -> list[float]:
    """Reference pure-Python EWMA (GIL-bound, boxed float objects).

    Args:
        x: Input observations as a Python list.
        lam: Smoothing factor.

    Returns:
        EWMA path as a Python list.
    """
    out = [x[0]]
    for t in range(1, len(x)):
        out.append(lam * x[t] + (1 - lam) * out[-1])
    return out


@njit(cache=True, fastmath=True)
def ewma_numba(x: np.ndarray, lam: float) -> np.ndarray:
    """Compiled EWMA kernel (see Q2 main listing for full docstring)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = x[0]
    for t in range(1, n):
        out[t] = lam * x[t] + (1.0 - lam) * out[t - 1]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.normal(size=500_000)
    data_list = data.tolist()

    t0 = time.perf_counter()
    _ = ewma_pure_python(data_list, 0.05)
    t_python = time.perf_counter() - t0

    ewma_numba(data[:10], 0.05)  # warm up JIT compilation
    t0 = time.perf_counter()
    _ = ewma_numba(data, 0.05)
    t_numba = time.perf_counter() - t0

    print(f"Pure Python: {t_python*1000:.1f} ms")
    print(f"Numba nogil: {t_numba*1000:.1f} ms")
    print(f"Speedup:     {t_python/t_numba:.1f}x  "
          f"(typically 50-150x for this memory-bound recursive kernel)")
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

### Why the Strategy Pattern, Specifically — the Alternative Designs Considered

**Feynman explanation of the design trade-off:** there are three ways to let a PM swap models: (1) a giant `if model_type == "ridge": ... elif model_type == "lstm": ...` branch inside the engine (violates open-closed principle — every new model requires editing the engine's core), (2) inheritance where `LSTMSignalEngine` subclasses `SignalEngine` (leads to a fragile class hierarchy — the engine's *orchestration logic*, like vol-scaling, gets duplicated or awkwardly overridden per subclass), or (3) **composition via the Strategy pattern** — the engine holds a reference to a `SignalModel` interface and never needs to know which concrete implementation it's talking to. Option 3 wins because it isolates the two concerns that actually **change at different rates**: orchestration/risk-scaling logic (changes rarely, needs rigorous review) versus model internals (changes weekly during active research) — coupling them would force every model experiment through the same review gate as risk-critical plumbing.

### Testability — Why This Matters More Than It Looks

```python
"""Unit-test-style harness demonstrating why Strategy pattern aids testing."""
from __future__ import annotations

import numpy as np


class DeterministicStubModel:
    """A stub SignalModel for testing the engine in isolation from any
    real model — returns a fixed, known score regardless of input.

    Attributes:
        fixed_score: The constant score returned by every prediction.
    """

    def __init__(self, fixed_score: float) -> None:
        """Initializes the stub with a constant return value.

        Args:
            fixed_score: The value predict() will always return.
        """
        self.fixed_score = fixed_score

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns a constant score array regardless of input features.

        Args:
            features: Ignored; present only to satisfy the interface.

        Returns:
            Array of shape (n_assets,) filled with fixed_score.
        """
        return np.full(features.shape[0], self.fixed_score)


def test_vol_scaling_clips_correctly() -> None:
    """Verifies the engine's vol-scaling and clipping logic in isolation,
    with the model held fixed — this is the entire point of dependency
    injection: the risk-scaling logic can be exhaustively unit-tested
    without any real (nondeterministic, slow-to-train) model in the loop.
    """
    from_engine_module = SignalEngine(DeterministicStubModel(fixed_score=10.0), vol_target=0.10)
    features = np.zeros((3, 2))
    realized_vol = np.array([0.05, 0.20, 0.50])  # low, medium, high vol assets

    signals = from_engine_module.generate(features, realized_vol)

    # Low-vol asset: 10.0 * (0.10/0.05) = 20.0, should clip to 1.0.
    assert signals[0] == 1.0, "Expected clipping at +1.0 for low-vol asset"
    # High-vol asset: 10.0 * (0.10/0.50) = 2.0, should clip to 1.0 as well.
    assert signals[2] == 1.0, "Expected clipping at +1.0 for high-vol asset"
    print("All vol-scaling/clipping assertions passed.")


if __name__ == "__main__":
    test_vol_scaling_clips_correctly()
```

**Say it out loud:** *"Because the engine depends on an abstract `SignalModel` interface rather than a concrete class, I can inject a deterministic stub that always returns a known value and exhaustively test every edge case of the risk-scaling logic — division by near-zero realized vol, clipping at both boundaries, NaN propagation — in milliseconds, without ever touching a slow-to-train real model. This separation is what makes a 10-15 person engineering team able to safely modify risk logic without waiting on quant researchers to retrain anything, and vice versa."*

### Extending to Multi-Model Ensembles (the Q28 Tie-In)

The same interface trivially extends to an ensemble — a `SignalModel` implementation that itself holds several other `SignalModel`s and combines their outputs:

```python
@dataclass(slots=True)
class EnsembleSignalModel(SignalModel):
    """Combines multiple SignalModel predictions via learned meta-weights.

    Attributes:
        sub_models: List of constituent SignalModel implementations.
        meta_weights: Per-model blending weights, shape (n_models,),
            typically fit via out-of-fold stacked regression.
    """

    sub_models: list
    meta_weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Computes a weighted blend of constituent model predictions.

        Args:
            features: Array of shape (n_assets, n_features).

        Returns:
            Array of shape (n_assets,) — the meta-weighted ensemble score.
        """
        predictions = np.stack([m.predict(features) for m in self.sub_models])
        return self.meta_weights @ predictions
```

This is precisely how a Ridge + GBM + LSTM ensemble (Q28) plugs into the exact same `SignalEngine` orchestration code with zero changes — the entire point of the abstraction.

[🔝 Back to Top](#table-of-contents)

---
---

# 🤖 GEN AI

---

## Q4 · Fine-Tuning vs. Prompt Engineering vs. RAG — and LoRA Math

> **Verified against 2025–2026 LLM interview banks** (DataCamp "Top 36 LLM Interview Questions," InterviewBit, GeeksforGeeks Generative AI guide, KalyanKS-NLP LLM Interview Hub): the fine-tuning-vs-RAG-vs-prompting decision tree and LoRA/PEFT mechanics are the single most-repeated Gen AI technical question across current postings — expect it in some form even in a 1-hour round.

**Open with the intuition:**
> "These three sit on a spectrum of *how much you change the model* vs. *how much you change the context*. Prompt engineering changes nothing about the weights — cheapest, fastest, best first move. RAG keeps weights frozen but injects fresh, proprietary, and *citable* information at inference time — this is what you want when facts change daily (financing rates) or must be auditable. Fine-tuning changes the weights — you reach for it only when you need a **behavior** change (format compliance, domain jargon, latency via a smaller distilled model), not a **knowledge** change."

### Decision Framework

```
                    Does the answer need FRESH / PROPRIETARY facts?
                                    │
                    ┌────────yes────┴────no────────┐
                    ▼                              ▼
                   RAG                  Is it a FORMATTING / STYLE /
        (embeddings + retriever         DOMAIN-BEHAVIOR problem?
         + frozen LLM + citations)                 │
                                        ┌────yes───┴────no─────┐
                                        ▼                      ▼
                              FINE-TUNE (LoRA)          PROMPT ENGINEER
                              (compress a workflow      (few-shot, CoT,
                               into weights, or          structured output,
                               distill to smaller,       system prompt)
                               cheaper model)
```

### LoRA — Low-Rank Adaptation, Derived

Full fine-tuning updates weight matrix $W_0 \in \mathbb{R}^{d\times k}$ directly: $W = W_0 + \Delta W$, with $\Delta W$ having $d\times k$ free parameters — for a 7B-parameter model that's billions of trainable values.

LoRA's hypothesis: the *update* $\Delta W$ needed to adapt a pretrained model to a new task has **low intrinsic rank** $r \ll \min(d,k)$. So constrain:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k}, \quad r \ll \min(d,k)$$

Forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

**Line-by-line:**
1. $W_0$ stays **frozen** — no gradient computed for it, so no optimizer state (Adam moment estimates) needed for the base weights. This is where the memory savings come from.
2. $A$ is initialized from $\mathcal{N}(0,\sigma^2)$, $B$ initialized to **zero** — so at step 0, $BA = 0$ and $h = W_0 x$, i.e., training starts exactly at the pretrained behavior (no cold-start degradation).
3. Only $A$ and $B$ receive gradients: parameter count drops from $d\times k$ to $r(d+k)$. For $d=k=4096,\ r=8$: $16.7\text{M} \to 65.5\text{K}$ — a **256× reduction**.
4. At inference, $B A$ can be **merged** into $W_0$ (since it's just addition), so there is **zero latency overhead** versus the base model — unlike adapter layers that add sequential compute.

**Feynman explanation:** Imagine the pretrained model already speaks fluent "general English" and you just need it to pick up "financing-desk jargon." You don't need to re-teach it English (don't touch $W_0$) — you need a small correction, and that correction, empirically, lives in a low-dimensional subspace. $B$ and $A$ are two thin "translation lenses" bolted onto the frozen brain; the low rank $r$ is the size of that lens.

### Deriving the LoRA Gradient — Why Training Is Stable Even Starting From Zero

A natural worry: if $B$ starts at zero, is the gradient zero everywhere at initialization (a dead start)? Let's check by differentiating the loss $\mathcal{L}$ with respect to $A$ and $B$ directly. With $h = W_0 x + \tfrac{\alpha}{r}BAx$:

$$\frac{\partial\mathcal{L}}{\partial B} = \frac{\partial\mathcal{L}}{\partial h}\cdot\frac{\partial h}{\partial B} = \frac{\alpha}{r}\left(\frac{\partial\mathcal{L}}{\partial h}\right)(Ax)^\top, \qquad \frac{\partial\mathcal{L}}{\partial A} = \frac{\alpha}{r}B^\top\left(\frac{\partial\mathcal{L}}{\partial h}\right)x^\top$$

**Line-by-line:** even with $B=0$, the gradient with respect to $B$ is $\frac{\alpha}{r}(\partial\mathcal{L}/\partial h)(Ax)^\top$ — this is **generally non-zero** because $A$ is randomly initialized (not zero) and $Ax\neq 0$. So $B$ receives a real, non-degenerate gradient signal from step 1 and starts moving away from zero immediately. Meanwhile $\partial\mathcal{L}/\partial A = \frac{\alpha}{r}B^\top(\partial\mathcal{L}/\partial h)x^\top$ **is** exactly zero at step 0 (because $B=0$) — so $A$'s very first gradient step is zero, and $A$ only starts moving once $B$ has moved away from zero on the prior step. This is a deliberate, benign asymmetry: it guarantees $h=W_0x$ exactly at initialization (no output perturbation from an untrained adapter) while still guaranteeing training proceeds via $B$'s immediate gradient.

### QLoRA — Combining Quantization with LoRA

For a 70B-parameter model, even the *frozen* weights $W_0$ may not fit in a single GPU's memory in float16 (140GB+). QLoRA's insight: since $W_0$ is frozen and only used in forward/backward matrix-multiplies (never updated by an optimizer), it can be stored in a heavily compressed 4-bit format (NF4 — "NormalFloat4," a quantization grid optimized for the empirical distribution of pretrained weights, which is approximately Gaussian) and **de-quantized to bf16 on-the-fly** only for the duration of each matmul:

$$W_0^{\text{fp4}} \xrightarrow{\text{dequantize per-block}} W_0^{\text{bf16}} \xrightarrow{\text{matmul}} W_0^{\text{bf16}}x, \qquad \text{LoRA branch } BAx \text{ stays in full precision}$$

**Say it out loud:** *"QLoRA reduces the frozen base model's memory footprint by roughly 4x (16-bit to 4-bit) with only a small accuracy cost from quantization error, while the trainable LoRA adapters — which are tiny in parameter count — stay in full precision, so almost all of the model's actual learning capacity during fine-tuning is unaffected by the compression. This is precisely how a single GPU can fine-tune a 65B+ parameter model that would otherwise require an 8-GPU cluster just to hold the frozen weights."*

### Worked Numerical Example — Real Parameter Counts for a Financing-Terms Extraction Fine-Tune

Suppose we fine-tune a 7B-parameter open-weight model on Q attention projections across 32 transformer layers, each with $d_{\text{model}}=4096$:

```
CONFIGURATION                    TRAINABLE PARAMS      MEMORY (Adam, fp32 optimizer states)
────────────────────────────    ───────────────────   ───────────────────────────────────────
Full fine-tune (all 7B params)   7,000,000,000          7B * 4 bytes (weights) +
                                                        7B * 8 bytes (Adam m,v) = ~84 GB
LoRA r=8 on Q,V only (32 layers) 32 * 2 * 8*(4096+4096)
                                  = 32 * 2 * 65,536
                                  = 4,194,304 params     4.2M * 12 bytes ≈ 50 MB
                                                        (>1600x less optimizer memory)
LoRA r=8 on Q,K,V,O (32 layers)  32 * 4 * 65,536
                                  = 8,388,608 params    ≈ 100 MB
```

**Feynman explanation, tied to why this matters for a bank:** you don't need a research supercomputer to specialize a model to financing-desk jargon and term-sheet formats — you need one workstation-class GPU, a few hundred labeled examples, and an afternoon. This economic fact is *why* fine-tuning has become viable for individual desks rather than being centralized in a single "AI lab" — which directly supports the JD's "build an AI/ML capability from the ground up" mandate: a lean team can realistically own fine-tuning workflows in-house rather than depending entirely on vendor-hosted frontier models.

### When LoRA Is the *Wrong* Choice — Failure Modes to Volunteer

- **Task requires new factual knowledge, not new behavior** — LoRA (like all fine-tuning) bakes information into weights at training time; it cannot inject *today's* financing rate, which changes daily. This is a RAG problem, not a fine-tuning problem (ties back to the Q4 decision tree above).
- **Catastrophic forgetting under aggressive rank/learning rate** — even LoRA can measurably degrade a model's general instruction-following if $r$ is too large or too many layers are adapted; the fix is validating on a held-out **general-capability** benchmark alongside the domain-specific one, not just the target task.
- **Multiple conflicting fine-tunes on the same base model** — LoRA's practical advantage of "swap adapters without reloading the full model" only holds if you don't need two adapters' behaviors simultaneously; merging multiple LoRA adapters into one base model is non-trivial and can produce unpredictable interference.

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
 ┌─────────────────┐                 ┌────────────────────┐
 │ Embed chunks    │                 │ ANN search (HNSW/  │
 │ (bi-encoder)    │────────────────▶│ IVF-PQ) top-k=50   │
 └───────┬─────────┘                 └───────┬────────────┘
         ▼                                   ▼    
 ┌─────────────────┐                 ┌──────────────────────┐
 │ Vector store    │                 │ Cross-encoder rerank │
 │ (metadata:      │                 │ → top-k=6            │
 │ doc_id, date,   │                 └───────┬──────────────┘
 │ counterparty)   │                         ▼   
 └─────────────────┘                 ┌────────────────────┐
                                     │ LLM synthesis w/   │
                                     │ inline citations   │
                                     └───────┬────────────┘
                                             ▼    
                                     Answer + source spans
                                     (auditable for Compliance)
```

**Why cross-encoder rerank matters (the math):** a bi-encoder scores relevance as $\text{sim}(E_q(q), E_d(d)) = \frac{E_q(q)\cdot E_d(d)}{\lVert E_q(q)\rVert \lVert E_d(d)\rVert}$ — query and doc are embedded **independently**, so the model never sees them jointly; it's fast ($O(1)$ per doc at query time) but loses fine-grained token interaction. A cross-encoder instead scores $f(q \oplus d)$ jointly through a full transformer, capturing token-level interaction (e.g., matching "GC" specifically to "general collateral" rather than any repo term) at the cost of $O(N)$ forward passes — hence: bi-encoder for cheap top-50 recall, cross-encoder for expensive top-6 precision.

### Chunking Strategy — The Decision That Matters Most in Practice

**Feynman explanation:** chunking is the single highest-leverage RAG design decision, and it's almost never discussed as deeply as the model architecture — yet a bad chunking strategy silently caps your ceiling regardless of how good the embedding model is. Two failure modes dominate: **chunks too large** (a 2000-token chunk containing both the haircut clause and an unrelated collateral-eligibility clause dilutes the embedding — the vector becomes an "average" of two concepts, hurting retrieval precision for either), and **chunks too small** (a 50-token chunk containing only "8.5%" with no surrounding context is unusable even if perfectly retrieved — the LLM can't tell what the 8.5% refers to).

```
CHUNKING STRATEGY            TRADE-OFF
───────────────────────────  ──────────────────────────────────────────────
Fixed-size (512 tokens)      Simple, fast; can split mid-clause, destroying
                             semantic coherence of a legal definition
Sentence-boundary aware      Respects grammar; still can separate a defined
                             term from its definition if they span sentences
Semantic/structural          Splits at document structure (Section, Clause,
(recommended for legal       Article headers) — preserves the unit a lawyer
docs like ISDA/repo master)  actually reasons about; each chunk is a complete
                             clause with its heading as metadata
Recursive with overlap       10-15% overlap prevents a clause's opening
(10-15%)                     sentence from being orphaned in the prior chunk
```

**Say it out loud:** *"For ISDA/MSFTA/repo agreements, I chunk along the document's own structural hierarchy — Section, Clause, Sub-clause — rather than a fixed token count, because the unit of legal meaning is the clause, not an arbitrary 512-token window. I attach the clause heading and agreement type as metadata on every chunk, so even a chunk that reads ambiguously in isolation carries enough metadata for the reranker and the LLM to disambiguate it."*

### Hybrid Search — Why Pure Dense Retrieval Isn't Enough for Financing Documents

Dense embeddings excel at *semantic* similarity but can miss exact-match requirements that matter enormously in legal/financial text — a query for "ISIN US912828U816" needs an **exact** lexical match, not a semantically-nearby ISIN. This motivates combining dense retrieval with sparse lexical retrieval (BM25):

$$\text{BM25}(q,d) = \sum_{t\in q} \text{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d) + k_1\left(1-b+b\cdot\frac{|d|}{\text{avgdl}}\right)}$$

**Line-by-line:** $f(t,d)$ is the term frequency of query term $t$ in document $d$; $\text{IDF}(t)=\ln\!\big(\frac{N-n_t+0.5}{n_t+0.5}+1\big)$ downweights terms that appear in most documents (like "the" or "agreement") and upweights rare, discriminative terms (like a specific ISIN or counterparty name); the denominator's $\left(1-b+b\frac{|d|}{\text{avgdl}}\right)$ term normalizes for document length so a long document doesn't win purely by containing the term more times through sheer volume; $k_1$ controls how quickly additional term occurrences saturate in their contribution (diminishing returns — the 5th occurrence of a term matters much less than the 1st).

**Hybrid fusion** combines dense and sparse scores via **Reciprocal Rank Fusion (RRF)**, which is more robust than trying to calibrate raw score scales across two different retrieval systems:

$$\text{RRF}(d) = \sum_{\text{system}\in\{\text{dense, BM25}\}} \frac{1}{k + \text{rank}_{\text{system}}(d)}$$

**Feynman explanation:** RRF sidesteps the "how do I compare a cosine similarity of 0.82 to a BM25 score of 14.3?" problem entirely by throwing away raw scores and working only with **rank position** — a document that's rank 1 in either system contributes $1/(k+1)$, a large, comparable number; a document buried at rank 200 in both contributes almost nothing. $k$ (typically 60) is a smoothing constant that prevents rank-1 documents from dominating too absolutely.

```python
"""Hybrid dense + BM25 retrieval fused via Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuses multiple ranked retrieval result lists via RRF.

    Args:
        ranked_lists: One ranked list of doc_ids per retrieval system
            (e.g., [dense_results, bm25_results]), each ordered best-first.
        k: RRF smoothing constant; larger k flattens the influence of
            top-ranked documents relative to lower-ranked ones.

    Returns:
        List of (doc_id, fused_score) tuples sorted by descending score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


if __name__ == "__main__":
    dense_results = ["clauseA", "clauseC", "clauseB", "clauseD"]
    bm25_results = ["clauseB", "clauseA", "clauseE", "clauseC"]

    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
    print("Fused ranking (hybrid dense + BM25):")
    for doc_id, score in fused:
        print(f"  {doc_id}: {score:.5f}")
```

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

> **Verified against the CFA Institute's "For LLMs in the Financial Industry: A Practical Guide"** and current LLM interview banks: RAG is confirmed as the dominant financial-industry adaptation pattern (vs. full fine-tuning), and faithfulness/hallucination-rate evaluation is called out explicitly as the key production risk in finance, where "financial language is highly technical and context dependent, making LLMs susceptible to hallucinations."

**Feynman explanation:** You cannot grade an essay with a ruler. LLM outputs are open-ended, so evaluation needs metrics that separate **faithfulness** (did it stick to the retrieved facts?) from **relevance** (did it answer the actual question?) from **correctness** (is it factually right against ground truth?) — conflating these three is the single biggest mistake I see in Gen AI eval.

### Core RAG Metrics, Defined

$$\text{Faithfulness} = \frac{|\{\text{claims in answer supported by context}\}|}{|\{\text{claims in answer}\}|}$$

$$\text{Context Precision} = \frac{1}{\big|\{\text{relevant chunks retrieved}\}\big|}\sum_{k}\text{Precision@}k \cdot \mathbb{1}[\text{relevant}_k]$$

$$\text{Answer Relevance} = \frac{1}{N}\sum_{i=1}^{N}\cos\big(E(q),\, E(q_i')\big)$$

where $q_i'$ are LLM-generated "reverse questions" from the answer — **if the answer is truly relevant, an LLM asked to reconstruct the question from it should land close to the real query in embedding space.**

**Say it out loud:** *"Faithfulness decomposes the answer into atomic claims and checks each against retrieved context with a second LLM call — it's a hallucination detector for RAG specifically. Answer relevance flips the problem: generate synthetic questions that the answer would address, and if they don't align with what was actually asked, the model likely answered a different (perhaps easier) question — a common failure mode called 'answer drift'."*

### Worked Numerical Example — Computing Faithfulness by Hand

Suppose the RAG system answers: *"The GC repo haircut for HY collateral is 8.5%, the tenor is 30 days, and the desk head who approved this rate was Jane Smith."* Decompose into atomic claims:

```
CLAIM 1: "GC repo haircut for HY collateral is 8.5%"       → Check against retrieved context
CLAIM 2: "tenor is 30 days"                                → Check against retrieved context
CLAIM 3: "desk head who approved this rate was Jane Smith" → Check against retrieved context
```

If the retrieved context supports Claims 1 and 2 verbatim but contains **no mention** of any approver named Jane Smith, then

```math
\text{claims\_supported}=2, \quad \text{claims\_total}=3
```

, and:

```math
\text{Faithfulness} = \frac{2}{3} = 0.667
```

**Say it out loud:** *"A faithfulness score of 0.667 flags a specific, actionable problem: Claim 3 is a hallucination — the model fabricated an approver name that wasn't in any retrieved document, likely because 'Jane Smith' is a plausible-sounding name pattern from its pretraining distribution rather than an actual fact from our corpus. This is exactly the failure mode that matters most in a regulated environment — a plausible-sounding but fabricated attribution — and it's invisible to metrics that only measure 'did the answer sound relevant,' which is why faithfulness must be measured separately from relevance."*

### Extending the Framework — Groundedness at the Span Level

A more rigorous production check goes beyond claim-level faithfulness to require every claim to cite a **specific source span** (as required in the Q8 schema), then independently verifies the citation:

$$\text{Groundedness}(x) = \begin{cases}1 & \text{if span}(x) \text{ exists verbatim in the cited chunk AND supports claim } x\\0 & \text{otherwise}\end{cases}$$

This catches a subtler failure than plain faithfulness: a model can cite a *real* chunk that happens to be topically related but doesn't actually support the specific numeric claim made (e.g., citing the haircut clause for **Investment Grade** collateral to support a claim about **High Yield** collateral) — this "near-miss citation" passes a loose faithfulness check but fails groundedness, and it's a genuinely common LLM failure mode worth naming explicitly in an interview.

### Human Evaluation — Why Automated Metrics Are Necessary But Not Sufficient

**Feynman explanation:** automated metrics (faithfulness, relevance, groundedness) are a **cheap, fast, biased proxy** for what you actually care about — whether a trader would trust and correctly act on the answer. They're computed by another LLM (LLM-as-judge), which has its own blind spots and can be gamed by an answer that superficially "looks well-cited" without being substantively correct. The right production setup is a **two-tier evaluation**: automated metrics run on every single query in a CI-style regression suite (catching regressions cheaply, continuously), while a smaller sample (e.g., 50 queries/week) goes to human domain experts (financing desk traders) for a rubric-based review — precision, actionability, and "would I have caught this error before it caused a trading mistake." The automated metrics' correlation with human judgment should itself be tracked over time — if it drifts, the automated eval harness needs recalibration, not just the underlying RAG system.

```
EVALUATION TIER       FREQUENCY        CATCHES                         COST
────────────────────  ───────────────  ──────────────────────────────  ──────
Automated (LLM-judge) Every query /    Faithfulness regressions,       Cheap
                      every CI run     retrieval precision drops
Human expert review   Weekly sample    Subtle domain errors, near-     Expensive
                                       miss citations, tone/format
A/B live monitoring   Continuous       Real trader behavior — did      Requires
                                       they act on/correct the answer  instrumentation
```

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
──────────────────────────────────     ────────────────────────────     ─────────────────────────────
Agentic coding / repo-scale            Claude (Sonnet/Opus class)       Best long-horizon tool-use,
refactors, CI-integrated workflows     via Claude Code                  strong instruction-following
                                                                        over multi-file context

Long-document synthesis (ISDA,         Claude (large context)           Long context window +
research notes), RAG generation                                         low hallucination rate on
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

### Quantifying the Trade-off — A Cost/Latency/Quality Framework

**Feynman explanation, made concrete:** "pick the cheapest model that clears the bar" needs a *bar* — otherwise it's just an opinion. Define an expected-value framework for a routed task:

$$\mathbb{E}[\text{Value}] = P(\text{correct})\cdot V_{\text{correct}} - P(\text{incorrect})\cdot C_{\text{error}} - \text{Cost}_{\text{inference}} - \lambda\cdot\text{Latency}$$

**Line-by-line:** $V_{\text{correct}}$ is the business value of a correct answer (e.g., time saved by a trader not manually reading a term sheet); $C_{\text{error}}$ is the cost of an *incorrect* answer, which for a term-sheet extraction feeding into a trade booking system could be a real financial loss — this term is why financing/trading use cases justify paying for a more accurate (often more expensive) model, whereas a low-stakes internal FAQ bot does not; $\text{Cost}_{\text{inference}}$ scales with model size and token count; $\lambda$ is how much the business cares about latency (near-zero for async batch jobs like nightly report synthesis, large for a trader-facing chat interface with real-time expectations).

**Say it out loud:** **"I don't have a favorite model — I have a routing function. For a structured extraction feeding into a booking system, $C_{\text{error}}$ is high, so I pay for the more accurate/larger model despite the cost. For a high-volume internal classification microservice, $C_{\text{error}}$ per single miss is low and volume is high, so a fine-tuned small model wins on the cost term alone. This framework is what I'd propose the team formalize into an actual routing layer rather than hard-coding model choice per use case."**

### A Concrete Routing Architecture

```
                         INCOMING REQUEST
                                │
                    ┌───────────▼───────────────┐
                    │  Task Classifier          │   (cheap, fast — itself a
                    │  (structured-extraction   │    small fine-tuned model
                    │   vs. reasoning vs.       │    or rule-based router)
                    │   agentic-coding vs. chat)│
                    └───────────┬───────────────┘
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
  STRUCTURED EXTRACT     MULTI-STEP REASONING     AGENTIC CODE / REPO WORK
  → small fine-tuned     → frontier reasoning      → Claude Code (long-horizon
    model, JSON mode,      model (Claude Opus /       tool use, multi-file
    temp=0, <200ms          GPT o-series), CoT         context, CI-integrated)
    latency SLA             prompting enabled
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  Response validated       │  (Q8 schema validation,
                    │  (schema check / eval     │   Q6 faithfulness gate)
                    │  gate) before returning   │
                    └───────────────────────────┘
```

### Why This Matters for the JD's "Suggest the Right LLM Models" Requirement

The JD explicitly asks for someone who can suggest the right model beyond just Claude Code — this routing framework is the concrete deliverable that answers that requirement. It also directly ties to **cost governance at scale**: a 50+ problem statement pipeline (per the JD) run entirely through a frontier model, un-routed, would be needlessly expensive; building the routing/classification layer once (itself a small, cheap model) pays for itself as soon as more than a handful of the pipeline's problem statements go into production.

[🔝 Back to Top](#table-of-contents)

---
---

## Q8 · Structured Extraction & Prompt Engineering for Term Sheets

### Designing the Few-Shot Examples — Why Edge Cases Matter More Than "Easy" Cases

**Feynman explanation:** the biggest mistake in few-shot prompt design is choosing examples that are all "easy" — clean, unambiguous documents. The model needs to see the **boundary conditions** it will actually fail on:

```
EXAMPLE TYPE                           WHY IT MUST BE IN THE FEW-SHOT SET
────────────────────────────────────   ──────────────────────────────────────────
Clean, unambiguous document            Establishes the baseline expected format
Document with a MISSING field          Teaches the model to output null rather
                                       than hallucinate a plausible-sounding value
Document with AMBIGUOUS units          Teaches disambiguation (e.g., "8.5" without
(8.5 vs. 8.5% vs. 850bps)              a % sign — is it 8.5% or 0.085%?)
Document with a superseded/amended     Teaches the model to extract the CURRENT
clause (old rate crossed out, new      (amended) value, not the first value it
rate stated in an amendment)           encounters textually
Multi-tranche document (several        Teaches the model NOT to conflate terms
haircuts for different collateral      across tranches into a single answer
types in one document)
```

**Say it out loud:** *"If I only show the model clean documents in my few-shot examples, it will confidently extract a value from an ambiguous or superseded clause because it's never seen an example demonstrating the correct 'refuse or flag' behavior. I deliberately over-represent edge cases in few-shot prompts relative to their natural frequency, precisely because those are the failure modes with the highest cost if silently wrong."*

### Error Taxonomy for Structured Extraction — What Goes Wrong and Why

$$\text{Total Error Rate} = \underbrace{P(\text{schema violation})}_{\text{caught by Pydantic}} + \underbrace{P(\text{schema-valid but wrong})}_{\text{NOT caught by Pydantic — silent}}$$

**Line-by-line:** Pydantic validation (shown in the code below) catches the first term for free — a `haircut_pct` of 150 or a non-numeric `tenor_days` fails loudly and routes to human review. But the second term — a syntactically valid JSON with a **plausible but incorrect** number (e.g., extracting 8.0% when the document actually says 8.5%) — passes every schema check and is the truly dangerous failure mode, because nothing in the pipeline flags it. This is precisely why the `source_span` field is not optional decoration: it lets a downstream **human or automated groundedness check** (Q6) catch silent numeric errors by comparing the claimed span against the extracted value, something schema validation alone can never do.

### Escalation Path — What Happens When Validation Fails

```
EXTRACTION FAILS VALIDATION (schema OR groundedness check)
                    │
                    ▼
         Route to human-review queue with:
         - Original document (highlighted at retrieval span)
         - Failed extraction attempt (for context, not blind trust)
         - Specific failure reason (schema field X, or groundedness
           mismatch on field Y)
                    │
                    ▼
         Human corrects → correction logged as a NEW few-shot
         example candidate → periodic prompt-set refresh
         (this is how the extraction prompt improves over time
         without a full re-fine-tune, closing the loop the JD
         describes as "continuous improvement")
```

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

$$\frac{\partial S}{\partial \beta} = -2X^\top y + 2X^\top X\beta = 0 \;\;\Rightarrow\;\; X^\top X\,\hat\beta = X^\top y \;\;\Rightarrow\;\; \hat\beta = (X^\top X)^{-1}X^\top y$$

**Line-by-line:** the first term $-2X^\top y$ is the derivative of the linear cross-term $-2\beta^\top X^\top y$ with respect to $\beta$ (a standard $\partial(a^\top\beta)/\partial\beta = a$ rule, doubled by the scalar out front); the second term comes from $\partial(\beta^\top A \beta)/\partial\beta = 2A\beta$ for symmetric $A = X^\top X$. Setting the gradient to zero gives the **normal equations**; because $S(\beta)$ is convex (Hessian $2X^\top X \succeq 0$), this stationary point is the global minimum, and it's unique whenever $X^\top X$ is invertible (i.e., $X$ has full column rank — no perfect multicollinearity).

**Gauss-Markov proof sketch (OLS is BLUE — Best Linear Unbiased Estimator):**

*Unbiasedness:* $\hat\beta = (X^\top X)^{-1}X^\top(X\beta+\varepsilon) = \beta + (X^\top X)^{-1}X^\top\varepsilon$. Taking expectation with $X$ fixed and $\mathbb{E}[\varepsilon]=0$: $\mathbb{E}[\hat\beta]=\beta$.

*Minimum variance among linear unbiased estimators:* Consider any other linear unbiased estimator $\tilde\beta = Cy$ with $C = (X^\top X)^{-1}X^\top + D$ for some matrix $D$. Unbiasedness requires $\mathbb{E}[\tilde\beta]=CX\beta=\beta$ for **every** $\beta$, i.e. $CX=I$. Substituting: $CX = (X^\top X)^{-1}X^\top X + DX = I + DX$, so unbiasedness forces $DX = 0$. Now compute the variance:

$$\text{Var}(\tilde\beta) = C\,\text{Var}(y)\,C^\top = \sigma^2 CC^\top$$

$$CC^\top = \Big[(X^\top X)^{-1}X^\top + D\Big]\Big[X(X^\top X)^{-1} + D^\top\Big] = (X^\top X)^{-1} + (X^\top X)^{-1}X^\top D^\top + DX(X^\top X)^{-1} + DD^\top$$

The two cross terms vanish because $DX=0 \Rightarrow (DX)^\top = X^\top D^\top = 0$. So:

$$\text{Var}(\tilde\beta) = \sigma^2\Big[(X^\top X)^{-1} + DD^\top\Big] = \text{Var}(\hat\beta) + \sigma^2 DD^\top$$

Because $DD^\top$ is positive semi-definite (it is a Gram matrix — for any vector $v$, $v^\top DD^\top v = \lVert D^\top v\rVert^2 \geq 0$), we have $\text{Var}(\tilde\beta) \succeq \text{Var}(\hat\beta)$ in the positive-semi-definite (Loewner) ordering — meaning for **any** linear combination $a^\top\beta$, $\text{Var}(a^\top\tilde\beta) \geq \text{Var}(a^\top\hat\beta)$. Equality holds iff $D=0$, i.e., $\tilde\beta=\hat\beta$. $\blacksquare$

**Geometric interpretation (the picture the interviewer wants you to be able to draw):** $\hat y = X\hat\beta$ is the **orthogonal projection** of $y$ onto the column space of $X$, via the projection (hat) matrix $H = X(X^\top X)^{-1}X^\top$, which is idempotent ($H^2=H$) and symmetric ($H^\top=H$). The residual vector $\hat\varepsilon = y-\hat y = (I-H)y$ is, by construction, **orthogonal** to every column of $X$: $X^\top\hat\varepsilon = X^\top(I-H)y = (X^\top - X^\top)y = 0$. This orthogonality is exactly what the normal equations $X^\top X\hat\beta = X^\top y$ encode — "the residual is uncorrelated, in-sample, with every regressor," which is why you can never regain explanatory power by re-including a variable already in $X$.

### Worked Numerical Example — OLS by Hand on a Toy Dataset

Consider fitting a one-factor model of financing spread changes on a rate-differential factor, with just 4 observations for hand-tractability:

$$X = \begin{pmatrix}1 & -1\\ 1 & 0\\ 1 & 1\\ 1 & 2\end{pmatrix}, \qquad y = \begin{pmatrix}0.5\\ 1.0\\ 2.5\\ 3.0\end{pmatrix}$$

Compute $X^\top X$:

$$X^\top X = \begin{pmatrix}4 & 2\\ 2 & 6\end{pmatrix}$$

**Line-by-line:** the $(1,1)$ entry is $\sum 1^2 = 4$ (n observations); the $(1,2)$ and $(2,1)$ entries are $\sum x_i = (-1+0+1+2)=2$; the $(2,2)$ entry is $\sum x_i^2 = (1+0+1+4)=6$.

Compute $X^\top y$:

$$X^\top y = \begin{pmatrix}\sum y_i\\ \sum x_i y_i\end{pmatrix} = \begin{pmatrix}0.5+1.0+2.5+3.0\\ (-1)(0.5)+(0)(1.0)+(1)(2.5)+(2)(3.0)\end{pmatrix} = \begin{pmatrix}7.0\\ 8.0\end{pmatrix}$$

Invert the $2\times2$ matrix using:

$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}, \quad \text{with determinant} \quad 4\cdot6 - 2\cdot2 = 20
$$

:

$$
(X^\top X)^{-1} = \frac{1}{20}\begin{pmatrix}6 & -2\\ -2 & 4\end{pmatrix}
$$

Finally:

$$
\hat\beta = (X^\top X)^{-1}X^\top y = \frac{1}{20}\begin{pmatrix}6 & -2\\ -2 & 4\end{pmatrix}\begin{pmatrix}7.0\\ 8.0\end{pmatrix} = \frac{1}{20}\begin{pmatrix}6(7.0)-2(8.0)\\ -2(7.0)+4(8.0)\end{pmatrix} = \frac{1}{20}\begin{pmatrix}26\\ 18\end{pmatrix} = \begin{pmatrix}1.3\\ 0.9\end{pmatrix}
$$

**Say it out loud:** **"The fitted model is $\hat y = 1.3 + 0.9x$ — an intercept of 1.3 and a slope of 0.9 per unit of the rate-differential factor. I'd sanity-check this by eye: at $x=-1$, predicted $\hat y = 1.3-0.9=0.4$ against observed $0.5$; at $x=2$, predicted $\hat y=1.3+1.8=3.1$ against observed $3.0$ — small, plausible residuals, consistent with a good linear fit, without needing to run any code to build that intuition."** This hand-computation is exactly the kind of check I'd do at a whiteboard if asked to derive OLS live.

### R-squared, Derived — What It Actually Measures

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \qquad SS_{\text{res}}=\sum_i(y_i-\hat y_i)^2, \qquad SS_{\text{tot}}=\sum_i(y_i-\bar y)^2$$

**Line-by-line:** $SS_{\text{tot}}$ is the total variance in $y$ around its mean — "how much there was to explain in the first place." $SS_{\text{res}}$ is what's left over after the model's best effort — "how much remains unexplained." The ratio $SS_{\text{res}}/SS_{\text{tot}}$ is the fraction of variance *still unexplained*, so $1$ minus that ratio is the fraction *explained*. A crucial, often-missed subtlety: this decomposition $SS_{\text{tot}} = SS_{\text{reg}} + SS_{\text{res}}$ (with no cross-term) is only exact when the model includes an intercept and is fit by OLS — it relies on the same orthogonality property from the projection-matrix argument above ($X^\top\hat\varepsilon=0$), which guarantees the fitted values and residuals are uncorrelated. **Feynman explanation:** if someone reports $R^2$ for a model fit **without** an intercept, or for a nonlinear/non-OLS model, the identity can break and $R^2$ can even go negative — a fact worth volunteering to show you understand $R^2$ isn't a universal, always-well-behaved quantity.

### Why This Question Escalates — The Likely Interview Follow-Up

After deriving OLS, expect an immediate follow-up: **"What if $X^\top X$ is singular?"** — this is deliberately baited to test whether you jump straight to Ridge (Q10) or first understand *why* it's singular (perfect multicollinearity — one regressor is an exact linear combination of others, e.g., including both a rate level and its lag plus their exact difference as three separate regressors) and whether the fix should be **structural** (drop the redundant regressor — it carries zero unique information) versus **statistical** (Ridge, when the near-singularity is a matter of degree — correlated but not perfectly collinear factors, which is the much more common real-world case in financial factor panels).

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

> **Why this is Q10, not an afterthought:** practitioner reports on Wall Street Oasis's quant-interview forum are explicit that buy-side/sell-side quant research interviews weight "penalized regression and ensemble methods — LASSO, Ridge, Elastic Net, Gradient Boosted Trees, Random Forest" more heavily than exotic deep learning, precisely because financial factor panels are highly collinear. Expect this to be asked directly.

**Ridge — closed form.** Add an $\ell_2$ penalty:

$$S_{\text{ridge}}(\beta) = (y-X\beta)^\top(y-X\beta) + \lambda\beta^\top\beta$$

$$\frac{\partial S_{\text{ridge}}}{\partial\beta} = -2X^\top y + 2X^\top X\beta + 2\lambda\beta = 0 \;\;\Rightarrow\;\; \hat\beta_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$$

**Say it out loud:** **"Adding $\lambda I$ to $X^\top X$ before inverting guarantees invertibility even when $X$ is rank-deficient or near-collinear — this is literally 'ridge' because you're adding a ridge along the diagonal — and it shrinks every coefficient toward zero, trading a little bias for a large reduction in variance whenever predictors are correlated (which financial factors always are)."**

**Lasso — no closed form, subgradient condition.** The $\ell_1$ penalty $\lambda\lVert\beta\rVert_1$ is non-differentiable at 0. The KKT/subgradient stationarity condition for coordinate $j$:

$$-2x_j^\top(y - X\beta) + \lambda\, s_j = 0, \quad s_j \in \begin{cases}\{\text{sign}(\beta_j)\} & \beta_j\neq 0\\ [-1,1] & \beta_j = 0\end{cases}$$

This is why Lasso produces **exact sparsity**: whenever the correlation of a feature with the residual is smaller in magnitude than $\lambda/2$, the optimal $\beta_j$ is driven exactly to 0 (a corner solution of the $\ell_1$ ball), unlike Ridge's smooth, everywhere-differentiable penalty which shrinks but never zeroes out.

**Elastic Net** blends both:

$$S_{\text{EN}}(\beta) = \lVert y-X\beta\rVert_2^2 + \lambda_1\lVert\beta\rVert_1 + \lambda_2\lVert\beta\rVert_2^2$$

solved via **coordinate descent** — cycling through $j=1,\dots,k$, holding all other coefficients fixed, and applying the soft-thresholding operator:

$$\hat\beta_j \leftarrow \frac{S\big(x_j^\top r_{-j},\ \lambda_1/2\big)}{1 + \lambda_2}, \qquad S(z,\gamma) = \text{sign}(z)\max(|z|-\gamma,\,0)$$

where $r_{-j} = y - X_{-j}\beta_{-j}$ is the partial residual excluding feature $j$.

**Formal bias-variance derivation for Ridge.** Write $\hat\beta_{\text{ridge}} = (X^\top X+\lambda I)^{-1}X^\top y = W_\lambda \hat\beta_{\text{OLS}}$ where $W_\lambda = (X^\top X+\lambda I)^{-1}X^\top X$ is a "shrinkage matrix" (eigenvalues in $(0,1)$ for $\lambda>0$). Then:

$$\text{Bias}(\hat\beta_{\text{ridge}}) = \mathbb{E}[\hat\beta_{\text{ridge}}] - \beta = (W_\lambda - I)\beta = -\lambda(X^\top X+\lambda I)^{-1}\beta$$

**Line-by-line:** substitute $\mathbb{E}[\hat\beta_{\text{OLS}}]=\beta$ so $\mathbb{E}[\hat\beta_{\text{ridge}}]=W_\lambda\beta$; the bias is exactly zero only at $\lambda=0$ and grows (in magnitude) monotonically with $\lambda$ — this is the "cost" side of the trade, and it is **directional**: Ridge always shrinks $\hat\beta$ toward the origin.

$$\text{Var}(\hat\beta_{\text{ridge}}) = \sigma^2 (X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}$$

**Line-by-line:** derived the same way as the OLS variance in Q9 — substitute the linear map $\hat\beta_{\text{ridge}}=A_\lambda y$ with $A_\lambda=(X^\top X+\lambda I)^{-1}X^\top$ into $\text{Var}(Ay)=\sigma^2 AA^\top$. As $\lambda$ grows, the $(X^\top X+\lambda I)^{-1}$ factors shrink every eigenvalue of the variance matrix — this is where the "buy variance reduction" side of the trade comes from, and critically, this reduction is **most dramatic exactly where OLS variance was largest**: the smallest eigenvalues of $X^\top X$ (the near-collinear directions) get proportionally the biggest variance reduction, because $\lambda$ matters most relative to a small eigenvalue.

$$\text{MSE}(\hat\beta_{\text{ridge}}) = \underbrace{\lVert\text{Bias}(\hat\beta_{\text{ridge}})\rVert^2}_{\text{monotonically} \uparrow \text{ in }\lambda} + \underbrace{\text{tr}\big(\text{Var}(\hat\beta_{\text{ridge}})\big)}_{\text{monotonically} \downarrow \text{ in }\lambda}$$

**The key theorem (Hoerl-Kennard, 1970):** there always exists a $\lambda^\ast > 0$ such that $\text{MSE}(\hat\beta_{\text{ridge}}(\lambda^\ast)) < \text{MSE}(\hat\beta_{\text{OLS}})$ — i.e., **OLS is never the minimum-MSE estimator whenever $X^\top X$ is ill-conditioned**, because the variance term's derivative at $\lambda=0$ is strictly negative while the bias term's derivative at $\lambda=0$ is exactly zero (bias is quadratic in $\lambda$ near the origin, variance reduction is linear) — so a small step away from $\lambda=0$ strictly reduces total MSE. This is the rigorous justification for regularizing collinear financial factor panels, not just an empirical heuristic.

**Feynman explanation — the bias-variance trade:** OLS sits at zero bias but can have huge variance when features are correlated (near-singular $X^\top X$ means tiny data perturbations swing $\hat\beta$ wildly — you're perfectly tuned to this exact sample, and badly mistuned to the next one). Ridge/Lasso/EN accept a small, mathematically quantified amount of bias to buy a much larger reduction in variance — like a mechanic detuning an engine slightly to make it more reliable across road conditions rather than perfectly tuned for one road. The Hoerl-Kennard result is the proof that this trade is *always* favorable near $\lambda=0$ when collinearity exists — it isn't a matter of taste.

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

### Worked Example — How Much Do t-Stats Actually Change?

Suppose OLS on a daily financing-spread regression yields $\hat\beta = 0.42$ with naive standard error $SE_{\text{naive}}=0.10$, giving $t=4.2$ — comfortably "significant" at any conventional threshold. If the residuals exhibit AR(1) autocorrelation with $\rho=0.6$ (common in daily financial data — today's regression error is meaningfully related to yesterday's), the **effective** standard error under a simplified single-lag correction inflates by approximately:

$$SE_{\text{HAC}} \approx SE_{\text{naive}}\sqrt{\frac{1+\rho}{1-\rho}} = 0.10\sqrt{\frac{1.6}{0.4}} = 0.10\sqrt{4} = 0.20$$

**Say it out loud:** **"With $\rho=0.6$, the naive standard error understates the truth by a factor of 2 — the corrected t-statistic falls to $0.42/0.20=2.1$, which may or may not still clear your significance threshold depending on how conservative you're being, but the story has changed from 'obviously significant' to 'borderline.' This exact arithmetic is why I never trust an OLS t-stat on daily or higher-frequency financial data without first checking the Durbin-Watson statistic or residual ACF plot — reporting the naive SE to a PM as if it were reliable is a way to get a false-positive signal shipped to production."**

### Durbin-Watson — The Quick Diagnostic Before Reaching for Newey-West

$$
DW = \frac{\sum_{t=2}^n(\hat\varepsilon_t-\hat\varepsilon_{t-1})^2}{\sum_{t=1}^n\hat\varepsilon_t^2}
$$

**Line-by-line:** expand the numerator: $\sum(\hat\varepsilon_t-\hat\varepsilon_{t-1})^2 = \sum\hat\varepsilon_t^2 + \sum\hat\varepsilon_{t-1}^2 - 2\sum\hat\varepsilon_t\hat\varepsilon_{t-1} \approx 2\sum\hat\varepsilon_t^2(1-\hat\rho)$ for large $n$, where $\hat\rho$ is the lag-1 sample autocorrelation of residuals. So $DW \approx 2(1-\hat\rho)$ — **say it out loud:** **" $DW \approx 2$ means no autocorrelation ($\hat\rho\approx0$); $DW$ near 0 means strong positive autocorrelation ($\hat\rho\to1$); $DW$ near 4 means strong negative autocorrelation ($\hat\rho\to-1$). It's a fast, single-number triage before deciding whether the full Newey-West machinery (or a GLS/Cochrane-Orcutt re-specification) is warranted. "**

### GLS as the Alternative — Regaining Efficiency, Not Just Correcting Inference

Newey-West fixes the **standard errors** but leaves $\hat\beta_{\text{OLS}}$ itself unchanged — it's still unbiased but no longer the *minimum-variance* unbiased estimator once $\Omega\neq\sigma^2I$ (Gauss-Markov's efficiency claim required exactly that). **Generalized Least Squares (GLS)** instead transforms the model to restore efficiency:

$$\hat\beta_{\text{GLS}} = (X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}y$$

**Line-by-line:** this is derived by whitening the data — premultiply both sides of $y=X\beta+\varepsilon$ by $\Omega^{-1/2}$ (a matrix square root of $\Omega^{-1}$) to get $\Omega^{-1/2}y = \Omega^{-1/2}X\beta + \Omega^{-1/2}\varepsilon$, where the transformed error $\Omega^{-1/2}\varepsilon$ now has covariance $\Omega^{-1/2}\Omega\Omega^{-1/2}=I$ — homoskedastic and uncorrelated. Applying plain OLS to this whitened system and transforming back gives exactly $\hat\beta_{\text{GLS}}$, which **is** BLUE for the true $\Omega$. **Feynman explanation:** Newey-West says "keep the same estimate, just be honest about how uncertain it is"; GLS says "actually use knowledge of the error structure to get a *better* estimate, weighting each observation inversely by how noisy it was." The practical catch: GLS requires knowing (or well-estimating) $\Omega$, which is itself a $n\times n$ object — often infeasible without a parametric structure (e.g., assume errors follow a GARCH or AR(1) process specifically, then GLS becomes feasible/"FGLS"). This is precisely why, in practice, Newey-West's "correct the inference, don't touch the point estimate" is the more commonly deployed pragmatic choice for financial factor regressions.

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

### Worked Numerical Example — Computing Information Gain by Hand

Suppose a node contains 10 observations of a binary "financing-flow-is-adverse-selection" label: 6 labeled "benign" (class 0), 4 labeled "adverse" (class 1). Parent entropy:

$$H(\text{parent}) = -\left(\frac{6}{10}\log_2\frac{6}{10} + \frac{4}{10}\log_2\frac{4}{10}\right) = -(0.6\times(-0.737) + 0.4\times(-1.322)) = 0.442+0.529 = 0.971 \text{ bits}$$

Now test a candidate split on "order size $>$ 5x average" that separates the 10 into a left child (7 obs: 6 benign, 1 adverse) and right child (3 obs: 0 benign, 3 adverse):

$$H(\text{left}) = -\left(\frac{6}{7}\log_2\frac{6}{7}+\frac{1}{7}\log_2\frac{1}{7}\right) = -(0.857\times(-0.222)+0.143\times(-2.807)) = 0.190+0.401=0.591\text{ bits}$$

$$H(\text{right}) = -\left(\frac{0}{3}\log_2\frac{0}{3}+\frac{3}{3}\log_2\frac{3}{3}\right) = 0 \text{ bits (pure node — all adverse)}$$

$$IG = 0.971 - \left(\frac{7}{10}(0.591) + \frac{3}{10}(0)\right) = 0.971 - 0.414 = 0.557 \text{ bits}$$

**Say it out loud:** *"This split is highly informative — it isolates a pure 'adverse' subgroup (all 3 large orders in this sample were adverse selection) while leaving the left child still mostly benign but with residual impurity (1 adverse case remains, meaning order size alone isn't a complete explanation) — the tree would continue splitting the left child on a further feature, e.g., time-of-day or venue, to try to isolate that remaining adverse case."* This is exactly the kind of by-hand sanity check I'd walk through at a whiteboard rather than just citing the formula.

### Pruning — Why Greedy Growth Alone Overfits

A fully grown tree (splitting until every leaf is pure) essentially memorizes the training set — for the adverse-selection example, it will eventually create a leaf containing exactly the single anomalous benign order that happened to be large, rather than treating it as noise. **Cost-complexity pruning** formalizes the trade-off:

$$R_\alpha(T) = R(T) + \alpha\,|T|$$

**Line-by-line:** $R(T)$ is the tree's total misclassification (or squared-error) cost on training data, which strictly decreases as the tree grows larger (more leaves $|T|$ always fits training data at least as well); $\alpha|T|$ is an explicit complexity penalty — larger $\alpha$ prefers smaller, simpler trees. As $\alpha$ increases from 0, subtrees get pruned back one collapsed-split-at-a-time (via **weakest-link pruning**, which at each step removes whichever internal node has the smallest increase in $R(T)$ per leaf removed) — producing a nested sequence of trees $T_0 \supset T_1 \supset \cdots \supset T_k = \{\text{root}\}$. The optimal $\alpha$ (and hence tree size) is chosen by cross-validation, **not** by an in-sample criterion — since $R(T)$ always favors the largest tree in-sample by construction.

**Feynman explanation:** think of a fully grown tree as a student who memorized every practice exam question verbatim rather than learning the underlying method — perfect on the practice exam (training data), terrible on the real exam (test data) because a slightly rephrased question breaks the memorization. Pruning is deliberately making the student "forget" some memorized specifics, forcing them to rely on more general, more robust rules — accepting slightly worse practice-exam performance for much better real-exam performance, exactly the bias-variance trade from Q10 applied to a completely different model family.

### Handling Continuous vs. Categorical Splits — A Practical Nuance Often Skipped

For continuous features (the example above), candidate thresholds are the midpoints between adjacent sorted values — an $O(n\log n)$ sort plus an $O(n)$ scan per feature per node. For **categorical** features with $k$ levels, the naive approach considers all $2^{k-1}-1$ possible binary partitions — computationally infeasible for high-cardinality categoricals (e.g., counterparty ID with hundreds of levels). CART's practical trick for binary classification: **sort the categories by their within-category positive-class rate**, then only consider the $k-1$ splits along that sorted order — provably optimal for Gini/entropy in the binary case, reducing the search from exponential to linear. This is exactly the kind of implementation detail that separates "I've read about decision trees" from "I've actually had to make one scale to production categorical features like counterparty or venue ID."

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
──────────────────   ────────────────────────────────     ──────────────────────────────────
Ensembling logic     Bagging: parallel trees on           Boosting: sequential trees, each
                     bootstrap samples + feature          fitting the RESIDUAL/gradient of
                     subsampling, averaged                the previous ensemble
					 
Bias/Variance        Reduces VARIANCE (trees are          Reduces BIAS (each stage
                     decorrelated, roughly unbiased       actively corrects prior errors);
                     individually)                        variance controlled via
                                                          shrinkage/regularization
														  
Overfit risk         Lower — averaging is inherently      Higher — needs careful
                     robust                               learning-rate/depth/early-stop

Parallelizable       Fully (trees independent)            Sequential in boosting rounds
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

### Worked Numerical Example — Computing an XGBoost Leaf Weight and Gain

Take a squared-error objective ($\ell(y,\hat y)=\tfrac12(y-\hat y)^2$), so $g_i = \hat y_i - y_i$ and $h_i=1$ for every observation (the Hessian of squared error is constant — this is *why* squared-error boosting is the simplest special case to hand-derive). Suppose a candidate leaf split produces:

```
LEFT CHILD : 3 observations, residuals g_i = [-0.8, -0.5, -0.3]  → G_L = -1.6, H_L = 3
RIGHT CHILD: 2 observations, residuals g_i = [0.9, 1.1]          → G_R =  2.0, H_R = 2
```

With $\lambda=1.0$ (L2 regularization) and $\gamma=0.1$ (min-gain complexity penalty):

$$\text{Gain} = \frac12\left[\frac{(-1.6)^2}{3+1} + \frac{(2.0)^2}{2+1} - \frac{(-1.6+2.0)^2}{3+2+1}\right] - 0.1$$

$$= \frac12\left[\frac{2.56}{4} + \frac{4.0}{3} - \frac{0.16}{6}\right] - 0.1 = \frac12[0.64+1.333-0.0267]-0.1 = \frac12(1.947)-0.1 = 0.973-0.1=0.873$$

**Say it out loud:** **"A gain of 0.873, well above zero, means this split is worth making — the loss reduction from separating the negative-residual observations (left, currently over-predicted) from the positive-residual observations (right, currently under-predicted) comfortably exceeds the $\gamma=0.1$ cost of adding a new leaf. I'd contrast this against a split that produces a gain of, say, 0.05 — below a typical $\gamma$, meaning the split isn't worth the added model complexity and the algorithm correctly prunes it (this is what 'min_split_gain' controls in a real XGBoost config)."**

The corresponding optimal leaf weights: $w_L^\ast = -G_L/(H_L+\lambda) = -(-1.6)/4 = 0.4$, $w_R^\ast = -G_R/(H_R+\lambda) = -2.0/3 = -0.667$. **Feynman explanation:** the left leaf, which was over-predicting (negative residuals, i.e., $\hat y > y$), gets assigned a **positive** correction $+0.4$ to be added to the ensemble's running prediction — pushing predictions down toward the true value. The right leaf, under-predicting, gets a **negative** correction $-0.667$ — wait, that seems backwards; let's double check the sign convention: $g_i=\hat y_i - y_i$, so negative $g_i$ means $\hat y_i < y_i$ (under-prediction), and $w^\ast=-G/(H+\lambda)$ being positive when $G$ is negative means the correction is positive, pushing predictions **up** — consistent with correcting an under-prediction. This sign-tracking discipline (which direction does the residual point, which direction should the correction push) is exactly what an interviewer will probe if you rattle off the formula without being able to verify it makes directional sense.

### Regularization Comparison — Why XGBoost's $\lambda$ and $\gamma$ Matter More Than People Realize

```
HYPERPARAMETER    ROLE                                     ANALOGY TO Q10 (Ridge/Lasso)
───────────────   ──────────────────────────────────────   ─────────────────────────────
lambda (L2)       Shrinks leaf weights toward 0,           Directly analogous to Ridge's
                  same role as ridge's leaf-shrinkage      lambda*||beta||^2 penalty —
                  in the denominator H_j + lambda          both damp the "confidence" of
                                                           an estimate proportional to
                                                           how little data supports it
														   
gamma             Minimum gain required to make ANY        No direct linear-regression
                  split — a discrete, structural           analogue; closest is a
                  pruning threshold, unlike lambda's       significance threshold on
                  smooth shrinkage                         an F-test for adding a term

learning_rate     Shrinks EACH TREE's total contribution   Analogous to a small step
(eta)             to the ensemble (separate from lambda)   size in gradient descent —
                                                           trades convergence speed for
                                                           final-solution stability

max_depth /       Bounds tree complexity structurally,     Analogous to a hard
min_child_weight  independent of the loss-based gain       cardinality constraint on
                  criterion                                which/how many regressors
                                                           can enter (best-subset)
```

**Feynman explanation, tying it together:** every one of these hyperparameters is a different lever for the same underlying bias-variance trade from Q10 — they just act on a *tree ensemble's* degrees of freedom (leaf count, leaf confidence, per-tree contribution weight) instead of a *linear model's* coefficient magnitudes. An interviewer who has just heard your Q10 answer will often immediately ask "so how does this generalize to trees?" — this table is the bridge answer.

### Why Random Forest's Variance Reduction Requires Decorrelation, Not Just Averaging

A subtlety often glossed over: averaging $B$ identically-distributed but **correlated** trees with pairwise correlation $\rho$ and individual variance $\sigma^2$ gives:

$$\text{Var}\left(\frac1B\sum_{b=1}^B T_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**Line-by-line:** as $B\to\infty$, the second term vanishes, but the **first term does not** — it's a floor set entirely by how correlated the individual trees are. This is *exactly why* Random Forest doesn't just bootstrap-sample rows (bagging alone) but **also randomly subsamples features at each split** ($\sqrt{p}$ features considered, typically) — feature subsampling is what actively drives $\rho$ down, which is the only lever that keeps paying off as you add more trees. Bagging alone (bootstrap rows only, as in a "Bagged Trees" ensemble) leaves $\rho$ relatively high because every tree still has access to the same dominant feature and will tend to split on it near the root, correlating the trees' overall structure.

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

### Worked Numerical Example — Shapley Values for a 3-Feature Toy Model

Let $v(S)$ denote the model's expected output using only feature subset $S$ (others marginalized to their average effect), for a simple linear model $f(x)=2x_1 - x_2 + 0.5x_3$ evaluated at $x=(1,2,3)$ against a baseline (all-zero) reference. Since the model is linear and features are treated as independent, $v(S) = \sum_{i\in S} w_i x_i$ exactly, with $w=(2,-1,0.5)$.

For feature 1, enumerate all subsets $S\subseteq\{2,3\}$ (i.e., $S$ ranges over $\emptyset, \{2\}, \{3\}, \{2,3\}$) and compute the marginal contribution of adding feature 1:

```
S            |S|   v(S)                       v(S U {1})               marginal contrib.  weight = |S|!(n-|S|-1)!/n!
{}           0     0                          2(1) = 2                 2                  0!2!/3! = 1/3
{2}          1     -1(2) = -2                 2(1)-1(2) = 0            2                  1!1!/3! = 1/6
{3}          1     0.5(3) = 1.5               2(1)+0.5(3) = 3.5        2                  1!1!/3! = 1/6
{2,3}        2     -1(2)+0.5(3) = -0.5        2(1)-1(2)+0.5(3) = 1.5   2                  2!0!/3! = 1/3
```

**Line-by-line weight check:** $n=3$ features total; for $|S|=0$: weight $=0!\cdot(3-0-1)!/3! = 1\cdot2/6=1/3$; for $|S|=1$: weight $=1!\cdot(3-1-1)!/3!=1\cdot1/6=1/6$; for $|S|=2$: weight $=2!\cdot(3-2-1)!/3!=2\cdot1/6=1/3$ — and indeed these four weights $(1/3,1/6,1/6,1/3)$ sum to exactly 1, confirming they form a valid probability distribution over "orderings in which feature 1 could arrive."

$$\phi_1 = \frac13(2) + \frac16(2) + \frac16(2) + \frac13(2) = 2\left(\frac13+\frac16+\frac16+\frac13\right) = 2(1) = 2$$

**Say it out loud:** **"The marginal contribution of feature 1 happens to be exactly 2 under every possible coalition it joins — unsurprising, because the model is linear and features are treated as independent, so $\phi_i = w_i x_i$ exactly recovers the linear model's own coefficient-times-value attribution. This is a useful sanity check to know cold: for a linear model with independent features, Shapley values degenerate to the obvious 'coefficient times feature value' attribution — the entire machinery is only earning its keep when the model has interactions or nonlinearities that a single coefficient can't capture."** Repeating the same enumeration for features 2 and 3 gives $\phi_2=-2$ and $\phi_3=1.5$, and indeed $\phi_1+\phi_2+\phi_3 = 2-2+1.5=1.5=f(x)-f(0)$ — the **efficiency axiom**, guaranteeing Shapley values exactly decompose the prediction (minus baseline) with no leftover unexplained term.

### TreeSHAP — Why Exact Computation Is Tractable for Trees Specifically

Computing Shapley values exactly requires evaluating $v(S)$ for all $2^{|F|}$ subsets — intractable in general. **TreeSHAP**'s key insight: for a tree model, $v(S)$ (the expected prediction when only features in $S$ are "known") can be computed by pushing weighted probability mass through the tree structure itself — every internal split node either (a) is on a known feature in $S$, in which case you deterministically follow the branch matching $x$, or (b) is on an unknown feature, in which case you split the probability mass proportionally across both branches according to the training-data fraction that went each way. This recursive structure lets TreeSHAP compute **exact** Shapley values in $O(TLD^2)$ time ($T$=number of trees, $L$=max leaves per tree, $D$=max tree depth) rather than exponential time — a genuinely important algorithmic result (Lundberg et al., 2018) worth naming by name if asked "how does SHAP actually scale to production."

### SHAP vs. Gain Importance — A Concrete Case Where They Disagree

Consider a feature that is used in exactly one, very early, high-impact split near the root of every tree (say, a regime-indicator feature) versus a feature used in many splits deep in the tree, each individually low-impact. **Gain importance** naively sums loss-reduction across all splits, and can be dominated by the many-small-splits feature purely due to split *count*. **SHAP**, by construction, correctly attributes credit based on each feature's actual marginal contribution to *individual predictions*, and will correctly identify the regime-indicator feature as far more important **for the specific observations where the regime differs from the norm**, even though it appears in fewer total splits. **Say it out loud:** *"I've seen production feature-importance rankings flip between Gain and SHAP specifically for exactly this reason — a coarse regime/session-time indicator ranked low by Gain importance but was the dominant SHAP driver for adverse-selection flagging during the specific hours it mattered, which is the more decision-relevant answer for a trading desk."*

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

**Say it out loud:** **"If $\gamma = 0$, then $y_{t-1}$ has no explanatory power for the change $\Delta y_t$ beyond the lagged differences — meaning today's level doesn't pull tomorrow's change back toward anything, which is the definition of a random walk / unit root. If $\gamma < 0$ and statistically significant, the series exhibits mean reversion: a high $y_{t-1}$ predicts a negative $\Delta y_t$, pulling the series back down."** The ADF test statistic is $t_\gamma = \hat\gamma / SE(\hat\gamma)$, compared against **non-standard** critical values (not the usual t-distribution, because under $H_0$ the process is a random walk and the asymptotic distribution is the Dickey-Fuller distribution).

**ARIMA(p,d,q)**: difference $d$ times to induce stationarity, then model as AR(p) + MA(q):

$$\left(1-\sum_{i=1}^p \phi_i L^i\right)(1-L)^d y_t = \left(1+\sum_{j=1}^q \theta_j L^j\right)\varepsilon_t$$

### Worked Numerical Example — Interpreting an ADF Regression Output

Suppose fitting the ADF regression on a repo-GC spread level series (no trend term, for simplicity) yields:

$$\Delta y_t = 0.02 - 0.15\,y_{t-1} + 0.3\,\Delta y_{t-1} + \hat\varepsilon_t, \qquad SE(\hat\gamma) = 0.05$$

$$t_\gamma = \frac{-0.15}{0.05} = -3.0$$

Compare against Dickey-Fuller critical values (approximate, no trend, large sample): $-3.43$ (1%), $-2.86$ (5%), $-2.57$ (10%). **Say it out loud:** **"A test statistic of $-3.0$ is more negative than the 5% critical value of $-2.86$ but not quite as extreme as the 1% critical value of $-3.43$ — so I'd reject the unit-root null at the 5% level but not the 1% level, meaning there's reasonably strong, but not overwhelming, evidence the spread mean-reverts rather than following a pure random walk. The implied half-life of a shock is $\ln(0.5)/\ln(1-0.15) = \ln(0.5)/\ln(0.85) \approx 4.3$ days — actionable information for how long to expect a financing-spread dislocation to persist before mean-reversion trading becomes attractive."**

**Why the critical values are non-standard — the intuition, not just the assertion:** under the null $\gamma=0$, the regressor $y_{t-1}$ is itself a random walk (non-stationary), so the usual justification for a t-distribution (which relies on the regressor behaving like a fixed, well-behaved quantity as $n\to\infty$) breaks down — the sample variance of a random walk grows with $t$, so the asymptotic distribution of $t_\gamma$ under $H_0$ is not Normal/t but the (skewed, non-standard) Dickey-Fuller distribution, tabulated via simulation rather than derived in closed form. This is worth being able to say explicitly if asked "why can't I just use a normal t-test here" — a very natural, very common follow-up.

### Order Selection — How Many Lags $p$ to Include (the Augmented Part)

$$\text{AIC}(p) = \ln(\hat\sigma^2_p) + \frac{2(p+k)}{n}, \qquad \text{BIC}(p) = \ln(\hat\sigma^2_p) + \frac{(p+k)\ln n}{n}$$

**Line-by-line:** both criteria trade off in-sample fit ($\ln\hat\sigma^2_p$, which mechanically decreases as you add more lags) against a complexity penalty that grows with the number of parameters $p+k$; BIC's penalty grows with $\ln n$ rather than a flat $2$, so BIC penalizes additional lags more heavily as sample size grows, and asymptotically selects the **true** model order with probability 1 (BIC is "consistent"), while AIC asymptotically tends to slightly overfit (select too many lags) even with infinite data, because it optimizes for **predictive accuracy**, not exact recovery of a "true" finite-order model. **Say it out loud:** *"For a financing-desk model where I genuinely believe the DGP has some finite true order — e.g., a repo spread that mean-reverts with a clean weekly settlement-cycle structure — I lean BIC. For a model where I care purely about forecast accuracy and don't believe there's a single 'true' order (most market microstructure signals), I lean AIC, accepting its slight tendency to over-parameterize in exchange for better held-out predictive performance."*

### Seasonal ARIMA — Why Financing Curves Often Need the "S" in SARIMA

Repo and financing markets exhibit strong **calendar-driven seasonality** — month-end and quarter-end balance-sheet constraints reliably widen GC repo spreads on specific, predictable days (window dressing). A plain $ARIMA(p,d,q)$ cannot capture this without an enormous lag order; $SARIMA(p,d,q) \times (P,D,Q)_s$ adds a **second, seasonal** difference/AR/MA structure at lag $s$ (e.g., $s=21$ trading days for a monthly cycle):

$$\Phi(L^s)(1-L^s)^D\phi(L)(1-L)^d y_t = \Theta(L^s)\theta(L)\varepsilon_t$$

**Feynman explanation:** think of the seasonal terms as a **second, separate clock** ticking alongside the regular one — the regular ARIMA terms capture "what happened yesterday and the day before," while the seasonal terms capture "what happened at this exact point in the last cycle" (e.g., last month-end). A financing-desk forecasting model that ignores this and only uses plain ARIMA will systematically under-forecast spread widening every month-end — a predictable, exploitable, and embarrassing miss to explain to a PM if it shows up as a live forecast error.

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

$$\bar\sigma^2 = \omega + (\alpha+\beta)\bar\sigma^2 \;\;\Rightarrow\;\; \bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$$

which requires $\alpha+\beta<1$ for stationarity — **the persistence parameter** $\alpha+\beta$ determines the half-life of a volatility shock: $\text{half-life} = \frac{\ln(0.5)}{\ln(\alpha+\beta)}$.

**EGARCH** fixes GARCH's blind spot — it can't distinguish a positive shock from a negative one of the same size, but equities famously exhibit a **leverage effect** (down moves raise vol more than up moves of equal size):

$$\ln(\sigma_t^2) = \omega + \beta\ln(\sigma_{t-1}^2) + \alpha\left(\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}} - \mathbb{E}\left[\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}}\right]\right) + \gamma\frac{\varepsilon_{t-1}}{\sigma_{t-1}}$$

### Worked Numerical Example — Multi-Step-Ahead GARCH Volatility Forecast

Suppose a fitted GARCH(1,1) has $\omega=0.02$, $\alpha=0.10$, $\beta=0.85$ (persistence $\alpha+\beta=0.95$, long-run variance $\bar\sigma^2 = 0.02/(1-0.95)=0.40$), and today's conditional variance is $\sigma_t^2=0.60$ (currently elevated above the long-run level — a stressed day). The $h$-step-ahead forecast, derived by repeatedly substituting the recursion forward and taking conditional expectation ( $\mathbb{E}_t[\varepsilon_{t+j}^2]=\sigma_{t+j}^2$ for the realized future variance ):

$$\mathbb{E}_t[\sigma_{t+h}^2] = \bar\sigma^2 + (\alpha+\beta)^{h-1}\big(\sigma_{t+1}^2-\bar\sigma^2\big)$$

**Line-by-line derivation:** start from $\sigma_{t+1}^2 = \omega+\alpha\varepsilon_t^2+\beta\sigma_t^2$ (known at time $t$). For $h=2$ : $\mathbb{E}_t[\sigma_{t+2}^2] = \omega + (\alpha+\beta)\sigma_{t+1}^2 = \bar\sigma^2(1-\alpha-\beta) + (\alpha+\beta)\sigma_{t+1}^2$ — rearranging, $=\bar\sigma^2 + (\alpha+\beta)(\sigma_{t+1}^2-\bar\sigma^2)$ , confirming the formula at $h=2$ directly, and induction on $h$ extends it forward using the same substitution at each step (each additional step multiplies the "distance from long-run variance" by another factor of the persistence $\alpha+\beta$).

Plugging in numbers: $\sigma_{t+1}^2 = 0.02+0.10\varepsilon_t^2+0.85(0.60)$; assume $\varepsilon_t^2\approx\sigma_t^2=0.60$ (typical realized shock near its conditional expectation) giving $\sigma_{t+1}^2\approx0.02+0.06+0.51=0.59$. Then for $h=10$ days ahead:

$$\mathbb{E}_t[\sigma_{t+10}^2] = 0.40 + (0.95)^{9}(0.59-0.40) = 0.40 + 0.630\times0.19 = 0.40+0.120=0.52$$

**Say it out loud:** **"Ten trading days out, the forecast has decayed from today's elevated 0.59 only about 40% of the way back to the long-run level of 0.40 — the high persistence ($\alpha+\beta=0.95$) means the current stress is expected to linger for weeks, not days. This is directly actionable: if I'm pricing a 2-week financing tenor today during a volatility spike, GARCH tells me not to naively use today's spot volatility for the full 2 weeks — the average expected variance over that window is somewhere between today's 0.59 and the eventual 0.40, weighted toward the near-term elevated level."**

### Model Comparison — GARCH Family Variants and When Each Is Used

```
MODEL           KEY FEATURE                            WHEN TO PREFER
─────────────   ────────────────────────────────────   ─────────────────────────────────
GARCH(1,1)      Symmetric response to +/- shocks       Baseline; FX vol (less pronounced
                                                       leverage effect than equities)

EGARCH          Asymmetric (leverage effect), log-     Equity vol, where down-moves raise
                variance guarantees positivity         vol more than up-moves

GJR-GARCH       Asymmetric via an indicator term       Alternative to EGARCH; easier to
                (adds gamma*eps^2*I[eps<0] to plain    constrain for stationarity, less
                GARCH) rather than log-variance        elegant positivity guarantee

HAR-RV          Regresses realized vol on lagged       When high-frequency (intraday)
                daily/weekly/monthly realized vol      data is available — captures
                averages — a simple linear cascade     long-memory volatility clustering
                across time HORIZONS, not just lags    cheaply without a GARCH's MLE fit

GARCH-MIDAS     Mixes high- and low-frequency data     Combining daily return-based vol
                (daily returns + monthly macro data)   with slower-moving macro/regime
                                                       variables (e.g., financing stress
                                                       indices) in a single model
```

**Feynman explanation of HAR-RV specifically (since it appears on Shaikat's resume and is a natural cross-reference):** HAR-RV sidesteps GARCH's MLE fitting entirely by simply regressing today's realized variance on **three lagged averages** — yesterday's, the past week's average, and the past month's average:

$$RV_t = \beta_0 + \beta_D RV_{t-1} + \beta_W \overline{RV}_{t-5:t-1} + \beta_M \overline{RV}_{t-22:t-1} + \varepsilon_t$$

*Say it out loud:* "This is just OLS (Q9) — no iterative optimization needed — yet it captures 'long memory' in volatility (today's vol depends meaningfully on vol from a month ago, not just yesterday) purely through the multi-horizon averaging, which is a remarkably effective, cheap-to-implement, and easy-to-explain-to-a-PM alternative to fitting a full GARCH when high-frequency intraday data is available to construct the realized-variance target in the first place."

### Model Diagnostics — How You'd Know GARCH(1,1) Is Mis-Specified

After fitting, compute **standardized residuals** $z_t = \varepsilon_t/\sigma_t$ and check: (1) $z_t$ should show **no remaining autocorrelation in $z_t^2$** (Ljung-Box test on squared standardized residuals) — if it does, the variance dynamics aren't fully captured and a higher-order GARCH or additional asymmetry term (EGARCH) is needed; (2) $z_t$'s unconditional distribution is typically **fat-tailed even after standardization** — motivating a Student-t or skewed-t innovation distribution instead of the Gaussian assumed in the vanilla likelihood, materially changing tail-risk (VaR) estimates even when the point volatility forecast looks similar.

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

**Say it out loud:** **"$\delta_t(j)$ is the probability of the single best path that ends in state $j$ at time $t$ — we build it up recursively: to be in state $j$ optimally at time $t$, you must have arrived from whichever prior state $i$ maximized $\delta_{t-1}(i)A_{ij}$, then multiply by how well state $j$ explains the current observation. The backpointer $\psi_t(j)$ remembers which $i$ won, so a final backward pass from $\arg\max_j \delta_T(j)$ reconstructs the whole optimal regime path."**

**Baum-Welch (EM)** alternates: E-step computes $\gamma_t(j)=P(s_t=j\mid X,\theta)$ and $\xi_t(i,j)=P(s_t=i,s_{t+1}=j\mid X,\theta)$ via forward-backward; M-step re-estimates:

$$\hat A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad \hat\mu_j = \frac{\sum_t \gamma_t(j) x_t}{\sum_t \gamma_t(j)}$$

— i.e., the new transition probability is "expected number of $i\to j$ transitions" over "expected time spent in $i$," and the new emission mean is a $\gamma$-weighted average of observations, exactly analogous to soft-assignment k-means.

### Worked Numerical Example — One Step of the Forward Algorithm by Hand

Take a 2-state model (state 1 = "calm," state 2 = "stressed"), with $\pi=(0.9, 0.1)$, transition matrix $A=\begin{pmatrix}0.95&0.05\\0.10&0.90\end{pmatrix}$, and Gaussian emissions $b_1(x)\sim N(0,0.01)$, $b_2(x)\sim N(-0.005,0.04)$. Observe $x_1 = -0.02$.

$$b_1(-0.02) = \frac{1}{\sqrt{2\pi(0.01)}}\exp\left(-\frac{(-0.02)^2}{2(0.01)}\right) = 3.99\exp(-0.02)=3.91$$
$$b_2(-0.02) = \frac{1}{\sqrt{2\pi(0.04)}}\exp\left(-\frac{(-0.02+0.005)^2}{2(0.04)}\right) = 1.995\exp(-0.0028)=1.99$$

$$\alpha_1(1) = \pi_1 b_1(x_1) = 0.9\times3.91=3.52, \qquad \alpha_1(2)=\pi_2 b_2(x_1)=0.1\times1.99=0.199$$

Now observe $x_2=-0.03$ (a bigger negative move), with $b_1(-0.03)=3.99\exp(-0.045)=3.82$, $b_2(-0.03)=1.995\exp(-0.00078)=1.99$:

$$\alpha_2(1) = b_1(x_2)\big[\alpha_1(1)A_{11}+\alpha_1(2)A_{21}\big] = 3.82\times[3.52(0.95)+0.199(0.10)] = 3.82\times[3.344+0.0199]=3.82\times3.364=12.85$$

$$\alpha_2(2) = b_2(x_2)\big[\alpha_1(1)A_{12}+\alpha_1(2)A_{22}\big] = 1.99\times[3.52(0.05)+0.199(0.90)]=1.99\times[0.176+0.179]=1.99\times0.355=0.706$$

Normalizing (dividing by $\alpha_2(1)+\alpha_2(2)=13.56$): $P(s_2=\text{calm}\mid x_{1:2}) \approx 0.948$, $P(s_2=\text{stressed}\mid x_{1:2})\approx0.052$.

**Say it out loud:** **"Even after two consecutive negative moves, the model still assigns 95% probability to the 'calm' regime — because state 1's transition matrix is strongly self-persistent ($A_{11}=0.95$) and its emission variance is tight enough that these particular moves aren't yet extreme relative to what 'calm' can plausibly produce. This is the correct, conservative behavior: an HMM shouldn't flip regimes on the first couple of noisy observations — it should require either a genuinely extreme move or a sustained run of moderately unusual moves before the posterior meaningfully shifts, which is exactly the recursive weighting the forward algorithm performs."**

### Choosing the Number of Hidden States — A Model-Selection Question Interviewers Ask

**Feynman explanation:** more states always fit the training data at least as well (same overfitting logic as tree depth in Q12 or ARIMA lag order in Q15) — so the number of states $K$ cannot be chosen by in-sample likelihood alone. In practice: (1) use BIC (penalizing the added transition/emission parameters that scale roughly as $O(K^2)$) to compare candidate $K$ on held-out likelihood; (2) impose a **domain-driven prior** — for financing-regime detection, 2–3 states (calm / stressed / crisis) is usually both economically interpretable and empirically sufficient; a 6-state HMM might fit marginally better in-sample but produces states that don't correspond to anything a PM can act on, which defeats the purpose of using an interpretable regime model in the first place. **Say it out loud:** **"I'd resist the temptation to let BIC alone pick $K=5$ if the resulting states don't map onto an economically distinct, actionable interpretation — an HMM's value to the desk is as much about interpretability as pure predictive fit, and a model a PM can't explain to risk/compliance in one sentence is a model that won't survive model-governance review regardless of its likelihood score."**

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
────────────────────────────────────          ─────────────────────────────────────────
[ train ][ test ][ train ][ test ]            [ train ][ purge ][ test ][ embargo ][ train ]
Random fold assignment leaks                  Purge: drop training obs whose LABEL
future info into past training                window overlaps the test window
folds via shuffling                           Embargo: drop N obs immediately after
                                              test fold from the next training fold
                                              (residual serial correlation)
```

**CPCV construction:** partition the timeline into $N$ groups; form $\binom{N}{k}$ combinations choosing $k$ groups as test, evaluating each combination as an independent backtest path — this produces **multiple, non-overlapping "paths"** through the data, letting you build a full distribution of out-of-sample Sharpe ratios rather than a single point estimate, from which you can compute the **Probability of Backtest Overfitting (PBO)**:

$$\text{PBO} = P\big(\text{rank of IS-best config in OOS} > \text{median}\big)$$

### Worked Example — Counting CPCV Paths and Computing PBO

Suppose the timeline is split into $N=6$ groups, and CPCV chooses $k=2$ groups as the test set each time. The number of combinations is $\binom{6}{2}=15$ — fifteen distinct train/test splits, each giving an independent out-of-sample Sharpe estimate for a given model configuration. Suppose across these 15 paths, a candidate signal configuration's OOS Sharpe ranks are (out of, say, 10 competing configurations tested per path): median rank in 9 of the 15 paths, top-quartile rank in only 3 of 15, bottom-quartile in 3 of 15.

$$\widehat{\text{PBO}} = \frac{\#\{\text{paths where in-sample-best config ranks below median OOS}\}}{\text{total paths}}$$

If the specific configuration that was picked because it looked in-sample-best ranks **below median** in 9 of the 15 combinatorial OOS paths:

$$\widehat{\text{PBO}} = \frac{9}{15} = 0.60$$

**Say it out loud:** **"A PBO of 0.60 means that 60% of the time, across every reasonable way of slicing the historical data into train/test, the configuration that looked best in-sample turns out to be no better than a coin flip (or worse) out-of-sample — this is damning evidence the in-sample selection process is picking up noise specific to the particular historical path, not genuine, generalizable skill. A PBO below roughly 0.20-0.30 is the kind of evidence I'd want before defending a signal's Sharpe to a PM; a PBO near 0.5 (or worse, above it) means the signal-selection process itself is broken, independent of whether any individual configuration happens to look good."**

### Deflated Sharpe Ratio — The Complementary Check to PBO

PBO tells you *whether* your selection process is contaminated by overfitting; the **Deflated Sharpe Ratio (DSR)** tells you, for a *specific* Sharpe ratio you're about to report, how much of it survives after accounting for how many configurations you tried:

$$\text{DSR} = \Phi\left(\frac{(\widehat{SR}-SR_0)\sqrt{n-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right)$$

where $SR_0$ is the **expected maximum Sharpe ratio under the null of no skill**, given the number of independent trials $M$ attempted:

$$SR_0 \approx \sqrt{\text{Var}[\widehat{SR}]}\left[(1-\gamma)\Phi^{-1}\left(1-\frac1M\right) + \gamma\Phi^{-1}\left(1-\frac{1}{Me}\right)\right]$$

**Feynman explanation:** if you tried 200 different signal configurations and report only the single best one's backtested Sharpe of 1.5, that 1.5 is **not** directly comparable to a Sharpe of 1.5 you got from testing just one configuration — with 200 independent attempts, pure noise alone would be expected to produce a "best of 200" Sharpe substantially above zero (this is the multiple-comparisons/multiple-testing problem, exactly analogous to why you need a Bonferroni correction when running 200 statistical hypothesis tests). $SR_0$ quantifies exactly how much of a "best of $M$" Sharpe you'd expect from pure noise, and DSR tells you what fraction of your *reported* Sharpe genuinely survives after subtracting that expected noise floor. **Say it out loud:** *"At Millburn, with a research library of 30+ signals, I would never report a signal's raw backtested Sharpe to a PM without first deflating it by the number of configurations actually tried during development — reporting an un-deflated Sharpe after an extensive search process is, in a very concrete quantitative sense, misleading, even if every individual number in the report is 'true.'"*

### Concrete Splitter Debugging — A Common Implementation Bug to Volunteer

**Feynman explanation of a subtle bug:** a common mistake is purging/embargoing based on **row index** rather than **actual timestamp**, which silently breaks if the data has any gaps (e.g., missing observations on a holiday, or an intraday dataset with a variable number of ticks per day). The purge/embargo logic must operate on the **timestamp delta**, not the row-count delta — "purge training rows within 10 calendar days of the test set" is correct; "purge training rows within 10 array-index positions of the test set" silently under- or over-purges whenever the data isn't perfectly evenly spaced. This is exactly the kind of production bug that passes code review (the logic "looks" right) but produces a subtly leaky or overly conservative validation split — worth explicitly naming as a lesson learned if asked about production ML pitfalls.

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

### Worked Numerical Example — One Kalman Update Step by Hand

Suppose $Q=0.001$ (slow-moving hedge ratio), $R=0.09$ (moderate observation noise), current filtered state $\hat\beta_{t-1|t-1}=0.50$, $P_{t-1|t-1}=0.02$. Observe $x_t=1.2$, $y_t=0.70$.

**Predict:** $\hat\beta_{t|t-1}=0.50$ (random walk — no drift term), $P_{t|t-1}=0.02+0.001=0.021$.

**Innovation:** $y_t - x_t\hat\beta_{t|t-1} = 0.70 - 1.2(0.50) = 0.70-0.60=0.10$.

**Kalman gain:** $K_t = \dfrac{P_{t|t-1}\,x_t}{x_t^2P_{t|t-1}+R} = \dfrac{0.021(1.2)}{(1.2)^2(0.021)+0.09} = \dfrac{0.0252}{0.0302+0.09}=\dfrac{0.0252}{0.1202}=0.2096$

**Update:** $\hat\beta_{t|t} = 0.50 + 0.2096(0.10) = 0.50+0.021=0.521$

$$P_{t|t} = (1-K_tx_t)P_{t|t-1} = (1-0.2096\times1.2)(0.021) = (1-0.2515)(0.021)=0.7485\times0.021=0.01572$$

**Say it out loud:** **"The hedge ratio moved from 0.500 to 0.521 — a modest update, appropriately damped, because the Kalman gain of ~0.21 says 'trust this single new observation about 21% as much as your existing belief.' Notice the updated variance $P_{t|t}=0.0157$ is smaller than the prior $P_{t|t-1}=0.021$ — observing new data always (weakly) reduces uncertainty about the current state, exactly as Bayesian updating requires, and this shrinkage factor $(1-K_tx_t)$ is the direct multi-step-ahead analogue of the posterior-precision argument in Q29's Bayesian regression."**

### Relating the Kalman Filter Explicitly to Bayesian Updating (Q29) and to Ridge (Q10)

**Feynman explanation, connecting three questions into one coherent story an interviewer will appreciate:** the Kalman filter, Bayesian linear regression (Q29), and even Ridge regression (Q10) are all instances of the **same underlying idea** — combine a prior belief with new data, weighted by their relative reliabilities. Ridge is the *static, single-shot* version (one prior, one batch of data, solved once in closed form). Bayesian regression is the *general, single-update* version (one prior, one batch of data, but explicitly tracking the full posterior distribution, not just a point estimate). The Kalman filter is the *sequential, repeated-update, time-varying-state* version — it's exactly Bayesian updating run once per timestep, where each period's **posterior becomes next period's prior** (after inflating its variance by $Q$ to account for the state's own genuine drift between observations). Being able to draw this connection explicitly, unprompted, is a strong signal of genuinely unified understanding rather than three memorized, disconnected formulas.

### Extending to a Vector State — The Multi-Asset Hedge Ratio Case

The scalar version above generalizes directly to hedging a basket (e.g., a financing book's exposure across multiple collateral types simultaneously) by replacing scalars with matrices:

$$\hat\beta_{t|t-1} = F\hat\beta_{t-1|t-1}, \quad P_{t|t-1}=FP_{t-1|t-1}F^\top+Q$$
$$K_t = P_{t|t-1}H_t^\top\big(H_tP_{t|t-1}H_t^\top+R\big)^{-1}, \quad \hat\beta_{t|t}=\hat\beta_{t|t-1}+K_t\big(y_t-H_t\hat\beta_{t|t-1}\big), \quad P_{t|t}=(I-K_tH_t)P_{t|t-1}$$

**Line-by-line:** $F$ is now a state-transition matrix (allowing, e.g., different collateral-type hedge ratios to have correlated drift, not just independent random walks); $H_t$ is the observation matrix mapping the full state vector to the specific scalar (or vector) actually observed at time $t$; the structure of every equation is unchanged — same predict/update logic, same Bayesian-updating interpretation — just promoted from scalars to matrices. This is exactly the kind of natural, structured generalization an interviewer wants to see you produce on the spot, rather than needing an entirely new derivation from scratch for the multivariate case.

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

### Worked Numerical Example — One Full Forward + Backward Pass by Hand

Take a tiny network: input $x=(1.0, 2.0)$, a single hidden unit ($h=1$), weights $W^{(1)}=(0.5, -0.3)$, $b^{(1)}=0.1$, output weight $W^{(2)}=0.8$, $b^{(2)}=0$, target $y=1.0$, sigmoid activation, learning rate $\eta=0.1$.

**Forward pass:**
$$z^{(1)} = 0.5(1.0)+(-0.3)(2.0)+0.1 = 0.5-0.6+0.1=0.0 \implies a^{(1)}=\sigma(0)=0.5$$
$$z^{(2)} = 0.8(0.5)+0 = 0.4 = \hat y$$
$$\mathcal{L} = \tfrac12(0.4-1.0)^2 = \tfrac12(0.36)=0.18$$

**Backward pass:**
$$\delta^{(2)} = \hat y - y = 0.4-1.0=-0.6$$
$$\frac{\partial\mathcal{L}}{\partial W^{(2)}} = \delta^{(2)}a^{(1)} = -0.6\times0.5=-0.3, \qquad \frac{\partial\mathcal{L}}{\partial b^{(2)}}=\delta^{(2)}=-0.6$$
$$\delta^{(1)} = W^{(2)}\delta^{(2)}\cdot\sigma'(z^{(1)}) = 0.8\times(-0.6)\times\big[\sigma(0)(1-\sigma(0))\big] = -0.48\times(0.5\times0.5)=-0.48\times0.25=-0.12$$
$$\frac{\partial\mathcal{L}}{\partial W^{(1)}_1} = \delta^{(1)}x_1 = -0.12\times1.0=-0.12, \qquad \frac{\partial\mathcal{L}}{\partial W^{(1)}_2}=\delta^{(1)}x_2=-0.12\times2.0=-0.24$$

**Update ($\theta \leftarrow \theta - \eta\,\partial\mathcal{L}/\partial\theta$):**
$$W^{(2)}_{\text{new}} = 0.8-0.1(-0.3)=0.83, \qquad W^{(1)}_{\text{new}}=(0.5-0.1(-0.12),\ -0.3-0.1(-0.24)) = (0.512,\ -0.276)$$

**Say it out loud:** **"After one step, both weights moved in the direction that reduces the loss — $W^{(2)}$ increased because the hidden activation was positive and the output was too low relative to target, so increasing the weight that maps that hidden unit to the output moves $\hat y$ closer to 1.0. I'd verify this by recomputing the forward pass with the new weights and confirming the loss decreased — a habit worth stating explicitly, because it's exactly how you'd sanity-check a from-scratch backprop implementation against silent sign errors, which are the single most common bug when implementing this by hand."**

### Computational Graph View — Why Backprop Is "Just" Reverse-Mode Autodiff

**Feynman explanation, the deepest level of understanding an interviewer can probe for:** every operation in the forward pass (matrix multiply, add bias, apply nonlinearity) is a node in a directed acyclic computational graph, and each node knows two things: how to compute its own output given its inputs (forward), and how to compute the gradient of its inputs given the gradient of its output (backward — the "vector-Jacobian product," or VJP). Backpropagation is simply **topologically traversing this graph in reverse order**, at each node calling its local VJP and multiplying by the upstream gradient — this is exactly what PyTorch's `autograd` does automatically, and understanding it this way (rather than "backprop is a special algorithm for neural networks") is what lets you correctly reason about gradients through *arbitrary* differentiable operations — attention (Q26), Kalman filters (Q19) if made differentiable, or a custom loss function — not just the specific MLP case memorized from a textbook.

### Common Interview Trap — Vectorizing Across a Mini-Batch

The single-example derivation above must be extended to a batch of $m$ examples for any real implementation. **Say it out loud, to preempt the follow-up question:** **"For a batch, $\delta^{(2)}$ becomes a $(m,1)$ matrix rather than a scalar, and every gradient becomes a sum (or mean) over the batch dimension — $\partial\mathcal{L}/\partial W^{(2)} = \frac1m\sum_i \delta_i^{(2)}(a_i^{(1)})^\top$ — the mean is what makes the effective learning rate roughly independent of batch size, which is why 'mean over batch' rather than 'sum over batch' is the standard convention; using sum would implicitly scale your effective learning rate by the batch size, a subtle but real bug I've seen trip up from-scratch implementations."**

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
───────────  ───────────  ─────────────────  ─────────────────────────────
Sigmoid      (0, 1)       0.25               Vanishing gradient (deep nets)
Tanh         (-1, 1)      1.0                Vanishing (less severe than sigmoid)
ReLU         [0, ∞)       1.0 (z>0), 0 else  Dying ReLU (permanently zero grad)
LeakyReLU    (-∞, ∞)      1.0 / 0.01α        Mitigates dying ReLU
GELU/SiLU    smooth       ~1.0 near 0        Used in modern transformers
```

### Worked Numerical Example — Watching a Gradient Vanish Across Depth

Take a 10-layer sigmoid network where, favorably, every $\sigma'(z^{(l)})=0.25$ exactly (the theoretical maximum) and every weight matrix has spectral norm exactly 1 (also favorable — no additional shrinkage from the weights themselves). The gradient magnitude reaching layer 1 relative to the output layer's gradient is:

$$\prod_{l=2}^{10}\sigma'(z^{(l)}) = 0.25^9 \approx 3.8\times10^{-6}$$

**Say it out loud:** **"Even under the best-case assumption for sigmoid — maximal derivative at every single layer, which almost never actually holds since most neurons aren't sitting exactly at $z=0$ — a 10-layer network already attenuates the gradient by a factor of roughly 260,000. In practice, with typical saturation (many neurons sitting in the flatter parts of the sigmoid where $\sigma'\ll0.25$), a network deeper than 4-5 layers using sigmoid activations throughout is essentially untrainable in its earliest layers via plain SGD — which is precisely why sigmoid/tanh survive today only at output layers (where you specifically want a bounded, interpretable range) and never as the default choice for hidden layers in a deep architecture."**

### He and Xavier Initialization — The Other Half of the Vanishing/Exploding Gradient Story

**Feynman explanation:** even with ReLU (bounded-below-by-zero derivative issues aside), a deep network can still vanish or explode purely from **weight initialization scale**, independent of the activation function's own derivative bound. If weights are initialized too large, pre-activations $z^{(l)}$ grow in variance layer over layer, pushing even ReLU units into a regime where a large fraction saturate (go permanently negative, "dying ReLU," Q21's activation table); if weights are initialized too small, activations shrink toward zero and the effective signal vanishes regardless of the activation's own gradient properties.

**He initialization** (used for ReLU networks) sets $W^{(l)}_{ij}\sim N(0, 2/n_{l-1})$, where $n_{l-1}$ is the fan-in (number of units in the previous layer). **Derivation of the "2":** for a ReLU with zero-mean, symmetric input $z$, exactly half of $z$'s realizations pass through as themselves and half become exactly 0, so $\text{Var}(\text{ReLU}(z)) = \tfrac12\text{Var}(z)$ — to keep the variance of activations constant across layers (neither growing nor shrinking), the weight variance must compensate for this factor-of-2 loss, giving $\text{Var}(W)=2/n_{l-1}$ rather than the $1/n_{l-1}$ that would be correct for a linear (non-ReLU) layer (Xavier/Glorot initialization, derived the same way but without the factor-of-2 correction, since a linear layer doesn't zero out half its inputs).

**Say it out loud:** **"He initialization isn't an arbitrary convention — it's the specific variance that keeps the forward-pass activation variance constant across layers for a ReLU network, which in turn keeps the backward-pass gradient variance roughly constant as well, since the same $W^\top$ Jacobian structure applies in reverse. Getting this wrong — e.g., using Xavier initialization (designed for tanh/sigmoid) on a ReLU network — is a subtle, easy-to-miss bug that manifests as slower or unstable training without an obvious error message, which is exactly why frameworks like PyTorch default to He initialization for `nn.Linear` layers followed by ReLU."**

### Modern Activations — GELU/SiLU, and Why Transformers Moved Away From ReLU

$$\text{GELU}(z) = z\cdot\Phi(z), \qquad \text{SiLU}(z) = z\cdot\sigma(z)$$

**Line-by-line:** both multiply the input by a smooth, sigmoid-like "gate" (the standard Normal CDF $\Phi$ for GELU, the logistic sigmoid for SiLU) rather than ReLU's hard, non-differentiable-at-zero cutoff. **Feynman explanation:** ReLU's derivative is a discontinuous step function (0 or 1, nothing in between) — this hard boundary means a neuron sitting right at the boundary during training can flip discontinuously between "active" and "dead" from one gradient step to the next, adding optimization noise. GELU/SiLU's smooth, everywhere-differentiable gating removes this discontinuity, empirically producing smoother loss landscapes and marginally better convergence — a big enough effect at transformer scale (billions of parameters, extremely deep networks) that essentially every modern LLM (GPT, Claude's underlying architecture family, LLaMA-class models) uses a GELU or SiLU-family activation rather than plain ReLU, even though the difference is often negligible at small scale.

[🔝 Back to Top](#table-of-contents)

---
---

## Q22 · RNNs — Backpropagation Through Time

**Recurrence:** $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$, output $\hat y_t = W_{hy}h_t$.

**BPTT derivation.** The total loss is $\mathcal{L}=\sum_t \mathcal{L}_t$ . Because $h_t$ depends on $h_{t-1}$ which depends on $h_{t-2}$, etc., the gradient w.r.t. the **shared** weight $W_{hh}$ must sum contributions through **every** path back through time:

$$\frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t}\frac{\partial \mathcal{L}_t}{\partial h_t}\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)\frac{\partial h_k}{\partial W_{hh}}$$

**Line-by-line:** $\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1-\tanh^2(z_j))\,W_{hh}$ — the same $W_{hh}$ matrix reappears at **every** timestep because weights are shared across time (that's what makes it "recurrent"). So the product term $\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}$ is a product of $(t-k)$ copies of essentially the same matrix (modulated by the tanh-derivative diagonal) — and just like Q21's depth argument, repeated multiplication by a matrix with eigenvalues $<1$ **vanishes exponentially in $t-k$**, while eigenvalues $>1$ **explode exponentially**. This is precisely why vanilla RNNs cannot learn dependencies more than roughly 10–20 steps back — and precisely the motivation for LSTM (Q23).

### Worked Numerical Example — Why 60 Steps Back Is Effectively Unreachable

From the code's actual output pattern: suppose the dominant eigenvalue of $W_{hh}$ is $\lambda_{\max}=0.9$ and, favorably, $\tanh'(z_j)\approx1$ at every step (near-linear regime, not saturated). The gradient norm at lag $k$ scales roughly as $\lambda_{\max}^k$:

$$\lambda_{\max}^{5} = 0.9^5 = 0.590, \qquad \lambda_{\max}^{30}=0.9^{30}=0.0424, \qquad \lambda_{\max}^{60}=0.9^{60}=0.00180$$

**Say it out loud:** **"Even with a relatively benign eigenvalue of 0.9 — meaning the recurrent weights are barely contractive — the gradient has decayed to under 0.2% of its original magnitude by 60 steps back. In practice $\tanh'(z)$ is essentially never exactly 1 (most hidden units spend meaningful time away from $z=0$), so the true decay is faster still. This is the precise, quantitative reason a financing-desk model trying to learn 'today's spread widening is caused by a shock 2 months ago' cannot rely on a vanilla RNN — the gradient signal from that event has, for all practical purposes, vanished to numerical noise by the time it would need to update the weights responsible for encoding it."**

Conversely, if $\lambda_{\max}=1.3$ (recurrent weights mildly amplifying), the same computation **explodes**: $1.3^{30}\approx2620$ — a gradient tens of thousands of times larger than at the output layer, which in practice causes NaN losses or wildly oscillating training, motivating gradient clipping as a near-mandatory practical safeguard for any RNN training loop.

### Gradient Clipping — The Practical Band-Aid (Necessary, Not Sufficient)

$$g \leftarrow g \cdot \min\left(1,\ \frac{\text{threshold}}{\lVert g\rVert}\right)$$

**Line-by-line:** if the gradient's L2 norm exceeds a chosen threshold, rescale the *entire* gradient vector (preserving its direction) down to exactly that threshold's magnitude; if it's already below threshold, leave it untouched. **Feynman explanation:** clipping doesn't fix vanishing gradients at all — it only prevents the **exploding** case from derailing training with a single catastrophically large update. It's a necessary practical safeguard for any RNN/LSTM training loop (used almost universally, threshold typically 1.0–5.0), but it does nothing to help a genuinely vanishing gradient recover lost signal — that requires an architectural fix (LSTM/GRU's additive cell state, Q23) or a training-procedure fix (truncated BPTT with careful state-carrying, or gradient-friendly initialization of $W_{hh}$ as orthogonal, which keeps all eigenvalues at exactly 1 at initialization).

### Truncated BPTT — The Practical Compromise for Long Sequences

For a sequence of 10,000 timesteps (e.g., tick-level order flow over a full trading day), full BPTT back through the entire sequence is both computationally prohibitive ($O(T)$ memory to store every intermediate activation) and, per the vanishing-gradient argument above, largely pointless beyond ~20-100 steps anyway. **Truncated BPTT** processes the sequence in chunks (e.g., 50 steps), carrying the hidden state **forward** across chunk boundaries (so the model still "remembers" long-run context in its state) but only backpropagating gradients **within** each chunk. **Say it out loud:** *"This is a pragmatic acknowledgment of exactly the mathematical limit just derived — since gradients meaningfully vanish beyond roughly 20-50 steps for a vanilla RNN anyway, truncating BPTT to that same horizon costs almost nothing in practice while providing an enormous, necessary reduction in memory and compute for long financial sequences."*

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

**Why this solves vanishing gradients — derive $\partial C_t/\partial C_{t-1}$ properly.** Apply the product rule to $C_t = f_t\odot C_{t-1} + i_t\odot\tilde C_t$, treating $f_t, i_t, \tilde C_t$ as themselves functions of $h_{t-1}$ ( hence, indirectly, of $C_{t-1}$ via $h_{t-1}=o_{t-1}\odot\tanh(C_{t-1})$ ):

$$\frac{\partial C_t}{\partial C_{t-1}} = \underbrace{f_t}_{\text{direct path}} + \underbrace{\left(C_{t-1}\odot\frac{\partial f_t}{\partial h_{t-1}} + \tilde C_t\odot\frac{\partial i_t}{\partial h_{t-1}} + i_t\odot\frac{\partial\tilde C_t}{\partial h_{t-1}}\right)\odot\frac{\partial h_{t-1}}{\partial C_{t-1}}}_{\text{indirect paths, through the gates}}$$

**Line-by-line:** the first term $f_t$ comes directly from differentiating $f_t\odot C_{t-1}$ with $C_{t-1}$ held as the explicit variable — since this is **element-wise multiplication**, not matrix multiplication through a saturating nonlinearity, its "local Jacobian" is simply the diagonal matrix $\text{diag}(f_t)$, whose eigenvalues are exactly the forget-gate activations $f_t \in (0,1)$. This is qualitatively different from the vanilla-RNN case (Q22), where the **same** weight matrix $W_{hh}$ was reapplied every timestep, forcing repeated multiplication by a matrix whose eigenvalues are fixed by training and often $\ll 1$. Here, $f_t$ is *dynamically produced per-timestep by a sigmoid gate*, so the network can **learn to set $f_t\approx 1$ specifically for the features/timesteps that must be remembered**, making that direct-path term $\approx 1$ (identity Jacobian) — an additive "gradient highway," the same architectural trick later generalized into ResNet skip connections ($y = x + F(x)$, whose Jacobian is $I + \partial F/\partial x$, guaranteeing a gradient path of exactly 1 regardless of what $F$ learns).

The indirect terms are real and non-zero (this is why LSTM gradient flow isn't *perfectly* 1, just close to it for well-trained gates) — but they are second-order corrections modulated by the gate derivatives $\sigma'(\cdot)\in(0,0.25]$, i.e., the same bounded-derivative structure from Q21, just no longer the *dominant* term in the recursion the way $W_{hh}\odot\tanh'(\cdot)$ was for vanilla RNN. Empirically and by construction, this is sufficient to push the effective gradient-decay half-life from ~10–20 steps (vanilla RNN) to 100+ steps (LSTM).

### Worked Numerical Example — Computing One LSTM Timestep by Hand

Simplify to scalar hidden/cell state for tractability. Suppose $h_{t-1}=0.3$, $x_t=1.0$, $C_{t-1}=0.5$, and (post-sigmoid/tanh, i.e., already-computed) gate activations from some trained weights: $f_t=0.8$, $i_t=0.6$, $\tilde C_t=0.4$, $o_t=0.7$.

$$C_t = f_t\odot C_{t-1} + i_t\odot\tilde C_t = 0.8(0.5)+0.6(0.4) = 0.40+0.24=0.64$$
$$h_t = o_t\odot\tanh(C_t) = 0.7\times\tanh(0.64) = 0.7\times0.565=0.396$$

**Say it out loud:** **"The forget gate at 0.8 means the cell retains 80% of its previous content — mostly remembering, slightly forgetting. The input gate at 0.6 means 60% of the new candidate information $\tilde C_t=0.4$ gets written in. The new cell state 0.64 is higher than the old 0.5, meaning net-new information pushed the 'memory' up. The output gate at 0.7 then decides how much of that internal memory to actually expose to the rest of the network as the hidden state $h_t$ — note $h_t=0.396$ is a damped, filtered view of the cell state, not the raw cell state itself; this separation between 'what's remembered' ($C_t$) and 'what's shared with the outside world right now' ($h_t$) is the architectural feature that lets the cell state persist information across many timesteps even while the exposed hidden state fluctuates more responsively step-to-step."**

### Peephole Connections and Other LSTM Variants — What Interviewers May Probe Next

**Feynman explanation of peephole connections:** the vanilla LSTM's gates ($f_t, i_t, o_t$) are computed only from $h_{t-1}$ and $x_t$ — they never directly "see" the cell state $C_{t-1}$ they're about to modify. **Peephole LSTMs** (Gers & Schmidhuber, 2000) let each gate also take $C_{t-1}$ (or $C_t$ for the output gate) as direct input: $f_t=\sigma(W_f[h_{t-1},x_t]+V_f\odot C_{t-1}+b_f)$. **Say it out loud:** *"The intuition is that a gate deciding 'should I forget this memory' arguably should be able to look at the memory's actual current value, not just the previous hidden state's summary of it — in practice this gives a modest but not always significant empirical improvement, and most production systems today use plain LSTM or GRU rather than peephole variants because the added parameter count rarely pays for itself outside of very specific sequence-modeling benchmarks."*

### Initialization Detail Worth Volunteering — Why Forget-Gate Bias Is Initialized to 1

Notice in the code below, `self.b_f = np.ones(self.hidden_dim)` rather than zeros. **Feynman explanation:** at the very start of training, before any gradient signal has shaped the weights, a forget gate initialized near zero (via zero bias, with small random input weights) would default to $\sigma(\approx0)=0.5$ — a fairly aggressive "forget half of everything, every timestep" default, which actively fights against the network ever learning long-range dependencies before it's had a chance to learn otherwise. Initializing the forget-gate bias to a positive value (commonly 1, sometimes higher) instead defaults the sigmoid output near $\sigma(1)\approx0.73$ — "remember most things by default" — giving the network a head start toward the long-memory behavior it's supposed to learn, rather than requiring gradient descent to discover from scratch that remembering is often useful. This is a small, easily-missed implementation detail (Jozefowicz et al., 2015 empirically demonstrated its importance) that separates a textbook-correct LSTM implementation from a well-tuned production one.

### Multi-Layer (Stacked) LSTM — Extending Beyond a Single Cell

For genuinely complex sequence patterns (e.g., modeling a full financing curve's term structure jointly with cross-tenor dependencies), a single LSTM layer is often insufficient — **stacked LSTMs** feed the hidden-state sequence output of one LSTM layer as the input sequence to a second LSTM layer:

$$h_t^{(1)} = \text{LSTM}^{(1)}(x_t, h_{t-1}^{(1)}, C_{t-1}^{(1)}), \qquad h_t^{(2)} = \text{LSTM}^{(2)}(h_t^{(1)}, h_{t-1}^{(2)}, C_{t-1}^{(2)})$$

**Say it out loud:** *"Each additional stacked layer lets the network learn increasingly abstract temporal features — layer 1 might learn short-horizon momentum/reversion patterns directly from raw returns, while layer 2, operating on layer 1's already-processed hidden states, can learn longer-horizon regime-level patterns built on top of those lower-level features. This is directly analogous to how stacking convolutional layers in a CNN builds up from edges to textures to objects — depth in the recurrent-layer dimension serves the same hierarchical-feature-learning purpose as depth in a standard feedforward network (Q20/Q21), just applied to the temporal-abstraction axis instead."*

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
Gates               3 (forget, input, output)         2 (update, reset)
State               Separate cell (C) + hidden (h)    Single hidden state (h)
Params (per layer)  ~4×(d²+dk)                        ~3×(d²+dk)
Training speed      Slower (more params)              ~15-25% faster
Long-range memory   Slightly stronger (separate C)    Slightly weaker on very long seqs
When to prefer      Long sequences, ample compute     Latency-sensitive prod, smaller data
```

### Worked Numerical Example — GRU vs. LSTM on the Same Inputs

Using similar scalar values to the Q23 LSTM example: $h_{t-1}=0.3$, $x_t=1.0$, and (post-activation) gate values $z_t=0.4$ (update gate), $r_t=0.6$ (reset gate), yielding candidate $\tilde h_t=\tanh(W[r_t\odot h_{t-1}, x_t]) = 0.55$ (given).

$$h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde h_t = (1-0.4)(0.3) + 0.4(0.55) = 0.6(0.3)+0.4(0.55)=0.18+0.22=0.40$$

**Say it out loud:** **"The update gate $z_t=0.4$ acts as a direct interpolation weight — 40% new candidate content, 60% retained old state, computed in a single convex combination. Compare this to LSTM's two-step process (separate forget-then-add via $f_t\odot C_{t-1}+i_t\odot\tilde C_t$, which are NOT constrained to sum to 1) — GRU's update gate enforces $z_t + (1-z_t) = 1$ by construction, meaning the new state is always a weighted average, never able to independently scale up total 'memory magnitude' the way LSTM's unconstrained $f_t,i_t$ can. This is the precise mathematical sense in which GRU is a constrained special case of LSTM's more general gating freedom — fewer degrees of freedom, in exchange for fewer parameters and faster training."**

### When the GRU-vs-LSTM Choice Actually Matters in Practice — An Honest Answer

**Feynman explanation, calibrated to avoid overclaiming:** in the vast majority of applied sequence-modeling tasks (including most financial time-series problems), GRU and LSTM perform within noise of each other — the architectural difference matters far less than data quality, sequence length, and regularization. The cases where the choice is **not** a coin flip: (1) extremely long sequences (1000+ timesteps) where LSTM's separate, unconstrained cell state has a slight empirical edge in preserving very-long-range dependencies; (2) severe compute/latency constraints (e.g., a real-time signal engine running inference on every tick) where GRU's ~25% parameter reduction directly translates to lower inference latency, which can be the deciding factor for a production system with a hard latency SLA regardless of any marginal accuracy difference. **Say it out loud, to show calibrated judgment rather than dogma:** *"I wouldn't spend a research cycle agonizing over GRU vs. LSTM in isolation — I'd pick GRU as the default for a latency-sensitive production signal (the JD's cross-asset platform likely has real-time constraints), validate empirically against LSTM on the actual target task, and only invest in the LSTM's extra complexity if there's a demonstrated, statistically significant accuracy gain that justifies the added inference cost — this is itself an example of the cost/quality/latency trade-off framework from Q7, applied to architecture choice rather than model-provider choice."*

[🔝 Back to Top](#table-of-contents)

---
---

## Q25 · Regularization — Dropout & BatchNorm Math

**Dropout.** During training, each unit is independently zeroed with probability $p$:

$$\tilde a_j = \frac{m_j}{1-p}\, a_j, \qquad m_j \sim \text{Bernoulli}(1-p)$$

**Say it out loud:** **"The $1/(1-p)$ scaling is 'inverted dropout' — it rescales the surviving activations at train time so that the expected activation matches what it would be at test time with no dropout at all, meaning inference requires zero changes to the forward pass."** Mechanistically, dropout prevents co-adaptation: a unit can't rely on any specific set of other units always being present, which is mathematically equivalent to training an implicit ensemble of $2^h$ thinned sub-networks that share weights, averaged at test time.

**Batch Normalization.** For a mini-batch $\mathcal{B}=\{z_1,\dots,z_m\}$ of pre-activations for one unit:

$$\mu_\mathcal{B} = \frac1m\sum_i z_i, \qquad \sigma_\mathcal{B}^2 = \frac1m\sum_i(z_i-\mu_\mathcal{B})^2, \qquad \hat z_i = \frac{z_i-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}, \qquad y_i = \gamma\hat z_i + \beta$$

### Worked Numerical Example — BatchNorm on a Small Batch by Hand

Take a mini-batch of 4 pre-activation values for a single unit: $z = (2, 4, 4, 6)$.

$$\mu_\mathcal{B} = \frac{2+4+4+6}{4} = 4, \qquad \sigma_\mathcal{B}^2 = \frac{(2-4)^2+(4-4)^2+(4-4)^2+(6-4)^2}{4} = \frac{4+0+0+4}{4}=2$$

$$\hat z = \frac{z-4}{\sqrt{2+\epsilon}} \approx \left(\frac{-2}{1.414},\ 0,\ 0,\ \frac{2}{1.414}\right) = (-1.414,\ 0,\ 0,\ 1.414)$$

With learned $\gamma=1.5, \beta=0.5$: $y = 1.5\hat z + 0.5 = (-1.621,\ 0.5,\ 0.5,\ 2.621)$.

**Say it out loud:** **"Notice the normalized values $\hat z$ have exactly zero mean and unit variance by construction — I could verify this: mean of $(-1.414,0,0,1.414)$ is exactly 0, and the variance is exactly 1. The final output $y$ then has mean $\beta=0.5$ and standard deviation $\gamma=1.5$ — the network has learned (via $\gamma,\beta$) that it actually wants this unit's output centered at 0.5 with a spread of 1.5, not the raw normalized 0/1 — this is exactly the 'undo the normalization if that's optimal' escape hatch mentioned above, made concrete with real numbers."**

### Train vs. Inference Mode — The Subtlety That Trips Up Production Deployments

**Feynman explanation of a genuinely common production bug:** at inference time, you typically process a single example (or the model must give a deterministic output regardless of what other examples happen to be in the current batch) — so BatchNorm cannot use the current batch's $\mu_\mathcal{B},\sigma_\mathcal{B}^2$ the way it does during training (there may be no "batch" at all, or a batch of size 1 whose variance is undefined/zero). The fix: during training, maintain a **running exponential moving average** of $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ across all batches seen (using the exact same EWMA recursion from Q2!), and at inference time use these **frozen running statistics** instead of recomputing from the (possibly single-example, possibly nonexistent) current batch. **Say it out loud:** *"Forgetting to call `model.eval()` in PyTorch — which switches BatchNorm layers from batch-statistics mode to running-statistics mode — is one of the most common silent bugs in deployed deep learning models: the model still runs and produces a number, but that number is now dependent on whatever other examples happen to be batched alongside it in production, an obviously unacceptable behavior for a live pricing or signal-generation service that must be invariant to unrelated concurrent requests."*

### Dropout as Implicit Ensembling — Made Precise

**Feynman explanation, deepened:** with $h$ hidden units and dropout probability $p$, there are $2^h$ possible "thinned" sub-networks (each unit either present or absent). Training with dropout can be shown to approximately optimize the **average** loss across this entire ensemble of $2^h$ weight-sharing sub-networks (since a different random subset is "active" on every mini-batch). At test time, using **all** units with inverted-dropout scaling approximates averaging the predictions of all $2^h$ sub-networks in a single forward pass — a computationally free approximation to an otherwise combinatorially intractable ensemble average. **Say it out loud:** **"This is why dropout provides regularization benefits similar in spirit to Random Forest's bagging (Q13) — both are averaging over many slightly-different models trained on slightly-different views of the data (bootstrap samples for Random Forest, randomly-thinned sub-networks for dropout) — but dropout achieves this within a single network's training run rather than requiring $B$ literally separate models to be trained and stored, which is the practical reason it scales to deep learning where training even one model can be expensive, let alone hundreds."**

### Layer Normalization — Why Transformers Use It Instead of BatchNorm

**Feynman explanation:** BatchNorm normalizes **across the batch dimension** for each feature — this requires a reasonably large, representative batch to get stable statistics, and breaks down for variable-length sequences (padding tokens would corrupt the batch statistics) or when batch size must be small (e.g., 1, for autoregressive generation). **LayerNorm** instead normalizes **across the feature dimension**, independently for each individual example: $\hat z_i = (z_i - \mu_i)/\sqrt{\sigma_i^2+\epsilon}$ where $\mu_i,\sigma_i^2$ are computed across that single example's own features, not across the batch. This makes LayerNorm invariant to batch size and sequence padding, exactly why every modern transformer (including the attention mechanism in Q26) uses LayerNorm rather than BatchNorm — a detail worth volunteering to show awareness of why the architectural choice differs between CNN-era vision models and modern sequence/language models.

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

### Worked Numerical Example — Attention Weights for 3 Positions by Hand

Take a toy sequence of 3 positions with 2-dimensional query/key vectors: $q_1=(1,0)$, $k_1=(1,0)$, $k_2=(0,1)$, $k_3=(1,1)$, $v_1=(2,0)$, $v_2=(0,2)$, $v_3=(1,1)$, $d_k=2$.

$$q_1\cdot k_1 = 1, \quad q_1\cdot k_2=0, \quad q_1\cdot k_3=1$$

Scaled scores (divide by $\sqrt{2}\approx1.414$): $(0.707,\ 0,\ 0.707)$.

$$\text{softmax}(0.707,\,0,\,0.707) = \frac{(e^{0.707}, e^0, e^{0.707})}{e^{0.707}+e^0+e^{0.707}} = \frac{(2.028,\ 1,\ 2.028)}{5.056} = (0.401,\ 0.198,\ 0.401)$$

$$\text{output}_1 = 0.401(2,0)+0.198(0,2)+0.401(1,1) = (0.802,0)+(0,0.396)+(0.401,0.401) = (1.203,\ 0.797)$$

**Say it out loud:** **"Position 1's query attends almost equally to positions 1 and 3 (both scored 0.707, both getting 40.1% attention weight) and much less to position 2 (0 score, only 19.8% weight, which is nonzero purely because softmax never assigns exactly zero probability to any option) — the output is a blend dominated by values 1 and 3. This is a concrete illustration of why attention is described as a 'soft, differentiable lookup': rather than picking exactly one key (a hard argmax), the mechanism blends proportionally to similarity, and crucially, that blending weight is fully differentiable, so gradient descent can directly adjust which positions a query 'should' attend to by adjusting the $W_Q, W_K$ projections that produced these query/key vectors in the first place."**

### Multi-Head Attention — Why One Attention Pattern Isn't Enough

**Feynman explanation:** a single attention head computes one specific *type* of relationship between positions (e.g., perhaps a head that mostly learns "attend to the immediately preceding day's value" — a short-range momentum pattern). But a financing-curve model plausibly needs to capture *several qualitatively different* relationships simultaneously — short-range momentum, longer-range mean-reversion, day-of-month seasonality (Q15's SARIMA discussion) — and a single head's fixed $d_k$-dimensional query/key space may not have enough capacity to represent all of them at once without them interfering. **Multi-head attention** runs $H$ independent attention computations in parallel, each with its own learned $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$ projected into a smaller subspace (so total compute stays roughly constant), then concatenates and linearly projects the results:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O, \qquad \text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})$$

**Say it out loud:** **"Splitting into $H$ heads each with dimension $d_k/H$ costs the same total compute as one full-dimensional head, but lets different heads specialize — empirically, when you visualize learned attention patterns in trained transformers, different heads really do learn qualitatively distinct patterns (some attending locally, some attending to specific syntactic/positional patterns), which is direct empirical evidence the extra representational flexibility is being used, not wasted."**

### Positional Encoding — The Detail That's Easy to Forget and Costly to Skip

**Feynman explanation of a genuinely important gotcha:** the attention formula as written is **permutation-invariant** — if you shuffled the order of the input sequence's rows, the set of attention outputs would simply be correspondingly shuffled, with no change to their *values*. This means attention alone has **no innate sense of sequence order** — a critical gap for time-series data, where "yesterday" and "30 days ago" carry fundamentally different information despite possibly having similar feature values. The fix is to add a **positional encoding** to each input embedding before it enters the attention layers — commonly $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d})$ — a fixed (or, in modern architectures, learned) vector that varies smoothly and uniquely with position, letting the attention mechanism's dot-product implicitly recover relative-position information. **Say it out loud:** *"If I were building an attention-based model over a 60-day financing-rate lookback window and forgot positional encoding, the model would be mathematically unable to distinguish 'the spike happened yesterday' from 'the spike happened 45 days ago' — a subtle-sounding bug that in practice produces a model that trains fine and shows a plausible loss curve, but silently underperforms an LSTM on any task where the *order*, not just the *set*, of past observations matters, which is essentially every financial time-series task."*

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

**Relationship to PCA.** For a **linear** encoder/decoder with no nonlinearity and squared-error loss, the optimal solution spans the same subspace as the top-`k` principal components of $X$ (this can be shown via the Eckart–Young theorem: the best rank-`k` approximation of $X$ in Frobenius norm is exactly its truncated SVD). A nonlinear autoencoder generalizes this to a **nonlinear manifold** approximation — it can capture curved lower-dimensional structure (e.g., a nonlinear relationship across the financing curve/term structure) that linear PCA cannot.

**Anomaly detection.** Train the autoencoder only on "normal" regime data; at inference, compute the reconstruction error:

$$e(x) = \lVert x - g_\phi(f_\theta(x))\rVert_2^2$$

### Worked Numerical Example — PCA-Equivalent Reconstruction by Hand

Take a 2D dataset (already centered) with two points $x_1=(3,1)$, $x_2=(-3,-1)$ — perfectly correlated along direction $(3,1)/\sqrt{10}$. A rank-1 linear autoencoder ($k=1$) should learn to encode/decode along exactly this direction. Suppose the learned encoder weight (after convergence) is $w_{\text{enc}}=(0.949, 0.316)$ (normalized $(3,1)/\sqrt{10}$) and decoder $w_{\text{dec}}=(0.949, 0.316)^\top$ (tied to the same direction, as PCA would produce):

$$z_1 = w_{\text{enc}}\cdot x_1 = 0.949(3)+0.316(1) = 2.847+0.316=3.163$$
$$\hat x_1 = z_1\, w_{\text{dec}} = 3.163\times(0.949,0.316) = (3.002,\ 0.999) \approx (3,1)$$

**Say it out loud:** **"Reconstruction is essentially perfect — 3.163 is exactly $\lVert x_1\rVert$ projected onto the principal direction, and decoding recovers the original point almost exactly, confirming this rank-1 linear autoencoder has converged to the same subspace PCA would find via the first eigenvector of the covariance matrix. If I fed in an off-manifold point, say $(3,-1)$ — orthogonal to the learned direction — the encoding would project out all of that orthogonal information: $z=0.949(3)+0.316(-1)=2.531$, decoding to $\hat x=(2.402,0.800)$, a reconstruction error of $\lVert(3,-1)-(2.402,0.800)\rVert\approx1.90$ — a large residual, correctly signaling this point doesn't lie on the learned 'normal' manifold, exactly the anomaly-detection mechanism the question is testing."**

### Variational Autoencoders — Extending to a Generative, Probabilistic Model

**Feynman explanation of the key conceptual leap:** the plain (deterministic) autoencoder above learns a single point $z$ per input — useful for compression and anomaly scoring, but the latent space has no guaranteed structure, so you can't meaningfully *sample* a new, plausible synthetic point from it. A **Variational Autoencoder (VAE)** instead encodes each input to a full **distribution** over $z$ — typically a Gaussian with learned mean $\mu(x)$ and variance $\sigma^2(x)$ — and adds a regularization term that pulls this learned distribution toward a standard Normal prior $N(0,I)$:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{z\sim q(z|x)}\big[-\log p(x|z)\big]}_{\text{reconstruction term}} + \underbrace{D_{KL}\big(q(z|x)\,\Vert\,N(0,I)\big)}_{\text{regularization term}}$$

**Line-by-line:** the first term is exactly the plain autoencoder's reconstruction loss, just now averaged over samples from the encoded distribution rather than using a single deterministic point. The second term, the KL-divergence to a standard Normal, penalizes the encoder for producing latent distributions that deviate from a well-behaved, "nice" reference distribution — and it is this term specifically that gives the latent space enough regularity that you can later sample $z\sim N(0,I)$ directly (without ever encoding any real input) and get a decoded output that looks like a plausible member of the training distribution — the generative capability a plain autoencoder lacks entirely. **Say it out loud, tying to a financing use case:** *"For anomaly detection on financing flow, the plain deterministic autoencoder above is usually sufficient and simpler to train and reason about — I'd only reach for a VAE if the actual business need were generative, e.g., simulating synthetic-but-realistic order-flow scenarios for stress-testing a risk model against data patterns that haven't occurred historically, which is a genuinely different use case from 'flag this real observation as anomalous.'"*

### The Reparameterization Trick — Why VAEs Are Trainable At All

A natural objection: sampling $z\sim N(\mu(x),\sigma^2(x))$ is a **stochastic** operation — how can gradients flow back through a random sampling step to update $\mu(x),\sigma(x)$? The **reparameterization trick** rewrites the sample as a deterministic function of a separate, parameter-free noise source: $z = \mu(x) + \sigma(x)\odot\epsilon$, where $\epsilon\sim N(0,I)$ is sampled **independently of any learnable parameter**. **Say it out loud:** **"By moving all the randomness into $\epsilon$, which carries no gradient requirement, the path from $\mu(x),\sigma(x)$ to $z$ becomes a perfectly ordinary, differentiable deterministic computation — this is precisely the trick that made VAEs practically trainable via standard backpropagation in 2013, and the same reparameterization idea now appears throughout modern generative modeling (diffusion models use an analogous trick), making it a genuinely foundational piece of technique worth being able to state precisely rather than gesture at vaguely."**

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
   + batch ETL         + feature lineage     model registry         backtest PnL
                                                                    (drift alarm)
```

**Feynman explanation:** the hardest part of this system is not any individual model — it's the **feature store's point-in-time correctness**. If a feature computed "as of" day $t$ accidentally includes information only known on day $t+1$ (e.g., a restated economic figure, or an end-of-day close mislabeled as available intraday), every backtest built on it is silently invalid — this is the single most common cause of a strategy that backtests brilliantly and then fails live, and defending against it is an engineering discipline (immutable, timestamped feature snapshots), not a modeling one.

### Quantifying Champion/Challenger — When to Actually Promote a New Model

**Feynman explanation:** "the challenger model looks better in backtest" is not sufficient grounds to promote it to production — you need a statistical test that accounts for the fact that you're comparing two *correlated* return streams (both trading similar underlying assets), not two independent samples. The appropriate tool is a paired test on the **return differential** $d_t = r_t^{\text{challenger}} - r_t^{\text{champion}}$:

$$t = \frac{\bar d}{\hat\sigma_d/\sqrt{n}}, \qquad \hat\sigma_d^2 = \frac{1}{n-1}\sum_t(d_t-\bar d)^2$$

**Line-by-line:** this is testing $H_0: \mathbb{E}[d_t]=0$ (no genuine difference in expected return) using the **paired** differences directly, which automatically controls for common market-wide moves both models are exposed to (a source of correlated noise that an unpaired two-sample test would incorrectly treat as independent variance, understating the true precision of the comparison). **Say it out loud:** *"Given the paired nature of live-vs-backtest or champion-vs-challenger comparisons in trading, I always use a paired test on the return differential, not a naive two-sample comparison of Sharpe ratios — the latter throws away exactly the correlation structure that makes the comparison statistically powerful in the first place, and can lead to premature model promotion decisions based on what's actually just noise in overall market conditions during the comparison window."* Combined with the CPCV/PBO framework from Q18, this should be run across multiple non-overlapping evaluation windows before promotion, not a single backtest period.

### Live-vs-Backtest Drift Monitoring — The Concrete Alarm Condition

$$\text{Drift}_t = \frac{PnL_t^{\text{live, cumulative}} - PnL_t^{\text{backtest, cumulative}}}{\hat\sigma_{\text{backtest}}\sqrt{t}}$$

**Say it out loud:** **"This is a running z-score of live performance against the backtest's expected path, scaled by $\sqrt{t}$ because cumulative PnL variance grows linearly with time under the backtest's own assumptions — so the drift statistic should stay roughly bounded (not systematically grow) if live performance is behaving consistently with backtest expectations. I'd set an alarm threshold (e.g., $|\text{Drift}_t| > 2.5$ sustained over several days, not a single noisy print) that pages the desk for a live review — this is the concrete implementation of the 'live monitoring' box in the pipeline diagram above, not just a vague aspiration to 'keep an eye on it.'"**

### Capacity Estimation — Why This Gates the Priority Score (Q30 Tie-In)

**Feynman explanation:** a signal's theoretical Sharpe ratio is meaningless without an estimate of how much capital can actually be deployed before the signal's own trading erodes the edge (market impact). A standard square-root market impact model:

$$\text{Impact}(Q) = \eta\cdot\sigma\cdot\sqrt{\frac{Q}{V}}$$

**Line-by-line:** $Q$ is order size, $V$ is average daily volume, $\sigma$ is the asset's volatility, $\eta$ is an empirically-calibrated impact coefficient; the square-root (not linear) scaling reflects that impact grows sub-linearly with size for well-executed orders (a well-known empirical market-microstructure regularity). Capacity is estimated by finding the position size $Q^\ast$ at which expected impact cost equals the signal's raw expected edge — beyond that size, trading the signal is a net loser regardless of how "real" the underlying pattern is. **Say it out loud:** *"This is why the JD's 50+ problem statements need to be prioritized partly on capacity, not just Sharpe — a beautiful, statistically airtight signal with $2M of capacity isn't worth building out to production for a franchise the size of Liquid Financing, whereas a more modest Sharpe signal with $200M of capacity might be the better use of the team's limited engineering bandwidth, which is precisely the trade-off the Q30 priority-scoring framework is designed to make explicit and defensible rather than left to gut feel."*

[🔝 Back to Top](#table-of-contents)

---
---

## Q29 · Bayesian Inference for Regime-Adaptive Position Sizing

**Setup:** treat a macro factor exposure $\beta$ as uncertain with a prior $\beta \sim N(\mu_0, \tau_0^2)$, and update given observed data via Bayes' rule.

$$p(\beta \mid y) \propto p(y\mid\beta)\,p(\beta)$$

For a Gaussian likelihood $y\mid\beta \sim N(X\beta, \sigma^2)$ and Gaussian prior, the posterior is conjugate — also Gaussian — with closed form:

$$\tau_1^2 = \left(\frac{1}{\tau_0^2} + \frac{X^\top X}{\sigma^2}\right)^{-1}, \qquad \mu_1 = \tau_1^2\left(\frac{\mu_0}{\tau_0^2} + \frac{X^\top y}{\sigma^2}\right)$$

**Line-by-line:** the posterior precision (inverse variance) $1/\tau_1^2$ is the **sum** of the prior precision and the data precision — more data (larger $X^\top X$) always shrinks posterior uncertainty, and a tighter prior (smaller $\tau_0^2$, i.e., stronger conviction before seeing data) resists being moved by noisy data. The posterior mean $\mu_1$ is a **precision-weighted average** of the prior mean and the OLS estimate $\hat\beta_{\text{OLS}}=(X^\top X)^{-1}X^\top y$ — exactly the same "prior belief + gain × surprise" structure as the Kalman filter in Q19 (the Kalman filter *is*, in fact, sequential Bayesian updating of a Gaussian state).

### Worked Numerical Example — One Bayesian Posterior Update by Hand

Take a scalar case for tractability: prior belief about a macro factor's exposure $\beta\sim N(\mu_0=0.3, \tau_0^2=0.04)$ (moderately confident it's around 0.3, with std dev 0.2). Observe a single new data point $x=2.0$, $y=0.9$, with assumed noise variance $\sigma^2=0.25$.

$$\tau_1^2 = \left(\frac{1}{0.04}+\frac{(2.0)^2}{0.25}\right)^{-1} = \left(25+16\right)^{-1} = \frac{1}{41}=0.0244$$

$$\mu_1 = 0.0244\left(\frac{0.3}{0.04}+\frac{2.0\times0.9}{0.25}\right) = 0.0244\left(7.5+7.2\right)=0.0244\times14.7=0.359$$

**Say it out loud:** **"The posterior mean 0.359 sits between the prior 0.3 and the raw single-observation OLS estimate $\hat\beta_{OLS}=y/x=0.9/2.0=0.45$ — closer to the prior than to the noisy single data point, because $\tau_0^2=0.04$ represents fairly strong prior conviction relative to the information in just one noisy observation. If instead $\sigma^2$ were much smaller (say 0.01 — a much more informative single observation), the posterior would shift substantially further toward 0.45, correctly reflecting that a very precise single data point should move a moderately-confident prior more than an imprecise one does — this differential responsiveness IS the entire point of doing Bayesian updating rather than either ignoring new data or overwriting the prior wholesale."**

### The Kelly Criterion — Derived, Not Just Asserted

The code above implements a fractional-Kelly position sizer; the full Kelly formula deserves its own derivation since it will likely be probed directly if position-sizing comes up. For a bet with win probability $p$, and even-money payoff (win $b$ per unit staked, lose 1 unit per unit staked on a loss), the Kelly fraction $f^\ast$ maximizes the **expected log-growth rate** of wealth:

$$g(f) = p\ln(1+bf) + (1-p)\ln(1-f)$$

**Line-by-line:** this is the expected value, over many repeated bets, of the *logarithm* of wealth growth per bet — using log-wealth rather than raw expected wealth is the crucial modeling choice, because maximizing raw expected wealth would recommend betting 100% of capital on any positive-edge bet (ruinous under repeated play due to the asymmetry of loss — losing 100% means game over, no future bets possible), whereas maximizing expected *log* wealth correctly penalizes the compounding damage of a large drawdown. Differentiate and set to zero:

$$g'(f) = \frac{pb}{1+bf} - \frac{1-p}{1-f} = 0 \;\;\Rightarrow\;\; pb(1-f) = (1-p)(1+bf) \;\;\Rightarrow\;\; f^\ast = \frac{pb-(1-p)}{b} = p - \frac{1-p}{b}$$

For the special case $b=1$ (even-money bet): $f^\ast = 2p-1$. **Say it out loud:** **"For a signal with a 55% win rate at even-money payoff, full Kelly says bet $2(0.55)-1=10\%$ of capital — a surprisingly modest number for what sounds like a strong edge, which is exactly why practitioners almost never trade full Kelly: it's calibrated to a known, static $p$, whereas real trading edges are themselves estimated with uncertainty (exactly the posterior uncertainty $\tau_1^2$ computed above) — trading full Kelly on an uncertain edge estimate systematically over-bets, because Kelly's optimality assumes you know $p$ exactly. This is precisely why my position-sizing code scales down by both a fixed fractional-Kelly haircut AND divides by the posterior variance — the posterior uncertainty term is doing double duty as a second, dynamically-varying Kelly haircut on top of the fixed one."**

### Connecting Posterior Uncertainty to the Kelly Fraction — Why the Code Divides by Variance

**Feynman explanation of the specific functional form in the code:** the position-size formula `edge / (uncertainty^2 + eps)` is a simplified continuous analogue of Kelly sizing under parameter uncertainty — as posterior variance $\tau_1^2$ shrinks (more/better data, tighter conviction), the recommended position size grows for the same central edge estimate; as $\tau_1^2$ grows (regime uncertainty, per Q29's "widen the prior ahead of a central bank decision" framing), the recommended size shrinks even if the point-estimate edge is unchanged. **Say it out loud:** *"This gives a principled, continuously-varying answer to 'should I size up or down heading into an FOMC meeting' — rather than an arbitrary discrete rule like 'cut all positions 50% the day before FOMC,' the Bayesian framework says exactly how much to cut, tied directly to how much the posterior variance on the relevant macro factor genuinely widens under acknowledged pre-event uncertainty, which is both more defensible to a risk committee and more responsive to the *actual* degree of uncertainty in any specific upcoming event."*

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
DAYS 0–30: FOUNDATION                  DAYS 30–60: FIRST SHIPS                DAYS 60–90: SCALE
────────────────────────────           ────────────────────────────           ────────────────────────
• Audit the 50+ problem statements     • Ship 1-2 highest ROI models          • Templatize: turn the first
  against: data availability,            end-to-end (research→prod)             model's pipeline into a
  expected Sharpe/cost impact,         • Establish CPCV/walk-forward            reusable scaffold — this
  engineering lift, business urgency     validation as the DEFAULT              IS "templatizing AI/ML
• Stand up point-in-time-correct         gate, not an afterthought              solutions across business
  feature store (prevents lookahead    • Build the champion/challenger          lines" from the JD
  bias across ALL future models)         model registry so iteration          • Cross-train other business
• Define the model-risk governance       doesn't require re-plumbing            lines (Delta One, FX, Rates)
  process WITH compliance/risk up      • Set up live-vs-backtest PnL            on the same infra
  front — retrofitting governance        drift monitoring                     • Establish the Gen AI stack
  later is 10x more expensive                                                   (RAG for research library,
                                                                                Claude Code for engineering
                                                                                velocity) as a force
                                                                                multiplier across ALL 10
                                                                                prioritized problem statements
```

**Prioritization scoring — a simple, defensible framework to present:**

$$\text{Priority Score} = w_1\cdot\hat{\text{Sharpe}} + w_2\cdot\text{Capacity} - w_3\cdot\text{EngLift} - w_4\cdot\text{DataRisk}$$

**Say it out loud:** *"I don't pick problems by gut feel — I score each of the 50 against expected Sharpe contribution, capacity/business impact, engineering lift required, and data-quality risk, calibrate the weights with the desk head, and defend the ranking transparently. This turns 'here's my opinion on what to build' into 'here's a reproducible framework the business can audit and adjust.'"*

### A Concrete Narrative to Have Ready — Connecting Your Own Background to "Zero to One"

**Feynman explanation of why a concrete story beats an abstract framework in the actual interview:** Rishi has built teams from scratch himself — he will listen for whether your 90-day framework is genuinely something you've lived, not something you constructed for this interview. The strongest answer pairs the abstract framework above with a specific, concrete example from your own career: *"At Millburn, the systematic macro research library didn't exist when I joined in that form — I helped build the validation infrastructure (CPCV, the same framework in Q18) that every one of the 30+ signals now passes through before going live. That wasn't a single model-building exercise; it was building the **scaffolding** that let 30+ subsequent models get validated consistently, which is structurally the exact same problem as this JD's mandate, just for a different cross-asset platform and with Gen AI as an additional capability layer that didn't exist in the same form when I started at Millburn."*

### Organizational Design — How the 10-15 Engineer Partnership Actually Works Day to Day

```
                     RESEARCH (You)                    ENGINEERING (10-15 team)
               ──────────────────────────────      ─────────────────────────────────
Owns:          Model logic, feature research,      Deployment infra, data pipelines,
               validation methodology, signal      monitoring/alerting, CI/CD,
               interpretation for PMs              scaling/latency, security/access
               ──────────────────────────────      ─────────────────────────────────
Interface:     A well-defined "model contract" — inputs (feature schema), outputs
               (signal schema + confidence), and SLAs (latency, refresh frequency) —
               NOT a shared codebase where both teams edit the same files freely

Weekly ritual: Joint review of the champion/challenger registry (Q28) — research
               proposes promotions, engineering validates production-readiness
               (the Q1 gate checklist) before any live routing change ships
```

**Say it out loud:** *"I've found the biggest failure mode in research/engineering partnerships isn't technical — it's an ambiguous contract boundary, where research keeps changing feature schemas without warning and engineering has to firefight, or engineering makes 'small' infra changes that silently alter model inputs. My first deliverable in month one wouldn't be a model — it would be that contract: a versioned, reviewed schema for how research hands off to engineering, because that single artifact is what lets a lean 10-15 person engineering team support 10 simultaneously-active problem statements without linear headcount growth per model."*

### The Gen AI Force-Multiplier, Made Concrete for This Specific Team

Tying back to the JD's explicit mention of Claude Code usage: **Say it out loud:** *"Given the team already uses Claude Code heavily, one of the highest-leverage early wins I'd pursue is standing up an internal 'research assistant' RAG layer (Q5) over the growing research library itself — as the team accumulates model documentation, validation reports, and postmortems across the 50+ problem statements, a well-built internal RAG system turns 'has anyone tried X before and what happened' from a Slack-archaeology exercise into a 10-second query with citations back to the original research note. This is a very deliberate, concrete example of 'templatizing AI/ML solutions across multiple business lines' — the RAG infrastructure itself becomes reusable across Equities, FX, Rates, and Futures research, not just Liquid Financing specifically."*

### The Single Hardest Trade-off You'll Be Asked to Defend

Expect a pointed follow-up along the lines of: *"You have 50 problem statements and limited engineering bandwidth — how do you say no to a senior PM who wants their pet project prioritized ahead of your framework's ranking?"* **Say it out loud, showing you've actually thought about the political, not just technical, dimension of this role:** *"I'd bring the scoring framework itself into that conversation rather than making it a personal judgment call — show the PM exactly where their proposal ranks and why (e.g., low estimated capacity, or high data-quality risk that would need to be resolved first), and explicitly ask what would need to be true for it to rank higher — maybe there's context I'm missing about strategic value that the framework doesn't capture, in which case I'd rather update the framework's weights transparently than override it quietly. The goal is for prioritization disagreements to become a conversation about the *framework's* assumptions, not a recurring negotiation that erodes trust in whether the ranking means anything at all."*

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

[🔝 Back to Top](#table-of-contents)
