# On-Policy Distillation (GKD) for Liquid Financing AI/ML — Applied Playbook

> Source paper: Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024. arXiv:2306.13649.
> Target role: VP, AI/ML Modeler — Liquid Financing (Equities/Delta One, Rate/Credit Financing, FX, Risk, Futures & Prime Derivatives). Stack: Python 3.13, C++26, `uv`, Google style, Plotly/Kaleido, GitHub Actions CI, nanobind, CPCV, Deflated Sharpe, HRP, Ledoit-Wolf.


---

## Table of Contents


- [1. Core Theory Recap](#1-core-theory-recap)
- [2. Why This Is *Directly* Relevant to the Liquid Financing Role](#2-why-this-is-directly-relevant-to-the-liquid-financing-role)
- [3. Recipe 1 — Distilling a Financing-Domain LLM Copilot for BMC Inference](#3-recipe-1--distilling-a-financing-domain-llm-copilot-for-bmc-inference)
- [4. Recipe 2 — On-Policy Distillation for Sequential/Numeric Models (Pricing, Time Series)](#4-recipe-2--on-policy-distillation-for-sequentialnumeric-models-pricing-time-series)
- [5. Recipe 3 — RL + On-Policy Distillation for Risk-Aware Copilots (Eq. 5 Fusion)](#5-recipe-3--rl--on-policy-distillation-for-risk-aware-copilots-eq-5-fusion)
- [6. Recipe 4 — Data-Efficiency Play for the "50+ Problem Statement Backlog"](#6-recipe-4--data-efficiency-play-for-the-50-problem-statement-backlog)
- [7. Recipe 5 — Self-Distillation for Continuous Weekly Refresh](#7-recipe-5--self-distillation-for-continuous-weekly-refresh)
- [8. Cost/Latency Model for BMC Deployment Decisions](#8-costlatency-model-for-bmc-deployment-decisions)
- [9. Evaluation Harness (CPCV-Aligned, Google-Style, Plotly Outputs)](#9-evaluation-harness-cpcv-aligned-google-style-plotly-outputs)
- [10. Interview Talking Points Mapped to JD Bullets](#10-interview-talking-points-mapped-to-jd-bullets)
- [11. Key Formulas Quick Reference](#11-key-formulas-quick-reference)
- [12. References](#12-references)
- [Prior Experience](#prior-experience)
- [How I intend to use for this seat?](#how-i-intend-to-use-for-this-seat)
- [How the implementation changes for a closed-source model like Claude vs Open Soure Model?](#how-the-implementation-changes-for-a-closed-source-model-like-claude-vs-open-soure-model)
- [Use-Cases](#use-cases)

[🔝 Back to Top](#table-of-contents)

---

## 1. Core Theory Recap

### 1.1 Three Ways to Teach a Model

```
┌────────────┬────────────────────┬────────────────────┬────────────────────┐
│            │        SFT         │         RL         │         OPD        │
├────────────┼────────────────────┼────────────────────┼────────────────────┤
│ who writes │  TEACHER writes    │  STUDENT writes    │  STUDENT writes    │
│            │  the solution      │  its own attempt   │  its own attempt   │
├────────────┼────────────────────┼────────────────────┼────────────────────┤
│ who grades │  STUDENT learns it │  ENVIRONMENT grades│  TEACHER grades    │
│            │  token by token    │  one scalar        │  every token       │
└────────────┴────────────────────┴────────────────────┴────────────────────┘
```

| | SFT | RL | On-Policy Distillation (OPD) |
|---|---|---|---|
| Text author | teacher | student | student |
| Grader | none (copy target) | environment (verifiable) | teacher model |
| Feedback granularity | full target text | 1 scalar / episode | dense, per-token |
| Analogy | copying worked solutions | exam, pass/fail | tutor correcting every step |
| Sample efficiency (Qwen3-8B ref.) | cheap, plateaus ~55.0 | 17,920 GPU-h → 67.6 | **1,800 GPU-h → 74.4** |

### 1.2 Objective

Given input $x$, teacher $p_T$, student $p_S^\theta$, token-level divergence at position $n$:

$$
D\big(p_T \| p_S^\theta\big)(y\mid x) \;=\; \frac{1}{L_y}\sum_{n=1}^{L_y} D\Big(p_T(\cdot\mid y_{<n},x)\,\big\|\,p_S^\theta(\cdot\mid y_{<n},x)\Big)
$$

**Generalized KD (GKD)** objective, mixing on-policy student rollouts with a fixed corpus via student-data fraction $\lambda\in[0,1]$:

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\big[D(p_T\|p_S^\theta)(y\mid x)\big] \;+\; \lambda\,\mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot\mid x)}\big[D(p_T\|p_S^\theta)(y\mid x)\big]\Big]
$$

No gradient flows through the sampling of $y \sim p_S(\cdot\mid x)$ (stop-gradient), mirroring policy-gradient stability tricks without the variance.

### 1.3 Divergence Family — Generalized JSD($\beta$)

$$
D_{JSD(\beta)}(P\|Q) = \beta\, D_{KL}\!\Big(P \,\Big\|\, \beta P + (1-\beta) Q\Big) + (1-\beta)\, D_{KL}\!\Big(Q \,\Big\|\, \beta P + (1-\beta) Q\Big)
$$

- $\beta \to 0$: behaves like forward KL $D_{KL}(P\|Q)$ → **mode-covering / mean-seeking** (student spreads mass to cover teacher support; risk of hallucinating on low-prob teacher tokens).
- $\beta \to 1$: behaves like reverse KL $D_{KL}(Q\|P)$ → **mode-seeking** (student concentrates on teacher's highest-probability modes; less diverse, more precise).

```
Forward KL (mean-seeking)        Reverse KL (mode-seeking)
        ___                              ___
       /   \  P                        /   \  P
   ___/     \___                   ___/     \___
  /    ______   \                 /   |          \
 |    /  Qθ  \   |   Qθ spreads   |   Qθ picks ONE mode, ignores rest
  \__/        \__/  across both        |  Qθ  |
       modes                            \_____/
```

### 1.4 On-Policy Loop (PyTorch skeleton, this paper's convention)

```python
# prompt: 2 + 3 × 4 = ?  — student wrote: = 5, then 5 × 4, = 20
rollout = student.sample(prompt)                          # [B, T]
s_lp = student(rollout).log_softmax(-1).gather(rollout)    # [B, T]
t_lp = teacher(rollout).log_softmax(-1).gather(rollout)    # [B, T]
rkl  = s_lp - t_lp                                         # [B, T]  (reverse-KL-style per-token surprisal)
loss = (rkl.detach() * s_lp).mean()                        # REINFORCE-style, stop-grad on rkl
loss.backward(); optimizer.step()
```

### 1.5 Decision Table (from paper's practical guidance)

| Your situation | Use |
|---|---|
| Strong teacher exists, shared tokenizer | **On-policy distillation** |
| Pushing past your best model | RL |
| Copying a teacher's behavior, no log-probs available | SFT |

[🔝 Back to Top](#table-of-contents)

---

## 2. Why This Is *Directly* Relevant to the Liquid Financing Role

The JD's core loop — **"50+ problem statements, 10 actively prioritized," ship-to-production, templatize across Equities/Delta One, Rate/Credit Financing, FX, Risk, Futures/Prime**, fine-tune LLMs, configure BMC inference servers, evaluate ML vs. DL vs. Gen AI trade-offs — maps onto GKD in five concrete ways:

1. **Model compression for production inference (BMC servers).** Liquid Financing needs low-latency, on-prem/BMC-hosted models. GKD lets you distill a large in-house or vendor LLM (teacher) into a small deployable student (7B→1–3B or smaller), preserving desk-specific reasoning quality while meeting latency/cost SLAs — precisely the "fine-tune LLMs... configure inference infrastructure (BMC servers)" responsibility.
2. **Templatized AI/ML solutions across business lines.** On-policy distillation is architecture-agnostic (any autoregressive or seq2seq head — including a pricing-model "reasoning" LLM, or a classifier reframed as constrained generation). One teacher (say, a firm-wide financing-risk LLM) can seed multiple lightweight desk-specific students (FX financing, Rates repo, Prime margining) — exactly the templatization mandate.
3. **Distribution mismatch = the same failure mode as backtest/production mismatch.** The paper's central insight — *train on the student's own rollouts, not a fixed static dataset* — is the LLM analogue of **CPCV / walk-forward validation**: avoid overfitting to a fixed historical path; instead, continually correct the model on the paths it *actually generates/encounters* (auto-regressive cascading error ≈ compounding P&L drift from a stale static-trained model).
4. **RL + Distillation fusion (Eq. 5) → RLHF-style alignment for internal copilots.** For any Gen-AI documentation/trade-explainability/compliance copilot the team ships, combining a factuality/entailment reward (financing docs must be textually grounded — same spirit as this paper's summarization RLAIF experiment) with on-policy distillation toward a trusted teacher directly prevents hallucination in financing/legal-adjacent text generation.
5. **Self-distillation for continuous model refresh.** Appendix A.1 shows a same-architecture teacher→student self-distillation surpasses the teacher. This is a cheap, low-risk pattern for **weekly/monthly model refresh** on the financing desk without a full retrain-from-scratch cycle — directly reduces GPU-hour cost of the "50+ problem statement" backlog.

```
                Liquid Financing GKD Deployment Map
┌───────────────────────────────────────────────────────────────────┐
│  FIRM-WIDE TEACHER LLM (large, high-latency, GPU cluster)         │
│      │  fine-tuned on financing corpora, entailment-checked       │
└──────┼───────────────────────────────────────────────────────────-┘
       │  on-policy distillation (JSD sweep per use-case)
   ┌───┴─────────┬─────────────┬─────────────┬─────────────┐
   ▼             ▼             ▼             ▼             ▼
 Equities/     Rate/Credit     FX            Risk         Futures/Prime
 Delta One     Financing       Financing     Explainer    Derivatives
 Student       Student         Student       Student      Student
 (BMC, <50ms)  (BMC, <50ms)    (BMC)         (BMC)         (BMC)
```

[🔝 Back to Top](#table-of-contents)

---

## 3. Recipe 1 — Distilling a Financing-Domain LLM Copilot for BMC Inference

**Use case:** A 30–70B teacher LLM (fine-tuned on financing agreements, repo/margin docs, ISDA/GMRA text, internal desk playbooks) must be compressed to a 3–8B student that fits BMC inference server memory/latency budgets while retaining desk-specific Q&A / summarization quality.

### 3.1 Pipeline

```
┌─────────────┐   sample    ┌───────────┐   teacher logits   ┌─────────────────┐
│  Prompts    │ ──────────▶ │  Student  │ ──────────────────▶│ Divergence Loss │
│  (financing │             │ rollout y │                    │  D(p_T‖p_S)      │
│   corpus)   │             └───────────┘                    └───────┬─────────┘
└─────────────┘                  ▲                                   │
                                 │            gradient (stop-grad on sampling)
                                 └───────────────────────────────────┘
```

### 3.2 Implementation

```python
"""On-policy distillation trainer for a financing-domain LLM student.

Google Python Style Guide compliant. Python 3.13, torch>=2.4.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass(frozen=True)
class GKDConfig:
    """Hyperparameters for Generalized Knowledge Distillation.

    Attributes:
        student_data_fraction: Lambda in [0, 1]; fraction of batches drawn
            on-policy from the student vs. from the fixed (prompt, target)
            corpus.
        jsd_beta: Beta in (0, 1) for generalized JSD; beta -> 0 recovers
            forward KL (mode-covering), beta -> 1 recovers reverse KL
            (mode-seeking). Use 0.9 for factual/compliance-sensitive text.
        teacher_temperature: Softmax temperature applied to teacher logits.
        student_temperature: Softmax temperature applied to student logits
            during on-policy sampling (1.0 encourages rollout diversity).
        max_new_tokens: Rollout length cap for on-policy sampling.
    """

    student_data_fraction: float = 1.0
    jsd_beta: float = 0.9
    teacher_temperature: float = 1.0
    student_temperature: float = 1.0
    max_new_tokens: int = 512


def generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Computes token-level generalized JSD(beta) between teacher/student.

    Args:
        teacher_logits: Shape [B, T, V], teacher next-token logits.
        student_logits: Shape [B, T, V], student next-token logits.
        beta: Interpolation coefficient; beta -> 0 is forward-KL-like,
            beta -> 1 is reverse-KL-like.

    Returns:
        Per-position JSD(beta) divergence, shape [B, T].
    """
    p = F.log_softmax(teacher_logits, dim=-1).exp()
    q = F.log_softmax(student_logits, dim=-1).exp()
    mix = beta * p + (1.0 - beta) * q
    log_mix = mix.clamp_min(1e-9).log()

    kl_p_mix = (p * (p.clamp_min(1e-9).log() - log_mix)).sum(-1)
    kl_q_mix = (q * (q.clamp_min(1e-9).log() - log_mix)).sum(-1)
    return beta * kl_p_mix + (1.0 - beta) * kl_q_mix


def gkd_step(
    student: nn.Module,
    teacher: nn.Module,
    prompts: torch.Tensor,
    targets: torch.Tensor | None,
    config: GKDConfig,
    rng: torch.Generator,
) -> torch.Tensor:
    """Executes one GKD update and returns the scalar loss (pre-backward).

    Args:
        student: Trainable autoregressive LM.
        teacher: Frozen, larger financing-domain LM (eval mode).
        prompts: Tokenized input prompts, shape [B, P].
        targets: Optional fixed-corpus target continuations, shape [B, L].
            Required when a supervised branch is sampled (u > lambda).
        config: GKDConfig hyperparameters.
        rng: Torch random generator for reproducible branch selection.

    Returns:
        Scalar loss tensor ready for `.backward()`.
    """
    use_on_policy = torch.rand((), generator=rng).item() <= config.student_data_fraction

    if use_on_policy:
        with torch.no_grad():
            rollout = student.generate(
                prompts,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.student_temperature,
            )
    else:
        if targets is None:
            raise ValueError("targets required when sampling supervised branch")
        rollout = torch.cat([prompts, targets], dim=1)

    student_logits = student(rollout).logits
    with torch.no_grad():
        teacher_logits = teacher(rollout).logits / config.teacher_temperature

    prompt_len = prompts.shape[1]
    divergence = generalized_jsd(
        teacher_logits[:, prompt_len - 1 : -1],
        student_logits[:, prompt_len - 1 : -1],
        beta=config.jsd_beta,
    )
    return divergence.mean()


def train_loop(
    student: nn.Module,
    teacher: nn.Module,
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    config: GKDConfig,
    steps: int,
    lr: float = 3e-4,
) -> list[float]:
    """Runs the GKD training loop and returns the loss history.

    Args:
        student: Trainable student model.
        teacher: Frozen teacher model.
        data_iter: Yields (prompts, targets) batches.
        config: GKDConfig hyperparameters.
        steps: Number of optimizer steps.
        lr: Adafactor/AdamW learning rate.

    Returns:
        List of per-step scalar loss values.
    """
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(0)
    losses: list[float] = []

    for step in range(steps):
        prompts, targets = next(data_iter)
        optimizer.zero_grad(set_to_none=True)
        loss = gkd_step(student, teacher, prompts, targets, config, rng)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
```

### 3.3 Divergence Selection Guide for Financing Text

| Task | Recommended $\beta$ | Rationale |
|---|---|---|
| Repo/GMRA clause summarization | 0.5–0.9 | Precision > diversity; hallucinated clauses are unacceptable |
| Trade-idea generation / research brainstorm | 0.1–0.3 | Diversity valuable; forward-KL covers more plausible continuations |
| Compliance/regulatory Q&A copilot | 0.9 (reverse-KL-like) | Mode-seeking → commits to the single correct/most-likely teacher answer |
| Instruction-tuned general desk assistant | reverse KL (per paper §4.4) | Outperformed forward KL by largest margin on held-out MMLU/BBH-style eval |

[🔝 Back to Top](#table-of-contents)

---

## 4. Recipe 2 — On-Policy Distillation for Sequential/Numeric Models (Pricing, Time Series)

The JD explicitly lists regression (OLS/LASSO/Elastic Net), tree models, RNN/LSTM/GRU, and time-series pricing. GKD's autoregressive framing generalizes beyond text: any model that emits a **sequence of tokens/states one step at a time** (multi-step price path forecasts, autoregressive vol surfaces, sequential hedge-ratio decisions) suffers the identical **train/inference distribution mismatch** (exposure bias) the paper addresses.

### 4.1 Reframing: Autoregressive Price-Path Forecasting as a Distillation Problem

Teacher: a large, expensive ensemble (e.g., deep LSTM/Transformer/GBM ensemble trained on repo rate curves, financing spreads). Student: a lightweight GRU/MLP deployable at tick-rate for real-time financing rate quoting.

$$
\text{Student rollout: } \hat{y}_{1:T} \sim p_S(\cdot \mid x), \qquad
\text{Teacher grades every step: } D\big(p_T(\cdot\mid \hat y_{<n}, x)\,\|\,p_S^\theta(\cdot\mid \hat y_{<n}, x)\big)
$$

Instead of training the student only on the ground-truth historical path (classic teacher-forcing == SFT-analogue, suffers cascading error at inference), roll the student forward on its **own predicted path**, and have the teacher (larger model or the true generative process / simulator) score each step's conditional distribution.

```python
"""On-policy distillation for autoregressive financing-rate forecasting."""
from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class SeqGKDConfig:
    """Config for sequential (non-text) on-policy distillation.

    Attributes:
        horizon: Number of autoregressive steps to roll out.
        student_data_fraction: Lambda; fraction of on-policy rollouts.
        divergence: 'gaussian_kl' (closed-form) for continuous targets.
    """

    horizon: int = 20
    student_data_fraction: float = 1.0
    divergence: str = "gaussian_kl"


def gaussian_kl(
    mu_t: torch.Tensor, sigma_t: torch.Tensor,
    mu_s: torch.Tensor, sigma_s: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL(teacher || student) between diagonal Gaussians.

    Args:
        mu_t: Teacher mean, shape [B, T, D].
        sigma_t: Teacher std, shape [B, T, D].
        mu_s: Student mean, shape [B, T, D].
        sigma_s: Student std, shape [B, T, D].

    Returns:
        Per-(batch, time) KL summed over D, shape [B, T].
    """
    var_t, var_s = sigma_t.pow(2), sigma_s.pow(2)
    term = (
        torch.log(sigma_s / sigma_t)
        + (var_t + (mu_t - mu_s).pow(2)) / (2.0 * var_s)
        - 0.5
    )
    return term.sum(-1)


def rollout_and_distill(
    student: nn.Module,
    teacher: nn.Module,
    x0: torch.Tensor,
    config: SeqGKDConfig,
) -> torch.Tensor:
    """Rolls the student forward autoregressively and distills at each step.

    Args:
        student: Emits (mu, sigma) per step given history.
        teacher: Frozen larger model, same interface.
        x0: Initial conditioning state (e.g., current financing curve),
            shape [B, D0].
        config: SeqGKDConfig.

    Returns:
        Scalar mean divergence loss over the rollout horizon.
    """
    history = [x0]
    losses = []
    for _ in range(config.horizon):
        h = torch.stack(history, dim=1)
        mu_s, sigma_s = student(h)
        with torch.no_grad():
            mu_t, sigma_t = teacher(h)
            next_state = mu_s.detach() + sigma_s.detach() * torch.randn_like(mu_s)
        step_kl = gaussian_kl(mu_t, sigma_t, mu_s, sigma_s)
        losses.append(step_kl.mean())
        history.append(next_state)
    return torch.stack(losses).mean()
```

### 4.2 Why This Beats Plain SFT/Teacher-Forcing for Financing Curves

```
Teacher-forcing (SFT-analogue)          On-policy distillation (OPD-analogue)
┌────────────────────────────┐         ┌─────────────────────────────┐
│ t=1: fed TRUE y1           │         │ t=1: student predicts ŷ1    │
│ t=2: fed TRUE y2           │         │ t=2: fed ŷ1, predicts ŷ2    │
│ t=3: fed TRUE y3           │         │ t=3: fed ŷ1,ŷ2 predicts ŷ3  │
│  ...                       │         │  ...                        │
│ INFERENCE: fed own ŷ's →   │         │ INFERENCE: same regime as   │
│  compounding drift NEVER   │         │  training → error correction│
│  seen in training          │         │  learned end-to-end         │
└────────────────────────────┘         └─────────────────────────────┘
```

This is the same rationale behind **CPCV over walk-forward**: eliminate a mismatch between the distribution the model is scored/deployed under and the distribution it was fit on.

[🔝 Back to Top](#table-of-contents)

---

## 5. Recipe 3 — RL + On-Policy Distillation for Risk-Aware Copilots (Eq. 5 Fusion)

**Use case:** A financing-risk explainability assistant must (a) maximize a verifiable reward (e.g., factual entailment against margin/collateral docs, or numerically-checked P&L attribution correctness) while (b) staying distilled toward a trusted, larger teacher to avoid capability loss ("alignment tax").

$$
\mathbb{E}_{x\sim X}\Big[(1-\alpha)\,\mathbb{E}_{y\sim p_S^\theta}\big[r(y)\big] \;-\; \alpha\,\mathbb{E}_{y\sim p_S}\big[D(p_T\|p_S^\theta)(y\mid x)\big]\Big]
$$

```python
"""RL (REINFORCE-style) + on-policy distillation fusion, Eq. 5."""
from __future__ import annotations

import torch
from torch import nn


def rl_plus_gkd_loss(
    student: nn.Module,
    teacher: nn.Module,
    reward_fn,
    prompts: torch.Tensor,
    alpha: float = 0.25,
    jsd_beta: float = 0.9,
) -> torch.Tensor:
    """Computes the fused RL + on-policy distillation loss (Eq. 5).

    Args:
        student: Trainable student policy.
        teacher: Frozen teacher for distillation regularization.
        reward_fn: Callable(rollout_ids) -> torch.Tensor of shape [B],
            e.g., an entailment-classifier reward against source docs.
        prompts: Shape [B, P].
        alpha: Weight on distillation term vs. RL reward term.
        jsd_beta: Generalized JSD beta for the distillation term.

    Returns:
        Scalar loss (negative of the objective, ready to minimize).
    """
    rollout = student.generate(prompts, do_sample=True, temperature=1.0)
    logits_s = student(rollout).logits
    with torch.no_grad():
        logits_t = teacher(rollout).logits
        reward = reward_fn(rollout)  # [B], e.g. entailment score in [0, 1]

    logp = torch.log_softmax(logits_s, -1).gather(
        -1, rollout.unsqueeze(-1)
    ).squeeze(-1).sum(-1)  # [B] sequence log-prob

    reward_term = (reward.detach() * logp).mean()  # REINFORCE surrogate

    p = torch.softmax(logits_t, -1)
    q = torch.softmax(logits_s, -1)
    mix = jsd_beta * p + (1 - jsd_beta) * q
    kl_pq = (p * (p.clamp_min(1e-9).log() - mix.clamp_min(1e-9).log())).sum(-1)
    kl_qq = (q * (q.clamp_min(1e-9).log() - mix.clamp_min(1e-9).log())).sum(-1)
    distill_term = (jsd_beta * kl_pq + (1 - jsd_beta) * kl_qq).mean()

    objective = (1 - alpha) * reward_term - alpha * distill_term
    return -objective
```

**Tuning guidance from paper's Figure 5 (RLAIF + OPD trade-off):** sweep $\alpha \in \{0.05, 0.1, 0.25, 0.5\}$ — lower $\alpha$ favors reward (factual consistency ↑ fastest at $\alpha{=}0.05$: +45% entailment), higher $\alpha$ favors summary/task quality. For financing risk copilots where factual grounding to legal/collateral text is non-negotiable, bias toward $\alpha \le 0.1$.

[🔝 Back to Top](#table-of-contents)

---

## 6. Recipe 4 — Data-Efficiency Play for the "50+ Problem Statement Backlog"

Paper's Figure 3 result: **on-policy GKD trained on only 5% of labeled data outperforms supervised KD/ImitKD trained on 100% of labeled data.** For a backlog of 50+ problem statements with limited desk-specific labeled data (financing spread quotes, margin call sequences), this directly reduces time-to-first-production-model:

$$
\text{On-policy GKD}_{5\%\text{ data}} \;>\; \text{Supervised KD}_{100\%\text{ data}}
$$

**Operational recipe:**
1. Cold-start: SFT a small student on whatever ground-truth labels exist per problem statement (even 0.5–5%).
2. Swap to on-policy rollouts against the firm's best available teacher (largest in-house model or ensemble) for the remaining training budget — $\lambda = 1$.
3. Track ROUGE/BLEU-analogues replaced by domain metrics: e.g., Deflated-Sharpe-weighted P&L accuracy, calibration error on spread forecasts, entailment score vs. source docs for text.

[🔝 Back to Top](#table-of-contents)

---

## 7. Recipe 5 — Self-Distillation for Continuous Weekly Refresh

Appendix A.1: same-architecture teacher→student self-distillation **surpasses the teacher**, with on-policy variants performing best.

**Applied pattern for a live financing model:**

```
Week N:  Model_N (deployed, "teacher")
              │  generate rollouts on latest week's live prompts/market states
              ▼
Week N+1: Model_{N+1} = self-distill(teacher=Model_N, student=copy(Model_N))
              │  on-policy JSD(0.9), small LR (0.0003), few thousand steps
              ▼
         Model_{N+1} deployed if eval gate passed (CPCV OOS check)
```

This is materially cheaper than full retraining and, per the paper, tends to *improve* over the prior checkpoint rather than merely replicate it — useful for a "10 actively prioritized problem statements" cadence where full retrains aren't affordable weekly.

[🔝 Back to Top](#table-of-contents)

---

## 8. Cost/Latency Model for BMC Deployment Decisions

Paper's compute-overhead figures (§A.2): on-policy sampling overhead vs. fixed-dataset sampling is **1.8×–2.2×** for student:teacher size ratios of 38×, 12×, 3.8× respectively — i.e., overhead *shrinks* as the student gets relatively larger, and is dominated by inference serving cost, not training cost, once deployed:

$$
\text{Overhead}(\rho) \approx f(\rho), \quad \rho = \frac{\text{teacher params}}{\text{student params}} \in \{38\times, 12\times, 3.8\times\} \mapsto \{1.8\times, 2\times, 2.2\times\}
$$

**Decision rule to present to the 10–15-person IT engineering team:** if it's too costly to sample on-policy from the student during fine-tuning, it will likely also be too costly to serve to end users at target QPS — so on-policy training cost should be evaluated *relative to serving cost*, not in isolation. This is a direct talking point for the "collaborate with IT engineering to support deployment" responsibility.

[🔝 Back to Top](#table-of-contents)

---

## 9. Evaluation Harness (CPCV-Aligned, Google-Style, Plotly Outputs)

```python
"""Evaluation harness for distilled financing models: CPCV + divergence
sweep reporting, Plotly HTML/PNG via Kaleido.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Container for one (divergence, student_data_fraction) cell result.

    Attributes:
        divergence: Divergence name, e.g. 'forward_kl', 'jsd_0.9'.
        student_data_fraction: Lambda used during training.
        metric_name: Domain metric, e.g. 'deflated_sharpe' or 'rouge2'.
        metric_value: Scalar OOS metric via CPCV folds.
    """

    divergence: str
    student_data_fraction: float
    metric_name: str
    metric_value: float


def build_ablation_heatmap(results: list[EvalResult]) -> go.Figure:
    """Builds a heatmap mirroring the paper's GKD ablation figures.

    Args:
        results: List of EvalResult across the divergence x lambda grid.

    Returns:
        Plotly Figure ready for `.write_image(...)` / `.write_html(...)`.
    """
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    pivot = df.pivot(
        index="student_data_fraction",
        columns="divergence",
        values="metric_value",
    ).sort_index(ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[f"{v:.0%}" for v in pivot.index],
            colorscale="Magma",
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="On-Policy Distillation Ablation — Financing Student Model",
        xaxis_title="Divergence D",
        yaxis_title="Student Data Fraction (λ)",
        template="plotly_dark",
    )
    return fig
```

[🔝 Back to Top](#table-of-contents)

---

## 10. Interview Talking Points Mapped to JD Bullets

| JD Requirement | GKD-Grounded Talking Point |
|---|---|
| "Evaluate and articulate trade-offs between modeling techniques" | Present the SFT vs RL vs OPD table (§1.1) with cost numbers (1,800 vs 17,920 GPU-h) as a template for technique selection memos. |
| "Fine-tune LLMs... configure inference infrastructure (BMC servers)" | Recipe 1: teacher→student compression pipeline sized to BMC memory/latency budget, with $\beta$ selection per text-risk profile. |
| "Implement neural network frameworks for pricing" | Recipe 2: autoregressive rollout-and-distill reframing of price-path forecasting to eliminate teacher-forcing/inference mismatch. |
| "Read and implement cutting-edge AI/ML research" | This document — end-to-end reproduction of ICLR 2024 GKD paper mapped to production financing use cases within days of publication-style triage. |
| "Templatizing AI/ML solutions across business lines" | §2 deployment map: one teacher, five desk-specific distilled students sharing the same training harness/config dataclass. |
| "50+ problem statements, 10 prioritized" | Recipe 4: 5%-labeled-data efficiency result directly compresses time-to-MVP per problem statement. |
| "Start-up mindset... ship to production" | Recipe 5: self-distillation as a low-cost weekly refresh loop instead of full retrains. |

[🔝 Back to Top](#table-of-contents)

---

## 11. Key Formulas Quick Reference

$$
\begin{aligned}
D_{KL}(P\|Q) &= \sum_{c} P(c)\log\frac{P(c)}{Q(c)} \quad \text{(forward KL, mean-seeking, used in SFT/MLE)} \\[4pt]
D_{JSD(\beta)}(P\|Q) &= \beta D_{KL}(P\|\beta P + (1-\beta)Q) + (1-\beta) D_{KL}(Q\|\beta P+(1-\beta)Q) \\[4pt]
\mathcal{L}_{\text{OD}}(\theta) &= \mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot|x)}\big[D_{KL}(p_T\|p_S^\theta)(y|x)\big]\Big] \quad \text{(on-policy KD, } \lambda{=}1\text{, fwd KL)}\\[4pt]
\mathcal{L}_{\text{GKD}}(\theta) &= (1-\lambda)\,\mathcal{L}_{\text{fixed-data}} + \lambda\,\mathcal{L}_{\text{on-policy}} \\[4pt]
\text{RL+OPD} &= \mathbb{E}\Big[(1-\alpha)\mathbb{E}_{y\sim p_S^\theta}[r(y)] - \alpha\,\mathbb{E}_{y\sim p_S}[D(p_T\|p_S^\theta)(y|x)]\Big]
\end{aligned}
$$

[🔝 Back to Top](#table-of-contents)

---

## 12. References

- Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., Bachem, O. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024. arXiv:2306.13649v3.
- Roit et al. 2023 — RLEF (entailment-reward RLAIF), used as the RL-reward pattern in §5.
- Job description: *AI / ML Modeler — Liquid Financing*, Global Financial Firm, NYC Midtown, VP level.

[🔝 Back to Top](#table-of-contents)

---

## Prior Experience

**"This connects to something I dealt with directly in my systematic macro work at Balyasny. We had signal and regime-classification models — HMM-based regime detection feeding into portfolio construction — where the classic failure mode was training the model on the true historical label sequence, but at inference it had to condition on its own prior predictions, since the true regime isn't known in real time. That's exactly the exposure-bias problem this paper formalizes: distribution mismatch between training-time conditioning and inference-time conditioning, and compounding error along the sequence.**

**My approach at the time was effectively an on-policy correction loop, though we didn't have this vocabulary for it: instead of purely teacher-forcing on the realized regime path, I'd roll the model forward on its own predicted state sequence during validation, and use a higher-capacity ensemble — essentially a 'teacher' — to re-score and correct those self-generated paths before feeding them back into training, particularly for the CPCV out-of-sample folds where the standard walk-forward approach was hiding this compounding-error effect. It's the same intuition as this paper's core contribution — train on the model's own rollouts and get dense feedback from a stronger reference model, rather than only fitting a fixed historical path — just applied to regime classification and portfolio signal generation instead of language generation.**

**What I find valuable about this paper is that it gives that intuition a rigorous, general framework — the divergence formulation, the data-efficiency results, the RL-fusion extension — which is exactly the kind of thing I'd want to formalize and reuse here for the sequential/autoregressive parts of the financing pipeline, rather than reinventing it ad hoc the way I had to back then."**

[🔝 Back to Top](#table-of-contents)

---

## How I intend to use for this seat?

**"One paper I've been digging into recently is DeepMind's 'On-Policy Distillation of Language Models' from ICLR 2024. It addresses a core problem in knowledge distillation: if you train a student model only on a fixed dataset — either ground-truth labels or teacher-generated text — the student sees a different distribution during training than it encounters at inference, because errors compound as it generates autoregressively. That's the same exposure-bias problem you get in any sequential model.**

**Their fix — Generalized Knowledge Distillation, or GKD — has the student generate its own rollouts during training, and then has the teacher grade every token of that self-generated output, rather than just scoring a fixed target. It's a middle ground between supervised fine-tuning, where the teacher writes and the student copies, and RL, where the student writes and gets one scalar reward per episode. Here the student writes, but gets dense, token-level feedback from the teacher. Empirically it's dramatically more compute-efficient — on their Qwen3-8B benchmark, on-policy distillation hit a higher score in about 1,800 GPU-hours versus roughly 18,000 for RL to reach a lower score. They also show it's far more data-efficient: on-policy distillation on 5% of labeled data outperformed supervised KD trained on the full dataset.**

**For this seat, I see three direct applications. First, for the BMC (Baseboard Management Controller) inference infrastructure — compressing a large in-house financing LLM down to something deployable at low latency, while preserving desk-specific reasoning, using on-policy distillation instead of standard fine-tuning. Second, for the pricing and time-series models — the same distribution-mismatch problem shows up in autoregressive forecasting, like multi-step financing curve or spread forecasts, where teacher-forcing during training doesn't match how the model actually gets used at inference. The paper's on-policy framing generalizes directly to that. And third, given the backlog — 50+ problem statements, 10 prioritized — the data efficiency result matters practically: it means we can stand up production-quality models for new use cases with a fraction of the labeled data and compute we'd otherwise need."**

[🔝 Back to Top](#table-of-contents)

---

## How the implementation changes for a closed-source model like Claude vs Open Soure Model?

Good question — it changes the architecture meaningfully, because you don't have access to Claude's weights or full output distribution, which the paper's method assumes.

[🔝 Back to Top](#table-of-contents)

---

### Background
* **SeqKD** (**Seq**uence-Level **K**nowledge **D**istillation):
* GKD (Generalized Knowledge Distillation) 

Both, are techniques in machine learning used to compress large, complex teacher models into smaller, faster student models for natural language processing and generation tasks.

The primary difference lies in their sampling policy: 

SeqKD is an off-policy method that trains the student on static text generated by the teacher, while GKD is an on-policy method that trains the student on sequences it generates itself.


#### 1. SeqKD (Sequence-Level Knowledge Distillation)
Introduced by Kim and Rush (2016), [SeqKD](https://arxiv.org/abs/1606.07947) moves beyond traditional token-by-token logit matching.

* How it works: The teacher model processes the training input and runs a decoding strategy (like beam search) to generate full target text sequences. The student model is then fine-tuned using standard supervised learning on this static, teacher-generated dataset.
* The Flaw: It is inherently off-policy. Because the student only trains on "perfect" sequences from the teacher, it never learns how to recover when it makes a mistake during real-world inference. This gap is called exposure bias.

#### 2. GKD (Generalized Knowledge Distillation)
Introduced by Google DeepMind (Agarwal et al., 2023), [GKD](https://arxiv.org/abs/2306.13649) was explicitly designed to fix the exposure bias of off-policy distillation.

* How it works: During training, the student model generates its own token sequences (on-policy rollouts). Even if the student generates a flawed or nonsensical sequence, the teacher evaluates those exact token contexts step-by-step and tells the student what the correct next-token probabilities should have been.
* Handling Under-Specification: Beyond fixing exposure bias, GKD allows for alternate divergence metrics. If a small student lacks the physical capacity to mirror a massive teacher's entire broad distribution, GKD can optimize for Reverse KL divergence. This forces the student to focus strictly on a single high-probability "mode" (outputting safe, accurate text) rather than failing to learn everything.

#### Core Comparison

| Feature | SeqKD (Sequence-Level KD) | GKD (Generalized KD) |
|---|---|---|
| Policy Type | Off-policy (student learns from teacher's paths) | On-policy (student learns from its own paths) |
| Data Source | Fixed text sequences pre-generated by the teacher | Dynamic text sequences generated by the student at runtime |
| Primary Loss | Cross-entropy against teacher's token sequences | Forward KL, Reverse KL, or Jensen-Shannon Divergence |
| Exposure Bias | Suffers from it; cannot correct its own compounding mistakes | Mitigates it; teacher corrects student on its flawed context |
| Compute Overhead | Low (data is generated once, offline) | High (requires ongoing generation from student & teacher evaluations) |


#### Summary of Impact
Use SeqKD if you have strict budget constraints and want to quickly distill a model offline without handling expensive 
generation loops during training. Choose GKD (available via tools like the 
[Hugging Face TRL GKD Trainer](https://huggingface.co/docs/trl/en/gkd_trainer)) if you 
are distilling an LLM for complex tasks like reasoning, translation, or mathematical formatting, 
where long-sequence errors tend to compound heavily.

[🔝 Back to Top](#table-of-contents)

--- 

**What breaks:**
The core GKD loss requires token-level KL/JSD between teacher and student logit distributions — that needs the teacher's full softmax over the vocabulary at every position. Claude's API gives you sampled text and, at most, top-k logprobs per token, not the full distribution needed for the exact divergence computation in the paper.

**How I'd adapt the implementation:**

1. **Use Claude as the teacher via SeqKD, not full on-policy GKD.** Sample rollouts from my open-weight student, send those same rollouts to Claude to get a *judged/corrected* version or a scalar quality grade per token or per span (via structured critique prompts), and train the student to match Claude's corrections — closer to the paper's sequence-level KD baseline, but using Claude's judgment as a proxy grader rather than true teacher logits.

2. **Approximate token-level feedback with available logprobs where the API exposes them.** If top-logprobs are available, I'd compute a partial KL restricted to that top-k set, plus a fallback penalty for mass outside it — a practical approximation, not the exact JSD(β) formulation, but directionally the same on-policy correction idea: grade the student's own rollouts rather than a fixed dataset.

3. **Shift the RL+distillation fusion (§5) toward Claude-as-reward-model.** Instead of an entailment classifier, use Claude itself as the grader for factuality/compliance against financing source docs — that's a natural fit since Claude is already in the stack, and it turns the distillation signal into an RLAIF-style reward Claude produces on the student's own generations.

4. **Keep the open-weight student.** The whole point is still a small, BMC-deployable model — Claude stays server-side as the periodic teacher/grader during training, never in the low-latency inference path.

So practically: same on-policy philosophy — train on the student's own outputs, get dense correction from a stronger model — but implemented as **Claude-graded on-policy SFT / RLAIF**, not literal token-level GKD, because the API surface doesn't expose what the paper's method needs.

[🔝 Back to Top](#table-of-contents)

---

## Use-Cases

Yes — if in-house LLMs already exist and BMC servers are the deployment target, this is exactly the setting the paper assumes: you control the full forward pass of both teacher and student, so the exact GKD algorithm (token-level KL/JSD, on-policy rollouts, stop-gradient sampling) is directly implementable, not an approximation. Here's how I'd frame it and the concrete use cases.

**Why it fits cleanly:**
- In-house teacher LLM → full logit access, no API bottleneck.
- BMC servers → the latency/memory-constrained deployment target that *motivates* compression in the first place.
- Same firm infra → training and serving both stay inside the perimeter, satisfying the data-governance constraint from before.

**Concrete use cases at this seat:**

1. **BMC-deployable desk copilots (Recipe 1).** Distill the large in-house LLM into a 3–8B student per business line — Equities/Delta One, Rate/Credit Financing, FX, Risk, Prime — each fine-tuned on desk-specific documents and financing terminology. On-policy GKD lets each student hit near-teacher quality at a fraction of inference cost, which is the direct "templatize AI/ML solutions across business lines" mandate in the JD.

2. **Low-latency trade/margin explainability.** Real-time margin call or repo-rate explanations need sub-second responses on BMC hardware. A distilled student trained on-policy against the in-house teacher, with JSD(β) tuned toward mode-seeking (β≈0.9), gives commitment to the single correct/most-defensible answer rather than diverse hedged phrasing — important when compliance reviews the output.

3. **RL + distillation fusion for factual grounding (§5, Eq. 5).** Combine an internal entailment/consistency reward (checking generated text against source financing docs) with on-policy distillation toward the teacher. This directly reduces hallucination risk in anything client- or compliance-facing, while the teacher-regularization term prevents the "alignment tax" of pure RL fine-tuning.

4. **Continuous refresh without full retrain (Recipe 5 — self-distillation).** Weekly/monthly, treat last week's deployed student as the new teacher and self-distill an updated checkpoint on-policy against fresh market/financing data. Cheap relative to full retraining, and the paper shows this can *surpass* the prior checkpoint rather than just replicate it — useful for keeping 10+ prioritized problem statements current without a full MLOps retrain cycle each time.

5. **Data-efficient bootstrapping of new problem statements (Recipe 4).** For any of the 50+ backlog items with limited labeled data, use the in-house teacher to bootstrap a student via on-policy distillation rather than waiting for a large labeled dataset — the paper shows 5% labeled data with on-policy GKD beats 100% with standard supervised KD.

6. **Autoregressive pricing/curve forecasting (Recipe 2), non-text.** If the in-house LLM infrastructure extends to autoregressive numeric heads (e.g., LLM-adjacent sequence models for financing curve or spread pricing), the same rollout-and-distill idea applies: train the small production model on its own predicted path, scored by the larger teacher, to eliminate the teacher-forcing/inference mismatch.

**One important scoping note for the interview:** I'd flag that step 1 in practice — before any of this — is confirming the in-house teacher's serving stack exposes full logits (not just top-k), since that determines whether you implement exact JSD(β) or fall back to a top-k-restricted approximation. That's a specific, concrete technical question worth asking the interviewer directly — it shows you're already thinking about implementation constraints, not just theory.

[🔝 Back to Top](#table-of-contents)

---