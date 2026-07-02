# Presentation Talking Points & Structured Plan
### Barclays — AI/ML Modeler, Liquid Financing · Take-Home Discussion, Wed July 8 2026 · 5 QR/Traders, 60 min

> Built from the JD's explicit technical checklist (regression, tree models, time series, DL/RNN-LSTM-GRU,
> GenAI fine-tuning/prompting/RAG) plus recurring themes in recent Barclays QR/DS/ML-Engineer interview
> reports: heavy probability/stats grounding, resume-project deep-dives, Python coding fluency (pandas,
> mutable/immutable objects), model evaluation metrics, and a strong "can you actually ship this to
> production" bar. Sourced from Glassdoor and InterviewQuery guide patterns for Barclays QA/DS/MLE roles
> (2025–2026 reports) — see `LIQUID_FINANCING_TAKEHOME.md` for full technical detail per project.

---
---

[↩️ Back to README.md](./README.md)

---
---

## 1. Opening framing (60 seconds — say this before slide 1)

> "I scoped five projects directly against the roadmap language in the JD — 50+ identified problem
> statements, 10 actively prioritized. Rather than one deep dive, I picked five smaller, shippable
> pieces that together cover every modeling technique you listed: regularized regression, tree
> ensembles, time series, deep sequence models, and Gen AI/RAG — each tied to a specific financing
> P&L or risk lever, each with a validation and model-risk plan, not just a notebook."

This single sentence does three things a Barclays panel is listening for: (1) shows you read the JD
closely, (2) signals production-mindedness over "Kaggle-leaderboard" mindset, (3) sets up a
five-beat structure so the panel can steer time toward what they care about most.

---

## 2. Per-Project Talking Points (2 minutes each, then Q&A)

### P1 — Sec-Lending Fee Forecasting
- **Lead with the P&L hook:** "This directly prices new loans and flags re-pricing triggers before a squeeze."
- **Key technical point to volunteer:** the *non-linear threshold* at ~90% utilization — explain why
  Elastic Net alone can't express it and trees can. This shows you understand *when* to reach for
  which technique, not just that you know the names.
- **Anticipate the pushback:** "Why not just LSTM everything?" → Answer: model complexity must be earned;
  a tree ensemble on tabular utilization features beats a sequence model here because the signal is
  cross-sectional, not deeply sequential — reserve DL for P4 where it's actually justified.
- **Numbers to have ready:** weighted MAPE, directional hit-rate on 25bps+ fee changes, walk-forward not K-fold.

### P2 — Client Margin & Haircut Optimization
- **Lead with the risk hook:** "Every bps of over-margin is client attrition; every bps of under-margin
  is unhedged tail loss — this is a credit-risk objective wrapped in a regression problem."
- **Key technical point:** the GBM add-on isn't replacing the linear haircut, it's modeling the
  *shortfall residual* — the correlation-breakdown effect that rules-based grids miss in stress.
  This nuance (residual modeling vs. replacing the base model) reads as senior.
- **Anticipate the pushback:** "How do you keep this auditable?" → LASSO coefficient sign review +
  floor/cap clipping to the rules-based grid + Kupiec/Christoffersen backtests. Have these three
  words ready: **Kupiec, Christoffersen, floor/cap**.

### P3 — Funding-Spread Anomaly Detection
- **Lead with the surveillance hook:** "The tail risk isn't a bad forecast, it's not noticing a
  dislocation until it's already cost money."
- **Key technical point:** why Mahalanobis *and* Isolation Forest, not just one — MCD-robustified
  Mahalanobis assumes elliptical structure; Isolation Forest is the non-parametric cross-check for
  regime-dependent non-Gaussianity. Naming the *reason for redundancy* is a strong QR-panel signal.
- **This is your GenAI bridge project:** the NLP fusion term is a natural segue into P5 — use it to
  transition ("...and that same text-classification idea is what P5 generalizes into a full copilot").

### P4 — Prime Balance Forecasting (Deep Learning)
- **Lead with the capacity-planning hook:** "Balance sheet is the scarcest resource here — Treasury
  needs a forward view for RWA/LCR planning."
- **Key technical point — this is your strongest "I don't over-engineer" moment:** state explicitly
  that the GRU has to beat a tuned LightGBM baseline on walk-forward MAPE *before* you'd argue for
  the added production complexity of a PyTorch serving path. This single sentence answers the
  unspoken panel question "does this candidate reach for deep learning because it's exciting or
  because it's justified?"
- **Key technical point #2:** the explicit calendar embedding for quarter-end window-dressing — a
  *known* seasonal effect should be given to the model as a feature, not left for it to rediscover.
  This is a model-risk-friendly design choice; say so explicitly.

### P5 — RAG Financing-Desk Copilot (GenAI)
- **Lead with the JD-alignment hook:** "This directly answers the fine-tuning / prompt engineering /
  RAG / inference infrastructure line in the JD."
- **Key technical point:** the fine-tune vs. prompt-engineer trade-off table — walk through it live,
  don't just show it. State the staged rollout plan (prompt+RAG first to build eval harness and
  collect logs, fine-tune the narrow sub-tasks once ≥5k labeled queries exist).
- **Key technical point #2:** faithfulness metric and the explicit refusal behavior
  (`INSUFFICIENT_EVIDENCE`) — a Barclays QR panel will push hard on hallucination risk in a
  regulated environment; having a concrete, named metric and a designed refusal path pre-empts that.
- **BMC servers line from the JD:** mention the on-prem/BMC routing for non-public queries vs.
  external frontier API for public-market-context — shows you read the JD's infrastructure line, not
  just the modeling lines.

---

## 3. Anticipated Panel Questions (drawn from recent Barclays QR/DS/MLE interview report patterns)

```
THEME (from recent reports)              LIKELY QUESTION                              YOUR ANCHOR ANSWER
────────────────────────────────────    ────────────────────────────────────────    ─────────────────────────────
Probability/stats grounding              "Walk me through Kupiec/Christoffersen"      P2 backtesting section
Resume/project deep-dive                 "Why this architecture and not X?"           P4's "beat LightGBM first" line
ML fundamentals                          "Lasso vs Ridge, why here?"                  P1/P2 auditability argument
Python coding fluency                    Live code read / small coding exercise       Reference implementations, clean
Model evaluation metrics                 "How do you evaluate a forecasting model     Weighted MAPE, pinball loss,
                                          for a trading desk, not a Kaggle score?"     directional hit-rate framing
Production / deployment                  "How does this actually ship?"               Cross-Project Production Standards
                                                                                       section: walk-forward, PSI,
                                                                                       canary, shadow mode
GenAI specifics (JD-driven)              "Fine-tune vs RAG — when which?"              P5 trade-off table
Business alignment                       "How does this fit our 50+ item backlog?"     Opening framing + Conclusion
```

---

## 4. Structured 60-Minute Session Plan

```
MINUTE    SEGMENT                                    NOTES
───────   ────────────────────────────────────────   ──────────────────────────────────────────
0–2       Opening framing + roadmap tie-back          One slide: 5 projects mapped to JD bullets
2–12      P1 — Fee Forecasting                        2 min pitch, open to Q&A, expect drill-down
12–22     P2 — Margin/Haircut Optimization             Same cadence; expect Model Risk questions
22–30     P3 — Anomaly Detection                       Faster pace; bridges into P5
30–42     P4 — Balance Forecasting (DL)                 Slow down here — DL justification is the
                                                        highest-scrutiny moment of the session
42–54     P5 — RAG Copilot (GenAI)                      Most JD-aligned; expect the longest Q&A
54–60     Close: backlog fit + "what I'd build first"   End with a concrete 90-day plan, not a recap
```

## 5. Closing Line (have this memorized, not read)

> "If I joined this team, my first 90 days would be P1 and P3 — they're the fastest to a production
> signal and they share the same walk-forward validation harness, so I'd build that harness once and
> reuse it across the roadmap. P4 and P5 are quarter-two builds once that harness and the model-risk
> pack template exist. I'd rather ship two things well than five things as prototypes."

This closes on **sequencing and judgment**, which is what separates a VP-level hire from a strong
individual contributor in the panel's eyes — the JD says "the roadmap is built, what's needed is
someone to execute it," and this line directly answers that.

---

*Companion documents: `LIQUID_FINANCING_TAKEHOME.md` (full technical writeup), `Liquid_Financing_Takehome.ipynb`
(executable demo notebook), `dissertation.tex` / `dissertation.pdf` (extended research-presentation format).*
