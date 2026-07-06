# Presentation Talking Points & Structured Plan
### Barclays — AI/ML Modeler, Liquid Financing · Take-Home Discussion, Wed July 8 2026 · 5 QR/Traders, 60 min

> Built from the JD's explicit technical checklist (regression, tree models, time series, DL/RNN-LSTM-GRU,
> GenAI fine-tuning/prompting/RAG) plus recurring themes in recent Barclays QR/DS/ML-Engineer interview
> reports: heavy probability/stats grounding, resume-project deep-dives, Python coding fluency (pandas,
> mutable/immutable objects), model evaluation metrics, and a strong "can you actually ship this to
> production" bar. Sourced from Glassdoor and InterviewQuery guide patterns for Barclays QA/DS/MLE roles
> (2025–2026 reports) — see `LIQUID_FINANCING_TAKEHOME.md` for full technical detail per project.

---

## Table of Contents

- [1. Opening framing (60 seconds — say this before slide 1)](#1-opening-framing-60-seconds--say-this-before-slide-1)
- [2. Per-Project Talking Points (2 minutes each, then Q&A)](#2-per-project-talking-points-2-minutes-each-then-qa)
  - [P1 — Sec-Lending Fee Forecasting](#p1--sec-lending-fee-forecasting)
  - [P2 — Client Margin & Haircut Optimization](#p2--client-margin--haircut-optimization)
  - [P3 — Funding-Spread Anomaly Detection](#p3--funding-spread-anomaly-detection)
  - [P4 — Prime Balance Forecasting (Deep Learning)](#p4--prime-balance-forecasting-deep-learning)
  - [P5 — RAG Financing-Desk Copilot (GenAI)](#p5--rag-financing-desk-copilot-genai)
- [3. Anticipated Panel Questions (drawn from recent Barclays QR/DS/MLE interview report patterns)](#3-anticipated-panel-questions-drawn-from-recent-barclays-qrdsmle-interview-report-patterns)
- [4. Structured 60-Minute Session Plan](#4-structured-60-minute-session-plan)
- [5. Whiteboard-Ready Deep-Dive Q&A (Full Derivations, Not Bullet Points)](#5-whiteboard-ready-deep-dive-qa-full-derivations-not-bullet-points)
  - [5.1 P1 Deep-Dive — "Why does Elastic Net alone fail at the 90% utilization kink?"](#51-p1-deep-dive--why-does-elastic-net-alone-fail-at-the-90-utilization-kink)
  - [5.2 P1 Deep-Dive — "Derive the pinball loss and tell me why q=0.9 penalizes under-prediction 9x more."](#52-p1-deep-dive--derive-the-pinball-loss-and-tell-me-why-q09-penalizes-under-prediction-9x-more)
  - [5.3 P2 Deep-Dive — "Prove the Kupiec LR statistic is asymptotically chi-squared(1)."](#53-p2-deep-dive--prove-the-kupiec-lr-statistic-is-asymptotically-chi-squared1)
  - [5.4 P2 Deep-Dive — "Why not just estimate the full covariance matrix and do proper mean-variance VaR?"](#54-p2-deep-dive--why-not-just-estimate-the-full-covariance-matrix-and-do-proper-mean-variance-var)
  - [5.5 P3 Deep-Dive — "Derive why MCD uses the subset with smallest covariance determinant."](#55-p3-deep-dive--derive-why-mcd-uses-the-subset-with-smallest-covariance-determinant)
  - [5.6 P4 Deep-Dive — "If I asked you to add attention to this GRU, where would it go and why?"](#56-p4-deep-dive--if-i-asked-you-to-add-attention-to-this-gru-where-would-it-go-and-why)
  - [5.7 P5 Deep-Dive — "Formally, why is RRF better than min-max normalizing BM25 and cosine scores and averaging?"](#57-p5-deep-dive--formally-why-is-rrf-better-than-min-max-normalizing-bm25-and-cosine-scores-and-averaging)
- [6. One-Slide Architecture Recap Per Project (Use as Literal Slide Content)](#6-one-slide-architecture-recap-per-project-use-as-literal-slide-content)
- [7. Extended Anticipated-Questions Matrix (Question → Full Answer Pointer)](#7-extended-anticipated-questions-matrix-question--full-answer-pointer)

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

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

[🔝 Back to Top](#table-of-contents)

---

## 5. Whiteboard-Ready Deep-Dive Q&A (Full Derivations, Not Bullet Points)

> Use this section when a panelist says "walk me through the math" — each entry is written the way you'd
> actually talk while writing on a whiteboard: equation, then what each symbol means, then why it's the
> right tool. Organized by project so you can flip directly to the one being probed.

### 5.1 P1 Deep-Dive — "Why does Elastic Net alone fail at the 90% utilization kink?"

**Whiteboard sequence:**

1. Write the Elastic Net objective: $\hat\beta = \arg\min_\beta \frac{1}{2n}\lVert y-X\beta\rVert_2^2 + \lambda(\alpha\lVert\beta\rVert_1 + \frac{1-\alpha}{2}\lVert\beta\rVert_2^2)$.
2. Say: "This is fundamentally a **linear-in-parameters** model — no matter how I regularize $\beta$, the
   prediction surface $X\beta$ is a hyperplane. A kink at utilization=0.9 requires the prediction
   function's *slope* to change discontinuously at that point — a single linear model cannot do that
   without an engineered feature."
3. Point to `util_squared_over_90 = max(utilization - 0.9, 0)^2` in the feature list — "I *did* engineer
   a hinge feature to partially address this — this is a spline-basis trick, essentially a manual
   piecewise-linear-plus-quadratic basis expansion. It helps, but it commits me to a specific functional
   form (quadratic above the knot) chosen in advance."
4. Contrast with the tree: "A single decision-tree split at utilization≈0.9 gets the discontinuity for
   free, with zero functional-form assumption — and LightGBM's iterative boosting refines the shape above
   and below the knot separately, additively. That's the concrete 'why trees here' answer, not just
   'trees are more flexible' as a vague appeal."
5. Close the loop: "The NNLS blend is exactly the mechanism that lets the *data itself* decide, per fold,
   how much to trust each functional form — which is a more defensible design than me hand-picking one."

### 5.2 P1 Deep-Dive — "Derive the pinball loss and tell me why q=0.9 penalizes under-prediction 9x more."

$$\rho_q(y,\hat y) = \begin{cases} q(y-\hat y) & y \geq \hat y \\ (1-q)(\hat y - y) & y < \hat y\end{cases} = \max\big(q(y-\hat y),\ (q-1)(y-\hat y)\big)$$

Say: "At $q=0.9$: if I under-predict ($y>\hat y$), the penalty is $0.9\times(\text{error})$; if I
over-predict ($y<\hat y$), the penalty is $0.1\times(\text{error})$ — a 9:1 ratio, exactly $q/(1-q)$.
This asymmetry is *why* minimizing expected pinball loss at $q$ yields the true $q$-th conditional
quantile as the population minimizer — differentiate $\mathbb{E}[\rho_q(y,\hat y)]$ with respect to
$\hat y$ and set to zero: $-q\cdot P(y\geq\hat y) + (1-q)\cdot P(y<\hat y) = 0 \Rightarrow P(y<\hat y)=q$
— that's the definition of the $q$-th quantile. This is the one-line proof I'd write if asked to justify
why pinball loss targets a quantile rather than a mean."

### 5.3 P2 Deep-Dive — "Prove the Kupiec LR statistic is asymptotically chi-squared(1)."

Say: "This follows directly from Wilks' theorem — under $H_0$ (the true breach probability equals the
target rate $p$), the log-likelihood ratio $-2\ln(L_0/L_1)$ between a *restricted* model (probability
fixed at $p$) and the *unrestricted* MLE ($\hat p = x/n$) converges in distribution to $\chi^2_{df}$,
where $df$ is the number of restrictions — here exactly 1 (we've restricted one free parameter, the
breach probability, to a fixed value). I wouldn't re-derive Wilks' theorem from scratch on a whiteboard,
but I'd want to be able to name it and state the degrees-of-freedom logic, because 'it just is chi-squared,
trust me' is a weak answer to a QR panel that will have this exact theorem in their own toolkit."

### 5.4 P2 Deep-Dive — "Why not just estimate the full covariance matrix and do proper mean-variance VaR?"

Say: "For a large collateral universe, the full covariance matrix has $O(k^2)$ free parameters — with,
say, 500 distinct asset buckets, that's ~125,000 parameters to estimate from a comparatively short
historical window, which is a severe curse-of-dimensionality problem: the sample covariance matrix becomes
poorly conditioned or singular long before you have enough data to estimate it reliably (this is precisely
the same $X^\top X$ ill-conditioning problem that motivates Ridge/Elastic Net in the regression context —
covariance estimation has an exactly analogous shrinkage literature, e.g. Ledoit-Wolf shrinkage). The
two-layer design sidesteps this: the linear per-asset layer only needs $O(k)$ parameters (one haircut per
asset, from *univariate* features), and the correlation/concentration structure is captured *implicitly*
through a much lower-dimensional set of portfolio-level summary features (concentration, sector exposure)
rather than the full $k\times k$ covariance matrix — a deliberate dimensionality reduction that trades a
small amount of theoretical completeness for a model that's actually estimable and auditable at the
portfolio sizes this desk deals with."

### 5.5 P3 Deep-Dive — "Derive why MCD uses the subset with smallest covariance determinant."

Say: "The determinant of a covariance matrix is proportional to the squared volume of the confidence
ellipsoid it defines — $\det(\Sigma) \propto (\text{generalized variance})$. If I have contaminating
outliers in my sample, the subset that includes them will have inflated variance in whatever direction
the outliers pull, and hence a larger determinant, than a 'clean' subset. MCD's insight is to explicitly
search over subsets of size $h$ (typically 50-75% of $n$) and pick the one with the *smallest* determinant
— that's the tightest, most internally-consistent cluster of points, which by construction is the subset
least likely to include the true outliers. It's a combinatorial search in principle, made tractable by the
FAST-MCD algorithm's iterative 'concentration steps' (swap points in/out and re-check the determinant),
which is what scikit-learn's `MinCovDet` actually runs under the hood rather than brute-force enumeration."

### 5.6 P4 Deep-Dive — "If I asked you to add attention to this GRU, where would it go and why?"

Say: "Right now the decoder's hidden state $h_t$ has to compress the *entire* 60-day encoder history into
a single fixed-size vector $h_n$ passed once at the start of decoding — this is the classic
encoder-decoder bottleneck. I'd add an attention layer (per Q26 of my interview prep) that lets each
decoder step compute a fresh, query-specific weighted combination over *all* encoder hidden states, not
just the final one: $\text{context}_t = \sum_s \alpha_{ts} h_s^{\text{enc}}$, with $\alpha_{ts}$ computed
via scaled dot-product attention between the decoder's current state and every encoder step. Concretely,
this would let the model directly attend back to, say, the *previous* quarter-end 60 trading days ago
without that information having to survive being compressed through 60 sequential GRU updates — I'd only
add this complexity, though, after confirming (same baseline discipline as the GRU vs. LightGBM comparison)
that it measurably improves walk-forward pinball loss over the simpler encoder-decoder, since attention
roughly doubles the parameter count and inference latency for what may be a modest gain on a series this
short."

### 5.7 P5 Deep-Dive — "Formally, why is RRF better than min-max normalizing BM25 and cosine scores and averaging?"

Say: "Min-max normalization assumes the *distribution* of scores in the current query's result set is
representative of the scale you should calibrate against — but BM25 scores for a query with many rare,
highly-discriminative terms can be an order of magnitude larger than for a query with only common terms,
and that variation has nothing to do with genuine relevance differences. If I min-max normalize per-query,
I'm implicitly assuming the *best* match for every query deserves the same normalized score of 1.0,
regardless of how strong that match actually is in absolute terms — this can make a weak best-match look
identical to a strong one after normalization. RRF sidesteps this entirely by using **only ordinal rank
information**, which is invariant to whatever arbitrary scale each ranker happens to produce — it's a
non-parametric fusion method in the same sense that Isolation Forest (P3) is a non-parametric anomaly
detector: no distributional assumption to get wrong."

[🔝 Back to Top](#table-of-contents)

---

> "If I joined this team, my first 90 days would be P1 and P3 — they're the fastest to a production
> signal and they share the same walk-forward validation harness, so I'd build that harness once and
> reuse it across the roadmap. P4 and P5 are quarter-two builds once that harness and the model-risk
> pack template exist. I'd rather ship two things well than five things as prototypes."

This closes on **sequencing and judgment**, which is what separates a VP-level hire from a strong
individual contributor in the panel's eyes — the JD says "the roadmap is built, what's needed is
someone to execute it," and this line directly answers that.

[🔝 Back to Top](#table-of-contents)

---

## 6. One-Slide Architecture Recap Per Project (Use as Literal Slide Content)

> These are deliberately terse — meant to be read in under 10 seconds while you talk over them, not
> read by the panel. Full architecture detail lives in `README.md` §3.10 if anyone wants to go deeper.

```
P1 — FEE FORECASTING            P2 — MARGIN OPTIMIZATION        P3 — ANOMALY DETECTION
Elastic Net ─┐                  LASSO haircut ──┐                Mahalanobis (MCD) ─┐
              ├─ NNLS blend       (auditable)      │                                    ├─ fuse_scores
LightGBM Q10/50/90┘              GBM add-on ───────┴─ IM = linear   Isolation Forest ──┘        │
    │                             + addon, floor/cap-clipped        TF-IDF text classifier ─────┘
    ▼                                    │                                    │
Split-Conformal (CQR)              Kupiec + Christoffersen              Neyman-Pearson threshold
    │                                    │                                    │
80% guaranteed-coverage band       backtest + traffic-light            ranked AlertRecord + driver
```

```
P4 — BALANCE FORECASTING (DL)                    P5 — RAG COPILOT (GenAI)
naive → SARIMAX → LightGBM → GRU seq2seq          BM25 ─┐
(baseline ladder, DL must earn its spot)                 ├─ RRF ─ retrieve top-k
GRU: encoder(60d history) + decoder                TF-IDF cosine ┘        │
  (calendar-embedding-conditioned, 3 quantile heads)                       ▼
      │                                            max_dense_similarity < 0.12?
  P10/P50/P90 fan chart, quarter-end marked          │             │
                                                     YES            NO
                                                INSUFFICIENT_   synthesize +
                                                EVIDENCE          cite [doc_id]
```

[🔝 Back to Top](#table-of-contents)

---

## 7. Extended Anticipated-Questions Matrix (Question → Full Answer Pointer)

```
QUESTION CATEGORY                  SPECIFIC LIKELY QUESTION                          FULL ANSWER LOCATION
──────────────────────────────    ──────────────────────────────────────────────    ──────────────────────
Statistics fundamentals            "Derive the pinball loss / prove it targets       §5.2 above
                                    a quantile"
Statistics fundamentals             "Prove Kupiec's LR stat is chi-squared(1)"        §5.3 above
Model-risk / governance             "Why is the correlation add-on fit on the         README §3.10.2,
                                    residual, not as a second full model?"            "Why the correlation
                                                                                       add-on is fit on
                                                                                       the residual"
Model selection judgment            "Why not just use the GBM everywhere,             §5.1 above
                                    it's more flexible?"
Model selection judgment            "Why not full covariance / proper mean-variance   §5.4 above
                                    VaR instead of the two-layer haircut design?"
Statistical/robust methods          "Derive why MCD picks the smallest-determinant    §5.5 above
                                    subset"
Deep learning extension             "Where would attention go in the P4 GRU?"          §5.6 above
Information retrieval theory        "Why RRF over min-max normalize + average?"        §5.7 above
Production engineering              "Walk me through why you need THREE splits          README §3.10.1,
                                    (train/calib/test), not two, for P1"                "Why three data
                                                                                          splits"
GenAI specifics                     "How would you actually calibrate the 0.12          README §3.10.5,
                                    refusal threshold in production?"                    "Say it out loud,
                                                                                          anticipating..."
GenAI specifics                     "Isn't the faithfulness proxy just measuring         README §3.10.5,
                                    word overlap — what does it actually miss?"          "Faithfulness proxy
                                                                                          vs. true LLM..."
Business/strategic                  "Which of these 5 would you build first if           §5 Closing Line +
                                    this were real, and why?"                            Opening framing §1
```

[🔝 Back to Top](#table-of-contents)

---

*Companion documents: `README.md` §3.10–3.11 (full per-project architecture diagrams, extended
derivations, and JD-requirement mapping table), `LIQUID_FINANCING_TAKEHOME.md` (full technical writeup),
`Liquid_Financing_Takehome.ipynb` (executable demo notebook), `dissertation.tex` / `dissertation.pdf`
(extended research-presentation format).*

[🔝 Back to Top](#table-of-contents)

---
