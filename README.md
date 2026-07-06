# Barclays — AI / ML Modeler, Liquid Financing — Technical Interview Playbook
### Panel: Rishi Dhingra (MD, Global Markets — Electronification, eTrading & AI/ML | Prime Services, Financing & Delta One) · HLS Trading Co-Founders
#### 30 Questions × 7 Domains · Wednesday, July 8 2026 · 1-Hour Technical Round

> **Delivery philosophy:** every answer runs *intuition → math derived line-by-line (verified) → ASCII diagram → Feynman restatement → production-grade Python 3.13*. Code blocks carry a Google-style module docstring header, full class/constructor/method docstrings (Args/Returns), a `if __name__ == "__main__":` block, and — wherever a chart clarifies the concept — a Plotly figure persisted to both interactive HTML and static PNG under `plots/`.

---

## 📚 Sources Mined (role-specific, most-recent-first)

| # | Source | What it confirmed | Applied to |
|---|---|---|---|
| 1 | Barclays careers site / Built In NYC — *"AVP – Global Markets AI/ML Modeller"* posting | Verbatim JD language: "building an AI/ML capability from the ground up," "start-up mindset," "PyTorch or TensorFlow," "deploy scalable ML models in production," "shape strategic direction of AI adoption across Markets" | Q1, Q3, Q30 framing |
| 2 | Glassdoor — Barclays AVP Global Markets AI/ML Modeller listing (NYC, $125K–$175K band, reports to Global Markets) | Core requirements: PyTorch/TensorFlow deep-learning experience, production deployment, cloud, statistics/applied-math foundation | Q1, Q20–Q27 weighting |
| 3 | InterviewQuery — *Barclays Machine Learning Engineer Interview Guide* (2026) | Recurring format: supervised-vs-unsupervised fundamentals, "walk me through a project end-to-end," data-quality/validation probing, model evaluation & optimization follow-ups | Q1, Q3, Q14 |
| 4 | Dataford — *Barclays GenAI Engineer Interview Questions & Guide 2026* | Confirms GenAI rounds probe: model/algorithm trade-offs, "how would you choose the right model for a financial application," live coding ("implement a basic neural network"), ethics of AI in financial services | Q4, Q7, Q20 |
| 5 | Barclays careers site — *Liquid Financing Analyst, RMG Capital & Funding* / *AI/ML Modeller – Global Markets* postings | Confirms Liquid Financing spans repo/financing spreads, collateral optimization, and cross-asset (Equities, Rates, FX, Futures) — i.e., pricing-curve time series is the dominant business surface | Q15–Q19, Q28 |
| 6 | Wall Street Oasis — Quant Interview forum consensus | Buy-side/sell-side quant technical rounds over-index on penalized regression (Ridge/Lasso/Elastic Net) and ensemble trees over vanilla OLS | Q9–Q14 weighting |
| 7 | LLM interview banks (DataCamp "Top LLM Interview Questions 2026," KalyanKS-NLP LLM Interview Hub) | LoRA/PEFT mechanics, RAG pipeline (chunk→embed→retrieve→rerank→generate), fine-tune-vs-RAG-vs-prompt decision tree, hallucination mitigation are the highest-frequency 2025–2026 GenAI questions | Q4–Q8 |
| 8 | CFA Institute — *LLMs in the Financial Industry: A Practical Guide* | RAG is the dominant financial-industry LLM adaptation pattern over full fine-tuning; CoT + RLHF are standard hallucination controls | Q6, Q8 |
| 9 | Nomura interview reports (Glassdoor) for quant/MLE-adjacent roles | Technical rounds emphasize applied end-to-end ML pipeline walkthroughs and NLP case studies | Q28–Q30 |

Where a platform (Blind, LeetCode/HackerRank, StreetOfWalls) returned no role-specific technical content for this exact title beyond generic logistics, it was excluded rather than padded.

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

---

## Table of Contents

### 🧑‍💻 EXPERIENCE & PRODUCTION PYTHON
- [Q1 · The Full Research-to-Production Lifecycle](./PART1.md#q1--the-full-research-to-production-lifecycle)
- [Q2 · Python Performance, the GIL & Mechanical Sympathy](./PART1.md#q2--python-performance-the-gil--mechanical-sympathy)
- [Q3 · Designing a Production-Grade Signal Engine](./PART1.md#q3--designing-a-production-grade-signal-engine)

### 🤖 GEN AI
- [Q4 · Fine-Tuning vs. Prompt Engineering vs. RAG — and LoRA Math](./PART1.md#q4--fine-tuning-vs-prompt-engineering-vs-rag--and-lora-math)
- [Q5 · RAG Architecture Deep-Dive for Financing Docs](./PART1.md#q5--rag-architecture-deep-dive-for-financing-docs)
- [Q6 · LLM Evaluation — RAGAS, Hallucination Detection, LLM-as-Judge](./PART1.md#q6--llm-evaluation--ragas-hallucination-detection-llm-as-judge)
- [Q7 · Model Selection Matrix — Claude, GPT, Open-Weight](./PART1.md#q7--model-selection-matrix--claude-gpt-open-weight)
- [Q8 · Structured Extraction & Prompt Engineering for Term Sheets](./PART1.md#q8--structured-extraction--prompt-engineering-for-term-sheets)

### 📉 REGRESSION
- [Q9 · OLS — Normal Equations & Gauss-Markov Proof](./PART1.md#q9--ols--normal-equations--gauss-markov-proof)
- [Q10 · Ridge, Lasso & Elastic Net — Derivation and Bias-Variance](./PART1.md#q10--ridge-lasso--elastic-net--derivation-and-bias-variance)
- [Q11 · Heteroskedasticity, Autocorrelation & Newey-West](./PART1.md#q11--heteroskedasticity-autocorrelation--newey-west)

### 🌳 TREE-BASED MODELS
- [Q12 · Decision Trees — Entropy, Gini & Information Gain](./PART1.md#q12--decision-trees--entropy-gini--information-gain)
- [Q13 · Random Forest vs. Gradient Boosting — XGBoost 2nd-Order Taylor Expansion](./PART1.md#q13--random-forest-vs-gradient-boosting--xgboost-2nd-order-taylor-expansi./PART1.mdon)
- [Q14 · Feature Importance & SHAP Values](#q14--feature-importance--shap-values)

### ⏱️ TIME SERIES & FORECASTING
- [Q15 · Stationarity, ADF Test & ARIMA](./PART2.md#q15--stationarity-adf-test--arima)
- [Q16 · GARCH/EGARCH — Deriving Volatility Forecasts](./PART2.md#q16--garchegarch--deriving-volatility-forecasts)
- [Q17 · Hidden Markov Models — Baum-Welch & Viterbi for Regime Detection](./PART2.md#q17--hidden-markov-models--baum-welch--viterbi-for-regime-detection)
- [Q18 · Walk-Forward Validation & Combinatorial Purged Cross-Validation](./PART2.md#q18--walk-forward-validation--combinatorial-purged-cross-validation)
- [Q19 · Kalman Filter for Dynamic Hedge Ratios / Financing Spread Tracking](./PART2.md#q19--kalman-filter-for-dynamic-hedge-ratios--financing-spread-tracking)

### 🧠 DEEP LEARNING
- [Q20 · MLP — Forward Pass & Backpropagation Derived Line-by-Line](./PART2.md#q20--mlp--forward-pass--backpropagation-derived-line-by-line)
- [Q21 · Activation Functions & the Vanishing Gradient Problem](./PART2.md#q21--activation-functions--the-vanishing-gradient-problem)
- [Q22 · RNNs — Backpropagation Through Time](./PART2.md#q22--rnns--backpropagation-through-time)
- [Q23 · LSTM — Gate Equations Derived from First Principles](./PART2.md#q23--lstm--gate-equations-derived-from-first-principles)
- [Q24 · GRU vs. LSTM — Simplification Trade-offs](./PART2.md#q24--gru-vs-lstm--simplification-trade-offs)
- [Q25 · Regularization — Dropout & BatchNorm Math](./PART2.md#q25--regularization--dropout--batchnorm-math)
- [Q26 · Attention & the Transformer Building Block](./PART2.md#q26--attention--the-transformer-building-block)
- [Q27 · Autoencoders for Dimensionality Reduction & Anomaly Detection](./PART2.md#q27--autoencoders-for-dimensionality-reduction--anomaly-detection)

### 🏗️ SYSTEM DESIGN & STRATEGY
- [Q28 · Design an End-to-End Alpha/Pricing Signal Pipeline](./PART2.md#q28--design-an-end-to-end-alphapricing-signal-pipeline)
- [Q29 · Bayesian Inference for Regime-Adaptive Position Sizing](./PART2.md#q29--bayesian-inference-for-regime-adaptive-position-sizing)
- [Q30 · Building an AI/ML Capability From Zero — the Greenfield Roadmap](./PART2.md#q30--building-an-aiml-capability-from-zero--the-greenfield-roadmap)

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

## Quick-Reference Equation Sheet

| # | Concept | Core Equation |
|---|---|---|
| Q1 | Deflated Sharpe Ratio | $\text{DSR}=\Phi\!\left(\frac{(\widehat{SR}-\widehat{SR}^{*})\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)$ |
| Q3 | Vol-targeted sizing | $w_t = s_t\cdot\frac{\sigma^*}{\hat\sigma_t}\cdot\frac{1}{\sqrt{n_{\text{corr}}}}$ |
| Q4 | LoRA update | $\Delta W = BA,\ B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$ |
| Q5 | Cosine similarity | $\text{sim}(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}$ |
| Q9 | OLS normal equations | $\hat\beta=(X^\top X)^{-1}X^\top y$ |
| Q10 | Ridge / Lasso | $\hat\beta_{\text{ridge}}=(X^\top X+\lambda I)^{-1}X^\top y$; Lasso soft-threshold $S(z,\gamma)$ |
| Q11 | Newey-West HAC | $\hat S=\hat\Gamma_0+\sum_{\ell=1}^L w_\ell(\hat\Gamma_\ell+\hat\Gamma_\ell^\top),\ w_\ell=1-\frac{\ell}{L+1}$ |
| Q12 | Entropy / Gini | $H(p)=-\sum p_c\log_2 p_c$; $G(p)=1-\sum p_c^2$ |
| Q13 | XGBoost split gain | $\text{Gain}=\tfrac12\!\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$ |
| Q14 | Shapley value | $\phi_i=\sum_{S}\frac{|S|!(|F|-|S|-1)!}{|F|!}[v(S\cup i)-v(S)]$ |
| Q15 | ADF test | $\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\sum\delta_i\Delta y_{t-i}+e_t$ |
| Q16 | GARCH(1,1) | $\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$ |
| Q17 | HMM forward/Viterbi | $\alpha_t(j)=b_j(x_t)\sum_i\alpha_{t-1}(i)A_{ij}$; $\delta_t(j)=b_j(x_t)\max_i[\delta_{t-1}(i)A_{ij}]$ |
| Q19 | Kalman gain | $K_t=\frac{P_{t|t-1}x_t}{x_t^2P_{t|t-1}+R}$ |
| Q20 | Softmax+CE gradient | $\delta^{(2)}=\hat y-y$ |
| Q22 | BPTT Jacobian product | $\prod_{j=k+1}^t \partial h_j/\partial h_{j-1}$ |
| Q23 | LSTM cell update | $C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t$ |
| Q26 | Scaled dot-product attention | $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ |
| Q27 | Autoencoder objective | $\min_{\theta,\phi}\frac1n\sum\lVert x_i-g_\phi(f_\theta(x_i))\rVert_2^2$ |

---

## Final Prep Checklist for July 8

- [ ] Re-derive Q1 (DSR), Q13 (XGBoost gain), Q20/Q23 (backprop/LSTM) on a whiteboard from memory, unaided.
- [ ] Be ready to relate every answer back to Liquid Financing's actual surface: financing-rate curves are a time-series problem (Q15-Q19) wrapped in a production-ML lifecycle (Q1, Q3) with a GenAI layer on top (Q4-Q8) — that's the throughline Rishi Dhingra's own background (electronification, eTrading, AI/ML for Prime/Financing) suggests he'll be listening for.
- [ ] Have one concrete BAM/Highbridge/Millburn story ready per domain (production incident, a signal that failed OOS, a model-risk conversation) — first-principles math earns credibility, but a specific war story is what's memorable an hour later.
- [ ] All code in this document runs end-to-end (`python3.13 <file>.py`) and persists Plotly charts to `plots/` as both `.html` and `.png` for quick visual review before the interview.

[🔝 Back to Top](#table-of-contents)

---
