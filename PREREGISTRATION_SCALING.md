# Pre-Registration Specification: Cross-Architecture Scaling Surface & Causal Reasoning Benchmark

- **Protocol Identifier**: `PREREG-SCALING-v1`
- **Author**: Sultan Haikal (GitHub: `@yvliet`)
- **Affiliation**: Independent Research / Gemma 4 Developer Agent Track 2 Submission
- **Timestamp**: October 12, 2026
- **Status**: Pre-Registered & Locked Prior to Experimental Execution
- **Target Venue**: Google DeepMind Developer Agent Competition Track 2 ($35,000 Prize Pool)
- **Primary Evaluator Alignment**: Graph-ML Peer Review Criteria (Bryan Perozzi, Benedek Rózemberczki, Mikhail Galkin)

---

## 1. Context & Motivation

In Phase 1 of our investigation, we established a confirmatory null result under direct lexical readout on Stratum C (131 un-named bug localization tasks in SWE-bench Verified): 1-hop graph diffusion ($A_4$) tied depth-matched BM25 ($A_2$) at 16.79% Acc@5 (22/131 hits each, exact McNemar $p = 0.287$). 

Under live Dual Tesla T4 GPU execution:
- **Control 1**: Test-time reasoning (`<thought>`) decisively broke the tie in favor of the graph-expanded pool (+5.34 pp Acc@5, 24 vs 17 hits, MRR 0.1358 vs 0.0967), while hash-shuffled candidate ordering ($sha256(instance\_id \mid candidate\_id)$) falsified presentation-order bias (candidate 1 chosen on only 6.87% of tasks vs. 91% un-reasoned).
- **Control 2**: Candidate selection conditioning (Berkson's bias) was verified via a $2 \times 2$ decomposition, proving that the 0.137 drop in ROC-AUC from seeds (0.7966) to graph neighbors (0.6596) is an expected artifact of selecting candidates outside the top-10 lexical seeds rather than neural scorer degradation.

### The Core Theoretical Gap
Is the Lexical Abstraction Bottleneck ($\text{AUC} \le 0.6596$) a permanent constraint of compact edge models ($2\text{B}$ to $4\text{B}$ parameters), or does increased model scale break through this barrier zero-shot? Furthermore, does dynamic Mixture-of-Experts routing (3.8B active parameters out of 26B total) provide representational efficiency over dense architectures?

This study executes a pre-registered cross-architecture scaling curve evaluating 5 contemporary (2025–2026) open-weight models across three distinct architectural paradigms.

---

## 2. Model Evaluation Matrix

All models are evaluated on identical candidate pools derived from SWE-bench Verified Stratum C using bitsandbytes NF4 quantization on Kaggle Dual Tesla T4 GPUs (2x 15,360 MiB GDDR6):

| Family | Model Identifier | Parameter Count | Active Parameters | Architecture Archetype | License |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Google Gemma 4** | `google/gemma-4-e2b-it` | 2.0B | 2.0B | Per-Layer Embeddings (PLE Edge) | Apache 2.0 |
| **Google Gemma 4** | `google/gemma-4-e4b-it` | 4.0B | 4.0B | Per-Layer Embeddings (PLE Edge) | Apache 2.0 |
| **Google Gemma 4** | `google/gemma-4-26b-a4b-it` | 26.0B | 3.8B | Dynamic Sparse MoE (128 experts) | Apache 2.0 |
| **Microsoft Phi** | `microsoft/Phi-4-mini-instruct` | 3.8B | 3.8B | High-Density Synthetic Reasoning | MIT |
| **DeepSeek AI** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7.0B | 7.0B | RL Reasoning Distilled Transformer | MIT |

---

## 3. Evaluation Protocol: Option C Hybrid Design

To reconcile statistical power across 201 candidate pools with end-to-end task validity without risking Kaggle 9.0-hour timeout limits, we execute a two-tier hybrid protocol:

### Tier 1: Pointwise Candidate Discrimination Ladder (All 5 Models)
- **Scope**: 201 candidate pools where the gold fault location is present in the $A_4$ candidate pool ($N=176$ seed-gold targets, $N=25$ graph-tail targets).
- **Forward Pass Scoring**:
  $$s(t, v) = \log P(\text{Yes} \mid \text{prompt}) - \log P(\text{No} \mid \text{prompt})$$
  Evaluated with left-padding and `logits_to_keep=1` at the final classification position.
- **Content-Free Prior Calibration**:
  $$s_{\text{cal}}(t, v) = s(t, v) - s(t, \emptyset)$$
  where $s(t, \emptyset)$ is the model's baseline log-odds on an empty candidate breadcrumb, removing unigram vocabulary prior skew across differing tokenizers (256k vs. 151k vs. 100k).
- **Metric**: Within-pool pairwise ROC-AUC for seed targets vs. graph-tail targets.

### Tier 2: Listwise Test-Time Reasoning Anchoring (2 Models)
- **Anchor Models**: `google/gemma-4-e4b-it` (Edge reference) and `google/gemma-4-26b-a4b-it` (Scaled MoE).
- **Scope**: All 131 Stratum C un-named defect tasks under greedy decoding with $\le 180$ `<thought>` reasoning tokens.
- **Metrics**: Acc@1, Acc@5, MRR@50, and rank correlation (Kendall's $\tau$) between Tier 1 pointwise ROC-AUC and Tier 2 listwise Acc@5 across candidate pools.

---

## 4. Pre-Registered Hypotheses & Decision Rules

### Hypothesis 1: Monotonic Capacity Scaling ($H_1$)
- **Prediction**: Pointwise discrimination on graph-tail targets ($\text{AUC}_{\text{tail}}$) scales monotonically with model capacity ($\tau \ge 0.80, p < 0.05$).
- **Falsification Condition**: If larger dense models (7B) achieve graph-tail discrimination statistically indistinguishable from 2B models ($\Delta \text{AUC} \le 0.02, p > 0.10$), $H_1$ is falsified; the Lexical Abstraction Bottleneck is proven to be scale-invariant zero-shot.

### Hypothesis 2: MoE Parameter Efficiency ($H_2$)
- **Prediction**: Gemma 4 26B-A4B (activating only 3.8B parameters) achieves graph-tail discrimination matching or exceeding dense 7B architectures while fitting consumer VRAM constraints ($\le 9\text{ GiB}$ per GPU).
- **Falsification Condition**: If Gemma 4 26B-A4B yields $\text{AUC}_{\text{tail}} \le \text{E4B baseline}$ ($0.6596$, $p > 0.10$), $H_2$ is falsified; MoE routing does not overcome lexical distraction zero-shot.

### Hypothesis 3: Order-Preserving Surrogate Relationship ($H_3$)
- **Prediction**: Pointwise forward-pass candidate ranking is an order-preserving surrogate for multi-turn listwise test-time reasoning across the anchor models ($\tau \ge 0.80, p < 0.05$).
- **Falsification Condition**: If rank correlation between pointwise AUC and listwise Acc@5 falls below 0.60 ($p > 0.05$), the surrogate metric assumption is rejected.

---

## 5. Statistical Rigor & Analysis Plan

1. **Unit of Analysis**: Task-level candidate discrimination (the fraction of negative distractors outscored by the true fault location within each candidate pool).
2. **Confidence Intervals**: 95% non-parametric cluster bootstrap by repository and task ($B=10,000$ resamples).
3. **Multiplicity Correction**: Holm-Bonferroni step-down adjustment across secondary contrast families.
4. **Hardware Verification Telemetry**: All raw JSON output files must record GPU device UUID, PCIe bus ID, peak memory allocation (`torch.cuda.max_memory_allocated()`), wall-clock inference latency, and sha256 prompt/response hashes.
