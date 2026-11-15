# Multi-Seed Experimental Repeatability and Statistical Variance Protocol

## 1. Executive Summary & Epistemological Purpose

In external peer review of deep learning architectures, single-run benchmarks are frequently scrutinized for seed-selection bias or lucky initialization. Stochastic factors during training include:
1. **Network Weight Initialization**: Random orthogonal / Kaiming normal initializations of custom fusion modules and detection heads.
2. **Data Augmentation Ordering**: Mini-batch composition, letterboxing offsets, Mosaic combinations, and color jitter sequences.
3. **Hardware Non-Determinism**: GPU non-deterministic floating-point accumulation in atomic CUDA operations.

To guarantee that observed performance differentials between unimodal baselines, naive feature concatenation, and asymmetric gated fusion represent genuine algorithmic properties rather than stochastic artifacts, this protocol specifies a **5-seed replication study**.

---

## 2. Experimental Design and Scope

### 2.1 Model Scope
The multi-seed evaluation will be conducted for the three primary competing paradigms (plus the proposed 1-class architecture):

1. **Model A: Unimodal Thermal Baseline (YOLO26n Thermal / E003)**
   - Parameter count: 2.37M
   - Serves as the single-modality benchmark.
2. **Model B: Dual-Stream Feature Concatenation (E006)**
   - Parameter count: 4.97M
   - Serves as the naive multimodal benchmark.
3. **Model C: Asymmetric Gated Fusion (Proposed Architecture / E007 / E008)**
   - Parameter count: 4.97M (80-class) / 4.90M (1-class)
   - Evaluates whether learned adaptive gating consistently outperforms Models A and B across random initializations.

---

### 2.2 Standardized Seed Matrix

To eliminate researcher degrees of freedom, the five random seeds are fixed *a priori*:

$$\mathcal{S} = \{42, \; 123, \; 456, \; 789, \; 1024\}$$

For each seed $s \in \mathcal{S}$:
- PyTorch CPU and GPU random generators are initialized: `torch.manual_seed(s)`, `torch.cuda.manual_seed_all(s)`.
- NumPy random generator is initialized: `np.random.seed(s)`.
- Python native `random` is initialized: `random.seed(s)`.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are enforced to maximize reproducibility.

---

## 3. Strict Controlled Invariants

Every seed replication must strictly hold the following variables invariant:

| Invariant Parameter | Controlled Specification |
|:---|:---|
| **Dataset Splits** | Exactly 10,643 training pairs and 1,803 validation pairs (LLVIP canonical split). |
| **Test Partition Isolation** | Official test set (3,463 pairs) strictly locked and unaccessed. |
| **Input Resolution** | $640 \times 640$ letterboxed ($1280 \times 1024$ native source). |
| **Optimizer & Schedule** | SGD ($\text{momentum}=0.937, \text{weight\_decay}=0.0005$), Cosine LR schedule ($0.002 \to 0.00002$). |
| **Batch Size** | 4 image pairs per forward/backward pass. |
| **Stopping Criterion** | Early stopping with patience = 12 epochs on validation mAP50-95 fitness metric. |
| **Checkpoint Selection** | `best.pt` chosen strictly based on validation fitness during training. |
| **Evaluation Harness** | Standardized Unified Evaluator (`evaluate_unified.py`) in native $1280 \times 1024$ coordinate space. |

---

## 4. Statistical Metrics and Reporting Format

For each model family across the $K = 5$ seeds, results will be summarized using robust sample statistics:

### 4.1 Quantitative Aggregates
For metric $X \in \{\text{mAP50}, \text{mAP50-95}, \text{Precision}, \text{Recall}, \text{F1}\}$:
1. **Sample Mean**:
   $$\bar{X} = \frac{1}{K} \sum_{k=1}^K X_k$$
2. **Unbiased Sample Standard Deviation**:
   $$s = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (X_k - \bar{X})^2}$$
3. **Range**: $[\min(X), \max(X)]$.
4. **95% Confidence Interval** (Student's $t$-distribution with $df = 4$, $t_{0.025, 4} = 2.776$):
   $$\text{CI}_{95\%} = \left[ \bar{X} - 2.776 \cdot \frac{s}{\sqrt{5}}, \quad \bar{X} + 2.776 \cdot \frac{s}{\sqrt{5}} \right]$$

### 4.2 Standard Reporting Schema

| Model Paradigm | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1024 | Mean $\pm$ Std | 95% Confidence Interval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E003 (Thermal Only)** | - | - | - | - | - | $\bar{X} \pm s$ | $[\dots, \dots]$ |
| **E006 (Feature Concat)** | - | - | - | - | - | $\bar{X} \pm s$ | $[\dots, \dots]$ |
| **E007 (Asymmetric Gate 80c)**| - | - | - | - | - | $\bar{X} \pm s$ | $[\dots, \dots]$ |
| **E008 (Asymmetric Gate 1c)** | - | - | - | - | - | $\bar{X} \pm s$ | $[\dots, \dots]$ |

---

## 5. Formal Hypothesis Testing

To establish whether performance improvements are statistically significant, paired statistical tests will be applied across matched seeds:

1. **Paired Student's $t$-Test / Welch's $t$-Test**:
   - Null Hypothesis $H_0$: $\mu_{\text{Proposed}} - \mu_{\text{Baseline}} \le 0$
   - Alternative Hypothesis $H_1$: $\mu_{\text{Proposed}} - \mu_{\text{Baseline}} > 0$
   - Significance threshold: $\alpha = 0.05$.
2. **Effect Size (Cohen's $d$)**:
   $$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$
   Measures the standardized magnitude of difference independent of sample size.

---

## 6. Execution Governance

- **Execution Status**: **PROPOSED** — The multi-seed experiment requires $3 \text{ models} \times 4 \text{ additional seeds} = 12$ full training runs ($\approx 40$ epochs each).
- Training will commence only upon explicit authorization from the primary investigator, following approval of Experiment E008 and the ablation protocol.
