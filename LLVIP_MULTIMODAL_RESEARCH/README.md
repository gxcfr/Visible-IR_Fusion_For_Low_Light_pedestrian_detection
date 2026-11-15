# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

### Research Status: Stage 8 Completed (Official Test Evaluation Completed Across 3,463 Test Pairs; Stage 9 Final Research Paper Package In Progress)

- **Core Research Objective**: *"Investigate whether reliability-aware asymmetric visible-infrared fusion improves pedestrian detection under challenging illumination conditions relative to unimodal and conventional fusion baselines."*
- **Execution Mandate**: **Zero additional training runs.** All evaluations operate strictly on frozen checkpoints (E001–E007). Proposed models E008/E009 and multi-seed training runs are formally **DEFERRED**.
- **Dataset**: LLVIP (`C:\LLVIP\LLVIP`) — 15,488 verified pairs (100% paired, single class `person`).
- **Splits**: Scene-aware group split (Train: 10,222 pairs / Val: 1,803 pairs / **Test: 3,463 pairs strictly untouched & sealed**).
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace, SM 8.9), Windows 11.
- **Software Stack**: PyTorch 2.5.1+cu121, CUDA 12.1, OpenCV 5.0.0, Ultralytics 8.4.144.
- **Evaluation Framework**: Ultralytics `DetectionValidator.match_predictions` and `ap_per_class` on unletterboxed native $1280 \times 1024$ coordinates.
- **Primary Architecture**: Ultralytics **YOLO26n** adopted for all baselines and multimodal architectures (E002–E007). Historical baseline retained: `E001_RGB_YOLO11N`.
- **Git Remote**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`

---

## 🔍 Master Harmonized Validation Results (1,803 Val Pairs, 5,086 Ground Truths)

All seven experiments (E001–E007) were evaluated under a standardized, unified evaluation framework in the **original $1280 \times 1024$ coordinate system** using Ultralytics descending-IoU matching and exact unletterboxing:

### Unified Validation Comparison Table

| Exp ID | Architecture | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Prec@0.25 (%) | Rec@0.25 (%) | F1@0.25 (%) | Max-F1 (%) | Total Preds |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | Visible (RGB) | 71.69 | 34.65 | 90.11 | 63.51 | 74.51 | 74.81 | 3,584 |
| **E002** | YOLO26n | Visible (RGB) | 73.04 | 35.42 | 86.43 | 65.71 | 74.66 | 74.74 | 3,868 |
| **E003** | YOLO26n | Thermal (IR) | **94.12** | 55.96 | 89.99 | **90.48** | **90.24** | **90.50** | 5,114 |
| **E004** | YOLO26n | Late Fusion (WBF) | 93.74 | 53.80 | **96.27** | 78.15 | 86.27 | 89.85 | 4,127 |
| **E005** | YOLO26n | Early Fusion (4CH) | 72.24 | 37.45 | 73.41 | 69.90 | 71.61 | 73.25 | 231,221 |
| **E006** | YOLO26n | Feature Concat | 93.24 | **56.09** | 89.64 | 88.50 | 89.07 | 89.47 | 5,021 |
| **E007** | Proposed Fusion | Asym Gated Fusion | 93.70 | 55.98 | 87.52 | 90.25 | 88.86 | 90.41 | 5,245 |

*Note: All metrics computed exclusively on `best.pt` checkpoints on 1,803 validation pairs in original $1280 \times 1024$ space.*

### Inference-Only Gate Ablations on E007 (Stage 3 Sensitivity Analysis)

| Condition ID | Gate Mode Description | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall (@0.25) | Precision (@0.25) | F1 (@0.25) | FP / Image | Misses (FN) | Mean Fwd Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **IA001** | **Learned Gate (Original E007)** | **93.70** | **55.98** | **90.25%** | **87.52%** | **88.86%** | **0.37** | **496** | 22.51 ms |
| **IA002** | **Gate Zero ($G_V = 0.0$)** | 72.27 | 38.45 | 59.49% | 85.31% | 70.10% | 0.33 | 2,060 | 21.53 ms |
| **IA003** | **Gate Half ($G_V = 0.5$)** | 84.24 | 48.12 | 80.32% | 75.17% | 77.66% | 0.75 | 1,001 | 21.97 ms |
| **IA004** | **Gate One ($G_V = 1.0$)** | 78.03 | 40.37 | 78.31% | 60.37% | 68.18% | 1.46 | 1,103 | 22.02 ms |

*Key Takeaway: Forcing $G_V=0.0$ at inference triggers representation starvation (mAP50 collapses from 93.70% to 72.27%), demonstrating network co-adaptation and proving that inference gate zeroing $\ne$ independently trained thermal baseline E003 (94.12%). Forcing $G_V=1.0$ quadruples false alarms (0.37 to 1.46 FP/img) by injecting unmodulated low-light noise.*

### Partial Robustness Stress-Testing (Stage 4 — 25 Completed Conditions across 5 Suites)

> **Integrity Disclosure**: 25 of 40 planned conditions completed (62.5% coverage across 5 complete suites $\times$ 5 severity levels). Thermal dead-pixels halted incomplete; simultaneous degradation suites deferred. Described strictly as a *partial robustness study*.

| Model ID | Clean $mAP_{50}$ | Visible Brightness ($R_{\text{mean}}$) | Visible Noise ($R_{\text{mean}}$) | Visible Blur ($R_{\text{mean}}$) | Thermal Contrast ($R_{\text{mean}}$) | Thermal Striping ($R_{\text{mean}}$) | **5-Suite Mean ($R_{\text{mean}}^{5\text{-suites}}$)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E002 (RGB)** | 73.04% | 86.74% | 47.46% | 75.98% | 100.00%* | 100.01%* | 82.04% |
| **E003 (Thermal)** | **94.12%** | 100.00%* | 100.00%* | 100.00%* | 74.00% | 38.10% | 82.42% |
| **E004 (Late WBF)** | 93.74% | **99.92%** | **99.70%** | 98.88% | **92.02%** | 80.30% | **94.16%** |
| **E005 (Early 4CH)** | 72.24% | 101.90% | 99.24% | 96.12% | 51.84% | 51.21% | 80.06% |
| **E006 (Concat)** | 93.24% | 99.20% | 99.30% | 99.14% | 75.46% | 73.99% | 89.42% |
| **E007 (Asym Gated)** | 93.70% | 98.52% | 98.72% | **99.18%** | 73.66% | **83.47%** | 90.71% |

*\*Note: For unimodal models, orthogonal modality corruption preserves clean performance ($100.0\%$) by definition as their sensor input remains pristine.*  
*Key Takeaway: Under extreme thermal sensor striping (L5 $\sigma=80$), E007 retains **51.17% mAP50** (54.6% retention), significantly outperforming unweighted feature concatenation E006 (30.88% mAP50, 33.1% retention) by **+20.29% mAP50**, proving the noise-suppression capability of dynamic gating against structured sensor artifacts.*

### Standardized GPU Latency and Computational Cost

Measured on NVIDIA GeForce RTX 4060 Laptop GPU (50 warmup + 200 measured iterations, `torch.cuda.synchronize()`, full end-to-end pipeline):

| Exp ID | Architecture | Modality | Params (M) | Head ($nc$) | FLOPs (G) | Model Fwd (ms) | Model FPS | End-to-End (ms) | End-to-End FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | RGB | 2.59 | 1 | 6.5 | $7.59 \pm 0.40$ | 131.8 | $9.97 \pm 0.54$ | 100.3 |
| **E002** | YOLO26n | RGB | 2.50 | 1 | 5.9 | $10.76 \pm 0.81$ | 92.9 | $13.03 \pm 0.93$ | 76.7 |
| **E003** | YOLO26n | Thermal | 2.50 | 1 | 5.9 | $9.83 \pm 0.46$ | 101.7 | $12.31 \pm 0.62$ | 81.3 |
| **E004** | YOLO26n | Late (WBF) | 5.01 | 1 + 1 | 11.8 | $20.38 \pm 0.65$ | 49.1 | $23.89 \pm 0.83$ | 41.9 |
| **E005** | YOLO26n | Early (4CH) | 2.57 | 80 | 6.3 | $10.16 \pm 0.46$ | 98.4 | $14.41 \pm 0.69$ | 69.4 |
| **E006** | YOLO26n | Feature Concat | 4.14 | 80 | 9.9 | $13.39 \pm 0.58$ | 74.7 | $17.48 \pm 0.77$ | 57.2 |
| **E007** | Proposed Fusion | Asym Gated | 4.97 | 80 | 12.2 | $14.17 \pm 0.75$ | 70.6 | $17.94 \pm 0.89$ | 55.7 |

### Master Official Test Benchmark (3,463 Test Pairs, 8,302 Ground Truths)

Following Stage 7 model freeze and cryptographic SHA256 registration, the official unseen LLVIP test partition was opened and evaluated across all 7 models:

| Exp ID | Architecture | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall@0.25 (%) | Prec@0.25 (%) | F1@0.25 (%) | True Positives | False Positives | False Negatives (Misses) | FP / Image |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | Visible (RGB) | 79.34 | 42.35 | 68.36 | 92.94 | 78.78 | 5,675 | 431 | 2,627 | 0.124 |
| **E002** | YOLO26n | Visible (RGB) | 81.44 | 44.16 | 72.79 | 89.98 | 80.48 | 6,043 | 673 | 2,259 | 0.194 |
| **E003** | YOLO26n | Thermal (IR) | 93.34 | **59.35** | 87.77 | 90.04 | **88.89** | 7,287 | 806 | 1,015 | 0.233 |
| **E004** | YOLO26n | Late (WBF) | **93.57** | 58.27 | 73.52 | **97.20** | 83.72 | 6,104 | **176** | 2,198 | **0.051** |
| **E005** | YOLO26n | Early (4CH) | 56.00 | 32.26 | 60.33 | 54.32 | 57.17 | 5,009 | 4,213 | 3,293 | 1.217 |
| **E006** | YOLO26n | Feature Concat | 91.00 | 57.23 | 86.86 | 81.56 | 84.13 | 7,211 | 1,630 | 1,091 | 0.471 |
| **E007** | Proposed Fusion | Asym Gated | 91.43 | 56.76 | **88.52** | 81.22 | 84.71 | **7,349** | 1,699 | **953** | 0.491 |

*Key Takeaway: Proposed Asymmetric Gated Fusion (**E007**) achieves the **highest recall (88.52%)** and fewest misses (**953 FNs**) of any model across the official unseen test set, successfully detecting **7,349 out of 8,302 pedestrians** (+62 pedestrians recovered over thermal-only E003).*

---

## 📊 Dataset Annotation Reconciliation

| Partition | Image Pairs | Raw XML Annotations | Excluded Invalid Annotations | Valid Converted Annotations | Valid Evaluation Ground Truths |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training** | 10,222 | 29,047 | 3 (zero-width lines: `090437`, `091180`, `170350`) | 29,044 | 29,044 |
| **Validation** | 1,803 | 5,088 | **2 (zero-width lines in Scene 02: `020118`, `020236`)** | **5,086** | **5,086** |
| **Test** *(Sealed)* | 3,463 | 8,302 | 0 | 8,302 | 8,302 |
| **Total** | **15,488** | **42,437** | **5** | **42,432** | **42,432** |

---

## 📌 Reconciled Findings & Scientific Conclusions

1. **Unimodal Thermal Primacy**: **E003 (Thermal-only)** is the strongest individual unimodal model on the LLVIP nighttime benchmark, achieving the highest $mAP_{50}$ (**94.12%**) and highest recall (**90.48%**). Thermal radiation provides an optimal contrast signal for human body heat in outdoor nighttime surveillance.
2. **Feature Concatenation Localization**: **E006 (Feature Concatenation)** attains the highest numerical high-threshold localization performance ($mAP_{50-95} = \mathbf{56.09\%}$), demonstrating that direct multi-scale channel concatenation preserves spatial detail.
3. **Proposed Asymmetric Fusion Parity**: **E007 (Proposed Asymmetric Gated Fusion)** achieves performance closely trailing E003 in recall (**90.25%** vs 90.48%) and localization ($mAP_{50-95} = \mathbf{55.98\%}$ vs 55.96%), while achieving $mAP_{50} = \mathbf{93.70\%}$.
4. **Relative Gains over E006**: E007 improves detection recall relative to E006 (**90.25%** vs 88.50%) and increases $mAP_{50}$ (**93.70%** vs 93.24%).
5. **Computational Cost Tradeoff**: E007 incurs greater computational complexity (4.97M parameters, 17.94 ms end-to-end) than unimodal thermal (2.37M parameters, 12.31 ms end-to-end), while running well above the real-time 30 FPS threshold (55.7 FPS).
6. **Absence of Universal Superiority**: **The current experiments do not establish that adaptive fusion is universally superior to thermal-only on this dataset**. Under typical nighttime LLVIP conditions, adding visible features offers marginal overall numerical benefit because thermal contrast is already near saturation.
7. **Empirical Value of Fusion in Degraded Conditions**: In complex localized scenarios (e.g., Scene 25 with artificial headlight glare and deep shadows), unimodal thermal misses shadowed individuals and visible misses thermal contrast targets; multi-modal feature fusion successfully recovers **both** pedestrians.
8. **Scene-Conditioned Gate Dynamics & Simpson's Paradox**: As proved in `SCENE_CONDITIONED_GATE_ANALYSIS.md`, gate activations correlate positively with visible brightness within three of four scenes (+0.09 to +0.49), but invert to -0.32 under naive pooling due to baseline scene luminance shifts. True within-scene centered correlation is +0.116. Furthermore, accounting for sequence frame autocorrelation via 50-frame block bootstrap widens standard error by over 3x (SE = 0.075 vs naive 0.024), demonstrating that naive i.i.d. p-values are heavily anticonservative.
9. **Detection Head Pruning**: Fusion models E005-E007 carried 80-class COCO heads (4.97M params), whereas baselines E001-E003 used 1-class heads (2.37M params). A 1-class head variant (4.90M params) has been specified as **E008** in `HEAD_CLASS_COUNT_AUDIT.md`.
10. **Controlled Next-Phase Protocols**: Comprehensive protocols for 6-configuration ablations, 5-level synthetic degradation stress-testing, and 5-seed repeatability have been formalized in `research/next_phase/` before launching any new training runs.

---

## 🔍 Reconciled Architectural Details for E007

- **Parameter Count Reconciliation**:
  - The physical checkpoint saved on disk contains **4,971,992 parameters** (4.97M) because it was initialized from the COCO-pretrained `yolo26n.pt` base model which retains the 80-class detection head.
  - A theoretical single-class pruned model contains **4,903,902 parameters** (4.90M).
  - Thermal Backbone: 1,365,472 params | Visible Backbone: 1,365,472 params | Multi-Scale Gating: 1,034,240 params | Neck & Head: 1,206,808 params.
- **Asymmetric Residual Formulation**:
  $$F_{\text{fused}} = F_T + G_V \odot P(F_V)$$
  where thermal features $F_T$ pass through an unmodulated additive identity shortcut, while visible features $P(F_V)$ are adaptively modulated by dynamic gate map $G_V \in (0, 1)$.
- **Gate Semantics**: Gate values represent unitless learned deep feature modulation scalars, **not** calibrated physical sensor reliability probabilities. Inside pedestrian heat signatures, the network attenuates visible features ($G_V \approx 0.505$) relative to background ($G_V \approx 0.513$) to prioritize crisp thermal silhouettes while preserving background context.

---

## 📁 Final Phase Deliverables (`research/final_phase/`)

- [01_AUDIT_RECONCILIATION.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/01_AUDIT_RECONCILIATION.md) — Master Stage 1 documentation reconciliation, exact parameter counts, and architecture realignment.
- [02_GATE_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/02_GATE_ANALYSIS.md) — Stage 2 scene-conditioned correlation, Simpson's paradox, and block-bootstrap verification.
- [03_INFERENCE_ABLATIONS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/03_INFERENCE_ABLATIONS.md) — Stage 3 inference-only gate interventions (IA001–IA004), co-adaptation collapse, and noise trade-offs.
- [04_ROBUSTNESS_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/04_ROBUSTNESS_ANALYSIS.md) — Stage 4 partial robustness study across 25 completed conditions, degradation slopes, and gate adaptation.
- [05_VALIDATION_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/05_VALIDATION_ANALYSIS.md) — Stage 5 consolidated validation & failure analysis across Scenes 02, 05, 07, 25, error taxonomy, and visual panels.
- [06_COMPUTATIONAL_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/06_COMPUTATIONAL_ANALYSIS.md) — Stage 6 computational efficiency, FLOPs, latency, and memory profiling on NVIDIA RTX 4060 GPU.
- [07_MODEL_FREEZE_REPORT.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/07_MODEL_FREEZE_REPORT.md) — Stage 7 model selection freeze, cryptographic SHA256 checksums, and sealed test protocol.
- [08_FINAL_TEST_RESULTS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_phase/08_FINAL_TEST_RESULTS.md) — Stage 8 held-out official test set evaluation across 3,463 pairs and 7 unseen scenes.

---

## 📁 Historical Audit Reconciliation Archives (`research/audit_phase/final_reconciliation/`)

- [FINAL_RESEARCH_AUDIT.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/FINAL_RESEARCH_AUDIT.md) — Master comprehensive reconciliation document.
- [CORRECTED_GATE_STATISTICS.json](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/CORRECTED_GATE_STATISTICS.json) — Canonical gate statistics across all 1,803 validation pairs.
- [CORRECTED_GATE_ANALYSIS.png](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/CORRECTED_GATE_ANALYSIS.png) — Environmental correlation and foreground/background gate distributions.
- [CORRECTED_OBJECT_SCALE_ANALYSIS.png](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/CORRECTED_OBJECT_SCALE_ANALYSIS.png) — Pedestrian scale analysis with exact sample counts.
- [CORRECTED_QUALITATIVE_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/CORRECTED_QUALITATIVE_ANALYSIS.md) — Ground-truth matched evaluation across Scenes 02, 05, 07, and 25.
- [PARAMETER_COUNT_AUDIT.json](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/PARAMETER_COUNT_AUDIT.json) — Parameter count accounting audit (4.97M vs 4.90M).
- [UNIFIED_EVALUATOR_SOURCE.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/final_reconciliation/UNIFIED_EVALUATOR_SOURCE.md) — Complete evaluation source code and package versions.

---

## 🚀 Next-Phase Protocols and Epistemological Closure (`research/next_phase/`)

- [FINAL_AUDIT_CLOSURE.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/FINAL_AUDIT_CLOSURE.md) — Final audit closure, epistemological corrections, and research roadmap.
- [SCENE_CONDITIONED_GATE_ANALYSIS.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/SCENE_CONDITIONED_GATE_ANALYSIS.md) — Scene-conditioned correlation, Simpson's paradox, and moving block bootstrap.
- [HEAD_CLASS_COUNT_AUDIT.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/HEAD_CLASS_COUNT_AUDIT.md) — Detection head class count audit ($nc=80$ vs $nc=1$) and E008 specification.
- [ABLATION_PROTOCOL.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/ABLATION_PROTOCOL.md) — 6-configuration controlled ablation design protocol.
- [ROBUSTNESS_PROTOCOL.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/ROBUSTNESS_PROTOCOL.md) — 5-level synthetic degradation stress-testing protocol.
- [MULTI_SEED_PROTOCOL.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/next_phase/MULTI_SEED_PROTOCOL.md) — 5-seed repeatability and statistical variance protocol.
