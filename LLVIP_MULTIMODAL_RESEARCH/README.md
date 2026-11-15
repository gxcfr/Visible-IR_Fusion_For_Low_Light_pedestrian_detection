# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

### Research Status: Phase 2 Experimental Audit & Harmonization Completed

- **Core Research Objective**: *"Investigate whether reliability-aware asymmetric visible-infrared fusion improves pedestrian detection under challenging illumination conditions relative to unimodal and conventional fusion baselines."*
- **Dataset**: LLVIP (`C:\LLVIP\LLVIP`) — 15,488 verified pairs (100% paired, single class `person`).
- **Splits**: Scene-aware group split (Train: 10,222 pairs / Val: 1,803 pairs / **Test: 3,463 pairs strictly untouched & sealed**).
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace, SM 8.9).
- **Primary Architecture**: Ultralytics **YOLO26n** adopted for all baselines and multimodal architectures (E002–E007). Historical baseline retained: `E001_RGB_YOLO11N`.
- **Git Remote**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`

---

## 🔍 Phase 2 Research Audit and Harmonized Validation Results

Following external academic peer review, all seven experiments (E001–E007) were rigorously audited under a single unified evaluation framework. All models were evaluated on the **exact same 1,803 validation pairs** (5,086 ground truth pedestrians) in the **original $1280 \times 1024$ coordinate system** using descending-IoU matching and exact unletterboxing:

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

### Standardized GPU Latency and Computational Cost

Measured on NVIDIA GeForce RTX 4060 Laptop GPU under strict controls (batch size 1, 50 warmup + 200 measured iterations, `torch.cuda.synchronize()`):

| Exp ID | Architecture | Modality | Params (M) | FLOPs (G) | Model Fwd (ms) | Model FPS | End-to-End (ms) | End-to-End FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | RGB | 2.59 | 6.4 | $7.59 \pm 0.40$ | 131.8 | $9.97 \pm 0.54$ | 100.3 |
| **E002** | YOLO26n | RGB | 2.37 | 5.2 | $10.76 \pm 0.81$ | 92.9 | $13.03 \pm 0.93$ | 76.7 |
| **E003** | YOLO26n | Thermal | 2.37 | 5.2 | $9.83 \pm 0.46$ | 101.7 | $12.31 \pm 0.62$ | 81.3 |
| **E004** | YOLO26n | Late (WBF) | 4.74 | 10.4 | $20.38 \pm 0.65$ | 49.1 | $23.89 \pm 0.83$ | 41.9 |
| **E005** | YOLO26n | Early (4CH) | 2.37 | 5.2 | $10.16 \pm 0.46$ | 98.4 | $14.41 \pm 0.69$ | 69.4 |
| **E006** | YOLO26n | Feature Concat | 3.87 | 8.5 | $13.39 \pm 0.58$ | 74.7 | $17.48 \pm 0.77$ | 57.2 |
| **E007** | Proposed Fusion | Asym Gated | 4.90 | 9.8 | $14.17 \pm 0.75$ | 70.6 | $17.94 \pm 0.89$ | 55.7 |

---

## 📌 Critical Audit Findings and Corrections

1. **E003 (Thermal) vs E007 (Proposed Fusion)**:
   - On the nighttime LLVIP benchmark, **Thermal-only (E003: 94.12% $mAP_{50}$)** slightly outperforms all multimodal architectures, including E007 (93.70%).
   - In nighttime urban surveillance, thermal radiation provides exceptionally crisp contrast, whereas visible sensors suffer from underexposure and noise. Adding raw or weakly conditioned visible features inevitably injects a small degree of background noise.
   - Previous claims of E007 outperforming thermal-only were artifactual, resulting from disparate evaluation scripts and coordinate scaling.
2. **Architecture Clarification for E007**:
   - E007 does **not** compute explicit scalar quality descriptors $Q_T, Q_V$. It is a **learned feature-conditioned adaptive gating module**.
   - The visible branch is **not** a lightweight encoder; it duplicates the full 11-layer YOLO26n backbone (1.365M parameters per stream, 4.90M total model parameters).
   - Asymmetry is **functional/semantic**: thermal features $F_T$ pass through an unmodulated additive identity shortcut, while visible features $P(F_V)$ are adaptively modulated by gate $G_V \in (0, 1)$:
     $$F_{\text{fused}} = F_T + G_V \odot P(F_V)$$
3. **Gate Activation Semantics**:
   - Gate activations represent learned spatial/channel modulations, **not** calibrated physical sensor reliability probabilities.
   - Gate values inside pedestrian bounding boxes are systematically lower (~0.36) than in background areas (~0.51), confirming that the network actively dampens noisy visible features on pedestrian heat signatures.
4. **Early Stopping and Checkpoints**:
   - Validation fitness peaked at **Epoch 5** for both E006 and E007, triggering early stopping at Epoch 17. All evaluated metrics strictly reflect official `best.pt` checkpoints.

---

## 📁 Audit Documentation Index (`research/audit_phase/`)

- [AUDIT_REPORT.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/AUDIT_REPORT.md) — Master comprehensive 12-section audit report.
- [architecture_audit_report.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/architecture_audit_report.md) — Full code-level graph and parameter audit.
- [convergence_analysis.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/convergence_analysis.md) — Training loss and validation fitness convergence study.
- [standardized_latency_report.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/standardized_latency_report.md) — GPU benchmarking protocol and results.
- [evaluation_protocol.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/evaluation_protocol.md) — Formal standardized evaluation specification.
- [unified_validation_metrics.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/unified_validation_metrics.csv) — Harmonized metrics in tabular format.
- [gate_analysis/](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/gate_analysis/) — Extended gate activation distributions and environmental correlation charts.
- [qualitative_analysis/](file:///C:/VI%20Fusion%20for%20LL-PD/research/audit_phase/qualitative_analysis/) — 8-panel high-resolution detection comparisons across all 4 validation scenes.
