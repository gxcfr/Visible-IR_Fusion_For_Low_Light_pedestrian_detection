# Phase C / Stage 3: Inference-Only Gate Interventions and Feature Co-adaptation Analysis

**Project**: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection  
**Model Under Test**: Frozen Checkpoint `E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N/weights/best.pt`  
**Dataset**: LLVIP Validation Split (1,803 image pairs across 4 surveillance scenes)  
**Deliverable ID**: `STAGE_03_INFERENCE_ABLATIONS`  
**Date**: November 15, 2026  

---

## 1. Executive Summary & Epistemological Boundaries

In compliance with the mandate freezing all model training, this stage conducts **controlled inference-time interventions** to isolate the functional contribution of the learned gating module and evaluate cross-modal feature co-adaptation within the proposed Asymmetric Gated Fusion framework (**E007**).

### Crucial Epistemological Distinction:
> [!IMPORTANT]
> **Inference-Time Intervention $\ne$ Training-Time Architectural Ablation**:
> Forcing the gate tensor to a constant value ($G_V \in \{0.0, 0.5, 1.0\}$) at runtime tests how the **fully-trained network responds to feature disconnection or static scaling**. 
> 
> In particular: **Forcing $G_V = 0.0$ does NOT equal the unimodal thermal baseline (E003)**.
> - In E003, the backbone, neck, and detection head were optimized over 40+ epochs to extract pedestrian features exclusively from thermal imagery.
> - In E007, the thermal stream and shared detection neck co-adapted during joint gradient descent with the visible stream. The upper neck layers expect combined representations ($F_T + G_V \odot P(F_V)$). Forcing $G_V = 0.0$ starves downstream filters of expected activation energy.
> 
> Therefore, this protocol evaluates **architectural reliance, feature co-dependence, and noise suppression efficacy**, rather than defining newly trained models.

---

## 2. Experimental Setup and Gate Conditions

All four conditions were evaluated on the exact same **1,803 validation image pairs** (5,086 ground-truth pedestrians) using the frozen checkpoint `best.pt` in native $1280 \times 1024$ space with identical NMS and evaluation post-processing:

- **IA001 (Learned Gate - Original)**: $G_V = \sigma(\text{Conv}([F_T, P(F_V)]))$. Operates with dynamic, data-driven spatial-channel feature modulation.
- **IA002 (Gate Zero)**: $G_V \equiv 0.0$. Completely severs optical feature injection; thermal features pass through the residual path alone.
- **IA003 (Gate Half)**: $G_V \equiv 0.5$. Replaces the dynamic gate with a constant $50\%$ linear feature injection.
- **IA004 (Gate One)**: $G_V \equiv 1.0$. Replaces the dynamic gate with unattenuated $100\%$ additive feature injection ($F_{\text{fused}} = F_T + P(F_V)$).

---

## 3. Master Quantitative Evaluation Results

| Condition ID | Gate Mode Description | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall (@0.25) | Precision (@0.25) | F1 (@0.25) | Max-F1 (%) | Total Preds ($\tau \ge 0.25$) | FP / Image | Misses (FN) | Mean Fwd Latency | FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **IA001** | **Learned Gate (Original)** | **93.70** | **55.98** | **90.25%** | **87.52%** | **88.86%** | **90.41%** | 5,250 | **0.3661** | **496** | 22.51 ms | 44.4 |
| **IA002** | **Gate Zero ($G_V = 0.0$)** | 72.27 | 38.45 | 59.49% | 85.31% | 70.10% | 70.53% | 3,627 | 0.3333 | 2,060 | 21.53 ms | 46.5 |
| **IA003** | **Gate Half ($G_V = 0.5$)** | 84.24 | 48.12 | 80.32% | 75.17% | 77.66% | 80.52% | 5,442 | 0.7526 | 1,001 | 21.97 ms | 45.5 |
| **IA004** | **Gate One ($G_V = 1.0$)** | 78.03 | 40.37 | 78.31% | 60.37% | 68.18% | 74.49% | 6,607 | 1.4554 | 1,103 | 22.02 ms | 45.4 |

```
                       mAP50 Sensitivity by Gate Intervention
          100 ┌────────────────────────────────────────────────────────┐
              │                                                        │
           90 │      [IA001: 93.7%]                                    │
              │      ┌────────────┐                                    │
           80 │      │            │                     [IA003: 84.2%] │
              │      │            │                     ┌────────────┐ │
           70 │      │            │      [IA002: 72.3%] │            │ │      [IA004: 78.0%]
              │      │            │      ┌────────────┐ │            │ │      ┌────────────┐
           60 │      │            │      │            │ │            │ │      │            │
              └──────┴────────────┴──────┴────────────┴─┴────────────┴─┴──────┴────────────┴─
                      IA001 (Learned)    IA002 (G_V=0)  IA003 (G_V=0.5)  IA004 (G_V=1.0)
```

---

## 4. Per-Scene Sensitivity Breakdown

The table below breaks down detection performance across the four surveillance camera scenes:

| Gate Condition | Scene 02 (Standard Urban) | Scene 05 (Dark Alley / Distant) | Scene 07 (Crossing Cluster) | Scene 25 (Headlight Glare) |
|:---|:---:|:---:|:---:|:---:|
| **IA001: Learned Gate** | $mAP_{50}$: **94.62%**<br>Rec: **91.54%** \| FP/img: 0.38 | $mAP_{50}$: **97.91%**<br>Rec: **95.39%** \| FP/img: 0.29 | $mAP_{50}$: **87.21%**<br>Rec: **80.79%** \| FP/img: 0.37 | $mAP_{50}$: **94.17%**<br>Rec: **94.20%** \| FP/img: 0.42 |
| **IA002: Gate Zero** | $mAP_{50}$: 62.41%<br>Rec: 38.35% \| FP/img: 0.23 | $mAP_{50}$: 84.94%<br>Rec: 79.25% \| FP/img: 0.73 | $mAP_{50}$: 69.24%<br>Rec: 53.56% \| FP/img: 0.10 | $mAP_{50}$: 73.10%<br>Rec: 62.84% \| FP/img: 0.30 |
| **IA003: Gate Half** | $mAP_{50}$: 84.72%<br>Rec: 71.73% \| FP/img: 0.31 | $mAP_{50}$: 92.91%<br>Rec: 93.33% \| FP/img: 1.08 | $mAP_{50}$: 73.61%<br>Rec: 68.75% \| FP/img: 0.98 | $mAP_{50}$: 87.51%<br>Rec: 87.34% \| FP/img: 0.70 |
| **IA004: Gate One** | $mAP_{50}$: 77.38%<br>Rec: 70.51% \| FP/img: 0.69 | $mAP_{50}$: 90.27%<br>Rec: 90.72% \| FP/img: 1.34 | $mAP_{50}$: 64.08%<br>Rec: 65.73% \| FP/img: 2.17 | $mAP_{50}$: 82.46%<br>Rec: 86.98% \| FP/img: 1.71 |

---

## 5. Detailed Scientific Deductions

### 5.1 Empirical Demonstration of Representation Co-adaptation
- **The Finding**: When visible feature injection is completely severed ($G_V = 0.0$, IA002), overall $mAP_{50}$ collapses by **21.43 percentage points** ($93.70\% \to 72.27\%$), and recall plummets from $90.25\%$ to $59.49\%$. The number of missed pedestrians jumps from 496 to **2,060**.
- **Theoretical Explanation**: This dramatic degradation proves that the thermal stream in E007 did not train in isolation. During training, backpropagation adjusted both backbones and the shared neck simultaneously. Subsequent layers learned to expect non-zero feature inputs from both streams. Removing the optical stream at inference creates an out-of-distribution feature magnitude shift that downstream convolutions cannot decode.
- **Scientific Takeaway**: This conclusively proves that **post-hoc gate zeroing cannot be used as a proxy for an independently trained thermal-only model** (E003, which achieved 94.12% mAP50).

### 5.2 Spurious Detections and False Alarm Escalation ($G_V = 1.0$)
- **The Finding**: When visible features are fully injected without attenuation ($G_V = 1.0$, IA004), false alarms explode:
  - Total candidate detections output by the network surge to **308,818** (vs. 59,853 for learned gate).
  - False positives per image quadruple from **0.37 to 1.46** (2,624 spurious detections).
  - Precision collapses from $87.52\%$ to $60.37\%$.
- **The Mechanism**: In nighttime surveillance imagery, large regions of the visible camera frame consist of deep darkness, sensor photon noise, and low-contrast shadows. When these unmodulated optical features are added directly to the thermal representation with weight $1.0$, noisy spatial gradients corrupt the clean thermal heat signatures and induce false edge responses that trigger high-confidence false pedestrian detections.

### 5.3 Superiority of Content-Aware Adaptive Gating
- The learned gate (**IA001**) decisively outperforms both equal linear blending ($G_V = 0.5$, $+9.46\%$ mAP50 advantage) and direct addition ($G_V = 1.0$, $+15.67\%$ mAP50 advantage).
- By adaptively suppressing visible features on human heat silhouettes ($G_V \approx 0.367$ at P3) while admitting contextual edges in the background ($G_V \approx 0.518$), the learned gate prevents the false-alarm explosion of IA004 and the representation starvation of IA002.

---

## 6. Artifact Verification and Handoff Manifest

All generated result files and plots are verified and stored in:
`research/inference_ablations/`

- [gate_intervention_metrics.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/inference_ablations/gate_intervention_metrics.csv) — Complete summary metrics table for IA001–IA004.
- [gate_intervention_per_scene.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/inference_ablations/gate_intervention_per_scene.csv) — Scene-stratified metrics across Scenes 02, 05, 07, 25.
- [gate_intervention_metrics.json](file:///C:/VI%20Fusion%20for%20LL-PD/research/inference_ablations/gate_intervention_metrics.json) — Full JSON record with optimal operating thresholds and candidate counts.
- [gate_intervention_comparison.png](file:///C:/VI%20Fusion%20for%20LL-PD/research/inference_ablations/gate_intervention_comparison.png) — 3-panel publication-ready comparison plot (mAP, Precision/Recall, and FP vs. Misses).
- [gate_intervention_analysis.md](file:///C:/VI%20Fusion%20for%20LL-PD/research/inference_ablations/gate_intervention_analysis.md) — Detailed standalone engineering report.
- Checkpoint integrity: Checkpoint `best.pt` remains completely unaltered (hash and byte count unchanged).

**Stage 3 Status**: **COMPLETED**.
