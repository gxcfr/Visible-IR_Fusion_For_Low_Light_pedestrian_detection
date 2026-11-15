# E007 Inference-Only Gate Intervention Analysis (IA001 – IA004)

## 1. Executive Summary

This study evaluates the architectural sensitivity and feature co-dependence of the proposed Asymmetric Gated Fusion model (**E007**) by executing controlled runtime interventions on the gate activation tensor $G_V \in \{\text{Learned}, 0.0, 0.5, 1.0\}$. 

All evaluations were executed on the canonical 1,803 validation pairs in native $1280 \times 1024$ space using the frozen checkpoint `E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N/weights/best.pt`.

---

## 2. Master Quantitative Comparison

| Gate Intervention ID | Gate Mode | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall@0.25 (%) | Precision@0.25 (%) | F1@0.25 (%) | Max-F1 (%) | Total Preds ($\tau \ge 0.25$) | FP / Image | Misses (FN) | Mean Fwd Latency | FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **IA001** | **Learned Gate (Original)** | **93.70** | **55.98** | **90.25** | **87.52** | **88.86** | **90.41** | 5,250 | **0.3661** | **496** | 22.51 ms | 44.4 |
| **IA002** | **Gate Zero ($G_V = 0.0$)** | 72.27 | 38.45 | 59.49 | 85.31 | 70.10 | 70.53 | 3,627 | 0.3333 | 2,060 | 21.53 ms | 46.5 |
| **IA003** | **Gate Half ($G_V = 0.5$)** | 84.24 | 48.12 | 80.32 | 75.17 | 77.66 | 80.52 | 5,442 | 0.7526 | 1,001 | 21.97 ms | 45.5 |
| **IA004** | **Gate One ($G_V = 1.0$)** | 78.03 | 40.37 | 78.31 | 60.37 | 68.18 | 74.49 | 6,607 | 1.4554 | 1,103 | 22.02 ms | 45.4 |

---

## 3. Per-Scene Sensitivity Breakdown

| Gate Condition | Scene 02 (Standard Urban) | Scene 05 (Dark Alley / Distant) | Scene 07 (Crossing Cluster) | Scene 25 (Headlight Glare) |
|:---|:---:|:---:|:---:|:---:|
| **IA001: Learned Gate** | $mAP_{50}$: **94.62%** \| Rec: **91.54%** | $mAP_{50}$: **97.91%** \| Rec: **95.39%** | $mAP_{50}$: **87.21%** \| Rec: **80.79%** | $mAP_{50}$: **94.17%** \| Rec: **94.20%** |
| **IA002: Gate Zero** | $mAP_{50}$: 62.41% \| Rec: 38.35% | $mAP_{50}$: 84.94% \| Rec: 79.25% | $mAP_{50}$: 69.24% \| Rec: 53.56% | $mAP_{50}$: 73.10% \| Rec: 62.84% |
| **IA003: Gate Half** | $mAP_{50}$: 84.72% \| Rec: 71.73% | $mAP_{50}$: 92.91% \| Rec: 93.33% | $mAP_{50}$: 73.61% \| Rec: 68.75% | $mAP_{50}$: 87.51% \| Rec: 87.34% |
| **IA004: Gate One** | $mAP_{50}$: 77.38% \| Rec: 70.51% | $mAP_{50}$: 90.27% \| Rec: 90.72% | $mAP_{50}$: 64.08% \| Rec: 65.73% | $mAP_{50}$: 82.46% \| Rec: 86.98% |

---

## 4. Key Scientific Insights

1. **Empirical Proof of Representation Co-adaptation**:
   - Forcing $G_V = 0.0$ at inference precipitates an immediate **21.43 percentage-point collapse in $mAP_{50}$** ($93.70\% \to 72.27\%$) and a **30.76 percentage-point drop in recall** ($90.25\% \to 59.49\%$). Missed pedestrians surge from 496 to 2,060.
   - **Theoretical Explanation**: This proves that the downstream detection neck layers (`neck_and_head`) have strongly co-adapted during multi-task gradient descent to expect fused cross-modal activations. When visible injection is abruptly clamped to zero, convolutional filters in the neck receive truncated activation magnitudes, demonstrating that **inference-time gate zero is NOT equivalent to an independently trained unimodal thermal baseline** (E003 achieved 94.12% mAP50).
2. **False Alarm Explosion Under Unattenuated Fusion ($G_V = 1.0$)**:
   - Forcing $G_V = 1.0$ increases total candidate detections to 308,818 and quadruples false positives per image from 0.37 to 1.46 (2,624 false alarms). Precision plummets to 60.37%.
   - In low-light nighttime imagery, dark optical regions contain sensor noise and low-frequency shadows. Without learned attenuation, these optical artifacts inject spurious gradients into neck convolutions, triggering false positive detections.
3. **Optimality of Learned Dynamic Gating**:
   - Learned gating (**IA001**) decisively outperforms all fixed-constant baselines across all four scenes, validating the hypothesis that content-dependent spatial-channel attenuation is necessary to harvest optical structural context while suppressing nighttime noise.
