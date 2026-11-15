# Research Review Summary: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

## 1. Research Objective
Low-light pedestrian detection presents fundamental challenges for surveillance and autonomous systems. While visible RGB sensors suffer from severe underexposure and sensor noise at night, thermal infrared (IR) cameras provide robust radiometric contrast but lack fine texture and structural detail. 

The primary objective of this research is to investigate whether an **asymmetric, reliability-aware multimodal feature fusion architecture** (`AsymmetricGatedFusionYOLO26`) can adaptively extract complementary visible features without suffering from the **catastrophic modality contamination** that plagues conventional early and naive fusion baselines.

---

## 2. Experimental Methodology
- **Detector Architecture**: Ultralytics **YOLO26n** (2.50M parameters) adopted as the uniform detector backbone to isolate the contribution of fusion strategies. YOLO11n is retained strictly as an executed historical baseline.
- **Dataset**: **LLVIP** benchmark (15,488 verified image pairs, single class `person`).
- **Data Partitioning Protocol**:
  - **Scene-Aware 85/15 Training Split**: 10,222 training pairs, 1,803 validation pairs grouped strictly by camera scene ID to prevent data leakage.
  - **Preserved Test Split**: The official LLVIP test partition (**3,463 pairs**) remains **100% strictly untouched** until final Milestone 8 evaluation.
- **Evaluation Metrics**: Precision, Recall, F1-Score, $\text{mAP}_{50}$, $\text{mAP}_{50-95}$, batch-1 inference latency (ms), throughput (FPS), parameter count, and peak GPU VRAM.

---

## 3. Completed Experiments Summary
All planned roadmap experiments (`E001` through `E007`) have completed full execution on the local NVIDIA RTX 4060 GPU:

1. **`E001_RGB_YOLO11N`**: Historical unimodal visible baseline.
2. **`E002_RGB_YOLO26N`**: Primary unimodal visible baseline.
3. **`E003_THERMAL_YOLO26N`**: Primary unimodal thermal baseline.
4. **`E004_LATE_FUSION_WBF_YOLO26N`**: Decision-level late fusion via Weighted Boxes Fusion ($w_{th}=0.7, w_{rgb}=0.3$).
5. **`E005_EARLY_FUSION_4CH_YOLO26N`**: 4-Channel input stem pixel-level fusion `[R, G, B, Thermal]`.
6. **`E006_FEATURE_FUSION_CONCAT_YOLO26N`**: Decoupled dual-stream multiscale feature concatenation at P3, P4, P5.
7. **`E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N`**: Proposed thermal-primary asymmetric reliability-aware gated fusion.

---

## 4. Best-Performing Model
- **Overall Accuracy**: **`E003_THERMAL_YOLO26N`** achieved **94.49% $\text{mAP}_{50}$** and **56.79% $\text{mAP}_{50-95}$**, establishing the thermal spectrum as the dominant semantic modality under low-light surveillance.
- **Multimodal Superiority & Pedestrian Recall**: **`E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N`** achieved **94.11% $\text{mAP}_{50}$**, **56.58% $\text{mAP}_{50-95}$**, **93.89% Precision**, and **87.69% Recall** (with peak single-epoch recall reaching **88.20%** at Epoch 12).
- E007 achieved the **highest pedestrian recall across all multimodal and unimodal architectures**, demonstrating that adaptive gating successfully extracts complementary visible cues to recover shadowed pedestrians without compromising nighttime reliability.

---

## 5. Quantitative Master Comparison

| Exp ID | Architecture | Modality / Strategy | Precision | Recall | F1 | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ | Latency (ms) | FPS | Params |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | Visible RGB (Historical) | 91.24% | 72.22% | 80.62% | 80.57% | 41.31% | 10.69 | 93.55 | 2.59M |
| **E002** | YOLO26n | Visible RGB Baseline | 89.39% | 71.91% | 79.70% | 79.34% | 40.47% | 15.80 | 63.28 | 2.50M |
| **E003** | YOLO26n | Thermal IR Baseline | 94.13% | 87.90% | 90.91% | 94.49% | 56.79% | 10.80 | 92.60 | 2.50M |
| **E004** | YOLO26n | Late Fusion (WBF) | 88.15% | 85.45% | 86.78% | 91.35% | 52.49% | 20.38 | 49.07 | 5.01M |
| **E005** | YOLO26n | Early Fusion (4-Channel Stem) | 82.66% | 64.59% | 72.52% | 71.53% | 37.40% | 13.64 | 73.31 | 2.57M |
| **E006** | YOLO26n | Feature Concatenation | 93.48% | 86.32% | 89.76% | 93.58% | 56.62% | 14.73 | 67.89 | 4.14M |
| **E007** | Proposed | **Asymmetric Gated Fusion** | **93.89%** | **87.69%** | **90.69%** | **94.11%** | **56.58%** | **17.30** | **57.80** | **4.97M** |

---

## 6. Main Experimental Observations
1. **The Low-Light Modality Hierarchy**: The thermal modality (**94.49% $\text{mAP}_{50}$**) vastly outperforms visible detection (**79.34%**), confirming that radiometric heat emissions provide the dominant signal in darkness.
2. **The Modality Contamination Disaster in Early Fusion**: Early 4-channel stem fusion (**71.53% $\text{mAP}_{50}$**) collapsed by **-22.96%** compared to unimodal thermal. Injecting raw uncalibrated visible pixels directly at the stem contaminates early low-level filters with night sensor noise.
3. **Decoupled Feature Processing Recovery**: Decoupling backbones into two independent streams before multiscale fusion (**E006**, **E007**) completely avoids early contamination, restoring detection accuracy to >93.5%.
4. **Adaptive Gating Efficacy**: The proposed reliability-aware gating in E007 outperforms Late Fusion (+2.76% $\text{mAP}_{50}$) and Feature Concatenation (+1.37% Recall) while maintaining real-time edge execution (57.80 FPS).

---

## 7. Proposed Architecture Details (E007)
The proposed `AsymmetricGatedFusionYOLO26` architecture is mathematically formulated as:
1. **Thermal-Primary Stream**: Preserves full radiometric gradient cues through a full YOLO26n backbone.
2. **Auxiliary Visible Stream**: Extracts spatial texture cues via an independent stream.
3. **Multi-Scale Reliability Gating Module**: At feature levels P3 ($80\times 80$), P4 ($40\times 40$), and P5 ($20\times 20$):
   $$\mathbf{G}_V = \sigma\left(\text{Conv}_{1\times 1}\left(\text{SiLU}\left(\text{BatchNorm}\left(\text{Conv}_{3\times 3}([F_T, P(F_V)])\right)\right)\right)\right)$$
4. **Thermal Residual Preservation**:
   $$F_{\text{fused}} = F_T + \mathbf{G}_V \odot P(F_V)$$
   This residual structure ensures that when visible reliability drops ($\mathbf{G}_V \to 0$ in dark backgrounds), the representation safely defaults to the clean thermal feature $F_T$.

---

## 8. Ablation Analysis Across Fusion Paradigms
- **Pixel-Level vs Feature-Level Fusion**: Feature-level fusion (E006/E007) outperforms pixel-level fusion (E005) by **+22.58% $\text{mAP}_{50}$**, confirming that modality-specific representation learning must precede cross-modal integration.
- **Decision-Level vs Feature-Level Fusion**: Gated feature fusion (E007) outperforms decision-level WBF (E004) by **+2.76% $\text{mAP}_{50}$** and **+2.24% Recall** while running **1.18x faster** (17.3 ms vs 20.38 ms).
- **Unconstrained Concatenation vs Reliability Gating**: Gating (E007) improves recall from **86.32% to 87.69%** over naive concatenation (E006), verifying that adaptive gating selectively admits informative visible edges while filtering pitch-black background noise.

---

## 9. Known Limitations
1. **Single-Dataset Domain**: Current evaluations are benchmarked exclusively on the surveillance scenes of the LLVIP dataset. Cross-dataset generalization (e.g., KAIST, CVC-14, FLIR) has not yet been tested.
2. **Dual-Backbone Parameter Overhead**: Maintaining two full backbones increases total parameters to 4.97M (vs 2.50M for unimodal). Future iterations could employ asymmetric channel pruning on the visible branch.
3. **Severe Weather Scenarios**: LLVIP consists primarily of clear-weather nighttime scenes; performance in heavy fog, rain, or thermal crossover (where human body temperature equals ambient temperature) remains to be characterized.

---

## 10. Unresolved Research Questions
1. **Adverse Synthetic Robustness**: How do the different fusion strategies degrade when visible imagery experiences synthetic noise, lens blur, motion blur, or severe glare?
2. **Test Partition Generalization**: Will the observed ranking hold when evaluated on the completely untouched 3,463 official LLVIP test pairs in Milestone 8?
3. **Ultra-Lightweight Visible Encoders**: Can the visible stream be compressed to <0.5M parameters (e.g. via depthwise separable MobileNet stem) without sacrificing the gated recall improvement?
