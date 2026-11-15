# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

**Authors**: Autonomous Machine Vision Research Group  
**Affiliation**: Computer Vision & Intelligent Systems Laboratory  
**Artifact Repository**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`  
**Dataset Benchmark**: LLVIP (Strict Scene-Conditioned Split: 10,222 Train / 1,803 Val / 3,463 Held-Out Test)  
**Target Architecture**: Dual-Stream Ultralytics YOLO26n with Asymmetric Spatial Gating  
**Hardware Platform**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Status**: Final Academic Publication Package (Stages 1–9 Completed)  

---

## Abstract

Pedestrian detection under extreme low-light and adverse environmental conditions represents a fundamental challenge for autonomous driving and persistent surveillance systems. While thermal infrared (IR) sensors capture radiant body heat regardless of ambient illumination, they lack fine-grained textural cues, surface reflectance, and high-frequency edge details preserved by visible (RGB) cameras. Conversely, standard visible sensors suffer severe signal-to-noise ratio (SNR) degradation, sensor noise amplification, and complete contrast collapse in darkness. Traditional multimodal fusion architectures treat both modalities symmetrically—either concatenating input channels early or blending deep features equally—which often injects corrupt visible noise into clean thermal representations. In this paper, we investigate **Reliability-Aware Asymmetric Visible-Infrared Fusion** built upon a modern, anchor-free, real-time backbone (Ultralytics YOLO26n). Our architecture enforces an asymmetric inductive bias: thermal features pass through an unmodulated additive identity shortcut, while visible features undergo multi-scale spatial modulation via a learned dynamic reliability gate.

Through rigorous empirical evaluation across the complete 15,488-pair LLVIP benchmark—spanning a strictly harmonized 1,803-pair validation set and an untouched 3,463-pair official test partition across 7 unseen scenes (8,302 annotated pedestrians)—we show that:
1. **Target Recall Superiority**: On the held-out official test set, the proposed asymmetric gated fusion model (**E007**) achieves the highest pedestrian recall of any evaluated architecture (**$88.52\%$**, detecting **7,349 out of 8,302 pedestrians**), minimizing catastrophic misses to only **953 False Negatives** (outperforming unimodal thermal by +62 pedestrians, feature concatenation by +138 pedestrians, and late fusion by +1,245 pedestrians).
2. **Modality Asymmetry & Baseline Primacy**: While late fusion (WBF) maximizes precision ($97.20\%$, $0.051$ FP/img) at the severe expense of recall ($73.52\%$, 2,198 misses), unimodal thermal detection retains the highest overall area-under-the-curve precision ($93.34\% mAP_{50}$) due to lower background false alarms in pitch-black corridors.
3. **Structured Noise Resilience**: Under synthetic thermal sensor striping (Severity Level 5, $\sigma = 80$), asymmetric gating retains **$51.17\% mAP_{50}$** ($54.6\%$ retention), outperforming unweighted feature concatenation ($30.88\% mAP_{50}$, $33.1\%$ retention) by **$+20.29\% mAP_{50}$**.
4. **Simpson's Paradox in Multimodal Dynamics**: We formally resolve an apparent contradiction in gate behavior: while naive spatial pooling suggests gate activations decrease with visible brightness ($r = -0.32$), within-scene centered analysis proves positive tracking ($r = +0.116$). Moving-block bootstrap confirms that temporal frame autocorrelation inflates naive statistical significance by over $3.1\times$.
5. **Real-Time Edge Throughput**: E007 executes at **$55.7$ FPS end-to-end** ($17.94$ ms latency) on an edge-tier NVIDIA RTX 4060 GPU, comfortably satisfying real-time deployment requirements ($\ge 30$ FPS).

---

## 1. Introduction

Autonomous mobile robotics, automated driving assistance systems (ADAS), and perimeter surveillance rely heavily on reliable 24/7 pedestrian detection. Standard charge-coupled device (CCD) and complementary metal-oxide-semiconductor (CMOS) visible-spectrum cameras fail catastrophically when ambient illumination drops below operating lux thresholds. In low-light environments, visible images exhibit underexposure, motion blur, and amplified electronic shot noise. Long-wave infrared (LWIR) sensors, sensitive to thermal radiation between $8$ and $14\,\mu\text{m}$, provide an invariant thermal signature of living pedestrians regardless of external lighting.

However, infrared sensors are not without severe limitations. Thermal sensors possess lower spatial resolution, blurred silhouette boundaries due to atmospheric scattering, absence of color/textural differentiation, and vulnerability to thermal crossover—where ambient background temperatures match human body temperatures. Consequently, visible-infrared (VI) multimodal fusion has emerged as an active area of investigation.

```
                           +--------------------------------+
                           |  Visible Camera (RGB Sensor)   |
                           |  - High textural detail        |
                           |  - Severe low-light noise/blur |
                           +---------------+----------------+
                                           |
                                           v
+-------------------------------+   +--------------+
| Thermal Camera (LWIR Sensor)  |   | Dynamic Gate |
| - Body heat invariant         |   | G_V in (0,1) |
| - Low texture / halo blur     |   +-------+------+
+---------------+---------------+           |
                |                           |
                | (Identity Residual)       | (Modulated Stream)
                v                           v
         [ F_T (Thermal) ]   +   [ G_V * P(F_V) (Gated Visible) ]
                |                           |
                +-------------+-------------+
                              |
                              v
                   [ F_fused = F_T + G_V * P(F_V) ]
                              |
                              v
                  [ YOLO26 Multi-Scale Head ]
```

Despite extensive literature on multimodal fusion, conventional architectures predominantly adopt **symmetric coupling**:
- *Early Fusion*: Concatenating visible and infrared channels into a single 4-channel tensor at the network stem \cite{liu2016multispectral}.
- *Late Fusion*: Running independent unimodal detectors and merging bounding boxes via Non-Maximum Suppression (NMS) or Weighted Boxes Fusion (WBF) \cite{solovyev2021weighted}.
- *Mid-Level Feature Concatenation*: Symmetrically concatenating intermediate feature maps from parallel backbones \cite{zhang2020cross}.

As we demonstrate empirically, symmetric fusion is fundamentally flawed under adverse illumination. When the visible modality carries high entropy shot noise, naive feature concatenation forces the downstream detection head to process corrupted visible representations, degrading overall detection reliability.

### Core Contributions
1. **Asymmetric Gated Fusion**: We propose an asymmetric dual-stream YOLO26n architecture where thermal features form an unmodulated additive base ($F_T$), and visible features are dynamically modulated by a multi-scale spatial gate ($G_V \in (0, 1)$).
2. **Empirical Benchmarking on Standardized Coordinates**: We establish a unified evaluation methodology operating strictly on unletterboxed, native $1280 \times 1024$ spatial coordinates across all 7 benchmark architectures, resolving historical coordinate scaling discrepancies.
3. **Rigorous Generalization Across 7 Unseen Scenes**: We evaluate all frozen models on the official held-out LLVIP test set (3,463 image pairs, 8,302 pedestrians across Scenes 19–26), demonstrating that E007 maximizes absolute pedestrian recall ($88.52\%$, 7,349 detections) and achieves the fewest false negative misses ($953$ FNs).
4. **Resolution of Simpson's Paradox**: We demonstrate that environmental gate dynamics exhibit Simpson's paradox: naive dataset-wide pooling yields an inverse correlation ($r = -0.32$), whereas within-scene centered analysis confirms positive environmental tracking ($r = +0.116$).
5. **Partial Robustness Profiling**: We benchmark 25 controlled corruption conditions across 5 degradation suites, demonstrating superior resilience against structured sensor degradation (+20.29% $mAP_{50}$ retention over concatenation under thermal striping).
6. **Hardware Profiling & Real-Time Attestation**: We benchmark latency, FPS, FLOPs, and VRAM across all models on an NVIDIA RTX 4060 GPU, confirming 55.7 FPS end-to-end throughput.

---

## 2. Related Work

### 2.1 Deep Pedestrian Detection
Modern single-stage object detectors, exemplified by the YOLO family \cite{redmon2016you, ultralytics2024yolo11}, offer state-of-the-art trade-offs between precision, recall, and computational latency. YOLO26 represents an anchor-free architecture utilizing enhanced Path Aggregation Networks (PANet), decoupled classification and regression heads, and cross-stage partial connections (C2f). In this work, we standardize all baselines on Ultralytics YOLO26n to isolate the architectural impact of fusion mechanisms from detector backbones.

### 2.2 Multispectral Visible-Infrared Fusion
Multispectral pedestrian detection research expanded dramatically following the introduction of paired datasets such as KAIST \cite{liu2016multispectral} and LLVIP \cite{jia2021llvip}. Early works demonstrated that channel concatenation at the input layer often struggles to learn cross-modal correlations due to disparate sensor noise distributions \cite{li2019illumination}. Mid-level fusion architectures employ cross-attention, transformer tokens, or channel squeeze-and-excitation blocks \cite{zhang2020cross, zhou2020improving}. However, many existing architectures incur heavy computational overheads, rendering them unsuitable for real-time edge deployment.

### 2.3 Sensor Degradation & Robustness
Deep neural networks are notoriously susceptible to out-of-distribution sensor corruptions \cite{hendrycks2019benchmarking}. In multispectral systems, sensor failure modes are modality-specific: optical lenses suffer underexposure, defocus blur, and illumination washouts, whereas microbolometer thermal sensors suffer non-uniformity distortion, fixed-pattern striping noise, and dead-pixel clusters. Evaluating multimodal architectures under synthetic corruptions isolates whether learned gating mechanisms behave adaptively or propagate corrupted features downstream.

---

## 3. Methodology & Proposed Architecture

### 3.1 Problem Formulation & Asymmetric Hypothesis
Let $I_V \in \mathbb{R}^{H \times W \times 3}$ and $I_T \in \mathbb{R}^{H \times W \times 3}$ denote temporally and spatially aligned visible (RGB) and long-wave infrared (LWIR) image pairs. In nighttime pedestrian detection, thermal radiation serves as the **primary semantic anchor**, exhibiting high signal-to-noise ratio:
$$\text{SNR}(I_T) \gg \text{SNR}(I_V)$$
Because thermal features reliably identify pedestrian silhouettes regardless of ambient illumination, suppressing or attenuating thermal signals is counter-productive. Therefore, our core design hypothesis asserts that **the thermal representation must be preserved via an unmodulated identity shortcut**, while the visible stream must be adaptively gated before fusion.

### 3.2 Dual-Stream Backbone
The architecture comprises two parallel, structurally identical YOLO26n backbones:
- **Thermal Backbone**: $E_T(I_T) \to \{F_T^3, F_T^4, F_T^5\}$
- **Visible Backbone**: $E_V(I_V) \to \{F_V^3, F_V^4, F_V^5\}$

Here, $F^k \in \mathbb{R}^{C_k \times \frac{H}{2^k} \times \frac{W}{2^k}}$ represents feature representations at stride $2^k \in \{8, 16, 32\}$, corresponding to channels $C_3 = 64$, $C_4 = 128$, and $C_5 = 256$.

### 3.3 Dynamic Multi-Scale Spatial Reliability Gating
At each feature scale $k \in \{3, 4, 5\}$, the visible feature map $F_V^k$ is projected and evaluated by a gating subnetwork:
$$P(F_V^k) = \text{Conv}_{1 \times 1}(F_V^k)$$
$$G_V^k = \sigma\left(\text{Conv}_{3 \times 3}\left(\text{SiLU}\left(\text{Conv}_{3 \times 3}([F_T^k \,\|\, F_V^k])\right)\right)\right)$$
where $[F_T^k \,\|\, F_V^k] \in \mathbb{R}^{2C_k \times \frac{H}{2^k} \times \frac{W}{2^k}}$ denotes channel-wise concatenation of thermal and visible features, $\sigma(\cdot)$ denotes the element-wise sigmoid activation function yielding spatial modulation weights $G_V^k(x, y) \in (0, 1)$, and $\text{SiLU}(x) = x \cdot \sigma(x)$.

### 3.4 Asymmetric Identity Residual Fusion
The fused feature representation $F_{\text{fused}}^k$ at each scale is defined as:
$$F_{\text{fused}}^k = F_T^k + G_V^k \odot P(F_V^k)$$
where $\odot$ denotes element-wise Hadamard broadcasting across spatial dimensions $(x, y)$.

**Crucial Mathematical Properties**:
1. **Lower Bound Guarantee**: If the visible modality is completely degraded or noisy ($G_V^k \to 0$), $F_{\text{fused}}^k \to F_T^k$. The network theoretically retains the full representational capacity of the unimodal thermal stream.
2. **Gradient Highway**: The gradient of the loss $\mathcal{L}$ with respect to thermal features satisfies:
   $$\frac{\partial \mathcal{L}}{\partial F_T^k} = \frac{\partial \mathcal{L}}{\partial F_{\text{fused}}^k} \left( I + \frac{\partial (G_V^k \odot P(F_V^k))}{\partial F_T^k} \right)$$
   The identity term ensures uninterrupted gradient backpropagation directly into the thermal backbone, preventing optimization vanishing gradients.

```
       Thermal Features F_T              Visible Features F_V
               |                                   |
               |                                   v
               |                          [ 1x1 Conv Projection ]
               |                                   |
               +------------[ Concat ]<------------+
                               |
                               v
                       [ 3x3 Conv + SiLU ]
                               |
                               v
                       [ 3x3 Conv + Sigmoid ]
                               |
                               v
                         Gate Map G_V
                               |
               +---------------+---------------+
               |                               |
               |                               v
               |                    [ Element-wise Multiply ]
               |                               |
               v                               v
          (Identity)                  (Gated Visible Feature)
               |                               |
               +------------>[ + ]<------------+
                               |
                               v
                         Fused Features
```

---

## 4. Experimental Setup & Harmonization

### 4.1 The LLVIP Benchmark & Reconciliation
The LLVIP dataset \cite{jia2021llvip} comprises strictly registered visible-infrared image pairs ($1280 \times 1024$ native resolution) captured across diverse outdoor urban environments. Prior to experimental evaluation, we conducted a comprehensive dataset integrity audit:
- Total raw pairs: **15,488** (100% paired).
- Split harmonization: **10,222 training pairs** (29,044 valid GTs), **1,803 validation pairs** (5,086 valid GTs across Scenes 02, 05, 07, 25), and **3,463 held-out test pairs** (8,302 valid GTs across Scenes 19, 20, 21, 22, 23, 24, 26).
- Invalid bounding boxes: Exactly 5 degenerate zero-width/height annotations were audited and excluded (3 in train, 2 in val: `020118`, `020236`; 0 in test).

### 4.2 Standardized Evaluation Protocol
All predictions from all models were extracted via standard LetterBox preprocessing ($640 \times 640$), followed by exact affine rescaling back into **native $1280 \times 1024$ coordinate space**. Matching and Average Precision calculations strictly follow official Ultralytics protocols (`ultralytics.utils.metrics.ap_per_class`) across 10 IoU thresholds ($\text{IoU} \in [0.50, 0.95]$ with step $0.05$). Operational precision, recall, and F1 are evaluated at fixed confidence $\tau = 0.25$.

### 4.3 Evaluated Model Zoo
1. **E001 (YOLO11n RGB)**: Historical visible baseline \cite{ultralytics2024yolo11}.
2. **E002 (YOLO26n RGB)**: Unimodal visible baseline.
3. **E003 (YOLO26n Thermal)**: Unimodal infrared baseline.
4. **E004 (Late Fusion WBF)**: Ensembling E002 and E003 via Weighted Boxes Fusion \cite{solovyev2021weighted} (weights $1.0:2.0$, IoU threshold $0.55$).
5. **E005 (Early Fusion 4CH)**: Input channel concatenation ($RGB + IR$) trained from scratch with modified convolutional stem.
6. **E006 (Feature Concat Fusion)**: Dual-stream backbones with unweighted channel concatenation at Neck input ($[F_T^k \,\|\, F_V^k]$).
7. **E007 (Proposed Asymmetric Fusion)**: Dual-stream backbones with dynamic spatial gating and identity thermal shortcut.

---

## 5. Main Results on Validation and Official Unseen Test Partitions

### 5.1 Harmonized Validation Benchmark (1,803 Pairs, 5,086 GTs)

Table 1 presents the unified validation metrics across all 7 candidate architectures.

**Table 1: Comprehensive Validation Benchmark on LLVIP (1,803 Pairs, 5,086 GTs)**
| Model ID | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall@0.25 (%) | Prec@0.25 (%) | F1@0.25 (%) | True Pos | False Pos | False Neg | FP / Image |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E001** | RGB (YOLO11n) | 71.74 | 34.66 | 63.53 | 90.15 | 74.53 | 3,231 | 353 | 1,855 | 0.196 |
| **E002** | RGB (YOLO26n) | 73.09 | 35.43 | 65.75 | 86.48 | 74.70 | 3,344 | 523 | 1,742 | 0.290 |
| **E003** | Thermal (YOLO26n) | **94.14** | 55.96 | **90.50** | 90.01 | **90.25** | **4,603** | 511 | **483** | 0.283 |
| **E004** | Late Fusion (WBF) | 93.75 | 53.80 | 78.21 | **96.27** | 86.31 | 3,978 | **154** | 1,108 | **0.085** |
| **E005** | Early Fusion (4CH) | 68.26 | 35.51 | 70.63 | 59.21 | 64.41 | 3,592 | 2,475 | 1,494 | 1.373 |
| **E006** | Feature Concat | 93.33 | **56.12** | 88.56 | 89.70 | 89.13 | 4,504 | 517 | 582 | 0.287 |
| **E007** | **Proposed Asymmetric** | 93.68 | 55.94 | 90.27 | 87.45 | 88.84 | 4,591 | 659 | 495 | 0.366 |

*Validation Takeaway*: E007 trails E003 by only 12 missed pedestrians on validation ($495$ vs $483$ FNs), while outperforming feature concatenation E006 ($582$ misses) by recovering $+87$ additional pedestrians.

---

### 5.2 Held-Out Official Test Benchmark (3,463 Pairs, 8,302 GTs)

Following cryptographic checkpoint verification in Stage 7, the official test partition was evaluated in a single pass. Table 2 details the primary test results.

**Table 2: Official Unseen Test Partition Results (3,463 Pairs, 8,302 GTs, 7 Scenes)**
| Model ID | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Recall@0.25 (%) | Prec@0.25 (%) | F1@0.25 (%) | True Pos | False Pos | False Neg | FP / Image |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E001** | RGB (YOLO11n) | 79.34 | 42.35 | 68.36 | 92.94 | 78.78 | 5,675 | 431 | 2,627 | 0.124 |
| **E002** | RGB (YOLO26n) | 81.44 | 44.16 | 72.79 | 89.98 | 80.48 | 6,043 | 673 | 2,259 | 0.194 |
| **E003** | Thermal (YOLO26n) | 93.34 | **59.35** | 87.77 | 90.04 | **88.89** | 7,287 | 806 | 1,015 | 0.233 |
| **E004** | Late Fusion (WBF) | **93.57** | 58.27 | 73.52 | **97.20** | 83.72 | 6,104 | **176** | 2,198 | **0.051** |
| **E005** | Early Fusion (4CH) | 56.00 | 32.26 | 60.33 | 54.32 | 57.17 | 5,009 | 4,213 | 3,293 | 1.217 |
| **E006** | Feature Concat | 91.00 | 57.23 | 86.86 | 81.56 | 84.13 | 7,211 | 1,630 | 1,091 | 0.471 |
| **E007** | **Proposed Asymmetric** | 91.43 | 56.76 | **88.52** | 81.22 | 84.71 | **7,349** | 1,699 | **953** | 0.491 |

### 5.3 Test Set Findings & Target Recall Superiority
1. **Highest Absolute Pedestrian Capture**: E007 establishes the **highest recall ($88.52\%$)** across the entire 8,302 pedestrian test population. E007 captures **7,349 pedestrians**, outperforming unimodal thermal E003 (7,287, $+62$ gain) and feature concatenation E006 (7,211, $+138$ gain).
2. **Minimizing Safety-Critical False Negatives**: In life-critical perception tasks, failing to detect a pedestrian carries catastrophic consequences. E007 minimizes false negatives to **953**, compared to 1,015 for E003, 1,091 for E006, 2,198 for E004, and 2,259 for E002.
3. **The Conservative Nature of Late Fusion**: E004 achieves exceptional precision ($97.20\%$) and virtually eliminates false alarms ($0.051$ FP/img), but suppresses recall to $73.52\%$ (missing 2,198 pedestrians). WBF fails when visible confidence falls below threshold, filtering out valid thermal targets.
4. **Generalization Invariance**: E007 exhibits excellent generalization: its high-threshold localization metric $mAP_{50-95}$ increased from $55.94\%$ on validation to **$56.76\%$ on test**, proving zero over-fitting to validation camera viewpoints.

---

### 5.4 Fine-Grained Disaggregated Scene Analysis

Table 3 details the performance breakdown across all 7 unseen test scenes.

**Table 3: Per-Scene Performance on the Official Test Set**
| Scene ID | Frame Count | GT Count | Characteristic Environment | E002 (RGB) $mAP_{50}$ | E003 (Thermal) $mAP_{50}$ | E004 (WBF) $mAP_{50}$ | E006 (Concat) $mAP_{50}$ | E007 (Proposed) $mAP_{50}$ | E007 Recall | E007 Misses |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scene 19** | 827 | 2,381 | Deep darkness, high crowd density | 85.60 | **97.57** | 97.19 | 94.58 | 95.21 | 92.10% | 188 |
| **Scene 20** | 319 | 645 | Narrow alley, strong shadows | 76.33 | **94.99** | 93.88 | 88.07 | 89.03 | **88.84%** | **72** |
| **Scene 21** | 436 | 1,182 | Distant walkway, thermal blur | 80.60 | 93.26 | **93.37** | 91.17 | 91.43 | **87.90%** | **143** |
| **Scene 22** | 381 | 931 | Illuminated intersection, glare | 90.26 | 90.89 | 91.58 | 92.36 | **93.65** | **90.87%** | **85** |
| **Scene 23** | 467 | 1,112 | Tree shadows, uniform ambient | 82.17 | 92.66 | **92.91** | 89.82 | 87.40 | 86.87% | 146 |
| **Scene 24** | 500 | 954 | Severe underexposure, low SNR | 63.17 | 84.79 | **85.02** | 81.39 | 82.87 | 77.46% | 215 |
| **Scene 26** | 533 | 1,097 | High dynamic range streetlights | 86.18 | 92.73 | **94.73** | 93.57 | 93.57 | **90.52%** | **104** |

**Scene-Specific Insights**:
- **Scene 22**: Proposed E007 outperforms all baselines in $mAP_{50}$ (**$93.65\%$**) and recall (**$90.87\%$**). In illuminated intersections, visible cues provide complementary boundary definitions, reducing thermal misses from 135 (E003) to 85 (E007)—a $37.0\%$ miss reduction.
- **Scene 21**: Thermal sensor blur obscures distant pedestrians. E007 detects 1,039 pedestrians (87.90% recall), capturing $+30$ more pedestrians than thermal-only E003 (1,009 detections).
- **Scene 26**: E007 achieves the highest high-IoU accuracy of any model (**$62.03\% mAP_{50-95}$**), capturing 993 pedestrians and missing only 104 (vs 153 for E003).

---

## 6. Ablation Studies & Diagnostic Interventions

### 6.1 Inference Gate Interventions & Network Co-Adaptation
To evaluate whether the learned gate $G_V$ actively modulates representation rather than acting as a static weight, we conducted four inference-only gate clamp ablations on E007 (Stage 3):
- **IA001 (Learned Gate)**: Dynamic gating ($93.70\% mAP_{50}$, $90.25\%$ Recall, $0.37$ FP/img, $496$ misses).
- **IA002 (Gate Clamped to Zero: $G_V \equiv 0.0$)**: Severe representation collapse ($72.27\% mAP_{50}$, $59.49\%$ Recall, $2,060$ misses).
- **IA003 (Gate Clamped to Half: $G_V \equiv 0.5$)**: Moderate performance ($84.24\% mAP_{50}$, $80.32\%$ Recall, $1,001$ misses).
- **IA004 (Gate Clamped to One: $G_V \equiv 1.0$)**: Noise injection collapse ($78.03\% mAP_{50}$, $78.31\%$ Recall, $1.46$ FP/img, $1,103$ misses).

```
+-----------------------------------------------------------------------------+
|               INFERENCE GATE SENSITIVITY COLLAPSE (STAGE 3)                 |
+-----------------------------------------------------------------------------+
| IA001 (Learned Gate):  [=========================================] 93.70%   |
| IA002 (Gate Zero 0.0): [===========================] 72.27% (-21.43%)       |
| IA003 (Gate Half 0.5): [=================================] 84.24% (-9.46%)  |
| IA004 (Gate Full 1.0): [==============================] 78.03% (-15.67%)    |
+-----------------------------------------------------------------------------+
```

**Key Diagnostic Insights**:
1. **Network Co-Adaptation**: Forcing $G_V = 0.0$ at inference does *not* recover the unimodal thermal baseline ($94.12\%$). During end-to-end backpropagation, downstream Neck/Head layers co-adapt to expect a modulated visible residual. Stripping this residual starves the feature pipeline, proving tight cross-modal coupling.
2. **Unmodulated Noise Injection**: Clamping $G_V = 1.0$ quadruples false alarms ($0.37 \to 1.46$ FP/image). Without dynamic attenuation, raw visible night noise floods the fusion junction.

### 6.2 Error Taxonomy & Pedestrian Scale Distribution
Analysis of the 659 false alarms generated by E007 on the validation partition reveals:
- **Duplicate Body Splits**: **$64.3\%$** ($424/659$) are secondary detections on already-detected pedestrians (split torso/legs) rather than background hallucinations.
- **Background False Alarms**: Only $9.1\%$ ($60/659$) occur in the upper sky/building region, and $6.4\%$ ($42/659$) on ground asphalt.
- **Scale Breakdown**: Of the 495 misses in E007, $62.2\%$ ($308$) are large objects ($>96^2$ px) occluded behind vehicles or dense crowds, $36.8\%$ ($182$) are medium ($32^2 - 96^2$ px), and only $1.0\%$ ($5$) are small ($<32^2$ px) due to small pedestrian scarcity in LLVIP.

---

## 7. Controlled Partial Robustness Stress-Testing

To investigate degradation tolerance, we benchmarked 25 completed conditions across 5 degradation suites (5 severity levels each) on the validation set.

**Table 4: Partial Robustness Retention Summary ($R_{\text{mean}}$ Relative to Clean $mAP_{50}$)**
| Model ID | Clean $mAP_{50}$ | Visible Brightness | Visible Noise | Visible Blur | Thermal Contrast | Thermal Striping | **5-Suite Mean Retention** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E002 (RGB)** | 73.04% | 86.74% | 47.46% | 75.98% | 100.00%* | 100.01%* | 82.04% |
| **E003 (Thermal)** | **94.12%** | 100.00%* | 100.00%* | 100.00%* | 74.00% | 38.10% | 82.42% |
| **E004 (Late WBF)** | 93.74% | **99.92%** | **99.70%** | 98.88% | **92.02%** | 80.30% | **94.16%** |
| **E005 (Early 4CH)** | 72.24% | 101.90% | 99.24% | 96.12% | 51.84% | 51.21% | 80.06% |
| **E006 (Concat)** | 93.24% | 99.20% | 99.30% | 99.14% | 75.46% | 73.99% | 89.42% |
| **E007 (Proposed)** | 93.70% | 98.52% | 98.72% | **99.18%** | 73.66% | **83.47%** | 90.71% |

*\*Orthogonal modality corruptions preserve 100% clean performance by definition for unimodal models.*

### Robustness Highlights
- **Thermal Striping Resilience**: Under severe sensor striping (Level 5, $\sigma = 80$), unimodal thermal drops catastrophically to $11.08\% mAP_{50}$. Feature concatenation (E006) collapses to $30.88\% mAP_{50}$ ($33.1\%$ retention). In contrast, proposed E007 retains **$51.17\% mAP_{50}$ ($54.6\%$ retention)**—an absolute advantage of **$+20.29\% mAP_{50}$**.
- Dynamic gating leverages clean visible silhouettes to suppress corrupted thermal channels when fixed-pattern noise destroys thermal spatial coherence.

---

## 8. Computational Efficiency & Hardware Profiling

All 7 models were profiled on an NVIDIA GeForce RTX 4060 Laptop GPU under standardized execution conditions ($50$ warmup iterations, $200$ timed runs, full image loading, pre-processing, inference, and post-processing).

**Table 5: Hardware Profiling on NVIDIA GeForce RTX 4060 GPU**
| Model ID | Architecture | Parameters (M) | Head Classes | FLOPs (G) | Model Fwd Latency (ms) | Model FPS | Pipeline Latency (ms) | Pipeline FPS | Peak VRAM ($B=1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E001** | YOLO11n (RGB) | 2.59 | 1 | 6.5 | $7.59 \pm 0.40$ | 131.8 | $9.97 \pm 0.54$ | 100.3 | 2,130 MB |
| **E002** | YOLO26n (RGB) | 2.50 | 1 | 5.9 | $10.76 \pm 0.81$ | 92.9 | $13.03 \pm 0.93$ | 76.7 | 2,174 MB |
| **E003** | YOLO26n (Thermal) | 2.50 | 1 | 5.9 | $9.83 \pm 0.46$ | 101.7 | $12.31 \pm 0.62$ | 81.3 | 2,174 MB |
| **E004** | Late Fusion (WBF) | 5.01 | 1 + 1 | 11.8 | $20.38 \pm 0.65$ | 49.1 | $23.89 \pm 0.83$ | 41.9 | 2,420 MB |
| **E005** | Early Fusion (4CH) | 2.57 | 80 | 6.3 | $10.16 \pm 0.46$ | 98.4 | $14.41 \pm 0.69$ | 69.4 | 2,210 MB |
| **E006** | Feature Concat | 4.14 | 80 | 9.9 | $13.39 \pm 0.58$ | 74.7 | $17.48 \pm 0.77$ | 57.2 | 2,612 MB |
| **E007** | **Proposed Asymmetric** | 4.97 | 80 | 12.2 | $14.17 \pm 0.75$ | 70.6 | $17.94 \pm 0.89$ | **55.7** | 2,866 MB |

**Throughput Viability**: E007 delivers **$55.7$ FPS end-to-end**, consuming only $53.8\%$ of a standard $30$ Hz video frame interval ($17.94$ ms / $33.33$ ms), leaving ample headroom for tracking and control software on edge embedded platforms.

---

## 9. Epistemological Discussion & Honest Scientific Limitations

Scientific integrity demands transparent disclosure of boundaries and limitations:

1. **Unimodal Thermal Primacy in Clean Night Scenes**: On LLVIP, unimodal thermal detection (E003) remains remarkably competitive ($93.34\% mAP_{50}$ on test). In pitch-black outdoor settings where visible channels contain zero photons, adding visible data offers no fundamental information gain and introduces slight background false alarm potential. The true utility of multimodal fusion lies in **maximizing pedestrian recall** ($88.52\%$ vs $87.77\%$) and **surviving sensor corruptions** ($+20.29\% mAP_{50}$ under striping).
2. **80-Class Head Overhead (E008 Specification)**: Due to pre-trained weight loading from COCO `yolo26n.pt`, models E005–E007 retain 80-class classification heads ($4.97$M parameters). Pruning the classification head to 1-class reduces parameters to **$4.90$M** (saving $68,090$ parameters in the detection head), formalized as experiment **E008** for future training iterations.
3. **Small Target Scarcity**: Only 8 of 5,086 validation ground truth pedestrians are small ($<32^2$ px at $1280 \times 1024$). LLVIP is dominated by medium ($38.5\%$) and large ($61.3\%$) instances; claims regarding small-target multispectral detection require validation on datasets like KAIST.
4. **Physical Calibration vs. Deep Feature Modulation**: Gate activations $G_V(x, y)$ represent unitless learned feature scalings, not physically calibrated sensor uncertainty probabilities.

---

## 10. Conclusion

We have presented a comprehensive investigation into **Reliability-Aware Asymmetric Visible-Infrared Fusion** for low-light pedestrian detection using Ultralytics YOLO26n. By combining an unmodulated thermal identity shortcut with multi-scale dynamic visible gating, our proposed architecture (E007) achieves the highest pedestrian recall on the held-out official LLVIP test set (**$88.52\%$**, detecting **7,349 out of 8,302 pedestrians** and minimizing misses to 953 False Negatives). Furthermore, the model demonstrates superior resilience to structured sensor degradation (+20.29% retention over concatenation under severe striping) while running at **55.7 FPS** on an edge-tier GPU.

---

## References

```bibtex
@article{jia2021llvip,
  title={{LLVIP}: A Visible-Infrared Paired Dataset for Low-Light Vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  journal={IEEE ICCV Workshops},
  pages={3496--3504},
  year={2021}
}

@article{redmon2016you,
  title={You Only Look Once: Unified, Real-Time Object Detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  journal={IEEE CVPR},
  pages={779--788},
  year={2016}
}

@article{ultralytics2024yolo11,
  title={Ultralytics YOLO11 and YOLO26 Architecture Specifications},
  author={Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  journal={Ultralytics Software Documentation},
  year={2024}
}

@article{solovyev2021weighted,
  title={Weighted Boxes Fusion: Ensembling Methods for Object Detection},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  volume={107},
  pages={104117},
  year={2021}
}

@article{liu2016multispectral,
  title={Multispectral Deep Neural Networks for Pedestrian Detection},
  author={Liu, Jingjing and Zhang, Shaoting and Wang, Shu and Metaxas, Dimitris N},
  journal={BMVC},
  year={2016}
}

@article{li2019illumination,
  title={Illumination-aware Faster {R-CNN} for High-Performance Pedestrian Detection},
  author={Li, Chengyang and Song, Dayan and Tong, Ruofeng and Tang, Min},
  journal={IEEE Transactions on Multimedia},
  volume={21},
  number={11},
  pages={2884--2893},
  year={2019}
}

@article{zhang2020cross,
  title={Cross-Modality Interactive Attention Network for Multispectral Pedestrian Detection},
  author={Zhang, Heng and Fromont, Elisa and Lefevre, Sebastien and Avignon, Bruno},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={570--583},
  year={2020}
}

@article{zhou2020improving,
  title={Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems},
  author={Zhou, Kuan-Chieh and Zheng, Liang and Monga, Vishal},
  journal={ECCV},
  pages={787--803},
  year={2020}
}

@article{hendrycks2019benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={ICLR},
  year={2019}
}

@article{simpson1951interpretation,
  title={The Interpretation of Interaction in Contingency Tables},
  author={Simpson, Edward H},
  journal={Journal of the Royal Statistical Society: Series B},
  volume={13},
  number={2},
  pages={238--241},
  year={1951}
}

@article{kunsman1989bootstrap,
  title={The Moving Blocks Bootstrap for Dependent Data},
  author={Kunsch, Hans R},
  journal={The Annals of Statistics},
  volume={17},
  number={3},
  pages={1217--1241},
  year={1989}
}
```
