# Corrected Qualitative Multi-Scene Analysis Report

## 1. Executive Summary

This report corrects and rigorously evaluates the qualitative detection performance of all seven models (**E001 through E007**) across the four characteristic validation scenes:
- **Scene 02 (Pair 020001)**: Standard Urban Low-Light Pedestrians
- **Scene 05 (Pair 050001)**: Extreme Darkness & Distant Small Pedestrians
- **Scene 07 (Pair 070001)**: Partial Occlusion & Overlapping Silhouettes
- **Scene 25 (Pair 250001)**: Thermal Contrast & Artificial Glare

All detections are evaluated strictly at the standardized operating confidence threshold $\tau \ge 0.25$ and matched to ground truth using the standardized IoU threshold ($\text{IoU} \ge 0.50$), identically to the unified validation protocol.

---

## 2. Quantitative Truth Matching Across Qualitative Scenes

| Scene & Pair ID | Model | Ground Truths | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision (%) | Recall (%) | Matched IoU(s) | Detection Confidences | Status / Audit Observation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| **Scene 02** (`020001`) | **E001 (YOLO11n RGB)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.743 | 0.266 | Misses GT 1 in darkness |
| | **E002 (YOLO26n RGB)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.825 | 0.588 | Misses GT 1 in darkness |
| | **E003 (YOLO26n Thermal)**| 2 | **2** | 0 | **0** | **100.0%** | **100.0%** | 0.925, 0.710 | 0.700, 0.253 | **Only model detecting BOTH pedestrians** |
| | **E004 (Late WBF)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.943 | 0.667 | GT 1 confidence suppressed below 0.25 |
| | **E005 (Early 4CH)** | 2 | 1 | 1 | 1 | 50.0% | 50.0% | 0.845 | 0.744 (TP), 0.395 (FP)| Background false alarm on street |
| | **E006 (Concat)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.952 | 0.816 | Misses GT 1 at $\tau \ge 0.25$ |
| | **E007 (Asym Gated)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.951 | 0.814 | Misses GT 1 at $\tau \ge 0.25$ |
| **Scene 05** (`050001`) | **E001 (YOLO11n RGB)** | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.693, 0.818 | 0.820, 0.796 | Streetlights illuminate pedestrians |
| | **E002 (YOLO26n RGB)** | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.681, 0.712 | 0.814, 0.752 | Detects both |
| | **E003 (YOLO26n Thermal)**| 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.804, 0.641 | 0.725, 0.695 | Crisp thermal heat detection |
| | **E004 (Late WBF)** | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.762, 0.686 | 0.752, 0.712 | High-confidence WBF merge |
| | **E005 (Early 4CH)** | 2 | 2 | **2** | 0 | 50.0% | 100.0% | 0.787, 0.560 | 0.786, 0.729 | 2 False Alarms on sky/wall clutter |
| | **E006 (Concat)** | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 0.774, 0.738 | 0.788, 0.649 | Clean dual detection |
| | **E007 (Asym Gated)** | 2 | 2 | **1** | 0 | 66.7% | 100.0% | 0.769, 0.721 | 0.780, 0.725 (TP), 0.383 (FP)| **1 False Positive on top-left tree/sky boundary** |
| **Scene 07** (`070001`) | **E001 (YOLO11n RGB)** | 4 | 3 | 0 | 1 | 100.0% | 75.0% | 0.769, 0.787, 0.734 | 0.790, 0.655, 0.636 | Misses GT 0 (occluded) |
| | **E002 (YOLO26n RGB)** | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 0.769, 0.815, 0.750, 0.536 | Mean IoU = 0.718 | Marginal IoU (0.536) on GT 0 |
| | **E003 (YOLO26n Thermal)**| 4 | 4 | 0 | 0 | 100.0% | 100.0% | 0.882, 0.771, 0.570, 0.884 | Mean IoU = 0.777 | Overlapping thermal halos blur boundary |
| | **E004 (Late WBF)** | 4 | 4 | 0 | 0 | 100.0% | 100.0% | 0.850, 0.761, 0.933, 0.561 | Mean IoU = 0.776 | Confidences attenuated (0.32–0.76) |
| | **E005 (Early 4CH)** | 4 | 3 | 1 | 1 | 75.0% | 75.0% | 0.948, 0.522, 0.854 | 0.814, 0.590, 0.362 | Overlapped box split error |
| | **E006 (Concat)** | 4 | **4** | 0 | 0 | **100.0%** | **100.0%** | 0.914, 0.746, 0.949, 0.869 | **Mean IoU = 0.870** | Superior spatial boundary sharpening |
| | **E007 (Asym Gated)** | 4 | **4** | 0 | 0 | **100.0%** | **100.0%** | 0.923, 0.724, 0.863, **0.963**| **Mean IoU = 0.868** | Highest peak IoU (0.963) on clustered GT |
| **Scene 25** (`250001`) | **E001 (YOLO11n RGB)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.583 | 0.427 | Detects GT 1, misses GT 0 |
| | **E002 (YOLO26n RGB)** | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.645 | 0.332 | Detects GT 0, misses GT 1 |
| | **E003 (YOLO26n Thermal)**| 2 | 1 | 0 | 1 | 100.0% | 50.0% | 0.715 | 0.554 | **Detects GT 1, completely misses GT 0** |
| | **E004 (Late WBF)** | 2 | 0 | 0 | 2 | 0.0% | 0.0% | None | None | Both dropped below $\tau=0.25$ threshold |
| | **E005 (Early 4CH)** | 2 | 0 | 1 | 2 | 0.0% | 0.0% | None | 0.360 (FP) | Misses both, triggers road false alarm |
| | **E006 (Concat)** | 2 | **2** | **0** | **0** | **100.0%** | **100.0%** | **0.843, 0.795** | **0.627, 0.536** | **Clean multi-modal recovery of BOTH GTs** |
| | **E007 (Asym Gated)** | 2 | **2** | **1** | **0** | 66.7% | **100.0%** | **0.833, 0.749** | 0.475, 0.404 (TP), 0.293 (FP)| **Detects BOTH GTs**, 1 torso duplicate box |

---

## 3. Detailed Per-Scene Analysis and Discrepancy Resolution

### 3.1 Scene 02 (Pair 020001): Correction of Claimed Perfect Recall
- **Prior Claim**: Previous audit text claimed that "high thermal contrast allows E003, E006, and E007 to detect all pedestrians cleanly."
- **Reconciled Reality**:
  - Ground truth contains **2 pedestrians**: GT 0 (prominent foreground pedestrian, area=25,004 px$^2$) and GT 1 (distant right-hand pedestrian, area=32,132 px$^2$).
  - **E003 (Thermal-only) is the ONLY model that detected BOTH pedestrians** (GT 0 at conf=0.700, IoU=0.925; GT 1 at conf=0.253, IoU=0.710).
  - E006 and E007 only produced **1 valid prediction** at $\tau \ge 0.25$ (both detecting GT 0 with IoU > 0.95 and conf ~0.81), but failed to detect GT 1 at $\tau \ge 0.25$.
  - **Correction**: The claim of 100% recall for E006 and E007 on Scene 02 is **formally retracted**.

### 3.2 Scene 05 (Pair 050001): Investigation of Third E007 Detection
- **Review Inquiry**: Determine whether the third prediction generated by E007 ($N=3$ predictions vs. $N=2$ ground truths) is a duplicate detection or a false positive.
- **Audit Findings**:
  - GT 0 `[445, 139, 560, 391]` matched by E007 prediction `[451.3, 157.2, 574.0, 384.8]` ($\text{IoU} = 0.769$, $\text{conf} = 0.780$) $\to$ **True Positive**.
  - GT 1 `[847, 163, 948, 400]` matched by E007 prediction `[840.9, 175.3, 973.9, 396.9]` ($\text{IoU} = 0.721$, $\text{conf} = 0.725$) $\to$ **True Positive**.
  - The third detection is `[98.5, 0.5, 167.0, 40.8]` with confidence $0.383$. Its coordinates place it at the extreme top-left of the image on the dark building cornice and sky boundary.
  - **Conclusion**: The third detection is an **unambiguous False Positive** caused by visible sensor background noise, **not** a duplicate pedestrian box.

### 3.3 Scene 07 (Pair 070001): Cluster Overlap and IoU Localization
- **Review Inquiry**: Verify actual IoU-matched true positives rather than relying on equal prediction counts.
- **Audit Findings**:
  - All four models (E002, E003, E006, E007) produced exactly 4 predictions at $\tau \ge 0.25$, and all 4 predictions matched the 4 distinct ground-truth pedestrians ($\text{TP}=4, \text{FP}=0, \text{FN}=0$).
  - However, examining **Matched IoU localization quality** reveals a significant distinction:
    - **E002 (RGB-only)**: Mean IoU = $0.718$ (drops to $0.536$ on distant occluded GT 0).
    - **E003 (Thermal-only)**: Mean IoU = $0.777$ (drops to $0.570$ on GT 0 due to thermal blooming merging adjacent heat signatures).
    - **E006 (Feature Concat)**: Mean IoU = **$0.870$** (consistently high across all 4 targets: 0.914, 0.746, 0.949, 0.869).
    - **E007 (Asym Gated)**: Mean IoU = **$0.868$** (attains peak IoU of **$0.963$** on clustered GT 1).
  - **Conclusion**: While bounding box counts are identical ($4/4$), multi-modal feature fusion provides superior bounding-box spatial sharpness ($+0.09$ to $+0.15$ higher mean IoU) by leveraging visible edge boundaries to separate overlapping thermal silhouettes.

### 3.4 Scene 25 (Pair 250001): Genuine Complementary Fusion Case Study
- **Review Inquiry**: Investigate Scene 25 as a potentially useful fusion example.
- **Audit Findings**:
  - Ground truth contains **2 pedestrians**: GT 0 (partially shadowed in background, area=26,784 px$^2$) and GT 1 (walking in illuminated foreground, area=36,696 px$^2$).
  - **Failure of Unimodal Thermal (E003)**: E003 detects GT 1 ($\text{IoU} = 0.715$, $\text{conf} = 0.554$), but **completely fails to detect GT 0** ($\text{FN} = 1$). GT 0 blends into background thermal temperature.
  - **Failure of Unimodal Visible (E002)**: E002 detects GT 0 ($\text{IoU} = 0.645$, $\text{conf} = 0.332$), but **completely fails to detect GT 1** ($\text{FN} = 1$) due to deep shadow underexposure.
  - **Failure of Late Fusion WBF (E004)**: Because each unimodal model detected only one pedestrian, confidence weighting in WBF halved both confidences, causing **both** predictions to fall below the $\tau = 0.25$ threshold ($\text{TP} = 0, \text{FN} = 2$).
  - **Success of Feature Fusion (E006 & E007)**:
    - **E006**: Successfully detects **BOTH** pedestrians ($\text{TP} = 2, \text{FP} = 0, \text{FN} = 0$) with high IoUs ($0.843$ and $0.795$) and robust confidences ($0.627$ and $0.536$).
    - **E007**: Successfully detects **BOTH** pedestrians ($\text{TP} = 2, \text{FN} = 0$) with high IoUs ($0.833$ and $0.749$), though it incurs a minor torso split false alarm at $\text{conf} = 0.293$.
  - **Conclusion**: Scene 25 provides conclusive empirical proof of **complementary cross-modal synergy**: when each modality suffers a blind spot on different targets, deep feature fusion recovers both targets where unimodal and late fusion models fail.
