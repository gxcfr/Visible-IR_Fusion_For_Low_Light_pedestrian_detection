# Comprehensive Research Audit and Result Harmonization Report

**Phase 2: External Review Response, Unified Validation, and Architectural Audit**  
**Project**: Visible-Infrared Fusion for Low-Light Pedestrian Detection (LLVIP Benchmark)  
**Date of Audit**: November 15, 2026  
**Auditor**: Antigravity Research Review Team  
**Audit Scope**: Experiments E001 through E007  

---

## 1. Executive Summary and Audit Overview

Following an external academic review of experiments **E001 through E007**, several critical methodological discrepancies, documentation inconsistencies, and evaluation divergence factors were identified. 

In strict adherence to academic rigor:
1. **Zero Checkpoint Overwrites**: All historical model checkpoints (`best.pt`, `last.pt`), training logs, and configuration files have been preserved unmodified.
2. **Untouched Test Partition**: The official LLVIP test set (3,463 image pairs) remains completely untouched and sealed.
3. **No Unapproved Training**: No new training experiments have been executed; all findings stem directly from auditing existing artifacts.
4. **Unified Validation Framework**: A standardized, single evaluation pipeline was built from first principles to evaluate all seven models on all 1,803 validation pairs in original $1280 \times 1024$ image coordinates with exact unletterboxing and descending-IoU matching.
5. **Architectural Clarification**: The proposed model (E007) has been formally audited. It is confirmed to feature dual symmetric backbones with feature-conditioned adaptive residual gating, rather than an explicit quality estimator ($Q_T, Q_V$) or a lightweight asymmetric visible encoder.

---

## 2. Unified Validation Methodology

### 2.1 The Historical Discrepancy
In earlier experiment reports:
- Unimodal baselines (E002, E003) were evaluated using Ultralytics native `model.val()` which applies letterboxed square tensors ($640 \times 640$) and native COCO-style greedy IoU matching.
- Multimodal experiments (E004, E005, E006, E007) were evaluated via custom evaluation scripts (`evaluate_detections()`) using confidence-greedy matching and normalized square coordinate scaling.
- Experiment E004 (WBF) converted normalized coordinates to $640 \times 640$ square space, introducing an aspect ratio distortion relative to the true $1280 \times 1024$ ground truth.

### 2.2 Standardized Harmonization Protocol
To guarantee 100% scientific comparability across all 7 experiments:
- **Dataset**: Identical 1,803 image pairs from the LLVIP validation split (5,086 ground truth pedestrian annotations).
- **Coordinate System**: True original image coordinate system ($1280 \times 1024$).
- **Bounding Box Recovery**: Precise letterbox unscaling reversing padding and resizing ($640 \times 640 \to 1280 \times 1024$).
- **Assignment Algorithm**: Descending IoU assignment identical to standard COCO / Ultralytics protocol across 10 IoU thresholds (0.50 to 0.95 at 0.05 increments).
- **Operating Points**: Precision, Recall, and F1 documented at standard confidence threshold $\tau = 0.25$, accompanied by optimal Max-F1 operating thresholds.
- **Detections Cap**: Consistent maximum of 300 detections per image.

---

## 3. Harmonized Cross-Experiment Validation Metrics

The table below presents the official unified validation results across all completed experiments. All models were evaluated under the exact same protocol:

| Exp ID | Architecture / Model | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Prec@0.25 (%) | Rec@0.25 (%) | F1@0.25 (%) | Max-F1 (%) | Total Preds |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n (Baseline) | Visible (RGB) | 71.69 | 34.65 | 90.11 | 63.51 | 74.51 | 74.81 | 3,584 |
| **E002** | YOLO26n (Baseline) | Visible (RGB) | 73.04 | 35.42 | 86.43 | 65.71 | 74.66 | 74.74 | 3,868 |
| **E003** | YOLO26n (Thermal Baseline) | Thermal (IR) | **94.12** | 55.96 | 89.99 | **90.48** | **90.24** | **90.50** | 5,114 |
| **E004** | YOLO26n (Late Fusion WBF) | RGB + IR | 93.74 | 53.80 | **96.27** | 78.15 | 86.27 | 89.85 | 4,127 |
| **E005** | YOLO26n (Early Fusion 4CH) | RGB + IR | 72.24 | 37.45 | 73.41 | 69.90 | 71.61 | 73.25 | 231,221 |
| **E006** | YOLO26n (Feature Concat) | RGB + IR | 93.24 | **56.09** | 89.64 | 88.50 | 89.07 | 89.47 | 5,021 |
| **E007** | Proposed Asym Gated Fusion | RGB + IR | 93.70 | 55.98 | 87.52 | 90.25 | 88.86 | 90.41 | 5,245 |

*Note: All metrics computed exclusively on `best.pt` checkpoints on 1,803 validation pairs in original $1280 \times 1024$ space.*

---

## 4. In-Depth Discussion of Metric Changes and Key Findings

### 4.1 Clarification of E003 (Thermal) vs. E007 (Proposed Fusion)
- **Empirical Reality**: Under standardized evaluation, **Thermal-only (E003: 94.12% $mAP_{50}$)** slightly outperforms **Proposed Fusion (E007: 93.70% $mAP_{50}$)** by 0.42 percentage points, and equals it in high-IoU localization ($mAP_{50-95}$: 55.96% vs 55.98%).
- **Physical Reason in LLVIP**: The LLVIP benchmark consists exclusively of surveillance scenes captured in outdoor urban night settings. In deep nighttime environments, thermal infrared radiation provides an exceptionally crisp, high-contrast signature for human body heat, while visible cameras suffer from heavy sensor noise, underexposure, and motion blur. Adding visible information directly into an uncorrupted thermal signal inevitably injects a small degree of background noise.
- **Scientific Honesty**: Prior informal claims that E007 achieved superior $mAP_{50}$ over thermal-only arose from mismatched coordinate systems (evaluating E003 under square letterbox matching while evaluating E007 under custom greedy matching). When evaluated under identical conditions, thermal alone is already near saturation on this dataset. E007 matches thermal recall (90.25% vs 90.48%) while maintaining high localization accuracy.

### 4.2 Failure Mode of Early Fusion (E005)
- Early Fusion achieves only **72.24% $mAP_{50}$** and spawned **231,221 detections** (over 128 candidate boxes per image).
- Blending 3 RGB channels with 1 thermal channel at the first convolutional layer destroys the clean gradient structure of thermal heat signatures. Low-confidence visible noise triggers thousands of false alarms across dark asphalt and walls.

### 4.3 Behavior of Late Fusion WBF (E004)
- Late Fusion achieves the highest precision of any model (**96.27%**), but suffers lower recall (**78.15%**).
- Because WBF matches and merges bounding boxes from both models, pedestrians that fail detection in the RGB branch (which only has 65.7% recall) often receive lower fusion confidence weights, causing them to drop below detection thresholds.

---

## 5. Architectural Audit of E007 Proposed Fusion

### 5.1 Verification of Explicit Quality Descriptors ($Q_T, Q_V$)
- **Audit Finding**: The implementation in `src/models/fusion/asymmetric_gated_fusion_yolo26.py` **does not** compute or consume explicit scalar or spatial quality descriptors $Q_T, Q_V$.
- **Correct Description**: The model utilizes a **learned feature-conditioned cross-modal gating module**. Gating coefficients are generated dynamically by processing concatenated intermediate features $[F_T, P(F_V)]$.

### 5.2 Verification of Stream Symmetry
- **Audit Finding**: The visible branch is **not** a lightweight encoder. It is an exact duplicate of the 11-layer YOLO26n backbone (1,365,472 parameters each).
- **Total Parameters**: The model contains **4,903,902 parameters**:
  - Thermal Backbone: 1,365,472 params (27.8%)
  - Visible Backbone: 1,365,472 params (27.8%)
  - Multi-Scale Gating Modules (P3, P4, P5): 1,034,240 params (21.1%)
  - Shared Neck & Head: 1,138,718 params (23.2%)
- **Nature of Asymmetry**: The asymmetry is **functional/semantic**: thermal infrared acts as the uninterrupted base feature trunk ($F_T$), while visible features ($F_V$) are projected and adaptively injected via gated residual modulation ($G_V \odot P(F_V)$).

### 5.3 Mathematical Formulation
The feature fusion at stage $l \in \{\text{P3}, \text{P4}, \text{P5}\}$ is strictly defined as:

$$F_{\text{fused}}^{(l)} = F_T^{(l)} + G_V^{(l)} \odot P^{(l)}\left(F_V^{(l)}\right)$$

Where:
- $P^{(l)}(F_V) = \text{SiLU}\left(\text{BatchNorm2d}\left(\text{Conv}_{1\times 1}(F_V^{(l)})\right)\right)$
- $G_V^{(l)} = \sigma\left(\text{Conv}_{1\times 1}\left(\text{SiLU}\left(\text{BatchNorm2d}\left(\text{Conv}_{3\times 3}\left([F_T^{(l)}, P^{(l)}(F_V^{(l)})]\right)\right)\right)\right)\right) \in (0, 1)$
- $\odot$ denotes element-wise spatial and channel Hadamard multiplication.

---

## 6. Extended Gate Activation Analysis

An extended statistical analysis across all 1,803 validation scenes was conducted to determine how gate weights $G_V$ behave under varying environmental conditions:

### 6.1 Environmental Correlations and Foreground vs. Background

| Feature Level | Foreground Mean ($\pm$ std) | Background Mean ($\pm$ std) | Corr with RGB Brightness | Corr with RGB Contrast | Corr with IR Contrast |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **P3** ($80 \times 80$) | $0.3668 \pm 0.160$ | $0.5167 \pm 0.111$ | -0.2608 | -0.2597 | -0.0632 |
| **P4** ($40 \times 40$) | $0.3601 \pm 0.144$ | $0.4578 \pm 0.113$ | -0.1983 | -0.2114 | -0.0984 |
| **P5** ($20 \times 20$) | $0.4430 \pm 0.150$ | $0.4907 \pm 0.117$ | +0.1601 | +0.1332 | -0.0210 |

### 6.2 Scale-Dependent Gate Activations (P3 Level)
- **Small Pedestrians** ($area < 32^2$): $0.3392 \pm 0.155$
- **Medium Pedestrians** ($32^2 \le area < 96^2$): $0.3670 \pm 0.160$
- **Large Pedestrians** ($area \ge 96^2$): $0.3702 \pm 0.161$

### 6.3 Scientific Deductions
1. **Not Calibrated Probabilities**: Gate values do not represent physical sensor reliability probabilities; background regions exhibit an activation floor around $0.45 - 0.52$.
2. **Foreground Attenuation**: Gating values inside pedestrian bounding boxes are systematically lower ($~0.36$) than background regions ($~0.51$). The network actively learns to dampen noisy visible pedestrian features to avoid corrupting the clean thermal pedestrian profile.
3. **Weak Environmental Correlation**: Correlation with visible brightness is slightly negative at P3 ($-0.26$), indicating the network relies on spatial structural cues rather than raw scalar luminance.

---

## 7. Training Convergence and Checkpoint Selection

- **Trajectory Verification**: Training loss in both E006 and E007 decreased monotonically through Epoch 17 (from 9.10 to 6.37).
- **Validation Peaking**: Validation fitness ($0.1 \times \text{mAP}_{50} + 0.9 \times \text{mAP}_{50-95}$) reached its global maximum at **Epoch 5** (Fitness: 0.6033).
- **Early Stopping**: With an early-stopping patience of 12 epochs, no subsequent epoch exceeded Epoch 5's fitness, triggering termination at Epoch 17.
- **Integrity Guarantee**: All validation metrics reported in this audit are derived strictly from the official `best.pt` checkpoint (Epoch 5). No metrics are cherry-picked from secondary epochs.

---

## 8. Standardized Latency and Computational Benchmarking

Latency was benchmarked on an NVIDIA GeForce RTX 4060 Laptop GPU under strict scientific controls: batch size 1, 50 warmup iterations, 200 timed iterations, `torch.cuda.synchronize()`, and full end-to-end accounting (including preprocessing, model forward pass, postprocessing, NMS, and WBF).

| Exp ID | Architecture | Modality | Params (M) | FLOPs (G) | Model Fwd (ms) | Model FPS | End-to-End (ms) | End-to-End FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | RGB | 2.59 | 6.4 | $7.59 \pm 0.40$ | 131.8 | $9.97 \pm 0.54$ | 100.3 |
| **E002** | YOLO26n | RGB | 2.37 | 5.2 | $10.76 \pm 0.81$ | 92.9 | $13.03 \pm 0.93$ | 76.7 |
| **E003** | YOLO26n | Thermal | 2.37 | 5.2 | $9.83 \pm 0.46$ | 101.7 | $12.31 \pm 0.62$ | 81.3 |
| **E004** | YOLO26n WBF | RGB + IR | 4.74 | 10.4 | $20.38 \pm 0.65$ | 49.1 | $23.89 \pm 0.83$ | 41.9 |
| **E005** | YOLO26n Early | RGB + IR | 2.37 | 5.2 | $10.16 \pm 0.46$ | 98.4 | $14.41 \pm 0.69$ | 69.4 |
| **E006** | YOLO26n Concat | RGB + IR | 3.87 | 8.5 | $13.39 \pm 0.58$ | 74.7 | $17.48 \pm 0.77$ | 57.2 |
| **E007** | Proposed Asym | RGB + IR | 4.90 | 9.8 | $14.17 \pm 0.75$ | 70.6 | $17.94 \pm 0.89$ | 55.7 |

### Benchmarking Insights:
1. **Unimodal Parity**: E002 (RGB) and E003 (Thermal) share identical architecture and parameter count; their model-only latencies are within ~0.9 ms of each other under thermal steady state.
2. **Late Fusion Overhead**: Executing actual WBF fusion across dual inference streams incurs a measurable overhead, bringing E004 end-to-end latency to 23.89 ms (41.9 FPS).
3. **Proposed Fusion Efficiency**: E007 executes in **17.94 ms end-to-end (55.7 FPS)**, comfortably exceeding real-time requirements (30 FPS) while running end-to-end within a single unified computational graph.

---

## 9. Qualitative Multi-Scene Analysis

High-resolution visual comparisons across all 7 models were generated for four characteristic validation scenes:
1. **Scene 02 (Standard Urban Street)**:
   - High thermal contrast allows E003, E006, and E007 to detect all pedestrians cleanly.
   - E001 and E002 miss dark-clothed pedestrians in deep shadow.
2. **Scene 05 (Distant Small Pedestrians)**:
   - Demonstrates the critical value of thermal sensing; visible models completely fail to distinguish distant pedestrians ($area < 32^2$).
   - E007 and E003 detect all distant individuals.
3. **Scene 07 (Occluded Pedestrian Cluster)**:
   - Pedestrians walking in close proximity create overlapping thermal silhouettes.
   - Feature fusion (E006 and E007) leverages visible boundary edges to resolve individual bounding box boundaries more sharply than unimodal thermal.
4. **Scene 25 (Thermal Glare / Headlight Washout)**:
   - Vehicle headlights create intense thermal blooming, partially saturating the thermal sensor.
   - E007 successfully suppresses blooming artifacts by fusing visible textural cues, avoiding false positive splits.

---

## 10. Proposed Research Directions: Option A vs. Option B

We present two clear research pathways for senior researcher evaluation:

### Option A: Retain E007 as Feature-Conditioned Adaptive Fusion (Recommended)
- **Concept**: Accept E007 as an empirical model of **learned feature-conditioned cross-modal residual gating**.
- **Action**: Retain existing weights and architecture; correct all manuscript text to accurately describe it as an adaptive feature-conditioned fusion mechanism without claiming explicit $Q_T, Q_V$ estimation.
- **Advantage**: Fully backed by completed, reproducible experiments (E001–E007). Zero retraining required. Directly answers the core hypothesis.

### Option B: Design True Explicit Quality-Guided Fusion (Future Phase)
- **Concept**: Explicitly engineer a standalone **Quality Estimation Module (QEM)** that measures quantitative modality attributes (e.g., local image entropy, Michelson contrast, gradient energy, spatial illuminance).
- **Action**: Implement QEM as a separate module, assign a new experiment identifier (**E008**), conduct ablation studies isolating the quality estimator, and train on the LLVIP training set.
- **Requirement**: Requires researcher approval and significant compute budget for multi-epoch training.

---

## 11. Required Documentation and Repository Corrections

The following corrections are now mandated across all project documentation:
1. **Research Objective**: Formally update to:
   *"Investigate whether reliability-aware asymmetric visible-infrared fusion improves pedestrian detection under challenging illumination conditions relative to unimodal and conventional fusion baselines."*
2. **Metric Claims**: Cease claiming that E007 beats thermal-only on LLVIP. State clearly that Thermal-only (E003: 94.12%) is the strongest unimodal baseline, and E007 achieves comparable high-accuracy performance (93.70%) while providing robust cross-modal feature injection.
3. **Architecture Description**: Cease claiming lightweight visible backbones or explicit $Q_T, Q_V$ descriptors. Describe the true dual-symmetric backbone and feature-conditioned gating equation.
4. **Test Set Partition**: Reiterate that the official LLVIP test set (3,463 pairs) remains sealed and untouched.

---

## 12. Audit Sign-Off and Index of Artifacts

The following audited artifacts are permanently preserved in `research/audit_phase/`:
- `unified_validation_metrics.csv` & `.json` — Harmonized validation metrics.
- `evaluation_protocol.md` — Complete formal validation protocol.
- `predictions/` — Full raw prediction JSON files for all 7 experiments.
- `standardized_latency_report.md` & `.json` — Rigorous GPU latency benchmark.
- `convergence_analysis.md` — Detailed training convergence and early stopping audit.
- `architecture_audit_report.md` — Code-level graph and parameter verification report.
- `gate_analysis/` — Extended statistical distributions and correlation plots.
- `qualitative_analysis/` — High-resolution 8-panel detection comparison imagery.
