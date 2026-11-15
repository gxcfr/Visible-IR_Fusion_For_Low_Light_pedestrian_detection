# Phase A / Stage 1: Master Documentation & Experimental Audit Reconciliation

**Project**: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection  
**Benchmark**: LLVIP Paired Dataset (`C:\LLVIP\LLVIP`)  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace), Windows 11  
**Software Environment**: PyTorch 2.5.1+cu121, CUDA 12.1, OpenCV 5.0.0, Ultralytics 8.4.144  
**Date**: November 15, 2026  
**Auditor**: Antigravity Research Engineering Team  

---

## 1. Executive Summary & Mandatory Execution Constraints

In accordance with the **Master Research Execution Directive**, the model-training pipeline is officially and permanently frozen for the current research cycle. 

### Mandatory Governance Rules:
1. **Zero New Training**: Experiments E008 (1-class gated fusion), E009 (1-class feature concatenation), five-seed repeatability training, training-time fixed-gate ablations, and explicit quality estimator training are formally **DEFERRED**.
2. **Preservation of Completed Work**: All seven existing experiment directories (**E001 through E007**), checkpoints (`best.pt`, `last.pt`), training logs, metrics, and validation predictions are preserved in their original state.
3. **Official Test Partition Sealed**: The 3,463 official LLVIP testing pairs remain strictly locked and unaccessed. Model freeze and explicit approval are required before opening the test set in Stage 8.
4. **No Dataset Duplication**: The canonical dataset resides exclusively at `C:\LLVIP\LLVIP`. No re-downloading or full local duplication has occurred or will occur.

---

## 2. Canonical Harmonized Validation Metrics (E001–E007)

The table below records the verified, canonical validation metrics evaluated across all **1,803 validation image pairs** (5,086 ground-truth pedestrians) in the **native $1280 \times 1024$ coordinate system**. 

These numbers originate directly from the canonical evaluation output files:
- `research/audit_phase/unified_validation_metrics.csv`
- `research/audit_phase/unified_validation_metrics.json`

### Master Validation Results Table

| Exp ID | Architecture Description | Input Modality | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) | Precision (@0.25) | Recall (@0.25) | F1 (@0.25) | Max-F1 (%) | Optimal Conf ($\tau$) | Total Preds ($\tau \ge 0.25$) | Total Candidates Evaluated |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n *(Baseline)* | Visible (RGB) | 71.69 | 34.65 | 90.11% | 63.51% | 74.51% | 74.81% | 0.1912 | 3,584 | 39,504 |
| **E002** | YOLO26n *(Baseline)* | Visible (RGB) | 73.04 | 35.42 | 86.43% | 65.71% | 74.66% | 74.74% | 0.2843 | 3,868 | 44,311 |
| **E003** | YOLO26n *(Baseline)* | Thermal (IR) | **94.12** | 55.96 | 89.99% | **90.48%** | **90.24%** | **90.50%** | 0.3293 | 5,114 | 34,035 |
| **E004** | YOLO26n *(Late WBF)* | RGB + IR | 93.74 | 53.80 | **96.27%** | 78.15% | 86.27% | 89.85% | 0.1582 | 4,127 | 44,120 |
| **E005** | YOLO26n *(Early 4CH)* | RGB + IR | 72.24 | 37.45 | 73.41% | 69.90% | 71.61% | 73.25% | 0.3704 | 4,845 | 231,221 |
| **E006** | YOLO26n *(Feature Concat)*| RGB + IR | 93.24 | **56.09** | 89.64% | 88.50% | 89.07% | 89.47% | 0.3173 | 5,021 | 60,664 |
| **E007** | **Proposed Asym Gated Fusion** | RGB + IR | 93.70 | 55.98 | 87.52% | 90.25% | 88.86% | 90.41% | 0.4044 | 5,245 | 59,853 |

*Documentation Resolution Note on Detection Counts*: Previous narrative audit drafts listed "Total Preds" without clarifying the operating threshold. Detections at the standard evaluation threshold $\tau \ge 0.25$ range from 3,584 to 5,245 across models, whereas total candidate detections output by the network prior to aggressive thresholding (confidence $\ge 0.001$, used for full PR-curve integration) range from 34,035 to 60,664 (with E005 producing 231,221 due to low confidence suppression). Both columns are now explicitly tabulated above.

---

## 3. Dataset Split and Annotation Reconciliation

### 3.1 Verified Dataset Partition Allocation
The canonical LLVIP dataset contains **15,488 paired visible-infrared frames**:
- **Official Training Pool**: 12,025 pairs total.
  - **Internal Training Split**: **10,222 pairs** (`train_rgb.txt`, `train_thermal.txt`).
  - **Internal Validation Split**: **1,803 pairs** (`val_rgb.txt`, `val_thermal.txt`).
  - Allocation Check: $10,222 + 1,803 = 12,025$ pairs ($85.0\% / 15.0\%$ scene-aware split).
- **Official Test Set**: **3,463 pairs** (`test_rgb.txt`, `test_thermal.txt`). **Strictly sealed.**
- **Total Dataset Size**: $10,222 + 1,803 + 3,463 = \mathbf{15,488}$ pairs.

> [!WARNING]
> **Correction of Historical Error**: Older draft protocols cited a training count of 10,643 pairs. This was an erroneous draft calculation. The actual, verified line count across `train_rgb.txt` is exactly **10,222**. All future documents must use 10,222 exclusively.

### 3.2 Annotation Accounting and Exclusion Justification

| Partition | Total Images | Raw XML Annotations | Excluded Line Boxes | Valid Annotations | Valid Ground Truths |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training** | 10,222 | 29,047 | 3 (`090437` box 1, `091180` box 2, `170350` box 0) | 29,044 | 29,044 |
| **Validation** | 1,803 | 5,088 | 2 (`020118` box 1, `020236` box 5 in Scene 02) | **5,086** | **5,086** |
| **Official Test** *(Sealed)* | 3,463 | 8,302 | 0 | 8,302 | 8,302 |
| **Total Benchmark** | **15,488** | **42,437** | **5** | **42,432** | **42,432** |

All 5 excluded bounding boxes have width equal to 0.0 px ($x_{\min} = x_{\max}$), representing geometric line artifacts that cannot physically delimit a pedestrian. Exactly **5,086 valid pedestrian bounding boxes** exist across the 1,803 validation pairs.

---

## 4. Hardware Environment Verification

All evaluations and latency benchmarks were conducted on the verified hardware environment:
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
  - Architecture: Ada Lovelace (SM 8.9)
  - VRAM: 8.0 GB GDDR6 (7.995 GB addressable)
  - Tensor Cores: 4th Generation
- **Host OS**: Windows 11 (build 26100)
- **Host Memory**: 16 GB System RAM
- **Software Stack**:
  - Python: 3.11.9
  - PyTorch: 2.5.1+cu121
  - CUDA: 12.1
  - OpenCV: 5.0.0
  - Ultralytics: 8.4.144
  - Ensemble-Boxes (WBF): 1.0.9

> [!IMPORTANT]
> **Correction of Hardware Reporting**: Earlier protocol drafts inadvertently mentioned an RTX 3050 Laptop GPU (4 GB). That was an incorrect placeholder from an earlier test rig. The actual GPU used for all training runs E001–E007, gate analyses, and benchmarks is the **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)**.

---

## 5. Architectural Audits and Checkpoint Realignment

### 5.1 Architecture Definitions

- **E001 (YOLO11n RGB)**: Single-stream baseline using Ultralytics YOLO11n topology with 3-channel visible RGB input.
- **E002 (YOLO26n RGB)**: Single-stream baseline using Ultralytics YOLO26n topology with 3-channel visible RGB input and a dedicated 1-class detection head ($nc=1$).
- **E003 (YOLO26n Thermal)**: Single-stream baseline using Ultralytics YOLO26n topology with 3-channel replicated thermal infrared input and a dedicated 1-class detection head ($nc=1$).
- **E004 (YOLO26n Late Fusion WBF)**: Post-processing ensemble combining bounding-box predictions from independently trained E002 and E003 models using Weighted Boxes Fusion (weights: RGB=1.0, Thermal=2.0, IoU threshold=0.55).
- **E005 (YOLO26n Early Fusion 4-Channel)**: Single-backbone YOLO26n model whose first convolutional stem (`model.0.conv`) was modified to accept **4 input channels** ($[R, G, B, \text{IR}]$). Pretrained RGB weights were assigned to channels 0–2, and channel 3 was initialized with the channel-wise mean of RGB weights. **It is NOT a 6-channel model, and NOT a dual-backbone architecture.**
- **E006 (YOLO26n Intermediate Feature Concatenation)**: Dual parallel YOLO26n backbones (Thermal + RGB, layers 0–10). Cross-modal feature maps at P3, P4, P5 are concatenated along channel dimensions and compressed back to single-stream width via $1\times 1$ convolutions before passing into shared neck and head layers (layers 11–23).
- **E007 (Proposed Asymmetric Gated Fusion)**: Dual parallel YOLO26n backbones (Thermal as primary semantic stream, RGB as auxiliary stream). Visible features are projected and dynamically modulated via learned `ReliabilityGatingModule` subnets at P3, P4, P5 before additive injection into the thermal residual stream:
  $$F_{\text{fused}} = F_T + G_V \odot P(F_V)$$
  where $G_V = \sigma(\text{Conv}_{1\times 1}(\delta(\text{BatchNorm}(\text{Conv}_{3\times 3}([F_T, P(F_V)])))))$.

### 5.2 Detection Head Class Count Discrepancy ($nc = 1$ vs. $nc = 80$)

A thorough checkpoint audit revealed that:
1. **Unimodal Baselines (E001, E002, E003)**: Trained via standard Ultralytics configuration with `data.yaml` setting `nc: 1`. The detection head was created with a single output channel ($nc = 1$).
2. **Multimodal Fusion Architectures (E005, E006, E007)**: Instantiated from the COCO-pretrained base model `yolo26n.pt` without explicitly pruning or reinitializing Layer 23 (`Detect.cv3` and `Detect.one2one_cv3`). Consequently, these models retained an **80-class COCO detection head** ($nc = 80$) during both training and evaluation.

### Detailed Checkpoint Parameter & Dimension Audit

| Exp ID | Model Architecture | Checkpoint Path | Total Parameters | Trainable Parameters | Detection Head ($nc$) | Layer 23 Head Params | Model Size on Disk | GFLOPs ($640\times 640$) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n RGB | `E001_RGB_YOLO11N/weights/best.pt` | **2,590,035** | 2,590,035 | 1 | 308,837 | 5.46 MB | 6.50 |
| **E002** | YOLO26n RGB | `E002_RGB_YOLO26N/weights/best.pt` | **2,504,190** | 2,504,190 | 1 | 297,560 | 5.37 MB | 5.89 |
| **E003** | YOLO26n Thermal | `E003_THERMAL_YOLO26N/weights/best.pt` | **2,504,190** | 2,504,190 | 1 | 297,560 | 5.37 MB | 5.89 |
| **E004** | YOLO26n Late WBF | E002 + E003 checkpoints | **5,008,380** | N/A (Ensemble) | 1 + 1 | 595,120 | 10.74 MB | 11.78 |
| **E005** | YOLO26n Early 4CH | `E005_EARLY_FUSION_4CH_YOLO26N/weights/best.pt` | **2,572,424** | 2,572,424 | 80 | 365,650 | 10.58 MB | 6.27 |
| **E006** | YOLO26n Feature Concat | `E006_FEATURE_FUSION_CONCAT_YOLO26N/weights/best.pt` | **4,135,384** | 4,135,384 | 80 | 365,650 | 16.96 MB | 9.88 |
| **E007** | Proposed Asym Gated | `E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N/weights/best.pt` | **4,971,992** | 4,971,992 | 80 | 365,650 | 20.31 MB | 12.16 |
| *E008* | *1-Class Gated Fusion* | *Theoretical (DEFERRED)* | *4,903,902* | *4,903,902* | *1* | *297,560* | *~19.8 MB* | *12.04* |

### Scientific Implications of the Detection Head Mismatch:
- **The Parameter Delta**: The physical E007 checkpoint has **4,971,992 parameters**, exactly 68,090 parameters larger than the theoretical single-class architecture (4,903,902 parameters). E006 has **4,135,384 parameters** (not 3.87M or 4.97M as inconsistently written in early drafts).
- **Training Loss Impact**: During training of E005, E006, and E007, the binary cross-entropy (BCE) classification loss computed penalties across all 80 classes. Class 0 was matched against ground-truth persons, while classes 1–79 received zero targets, consuming optimizer capacity to suppress logits for 79 irrelevant classes.
- **Evaluation Neutrality**: During unified evaluation, predictions are filtered for class 0 (`pred_cls == 0`) before AP integration. Because $>99.8\%$ of pedestrian predictions were assigned class 0, the reported AP accurately reflects person detection performance.
- **Conclusion**: This discrepancy represents an uncontrolled variable of the historical training phase. Because all new training is prohibited, this discrepancy is permanently documented as a research limitation.

---

## 6. Evaluation Framework Clarification

A critical audit objective was resolving contradictory statements regarding the evaluation implementation:
- **Actual Evaluator**: The unified validation script (`audit_unified_evaluation.py`) uses **Ultralytics `DetectionValidator.match_predictions` and `ultralytics.utils.metrics.ap_per_class`**, evaluating bounding boxes transformed back to the **native $1280 \times 1024$ image coordinate system** via unletterbox scaling (`scale_boxes`).
- **Clarification**: The evaluator does **NOT** use `pycocotools.cocoeval.COCOeval`. Prior documents describing the evaluation as "pycocotools COCOeval" were factually inaccurate.
- **Evaluation Settings**:
  - Maximum detections per image: 300
  - Candidate confidence threshold for PR integration: $\tau_{\min} = 0.001$
  - IoU evaluation thresholds: 10 steps from 0.50 to 0.95 ($\text{step} = 0.05$)
  - NMS IoU threshold: 0.65

---

## 7. Recalculated Object-Scale Distributions

To resolve conflicting scale distributions across older reports, object scales were recalculated directly from the 5,086 validated ground-truth XML annotations:

### Official COCO Area Categories:
- **Small ($S < 32^2 = 1,024\text{ px}^2$)**
- **Medium ($1,024 \le S \le 96^2 = 9,216\text{ px}^2$)**
- **Large ($S > 9,216\text{ px}^2$)**

### Empirical Scale Breakdown Across the 1,803 Validation Pairs

| Coordinate Space | Small ($< 1,024\text{ px}^2$) | Medium ($1,024\text{--}9,216\text{ px}^2$) | Large ($> 9,216\text{ px}^2$) | Total Valid Ground Truths |
|:---|:---:|:---:|:---:|:---:|
| **Native Resolution ($1280 \times 1024$)** | **8** ($0.16\%$) | **330** ($6.49\%$) | **4,748** ($93.35\%$) | **5,086** |
| **Model Input Letterbox ($640 \times 640$)** | **66** ($1.30\%$) | **4,820** ($94.77\%$) | **200** ($3.93\%$) | **5,086** |

### Crucial Empirical Findings:
1. In native $1280 \times 1024$ space, exactly **8 pedestrians** across the entire validation partition are smaller than $1,024\text{ px}^2$. Because $N = 8$, small-object metrics ($\text{AP}_S$) have **insufficient sample support** and exhibit high statistical instability.
2. In model input space ($640 \times 640$), bilinear downsampling compresses areas by $0.50^2 = 0.25$, shifting 4,548 native large objects into the medium category and 58 native medium objects into the small category.
3. Prior draft texts reporting 2,207 medium and 2,871 large objects used an undocumented, non-standard area threshold. The counts above (330 medium, 4,748 large native) are verified.

---

## 8. Summary of Reconciled Research Status

| Subject Area | Reconciled Canonical Truth |
|:---|:---|
| **Training Pipeline** | **Permanently frozen.** E008/E009/Multi-seed are **DEFERRED**. |
| **Primary Checkpoints** | E001–E007 preserved on disk and verified. |
| **Dataset Splits** | 10,222 train / 1,803 val / 3,463 sealed test (Total: 15,488). |
| **Valid Annotations** | 29,044 train / 5,086 val / 8,302 test (5 excluded zero-width line boxes). |
| **Hardware** | NVIDIA RTX 4060 Laptop GPU (8GB VRAM), PyTorch 2.5.1+cu121. |
| **E005 Architecture** | Single-backbone 4-channel input ($[R, G, B, \text{IR}]$), 2.57M params, $nc=80$. |
| **E006 Architecture** | Dual-backbone feature concatenation, 4.14M params, $nc=80$. |
| **E007 Architecture** | Dual-backbone asymmetric gated fusion, 4.97M params, $nc=80$. |
| **Unified Evaluator** | Ultralytics `DetectionValidator.match_predictions` + `ap_per_class` in native $1280 \times 1024$ space. |
| **Official Test Set** | **Strictly sealed until Stage 7 model freeze approval.** |
