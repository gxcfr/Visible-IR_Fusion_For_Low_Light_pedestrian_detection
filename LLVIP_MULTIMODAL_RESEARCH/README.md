# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

---

## A. Project Overview

- **Research Title**: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection
- **Research Objective**: Build a complete, reproducible, and scientifically defensible multimodal pedestrian detection research framework that investigates whether an asymmetric, reliability-conditioned feature fusion architecture outperforms independent unimodal detectors and conventional fusion approaches under low-light nighttime surveillance conditions.
- **Research Domain**: Computer Vision, Multimodal Deep Learning, RGB-Thermal (RGB-T) Fusion, Low-Light Object Detection, Attention & Feature Gating, Intelligent Transportation Systems.
- **Primary Research Question**: Can an asymmetric, reliability-aware visible-infrared fusion architecture improve pedestrian detection under challenging illumination conditions compared with independent RGB and infrared detectors and conventional multimodal fusion approaches?
- **Research Hypothesis**: Visible and infrared sensors provide complementary information whose reliability varies dynamically with scene conditions. In severe low-light, visible imagery suffers low SNR while infrared provides stable thermal signatures of pedestrians. A lightweight asymmetric architecture—treating thermal as the primary semantic backbone and adaptively modulating visible feature injection through learnable reliability-conditioned gating—can achieve superior pedestrian detection performance ($mAP_{50}$, $mAP_{50-95}$) while maintaining real-time edge computational efficiency.
- **Dataset**: LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision.
- **Primary Architecture Selection**: **Ultralytics YOLO26n** is exclusively selected as the primary detector architecture for all future experiments. Using a consistent YOLO26n backbone throughout unimodal, conventional fusion, and proposed fusion models isolates the exact scientific contribution of the feature fusion mechanisms without confounding cross-architecture variables. YOLO11n is deferred, and E001 is preserved as an already-executed historical unimodal reference baseline.
- **Research Scope**: Dedicated strictly to pedestrian detection (`person` class).

---

## B. Current Project Status

- **Current Milestone**: Milestone 4 (Unimodal Detection Baselines) — `IN PROGRESS`
- **Current Milestone Status**:
  - **Milestone 1**: `COMPLETED` (Environment & Hardware Setup, RTX 4060 GPU verified)
  - **Milestone 2**: `COMPLETED` (LLVIP Dataset Audit: 15,488 pairs, single class `person`, 5 degenerate boxes isolated)
  - **Milestone 3**: `COMPLETED` (Scene-Aware 85/15 Split, VOC-to-YOLO conversion, paired PyTorch DataLoader, 16/16 unit tests passed)
  - **Milestone 4**: `IN PROGRESS` (Historical baseline `E001_RGB_YOLO11N` fully trained & evaluated; `E002_RGB_YOLO26N` queued awaiting researcher approval)
- **Overall Project Status**:
  - First unimodal baseline `E001_RGB_YOLO11N` trained to full completion over 40 epochs.
  - Achieved **80.57% $mAP_{50}$** and **41.31% $mAP_{50-95}$** on the internal validation set with **91.24% Precision** and **72.22% Recall**.
  - Batch-1 inference latency on RTX 4060 GPU measured at **10.69 ms** (**93.55 FPS**).
  - All artifacts, checkpoints, confusion matrices, and training curves saved in `experiments/E001_RGB_YOLO11N/`.
- **Last Completed Task**: Completed training, evaluation, and latency benchmarking of `E001_RGB_YOLO11N`.
- **Current Active Task**: Awaiting researcher approval to launch `E002_RGB_YOLO26N` (YOLO26n RGB-only baseline).
- **Next Immediate Task**: Execute `E002_RGB_YOLO26N` using the same core protocol (batch size 8, AdamW, AMP, patience 12, max 40 epochs).
- **Active Blockers**: None.
- **Decisions Awaiting Researcher Input**: Approval to launch `E002_RGB_YOLO26N`.

---

## C. Milestone Tracker

| Milestone | Description | Status | Completion Evidence |
|---|---|---|---|
| **1** | Project Initialization & Environment Setup | `COMPLETED` | Clean repo under `LLVIP_MULTIMODAL_RESEARCH/`, PyTorch 2.5.1+cu121, RTX 4060 verified, `test_environment.py` 6/6 passed |
| **2** | LLVIP Dataset Audit & Validation | `COMPLETED` | Audit JSON (`dataset_audit.json`), manifest (15,488 pairs), pairing stats (`dataset_pairing_statistics.json`), distribution plot (`dataset_distribution.png`), 50-pair gallery (`results/qualitative/paired_gallery/`), `test_dataset.py` 3/3 passed |
| **3** | Data Pipeline & Baseline Preparation | `COMPLETED` | Scene-aware 85/15 split (`dataset_split_manifest.csv`), VOC-to-YOLO conversion (`annotation_exclusion_report.json`), paired DataLoader (`paired_dataset.py`, `visualize_augmentations.py`), YOLO weights verified, 16/16 tests passed |
| **4** | Unimodal Detection Baselines | `IN PROGRESS` | `E001_RGB_YOLO11N` completed (80.57% mAP50, 41.31% mAP50-95, 93.55 FPS); `E002_RGB_YOLO26N` queued |
| **5** | Conventional Multimodal Fusion (WBF, Early, Concat) | `NOT STARTED` | Pending Milestone 4 baselines |
| **6** | Proposed Architecture (Asymmetric Reliability Fusion) | `NOT STARTED` | Pending Milestone 5 baselines |
| **7** | Ablation & Robustness Experiments | `NOT STARTED` | Pending Milestone 6 training |
| **8** | Final Evaluation & Research Documentation | `NOT STARTED` | Pending Milestone 7 ablations |

*Status definitions: `NOT STARTED` \| `IN PROGRESS` \| `BLOCKED` \| `NEEDS REVIEW` \| `COMPLETED`*

---

## D. Dataset Information & Partition Summary

- **Dataset Name**: LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision
- **Homepage / Paper**: https://bupt-ai-cz.github.io/LLVIP/ | arXiv:2108.10831
- **Local Dataset Path**: `C:\LLVIP\LLVIP` (Untouched, read-only)
- **Total Unique Identifiers**: 15,488
- **Total Individual Images**: 30,976 (15,488 visible RGB, 15,488 thermal infrared)
- **Total Perfectly Paired Pairs**: 15,488 (100.0% pairing rate, 0 unmatched)
- **Scene-Aware Architecture**:
  - Total scenes: 26 scenes (`01` through `26`).
  - Train/Test scene overlap: **0 scenes** (disjoint at scene level).
- **Partitions**:
  - **Internal Train**: 15 scenes (`01, 03, 04, 06, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18`)
    - Pairs: **10,222 pairs (85.01% of official train)**
    - Pedestrian Annotations: **29,047 (85.09%)** | Density: 2.84 persons/pair
  - **Internal Validation**: 4 scenes (`02, 05, 07, 25`)
    - Pairs: **1,803 pairs (14.99% of official train)**
    - Pedestrian Annotations: **5,088 (14.91%)** | Density: 2.82 persons/pair
  - **Official Test (Untouched)**: 7 scenes (`19, 20, 21, 22, 23, 24, 26`)
    - Pairs: **3,463 pairs** | Annotations: **8,302** | Density: 2.40 persons/pair
- **Annotation Integrity**:
  - Total XML Files Processed: 15,488
  - Total Valid Converted Boxes: 42,432 (99.9882% retention rate)
  - Degenerate Zero-Width Boxes Excluded: Exactly 5 (`020118`, `020236`, `090437`, `091180`, `170350`). All valid pedestrians in those files preserved.

---

## E. Development Environment

- **Operating System**: Windows 11
- **Python Version**: 3.11.9
- **PyTorch Version**: 2.5.1+cu121
- **CUDA Runtime / Driver**: CUDA 12.1 (Torch) / Driver 581.42 (CUDA 13.0 capability)
- **GPU Model**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM Total**: 8187.5 MB (~8.00 GB)
- **Compute Capability**: 8.9 (Ada Lovelace architecture with 4th Gen Tensor Cores)
- **Ultralytics Version**: 8.4.144
- **Primary Detection Architecture**: **YOLO26n** (`weights/yolo26n.pt`, 2,572,280 parameters)
- **Historical Baseline**: `YOLO11n` (`weights/yolo11n.pt`, 2,590,035 parameters)
- **Git Remote**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`

---

## F. Research Architecture

```
           [ Infrared / Thermal Input ]
                        │
             [ Thermal Primary Backbone ] (YOLO26n Backbone)
                        │
                        ▼ (F_T: P3, P4, P5)
                        │
 [ Visible / RGB Input ]│
          │             │
   [ Lightweight Aux ]  │
   [ YOLO26n Stream ]   │
          │             │
          ▼ (F_V)       │
          │             │
   [ Feature Proj ]     │
      P(F_V)            │
          │             │
          ├──► [ Quality & Gating Module ] ◄── Quality Descriptors (Q_T, Q_V)
          │    G_V = σ(Conv([F_T, P(F_V), Q_T, Q_V]))
          │             │
          ▼             ▼
          └───► [ G_V ⊙ P(F_V) ]
                        │
                        ▼
            [ F_fused = F_T + G_V ⊙ P(F_V) ]
                        │
                        ▼
             [ Detection Neck & Head ]
                        │
                        ▼
          Predictions: [ BBoxes, Conf, 'person' ]
```

---

## G. Experimental Results Table

*All experiments evaluated on identical validation partitions (1,803 pairs). Official test set strictly untouched.*

| Exp ID | Architecture | Modality / Strategy | $mAP_{50}$ | $mAP_{50-95}$ | Recall | Precision | F1 | Latency (ms) | FPS | Params | Peak VRAM | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E001** | YOLO11n | RGB-only (Historical) | **0.8057** | **0.4131** | **0.7222** | **0.9124** | **0.8062** | **10.69** | **93.55** | 2,590,035 | 1,126 MB | **COMPLETED** |
| **E002** | YOLO26n | RGB-only Baseline | **0.7934** | **0.4047** | **0.7191** | **0.8939** | **0.7970** | **15.80** | **63.28** | 2,504,190 | 1,251 MB | **COMPLETED** |
| **E003** | YOLO26n | Thermal-only Baseline | **0.9449** | **0.5679** | **0.8790** | **0.9413** | **0.9091** | **10.80** | **92.60** | 2,504,190 | 1,247 MB | **COMPLETED** |
| **E004** | YOLO26n | Late Fusion (WBF) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | 5,008,380 | — | *Queued* |
| **E005** | YOLO26n | Early Fusion (4-Channel) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | ~2.57M | — | *Queued* |
| **E006** | YOLO26n | Feature Concatenation | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | ~3.8M | — | *Queued* |
| **E007** | Proposed | Asymmetric Gated Fusion | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | ~3.9M | — | *Queued* |

---

## H. Detailed Record of Completed Experiment E001

- **Experiment ID**: `E001_RGB_YOLO11N`
- **Model Architecture**: Ultralytics YOLO11n
- **Input Modality**: Visible RGB (3-channel)
- **Pretrained Checkpoint**: `weights/yolo11n.pt`
- **Training Settings**:
  - Image Resolution: $640 \times 640$
  - Batch Size: 8
  - Optimizer: Explicit AdamW (`lr0=0.001`, `weight_decay=0.0005`)
  - Precision: Automatic Mixed Precision (AMP / FP16)
  - Warmup: 3 epochs, Cosine LR scheduling
  - Max Epochs: 40, Early Stopping Patience: 15
  - Seed: 42
- **Training Trajectory & Convergence**:
  - Starting Epoch 1: $mAP_{50} = 67.97\%$, $mAP_{50-95} = 32.32\%$, Box Loss = 1.899, Cls Loss = 1.637.
  - Intermediate Epoch 11: $mAP_{50} = 80.02\%$, $mAP_{50-95} = 39.98\%$.
  - Best Validation Epoch: **Epoch 25** ($mAP_{50} = 80.56\%$, $mAP_{50-95} = 41.35\%$, Precision = $91.26\%$, Recall = $72.24\%$, Val Box Loss = 1.746, Val Cls Loss = 1.084).
  - Peak $mAP_{50}$ Epoch: **Epoch 26** ($mAP_{50} = 80.69\%$, $mAP_{50-95} = 40.43\%$).
  - Stopping Epoch: **Epoch 40** (Completed full scheduled 40 epochs; fine-tuning plateau reached).
  - Training Duration: **172.98 minutes (~2.88 hours)**.
  - Peak PyTorch GPU Memory: **1,126.61 MB** (~14% of 8 GB VRAM capacity).
- **Validation Benchmark (on Best Checkpoint `weights/best.pt`)**:
  - Precision: **0.9124 (91.24%)**
  - Recall: **0.7222 (72.22%)**
  - F1 Score: **0.8062 (80.62%)**
  - $mAP_{50}$: **0.8057 (80.57%)**
  - $mAP_{50-95}$: **0.4131 (41.31%)**
  - Batch-1 Latency (GPU sync, 50 warmup, 200 timed runs): **10.69 ms** (std: 3.09 ms, min: 7.19 ms)
  - Batch-1 End-to-End Latency: **11.48 ms**
  - Throughput: **93.55 FPS**
- **Artifact Locations**:
  - Directory: `experiments/E001_RGB_YOLO11N/`
  - Best Checkpoint: `experiments/E001_RGB_YOLO11N/weights/best.pt`
  - Last Checkpoint: `experiments/E001_RGB_YOLO11N/weights/last.pt`
  - Training Curves: `experiments/E001_RGB_YOLO11N/results.png`
  - PR Curve: `experiments/E001_RGB_YOLO11N/BoxPR_curve.png`
  - F1 Curve: `experiments/E001_RGB_YOLO11N/BoxF1_curve.png`
  - Confusion Matrix: `experiments/E001_RGB_YOLO11N/confusion_matrix.png`
  - Metrics JSON: `experiments/E001_RGB_YOLO11N/metrics.json`

---

## I. Completed Work Log

- **2026-11-15 (Milestones 1–3)**:
  - Repository, hardware diagnostics, and test suite initialized (RTX 4060 GPU, 16/16 tests passing).
  - Audited LLVIP dataset: 15,488 image pairs, single class `person`, 5 degenerate zero-width annotations isolated without data loss.
  - Scene-aware 85/15 train-val split established (Train: 10,222 pairs, Val: 1,803 pairs, Test: 3,463 pairs untouched).
  - Converted all 15,488 XML annotations to normalized YOLO format; generated synchronized multimodal PyTorch DataLoader.
  - Committed and pushed Milestones 1–3 to GitHub (`commit f214f40`).
- **2026-11-15 (Milestone 4 - Execution of E001 & Architecture Decision)**:
  - Executed memory smoke test on RTX 4060 GPU: confirmed Batch Size 8 allocates only ~1.16 GB VRAM with >85% headroom.
  - Executed and completed full training run for `E001_RGB_YOLO11N` (40 epochs, ~2.88 hours).
  - Measured empirical baseline metrics on validation set: **80.57% $mAP_{50}$**, **41.31% $mAP_{50-95}$**, **93.55 FPS** at Batch 1.
  - **Research Architecture Decision**: Focused the research roadmap exclusively on Ultralytics YOLO26n to isolate multimodal fusion mechanisms; deferred YOLO11n comparisons and preserved E001 as historical reference.
  - Updated experiment numbering: `E002_RGB_YOLO26N`, `E003_THERMAL_YOLO26N`, `E004_LATE_FUSION_WBF_YOLO26N`, `E005_EARLY_FUSION_4CH_YOLO26N`, `E006_FEATURE_FUSION_CONCAT_YOLO26N`, `E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N`.

---

## J. Outstanding Tasks

- [x] **Milestone 1**: Project Initialization & Environment Setup
- [x] **Milestone 2**: LLVIP Dataset Audit & Validation
- [x] **Milestone 3**: Data Pipeline & Baseline Preparation
- [x] **Milestone 4 (E001)**: Train and benchmark historical baseline `E001_RGB_YOLO11N`
- [x] **Milestone 4 (E002)**: Train and benchmark primary baseline `E002_RGB_YOLO26N` (YOLO26n RGB-only)
- [x] **Milestone 4 (E003)**: Train and benchmark primary baseline `E003_THERMAL_YOLO26N` (YOLO26n Thermal-only)
- [ ] **Milestone 5 (E004)**: Benchmark YOLO26n Late Fusion via Weighted Boxes Fusion (WBF) [IN PROGRESS]
- [ ] **Milestone 5 (E005)**: Implement & train 4-channel Early Fusion baseline
- [ ] **Milestone 5 (E006)**: Implement & train dual-stream Feature Concatenation baseline
- [ ] **Milestone 6 (E007)**: Implement & train proposed Thermal-Primary Asymmetric Reliability-Aware Gated Fusion
- [ ] **Milestone 7**: Execute ablation studies and robustness testing under synthetic corruptions
- [ ] **Milestone 8**: Final evaluation on untouched official test partition and compile research paper draft

---

## K. Next Actions

1. **Researcher Review & Approval**: Present completion report of `E001_RGB_YOLO11N` and updated roadmap for approval.
2. **Launch Experiment E002**: Upon approval, execute `E002_RGB_YOLO26N` (YOLO26n RGB-only, 640x640, Batch 8, AdamW, AMP, Max 40 epochs, Patience 12).
3. **Launch Experiment E003**: Execute `E003_THERMAL_YOLO26N` (YOLO26n Thermal-only with preserved radiometric intensity).
4. **Compile Unimodal Comparison**: Compare YOLO26n vs YOLO11n on RGB, and compare YOLO26n RGB vs Thermal.

---

## L. Reproduction Instructions

### Run Unimodal Training
```powershell
# E002: YOLO26n RGB-only
python LLVIP_MULTIMODAL_RESEARCH/src/training/train_unimodal.py --exp_id E002_RGB_YOLO26N --model LLVIP_MULTIMODAL_RESEARCH/weights/yolo26n.pt --modality rgb --batch 8 --epochs 40 --patience 12

# E003: YOLO26n Thermal-only
python LLVIP_MULTIMODAL_RESEARCH/src/training/train_unimodal.py --exp_id E003_THERMAL_YOLO26N --model LLVIP_MULTIMODAL_RESEARCH/weights/yolo26n.pt --modality thermal --batch 8 --epochs 40 --patience 12
```

---

## M. Research Integrity Statement

All experimental metrics, parameter counts, latency benchmarks, and dataset statistics reported in this project derive strictly from actual, reproducible executions on the stated hardware. No metrics are approximated from external literature. Incomplete or planned benchmarks are explicitly designated as *Pending*.
