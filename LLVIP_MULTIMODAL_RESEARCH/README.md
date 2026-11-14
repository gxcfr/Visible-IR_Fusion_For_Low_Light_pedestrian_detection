# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

---

## A. Project Overview

- **Research Title**: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection
- **Research Objective**: Build a complete, reproducible, and scientifically defensible multimodal pedestrian detection research framework that outperforms independent unimodal detectors and conventional fusion architectures under low-light nighttime surveillance conditions.
- **Research Domain**: Computer Vision, Multimodal Deep Learning, RGB-Thermal (RGB-T) Fusion, Low-Light Object Detection, Attention & Feature Gating, Intelligent Transportation Systems.
- **Primary Research Question**: Can an asymmetric, reliability-aware visible-infrared fusion architecture improve pedestrian detection under challenging illumination conditions compared with independent RGB and infrared detectors and conventional multimodal fusion approaches?
- **Research Hypothesis**: Visible and infrared sensors provide complementary information whose reliability varies dynamically with scene conditions. In severe low-light, visible imagery suffers low SNR while infrared provides stable thermal signatures of pedestrians. A lightweight asymmetric architecture—treating thermal as the primary semantic backbone and adaptively modulating visible feature injection through learnable reliability-conditioned gating—can achieve superior pedestrian detection performance ($mAP_{50}$, $mAP_{50-95}$) while maintaining real-time edge computational efficiency.
- **Dataset**: LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision.
- **Architecture Direction**: Thermal-Primary Asymmetric Reliability-Aware Gated Fusion built upon lightweight YOLO architectures (YOLO26n primary, YOLO11n comparison baseline).
- **Research Scope**: Dedicated strictly to pedestrian detection (`person` class). Multi-class VRU categories are outside current research scope as LLVIP strictly annotates pedestrians.

---

## B. Current Project Status

- **Current Milestone**: Milestone 4 (Unimodal Detection Baselines: YOLO11n & YOLO26n)
- **Current Milestone Status**:
  - **Milestone 1**: `COMPLETED`
  - **Milestone 2**: `COMPLETED`
  - **Milestone 3**: `COMPLETED`
  - **Milestone 4**: `NOT STARTED / READY TO LAUNCH`
- **Overall Project Status**: Environment setup, dataset audit, and multimodal data pipeline are fully completed with 100% test pass rate (16/16 unit tests passing).
  - The 15,488 pairs were analyzed at the scene level (26 distinct scenes, 0 overlap between official train and test sets).
  - A deterministic group-based 85/15 train-validation split was established using training scenes `[02, 05, 07, 25]` for validation, yielding exactly 1,803 validation pairs (14.99% of official train) and 10,222 training pairs (85.01%), with the 3,463 official test pairs preserved 100% untouched.
  - All 15,488 VOC XML annotations were converted to normalized YOLO format; 5 degenerate zero-width annotations in raw data were documented in an exclusion report and filtered, retaining 42,432 valid pedestrian annotations (99.9882% retention rate, 0 valid boxes lost).
  - Paired PyTorch Dataset and DataLoader with synchronized letterboxing, horizontal flips, and photometric variations were implemented and visually verified.
  - Pretrained detection weights for both YOLO11n (2.62M params) and YOLO26n (2.57M params) load offline and execute forward passes without dependency issues.
- **Last Completed Task**: Completed Milestone 3 deliverables, executed full test suite (16/16 passed), generated augmentation verification figure (`results/qualitative/augmentation_verification.png`), scene distribution chart (`results/plots/scene_distribution.png`), and exclusion report (`results/metrics/annotation_exclusion_report.json`).
- **Current Active Task**: Awaiting researcher approval to initiate training of Milestone 4 unimodal baselines (E001–E004).
- **Next Immediate Task**: Execute Experiment E001 (YOLO11n RGB-only) on RTX 4060 GPU with AMP.
- **Active Blockers**: None.
- **Decisions Awaiting Researcher Input**: Confirmation to launch baseline training runs (E001–E004).

---

## C. Milestone Tracker

| Milestone | Description | Status | Completion Evidence |
|---|---|---|---|
| **1** | Project Initialization & Environment Setup | `COMPLETED` | Clean repo under `LLVIP_MULTIMODAL_RESEARCH/`, PyTorch 2.5.1+cu121, RTX 4060 verified, `test_environment.py` 6/6 passed |
| **2** | LLVIP Dataset Audit & Validation | `COMPLETED` | Audit JSON (`dataset_audit.json`), manifest (15,488 pairs), pairing stats (`dataset_pairing_statistics.json`), distribution plot (`dataset_distribution.png`), 50-pair gallery (`results/qualitative/paired_gallery/`), `test_dataset.py` 3/3 passed |
| **3** | Data Pipeline & Baseline Preparation | `COMPLETED` | Scene-aware 85/15 split (`dataset_split_manifest.csv`, `dataset_split_statistics.json`, `scene_distribution.png`), VOC-to-YOLO conversion (`annotation_exclusion_report.json`), paired DataLoader (`paired_dataset.py`, `visualize_augmentations.py`), YOLO11n/YOLO26n weights verified, 16/16 tests passed |
| **4** | Unimodal Detection Baselines (YOLO11n / YOLO26n) | `NOT STARTED` | Experiment configs generated (`configs/yolo11_rgb.yaml`, etc.); awaiting launch instruction |
| **5** | Conventional Multimodal Fusion (WBF, Early, Concat) | `NOT STARTED` | Pending Milestone 4 baselines |
| **6** | Proposed Architecture (Asymmetric Reliability Fusion) | `NOT STARTED` | Pending Milestone 5 baselines |
| **7** | Ablation & Robustness Experiments | `NOT STARTED` | Pending Milestone 6 training |
| **8** | Final Evaluation & Research Documentation | `NOT STARTED` | Pending Milestone 7 ablations |

*Status definitions: `NOT STARTED` \| `IN PROGRESS` \| `BLOCKED` \| `NEEDS REVIEW` \| `COMPLETED`*

---

## D. Dataset Information & Partition Summary

- **Dataset Name**: LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision
- **Homepage / Paper**: https://bupt-ai-cz.github.io/LLVIP/ | arXiv:2108.10831
- **Local Dataset Path**: `C:\LLVIP\LLVIP`
- **Total Unique Identifiers**: 15,488
- **Total Individual Images**: 30,976 (15,488 visible RGB, 15,488 thermal infrared)
- **Total Perfectly Paired Pairs**: 15,488 (100.0% pairing rate, 0 unmatched)
- **Scene-Aware Architecture**:
  - Total scenes detected: 26 scenes (identified by first 2 digits `01`–`26`)
  - Train/Test scene overlap: **0 scenes** (official splits are 100% disjoint at scene level)
  - Official Train Scenes (19 scenes): `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08`, `09`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `25`
  - Official Test Scenes (7 scenes): `19`, `20`, `21`, `22`, `23`, `24`, `26`
- **Deterministic 85/15 Internal Split**:
  - **Internal Train**: 15 scenes (`01, 03, 04, 06, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18`)
    - Pairs: **10,222 pairs (85.01% of official train)**
    - Pedestrian Annotations: **29,047 (85.09%)**
    - Density: 2.84 persons/pair
  - **Internal Validation**: 4 scenes (`02, 05, 07, 25`)
    - Pairs: **1,803 pairs (14.99% of official train)**
    - Pedestrian Annotations: **5,088 (14.91%)**
    - Density: 2.82 persons/pair
  - **Official Test (Untouched)**: 7 scenes (`19, 20, 21, 22, 23, 24, 26`)
    - Pairs: **3,463 pairs**
    - Pedestrian Annotations: **8,302**
    - Density: 2.40 persons/pair
- **Annotation Conversion Integrity**:
  - Total XML Files Processed: 15,488
  - Total Original Boxes: 42,437
  - Total Converted Valid Boxes: 42,432 (99.9882% retention rate)
  - Total Excluded Degenerate Boxes: Exactly 5 zero-width line boxes (`020118`, `020236`, `090437`, `091180`, `170350`). Remaining valid annotations in these 5 files were fully preserved.
- **Dataset Artifact Paths**:
  - Split Manifest: `results/metrics/dataset_split_manifest.csv`
  - Split Statistics: `results/metrics/dataset_split_statistics.json`
  - Scene Distribution Report: `results/metrics/scene_distribution_report.json`
  - Exclusion Report: `results/metrics/annotation_exclusion_report.json`
  - Scene Distribution Chart: `results/plots/scene_distribution.png`
  - Augmentation Verification: `results/qualitative/augmentation_verification.png`
  - Partition Image Lists: `datasets/train_rgb.txt`, `val_rgb.txt`, `test_rgb.txt`, `train_thermal.txt`, `val_thermal.txt`, `test_thermal.txt`

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
- **Verified Pretrained Models**:
  - `LLVIP_MULTIMODAL_RESEARCH/weights/yolo11n.pt` (2,624,080 parameters)
  - `LLVIP_MULTIMODAL_RESEARCH/weights/yolo26n.pt` (2,572,280 parameters)
- **Git Remote**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git` (Commits strictly upon researcher instruction)
- **Environment Verification**: `pytest LLVIP_MULTIMODAL_RESEARCH/tests -v` (16/16 tests passing)

---

## F. Research Architecture

```
           [ Infrared / Thermal Input ]
                        │
             [ Thermal Primary Backbone ] (YOLO26n / YOLO11n compatible)
                        │
                        ▼ (F_T: P3, P4, P5)
                        │
 [ Visible / RGB Input ]│
          │             │
   [ Lightweight Aux ]  │
       [ Backbone ]     │
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

## G. Experimental Results

*All experiments evaluated on identical validation partitions. Official test set evaluated strictly in Milestone 8.*

| Experiment ID | Model Architecture | Modality | mAP50 | mAP50-95 | Recall | Precision | Latency (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| **E001** | YOLO11n | RGB-only | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Baseline RGB benchmark |
| **E002** | YOLO11n | Thermal-only | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Baseline Thermal benchmark |
| **E003** | YOLO26n | RGB-only | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Primary RGB benchmark |
| **E004** | YOLO26n | Thermal-only | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Primary Thermal benchmark |
| **E005** | YOLO26n | Late Fusion (WBF) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Decision-level fusion |
| **E006** | Custom 4-Ch | Early Fusion | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Pixel-level 4-channel input |
| **E007** | Custom Dual | Feature Concat | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Intermediate concatenation |
| **E008** | Proposed | Asymmetric Gated | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Reliability-aware adaptive fusion |

---

## H. Research Literature

| Paper | Year | Architecture | Core Contribution | Relevance to Our Work | Read Status |
|---|---|---|---|---|---|
| **LLVIP** (Jia et al.) | 2021 | YOLOv5 / Faster R-CNN | Introduced strictly aligned low-light RGB-T benchmark | Benchmark dataset reference and baseline | Verified |
| **MBNet** (Zhou et al.) | 2020 | ResNet-50 + Modality Balance | Modality-balance and differential illumination weighting | Foundational motivation for dynamic gating | Verified |
| **KAIST Benchmark** (Hwang et al.) | 2015 | ACF / Faster R-CNN | Early pioneer in multispectral pedestrian detection | Historical context & alignment reference | Verified |
| **TFDet** | 2025 | Target-aware feature extractor | Target-aware cross-spectral fusion | Comparison architecture candidate | Scheduled |
| **Deformable Cross-Attention** | 2025 | Transformer Cross-Attention | Weakly-aligned RGB-T detection via deformable attention | Attention-based cross-modal comparison | Scheduled |
| **MCFusion** | 2026 | Multi-cue fusion | Multi-cue feature aggregation | Advanced fusion comparison | Scheduled |
| **Uni-MPedFormer** | 2026 | Vision Transformer | Unified transformer framework for multispectral detection | Transformer state-of-the-art reference | Scheduled |
| **WCCNet** | 2026 | Weak calibration network | Calibration under misaligned sensor setups | Robustness reference | Scheduled |

---

## I. Completed Work Log

- **2026-11-15 (Milestone 1)**:
  - Initialized repository structure under `LLVIP_MULTIMODAL_RESEARCH/`.
  - Configured hardware verification: Python 3.11.9, PyTorch 2.5.1+cu121, NVIDIA RTX 4060 Laptop GPU (8.0 GB VRAM).
  - Executed test suite `tests/test_environment.py` with 6/6 tests passing.
  - Initialized git tracking with remote `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`.
- **2026-11-15 (Milestone 2)**:
  - Audited confirmed local dataset at `C:\LLVIP\LLVIP`.
  - Verified 15,488 pairs (12,025 train, 3,463 test) with zero missing files or corrupted images.
  - Confirmed strict single class: `person` (42,437 instances).
  - Identified and isolated 5 degenerate 0-width annotations in raw data.
  - Generated pairing manifest (`dataset_pairing_manifest.csv`), audit JSON (`dataset_audit.json`), and pairing statistics (`dataset_pairing_statistics.json`).
  - Generated dataset distribution chart (`results/plots/dataset_distribution.png`).
  - Generated 50-pair side-by-side visible-infrared annotated qualitative gallery (`results/qualitative/paired_gallery/`).
  - Executed test suite `tests/test_dataset.py` with 3/3 tests passing.
- **2026-11-15 (Milestone 3)**:
  - Verified 26 distinct scene groups via first 2 digits of filenames; confirmed 0 scene overlap between official train and test sets.
  - Implemented scene-aware 85/15 train-val split assigning scenes `02, 05, 07, 25` to validation (1,803 pairs, 14.99%; 5,088 persons, 14.91%) and remaining 15 scenes to train (10,222 pairs, 85.01%; 29,047 persons, 85.09%), preserving 3,463 test pairs untouched.
  - Completed VOC-to-YOLO conversion for all 15,488 XML files; filtered 5 degenerate line annotations, documented in `annotation_exclusion_report.json`, retaining 42,432 valid pedestrian annotations.
  - Implemented `SynchronizedMultimodalAugmentor` and `LLVIPPairedDataset` with PyTorch `DataLoader`.
  - Verified batch loading: `(B, 3, 640, 640)` for both RGB and Thermal, normalized $[0, 1]$, and `(N, 6)` indexed detection targets.
  - Verified offline loading and forward pass of `yolo11n.pt` and `yolo26n.pt` on Ultralytics 8.4.144.
  - Generated split manifest, statistics, scene distribution chart, and augmentation verification figure.
  - Executed full unit test suite with 16/16 tests passing.

---

## J. Outstanding Tasks

- [x] **Milestone 1**: Project Initialization & Environment Setup
- [x] **Milestone 2**: LLVIP Dataset Audit & Validation
- [x] **Milestone 3**: Data Pipeline & Baseline Preparation
- [ ] **Milestone 4**: Train unimodal baselines E001 (YOLO11n RGB), E002 (YOLO11n Thermal), E003 (YOLO26n RGB), E004 (YOLO26n Thermal).
- [ ] **Milestone 5**: Implement and benchmark conventional fusion baselines (WBF Late Fusion, Early Fusion, Feature Concatenation).
- [ ] **Milestone 6**: Implement proposed Thermal-Primary Asymmetric Reliability-Aware Gated Fusion architecture.
- [ ] **Milestone 7**: Execute ablation studies (A1 - A9) and synthetic robustness evaluations.
- [ ] **Milestone 8**: Freeze hyperparameters, evaluate on official LLVIP test set, and prepare final research paper draft.

---

## K. Next Actions

1. **Researcher Review**: Review the Milestone 3 completion report and verify all deliverables.
2. **Execute Experiment E001**: Launch YOLO11n RGB-only unimodal baseline training on RTX 4060 GPU with AMP.
3. **Execute Experiment E002**: Launch YOLO11n Thermal-only unimodal baseline training.
4. **Execute Experiment E003 & E004**: Launch YOLO26n RGB and Thermal unimodal baseline training.
5. **Compile Unimodal Benchmark**: Populate experimental results table with empirical mAP50, mAP50-95, recall, precision, and latency metrics.

---

## L. Reproduction Instructions

### 1. Environment Verification & Full Test Suite
```powershell
python LLVIP_MULTIMODAL_RESEARCH/src/utils/device.py
pytest LLVIP_MULTIMODAL_RESEARCH/tests -v
```

### 2. Dataset Splitting & Annotation Conversion (Milestone 3)
```powershell
# Execute scene-aware 85/15 train-val split
python LLVIP_MULTIMODAL_RESEARCH/src/data/dataset_splitter.py

# Convert annotations from VOC XML to YOLO format
python LLVIP_MULTIMODAL_RESEARCH/src/data/annotation_converter.py

# Verify augmentation and paired DataLoader batching
python LLVIP_MULTIMODAL_RESEARCH/src/visualization/visualize_augmentations.py
```

---

## M. Known Limitations & Constraints

- **Hardware**: NVIDIA RTX 4060 Laptop GPU with 8.0 GB VRAM. Requires conservative batch sizes (8 for single stream, 2–4 for dual stream) and AMP FP16.
- **Dataset Scope**: LLVIP contains only `person` category. Multi-class detection is out of scope.
- **Degenerate Bounding Boxes**: 5 XML annotations in raw data have zero width. These were documented and safely filtered in `datasets/labels/` without touching raw dataset files.
- **Scene Granularity**: Validation set is constrained to whole scene units (`02, 05, 07, 25`) to prevent adjacent video frame leakage.

---

## N. Research Integrity Statement

All experimental metrics, parameter counts, latency benchmarks, and dataset statistics reported in this project derive strictly from actual, reproducible executions on the stated hardware. No metrics are approximated from external literature. Incomplete or planned benchmarks are explicitly designated as *Pending*.
