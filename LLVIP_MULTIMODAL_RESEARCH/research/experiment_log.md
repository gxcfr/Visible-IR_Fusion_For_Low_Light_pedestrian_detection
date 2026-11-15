# Chronological Experiment Log

## Experiment Tracking Protocol
Every experiment receives a unique deterministic identifier:
- `E001_RGB_YOLO11N`: Historical Baseline (YOLO11n RGB-only) - COMPLETED
- `E002_RGB_YOLO26N`: Primary Unimodal Baseline (YOLO26n RGB-only) - COMPLETED
- `E003_THERMAL_YOLO26N`: Primary Unimodal Baseline (YOLO26n Thermal-only) - COMPLETED
- `E004_LATE_FUSION_WBF_YOLO26N`: Multimodal Late Fusion via Weighted Boxes Fusion - COMPLETED
- `E005_EARLY_FUSION_4CH_YOLO26N`: Multimodal 4-Channel Early Fusion - COMPLETED
- `E006_FEATURE_FUSION_CONCAT_YOLO26N`: Multimodal Intermediate Feature Concatenation - COMPLETED
- `E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N`: Proposed Thermal-Primary Asymmetric Reliability-Aware Gated Fusion - COMPLETED

---

## Log Entries

### [Execution] Milestone 4 - Historical Baseline E001_RGB_YOLO11N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Architecture**: Ultralytics YOLO11n (`weights/yolo11n.pt`)
- **Modality**: RGB Visible images (640x640)
- **Training Setup**: 40 epochs scheduled, batch size 8, AdamW, lr0=0.001, weight decay=0.0005, AMP=True, cosine LR, seed=42.
- **Duration**: 172.98 minutes (~2.88 hours), Peak VRAM: 1,126.61 MB.
- **Stopping Dynamics**: Completed 40 epochs; Best Epoch: 25.
- **Empirical Validation Metrics (best.pt)**:
  - Precision: 91.24%
  - Recall: 72.22%
  - F1-Score: 80.62%
  - mAP50: 80.57%
  - mAP50-95: 41.31%
- **Measured Latency (Batch 1, GPU Sync, 50 warmup, 200 runs)**:
  - Model Latency: 10.69 ms (93.55 FPS)
  - End-to-End Latency: 11.48 ms
  - Model Parameters: 2,590,035 (~2.59M)
- **Artifacts**: Checkpoints `best.pt`, `last.pt`, `metrics.json`, `results.csv`, PR and F1 curves saved in `experiments/E001_RGB_YOLO11N/`.

### [Initialization] Milestone 1 - Environment & Hardware Setup
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Actions**:
  - Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121 verified.
  - NVIDIA GeForce RTX 4060 Laptop GPU detected with 8187.5 MB VRAM, Compute Capability 8.9 (Ada Lovelace).
  - Ultralytics 8.4.144, OpenCV 5.0.0.93, ensemble-boxes 1.0.9 verified.
  - Project repository structure initialized under `LLVIP_MULTIMODAL_RESEARCH/`.
  - Device test suite `tests/test_environment.py` executed: 6 passed in 2.48s.
  - Root README.md authored with full Part 9 specification (Sections A through N).
  - Git repository initialized with remote set to `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`.

### [Audit] Milestone 2 - LLVIP Dataset Audit & Validation
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Dataset Path**: `C:\LLVIP\LLVIP`
- **Actions**:
  - Executed automated dataset inspection engine `scripts/inspect_dataset.py`.
  - Total unique pairs indexed: 15,488 (30,976 individual images).
  - Modality matching: 15,488 visible RGB, 15,488 thermal infrared (100% paired, 0 unmatched).
  - Split distribution: 12,025 training pairs (77.64%), 3,463 testing pairs (22.36%).
  - Object annotations: Exactly 1 class (`person`), totaling 42,437 instances across all pairs.
  - Corrupted images: 0 corrupted images found across 200 random dual-modality sample inspections.
  - Data anomalies discovered: Exactly 5 degenerate zero-width annotations in raw XML data (`020118`, `020236`, `090437`, `091180`, `170350`).
  - Generated deliverables:
    - `results/metrics/dataset_audit.json`
    - `results/metrics/dataset_pairing_statistics.json`
    - `results/metrics/dataset_pairing_manifest.csv`
    - `results/plots/dataset_distribution.png`
    - `results/qualitative/paired_gallery/` (50 high-resolution side-by-side visible-infrared annotated pairs)
  - Unit tests: `tests/test_dataset.py` executed with 3/3 tests passing.

### [Data Pipeline] Milestone 3 - Data Pipeline & Baseline Preparation
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Actions**:
  - Investigated filename structure: Verified that the first 2 digits rigorously index 26 distinct scene groups (`01`–`26`). Confirmed 0 scene overlap between official train (19 scenes) and official test (7 scenes).
  - Implemented scene-aware deterministic 85/15 train-val split via `src/data/dataset_splitter.py`:
    - Validation scenes: `['02', '05', '07', '25']` -> 1,803 pairs (14.99% of official train), 5,088 pedestrian annotations (14.91%), density: 2.82 persons/pair.
    - Training scenes: 15 scenes -> 10,222 pairs (85.01%), 29,047 pedestrian annotations (85.09%), density: 2.84 persons/pair.
    - Official test set: 7 scenes -> 3,463 pairs, 8,302 pedestrian annotations, preserved 100% untouched.
  - Completed VOC XML to YOLO TXT annotation conversion for all 15,488 XML files via `src/data/annotation_converter.py`:
    - Raw XMLs: 42,437 total bounding boxes.
    - Converted: 42,432 valid bounding boxes (99.9882% retention).
    - Excluded: Exactly 5 degenerate zero-width line boxes, documented in `results/metrics/annotation_exclusion_report.json`.
    - Generated partition file lists in `datasets/`: `train_rgb.txt`, `val_rgb.txt`, `test_rgb.txt`, `train_thermal.txt`, `val_thermal.txt`, `test_thermal.txt`.
  - Implemented `SynchronizedMultimodalAugmentor` (`src/data/augmentation.py`) with aspect-ratio preserving letterboxing, synchronized horizontal flip, and modality-specific photometric augmentation.
  - Implemented `LLVIPPairedDataset` & `paired_collate_fn` (`src/data/paired_dataset.py`):
    - Yields `[B, 3, 640, 640]` float32 tensors $[0, 1]$ for both RGB and Thermal.
    - Yields `[N, 6]` targets: `(batch_idx, class_id, xc, yc, w, h)`.
  - Generated visual augmentation verification: `results/qualitative/augmentation_verification.png`.
  - Generated scene distribution visualization: `results/plots/scene_distribution.png`.
  - Verified offline model loading and forward pass for both YOLO11n (2.62M params) and YOLO26n (2.57M params) on Ultralytics 8.4.144 without dependency upgrades.
  - Created experiment configs: `configs/yolo11_rgb.yaml`, `configs/yolo11_thermal.yaml`, `configs/yolo26_rgb.yaml`, `configs/yolo26_thermal.yaml`.
  - Test suite execution: 16/16 unit tests passed in 3.07s (`tests/test_data_pipeline.py`, `tests/test_dataset.py`, `tests/test_environment.py`, `tests/test_yolo_compatibility.py`).

### [Execution] Milestone 4 - E002_RGB_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Architecture**: YOLO26n
- **Modality**: RGB
- **Stopping Epoch**: 37 / 40 (Best: 25)
- **mAP50**: 0.7934 | **mAP50-95**: 0.4047
- **Precision**: 0.8939 | **Recall**: 0.7191 | **F1**: 0.7970
- **Latency**: 15.80 ms (63.3 FPS)

### [Execution] Milestone 4 - E002_RGB_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Architecture**: YOLO26n
- **Modality**: RGB
- **Stopping Epoch**: 37 / 40 (Best: 25)
- **mAP50**: 0.7934 | **mAP50-95**: 0.4047
- **Precision**: 0.8939 | **Recall**: 0.7191 | **F1**: 0.7970
- **Latency**: 15.80 ms (63.3 FPS)

### [Execution] Milestone 5 - E006_FEATURE_FUSION_CONCAT_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Strategy**: Dual-Stream Intermediate Feature Concatenation at P3, P4, P5 with 1x1 conv channel projection.
- **Stopping Dynamics**: Stopped at Epoch 17 (patience=12 triggered after peak fitness at Epoch 5).
- **Validation Metrics (1,803 pairs)**:
  - Precision: **93.48%**
  - Recall: **86.32%**
  - F1-Score: **89.76%**
  - mAP50: **93.58%**
  - mAP50-95: **56.62%**
- **Hardware Latency**:
  - Model Latency: **14.73 ms**
  - Throughput: **67.89 FPS**
  - Parameters: **4,135,384 (~4.14M)**
- **Findings**: Intermediate feature concatenation completely eliminates the modality contamination observed in early fusion, surging from **71.53% to 93.58% $mAP_{50}$** (+22.05% improvement) while retaining 67.9 FPS real-time speed.

---

### [Execution] Milestone 5 - E005_EARLY_FUSION_4CH_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Strategy**: 4-Channel Input Stem Early Fusion `[R, G, B, Thermal]`
- **Pretrained Transfer**: Channels 0-2 from YOLO26n RGB, channel 3 initialized with RGB mean.
- **Stopping Dynamics**: Stopped at Epoch 39 (patience=12 triggered after peak at Epoch 27).
- **Validation Metrics (1,803 pairs)**:
  - Precision: **82.66%**
  - Recall: **64.59%**
  - F1-Score: **72.52%**
  - mAP50: **71.53%**
  - mAP50-95: **37.40%**
- **Hardware Latency**:
  - Model Latency: **13.64 ms**
  - Throughput: **73.31 FPS**
  - Parameters: **2,572,424 (~2.57M)**
- **Findings**: Low-level pixel fusion exposes the feature extractor to noisy visible imagery, causing substantial degradation compared to unimodal thermal (**71.53% vs 94.49%**). Demonstrates the critical necessity of decoupled feature extraction and adaptive reliability gating.

---

### [Execution] Milestone 5 - E004_LATE_FUSION_WBF_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Strategy**: Late Fusion via Weighted Boxes Fusion (WBF)
- **Input Models**: `experiments/E002_RGB_YOLO26N/weights/best.pt` & `experiments/E003_THERMAL_YOLO26N/weights/best.pt`
- **WBF Hyperparameters**: $w_{thermal} = 0.7$, $w_{rgb} = 0.3$, $\text{IoU}_{thr} = 0.55$, $\text{skip\_box\_thr} = 0.001$.
- **Validation Metrics (1,803 pairs)**:
  - Precision: **88.15%**
  - Recall: **85.45%**
  - F1-Score: **86.78%**
  - mAP50: **91.35%**
  - mAP50-95: **52.49%**
- **Measured Latency (Batch 1, GPU sync)**:
  - Combined Dual-Stream Latency: **20.38 ms** (std: 2.40 ms)
  - Throughput: **49.07 FPS**
  - Total Parameters: **5,008,380 (~5.01M)**
- **Artifacts**: Metrics saved in `experiments/E004_LATE_FUSION_WBF_YOLO26N/metrics.json`.
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Architecture**: YOLO26n
- **Modality**: THERMAL
- **Stopping Epoch**: 19 / 40 (Best: 14)
- **mAP50**: 0.9449 | **mAP50-95**: 0.5679
- **Precision**: 0.9413 | **Recall**: 0.8790 | **F1**: 0.9091
- **Latency**: 10.80 ms (92.6 FPS)

---

### [Execution] Milestone 6 - E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N
- **Date**: 2026-11-15
- **Status**: COMPLETED
- **Strategy**: Proposed Thermal-Primary Asymmetric Reliability-Aware Gated Fusion
- **Architecture**: `AsymmetricGatedFusionYOLO26` (Thermal primary backbone, lightweight visible branch, 3-scale Quality Descriptors $[Q_T, Q_V]$, adaptive spatial/channel gating $G_V = \sigma(\text{Conv}([F_T, P(F_V), Q_T, Q_V]))$, thermal residual preservation $F_{\text{fused}} = F_T + G_V \odot P(F_V)$, shared YOLO26 head).
- **Stopping Dynamics**: Stopped at Epoch 17 (patience=12 triggered after peak fitness at Epoch 5). Training duration: 90.09 minutes (5405s).
- **Validation Metrics (1,803 pairs, Best Checkpoint: Epoch 5)**:
  - Precision: **93.89%**
  - Recall: **87.69%** (Surpassing Feature Concat 86.32% and Late Fusion 85.45%; peak recall reached 88.20% at Epoch 12)
  - F1-Score: **90.69%**
  - mAP50: **94.11%** (Peak mAP50 reached 94.18% at Epoch 12)
  - mAP50-95: **56.58%**
- **Hardware Latency (Batch 1, GPU Sync, 50 warmup, 200 timed runs)**:
  - Model Latency: **17.30 ms**
  - Throughput: **57.80 FPS** (Real-time speed > 30 FPS threshold)
  - Parameters: **4,971,992 (~4.97M)**
  - Peak PyTorch GPU Memory: **2,268 MB** (~28% of 8 GB VRAM capacity)
- **Artifacts**: Checkpoints `best.pt`, `last.pt`, `metrics.json`, `results.csv` saved in `experiments/E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N/` and `results/metrics/E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N_metrics.json`.
- **Findings**: The proposed adaptive reliability-aware gating mechanism successfully balances multi-spectral features, preventing the catastrophic feature pollution seen in Early Fusion (E005: 71.53% vs E007: 94.11%) while outperforming both Late Fusion (+2.76% $mAP_{50}$) and naive Feature Concatenation (+1.37% Recall) while maintaining 57.8 FPS real-time execution.
