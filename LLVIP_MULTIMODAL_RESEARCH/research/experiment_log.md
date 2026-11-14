# Chronological Experiment Log

## Experiment Tracking Protocol
Every experiment receives a unique deterministic identifier:
- `E001_YOLO11N_RGB`
- `E002_YOLO11N_THERMAL`
- `E003_YOLO26N_RGB`
- `E004_YOLO26N_THERMAL`
- `E005_YOLO26N_WBF_LATE_FUSION`
- `E006_CUSTOM_EARLY_FUSION`
- `E007_CUSTOM_FEATURE_FUSION`
- `E008_PROPOSED_ADAPTIVE_FUSION`

---

## Log Entries

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
