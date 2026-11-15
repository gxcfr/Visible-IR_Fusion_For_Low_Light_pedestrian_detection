# Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection

### Project Status: Milestones 1–6 COMPLETED (All Experiments E001–E007 Fully Executed & Benchmarked)

- **Dataset**: LLVIP (`C:\LLVIP\LLVIP`) — 15,488 verified pairs (100% paired, single class `person`).
- **Splits**: Scene-aware group split (Train: 10,222 pairs / Val: 1,803 pairs / Test: 3,463 pairs strictly untouched).
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace, SM 8.9).
- **Primary Architecture**: Ultralytics **YOLO26n** adopted for all baselines and multimodal architectures (E002–E007). Historical baseline retained: `E001_RGB_YOLO11N`.
- **Git Remote**: `https://github.com/gxcfr/Visible-IR_Fusion_For_Low_Light_pedestrian_detection.git`

---

## Master Experimental Results (Evaluated on Identical 1,803 Val Pairs)

| Exp ID | Architecture | Modality / Strategy | Precision | Recall | F1 | $mAP_{50}$ | $mAP_{50-95}$ | Latency (ms) | FPS | Parameters | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | YOLO11n | RGB-only *(Historical)* | 91.24% | 72.22% | 80.62% | 80.57% | 41.31% | **10.69** | **93.55** | 2.59M | **COMPLETED** |
| **E002** | YOLO26n | RGB-only Baseline | 89.39% | 71.91% | 79.70% | 79.34% | 40.47% | 15.80 | 63.28 | **2.50M** | **COMPLETED** |
| **E003** | YOLO26n | Thermal-only Baseline | **94.13%** | **87.90%** | **90.91%** | **94.49%** | **56.79%** | 10.80 | 92.60 | **2.50M** | **COMPLETED** |
| **E004** | YOLO26n | Late Fusion (WBF) | 88.15% | 85.45% | 86.78% | 91.35% | 52.49% | 20.38 | 49.07 | 5.01M | **COMPLETED** |
| **E005** | YOLO26n | Early Fusion (4-Channel Stem) | 82.66% | 64.59% | 72.52% | 71.53% | 37.40% | 13.64 | 73.31 | 2.57M | **COMPLETED** |
| **E006** | YOLO26n | Feature Concatenation | 93.48% | 86.32% | 89.76% | 93.58% | 56.62% | 14.73 | 67.89 | 4.14M | **COMPLETED** |
| **E007** | Proposed | **Asymmetric Gated Fusion** | **93.89%** | **87.69%** | **90.69%** | **94.11%** | **56.58%** | **17.30** | **57.80** | **4.97M** | **COMPLETED** |

---

## Research Review Deliverables
- **External Review Package**: `research_review_bundle.zip`
- **Review Summary**: [research_review_summary.md](file:///C:/VI%20Fusion%20for%20LL-PD/research_review_summary.md)
- **Detailed Research Spec & Logs**: 👉 [LLVIP_MULTIMODAL_RESEARCH/README.md](file:///C:/VI%20Fusion%20for%20LL-PD/LLVIP_MULTIMODAL_RESEARCH/README.md)
