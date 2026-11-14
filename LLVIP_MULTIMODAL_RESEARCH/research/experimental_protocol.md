# Experimental Protocol

## 1. Dataset Partitions and Integrity
- **Dataset**: LLVIP (Strict Visible-Infrared Paired Dataset for Low-Light Vision).
- **Official Test Set**: 3,463 pairs (to be verified). FROZEN until final Milestone 8 evaluation. NEVER used for training, validation, or hyperparameter selection.
- **Official Train Set**: 12,025 pairs (to be verified).
  - Internal Train Partition: 85% (~10,221 pairs), deterministic split (Seed = 42).
  - Internal Validation Partition: 15% (~1,804 pairs), deterministic split (Seed = 42).
- **Class**: Single class detection — `person` (Class ID: 0).

## 2. Evaluation Metrics
All detectors evaluated on identical splits using standardized COCO / VOC detection protocols:
- **mAP50**: Mean Average Precision at IoU threshold 0.50.
- **mAP50-95**: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 (step 0.05).
- **Precision (P)** and **Recall (R)** at optimal F1 threshold.
- **Inference Latency (ms)**: Measured at Batch Size 1 on NVIDIA GeForce RTX 4060 Laptop GPU.
- **Throughput (FPS)**: $1000 / \text{latency (ms)}$.
- **Model Efficiency**: Parameter count (M), Computational Complexity (GFLOPs), Peak VRAM (MB).

## 3. Hardware and Training Environment
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).
- **Precision**: Automatic Mixed Precision (AMP / FP16) enabled for all experiments.
- **Input Resolutions**: Baseline 640x640, secondary investigation 960x960. Letterboxing with aspect ratio preservation.
- **Sequential Training**: Strict sequential execution to avoid VRAM exhaustion.
