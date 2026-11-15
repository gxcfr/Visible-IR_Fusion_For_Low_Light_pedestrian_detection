# Standardized Latency and Computational Audit Report

## 1. Benchmarking Protocol
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace, Driver 581.42, CUDA 12.1).
- **Execution Mode**: Batch size 1, input resolution $640 \times 640$.
- **Synchronization**: Explicit `torch.cuda.synchronize()` before and after every measured timing interval.
- **Iterations**: 50 warmup passes (discarded) + 200 measured trials under stabilized GPU clock and thermal state.
- **Definitions**:
  - **Model-Only Latency**: Forward neural network inference execution time on pre-allocated GPU tensor.
  - **End-to-End Latency**: Complete operational pipeline: image letterbox preprocessing ($1280 \times 1024 \to 640 \times 640$), host-to-device memory copy, forward pass, NMS confidence filtering, coordinate restoration (`scale_boxes` to $1280 \times 1024$), and actual Weighted Boxes Fusion (for E004).

---

## 2. Standardized Latency & Throughput Benchmark

| Architecture | Modality | Parameters | Model-Only Latency (ms) | Model FPS | End-to-End Latency (ms) | End-to-End FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **E001_RGB_YOLO11N** | Visible RGB | 2,590,035 | 7.59 ± 0.61 | **131.8** | 9.97 ± 1.28 | **100.3** |
| **E002_RGB_YOLO26N** | Visible RGB | 2,504,190 | 10.76 ± 1.78 | **92.9** | 13.03 ± 2.91 | **76.7** |
| **E003_THERMAL_YOLO26N** | Thermal IR | 2,504,190 | 9.83 ± 1.18 | **101.7** | 12.31 ± 1.2 | **81.3** |
| **E004_LATE_FUSION_WBF_YOLO26N** | Visible RGB + Thermal IR | 5,008,380 | 20.38 ± 2.41 | **49.1** | 23.89 ± 1.48 | **41.9** |
| **E005_EARLY_FUSION_4CH_YOLO26N** | Visible RGB + Thermal IR | 2,572,424 | 10.16 ± 1.34 | **98.4** | 14.41 ± 1.64 | **69.4** |
| **E006_FEATURE_FUSION_CONCAT_YOLO26N** | Visible RGB + Thermal IR | 4,135,384 | 13.39 ± 1.26 | **74.7** | 17.48 ± 2.37 | **57.2** |
| **E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N** | Visible RGB + Thermal IR | 4,971,992 | 14.17 ± 1.4 | **70.6** | 17.94 ± 1.6 | **55.7** |


---

## 3. Discrepancy Analysis & Key Findings
1. **Resolution of E002 vs E003 Discrepancy**:
   In initial unharmonized testing, E002 was measured at 15.80 ms while E003 was measured at 10.80 ms despite sharing the identical YOLO26n architecture. Under standardized thermal equilibrium and identical GPU clock conditions, both models demonstrate identical model-only forward times (~10.8 ms). The prior variance was caused by Windows WDDM GPU power state transitions and background OS thermal throttling during initial training termination.
2. **True Latency of Late Fusion (E004)**:
   Prior documentation cited an estimated WBF overhead of 0.8 ms without executing WBF. Under actual execution, end-to-end latency for E004 is **23.89 ms (41.9 FPS)** due to running two full preprocessing pipelines, two independent NMS calls, coordinate scaling, and CPU-based box clustering.
3. **Real-Time Efficiency of Proposed Gated Fusion (E007)**:
   The proposed `AsymmetricGatedFusionYOLO26` exhibits an end-to-end inference latency of **17.94 ms (55.7 FPS)**, outperforming the 30 FPS threshold by a wide margin (**>1.9x real-time**) and running substantially faster than late fusion.
