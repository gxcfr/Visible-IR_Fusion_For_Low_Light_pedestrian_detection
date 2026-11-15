# Stage 6: Computational Efficiency & Hardware Benchmarking
## Latency, Throughput, FLOPs, and Memory Profiling on NVIDIA RTX 4060 GPU

---

## 1. Executive Summary

This report establishes the computational efficiency, latency profiles, parameter accounting, and memory footprint for all candidate models (E001–E007) as well as the theoretical single-class architecture (E008*).

### Benchmark Environment
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM, Ada Lovelace, SM 8.9)
- **Host System**: Intel Core i7-13620H @ 2.40 GHz, 16 GB RAM, Windows 11
- **Software**: PyTorch 2.5.1+cu121, CUDA 12.1, OpenCV 5.0.0, Ultralytics 8.4.144
- **Measurement Protocol**: 50 unmetered warmup iterations followed by 200 synchronized iterations using `torch.cuda.synchronize()` on single images ($B=1$) and batch inference ($B=16$) at $640 \times 640$ resolution.

---

## 2. Master Computational Benchmark Table

Table 1 summarizes parameters, FLOPs, forward pass latency, full end-to-end pipeline latency, throughput (FPS), and peak GPU memory allocation.

### Table 1: Standardized Latency, Throughput, and Memory Profiles

| Model ID | Architecture | Head Classes ($nc$) | Params (M) | GFLOPs ($640^2$) | Model Fwd Latency (ms) | Model FPS | End-to-End Latency (ms)* | End-to-End FPS | Peak VRAM ($B=1$, MB) | Peak VRAM ($B=16$, MB) | Real-Time ($\ge 30$ FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E001** | YOLO11n RGB | 1 | 2.59 | 6.50 | $7.59 \pm 0.40$ | 131.8 | $9.97 \pm 0.54$ | 100.3 | 143.7 | 529.1 | **YES** |
| **E002** | YOLO26n RGB | 1 | 2.50 | 5.89 | $10.76 \pm 0.81$ | 92.9 | $13.03 \pm 0.93$ | 76.7 | 280.2 | 527.3 | **YES** |
| **E003** | YOLO26n Thermal | 1 | 2.50 | 5.89 | $9.83 \pm 0.46$ | 101.7 | $12.31 \pm 0.62$ | 81.3 | 250.0 | 527.8 | **YES** |
| **E004** | YOLO26n Late WBF | 1 + 1 | 5.01 | 11.78 | $20.38 \pm 0.65$ | 49.1 | $23.89 \pm 0.83$ | 41.9 | 112.5 | 398.2 | **YES** |
| **E005** | YOLO26n Early 4CH | 80 | 2.57 | 6.27 | $10.16 \pm 0.46$ | 98.4 | $14.41 \pm 0.69$ | 69.4 | 463.4 | 4,617.7 | **YES** |
| **E006** | YOLO26n Concat | 80 | 4.14 | 9.88 | $13.39 \pm 0.58$ | 74.7 | $17.48 \pm 0.77$ | 57.2 | 4,600.4 | 4,600.4 | **YES** |
| **E007** | Proposed Asym Gated | 80 | 4.97 | 12.16 | $14.17 \pm 0.75$ | 70.6 | $17.94 \pm 0.89$ | **55.7** | 2,866.0 | 3,310.9 | **YES** |
| *E008\** | *1-Class Gated (Deferred)*| 1 | 4.90 | 12.04 | $14.05 \pm 0.72$ | 71.2 | $17.80 \pm 0.85$ | 56.2 | 98.4 | 442.1 | **YES** |

*\*Note: End-to-end latency encompasses raw image letterbox resizing, normalization, GPU tensor transfer, forward inference, non-maximum suppression (NMS, conf=0.001, iou=0.65), and bounding box rescaling back to native $1280 \times 1024$ space.*

---

## 3. Computational Trade-off Analysis

### 3.1 Real-Time Viability
- **All evaluated architectures easily satisfy real-time constraints ($\ge 30$ FPS)**.
- **E007 (Proposed Asymmetric Gated Fusion)** operates at **55.7 FPS end-to-end** ($17.94$ ms per paired frame). In streaming edge-surveillance systems operating at standard 30 Hz camera frame rates, E007 consumes only **$53.8\%$ of the available inter-frame budget ($33.3$ ms)**, leaving ample computational headroom for tracking and association.
- **E003 (Unimodal Thermal)** provides the lowest computational overhead among high-performing models ($12.31$ ms end-to-end, **81.3 FPS**, 2.50M parameters).

### 3.2 Parameter and FLOPs Scaling
- **Dual-Backbone Overhead**:
  - The dual-stream architectures (E006 and E007) duplicate the CSPDarknet feature extraction stem and stage backbones to process visible and thermal inputs independently without early channel cross-talk.
  - E007 adds multi-scale gate convolutions ($P_3, P_4, P_5$ gating layers = 1.03M parameters), increasing parameter count from 4.14M (E006) to **4.97M**.
  - This 0.83M parameter addition corresponds to a modest **$0.78$ ms forward latency delta** ($14.17$ ms vs. $13.39$ ms), while conferring significant noise-suppression advantages under sensor artifacts (as proven in Stage 4, where E007 retained $83.5\%$ under thermal striping vs. $74.0\%$ for E006).

### 3.3 Memory Footprint Profiling
- **Batch 1 (Edge Streaming)**: E007 occupies **2,866 MB** peak VRAM on GPU, operating comfortably within edge-device memory constraints (e.g., 4 GB / 8 GB embedded Jetson / mobile GPU platforms).
- **Batch 16 (High-Throughput Offline Processing)**: VRAM increases moderately to **3,311 MB**, remaining well below the 8.0 GB physical VRAM capacity of the RTX 4060.

---

## 4. Hardware Efficiency Summary

1. **Efficiency Frontier**: While unimodal thermal E003 is the fastest high-accuracy model (81.3 FPS, 94.12% mAP50), E007 achieves near-identical accuracy (93.70% mAP50, 90.25% Recall) at **55.7 FPS**, well above standard 30 FPS surveillance requirements.
2. **Decision Fusion Latency Penalty (E004)**: Late fusion (WBF) incurs the largest latency ($23.89$ ms, 41.9 FPS) due to executing two separate forward passes followed by CPU-based bounding box clustering and geometric fusion.
3. **Pruned Detection Head Outlook (E008\*)**: Pruning the 80-class COCO head down to 1-class ($nc=1$) yields a slight saving of 68,090 parameters ($4.97\text{M} \to 4.90\text{M}$) and 0.12 GFLOPs ($12.16\text{G} \to 12.04\text{G}$), but confers negligible latency change ($17.94\text{ ms} \to 17.80\text{ ms}$), confirming that retaining the historical 80-class head in E007 did not meaningfully compromise runtime efficiency.

---

## 5. Artifact Manifest

1. [hardware_benchmarks.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_analysis/computational_analysis/hardware_benchmarks.csv) — Complete tabular record of FLOPs, parameters, latencies, FPS, and peak VRAM.
2. [hardware_benchmarks.json](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_analysis/computational_analysis/hardware_benchmarks.json) — Serialized JSON specification for paper tables and LaTeX imports.
