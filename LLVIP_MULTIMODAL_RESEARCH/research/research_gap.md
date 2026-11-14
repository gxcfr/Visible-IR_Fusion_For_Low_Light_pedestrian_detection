# Research Gap Analysis

## Current Limitations in Multimodal RGB-T Pedestrian Detection

1. **Symmetric vs. Asymmetric Resource Allocation**:
   - Most dual-stream architectures (e.g., standard two-stream ResNet or CSPDarknet backbones) deploy identical, symmetric feature extractors for visible and infrared modalities.
   - In nighttime and extreme low-light regimes, the visible stream carries severely degraded SNR, yet symmetric models dedicate 50% of computational budget and parameter capacity to processing near-black or noisy RGB inputs.
   - **Research Gap**: How to design an *asymmetric* architecture where the reliable modality (thermal) forms the primary semantic backbone, while the auxiliary modality (visible) is parsed via a lightweight, computationally lean auxiliary pathway that injects color/texture only when informative.

2. **Absence of Explicit Environmental Reliability Conditioning**:
   - Conventional intermediate fusion methods rely on fixed concatenation or static convolutional weights across all lighting scenarios.
   - In real-world urban surveillance, camera sensors encounter abrupt illumination shifts (e.g., headlights, streetlights, total darkness). Fixed fusion cannot suppress corrupted channels dynamically.
   - **Research Gap**: Formulation of a dynamic, learnable reliability gating mechanism conditioned on both low-level image statistics (luminance, local contrast) and high-level feature activations.

3. **Latency-Accuracy Trade-off for Edge Deployment**:
   - Heavy transformer-based multimodal cross-attention networks (e.g., Deformable Cross-Attention, Swin-based backbones) incur high latency and memory footprints unsuited for edge GPU constraints (such as the 8 GB RTX 4060 Laptop GPU or onboard embedded platforms).
   - **Research Gap**: Real-time capable, lightweight fusion built on next-generation lightweight detectors (YOLO26n / YOLO11n) achieving superior mAP50 without exceeding real-time latency budgets (>30 FPS).
