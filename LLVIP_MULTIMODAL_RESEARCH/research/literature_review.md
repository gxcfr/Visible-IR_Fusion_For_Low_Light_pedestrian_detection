# Literature Review: Multimodal Visible-Infrared Pedestrian Detection

## 1. Domain Foundations: Multispectral / Visible-Infrared Detection
Multispectral pedestrian detection leverages complementary information from visible (RGB) and infrared (thermal / LWIR) spectra:
- **Visible Spectrum (RGB)**: High spatial resolution, rich color and texture cues, semantic edge definition. Highly vulnerable to adverse illumination (nighttime, deep shadows, glare, smoke).
- **Thermal Infrared (LWIR)**: Emitted thermal radiation invariant to illumination changes, high contrast for biological targets against cooler backgrounds. Suffers from blur, lack of interior textures, halo artifacts, and background thermal crossover (summer daytime).

## 2. Benchmark Datasets
- **LLVIP (2021)**: Visible-Infrared Paired Dataset for Low-Light Vision (Jia et al., ICCV Workshops 2021). Strict spatial-temporal alignment in urban surveillance scenarios under low-light nighttime settings. 15,488 aligned pairs (12,025 train / 3,463 test), class: `person`.
- **KAIST Multispectral**: Earlier benchmark (Hwang et al., CVPR 2015), known for spatial misalignment and annotation noise.
- **CVC-14 / FLIR**: Widely used automotive datasets with varying alignment and environmental conditions.

## 3. Classical vs. Modern Fusion Architectures
- **Early Fusion (Pixel-level)**: Stacking RGB and Thermal into a 4-channel input $(R, G, B, T)$. Fast and simple, but forces early convolutional filters to learn cross-spectral representations without allowing spectral-specific feature specialization.
- **Late Fusion (Decision-level)**: Independent RGB and Thermal detector networks combined via post-processing (e.g., NMS, Soft-NMS, Weighted Boxes Fusion / WBF). Preserves unimodal integrity but doubles compute overhead and fails to exploit cross-modal intermediate feature synergies.
- **Intermediate Feature Fusion**: Separate backbones extract multi-scale feature pyramids $(P_3, P_4, P_5)$ merged via concatenation, element-wise addition, or attention mechanisms before passing to a shared detection head.

## 4. Key Candidate Architectures and Literature Candidates (To Be Systematically Audited)
1. **MBNet (2020)**: Modality-balance Network for multispectral pedestrian detection (Zhou et al., ECCV 2020). Addresses modality imbalance and differential feature discrimination.
2. **LLVIP Baseline Paper (2021)**: Jia et al., "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision", ICCV 2021.
3. **TFDet (2025)**: Target-aware fusion network for multimodal detection.
4. **Deformable Cross-Attention Transformer for Weakly Aligned RGB-T Pedestrian Detection (2025)**.
5. **MCFusion (2026)**: Multi-cue feature fusion for multispectral detection.
6. **Uni-MPedFormer (2026)**: Unified multispectral pedestrian detection transformer.
7. **WCCNet (2026)**: Weak cross-modal calibration network.
8. **Ultralytics YOLO Family (YOLO11 / YOLO26)**: State-of-the-art real-time detection architectures.

*Note: All citations and empirical claims will be verified against verified publisher DOIs before final citation.*
