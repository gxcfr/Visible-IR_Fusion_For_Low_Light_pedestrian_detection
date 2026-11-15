# Comprehensive Multimodal Fusion Ablation Protocol

## 1. Executive Summary & Purpose

The objective of this ablation study is to systematically isolate, evaluate, and quantify the performance contributions of each architectural component within the proposed Asymmetric Gated Fusion framework. 

Previous reviews noted that post-hoc or informal feature comparisons are insufficient for external peer review. This protocol defines a rigorous, controlled suite of **six distinct experimental configurations**, establishing:
1. Clear demarcation between **training-time structural ablations** and **inference-time interventions**.
2. Controlled experimental conditions ensuring cross-configuration parity.
3. Theoretical grounding for why inference-time gate interventions ($G_V = 0$) do not equal independently trained unimodal models.
4. An empirical roadmap for future execution upon authorization.

---

## 2. Controlled Experimental Variables and Parity Guarantees

To ensure rigorous scientific validity, all training-time ablations must adhere to identical hyperparameters and environmental constraints:

| Parameter / Condition | Standardized Specification | Rationale for Parity |
|:---|:---|:---|
| **Dataset Partitions** | LLVIP Train (10,643 pairs) / LLVIP Val (1,803 pairs) | Fixed split; official test set (3,463 pairs) strictly untouched. |
| **Input Resolution** | $640 \times 640$ letterboxed ($1280 \times 1024$ native) | Consistent receptive field and anchor scaling. |
| **Optimization Algorithm** | SGD with momentum ($0.937$), weight decay ($0.0005$) | Identical trajectory dynamics. |
| **Learning Rate Schedule** | Cosine annealing, initial $\text{lr}_0 = 0.002$, final $\text{lr}_f = 0.01 \times \text{lr}_0$ | Consistent convergence rate. |
| **Batch Size & Accumulation** | Batch size = 4 (hardware constrained on RTX 3050 Laptop 4GB) | Identical effective batch statistics. |
| **Early Stopping Policy** | Patience = 12 epochs on validation mAP50-95 | Objective termination preventing overfitting. |
| **Random Seed** | Fixed Seed = 42 for primary runs | Deterministic initialization and augmentations. |
| **Evaluation Harness** | Unified Evaluator (`evaluate_unified.py`) in native $1280 \times 1024$ space | Identical IoU matching, COCOeval AP integration, confidence thresholds. |

---

## 3. Systematic Breakdown of the Six Ablation Configurations

```
                                  ┌──────────────────────────────┐
                                  │   Input Modality Streams     │
                                  │  Visible (RGB) & Thermal (IR) │
                                  └──────────────┬───────────────┘
                                                 │
            ┌─────────────────────┬──────────────┴───────┬──────────────────────┐
            ▼                     ▼                      ▼                      ▼
    [Conf 1: E003]         [Conf 2: E006]         [Conf 3: E007/8]       [Conf 4: Fixed Gate]
     Thermal-Only          Feature Concat        Learned Gated Fusion    Static Weights
     (Single Stream)      (Direct Addition/Cat)   (Feature Conditioned)   (G_V = 0.5 / 1.0)
            │                     │                      │                      │
            └─────────────────────┼──────────────────────┴──────────────────────┘
                                  ▼
                    [Conf 5: Inference Intervention]
                     G_V = 0.0 on Trained Model
                     (Co-adaptation Analysis)
                                  │
                                  ▼
                    [Conf 6: Explicit Quality Fusion]
                     QEM-Guided Gating (Entropy/Grad)
```

---

### Configuration 1: Thermal-Only Baseline Reference (Unimodal Lower Bound / Reference)
- **Nature**: Training-time unimodal baseline.
- **Model Equivalent**: Existing experiment **E003**.
- **Architecture**: Single-stream YOLO26n receiving purely 3-channel thermal infrared imagery.
- **Scientific Role**: Establishes the performance ceiling of the dominant modality. In dark surveillance settings, thermal infrared provides the majority of pedestrian heat signatures. Any proposed fusion mechanism must justify its parameter and computational overhead against this strong baseline.

---

### Configuration 2: Dual-Stream Feature Concatenation (Ungated Multimodal Baseline)
- **Nature**: Training-time structural ablation.
- **Model Equivalent**: Existing experiment **E006**.
- **Architecture**: Dual identical backbones (Visible + Thermal). At fusion stages (P3, P4, P5), feature maps from both streams are concatenated along channel dimensions ($\mathbb{R}^{2C \times H \times W}$) and compressed back to $C$ channels using a $1\times 1$ convolution:
  $$F_{\text{fused}} = \text{Conv}_{1\times 1}([F_{\text{Thermal}}, F_{\text{Visible}}])$$
- **Scientific Role**: Evaluates whether an unweighted, non-adaptive combination of modalities provides synergy or introduces noise corruption from degraded visible channels.

---

### Configuration 3: Learned Feature-Conditioned Adaptive Gating (Proposed Model)
- **Nature**: Training-time proposed architecture.
- **Model Equivalent**: Existing experiment **E007** (80-class) / Proposed **E008** (1-class).
- **Architecture**: Asymmetric dual-stream architecture where visible features are dynamically modulated via a spatial-channel gating module before being injected into the thermal backbone:
  $$G_V = \sigma(\text{Conv}_{1\times 1}(\delta(\text{Conv}_{3\times 3}([F_{\text{Thermal}}, F_{\text{Visible}}]))))$$
  $$F_{\text{fused}} = F_{\text{Thermal}} + G_V \odot F_{\text{Visible}}$$
- **Scientific Role**: Evaluates whether learned, feature-driven gating selectively admits helpful visible spatial details (e.g. edge sharpness, texture) while suppressing darkness and sensor noise.

---

### Configuration 4: Fixed-Gate Fusion (Static Weight Ablation)
- **Nature**: Training-time structural ablation.
- **Variants**:
  - **4A (Equal Fusion)**: $G_V \equiv 0.5$ (constant across all spatial locations and channels).
  - **4B (Direct Additive Fusion)**: $G_V \equiv 1.0$ ($F_{\text{fused}} = F_{\text{Thermal}} + F_{\text{Visible}}$).
- **Implementation**: The gating subnet is removed or replaced with a non-trainable constant tensor during training and validation.
- **Scientific Role**: Disentangles the contribution of **learnable adaptive gating** from simple **linear feature blending**. If Configuration 3 outperforms Configuration 4, the advantage stems from content-aware selection rather than merely broadening feature representation.

---

### Configuration 5: Gate-Disabled Inference Analysis (Post-Hoc Intervention)
- **Nature**: Inference-time intervention on the fully trained Configuration 3 (E007/E008) checkpoint.
- **Execution**: The model is evaluated without retraining, but the gate values are forced to zero ($G_V \leftarrow 0.0$) at every forward pass, completely severing visible feature injection into the thermal stream.
- **Theoretical Distinction (Why Inference Intervention $\ne$ Unimodal Training)**:
  > **Crucial Methodological Concept: Co-adaptation of Weights**
  >
  > A common reviewer misconception is to equate $G_V = 0$ inference with a thermal-only model. They are fundamentally different:
  > 1. **In Configuration 1 (E003)**: The thermal backbone weights were optimized over 40+ epochs with the sole objective of extracting pedestrian features independently from thermal input alone.
  > 2. **In Configuration 3 (E007/E008)**: The thermal backbone weights co-adapted during gradient descent with the visible stream. The upper neck and detection heads expect a blended representation ($F_T + G_V \odot F_V$). Forcing $G_V = 0$ at inference removes features that subsequent convolutional filters have learned to anticipate, causing feature magnitude and distributional shifts.
  >
  > Therefore, Configuration 5 measures **model reliance and feature co-dependence**, NOT unimodal thermal performance.

---

### Configuration 6: Explicit Quality-Estimation-Guided Fusion (QEM-Guided Fusion)
- **Nature**: Training-time architectural extension.
- **Architecture**: Replaces the implicit convolutional gate with an **Explicit Quality Estimation Module (QEM)**:
  - Input: Raw image-space or early feature statistics measuring local visible contrast (standard deviation in local patches), local high-frequency energy (Laplacian gradient magnitude), and local entropy.
  - The gate $G_V$ is conditioned directly on these explicit quality indicators:
    $$Q_V = \text{Norm}(\nabla^2 I_V) \in [0, 1]$$
    $$G_V = \alpha \cdot Q_V + (1 - \alpha) \cdot \text{Gate}_{\text{feature}}(F_T, F_V)$$
- **Scientific Role**: Evaluates whether injecting domain-specific physical image quality priors improves over pure data-driven convolutional feature gating, directly addressing the external review question on calibrated sensor reliability.

---

## 4. Comprehensive Metrics and Comparison Matrix

All six configurations will be evaluated using the standardized Unified Evaluator on the 1,803 validation image pairs:

| Config ID | Description | Mode | Params | Gating Mechanism | Primary Question Addressed |
|:---:|:---|:---:|:---:|:---:|:---|
| **Conf 1** | Thermal-Only Baseline (E003) | Train | 2.37M | None (Single Stream) | What is the unimodal thermal performance ceiling? |
| **Conf 2** | Feature Concatenation (E006) | Train | 4.97M | Static $1\times 1$ conv | Does unweighted multimodal concatenation degrade thermal quality? |
| **Conf 3** | Learned Gated Fusion (E007 / E008) | Train | 4.97M / 4.90M | Learned Spatial-Channel $\sigma(\cdot)$ | Does adaptive feature gating selectively harvest visible cues? |
| **Conf 4A** | Fixed Gate ($G_V = 0.5$) | Train | 4.74M | Fixed constant $0.5$ | Is adaptive gating superior to equal linear blending? |
| **Conf 4B** | Fixed Gate ($G_V = 1.0$) | Train | 4.74M | Fixed constant $1.0$ | Is adaptive gating superior to direct feature addition? |
| **Conf 5** | Gate Severing ($G_V = 0.0$ at test) | Eval only | 4.97M / 4.90M | Clamped to $0.0$ at inference | How strongly do neck/head layers co-adapt to visible features? |
| **Conf 6** | Explicit QEM Quality Fusion | Train | 4.98M | Physical metric + feature gate | Do explicit physical quality priors outperform pure convolutional gating? |

### Target Evaluation Metrics
For every configuration, the following metrics will be tabulated:
1. **mAP50** (Standard IoU threshold 0.50).
2. **mAP50-95** (COCO standard average over 10 thresholds).
3. **Precision, Recall, F1** at maximum F1 operating threshold.
4. **Latency** (Batch size 1, FP32, milliseconds on target GPU).
5. **Gated Feature Distribution** (Mean $\mu_G$, standard deviation $\sigma_G$, activation histograms across scales).
