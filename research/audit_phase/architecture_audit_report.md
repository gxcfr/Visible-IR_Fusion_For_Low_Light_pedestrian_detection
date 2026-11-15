# Architecture Audit Report: E007 Proposed Asymmetric Gated Fusion

## 1. Executive Summary

This report provides a formal architectural audit of **Experiment E007: Proposed Asymmetric Gated Fusion YOLO26n**, implemented in:
`src/models/fusion/asymmetric_gated_fusion_yolo26.py`

The objective of this audit is to rigorously address discrepancies identified by external peer review regarding:
1. Whether explicit quality estimators ($Q_T, Q_V$) are implemented.
2. The true structural symmetry vs. asymmetry of the visible and thermal streams.
3. The exact mathematical formulation of the feature gating operation.
4. The structural integrity and layer indices of the PyTorch computational graph.

---

## 2. Explicit Quality Descriptors ($Q_T, Q_V$) Audit

### 2.1 Peer Review Inquiry
*Does the submitted implementation explicitly compute or receive modality quality descriptors $Q_T$ (thermal quality) and $Q_V$ (visible quality) as claimed in preliminary research documentation?*

### 2.2 Direct Code Inspection Findings
Inspection of `ReliabilityGatingModule` (lines 13–39):
```python
class ReliabilityGatingModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat_thermal: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        proj_rgb = self.proj(feat_rgb)
        concat = torch.cat([feat_thermal, proj_rgb], dim=1)
        gate = self.gate_conv(concat)
        return feat_thermal + gate * proj_rgb
```

### 2.3 Verification Verdict
- **No Explicit Quality Descriptors**: The architecture **does not** compute, estimate, or ingest scalar or spatial quality metrics ($Q_T, Q_V$) such as illumination level, Michelson/RMS contrast, edge gradient density, or signal-to-noise ratio.
- **Accurate Description**: The gating mechanism is an **implicit, feature-conditioned adaptive gating module**. It computes dynamic gate weights directly from the concatenation of intermediate deep convolutional features $[F_T, P(F_V)]$.
- **Correction Action**: All research documentation must cease referring to an "explicit quality estimator" or "calibrated quality descriptors $Q_T, Q_V$". The mechanism must be accurately described as a **learned feature-conditioned cross-attention gating module**.

---

## 3. Stream Symmetry and Parameter Audit

### 3.1 Peer Review Inquiry
*Is the visible encoder a lightweight, structurally asymmetric branch, or does it duplicate the full thermal backbone?*

### 3.2 Quantitative Parameter and Layer Breakdown

| Submodule | Class / Architecture | Layers | Parameters | % of Total Params |
|:---|:---|:---:|:---:|:---:|
| **Thermal Backbone** | YOLO26n layers 0–10 | 11 | 1,365,472 | 27.84% |
| **Visible (RGB) Backbone** | YOLO26n layers 0–10 | 11 | 1,365,472 | 27.84% |
| **Gating Module P3** | Conv-BN-SiLU + Conv-BN-SiLU-Conv-Sigmoid | - | 172,544 | 3.52% |
| **Gating Module P4** | Conv-BN-SiLU + Conv-BN-SiLU-Conv-Sigmoid | - | 172,544 | 3.52% |
| **Gating Module P5** | Conv-BN-SiLU + Conv-BN-SiLU-Conv-Sigmoid | - | 689,152 | 14.05% |
| **Shared Neck & Detection Head** | YOLO26n layers 11–23 | 13 | 1,138,718 | 23.22% |
| **Total Model** | `AsymmetricGatedFusionYOLO26` | **35** | **4,903,902** | **100.0%** |

### 3.3 Verification Verdict
- **Structural Symmetry**: The visible branch is **not** a lightweight encoder. It is an exact structural duplicate of the 11-layer YOLO26n backbone, possessing identical depth, channel capacities, and parameter count (1,365,472 parameters each).
- **Nature of Asymmetry**: The asymmetry in E007 is purely **functional and operational**:
  1. **Primary Semantic Backbone**: Thermal infrared features ($F_T$) form the main backbone trunk and pass through an unmodulated additive identity shortcut ($F_T + \dots$).
  2. **Auxiliary Modulated Stream**: Visible RGB features ($F_V$) are treated as secondary; their injection is linearly projected and adaptively suppressed or scaled by the gating map $G_V \in (0, 1)$.
- **Correction Action**: The architecture must be formally documented as a **dual-symmetric-backbone architecture with asymmetric residual feature injection**, rather than a lightweight asymmetric encoder.

---

## 4. Mathematical Formulation of Feature Gating

The fusion operation performed at feature hierarchy levels $l \in \{\text{P3}, \text{P4}, \text{P5}\}$ is formally defined as:

$$F_{\text{fused}}^{(l)} = F_T^{(l)} + G_V^{(l)} \odot P^{(l)}\left(F_V^{(l)}\right)$$

### 4.1 Component Details:
1. **Visible Feature Projection $P^{(l)}(\cdot)$**:
   - $P^{(l)}(F_V) = \text{SiLU}\left(\text{BatchNorm2d}\left(\text{Conv}_{1\times 1}\left(F_V^{(l)}\right)\right)\right)$
   - Aligns the representation space of the visible features prior to gating and addition without changing channel dimensions $C_l$.
2. **Dynamic Gating Tensor $G_V^{(l)}$**:
   - $G_V^{(l)} = \sigma\left(\text{Conv}_{1\times 1}\left(\text{SiLU}\left(\text{BatchNorm2d}\left(\text{Conv}_{3\times 3}\left([F_T^{(l)}, P^{(l)}(F_V^{(l)})]\right)\right)\right)\right)\right)$
   - $[F_T, P(F_V)]$ denotes channel concatenation across dimension 1, resulting in $2C_l$ channels.
   - The intermediate bottleneck convolution reduces dimensionality to $C_l / 2$ with a $3 \times 3$ spatial receptive field.
   - The final $1 \times 1$ convolution expands channels back to $C_l$.
   - $\sigma(\cdot)$ is the standard element-wise sigmoid activation, constraining all gate coefficients strictly to $(0, 1)$.
3. **Tensor Dimensions**:
   - Level P3: $B \times 128 \times 80 \times 80$ (at $640 \times 640$ input)
   - Level P4: $B \times 128 \times 40 \times 40$
   - Level P5: $B \times 256 \times 20 \times 20$
4. **Thermal Residual Guarantee**:
   - The identity path $F_T^{(l)}$ ensures that in scenarios where visible information is completely corrupted or uninformative ($G_V^{(l)} \to 0$), $F_{\text{fused}}^{(l)} \to F_T^{(l)}$. The network inherently preserves thermal baseline feature integrity.

---

## 5. Computational Graph and Layer Index Verification

Inspection of the forward pass in `AsymmetricGatedFusionYOLO26` confirms the following layer mappings:

```python
# Backbone feature extraction stages
# Layer 4 (C3k2) -> P3 (stride 8, 128 channels)
# Layer 6 (C3k2) -> P4 (stride 16, 128 channels)
# Layer 10 (C2PSA) -> P5 (stride 32, 256 channels)
```

### 5.1 Neck and Head Connections:
The shared neck and head are constructed from layers 11 through 23 of the base YOLO26n model:
- **Layer 11**: `nn.Upsample` (scale_factor=2) applied to `fused_p5` $\to (B, 256, 40, 40)$.
- **Layer 12**: `Concat([Upsample(P5), fused_p4])` $\to (B, 384, 40, 40)$.
- **Layer 13**: `C3k2` block $\to (B, 128, 40, 40)$, saved as intermediate feature `13`.
- **Layer 14**: `nn.Upsample` (scale_factor=2) applied to output of layer 13 $\to (B, 128, 80, 80)$.
- **Layer 15**: `Concat([Upsample(13), fused_p3])` $\to (B, 256, 80, 80)$.
- **Layer 16**: `C3k2` block $\to (B, 128, 80, 80)$, saved as detection feature `16` (P3 scale).
- **Layer 17**: `Conv` (stride 2) applied to layer 16 $\to (B, 128, 40, 40)$.
- **Layer 18**: `Concat([Conv(16), saved[13]])` $\to (B, 256, 40, 40)$.
- **Layer 19**: `C3k2` block $\to (B, 128, 40, 40)$, saved as detection feature `19` (P4 scale).
- **Layer 20**: `Conv` (stride 2) applied to layer 19 $\to (B, 128, 20, 20)$.
- **Layer 21**: `Concat([Conv(19), saved[10]])` $\to (B, 384, 20, 20)$ (where saved[10] is `fused_p5`).
- **Layer 22**: `C3k2` block $\to (B, 256, 20, 20)$, saved as detection feature `22` (P5 scale).
- **Layer 23**: `Detect([saved[16], saved[19], saved[22]])` generating final multi-scale anchor-free detection tensors.

### 5.2 Verification Verdict
- All backbone tapping points (indices 4, 6, 10) and neck routing indices (11 through 23) match the installed YOLO26n PyTorch module graph.
- The forward execution graph is valid and mathematically consistent.

---

## 6. Recommendations for Future Research

1. **Retain Experiment E007 as Feature-Conditioned Adaptive Fusion (Option A)**:
   - Accept E007 as an empirical baseline for feature-conditioned adaptive residual fusion.
   - Accurately document its dual-symmetric backbone and lack of explicit $Q_T, Q_V$ metrics.
2. **Future Investigation of Explicit Quality-Guided Fusion (Option B)**:
   - If explicit quality estimation is desired, design and validate a dedicated Quality Estimation Module (QEM) that extracts measurable image statistics (e.g., local entropy, high-frequency gradient power, illuminance distributions) to steer gating weights.
   - This should be conducted in a newly numbered experiment (e.g., **E008**) with an explicit ablation isolating the quality estimator from the feature gating trunk.
