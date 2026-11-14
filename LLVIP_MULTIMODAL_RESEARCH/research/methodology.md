# Proposed Research Methodology

## Working Architecture: Thermal-Primary Asymmetric Reliability-Aware Fusion

### 1. Conceptual Framework
Under low-light conditions, thermal radiation provides the persistent geometric and localization baseline of human presence. Visible imagery provides discriminating fine-grained textures, boundary sharpness, and background differentiation when illuminated. The system models feature integration as an asymmetric reliability-gated residual process:

$$\mathbf{F}_{\text{fused}} = \mathbf{F}_T + \mathcal{G}_V(\mathbf{F}_T, \mathcal{P}(\mathbf{F}_V), \mathbf{Q}_T, \mathbf{Q}_V) \odot \mathcal{P}(\mathbf{F}_V)$$

where:
- $\mathbf{F}_T$: Primary thermal feature representation extracted by the thermal backbone.
- $\mathbf{F}_V$: Auxiliary visible feature representation extracted by a lightweight branch.
- $\mathcal{P}(\cdot)$: Dimension projection layer aligning channel and spatial dimensions.
- $\mathbf{Q}_T, \mathbf{Q}_V$: Modality quality descriptors (luminance, contrast, activation variance).
- $\mathcal{G}_V(\cdot)$: Learnable gating module outputting spatial-channel attenuation weights $\in [0, 1]$.

### 2. Dual-Gating Formulation (Ablation Variant)
To test whether the thermal modality should also be subject to suppression (e.g. during thermal washout or thermal crossover):
$$\mathbf{F}_{\text{fused}} = \mathcal{G}_T \odot \mathcal{P}_T(\mathbf{F}_T) + \mathcal{G}_V \odot \mathcal{P}_V(\mathbf{F}_V)$$

Both formulations will be tested empirically against fixed concatenation and unimodal baselines.
