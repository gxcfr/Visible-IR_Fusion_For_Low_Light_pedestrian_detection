# Multimodal Robustness Stress-Testing Protocol

## 1. Executive Summary & Epistemological Boundaries

A central motivation for multimodal pedestrian detection is maintaining perceptual reliability when one sensing modality experiences physical failure or environmental degradation. 

However, external academic reviews highlighted that evaluating models on naturally occurring validation sets (such as the four scenes of LLVIP) does not provide systematically controlled stress envelopes. Furthermore, claims of "proven real-world environmental resilience" based purely on synthetic corruptions are scientifically flawed.

This protocol defines a **rigorous synthetic stress-testing framework** to evaluate the degradation resistance and failure modes of:
1. **E002**: Unimodal Visible Baseline (YOLO26n RGB)
2. **E003**: Unimodal Thermal Baseline (YOLO26n Thermal)
3. **E006**: Dual-Stream Feature Concatenation
4. **E007**: Proposed Asymmetric Gated Fusion (80-class)
5. **E008**: Proposed Asymmetric Gated Fusion (1-class, upon training)

### Critical Epistemological Boundary & Qualification
> [!IMPORTANT]
> **Scientific Qualification of Synthetic Corruptions**:
> Synthetic perturbations (digital Gaussian noise, artificial dimming, mathematical blurring) are mathematical abstractions. They **do not perfectly model real-world atmospheric obscurants** (e.g. advection fog, rain droplet backscatter, road salt encrustation, or thermal microbolometer drift).
> 
> Therefore, this protocol evaluates **algorithmic stress-tolerance and gating responsiveness under controlled perturbations**, rather than claiming "field-proven all-weather survivability". All findings from this protocol must be reported as stress-testing case analyses.

---

## 2. Controlled Degradation Taxonomy & Severity Levels

All degradations are applied deterministically to the 1,803 validation image pairs across **five increasing severity levels** ($L_1$ to $L_5$). All evaluations are conducted without modifying or fine-tuning model weights.

```
                           ┌─────────────────────────────────────┐
                           │      Original Validation Pair       │
                           │      (RGB $I_V$, Thermal $I_T$)     │
                           └──────────────────┬──────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
     [Branch A: Visible Only]       [Branch B: Thermal Only]       [Branch C: Simultaneous]
     - Brightness Attenuation       - Contrast Blooming/Washout    - Visible Gaussian Noise
     - Additive Gaussian Noise      - Fixed-Pattern Striping Noise   + Thermal Contrast Loss
     - Gaussian Defocus Blur        - Random Dead Pixel Dropout
```

---

### Branch A: Visible Modality Corruptions (Thermal Stream Pristine)

Simulates adverse optical conditions (failing streetlights, dense atmospheric haze, electronic sensor noise) while the long-wave infrared sensor operates under optimal thermal differential:

1. **A1: Brightness Attenuation (Severe Underexposure / Lighting Failure)**:
   $$I_V'(x, y) = \text{clip}(I_V(x, y) \cdot \alpha, 0, 255)$$
   - Levels: $L_1: \alpha = 0.80$, $L_2: \alpha = 0.60$, $L_3: \alpha = 0.40$, $L_4: \alpha = 0.20$, $L_5: \alpha = 0.05$ (near-pitch black).
   - **Hypothesis**: As $\alpha \to 0.05$, unimodal Visible (E002) will collapse to $\approx 0$ mAP, while Gated Fusion (E007/E008) should dynamically attenuate gate weights $G_V$ to prevent visible noise from corrupting the intact thermal representation.

2. **A2: Additive High-Frequency Sensor Noise (Gaussian White Noise)**:
   $$I_V'(x, y) = \text{clip}(I_V(x, y) + \mathcal{N}(0, \sigma^2 \cdot 255^2), 0, 255)$$
   - Levels: $L_1: \sigma = 0.02$, $L_2: \sigma = 0.05$, $L_3: \sigma = 0.10$, $L_4: \sigma = 0.18$, $L_5: \sigma = 0.28$.
   - **Hypothesis**: High-frequency noise injects spurious edge gradients. Unweighted concatenation (E006) will propagate high-frequency artifacts into neck features, whereas gated fusion should isolate the thermal stream.

3. **A3: Defocus / Atmospheric Scattering Blur (Gaussian Filter)**:
   $$I_V'(x, y) = I_V(x, y) * \mathcal{K}_{\text{Gaussian}}(k, \sigma_b)$$
   - Levels (Kernel size $k$, $\sigma_b$): $L_1: (3\times 3, 1.0)$, $L_2: (7\times 7, 2.0)$, $L_3: (11\times 11, 3.5)$, $L_4: (17\times 17, 5.0)$, $L_5: (25\times 25, 8.0)$.
   - **Hypothesis**: Severe blur eliminates high-frequency texture while preserving low-frequency illumination. Tests whether the fusion mechanism can utilize thermal edge information when visible contours are eliminated.

---

### Branch B: Thermal Modality Corruptions (Visible Stream Pristine)

Simulates thermal crossover, diurnal thermal inversion, or sensor calibration degradation (e.g., thermal blooming where background and human skin temperatures equilibrate, or microbolometer readout artifacts):

1. **B1: Thermal Contrast Compression (Thermal Inversion / Washout)**:
   $$I_T'(x, y) = \text{clip}(\bar{I}_T + \gamma \cdot (I_T(x, y) - \bar{I}_T), 0, 255)$$
   Where $\bar{I}_T$ is the global frame mean thermal intensity.
   - Levels: $L_1: \gamma = 0.75$, $L_2: \gamma = 0.50$, $L_3: \gamma = 0.30$, $L_4: \gamma = 0.15$, $L_5: \gamma = 0.02$ (complete thermal washout).
   - **Hypothesis**: As thermal contrast collapses, unimodal thermal performance drops sharply. Multimodal models must rely on visible structural boundaries (e.g. clothing contrast, silhouette outlines) where visible illumination allows.

2. **B2: Uncooled Microbolometer Non-Uniformity / Fixed-Pattern Striping Noise**:
   Simulates uncalibrated readout columns common in commercial infrared focal plane arrays:
   $$I_T'(x, y) = \text{clip}(I_T(x, y) + S(x), 0, 255), \quad S(x) \sim \mathcal{N}(0, \sigma_s^2)$$
   - Levels: $L_1: \sigma_s = 5$, $L_2: \sigma_s = 15$, $L_3: \sigma_s = 30$, $L_4: \sigma_s = 50$, $L_5: \sigma_s = 80$.

---

### Branch C: Joint / Simultaneous Modality Stress-Testing

Simulates catastrophic combined environmental conditions:
- **C1: Optical Darkness + Thermal Washout**:
  Visible brightness attenuated by $L_k$ alongside thermal contrast compressed by $L_k$.
- **C2: Optical Sensor Noise + Thermal Striping**:
  Simultaneous additive electronic noise in both optical and infrared channels.

---

## 3. Quantitative Evaluation Framework & Degradation Metrics

For each model and degradation scenario at each severity level $L \in \{L_1, \dots, L_5\}$, the following quantitative indicators will be logged:

### 3.1 Primary Detection Metrics
- **mAP50**: Mean Average Precision at IoU 0.50 (in native $1280 \times 1024$ space).
- **mAP50-95**: Mean Average Precision averaged across 10 IoU thresholds.
- **Recall**: Proportion of true pedestrians detected at confidence threshold $0.25$.
- **False Positive Rate (FPR)**: Rate of spurious background bounding boxes generated under noise.

### 3.2 Robustness Indices
To provide single-number summary statistics for scientific comparison, we define:
1. **Relative Robustness Ratio ($R_{\text{rel}}$)**:
   $$R_{\text{rel}}(M, \text{degradation}) = \frac{\sum_{k=1}^5 \text{mAP50}(M, L_k)}{5 \cdot \text{mAP50}_{\text{clean}}(M)}$$
   An ideal robust model maintains $R_{\text{rel}} \approx 1.0$; an fragile model displays $R_{\text{rel}} \to 0.0$.
2. **Degradation Sensitivity Slope ($\beta_{\text{deg}}$)**:
   Linear regression slope of mAP50 degradation as a function of corruption severity level.

### 3.3 Dynamic Gate Adaptation Metrics (for E007 / E008)
- **Mean Gate Activation ($\mu_{G_V}(L_k)$)**: Average spatial gate value across all validation frames.
- **Modality Shift Index ($\Delta G_V$)**:
  $$\Delta G_V(L_k) = \mu_{G_V}(L_k) - \mu_{G_V}(\text{clean})$$
  Measures whether the gate objectively suppresses visible features when visible corruptions are injected ($\Delta G_V < 0$), or whether the gate fails to adapt.

---

## 4. Execution Plan and Safety Safeguards

1. **No Checkpoint Retraining**: This protocol operates purely as an evaluation harness on existing checkpoints (`best.pt`).
2. **Strict Test Partition Isolation**: The official test set (3,463 pairs) remains untouched and locked.
3. **Reproducibility**: Corruptions are generated using fixed pseudo-random seeds (`seed=1337`) so that every model receives bit-identical perturbed inputs.
4. **Execution Status**: **PROPOSED** — Evaluation harness will be executed upon formal authorization.
