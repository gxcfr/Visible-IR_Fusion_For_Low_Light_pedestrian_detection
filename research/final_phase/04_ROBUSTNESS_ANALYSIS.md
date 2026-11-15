# Stage 4: Partial Robustness Stress-Testing Analysis
## Controlled Synthetic Sensor Perturbation Study across 25 Completed Conditions

---

> [!IMPORTANT]
> **SCOPE & INTEGRITY DISCLOSURE — PARTIAL ROBUSTNESS STUDY ONLY**  
> In accordance with the explicit research directive to halt experimental execution, this evaluation covers **25 of 40 planned conditions** (62.5% execution coverage).  
> - **Completed**: Five complete unimodal corruption suites across all five severity levels ($L_1 \to L_5$):
>   1. `rgb_brightness` (Visible luminance reduction, $L_1 \to L_5$)
>   2. `rgb_noise` (Additive visible Gaussian noise, $L_1 \to L_5$)
>   3. `rgb_blur` (Optical defocus blur, $L_1 \to L_5$)
>   4. `thermal_contrast` (Infrared contrast compression / thermal washout, $L_1 \to L_5$)
>   5. `thermal_striping` (Fixed-pattern horizontal sensor column striping, $L_1 \to L_5$)
> - **Incomplete**: `thermal_dead_pixels` (stopped gracefully prior to condition finalization; 0 of 5 levels finalized).
> - **Deferred**: `simul_darkness_washout` (simultaneous optical darkness + thermal washout, 5 levels deferred) and `simul_noise_striping` (simultaneous optical noise + thermal striping, 5 levels deferred).
> 
> **Scientific Caveat**: This report represents a **selected-corruption / partial robustness study**, NOT a comprehensive 40-condition multimodal evaluation. Claims regarding simultaneous cross-modality failure resilience or bad-pixel tolerance are explicitly withheld. Quantitative comparisons are strictly restricted to the 25 verified completed conditions.

---

## 1. Executive Summary & Experimental Methodology

Synthetic robustness testing isolates how detection architectures behave when primary sensory cues degrade due to environmental factors, optical defects, or sensor calibration loss. All evaluations were executed on the **1,803 validation image pairs** (5,086 ground truth pedestrians) using native $1280 \times 1024$ spatial coordinate matching.

### 1.1 Evaluated Architecture Matrix (Frozen Checkpoints)
- **E002 (RGB Baseline)**: Single-stream visible YOLO26n (Clean $mAP_{50} = 73.04\%$).
- **E003 (Thermal Baseline)**: Single-stream thermal YOLO26n (Clean $mAP_{50} = 94.12\%$).
- **E004 (Late Fusion WBF)**: Dual-stream decision fusion via Weighted Boxes Fusion ($w_V=1.0, w_T=2.0$, Clean $mAP_{50} = 93.74\%$).
- **E005 (Early Fusion 4CH)**: Single-stream 4-channel early concatenated input (Clean $mAP_{50} = 72.24\%$).
- **E006 (Feature Concat)**: Dual-stream deep feature concatenation at $P_3, P_4, P_5$ (Clean $mAP_{50} = 93.24\%$).
- **E007 (Proposed Asymmetric Gated Fusion)**: Dual-stream dynamic gating feature fusion with thermal anchor (Clean $mAP_{50} = 93.70\%$).

### 1.2 Evaluation Metric
For each model $m$ under degradation suite $s$ and severity level $L_k$, the **performance retention ratio** is defined as:
$$R(m, s, L_k) = \frac{mAP_{50}(m, s, L_k)}{mAP_{50}^{\text{clean}}(m)} \times 100\%$$
The mean retention across the five severity levels of a suite is denoted $R_{\text{mean}}(m, s)$, and the overall 5-suite mean robustness index is $R_{\text{mean}}^{5\text{-suites}}(m)$.

---

## 2. Consolidated Robustness Retention Summary

Table 1 reports clean baseline performance and mean retention ratios ($R_{\text{mean}}$) across the five completed corruption suites.

### Table 1: Performance Retention across 5 Completed Degradation Suites ($R_{\text{mean}} \%$)

| Model ID | Clean $mAP_{50}$ | Visible Brightness ($R_{\text{mean}}$) | Visible Noise ($R_{\text{mean}}$) | Visible Blur ($R_{\text{mean}}$) | Thermal Contrast ($R_{\text{mean}}$) | Thermal Striping ($R_{\text{mean}}$) | **5-Suite Mean ($R_{\text{mean}}^{5\text{-suites}}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E002 (RGB)** | 73.04% | 86.74% | 47.46% | 75.98% | 100.00%* | 100.01%* | 82.04% |
| **E003 (Thermal)** | **94.12%** | 100.00%* | 100.00%* | 100.00%* | 74.00% | 38.10% | 82.42% |
| **E004 (Late WBF)** | 93.74% | **99.92%** | **99.70%** | 98.88% | **92.02%** | 80.30% | **94.16%** |
| **E005 (Early 4CH)** | 72.24% | 101.90% | 99.24% | 96.12% | 51.84% | 51.21% | 80.06% |
| **E006 (Concat)** | 93.24% | 99.20% | 99.30% | 99.14% | 75.46% | 73.99% | 89.42% |
| **E007 (Asym Gated)** | 93.70% | 98.52% | 98.72% | **99.18%** | 73.66% | **83.47%** | 90.71% |

*\*Note: For unimodal models, performance under orthogonal modality corruption is identically preserved by definition ($100.0\%$), as the sensor input to that model remains pristine.*

---

## 3. Detailed Suite-by-Suite Analysis

```
                                PER-SUITE RETENTION CURVES (L1 to L5)
   100% +-------------------------------------------------------------------------------+
        |  Visible Noise (E002: 98% -> 1%)              Thermal Striping (E003: 84% -> 2%) |
        |  Multimodal: E004, E006, E007 stay >90%      E007 (55%) vs E006 (33%) vs E003 (2%)|
    50% |                                                                               |
        |                                                                               |
     0% +-------------------------------------------------------------------------------+
        L1              L2              L3              L4              L5 (Severe)
```

### 3.1 Suite 1: Visible Luminance Attenuation (`rgb_brightness`)
- **Perturbation**: Multiplicative scaling of visible channel intensities by factors $\alpha \in [0.80, 0.60, 0.40, 0.20, 0.05]$.
- **Observations**:
  - E002 (RGB) exhibits graceful degradation through $L_4$ ($\alpha=0.20$, retention $93.5\%$, $mAP_{50}=68.31\%$), before suffering catastrophic failure at $L_5$ ($\alpha=0.05$, retention $43.0\%$, $mAP_{50}=31.38\%$). Recall drops from $65.9\%$ to $19.6\%$.
  - All fusion models (E004, E006, E007) maintain $>98.5\%$ mean retention because the thermal anchor remains completely intact. E007 maintains $88.81\% mAP_{50}$ at $L_5$ ($94.8\%$ retention).

### 3.2 Suite 2: Additive Visible Gaussian Noise (`rgb_noise`)
- **Perturbation**: Zero-mean additive Gaussian noise with standard deviations $\sigma \in [0.02, 0.05, 0.10, 0.18, 0.28]$.
- **Observations**:
  - Unimodal visible detection E002 is acutely sensitive to high-frequency sensor noise:
    - At $L_2$ ($\sigma=0.05$): retention drops to $86.0\%$ ($mAP_{50}=62.85\%$).
    - At $L_3$ ($\sigma=0.10$): retention collapses to $40.1\%$ ($mAP_{50}=29.32\%$).
    - At $L_5$ ($\sigma=0.28$): total functional collapse ($mAP_{50}=1.02\%$, recall $1.08\%$).
  - **Fusion Resilience**: Deep fusion models (E006, E007) and decision fusion (E004) completely shield detection against visible sensor noise. E007 retains $90.84\% mAP_{50}$ at $L_5$ ($96.9\%$ retention). The dynamic gate progressively down-weights visible feature maps ($\Delta G_V = -0.1019$).

### 3.3 Suite 3: Visible Optical Defocus Blur (`rgb_blur`)
- **Perturbation**: Gaussian defocus kernels with sizes $k \in \{3, 7, 11, 17, 25\}$ and corresponding $\sigma \in [1.0, 8.0]$.
- **Observations**:
  - E002 degrades monotonically from $74.82\% mAP_{50}$ ($L_1$) down to $23.10\% mAP_{50}$ ($L_5$, $31.6\%$ retention).
  - Multimodal architectures exhibit near-perfect immunity, with E007 achieving $99.18\%$ mean retention ($92.39\% mAP_{50}$ at $L_5$).

### 3.4 Suite 4: Thermal Contrast Compression / Washout (`thermal_contrast`)
- **Perturbation**: Attenuation of thermal pixel deviations around scene mean by factors $\gamma \in [0.75, 0.50, 0.30, 0.15, 0.02]$.
- **Observations**:
  - E003 (Thermal) degrades as contrast compresses:
    - $L_1$ ($\gamma=0.75$): $94.11\% mAP_{50}$ ($100.0\%$ retention).
    - $L_3$ ($\gamma=0.30$): $88.69\% mAP_{50}$ ($94.2\%$ retention).
    - $L_4$ ($\gamma=0.15$): $73.46\% mAP_{50}$ ($78.0\%$ retention).
    - $L_5$ ($\gamma=0.02$): $0.00\% mAP_{50}$ (total thermal signal extinction).
  - **Decoupling vs. Co-adaptation in Fusion**:
    - **E004 (Late Fusion WBF)** demonstrates superior failure-mode independence: when thermal collapses to $0.00\%$ at $L_5$, E004 preserves **$72.60\% mAP_{50}$** ($77.4\%$ retention), gracefully falling back to the intact RGB stream ($73.04\%$).
    - **E006 (Concat)** and **E007 (Asym Gated)** drop to $13.70\%$ and $3.60\% mAP_{50}$ respectively at $L_5$. Because both deep architectures were trained with thermal as the dominant anchor backbone, complete thermal signal extinction starves the neck convolutions, proving that deep fusion models cannot effortlessly isolate an entirely extinguished primary backbone without explicit dropout regularization during training.

### 3.5 Suite 5: Thermal Fixed-Pattern Column Striping Noise (`thermal_striping`)
- **Perturbation**: Uncooled microbolometer column-amplifier read-out noise modeled as stationary vertical stripes with $\sigma_{\text{stripe}} \in [5, 15, 30, 50, 80]$.
- **Observations**:
  - E003 (Thermal) suffers severe degradation: retention falls to $55.0\%$ at $L_2$, $34.3\%$ at $L_3$, $15.2\%$ at $L_4$, and **$2.1\% mAP_{50}$** ($1.98\%$) at $L_5$. Spatial striping disrupts horizontal edge gradients essential for pedestrian shape recognition.
  - **Architectural Advantage of Dynamic Gating (E007 vs. E006)**:
    - At severity $L_4$ ($\sigma=50$): E007 retains **$71.70\% mAP_{50}$** ($76.5\%$ retention), whereas unweighted feature concatenation E006 achieves only **$55.33\% mAP_{50}$** ($59.3\%$ retention).
    - At severity $L_5$ ($\sigma=80$): E007 retains **$51.17\% mAP_{50}$** ($54.6\%$ retention), outperforming E006 ($30.88\%$) by **$+20.29\%$ mAP50** and outperforming unimodal thermal E003 ($1.98\%$) by **$+49.19\%$ mAP50**.
    - E007's dynamic gating mechanism suppresses corrupted spatial features before feeding the detection neck, demonstrating significant empirical resilience against structured sensor artifact noise.

---

## 4. Gate Adaptation Dynamics (E007)

Table 2 details the empirical behavior of E007's gate tensor $P_3$ under extreme ($L_5$) degradation relative to clean baseline activation ($\bar{G}_V^{\text{clean}} = 0.5121$).

### Table 2: E007 P3 Dynamic Gate Response under L5 Severity

| Perturbation Suite | Clean $\bar{G}_{V, P3}$ | Degraded $\bar{G}_{V, P3}$ ($L_5$) | Gate Shift ($\Delta G_V$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Visible Brightness** | 0.5121 | 0.4612 | $-0.0509$ | Moderate down-weighting as visible contrast diminishes |
| **Visible Noise** | 0.5121 | 0.4102 | $\mathbf{-0.1019}$ | Strongest down-weighting; suppresses spurious high-frequency noise |
| **Visible Blur** | 0.5121 | 0.4721 | $-0.0400$ | Moderate attenuation of low-frequency defocused maps |
| **Thermal Contrast** | 0.5121 | 0.5480 | $\mathbf{+0.0359}$ | Compensatory increase in visible reliance as thermal contrast fades |
| **Thermal Striping** | 0.5121 | 0.4952 | $-0.0169$ | Slight gate contraction due to disrupted joint spatial correlation |

The gate exhibits measurable adaptive responsiveness: it actively suppresses corrupted visible inputs when noise is injected ($\Delta G_V = -0.1019$), and increases visible weighting when thermal contrast is washed out ($\Delta G_V = +0.0359$). However, consistent with the Stage 2 audit findings, the gate remains bounded between $\approx 0.41$ and $0.55$, acting as a soft feature attentional re-weighting layer rather than a binary sensor cut-off switch.

---

## 5. Formal Manifest of Condition Coverage

### 5.1 Completed Conditions Manifest (25 / 40)
- **Visible Brightness ($L_1 \to L_5$)**: 5 conditions completed across all 6 models ($30$ evaluations).
- **Visible Gaussian Noise ($L_1 \to L_5$)**: 5 conditions completed across all 6 models ($30$ evaluations).
- **Visible Optical Blur ($L_1 \to L_5$)**: 5 conditions completed across all 6 models ($30$ evaluations).
- **Thermal Contrast Washout ($L_1 \to L_5$)**: 5 conditions completed across all 6 models ($30$ evaluations).
- **Thermal Column Striping ($L_1 \to L_5$)**: 5 conditions completed across all 6 models ($30$ evaluations).
- **Total Valid Completed Evaluations**: **150 model-condition pairs**.

### 5.2 Partial & Deferred Conditions Manifest (15 / 40)
- **Thermal Dead Pixels ($L_1 \to L_5$)**: Incomplete (execution halted by user directive prior to condition finalization; 0 of 5 finalized).
- **Simultaneous Darkness + Contrast Washout ($L_1 \to L_5$)**: Deferred / Unexecuted (5 conditions).
- **Simultaneous Noise + Column Striping ($L_1 \to L_5$)**: Deferred / Unexecuted (5 conditions).

---

## 6. Generated & Preserved Artifacts

All artifacts have been flushed, serialized, and mirrored into both `research/robustness/` and `LLVIP_MULTIMODAL_RESEARCH/research/robustness/`:
1. `robustness_metrics.csv` & `robustness_metrics.json`: Complete record of all 150 evaluations (clean and degraded mAP50, mAP50-95, recall, precision, and FP/img).
2. `robustness_retention_summary.csv`: Master retention ratio comparison table across the five completed suites.
3. `gate_response/e007_gate_dynamics.csv`: Quantitative dynamic gate response log for E007.
4. `completed_conditions_manifest.json`: Machine-readable metadata of the 25 completed stress conditions.
5. `partial_conditions_manifest.json`: Machine-readable record documenting the incomplete and deferred conditions.
6. `robustness_comparison_plots/retention_by_suite.png`: 5-panel comparative visualization illustrating degradation slopes from $L_1$ to $L_5$.
7. `corruption_visual_gallery/`: Sample visual inspections depicting corruption appearances at $L_1, L_3, L_5$.
