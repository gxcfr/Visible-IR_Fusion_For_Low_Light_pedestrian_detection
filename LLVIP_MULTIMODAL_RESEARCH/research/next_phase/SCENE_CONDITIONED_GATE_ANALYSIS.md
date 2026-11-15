# Scene-Conditioned Gate Correlation and Cluster Uncertainty Analysis

## 1. Executive Summary & Problem Formulation

In preliminary Phase 2 audit reports, gate correlation metrics were reported as pooled statistics across all 1,803 validation image pairs. However, the LLVIP validation split is not a collection of independent random photographs; it consists of sequential surveillance video recordings captured from exactly **four stationary camera locations**:
- **Scene 02**: Standard urban street (500 frames)
- **Scene 05**: Dark urban alley with distant background lighting (428 frames)
- **Scene 07**: Pavement pedestrian crossing with high clutter (452 frames)
- **Scene 25**: Street with vehicle artificial headlights and glare (423 frames)

Because frames within each scene share identical background geometry, sensor gain parameters, and ambient illumination, treating all 1,803 frames as statistically independent violates the fundamental i.i.d. assumption. Naive parametric $p$-values ($p < 10^{-40}$) are heavily anticonservative. 

Furthermore, pooling across scenes with different baseline illumination introduces **Simpson's Paradox**, where between-scene baseline shifts distort or completely invert the true intra-scene relationship.

---

## 2. Quantitative Per-Scene vs. Pooled Correlation Breakdown

The table below reports Pearson ($r$) and Spearman ($\rho$) correlations evaluated separately within each scene ($N = 423$ to $500$ frames), compared against the naive pooled correlation ($N = 1,803$) and the **Within-Scene Centered Correlation** (fixed-effects model subtracting each scene's mean):

| Modality Conditioning Attribute | Metric | Scene 02 ($N=500$) | Scene 05 ($N=428$) | Scene 07 ($N=452$) | Scene 25 ($N=423$) | Pooled Across All Scenes ($N=1,803$) | Within-Scene Centered ($N=1,803$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Visible Brightness** (Mean Pixel Gray) | **Scene Mean ($\bar{X}$)** | 55.84 | 37.38 | 62.11 | 87.49 | 59.88 | Centered ($\bar{X}=0$) |
| | **Gate Mean ($\bar{G}_V$)** | 0.4937 | 0.5482 | 0.5104 | 0.4994 | 0.5121 | Centered ($\bar{G}=0$) |
| | **Pearson $r$** | **$+0.3064$** | **$-0.2654$** | **$+0.4921$** | $+0.0898$ | **$-0.3216$** | **$+0.1162$** |
| | **95% Confidence Interval** | $[+0.225, +0.384]$ | $[-0.351, -0.175]$ | $[+0.419, +0.559]$ | $[-0.006, +0.184]$ | $[-0.362, -0.280]$ | (Within cluster) |
| | **Spearman $\rho$** | $+0.3161$ | $-0.2091$ | $+0.5802$ | $+0.1509$ | $-0.2302$ | $+0.2007$ |
| **Visible Contrast** (Pixel Std Dev) | **Scene Mean ($\bar{X}$)** | 44.77 | 21.18 | 46.45 | 56.56 | 42.36 | Centered ($\bar{X}=0$) |
| | **Pearson $r$** | **$+0.1920$** | **$-0.5266$** | **$-0.3243$** | $-0.0170$ | **$-0.5368$** | **$-0.0607$** |
| | **95% Confidence Interval** | $[+0.106, +0.275]$ | $[-0.592, -0.455]$ | $[-0.405, -0.239]$ | $[-0.112, +0.078]$ | $[-0.569, -0.503]$ | (Within cluster) |
| | **Spearman $\rho$** | $+0.2010$ | $-0.5482$ | $-0.4001$ | $-0.0901$ | $-0.4792$ | $-0.0773$ |
| **Thermal Contrast** (Pixel Std Dev) | **Scene Mean ($\bar{X}$)** | 43.03 | 51.97 | 46.18 | 49.17 | 47.38 | Centered ($\bar{X}=0$) |
| | **Pearson $r$** | **$-0.5474$** | **$-0.5361$** | **$+0.3611$** | $+0.1371$ | **$+0.4719$** | **$-0.1055$** |
| | **95% Confidence Interval** | $[-0.606, -0.483]$ | $[-0.600, -0.465]$ | $[+0.278, +0.439]$ | $[+0.042, +0.229]$ | $[+0.435, +0.507]$ | (Within cluster) |
| | **Spearman $\rho$** | $-0.5328$ | $-0.4537$ | $+0.3700$ | $-0.0026$ | $+0.4735$ | $-0.1688$ |

---

## 3. Methodological Deductions: Simpson's Paradox and Cross-Scene Inconsistency

### 3.1 Empirical Demonstration of Simpson's Paradox
The most striking finding is in **Visible Brightness**:
- When evaluated within individual scenes, three of the four scenes exhibit a **positive** correlation (Scene 02: $r = +0.306$, Scene 07: $r = +0.492$, Scene 25: $r = +0.090$). Within those camera positions, brighter lighting conditions slightly elevate visible feature gating.
- However, when naively pooling all 1,803 frames across scenes, the correlation **inverts to a negative value** ($r = -0.3216$).
- **Cause of Inversion**: Scene 05 happens to be the darkest camera location overall (mean brightness $37.38$) but produces the highest average gate activation ($0.5482$) because its background contains prominent building outlines. Conversely, Scene 25 is the brightest (mean brightness $87.49$) but has lower average gate activation ($0.4994$). This large between-scene cluster offset dominates the naive regression line and flips the correlation sign.
- **Within-Scene Centering**: When scene-specific mean brightness and mean gate values are subtracted ($\tilde{X} = X - \bar{X}_s$, $\tilde{Y} = Y - \bar{Y}_s$), isolating genuine intra-scene fluctuations, the centered correlation is **$+0.1162$** ($p = 7.51 \times 10^{-7}$).

### 3.2 Inconsistent Cross-Scene Correlation Directions
The gate responses do not follow a uniform, universal physical rule:
1. **Visible Contrast**: In Scene 02, visible contrast correlates positively with gating ($r = +0.192$), whereas in Scene 05 it correlates strongly negatively ($r = -0.527$). In Scene 25, there is virtually zero correlation ($r = -0.017$).
2. **Thermal Contrast**: In Scenes 02 and 05, higher thermal contrast correlates strongly negatively with visible gating ($r \approx -0.54$), meaning the model suppresses visible feature injection when thermal contrast is clean. But in Scene 07, thermal contrast correlates positively ($r = +0.361$).
3. **Scientific Conclusion**: These reversals prove conclusively that **the gate does not act as a calibrated physical sensor reliability estimator**. The gate convolutions respond to complex scene-specific visual textures (e.g. wall shadows, asphalt grain, vehicle reflections) rather than universal radiometric properties.

---

## 4. Block Bootstrap Uncertainty Analysis (Accounting for Frame Autocorrelation)

To rigorously quantify uncertainty without falsely assuming that consecutive video frames are independent, a non-parametric **Moving Block Bootstrap** was implemented:
- **Block Size**: 50 contiguous temporal frames (capturing local autocorrelation).
- **Total Blocks**: 38 blocks across the four scenes.
- **Bootstrap Replications**: 1,000 iterations resampling blocks with replacement.

### Standard Error and Confidence Interval Comparison

| Environmental Association | Naive i.i.d. Standard Error | Block Bootstrap Standard Error | Error Inflation Factor | Naive 95% CI ($N=1,803$) | Block Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|
| **P3 vs. Visible Brightness** | $0.0236$ | **$0.0746$** | **$3.16\times$** | $[-0.3624, -0.2796]$ | **$[-0.4552, -0.1598]$** |
| **P3 vs. Visible Contrast** | $0.0236$ | **$0.0758$** | **$3.21\times$** | $[-0.5689, -0.5031]$ | **$[-0.6582, -0.3610]$** |
| **P3 vs. Thermal Contrast** | $0.0236$ | **$0.0611$** | **$2.59\times$** | $[+0.4352, +0.5070]$ | **$[+0.3409, +0.5756]$** |

### Statistical Takeaways:
1. **Uncertainty is Over Three Times Larger**: Accounting for frame sequence continuity inflates the true standard error by a factor of $2.6\times$ to $3.2\times$.
2. **Rejection of Spurious Significance**: Ordinary i.i.d. $p$-values ($p < 10^{-40}$) severely exaggerate the certainty of the observations. While the pooled trends remain statistically bounded away from zero, they are dominated by scene-level sampling rather than independent frame evidence.
