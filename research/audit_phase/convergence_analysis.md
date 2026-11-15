# Training Convergence and Checkpoint Selection Audit

## 1. Overview and Objectives

This audit examines the training trajectories, loss convergence, validation fitness metrics, and checkpoint selection policies for experiments **E005** (Early Fusion 4CH), **E006** (Feature Concatenation), and **E007** (Proposed Asymmetric Gated Fusion).

Specifically, the external peer review requested verification of:
1. Validation fitness and mAP trajectories across epochs.
2. The divergence between continued training loss decrease and early validation fitness peaking.
3. Checkpoint selection criteria (`best.pt` vs `last.pt` vs single-epoch peaks).
4. Early stopping behavior and risk of overfitting or validation instability.

---

## 2. Checkpoint Selection Policy and Early Stopping

All training runs were configured under Ultralytics-standard early stopping criteria:
- **Fitness Definition**: $\text{Fitness} = 0.1 \times \text{mAP}_{50} + 0.9 \times \text{mAP}_{50-95}$
- **Checkpoint Rule**: The checkpoint `best.pt` is updated strictly when the current epoch achieves a higher validation fitness than all preceding epochs.
- **Early Stopping Patience**: 12 epochs without a new validation fitness peak triggers training termination.
- **Reporting Integrity Rule**: In accordance with rigorous academic standards, all reported evaluation metrics for an experiment must correspond strictly to a single, verified checkpoint (`best.pt`). Metrics must **never** be cherry-picked across disjoint epochs (e.g., combining peak recall from epoch 12 with peak precision from epoch 5).

---

## 3. Detailed Trajectory Analysis

### 3.1 Experiment E007: Proposed Asymmetric Gated Fusion (YOLO26n)

- **Total Epochs Trained**: 17
- **Termination Reason**: Early stopping triggered at Epoch 17 ($5 + 12 = 17$ epochs without fitness improvement).
- **Selected Checkpoint**: `best.pt` from **Epoch 5**.

#### Epoch-by-Epoch Convergence Table:

| Epoch | Train Loss | Val Precision | Val Recall | Val F1 | Val mAP50 | Val mAP50-95 | Fitness | Status / Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 9.1023 | 0.9184 | 0.8482 | 0.8819 | 0.9233 | 0.5507 | 0.5880 | Initial epoch |
| 2 | 8.1673 | 0.9322 | 0.8617 | 0.8956 | 0.9350 | 0.5656 | 0.6025 | Improvement |
| 3 | 7.8734 | 0.9248 | 0.8676 | 0.8953 | 0.9337 | 0.5634 | 0.6004 | Plateau |
| 4 | 7.6970 | 0.9423 | 0.8612 | 0.8999 | 0.9329 | 0.5630 | 0.6000 | Precision spike |
| **5** | **7.5155** | **0.9389** | **0.8769** | **0.9069** | **0.9411** | **0.5658** | **0.6033** | **BEST FITNESS CHECKPOINT (`best.pt`)** |
| 6 | 7.3641 | 0.9311 | 0.8740 | 0.9016 | 0.9370 | 0.5640 | 0.6013 | - |
| 7 | 7.2469 | 0.9443 | 0.8742 | 0.9079 | 0.9382 | 0.5644 | 0.6018 | - |
| 8 | 7.1343 | 0.9380 | 0.8755 | 0.9057 | 0.9389 | 0.5645 | 0.6019 | - |
| 9 | 7.0279 | 0.9342 | 0.8683 | 0.9000 | 0.9357 | 0.5492 | 0.5879 | Moderate drop |
| 10 | 6.9303 | 0.9322 | 0.8700 | 0.9000 | 0.9359 | 0.5546 | 0.5927 | - |
| 11 | 6.8520 | 0.9411 | 0.8703 | 0.9043 | 0.9405 | 0.5568 | 0.5952 | - |
| 12 | 6.7783 | 0.9359 | 0.8820 | 0.9082 | 0.9418 | 0.5559 | 0.5945 | Peak recall (0.8820), peak mAP50 (0.9418) |
| 13 | 6.6744 | 0.9382 | 0.8752 | 0.9056 | 0.9349 | 0.5523 | 0.5906 | - |
| 14 | 6.6040 | 0.9385 | 0.8701 | 0.9030 | 0.9360 | 0.5505 | 0.5891 | - |
| 15 | 6.5029 | 0.9355 | 0.8729 | 0.9031 | 0.9343 | 0.5480 | 0.5866 | - |
| 16 | 6.4323 | 0.9367 | 0.8663 | 0.9001 | 0.9364 | 0.5516 | 0.5901 | - |
| 17 | 6.3739 | 0.9393 | 0.8703 | 0.9035 | 0.9337 | 0.5477 | 0.5863 | Early stopping trigger (12 patience) |

#### Analysis:
1. **Loss vs. Fitness Divergence**:
   - Training loss steadily reduced from 9.1023 down to 6.3739 (-30.0%).
   - Validation fitness peaked early at Epoch 5 (0.6033), driven by a strong $\text{mAP}_{50-95}$ of 0.5658 and balanced precision (0.9389) and recall (0.8769).
   - After Epoch 8, despite continuing reduction in training loss, $\text{mAP}_{50-95}$ exhibited a gradual degradation (from 0.5645 down to 0.5477). This degradation indicates mild over-fitting to the training distribution, as the network sharpened predictions on training features at the expense of generalized localization precision on distant and occluded pedestrians.
2. **Peak Discrepancy Clarification**:
   - While Epoch 12 registered slightly higher $\text{mAP}_{50}$ (0.9418 vs 0.9411) and higher recall (0.8820 vs 0.8769), its $\text{mAP}_{50-95}$ (0.5559) was noticeably lower than Epoch 5 (0.5658). Under the standard fitness formula ($0.1 \times \text{mAP}_{50} + 0.9 \times \text{mAP}_{50-95}$), Epoch 5 remained the global maximum (0.6033 vs 0.5945).
   - In the audit, all validation results for E007 are derived strictly from the official `best.pt` saved at Epoch 5.

---

### 3.2 Experiment E006: Feature Concatenation (YOLO26n)

- **Total Epochs Trained**: 17
- **Termination Reason**: Early stopping triggered at Epoch 17 ($5 + 12 = 17$).
- **Selected Checkpoint**: `best.pt` from **Epoch 5**.

#### Epoch-by-Epoch Convergence Table:

| Epoch | Train Loss | Val Precision | Val Recall | Val F1 | Val mAP50 | Val mAP50-95 | Fitness | Status / Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 8.9162 | 0.9333 | 0.8628 | 0.8966 | 0.9359 | 0.5580 | 0.5958 | Initial epoch |
| 2 | 7.9461 | 0.9301 | 0.8609 | 0.8942 | 0.9345 | 0.5630 | 0.6001 | Improvement |
| 3 | 7.7573 | 0.9353 | 0.8645 | 0.8985 | 0.9344 | 0.5606 | 0.5980 | - |
| 4 | 7.5695 | 0.9286 | 0.8714 | 0.8991 | 0.9344 | 0.5641 | 0.6011 | - |
| **5** | **7.4507** | **0.9348** | **0.8632** | **0.8976** | **0.9358** | **0.5662** | **0.6032** | **BEST FITNESS CHECKPOINT (`best.pt`)** |
| 6 | 7.3314 | 0.9359 | 0.8685 | 0.9009 | 0.9378 | 0.5624 | 0.5999 | - |
| 7 | 7.2473 | 0.9379 | 0.8606 | 0.8976 | 0.9298 | 0.5577 | 0.5949 | - |
| 8 | 7.1732 | 0.9350 | 0.8774 | 0.9053 | 0.9424 | 0.5644 | 0.6022 | Peak mAP50 (0.9424), fitness 0.6022 |
| 9 | 7.0966 | 0.9356 | 0.8630 | 0.8978 | 0.9292 | 0.5563 | 0.5936 | - |
| 10 | 7.0387 | 0.9427 | 0.8671 | 0.9033 | 0.9378 | 0.5632 | 0.6007 | - |
| 11 | 6.9510 | 0.9399 | 0.8655 | 0.9012 | 0.9346 | 0.5584 | 0.5960 | - |
| 12 | 6.8988 | 0.9443 | 0.8664 | 0.9037 | 0.9353 | 0.5572 | 0.5950 | Peak precision (0.9443) |
| 13 | 6.8649 | 0.9336 | 0.8742 | 0.9029 | 0.9361 | 0.5618 | 0.5992 | - |
| 14 | 6.8014 | 0.9331 | 0.8669 | 0.8988 | 0.9342 | 0.5506 | 0.5890 | - |
| 15 | 6.7574 | 0.9407 | 0.8677 | 0.9027 | 0.9337 | 0.5541 | 0.5921 | - |
| 16 | 6.6957 | 0.9318 | 0.8649 | 0.8971 | 0.9344 | 0.5523 | 0.5905 | - |
| 17 | 6.6715 | 0.9368 | 0.8687 | 0.9014 | 0.9333 | 0.5537 | 0.5917 | Early stopping trigger (12 patience) |

#### Analysis:
- E006 exhibited very similar behavior to E007: training loss dropped monotonically from 8.9162 to 6.6715, while $\text{mAP}_{50-95}$ peaked at Epoch 5 (0.5662).
- Epoch 8 attained a slightly higher $\text{mAP}_{50}$ of 0.9424, but its fitness of 0.6022 fell just short of Epoch 5's 0.6032.
- Checkpoint `best.pt` corresponds strictly to Epoch 5.

---

### 3.3 Experiment E005: Early Fusion 4CH (YOLO26n)

- **Total Epochs Trained**: 39
- **Selected Checkpoint**: `best.pt` from **Epoch 21** (Fitness: 0.4081, $\text{mAP}_{50}$: 0.7121, $\text{mAP}_{50-95}$: 0.3743).
- **Convergence Behavior**:
  - Training loss started high at 15.4744 and dropped slowly to 12.6514 at Epoch 39.
  - Validation metrics remained severely depressed compared to two-stream architectures ($\text{mAP}_{50}$ plateaued around 70–71%).
  - Early-stage 4-channel convolution directly blends raw visible sensor noise into thermal features at the very first layer, preventing the network from separating modality-specific signals. This resulted in an enormous number of low-confidence false alarms (over 231,000 total candidate boxes across 1,803 validation images).

---

## 4. Key Checkpoint Identification and Integrity Summary

| Experiment | Best Fitness Epoch | Best Recall Epoch | Best mAP50 Epoch | Best mAP50-95 Epoch | Evaluated Checkpoint | Selection Criterion |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **E005** (Early Fusion) | 21 | 17 (0.6553) | 16 (0.7153) | 21 (0.3743) | `best.pt` (Epoch 21) | Maximum Validation Fitness ($0.1\times\text{mAP}_{50} + 0.9\times\text{mAP}_{50-95}$) |
| **E006** (Feature Concat) | 5 | 8 (0.8774) | 8 (0.9424) | 5 (0.5662) | `best.pt` (Epoch 5) | Maximum Validation Fitness ($0.1\times\text{mAP}_{50} + 0.9\times\text{mAP}_{50-95}$) |
| **E007** (Asym Gated) | 5 | 12 (0.8820) | 12 (0.9418) | 5 (0.5658) | `best.pt` (Epoch 5) | Maximum Validation Fitness ($0.1\times\text{mAP}_{50} + 0.9\times\text{mAP}_{50-95}$) |

---

## 5. Audit Conclusions

1. **Validity of Peak Selection**: The selection of Epoch 5 for both E006 and E007 is fully compliant with standard validation fitness tracking. It correctly optimizes bounding box localization quality ($\text{mAP}_{50-95}$) rather than raw classification recall.
2. **Absence of Cherry-Picking**: Previous summary tables that listed peak recall figures (e.g., recall of ~88.2% from Epoch 12) alongside $\text{mAP}_{50-95}$ from Epoch 5 have been strictly superseded. The harmonized audit tables report all metrics computed exclusively from `best.pt`.
3. **Overfitting Assessment**: The decline in $\text{mAP}_{50-95}$ after Epoch 8, while training loss fell, demonstrates that extended training without strong regularization or modality dropout causes multi-stream architectures to over-fit subtle visible background textures in nighttime scenes. Future training runs should incorporate higher weight decay or modality dropout to mitigate this tendency.
