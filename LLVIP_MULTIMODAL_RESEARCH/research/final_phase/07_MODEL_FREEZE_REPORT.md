# Stage 7: Model Selection Freeze & Pre-Test Integrity Protocol
## Formal Checkpoint Freezing, SHA256 Verification, and Sealed Test Protocol

---

> [!CAUTION]
> **MANDATORY EXECUTION HALT — RESEARCHER APPROVAL REQUIRED**  
> In accordance with academic research protocol, **all execution is strictly halted at Stage 7**.  
> The official LLVIP test partition (**3,463 image pairs / 8,302 ground truth pedestrians**) remains **100% sealed, unopened, and untouched**.  
> **Under no circumstances will Stage 8 (Official Test Evaluation) proceed without explicit, affirmative approval from the researcher.**

---

## 1. Executive Summary & Model Selection Rationales

Across Stages 1 through 6, candidate models were comprehensively audited, harmonized, ablated, stress-tested, and computationally profiled strictly on the **1,803 validation pairs**. All model architectures and weights are now permanently frozen.

### 1.1 Candidate Model Selection Rationales

| Experiment ID | Model Name | Role in Final Benchmark | Selection & Benchmarking Rationale |
| :--- | :--- | :--- | :--- |
| **E007** | **Proposed Asymmetric Gated Fusion** | **Primary Proposed Architecture** | Combines thermal backbone anchoring with dynamic multi-scale visible feature gating. Achieves $93.70\% mAP_{50}$ on validation, highest Scene 25 recall ($94.20\%$), fewest large-object misses ($308$ FNs), and superior sensor striping retention ($83.5\%$ vs. E006's $74.0\%$) at $55.7$ FPS. |
| **E003** | **YOLO26n Thermal** | **Primary Unimodal Competitor** | The strongest unimodal baseline on LLVIP ($94.12\% mAP_{50}$, $90.50\%$ Recall). Represents the primary benchmark hurdle that any multimodal fusion architecture must justify against. |
| **E006** | **YOLO26n Feature Concat** | **Primary Deep Fusion Competitor** | Unweighted multi-scale channel concatenation ($93.24\% mAP_{50}$, $56.09\% mAP_{50-95}$). Serves as the direct ablation counterpart to evaluate whether dynamic gating ($G_V$) confers measurable advantage over simple concatenation. |
| **E004** | **YOLO26n Late Fusion (WBF)** | **Decision-Level Fusion Competitor** | Geometric box-level ensembling of E002 and E003 ($93.74\% mAP_{50}$). Exhibits the highest precision ($96.27\%$) and isolates unimodal sensor failure (retains $72.60\%$ when thermal is completely washed out). |
| **E005** | **YOLO26n Early Fusion (4CH)** | **Pixel-Level Fusion Competitor** | 4-channel input stem ($72.24\% mAP_{50}$). Demonstrates the critical failure modes of naive early pixel concatenation ($2,475$ validation false alarms). |
| **E002** | **YOLO26n Visible (RGB)** | **Unimodal Visible Baseline** | Clean visible-only baseline ($73.04\% mAP_{50}$). Quantifies pedestrian detectability under standard low-light street illumination. |
| **E001** | **YOLO11n Visible (RGB)** | **Historical Baseline** | Predecessor architecture baseline ($71.69\% mAP_{50}$). Retained for historical continuity across university project reporting cycles. |

---

## 2. Checkpoint Fingerprinting & SHA256 Integrity Verification

All model weights evaluated in Stage 8 are cryptographically fingerprinted below. Checkpoint files are read-only and permanently locked against modification.

### Table 1: Cryptographic Checkpoint Register

| Exp ID | Architecture Name | Checkpoint File Path | File Size | SHA256 Checksum | Parameters | Detection Head ($nc$) |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: |
| **E001** | YOLO11n RGB | `experiments/E001_RGB_YOLO11N/weights/best.pt` | 5.21 MB | `39f4393e833915b3e5ccf2dbb7584ee9983b36afa9f9b590d8d77a8aa053b7bb` | 2,590,035 | 1 |
| **E002** | YOLO26n RGB | `experiments/E002_RGB_YOLO26N/weights/best.pt` | 5.13 MB | `cd12c93e56c0bfc4d0e8f6568d2b53666e3899678e0ba3796bf6dbd91fb5a204` | 2,504,190 | 1 |
| **E003** | YOLO26n Thermal | `experiments/E003_THERMAL_YOLO26N/weights/best.pt` | 5.12 MB | `652aa0558aaeb6fbf2873f7ac1fd3d45cd9d448cef2f9ad82e4fa7c22f3987d7` | 2,504,190 | 1 |
| **E004** | YOLO26n Late WBF | Checkpoints E002 + E003 (Ensemble) | 10.25 MB | *Deterministic WBF Merge ($w=[1.0, 2.0]$, IoU=$0.55$)* | 5,008,380 | 1 + 1 |
| **E005** | YOLO26n Early 4CH | `experiments/E005_EARLY_FUSION_4CH_YOLO26N/weights/best.pt` | 10.09 MB | `874dbe5e9843843bbc15741dbdeee771a2e1a7e20e201a3112f6d517b4672ed0` | 2,572,424 | 80 |
| **E006** | YOLO26n Concat | `experiments/E006_FEATURE_FUSION_CONCAT_YOLO26N/weights/best.pt` | 16.17 MB | `238971cce85a9bce49e03a95cb6b4daef9de79eff175d3464e0bab84e6580f91` | 4,135,384 | 80 |
| **E007** | Proposed Asym Gated | `experiments/E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N/weights/best.pt` | 19.37 MB | `0e876a8fc8bb42529aeaa8c3f54409aafe46c82802958f74efd44a0dd49ac59c` | 4,971,992 | 80 |

---

## 3. Test Partition Verification & Strict Seal Statement

### 3.1 Verification of Sealed Test Partition
- **Test Partition Manifest**: 3,463 paired visible-infrared frames located in `C:\LLVIP\LLVIP`.
- **Annotation Count**: 8,302 validated ground-truth pedestrian annotations.
- **Seal Attestation**:
  - Across all prior research phases (Stages 1 through 7), **zero images from `test_rgb.txt` or `test_thermal.txt` have been loaded, inferred, or evaluated**.
  - All prompt development, hyperparameter selection, gate analyses, inference ablations, synthetic robustness tests, and computational benchmarks operated **strictly and exclusively** on the 10,222 training pairs and 1,803 validation pairs.
  - The official test partition remains pristine and completely uncorrupted by data snooping or premature evaluation.

---

## 4. Single-Pass Execution Protocol for Stage 8

Upon receiving explicit researcher approval, Stage 8 will execute under the following strict scientific constraints:

1. **Single-Pass Evaluation**:
   - Each frozen model will be evaluated exactly **once** on the 3,463 test pairs.
   - No post-hoc hyperparameter tuning, confidence threshold shifting, or iterative re-running will take place.
2. **Standardized Coordinate Matching**:
   - Identical to the unified validation protocol, predictions will be mapped back to native $1280 \times 1024$ coordinates and evaluated via Ultralytics descending-IoU matching.
   - Evaluated metrics: $mAP_{50}$, $mAP_{50-95}$, Precision@0.25, Recall@0.25, F1@0.25, and False Positives per image.
3. **Reproducible Archival**:
   - Raw prediction outputs for all models on the test set will be saved as JSON artifacts in `research/test_evaluation/predictions/`.
   - Consolidated metrics will be generated as `research/final_phase/08_FINAL_TEST_RESULTS.md`.

---

## 5. Artifact Manifest

1. [hardware_benchmarks.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_analysis/computational_analysis/hardware_benchmarks.csv) — Latency and memory benchmarks supporting model selection.
2. [per_scene_metrics.csv](file:///C:/VI%20Fusion%20for%20LL-PD/research/final_analysis/validation_analysis/per_scene_metrics.csv) — Per-scene performance validation records.
3. [dataset_split_manifest.csv](file:///C:/VI%20Fusion%20for%20LL-PD/LLVIP_MULTIMODAL_RESEARCH/results/metrics/dataset_split_manifest.csv) — Official dataset split manifest containing the sealed 3,463 test pairs.

---

> [!IMPORTANT]
> **GATEWAY HOLD — AWAITING RESEARCHER CONFIRMATION**  
> Model checkpoints are locked. Checkpoint hashes are verified. The test partition is sealed.  
> **Please review the model selection rationales and checkpoint register. Do you grant approval to unseal the official test partition and execute Stage 8?**
