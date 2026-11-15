# Figure & Table Manifest: LLVIP Research Report Finalization

**Project Title**: Reliability-Aware Asymmetric Visible-Infrared Fusion for Low-Light Pedestrian Detection  
**Primary Deliverable**: `LLVIP_Research_Report_Final.docx`  
**Date**: November 15, 2026  
**Finalization Pipeline**: Single-pass authentic insertion; zero synthetic placeholders; 100% genuine project artifacts.  

---

## 1. Summary of Insertions

- **Total Figure Placeholders Filled**: 14 of 14 (100% filled with authentic images)
- **Total Table Placeholders Filled**: 3 of 3 (100% filled with authentic measured data tables)
- **Total Placeholders / Callouts Removed**: 18 of 18 (including Table 1 internal handoff callout)
- **Remaining `[[...]]` Markers**: 0
- **Remaining Antigravity Instructional Text**: 0

---

## 2. Comprehensive Figure Manifest

| Figure ID | In-Report Title | Category / Provenance | Source Path / Origin | Image Dimensions | Pair ID / Scene ID | Description / Rationale |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Figure VI-1** | End-to-end research workflow | Schematic Diagram | `research/final_report_delivery/figures/figure_vi_1_workflow.png` | 4200 × 2400 px | N/A (Overall) | Clean 8-stage end-to-end research execution pipeline illustrating dataset audit, scene-aware split, preprocessing, 7-model zoo, validation, diagnostics, model freeze, and official test evaluation. |
| **Figure VI-2** | Paired RGB and thermal samples with ground-truth annotations | Authentic Dataset Sample | `research/final_report_delivery/figures/figure_vi_2_paired_dataset_samples.png` | 4200 × 3000 px | Pairs `020001` (Scene 02) & `050001` (Scene 05) | Two authentic validation pairs from LLVIP showing deep visible darkness with high thermal contrast (Pair 020001) and streetlight illumination with distant pedestrians (Pair 050001). Green bounding boxes indicate verified ground-truth pedestrian annotations. |
| **Figure VI-3** | Implemented experiment families (E001–E007) | Architecture Comparison Schematic | `research/final_report_delivery/figures/figure_vi_3_experiment_architectures.png` | 4800 × 2700 px | All Models (E001–E007) | Comparative architectural topology across the six implemented model families: Unimodal RGB (E001/E002), Unimodal Thermal (E003), Late Fusion WBF (E004), 4-Channel Early Fusion (E005), Feature Concatenation (E006), and Proposed Asymmetric Gated Fusion (E007). |
| **Figure VI-4** | E007 thermal-primary asymmetric gated fusion architecture | Detailed Architecture Schematic | `research/final_report_delivery/figures/figure_vi_4_e007_architecture.png` | 4200 × 2400 px | E007 | Detailed structural specification derived directly from source code (`asymmetric_gated_fusion_yolo26.py`), showing dual YOLO26n backbones, multi-scale `ReliabilityGatingModule` at P3/P4/P5, unmodulated additive thermal identity residual shortcut, and shared 80-class COCO head. |
| **Figure VI-5** | Examples of selected synthetic corruption levels | Authentic Robustness Outputs | `research/final_report_delivery/figures/figure_vi_5_robustness_examples.png` | 4200 × 4500 px | 25 Completed Conditions | Authentic multi-panel composite showing samples across the 5 completed degradation suites (RGB Brightness, RGB Gaussian Noise, RGB Gaussian Blur, Thermal Contrast Compression, and Thermal Striping) at Severity Levels 1 (mild), 3 (moderate), and 5 (extreme). |
| **Figure VII-1** | LLVIP visible and thermal input pair | Authentic Dataset Sample | `research/final_report_delivery/figures/figure_vii_1_rgb_thermal_pair.png` | 4200 × 1800 px | Pair `250001` (Scene 25) | Validation pair from Scene 25 depicting challenging oncoming vehicular headlight glare, deep shadowed background, and verified ground-truth pedestrian annotations in native 1280×1024 coordinate space. |
| **Figure VII-2** | Dataset split and annotation audit | Authentic Measurement Plot | `research/final_report_delivery/figures/figure_vii_2_dataset_audit.png` | 4500 × 1500 px | 26 Scenes Total | Scene-aware split partition audit chart showing 10,222 training pairs (15 scenes, 29,044 valid GTs), 1,803 validation pairs (4 scenes, 5,086 valid GTs), and 3,463 official test pairs (7 scenes, 8,302 valid GTs), without sensitive local paths. |
| **Figure VII-3** | Training convergence across key architectures | Authentic Training Metric Curves | `research/final_report_delivery/figures/figure_vii_3_training_curves.png` | 4200 × 4500 px | E003, E006, E007 | Authentic multi-panel training loss and metric trajectories extracted from experimental logs for E003 (Thermal YOLO26n, best epoch 48 / early stopping 58), E006 (Feature Concat YOLO26n), and E007 (Proposed Asymmetric Gated YOLO26n). |
| **Figure VII-4** | Qualitative detection comparison on validation Scene 02 | Authentic Evaluation Output | `research/final_report_delivery/figures/figure_vii_4_qualitative_detection_comparison.png` | 2400 × 1200 px | Pair `020001` (Scene 02) | Ground-truth-matched 8-panel detection comparison on validation Scene 02: (1) Raw Visible, (2) Raw Thermal, (3) Ground Truth, (4) E002 RGB, (5) E003 Thermal, (6) E004 Late WBF, (7) E006 Feature Concat, and (8) E007 Proposed Asymmetric Gated Fusion. |
| **Figure VII-5** | Gate activation and scene-conditioned analysis | Authentic Diagnostic Plot | `research/final_report_delivery/figures/figure_vii_5_gate_analysis.png` | 2400 × 1714 px | Scenes 02, 05, 07, 25 | Consolidated 4-panel diagnostic analysis resolving Simpson's paradox: within-scene gate vs brightness scatter, within-scene centered residuals ($r = +0.116$), foreground vs background gate distribution ($G_V \approx 0.505$ vs $0.513$), and moving block bootstrap distribution (SE = 0.075). |
| **Figure VII-6** | Inference-only gate intervention comparison (IA001–IA004) | Authentic Intervention Plot | `research/final_report_delivery/figures/figure_vii_6_gate_intervention.png` | 2400 × 733 px | IA001–IA004 (E007) | Quantitative comparison of post-training gate clamping interventions on E007: Learned Gate (IA001, 93.70%), Gate Zero (IA002, 72.27%), Gate Half (IA003, 84.24%), and Gate One (IA004, 78.03%), demonstrating network co-adaptation and noise flood effects. |
| **Figure VII-7** | Selected synthetic-corruption robustness results | Authentic Robustness Plot | `research/final_report_delivery/figures/figure_vii_7_robustness_results.png` | 2500 × 500 px | 25 Completed Conditions | Metric retention curves ($mAP_{50} / \text{Clean } mAP_{50}$) across the 25 completed corruption conditions. Highlights E007 retaining 51.17% $mAP_{50}$ (54.6% retention) under severe thermal striping (L5 $\sigma=80$), outperforming feature concatenation E006 ($30.88\% mAP_{50}$, 33.1% retention) by +20.29% absolute $mAP_{50}$. |
| **Figure VII-8** | Computational performance and hardware profiling | Authentic Benchmark Plot | `research/final_report_delivery/figures/figure_vii_8_latency_comparison.png` | 4500 × 1800 px | E001–E007 on RTX 4060 | Dual-panel profiling chart: Left panel compares model forward latency vs full end-to-end pipeline latency in ms with real-time 30 FPS threshold line (E007 runs at 55.7 FPS / 17.94 ms); Right panel compares parameter count (M) vs GFLOPs at 640×640 input resolution. |
| **Figure VII-9** | Software execution evidence and audit ledger | Authentic Code & Execution Montage | `research/final_report_delivery/figures/figure_vii_9_software_execution.png` | 4800 × 3300 px | Project Directory & Logs | Multi-panel software execution montage: Panel A shows PyTorch code of `ReliabilityGatingModule`; Panel B shows terminal logs from the official Stage 8 test evaluation; Panel C shows project directory tree; Panel D shows the Stage 7 cryptographic SHA256 model freeze ledger. |

---

## 3. Comprehensive Table Manifest

| Table ID | In-Report Title | Category / Provenance | Source Data File | Dimensions | Target Scope | Key Quantities Reported |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Table VII-1** | Computational performance and hardware profiling (Stage 6) | Measured Hardware Benchmark | `research/final_analysis/computational_analysis/hardware_benchmarks.csv` | 8 rows × 9 cols | Models E001–E007 + E008* | Model-only forward latency (ms), pipeline latency (ms), pipeline throughput (FPS), peak VRAM (MB at $B=1$ and $B=16$), parameter counts (M), and GFLOPs on NVIDIA RTX 4060 Laptop GPU. |
| **Table VII-2** | Inference-time gate interventions on E007 (Stage 3) | Measured Diagnostic Interventions | `research/inference_ablations/gate_intervention_metrics.csv` | 4 rows × 9 cols | IA001–IA004 on Val Split | $mAP_{50}$, $mAP_{50-95}$, Recall@0.25, Precision@0.25, F1@0.25, FP/image, and False Negative misses under Learned Gate, $G_V=0.0$, $G_V=0.5$, and $G_V=1.0$ clamping. |
| **Table VII-3** | Selected synthetic-corruption results across 25 completed conditions (Stage 4) | Measured Robustness Retention | `research/robustness/robustness_retention_summary.csv` | 6 rows × 8 cols | 25 Conditions across 5 Suites | Clean $mAP_{50}$ and retention percentages ($R_{\text{mean}}$) across RGB Brightness, RGB Noise, RGB Blur, Thermal Contrast, Thermal Striping, and 5-Suite Mean. Explicitly notes partial status of dead-pixels and deferral of joint suites. |

---

## 4. Verification Attestation

- [x] All 14 embedded image files exist on disk in `research/final_report_delivery/figures/`.
- [x] All 3 data tables populated with verified metrics matching machine-readable CSV/JSON sources.
- [x] Zero synthetic, stock, or online imagery utilized.
- [x] All image aspect ratios and text resolutions verified for clean print layout.
- [x] Exactly 0 raw `[[...]]` markers remain in the output document.
