# LLVIP Research Project Report — Source Text

**Working title:** Reliability-Aware Asymmetric Visible–Infrared Fusion for Low-Light Pedestrian Detection

This Markdown companion contains the report text and tables from the editable DOCX. Use the DOCX as the primary layout source. Figure and table markers are deliberate insertion points for Antigravity; replace them only with authentic artifacts from the project directory.

---
RESEARCH PROJECT REPORT

RELIABILITY-AWARE ASYMMETRIC
VISIBLE–INFRARED FUSION FOR
LOW-LIGHT PEDESTRIAN DETECTION

A paired RGB–thermal pedestrian-detection study using LLVIP and YOLO-based detectors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


|Student name|[Insert student name]|
|---|---|
|Registration number|[Insert registration number]|
|Programme / branch|B.Tech. — Computer Science and Engineering|
|Department / institution|[Insert department and institution]|
|Research guide|[Insert guide name]|
|Academic session|[Insert academic session]|
|Project category|Research-based (primary); application-oriented software prototype (secondary)|

Reporting stage: Completed experiments and held-out evaluation


# REPORT STRUCTURE

I.  Introduction (What, Why and How)

II.  Literature Survey

III.  Comparative Study

IV.  Problem Statement

V.  Objectives

VI.  Planning of Work / Proposed Solution

VII.  Results / Screenshots

VIII.  Bibliography / References


> How to finalize this report The narrative and tables in this document are authored. Figure markers identify where Antigravity should insert authentic project plots, architecture diagrams and screenshots. It must not invent figures or change reported results. Exact Stage 6 latency/VRAM and inference-ablation details must be imported from their final project artifacts if available.


# PROJECT SUMMARY

This project investigates pedestrian detection under low illumination using paired visible RGB and thermal infrared imagery. Seven experiments were implemented: visible-only and thermal-only YOLO detectors, late decision-level fusion, early four-channel fusion, intermediate feature concatenation and a proposed thermal-primary asymmetric gated-fusion model. The work uses the LLVIP dataset and evaluates accuracy, false-positive/false-negative behaviour, scene-conditioned gate activation, inference-time gate interventions, selected synthetic corruptions and computational cost.

The final supplied held-out test results show that no single model dominates every metric. Thermal-only detection (E003) obtains the highest F1 score and mAP50–95. Weighted Boxes Fusion (E004) obtains the highest precision and the lowest false-positive rate per image. The proposed asymmetric model (E007) obtains the highest recall and the largest true-positive count, but at the cost of substantially more false positives than thermal-only. The central finding is therefore a precision–recall and compute trade-off, not universal superiority of multimodal fusion.

Project classification: research-based (primary), application-oriented software prototype (secondary), and not hardware-development-based. No custom camera or sensor was fabricated. Experiments were run on an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.


# I. INTRODUCTION (WHAT, WHY AND HOW)


## 1.1 Background and context

Pedestrian detection is a core computer-vision task for intelligent transportation, driver-assistance systems and outdoor surveillance. A detector must identify a pedestrian and localize the person with a bounding box. Visible RGB cameras provide colour, texture, edges and contextual information, but their useful signal can deteriorate under darkness, underexposure, glare, shadows and low contrast. Thermal infrared cameras respond to emitted radiation and can preserve pedestrian-to-background contrast when visible illumination is weak. Thermal images, however, generally provide less colour and fine texture information than RGB images. These complementary characteristics motivate visible–thermal (RGB-T) perception and multimodal feature fusion [1]–[3].

The LLVIP dataset was introduced for low-light visible–infrared vision and provides paired, temporally and spatially aligned images with pedestrian labels [1]. The locally audited release used in this project contains 15,488 image pairs. The original LLVIP publication described an earlier 16,836-pair release; all counts in this report refer to the local version that was downloaded, audited and partitioned for the experiments [1].

A central challenge in multimodal detection is that combining two inputs does not automatically improve accuracy. A degraded or weak modality may contribute noise, redundant features or false detections. Fusion may also increase parameter count, memory use and inference latency. Research has therefore examined illumination-aware weighting, modality balancing, target-aware feature selection, cross-modal attention, alignment-aware fusion and lightweight residual compensation [4]–[19].


## 1.2 What is developed?

The project develops and evaluates a software-based pedestrian detector that can process RGB and thermal imagery separately or jointly. YOLO26n is used as the main detector family for the principal experiments, while YOLO11n is retained as a historical visible-only baseline. The proposed E007 network uses two YOLO26n-derived feature-extraction branches. Thermal features retain a direct residual path; projected visible features are multiplied by a learned sigmoid gate before being added to the thermal representation.

The implemented gate is feature-conditioned adaptive gating. It is generated from learned feature maps and does not use a separately calibrated physical sensor-quality estimator. Gate values must therefore be interpreted as learned feature-modulation coefficients, not as probabilities that a camera or sensor is reliable.


## 1.3 Why is the problem important?

A thermal-only detector can be highly effective in the low-light conditions represented by LLVIP. Nevertheless, thermal imagery may also face weak thermal contrast, overlapping heat signatures and confusing background heat sources. Visible imagery can sometimes supply complementary texture or contextual evidence. The research question is therefore not whether fusion must always outperform thermal-only detection, but whether a thermal-primary strategy can use visible features selectively and improve particular operating characteristics—especially recall—without an unacceptable increase in false positives or computational cost.


## 1.4 How is the project carried out?

- Audit and pair the visible and infrared images; validate image dimensions and pedestrian annotations.
- Create a scene-aware training/validation split and retain the official test partition for held-out evaluation.
- Train visible-only and thermal-only baselines.
- Implement and compare late WBF fusion, early four-channel fusion and intermediate feature concatenation.
- Implement the thermal-primary asymmetric gated-fusion architecture.
- Evaluate detection metrics, scene-level cases, gate activation, inference-time gate interventions, selected synthetic corruptions and computational cost.
- Freeze model settings based on validation evidence and evaluate the selected checkpoints on the held-out test set.

## 1.5 Project classification

Research-based (primary): The primary work is an empirical investigation of RGB–thermal fusion strategies, their precision–recall trade-offs, scene-conditioned gate behaviour and response to selected controlled corruptions.

Application-oriented (secondary): The trained detectors form a software prototype for low-light pedestrian detection. Potential application domains include night-time surveillance and driver-assistance perception; the project does not claim deployment certification or safety validation.

Hardware-based (not the project category): No custom camera, thermal sensor, embedded board or sensing device was designed or fabricated. Public paired imagery was used. The RTX 4060 Laptop GPU is the development platform, not a new hardware contribution.


## 1.6 Scope and limitations

The dataset contains one detection class, person; the results do not establish multi-class vulnerable-road-user detection. The validation analysis covers four held-out scenes from the official training pool. The held-out test set supports an evaluation on this benchmark but cannot establish generalization to all cameras, climates or sensor designs. Synthetic corruptions are controlled stress tests rather than substitutes for collecting real fog, rain, physical sensor failure or thermal-crossover data. Historical multimodal models E005–E007 retained 80-class COCO detection heads, whereas unimodal YOLO26n models E002–E003 used one-class heads; this is disclosed as a comparability limitation.


# II. LITERATURE SURVEY


## 2.1 Overview of the research landscape

RGB–thermal pedestrian detection has progressed from handcrafted multispectral features and simple channel combination to dual-stream convolutional networks, illumination-aware fusion, modality balancing, target-aware feature selection, cross-modal attention, deformable alignment and lightweight gated compensation. Survey studies identify image fusion, pedestrian detection, segmentation, tracking and person re-identification as major RGB-T research directions [2], [3]. More recent work increasingly addresses three connected challenges: preserving modality-specific information, suppressing noisy or redundant cross-modal responses, and maintaining accuracy under computational constraints [8]–[18].

The following review covers 18 directly relevant studies and two foundational benchmark/method papers. The strengths and limitations are summarized from the published method descriptions. Where a limitation is an engineering trade-off inferred from the architecture rather than an author-reported weakness, it is labelled as a consideration.


### 1. LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision — Jia et al. (2021) [1]

Method: Introduces paired visible and infrared images for low-light vision, with temporal/spatial alignment and pedestrian annotations.

Advantages: Directly matches the project domain; aligned pairs support controlled modality experiments; pedestrian labels allow detection benchmarks.

Limitations: The dataset is predominantly low-light and single-class pedestrian data, so results should not be generalized to all-day, all-weather or multi-class road-user detection. The paper describes an earlier 16,836-pair release; the local audited release used here contains 15,488 pairs.


### 2. RGB-T Image Analysis Technology and Application: A Survey — Song et al. (2023) [2]

Method: Surveys RGB-T image fusion and application areas including salient-object detection, segmentation, pedestrian detection, tracking and person re-identification.

Advantages: Broad map of RGB-T methods and applications; useful for understanding the wider field and recurring fusion strategies.

Limitations: As a survey, it does not provide a directly executable detector or a single standardized benchmark. Methods summarized across different datasets and metrics cannot be ranked without protocol matching.


### 3. RGB-D and Thermal Sensor Fusion: A Systematic Literature Review — Brenner et al. (2023) [3]

Method: Reviews RGB-D-thermal fusion across applications and discusses calibration, registration, data representation and deep-learning approaches.

Advantages: Highlights sensor calibration and alignment as prerequisites; provides a systematic view of fusion challenges beyond pedestrian detection.

Limitations: Its scope includes depth-plus-thermal applications, so some reviewed methods are not direct RGB-T pedestrian detectors. The diversity of hardware and tasks limits direct metric comparison.


### 4. All-Weather Pedestrian Detection Based on Double-Stream Multispectral Network — Hsia et al. (2023) [4]

Method: Uses a double-stream multispectral detector with YOLO-oriented components and an illumination-aware weighting strategy for visible and thermal information.

Advantages: Explicitly models two modalities and adapts modality contributions; evaluates on KAIST and FLIR-style multispectral data.

Limitations: Multiple branches and illumination weighting increase implementation and tuning complexity. Results on KAIST/FLIR are not directly comparable with LLVIP results under a different split.


### 5. Illumination-Aware Cross-Modality Differential Fusion — Wang et al. (2023) [5]

Method: Proposes illumination-aware modality weights and cross-modality differential information to improve multispectral pedestrian detection.

Advantages: Recognizes that modality contribution should depend on scene conditions; uses differential cues to expose complementary information.

Limitations: The illumination estimator and differential-fusion components add design complexity. Scene illumination is not the only source of modality reliability, and estimated weights are not automatically calibrated.


### 6. Cascaded Information Enhancement and Cross-Modal Attention Feature Fusion — Yang et al. (2023) [6]

Method: Combines cascaded feature enhancement with channel/spatial attention and cross-modal attention-based fusion to reduce pedestrian/background confusion.

Advantages: Refines modality features and explicitly addresses background interference before or during fusion.

Limitations: Cascaded enhancement and attention add computational operations and tuning choices; attention values should not be interpreted as physical sensor-quality scores.


### 7. Real-Time Multispectral Pedestrian Detection with Weakly Aligned Cross-Modal Learning (CMLNet, 2023) [7]

Method: Uses spatial-semantic alignment and feature reselection with a YOLOX-based detector to address weak RGB-thermal registration.

Advantages: Recognizes that real sensor pairs may not be perfectly registered and combines alignment with feature-redundancy reduction.

Limitations: Alignment and feature-selection modules add complexity. Its reported KAIST protocol and miss-rate metrics are not directly comparable with LLVIP mAP results.


### 8. TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection — Zhang et al. (2025) [8]

Method: Develops target-aware fusion to address false positives caused by noisy fused feature maps.

Advantages: Directly considers target relevance and false-positive suppression, an issue also visible in this project’s WBF and gated-fusion results. Evaluates KAIST and LLVIP and extends to FLIR and M3FD.

Limitations: Dedicated target-aware modules and objectives add complexity. Its architecture and evaluation protocol differ from the YOLO26n implementation in this project.


### 9. Deformable Cross-Attention Transformer for Weakly Aligned RGB-T Pedestrian Detection — Hu et al. (2025) [9]

Method: Uses dual-branch deformable cross-attention and positional information to model spatial relationships in weakly aligned RGB-T pairs; local attention samples a restricted set of keys and values.

Advantages: Addresses geometric misalignment more directly than simple concatenation; local sampling is intended to reduce the cost of global attention.

Limitations: Deformable attention and offset generation introduce architectural complexity. LLVIP pairs are described as aligned, so the method solves a different alignment regime from the primary dataset setup.


### 10. Illumination-Guided with Crossmodal Transformer Fusion for RGB-T Object Detection — Xie et al. (2025) [10]

Method: Combines illumination-guided adaptive fusion, frequency-domain convolutions and cross-modal transformer interaction.

Advantages: Couples scene illumination with modal weighting and models spectral/contextual interactions; reports evaluations on FLIR-aligned, LLVIP and KAIST.

Limitations: Transformer and frequency-domain components increase design complexity. Multi-module systems require careful ablation to identify which component produces a gain.


### 11. Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection — Zhang et al. (2025) [11]

Method: Re-examines design choices in early fusion for multispectral object detection.

Advantages: Relevant to four-channel input fusion and challenges the assumption that raw modality stacking is sufficient. Offers a useful methodological reference for designing early-fusion baselines.

Limitations: It is a broader multispectral object-detection study rather than a direct replication of this project’s LLVIP four-channel setup. Dataset and implementation details must be matched before comparing metrics.


### 12. PedDet: Adaptive Spectral Optimization for Multimodal Pedestrian Detection — Zhao et al. (ECAI 2025 proceedings; first online 2026) [12]

Method: Uses multi-scale spectral feature perception and illumination-robustness feature decoupling for multimodal pedestrian detection.

Advantages: Explicitly targets complementary spectral features and changing illumination; reports work on LLVIP and M3FD.

Limitations: Dedicated spectral and decoupling modules increase implementation/tuning burden. Headline results cannot be ranked against this project without matching evaluation settings.


### 13. MCFusion: A Lightweight RGB-T Pedestrian Detection Method with Progressive Thermal Compensation — Li et al. (2026) [13]

Method: Uses an RGB-primary dual branch and progressively injects thermal information through a zero-initialized residual compensation gate at P4/P5, followed by a shared lightweight head.

Advantages: Highly relevant gated-residual design; explicitly considers lightweight implementation and evaluates on LLVIP and KAIST.

Limitations: It is RGB-primary, whereas E007 is thermal-primary. Its accuracy figures and training protocol differ from this project and should not be directly compared.


### 14. MSAFusion: A Lightweight Multispectral Pedestrian Detection Network with Multi-Scale and Adaptive Feature Fusion — Song et al. (2026) [14]

Method: Uses multi-scale feature refinement, cross-modality transformer interaction and adaptive feature recalibration.

Advantages: Addresses pre-fusion feature quality, cross-modal interaction and post-fusion recalibration; evaluates on KAIST, LLVIP and FLIR-aligned data.

Limitations: The combination of multiple modules increases design choices and analysis burden. Code availability and reproduction effort should be checked against the paper’s stated access conditions.


### 15. Uni-MPedFormer: A Unified Transformer-Based Framework for Multispectral Pedestrian Detection — Li et al. (2026) [15]

Method: Combines a cross-modality information-fusion transformer, feature-correlation recalibration, U-shaped multi-scale aggregation and illumination-aware weight generation.

Advantages: Evaluates five benchmarks (KAIST, CVC-14, FLIR-ADAS, LLVIP and UTokyo) and reports 25 FPS on an RTX 2080 Ti; explicitly addresses illumination, occlusion and scale variation.

Limitations: Transformer encoder/decoder and attention modules are more complex than convolutional fusion. Its hardware, inputs and metrics differ from the RTX 4060/LLVIP setup here.


### 16. FID-YOLO: A Pedestrian Detection Model Integrating Multispectral Information in Complex Environments — Yang et al. (2026) [16]

Method: Combines illumination-aware image fusion, cascaded feature aggregation with re-parameterization/channel shuffle, and a scale-adaptive detection head.

Advantages: Targets complex scenes and scale variation; reports LLVIP and M3FD experiments with component ablations.

Limitations: The fusion, aggregation and detection-head components create multiple design choices; reported results require protocol matching before comparison.


### 17. SSMPD: Semi-Supervised Learning for Multispectral Pedestrian Detection — Shin et al. (2026) [17]

Method: Extends multispectral pedestrian detection to semi-supervised learning using modality-aware pseudo-label learning, similarity-based contrastive loss and data selection.

Advantages: Addresses the cost of fully labelled multispectral data and evaluates on KAIST and LLVIP; an official implementation is available.

Limitations: Requires teacher/student or pseudo-label training and additional unlabeled-data management; it is not a direct replacement for the fully supervised comparison used in this project.


### 18. CMT-Det: A Cross-Modality Transformer Method for Road Scene Object Detection — Li et al. (2026) [18]

Method: Uses cross-modality transformer fusion, directional feature enhancement and scale-aware multi-level attention for RGB-T road-scene detection.

Advantages: Combines global cross-modal interaction, structural feature enhancement and multi-scale attention; includes LLVIP and RoadScene experiments.

Limitations: Transformer and multi-stage attention increase design complexity. Reported F1 is not directly comparable without matching split, preprocessing and threshold definitions.


## 2.2 Foundational methods and benchmark references


### Modality Balance Network (MBNet) — Zhou et al. (ECCV 2020) [19]

MBNet identifies modality imbalance as an optimization issue in multispectral pedestrian detection and proposes differential modality-aware fusion and illumination-aware feature alignment. Its advantage is a direct treatment of unequal modality contributions; its limitation for this report is that it predates the 2023–2026 work and uses different benchmark protocols. It remains a useful foundational reference for why equal treatment of modalities may be suboptimal.


### KAIST Multispectral Pedestrian Detection Benchmark — Hwang et al. (CVPR 2015) [20]

The KAIST benchmark established paired colour-thermal pedestrian detection as a formal benchmark and introduced a multispectral ACF baseline. Its value is the aligned day/night pedestrian benchmark and temporal annotations; its age and dataset-specific annotation protocol mean that contemporary detectors and LLVIP results require separate evaluation.


## 2.3 Software, methods and hardware used in this project

Ultralytics YOLO26n [21]: Primary detector family and feature-backbone basis. The nano scale reduces compute relative to larger variants; YOLO26 provides an end-to-end option and updated training/detection design. It is not a native RGB-T fusion network, so custom dual-stream/fusion code was required.

Ultralytics YOLO11n [22]: Historical visible-only baseline. Useful for an earlier lightweight YOLO reference; not the primary architecture for the later fusion experiments.

PyTorch [23]: Deep-learning implementation, automatic differentiation, custom module definition and CUDA execution. Flexible for fusion research; dual-stream models increase GPU-memory and debugging demands.

OpenCV [24]: Image reading, conversion, geometric transforms, visualization and deterministic synthetic corruptions. Broad and efficient image-processing toolkit; processing choices and library versions must be recorded.

Unified evaluation and fusion scripts: Common coordinate restoration, matching, AP computation, WBF and latency reporting. Enables consistent analysis, but validity depends on documented IoU matching, thresholds, NMS/WBF settings and exact software versions.

NVIDIA RTX 4060 Laptop GPU (8 GB VRAM): Development and inference hardware. CUDA acceleration supports the lightweight detector and custom fusion models; 8 GB VRAM limits batch size and makes dual-stream training more memory-intensive. It is an existing platform, not custom hardware developed in this project.

RGB-visible modality: Provides colour, texture, edges and context. Its limitations are darkness, glare, underexposure, shadows and contrast loss.

Thermal-infrared modality: Provides thermal-pattern information that is less dependent on visible illumination. Its limitations include reduced colour/texture, weak thermal contrast and confusing background heat sources.


## 2.4 Literature synthesis and research positioning

The literature shows that RGB-T detection is not solved by simply increasing the number of input channels. Illumination-aware methods [5], [10], [12], target-aware fusion [8], modality balancing [19], deformable alignment [9], transformer interaction [10], [14], [15], and progressive residual compensation [13] all attempt to preserve useful modality-specific information while controlling noise and redundancy. These methods also create different costs in computation, memory, calibration, alignment assumptions and implementation complexity.

This project positions E007 as a thermal-primary, feature-conditioned asymmetric residual-fusion implementation. Thermal features retain a direct identity path, while projected RGB features are modulated by a learned convolutional sigmoid gate before addition. Unlike explicit illumination- or quality-estimation approaches, E007 does not receive a separate physical image-quality descriptor. The gate is therefore interpreted as a learned feature selector, not a calibrated physical sensor-reliability estimator.


# III. COMPARATIVE STUDY


## 3.1 Comparison of implemented experiments

The seven experiments differ in input modality and fusion level. E001 is a historical YOLO11n RGB baseline; E002–E007 use YOLO26n-derived models. The comparison is intended to show how modality choice and fusion strategy alter the precision–recall and computational trade-offs.


|ID|Architecture|Input / fusion|Comparative purpose|
|---|---|---|---|
|E001|YOLO11n|RGB only|Historical visible-spectrum baseline.|
|E002|YOLO26n|RGB only|Primary visible-only reference for the YOLO26n family.|
|E003|YOLO26n|Thermal only|Primary thermal-only reference.|
|E004|Two YOLO26n detectors + WBF|Late decision fusion|Combines detections from independent RGB and thermal models.|
|E005|YOLO26n|Early 4-channel fusion|Stacks R, G, B and thermal at the input.|
|E006|Dual-stream YOLO26n|Intermediate feature concatenation|Combines modality feature maps before the shared detection pathway.|
|E007|Dual-stream YOLO26n + learned gate|Thermal-primary asymmetric feature fusion|Adds gated projected RGB features to thermal features through a residual path.|


## 3.2 Final held-out test comparison

The project’s final result table covers 3,463 official test image pairs and 8,302 valid ground-truth pedestrian instances. Precision, recall and F1 are reported at confidence threshold 0.25. The precise IoU matching configuration for TP/FP/FN must match the final evaluator source code retained in the project repository.


|ID|Strategy|mAP50|mAP50–95|Precision|Recall|F1|
|---|---|---|---|---|---|---|
|E001|RGB / YOLO11n|79.34|42.35|92.94|68.36|78.78|
|E002|RGB / YOLO26n|81.44|44.16|89.98|72.79|80.48|
|E003|Thermal / YOLO26n|93.34|59.35|90.04|87.77|88.89|
|E004|Late fusion / WBF|93.57|58.27|97.20|73.52|83.72|
|E005|Early fusion / 4CH|56.00|32.26|54.32|60.33|57.17|
|E006|Feature concat|91.00|57.23|81.56|86.86|84.13|
|E007|Proposed gated|91.43|56.76|81.22|88.52|84.71|


|ID|True positives|False positives|False negatives|FP / image|
|---|---|---|---|---|
|E001|5,675|431|2,627|0.124|
|E002|6,043|673|2,259|0.194|
|E003|7,287|806|1,015|0.233|
|E004|6,104|176|2,198|0.051|
|E005|5,009|4,213|3,293|1.217|
|E006|7,211|1,630|1,091|0.471|
|E007|7,349|1,699|953|0.491|

TP + FN = 8,302 for every model. FP/image is calculated over 3,463 held-out test pairs. Values are reproduced from the final table supplied for this report.


## 3.3 Comparative interpretation

Thermal-only (E003). E003 obtains the highest F1 (88.89%) and mAP50–95 (59.35%). It provides the strongest balanced result among these reported metrics, with 7,287 true positives, 806 false positives and 1,015 misses.

Late fusion (E004). WBF obtains the highest mAP50 (93.57%), precision (97.20%) and lowest FP/image (0.051), but recall is 73.52%. It is the precision-oriented option in this comparison, not the highest-recall option.

Proposed gated fusion (E007). E007 has the highest recall (88.52%) and most true positives (7,349). Compared with E003, it finds 62 additional pedestrians and reduces false negatives by 62, but false positives rise from 806 to 1,699. Its mAP50, mAP50–95 and F1 remain below E003.

Feature concatenation (E006). E006 reaches 57.23% mAP50–95 and 86.86% recall. E007 improves recall by 1.66 percentage points and F1 by 0.58 points, but E007 has lower mAP50–95 and 69 more false positives.

Early fusion (E005). The four-channel input model records 56.00% mAP50 and 4,213 false positives. This shows that simple channel stacking was ineffective under this implementation and training configuration; it does not prove that all early-fusion designs are ineffective.

Visible-only baselines (E001/E002). YOLO26n RGB (E002) improves over the historical YOLO11n RGB baseline (E001), but both visible-only models remain below the thermal-only model on the reported low-light test set.


> Research conclusion from the comparative study The results do not support a claim that multimodal fusion universally outperforms thermal-only detection. Instead, performance depends on the operating objective: E003 offers the strongest F1/high-IoU balance, E004 minimizes false alarms, and E007 maximizes recall at the cost of additional false positives and higher computation.


# IV. PROBLEM STATEMENT

Pedestrian detection based only on visible RGB imagery can become unreliable under low illumination because darkness, glare, shadows and reduced contrast weaken the visual evidence required to localize pedestrians, while thermal infrared imagery can preserve pedestrian heat signatures but may lack fine appearance detail and can be affected by weak thermal contrast or background heat sources. Although visible and thermal modalities provide complementary information, naïve fusion through raw channel stacking, fixed feature concatenation or decision-level aggregation can introduce redundant features and false detections, increase computational cost, or fail to improve upon a strong thermal-only model. The problem addressed in this project is to investigate and evaluate whether a thermal-primary asymmetric feature-fusion strategy—in which visible features are projected and selectively modulated by a learned gate before being added to a thermal representation—can provide a useful precision–recall trade-off for low-light pedestrian detection on LLVIP while maintaining practical inference performance and clearly characterizing its limitations.


# V. OBJECTIVES

1. Develop a reliable paired-data pipeline. Validate RGB–thermal correspondence, inspect pedestrian annotations, preserve the official test partition and create a scene-aware training/validation split for reproducible experiments.

2. Establish unimodal and conventional-fusion baselines. Evaluate RGB-only and thermal-only detectors and compare late decision-level fusion, early four-channel fusion and intermediate feature concatenation under a documented common evaluation procedure.

3. Implement and evaluate asymmetric gated fusion. Develop a thermal-primary dual-stream model that preserves a direct thermal feature path and adaptively modulates the visible feature contribution using a learned convolutional gate.

4. Measure detection and computational trade-offs. Compare mAP, precision, recall, F1, true/false detections, false positives per image, latency, parameter count, FLOPs and memory use.

5. Study gate behaviour and selected robustness conditions. Analyse gate activations by scene and image characteristics, perform inference-time gate interventions and evaluate selected synthetic visible/thermal corruptions using frozen checkpoints.

6. Identify operating points and limitations. Determine when thermal-only, WBF or asymmetric fusion is preferable and document the single-class scope, scene coverage, detection-head mismatch and incomplete corruption matrix.


# VI. PLANNING OF WORK / PROPOSED SOLUTION


## 6.1 Research workflow

The project follows a controlled experimental workflow. The raw dataset is not modified. Visible and thermal image pairs are identified by validated sample identifiers and checked against pedestrian annotations. A scene-aware split is used for training and validation, while the official test partition is kept separate for the final held-out evaluation. The main detector family is YOLO26n; YOLO11n is retained as a historical RGB baseline. Each architecture is recorded as an individual experiment and evaluated using the project’s unified evaluation pipeline.


> [[FIGURE_VI_1_WORKFLOW]] Figure VI-1. End-to-end research workflow Antigravity: insert an existing flowchart or create a clean diagram from the written steps only. Do not add uncompleted stages.


## 6.2 Dataset and scene-aware split

The local LLVIP release contains 15,488 validated visible–thermal pairs and one detection category (`person`). The training/validation pool contains 12,025 pairs and was split by scene: 10,222 training pairs across 15 scenes and 1,803 validation pairs across four held-out scenes (02, 05, 07 and 25). The official test partition contains 3,463 pairs across seven scenes. The scene-aware validation split reduces the risk of placing highly related neighbouring frames from the same surveillance sequence into both training and validation. It does not establish broad geographic generalization beyond the four validation scenes.


|Partition|Pairs|Scene allocation / role|
|---|---|---|
|Training|10,222|15 scenes from official training pool|
|Validation|1,803|Scenes 02, 05, 07 and 25|
|Official test|3,463|Seven official test scenes; held out for final evaluation|
|Total|15,488|26 scenes in the local audited release|

The original annotation audit recorded 42,437 pedestrian boxes. Five zero-width degenerate boxes were excluded from converted labels; all remaining valid boxes were retained, resulting in 42,432 valid boxes. No complete image pair was discarded solely because it contained one degenerate annotation. Validation contained 5,088 raw boxes and 5,086 valid evaluation boxes after the two validation exclusions.


> [[FIGURE_VI_2_PAIRED_DATASET_SAMPLES]] Figure VI-2. Paired RGB and thermal samples with ground-truth annotations Antigravity: use genuine pairs from the local project gallery; show both modalities and the corresponding person boxes.


## 6.3 Preprocessing and data management

- Pair visible and infrared images using validated sample identifiers, not directory order; verify both image files and their annotation file.
- Check image readability, dimensions, missing files, duplicate IDs, annotation classes and bounding-box coordinates. Preserve the original XML annotation files.
- Convert valid person boxes into the detector label format while recording the five degenerate zero-width exclusions in a separate audit report.
- Apply identical geometric transformations to both modalities and to their bounding boxes. Use letterboxing to 640 × 640 while preserving aspect ratio and applying the same coordinate transform to labels.
- Use modality-appropriate photometric transforms: visible brightness/contrast/colour augmentation is not blindly applied to thermal imagery.
- Keep the official test partition outside training, early stopping, hyperparameter selection and confidence-threshold tuning.

## 6.4 Baseline and fusion architectures


|ID|Architecture|Input / strategy|Purpose|
|---|---|---|---|
|E001|YOLO11n|RGB only|Historical visible baseline.|
|E002|YOLO26n|RGB only|Primary visible-only reference.|
|E003|YOLO26n|Thermal only|Primary thermal-only reference.|
|E004|Two YOLO26n models + WBF|Late decision fusion|Combines detections from independently trained RGB and thermal detectors.|
|E005|YOLO26n|Early 4-channel fusion|Stacks R, G, B and thermal at the input.|
|E006|Dual-stream YOLO26n|Intermediate feature concatenation|Concatenates modality features before shared detection components.|
|E007|Dual-stream YOLO26n + gate|Thermal-primary asymmetric gated fusion|Adds gated projected visible features to thermal features through a residual path.|


> [[FIGURE_VI_3_EXPERIMENT_ARCHITECTURES]] Figure VI-3. Implemented experiment families (E001–E007) Antigravity: distinguish RGB-only, thermal-only, WBF, 4-channel early fusion, feature concatenation and asymmetric gated fusion.


## 6.5 Proposed E007 architecture

E007 is a dual-stream YOLO26n-derived network. Both visible and thermal streams extract modality-specific features. The visible feature map is projected to dimensions compatible with the thermal feature map. A learnable convolutional gate is calculated from the thermal feature and projected visible feature. The thermal feature retains a direct residual/identity route; only the projected visible contribution is multiplied by the gate before addition. The fused multi-scale features are passed to the detection neck and head.


> Implemented formulation (feature-conditioned gate) G_V = sigmoid(GateNet(concat(F_T, P(F_V)))  
> F_fused = F_T + G_V ⊙ P(F_V)  
> F_T: thermal feature; F_V: visible feature; P: feature projection; G_V: learned gate. The actual E007 implementation does not contain explicit Q_T/Q_V physical quality descriptors and does not output calibrated sensor-reliability probabilities.

The architecture is asymmetric in the way information is fused: thermal features are retained through an unmodulated residual path, while visible features are conditionally injected. The source audit indicates two YOLO26n-derived backbones; the visible branch should not be described as a lightweight encoder. The saved E007 checkpoint retained an 80-class COCO detection head, whereas the unimodal YOLO26n baselines used one-class heads. This historical difference is disclosed as a limitation rather than silently corrected.


> [[FIGURE_VI_4_E007_ARCHITECTURE]] Figure VI-4. E007 thermal-primary asymmetric gated fusion architecture Antigravity: derive from the actual source. Show two full backbones, projection, learned gate, thermal residual path and shared neck/head. Do not add explicit Q_T/Q_V quality-estimator blocks.


## 6.6 Training and evaluation protocol

The initial detector experiments used 640 × 640 letterboxed inputs and an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM. The training environment used CUDA-enabled PyTorch and AMP where configured. The initial single-stream runs used a maximum 40-epoch schedule with early stopping; actual stopping epochs and optimizer settings should be quoted from each saved experiment configuration rather than inferred from the maximum budget. Dual-stream models required more memory and used conservative batch settings.

The evaluation pipeline maps predicted boxes back to the original image coordinate system before matching. It reports mAP50 and mAP50–95, plus precision, recall and F1 at a specified operating confidence threshold. The final held-out table reports precision/recall/F1 at confidence 0.25. WBF timing must include both detector forward passes and the fusion procedure; dual-stream latency must include both branches and the fusion modules. Software versions, NMS/WBF settings, confidence thresholds and IoU matching settings should be retained with the final evaluation artifacts.


## 6.7 Inference-only gate interventions and robustness protocol

The final analysis used the existing E007 checkpoint and temporary inference-time interventions: (i) the original learned gate, (ii) the gate forced to zero, (iii) a constant gate of 0.5 and (iv) a constant gate of 1.0. These interventions measure the trained network’s dependence on the visible feature stream. They are not equivalent to separately training a thermal-only model or to training fixed-gate architectures, because E007’s backbones, neck and detection head co-adapted during training.

Robustness evaluation used frozen checkpoints and deterministic synthetic corruptions on the validation partition. Twenty-five of the forty planned conditions were completed: five severity levels each for RGB brightness attenuation, RGB Gaussian noise, RGB blur, thermal contrast compression and thermal striping. Thermal dead-pixel testing was incomplete, and simultaneous-modality degradation suites were deferred. The resulting analysis is a selected-corruption study, not an exhaustive or real-world all-weather robustness evaluation.


> [[FIGURE_VI_5_ROBUSTNESS_EXAMPLES]] Figure VI-5. Examples of selected synthetic corruption levels Antigravity: use saved corruption outputs; label modality, corruption type and exact severity. Do not imply 40/40 completion.


## 6.8 High-level algorithm


> INPUT: paired visible image I_V, thermal image I_T  
> OUTPUT: pedestrian detections D  
>   
> 1. Load the validated RGB–thermal pair using its pair identifier.  
> 2. Apply the configured geometric preprocessing to both modalities.  
> 3. Extract thermal features F_T using the thermal backbone.  
> 4. Extract visible features F_V using the visible backbone.  
> 5. At each implemented fusion stage:  
>       a. Project visible features: F_Vp = P(F_V)  
>       b. Compute gate: G_V = sigmoid(GateNet(concat(F_T, F_Vp)))  
>       c. Fuse: F_fused = F_T + G_V elementwise-multiplied by F_Vp  
> 6. Pass fused multi-scale features through the detection neck/head.  
> 7. Decode person boxes and confidence scores.  
> 8. Apply the configured post-processing (NMS for the standard YOLO path).  
> 9. Restore boxes to original image coordinates.  
> 10. Return the final detections D.


## 6.9 Development milestones


|Milestone|Work|Status|
|---|---|---|
|1|Environment and repository setup|Completed|
|2|Dataset audit and annotation validation|Completed|
|3|Pairing, data loader and preprocessing|Completed|
|4|Unimodal baselines|Completed|
|5|Conventional fusion experiments|Completed|
|6|Proposed asymmetric gated fusion|Completed|
|7|Evaluation audit, gate analysis, inference interventions and selected robustness tests|Completed with limitations|
|8|Model freeze and held-out test evaluation|Reported complete|


# VII. RESULTS / SCREENSHOTS


## 7.1 Final held-out test results

The following table reproduces the final held-out result table supplied for this report. It covers 3,463 official test image pairs and 8,302 valid ground-truth pedestrian instances. Precision, recall and F1 are reported at confidence threshold 0.25. The exact IoU matching rule used for the TP/FP/FN operating-point counts must be taken from the final evaluation script retained in the repository.


|ID|Strategy|mAP50|mAP50–95|Precision|Recall|F1|
|---|---|---|---|---|---|---|
|E001|RGB / YOLO11n|79.34|42.35|92.94|68.36|78.78|
|E002|RGB / YOLO26n|81.44|44.16|89.98|72.79|80.48|
|E003|Thermal / YOLO26n|93.34|59.35|90.04|87.77|88.89|
|E004|Late fusion / WBF|93.57|58.27|97.20|73.52|83.72|
|E005|Early fusion / 4CH|56.00|32.26|54.32|60.33|57.17|
|E006|Feature concat|91.00|57.23|81.56|86.86|84.13|
|E007|Proposed gated|91.43|56.76|81.22|88.52|84.71|


|ID|TP|FP|FN|FP / image|
|---|---|---|---|---|
|E001|5,675|431|2,627|0.124|
|E002|6,043|673|2,259|0.194|
|E003|7,287|806|1,015|0.233|
|E004|6,104|176|2,198|0.051|
|E005|5,009|4,213|3,293|1.217|
|E006|7,211|1,630|1,091|0.471|
|E007|7,349|1,699|953|0.491|

TP + FN = 8,302 for every model. FP/image is reported over 3,463 test pairs. All values are reproduced from the final result table supplied for the report.


## 7.2 Main findings

Thermal-only detection. E003 obtains the highest F1 (88.89%) and mAP50–95 (59.35%), indicating the strongest balanced result and strict-IoU localization in this result set.

Precision-oriented late fusion. E004 obtains 97.20% precision and 0.051 FP/image, the best false-alarm profile, but recall is 73.52% and 2,198 pedestrians are missed.

Recall-oriented proposed fusion. E007 obtains 88.52% recall and 7,349 true positives. Relative to E003, it detects 62 additional pedestrians and reduces misses by 62, but increases false positives by 893.

Feature concatenation. E006 achieves 57.23% mAP50–95 and 86.86% recall. E007 improves recall and F1 over E006 but reduces mAP50–95 and increases false positives.

Early fusion. E005 underperforms under the implemented 4-channel configuration, indicating that simple channel stacking was not effective in this experiment.


## 7.3 Supplemental tables to be populated from final project artifacts

The final Stage 6 report and completed inference/robustness analysis contain additional measurements that were not present in the supplied summary table. Antigravity should import the exact final values from those authoritative artifacts. If a value is missing, retain an explicit “not available in supplied artifact” note rather than estimating it.


> [[TABLE_VII_1_COMPUTATIONAL_METRICS]] Table VII-1. Computational performance (Stage 6) Insert the verified model-only and end-to-end latency, FPS, peak VRAM, parameter count and FLOPs for each evaluated model.


> [[TABLE_VII_2_GATE_INTERVENTION_RESULTS]] Table VII-2. Inference-time gate interventions Insert the exact IA001–IA004 results for learned gate, G=0, G=0.5 and G=1.0. Identify the evaluation split and state that these are inference interventions.


> [[TABLE_VII_3_ROBUSTNESS_SUMMARY]] Table VII-3. Selected synthetic-corruption results Insert the exact completed 25-condition results. Mark the thermal dead-pixel suite partial and both joint-modality suites deferred.


## 7.4 Required figures and screenshots

Replace the following markers with authentic figures and screenshots from the project directory. Use original generated plots, saved detections and source-code screenshots. Every caption must identify the dataset split (validation or official test) and must not confuse results from different splits.


> [[FIGURE_VII_1_RGB_THERMAL_PAIR]] Figure VII-1. LLVIP visible and thermal input pair Insert a representative paired frame with ground-truth person boxes; label whether it is validation or official test.


> [[FIGURE_VII_2_DATASET_AUDIT]] Figure VII-2. Dataset split and annotation audit Insert scene distribution/audit plot showing 10,222 training, 1,803 validation and 3,463 test pairs.


> [[FIGURE_VII_3_TRAINING_CURVES]] Figure VII-3. Training convergence Insert authentic loss and mAP curves for E003, E006 and E007; label best epoch and stopping epoch from logs.


> [[FIGURE_VII_4_QUALITATIVE_DETECTION_COMPARISON]] Figure VII-4. Qualitative detection comparison Insert predictions from E002–E007 on identical frames, with ground truth and confidence labels.


> [[FIGURE_VII_5_GATE_ANALYSIS]] Figure VII-5. Gate activation and scene-conditioned analysis Insert corrected per-scene plots, within-scene centred correlations, foreground/background and object-scale plots with sample counts.


> [[FIGURE_VII_6_GATE_INTERVENTION]] Figure VII-6. Inference-only gate intervention Insert learned, G=0, G=0.5 and G=1.0 results from the inference ablation report; state these are inference interventions, not separately trained models.


> [[FIGURE_VII_7_ROBUSTNESS_RESULTS]] Figure VII-7. Selected robustness results Insert the 25 completed corruption conditions only; label five complete suites and identify partial/deferred conditions.


> [[FIGURE_VII_8_LATENCY_COMPARISON]] Figure VII-8. Computational performance Insert final Stage 6 model-only/end-to-end latency, FPS, parameters, FLOPs and peak VRAM from the final report; do not reuse provisional values.


> [[FIGURE_VII_9_SOFTWARE_EXECUTION]] Figure VII-9. Software execution evidence Insert authentic code, inference, evaluation and result-folder screenshots; remove unrelated desktop content and personal paths.


## 7.5 Limitations and interpretation

- The task contains one class (`person`); results do not establish multi-class vulnerable-road-user detection.
- The validation gate analysis uses four held-out scenes and should not be generalized to all camera locations.
- E005–E007 retained 80-class COCO detection heads whereas E002/E003 used one-class heads. This historical mismatch is disclosed and may confound some architectural comparisons.
- The experiments are single-run results; no multi-seed training was completed. Small numerical differences must not be described as statistically significant.
- The robustness evaluation completed 25 of 40 planned conditions: five levels each for RGB brightness, RGB Gaussian noise, RGB blur, thermal contrast compression and thermal striping. Thermal dead-pixel coverage was incomplete and joint-modality suites were deferred.
- Synthetic perturbations are controlled algorithmic stress tests, not proof of real-world fog, rain, physical sensor failure or all-weather reliability.
- The E007 gate is a learned feature-modulation map, not a calibrated physical sensor-quality estimator.
- Latency/FPS are hardware- and pipeline-dependent. Model-only throughput must be distinguished from end-to-end timing, which includes preprocessing and post-processing.

# VIII. BIBLIOGRAPHY / REFERENCES

The bibliography prioritizes original publisher, conference and official technical sources. Literature-reported results are described qualitatively and are not numerically ranked against this project because datasets, splits, preprocessing and evaluation protocols differ.

[1] X. Jia, C. Zhu, M. Li, W. Tang, and W. Zhou, “LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision,” in Proc. IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021, pp. 3496–3504. doi: 10.1109/ICCVW54120.2021.00389.

[2] K. Song, Y. Zhao, L. Huang, Y. Yan, and Q. Meng, “RGB-T image analysis technology and application: A survey,” Engineering Applications of Artificial Intelligence, vol. 120, Art. no. 105919, 2023. doi: 10.1016/j.engappai.2023.105919.

[3] M. Brenner, N. H. Reyes, T. Susnjak, and A. L. C. Barczak, “RGB-D and Thermal Sensor Fusion: A Systematic Literature Review,” IEEE Access, vol. 11, pp. 82410–82442, 2023. doi: 10.1109/ACCESS.2023.3301119.

[4] C.-H. Hsia, H.-C. Peng, and H.-T. Chan, “All-Weather Pedestrian Detection Based on Double-Stream Multispectral Network,” Electronics, vol. 12, no. 10, Art. no. 2312, 2023. doi: 10.3390/electronics12102312.

[5] C. Wang, J. Qian, J. Wang, and Y. Chen, “Illumination-Aware Cross-Modality Differential Fusion Multispectral Pedestrian Detection,” Electronics, vol. 12, no. 17, Art. no. 3576, 2023. doi: 10.3390/electronics12173576.

[6] Y. Yang, K. Xu, and K. Wang, “Cascaded information enhancement and cross-modal attention feature fusion for multispectral pedestrian detection,” Frontiers in Physics, vol. 11, Art. no. 1121311, 2023. doi: 10.3389/fphy.2023.1121311.

[7] “Real-Time Multispectral Pedestrian Detection with Weakly Aligned Cross-Modal Learning,” in Proc. IEEE International Conference on Real-time Computing and Robotics (RCAR), 2023. doi: 10.1109/RCAR58764.2023.10249327.

[8] X. Zhang, X. Zhang, J. Wang, J. Ying, Z. Sheng, H. Yu, C. Li, and H.-L. Shen, “TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 7, pp. 13276–13290, 2025. doi: 10.1109/TNNLS.2024.3443455.

[9] Y. Hu, X. Chen, S. Wang, L. Liu, H. Shi, L. Fan, J. Tian, and J. Liang, “Deformable Cross-Attention Transformer for Weakly Aligned RGB-T Pedestrian Detection,” IEEE Transactions on Multimedia, vol. 27, pp. 4400–4411, 2025. doi: 10.1109/TMM.2025.3543056.

[10] R. Xie, S. Jiang, Y. Bie, and M. Xia, “Illumination-guided with Crossmodal Transformer Fusion for RGB-T Object Detection,” Journal of Imaging Science and Technology, vol. 69, no. 2, Art. no. 020505, 2025. doi: 10.2352/J.ImagingSci.Technol.2025.69.2.020505.

[11] X. Zhang, S.-Y. Cao, F. Wang, R. Zhang, Z. Wu, X. Zhang, X. Bai, and H.-L. Shen, “Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection,” IEEE Transactions on Intelligent Vehicles, vol. 10, no. 6, pp. 3728–3742, 2025. doi: 10.1109/TIV.2024.3462488.

[12] R. Zhao, Z. Zhang, et al., “PedDet: Adaptive Spectral Optimization for Multimodal Pedestrian Detection,” in ECAI 2025 Proceedings, first published online 2026. doi: 10.3233/FAIA250915.

[13] H. Li, H. Xu, and D. Chen, “MCFusion: A Lightweight RGB-T Pedestrian Detection Method with Progressive Thermal Compensation,” Algorithms, vol. 19, no. 6, Art. no. 468, 2026. doi: 10.3390/a19060468.

[14] Y. Song, X. Zuo, C. Qu, Q. Qian, and D. Jiang, “MSAFusion: A Lightweight Multispectral Pedestrian Detection Network with Multi-Scale and Adaptive Feature Fusion,” Journal of Imaging, vol. 12, no. 6, Art. no. 246, 2026. doi: 10.3390/jimaging12060246.

[15] Y. Li, C. Yan, C. Zhao, et al., “Uni–MPedFormer: Towards a Unified Transformer-Based Framework for Multispectral Pedestrian Detection,” Journal of King Saud University Computer and Information Sciences, vol. 38, Art. no. 545, 2026. doi: 10.1007/s44443-026-00951-9.

[16] D. Yang, X. Zhang, and P. Wang, “FID-YOLO: A Pedestrian Detection Model Integrating Multispectral Information in Complex Environments,” PLOS ONE, vol. 21, no. 3, Art. no. e0342054, 2026. doi: 10.1371/journal.pone.0342054.

[17] S. Shin, C. Lee, G.-M. Park, and J. U. Kim, “SSMPD: Semi-Supervised Learning for Multispectral Pedestrian Detection,” IEEE Transactions on Multimedia, vol. 28, pp. 1806–1819, 2026. doi: 10.1109/TMM.2025.3645626.

[18] C. Li, Q. Qiang, and S. Zhou, “CMT-Det: A Cross-Modality Transformer Method for Road Scene Object Detection,” Measurement, vol. 290, Part D, Art. no. 122980, 2026. doi: 10.1016/j.measurement.2026.122980.

[19] K. Zhou, L. Chen, and X. Cao, “Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems,” in Computer Vision – ECCV 2020, Lecture Notes in Computer Science, vol. 12375, pp. 787–803, 2020. doi: 10.1007/978-3-030-58523-5_46.

[20] S. Hwang, J. Park, N. Kim, Y. Choi, and I. S. Kweon, “Multispectral Pedestrian Detection: Benchmark Dataset and Baseline,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1037–1045. doi: 10.1109/CVPR.2015.7298706.

[21] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, “Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models,” arXiv:2606.03748, 2026. doi: 10.48550/arXiv.2606.03748.

[22] Ultralytics, “Ultralytics YOLO11 Documentation,” released 2024. Available: https://docs.ultralytics.com/models/yolo11/ (accessed 15 Nov. 2026).

[23] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[24] OpenCV, “OpenCV Documentation,” Open Source Computer Vision Library. Available: https://docs.opencv.org/ (accessed 15 Nov. 2026).
