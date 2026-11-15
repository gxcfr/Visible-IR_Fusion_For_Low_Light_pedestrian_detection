# Unified Validation Evaluation Protocol

## 1. Objectives & Scope
This protocol defines the standardized, harmonized evaluation benchmark executed during Phase 2 of the research audit. It replaces heterogeneous evaluation pipelines with a single common ground-truth matching and coordinate transformation framework.

---

## 2. Benchmark Dataset & Ground Truth
- **Partition**: Scene-aware validation split containing exactly **1,803 image pairs** (Scenes 2, 5, 7, 25).
- **Official Test Partition**: Strictly preserved and **100% untouched** (3,463 pairs).
- **Ground Truth Source**: Raw XML annotations parsed directly from `C:\LLVIP\LLVIP\Annotations\`.
- **Target Coordinate System**: Original native **1280 × 1024** pixel coordinate system.
- **Classes**: Exactly 1 class (`person`, class ID 0).

---

## 3. Transformation & Unletterboxing Invariant
- **Model Input Size**: $640 \times 640$.
- **Letterbox Transform**: Downscales $1280 \times 1024$ by scale factor $0.5$ ($640 \times 512$) with symmetric padding of 64 pixels on top and bottom (`pad_x=0`, `pad_y=64`).
- **Coordinate Reversal**: All predicted bounding boxes are mapped back to $1280 \times 1024$ via Ultralytics `scale_boxes`:
  $$x_{\text{orig}} = \frac{x_{\text{box}} - \text{pad}_x}{\text{gain}_x}, \quad y_{\text{orig}} = \frac{y_{\text{box}} - \text{pad}_y}{\text{gain}_y}$$
- **WBF Aspect Ratio Handling**: In E004, predictions from E002 and E003 are normalized using their respective axes ($x / 1280.0$, $y / 1024.0$) before fusion, and unnormalized using the same factors, strictly avoiding square-distortion artifacts.

---

## 4. Detection Post-Processing & Metrics
- **Non-Maximum Suppression (NMS)**:
  - Confidence threshold: $\text{conf} = 0.001$ (standard evaluation threshold to extract full precision-recall curves).
  - IoU threshold: $\text{IoU}_{\text{nms}} = 0.65$.
  - Maximum detections: $\text{max\_det} = 300$.
- **IoU Matching**: Standard COCO-style 10 IoU thresholds $\text{IoU} \in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]$ evaluated using Ultralytics `match_predictions` (descending IoU assignment).
- **Reported Metric Policies**:
  1. $\text{mAP}_{50}$: Mean Average Precision at $\text{IoU} = 0.50$ via 101-point interpolation.
  2. $\text{mAP}_{50-95}$: Mean Average Precision averaged over all 10 IoU thresholds.
  3. Operating Point at standard $\text{conf} = 0.25$: Precision, Recall, and F1.
  4. Optimal Operating Point (Max F1): Peak F1 score, precision, recall, and optimal confidence threshold.
