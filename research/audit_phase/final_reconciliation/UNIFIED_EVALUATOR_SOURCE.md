# Unified Validation Evaluator: Implementation and Source Code

## 1. Overview and Environment Specification

This document provides the complete, unabridged source code used to generate the canonical validation metrics in:
- `research/audit_phase/unified_validation_metrics.csv`
- `research/audit_phase/unified_validation_metrics.json`

The evaluation pipeline was executed under identical software environments across all seven experiments (**E001 through E007**):

- **Python**: 3.11.9
- **PyTorch**: 2.5.1+cu124
- **Torchvision**: 0.20.1+cu124
- **Ultralytics**: 8.3.34
- **Ensemble-Boxes (WBF)**: 1.0.9
- **OpenCV**: 4.10.0
- **NumPy**: 1.26.4
- **Pandas**: 2.2.3
- **Device**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.4)
- **Target Partition**: LLVIP Validation Split (1,803 image pairs, 5,086 valid pedestrian bounding boxes)
- **Test Set Status**: Official LLVIP test partition (3,463 pairs) **strictly untouched and sealed**.

---

## 2. Complete Evaluation Source Code

The implementation is located at:
`LLVIP_MULTIMODAL_RESEARCH/scripts/audit_unified_evaluation.py`

```python
"""
Unified Validation Evaluation Framework for LLVIP Multimodal Research Audit
Harmonizes evaluation across all 7 experiments (E001 to E007) using identical
ground-truth annotations, letterbox reversals, NMS thresholds, and IoU matching
in the original 1280x1024 image coordinate system.
"""

import os
import sys
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ensemble_boxes import weighted_boxes_fusion

LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
sys.path.insert(0, str(LLVIP_ROOT))

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.models.yolo.detect.val import DetectionValidator

from src.models.baselines.early_fusion_4ch import build_4channel_yolo26
from src.models.fusion.feature_concat_yolo26 import DualStreamFeatureConcatYOLO26
from src.models.fusion.asymmetric_gated_fusion_yolo26 import AsymmetricGatedFusionYOLO26

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AUDIT_DIR = LLVIP_ROOT / "research" / "audit_phase"
AUDIT_DIR_ROOT = PROJECT_ROOT / "research" / "audit_phase"
PRED_DIR = AUDIT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
(AUDIT_DIR_ROOT / "predictions").mkdir(parents=True, exist_ok=True)

# Letterbox transformer: 1280x1024 -> 640x640 with padding
letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)


def load_validation_ground_truths(manifest_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Load the 1,803 validation pairs and parse original XML ground-truth annotations."""
    df = pd.read_csv(manifest_path)
    val_df = df[df["partition"] == "val"].reset_index(drop=True)
    
    samples = []
    gts = {}
    
    for _, row in val_df.iterrows():
        pair_id = str(row["pair_id"]).zfill(6)
        rgb_path = row["rgb_path"]
        th_path = row["infrared_path"]
        ann_path = row["annotation_path"]
        scene_id = int(row["scene_id"])
        
        # Parse XML bounding boxes in original 1280x1024 coordinate system
        tree = ET.parse(ann_path)
        boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            # Clip to image boundaries
            xmin = max(0.0, min(1280.0, xmin))
            ymin = max(0.0, min(1024.0, ymin))
            xmax = max(0.0, min(1280.0, xmax))
            ymax = max(0.0, min(1024.0, ymax))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                
        boxes_arr = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.empty((0, 4), dtype=np.float32)
        gts[pair_id] = boxes_arr
        
        samples.append({
            "pair_id": pair_id,
            "scene_id": scene_id,
            "rgb_path": rgb_path,
            "th_path": th_path,
            "ann_path": ann_path,
        })
        
    print(f"Loaded {len(samples)} validation pairs with ground truth annotations.")
    return samples, gts


def evaluate_model_predictions(
    exp_id: str,
    predictions: Dict[str, np.ndarray],
    ground_truths: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Standardized bounding-box evaluation using Ultralytics DetectionValidator.match_predictions
    and ap_per_class on original 1280x1024 coordinates.
    """
    validator = DetectionValidator()
    validator.iouv = torch.linspace(0.5, 0.95, 10).to(DEVICE)
    
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []
    
    for pair_id, gt_boxes in ground_truths.items():
        preds = predictions.get(pair_id, np.empty((0, 6), dtype=np.float32))
        n_gt = len(gt_boxes)
        n_p = len(preds)
        
        if n_gt > 0:
            target_cls_all.extend([0] * n_gt)
            
        if n_p == 0:
            continue
            
        conf_all.extend(preds[:, 4].tolist())
        pred_cls_all.extend(preds[:, 5].astype(int).tolist())
        
        if n_gt == 0:
            tp_all.append(np.zeros((n_p, 10), dtype=bool))
            continue
            
        p_tensor = torch.tensor(preds[:, :4], dtype=torch.float32, device=DEVICE)
        g_tensor = torch.tensor(gt_boxes, dtype=torch.float32, device=DEVICE)
        
        # Calculate pairwise IoU in original 1280x1024 coordinates
        ious = box_iou(g_tensor, p_tensor)  # [M, N]
        pred_classes = torch.tensor(preds[:, 5], dtype=torch.int64, device=DEVICE)
        true_classes = torch.zeros(n_gt, dtype=torch.int64, device=DEVICE)
        
        # Match using Ultralytics standard descending IoU assignment
        tp = validator.match_predictions(pred_classes, true_classes, ious)  # [N, 10]
        tp_all.append(tp.cpu().numpy())
        
    if len(tp_all) == 0:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "p_at_25": 0.0, "r_at_25": 0.0, "f1_at_25": 0.0}
        
    tp_cat = np.concatenate(tp_all, axis=0)
    conf_arr = np.array(conf_all)
    pred_cls_arr = np.array(pred_cls_all)
    target_cls_arr = np.array(target_cls_all)
    
    results = ap_per_class(
        tp=tp_cat,
        conf=conf_arr,
        pred_cls=pred_cls_arr,
        target_cls=target_cls_arr,
        plot=False,
    )
    tp, fp, p, r, f1, ap, u_cls, p_curve, r_curve, f1_curve, x, prec_val = results
    
    # 1. Standard COCO metrics
    map50 = float(ap[:, 0].mean()) if len(ap) > 0 else 0.0
    map50_95 = float(ap.mean()) if len(ap) > 0 else 0.0
    
    # 2. Operating Point at standard conf=0.25
    idx_25 = np.argmin(np.abs(x - 0.25))
    p_at_25 = float(p_curve[0, idx_25]) if len(p_curve) > 0 else 0.0
    r_at_25 = float(r_curve[0, idx_25]) if len(r_curve) > 0 else 0.0
    f1_at_25 = float(f1_curve[0, idx_25]) if len(f1_curve) > 0 else 0.0
    
    # 3. Peak F1 Operating Point
    best_f1_idx = int(np.argmax(f1_curve[0])) if len(f1_curve) > 0 else 0
    best_thresh = float(x[best_f1_idx])
    max_f1 = float(f1[0]) if len(f1) > 0 else 0.0
    p_at_max_f1 = float(p[0]) if len(p) > 0 else 0.0
    r_at_max_f1 = float(r[0]) if len(r) > 0 else 0.0
    
    return {
        "experiment_id": exp_id,
        "mAP50": round(map50 * 100, 2),
        "mAP50_95": round(map50_95 * 100, 2),
        "precision_at_conf025": round(p_at_25 * 100, 2),
        "recall_at_conf025": round(r_at_25 * 100, 2),
        "f1_at_conf025": round(f1_at_25 * 100, 2),
        "max_f1": round(max_f1 * 100, 2),
        "precision_at_max_f1": round(p_at_max_f1 * 100, 2),
        "recall_at_max_f1": round(r_at_max_f1 * 100, 2),
        "optimal_conf_threshold": round(best_thresh, 4),
        "total_detections_evaluated": int(len(conf_arr)),
        "total_ground_truth_pedestrians": int(len(target_cls_arr)),
    }


def run_unimodal_inference(model_path: Path, samples: List[Dict[str, Any]], modality: str) -> Dict[str, np.ndarray]:
    """Run inference for unimodal YOLO models, reversing letterbox to 1280x1024."""
    model = YOLO(str(model_path))
    model.model.to(DEVICE).eval()
    
    predictions = {}
    
    for s in samples:
        pair_id = s["pair_id"]
        img_path = s["rgb_path"] if modality == "rgb" else s["th_path"]
        img_raw = cv2.imread(img_path)
        orig_shape = img_raw.shape[:2]  # (1024, 1280)
        
        # Letterbox transform
        img_lb = letterbox(image=img_raw)
        t_in = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        with torch.no_grad():
            preds = model.model(t_in)
            
        # Standard Ultralytics non-max suppression at conf=0.001 to retain full PR curve
        nms_res = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.7, max_det=300)[0]
        
        if len(nms_res) > 0:
            boxes = nms_res[:, :4].clone()
            # Reverse letterbox transformation back to original (1024, 1280) space
            boxes = scale_boxes((640, 640), boxes, orig_shape)
            scores = nms_res[:, 4:5]
            classes = nms_res[:, 5:6]
            pred_arr = torch.cat([boxes, scores, classes], dim=1).cpu().numpy()
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        predictions[pair_id] = pred_arr
        
    return predictions


def run_late_fusion_wbf_inference(
    rgb_preds: Dict[str, np.ndarray],
    th_preds: Dict[str, np.ndarray],
    weights: Tuple[float, float] = (1.0, 1.5),
    iou_thr: float = 0.55
) -> Dict[str, np.ndarray]:
    """
    Fuse unimodal RGB (E002) and Thermal (E003) predictions using Weighted Boxes Fusion.
    Crucially converts to and from normalized space with TRUE original dimensions (1280x1024).
    """
    fused_predictions = {}
    
    for pair_id in rgb_preds.keys():
        p_rgb = rgb_preds[pair_id]
        p_th = th_preds[pair_id]
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        # Normalize coordinates using native 1280x1024 dimensions
        if len(p_rgb) > 0:
            norm_rgb_boxes = p_rgb[:, :4].copy()
            norm_rgb_boxes[:, [0, 2]] /= 1280.0
            norm_rgb_boxes[:, [1, 3]] /= 1024.0
            norm_rgb_boxes = np.clip(norm_rgb_boxes, 0.0, 1.0)
            boxes_list.append(norm_rgb_boxes.tolist())
            scores_list.append(p_rgb[:, 4].tolist())
            labels_list.append(p_rgb[:, 5].astype(int).tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            
        if len(p_th) > 0:
            norm_th_boxes = p_th[:, :4].copy()
            norm_th_boxes[:, [0, 2]] /= 1280.0
            norm_th_boxes[:, [1, 3]] /= 1024.0
            norm_th_boxes = np.clip(norm_th_boxes, 0.0, 1.0)
            boxes_list.append(norm_th_boxes.tolist())
            scores_list.append(p_th[:, 4].tolist())
            labels_list.append(p_th[:, 5].astype(int).tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            
        # Execute Weighted Boxes Fusion
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=list(weights), iou_thr=iou_thr, skip_box_thr=0.001
        )
        
        if len(f_boxes) > 0:
            # Restore to original 1280x1024 coordinates without aspect ratio distortion
            f_boxes[:, [0, 2]] *= 1280.0
            f_boxes[:, [1, 3]] *= 1024.0
            f_arr = np.column_stack([f_boxes, f_scores, f_labels])
        else:
            f_arr = np.empty((0, 6), dtype=np.float32)
            
        fused_predictions[pair_id] = f_arr
        
    return fused_predictions


def run_custom_multimodal_inference(
    exp_id: str,
    model_type: str,
    weights_path: Path,
    samples: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """Inference for E005 (Early), E006 (Concat), and E007 (Asymmetric Gated)."""
    base_weights = str(LLVIP_ROOT / "weights" / "yolo26n.pt")
    
    if model_type == "early":
        yolo_4ch = build_4channel_yolo26(base_weights)
        yolo_4ch.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model = yolo_4ch.model.to(DEVICE).eval()
    elif model_type == "concat":
        model = DualStreamFeatureConcatYOLO26(base_weights)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE).eval()
    elif model_type == "asymmetric":
        model = AsymmetricGatedFusionYOLO26(base_weights)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE).eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    predictions = {}
    
    for s in samples:
        pair_id = s["pair_id"]
        img_rgb_raw = cv2.imread(s["rgb_path"])
        img_th_raw = cv2.imread(s["th_path"])
        orig_shape = img_rgb_raw.shape[:2]
        
        img_rgb_lb = letterbox(image=img_rgb_raw)
        img_th_lb = letterbox(image=img_th_raw)
        
        t_rgb = torch.from_numpy(img_rgb_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_th = torch.from_numpy(img_th_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        with torch.no_grad():
            if model_type == "early":
                # Concat RGB + 1-channel thermal -> 4-channel input tensor
                t_4ch = torch.cat([t_rgb, t_th[:, 0:1, :, :]], dim=1)
                preds = model(t_4ch)
            else:
                preds = model(t_rgb, t_th)
                
        nms_res = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.7, max_det=300)[0]
        
        if len(nms_res) > 0:
            boxes = nms_res[:, :4].clone()
            boxes = scale_boxes((640, 640), boxes, orig_shape)
            scores = nms_res[:, 4:5]
            classes = nms_res[:, 5:6]
            pred_arr = torch.cat([boxes, scores, classes], dim=1).cpu().numpy()
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        predictions[pair_id] = pred_arr
        
    return predictions
```
