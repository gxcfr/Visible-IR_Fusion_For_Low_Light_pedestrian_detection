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
        
        # Load and letterbox
        img_raw = cv2.imread(img_path)
        img_lb = letterbox(image=img_raw)
        t_img = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        with torch.no_grad():
            preds = model.model(t_img)
            nms_res = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.65, max_det=300)[0]
            
        if len(nms_res) > 0:
            boxes = nms_res[:, :4].clone()
            # Reverse letterbox back to (1024, 1280)
            boxes_orig = scale_boxes((640, 640), boxes, (1024, 1280))
            conf = nms_res[:, 4:5]
            cls_id = nms_res[:, 5:6]
            pred_arr = torch.cat([boxes_orig, conf, cls_id], dim=1).cpu().numpy()
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        predictions[pair_id] = pred_arr
        
    return predictions


def run_wbf_fusion(
    preds_rgb: Dict[str, np.ndarray],
    preds_th: Dict[str, np.ndarray],
    w_th: float = 0.7,
    w_rgb: float = 0.3,
    iou_thr: float = 0.55
) -> Dict[str, np.ndarray]:
    """
    Weighted Boxes Fusion combining E002 (RGB) and E003 (Thermal) in original 1280x1024 space.
    Normalizes coordinates by respective dimensions (1280 for width, 1024 for height)
    to strictly avoid aspect ratio distortion during fusion.
    """
    fused_predictions = {}
    
    for pair_id in preds_rgb.keys():
        p_rgb = preds_rgb.get(pair_id, np.empty((0, 6), dtype=np.float32))
        p_th = preds_th.get(pair_id, np.empty((0, 6), dtype=np.float32))
        
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        # Normalize Thermal boxes: x by 1280, y by 1024
        if len(p_th) > 0:
            b_th_norm = p_th[:, :4].copy()
            b_th_norm[:, [0, 2]] /= 1280.0
            b_th_norm[:, [1, 3]] /= 1024.0
            b_th_norm = np.clip(b_th_norm, 0.0, 1.0)
            boxes_list.append(b_th_norm.tolist())
            scores_list.append(p_th[:, 4].tolist())
            labels_list.append(p_th[:, 5].astype(int).tolist())
            weights.append(w_th)
            
        # Normalize RGB boxes: x by 1280, y by 1024
        if len(p_rgb) > 0:
            b_rgb_norm = p_rgb[:, :4].copy()
            b_rgb_norm[:, [0, 2]] /= 1280.0
            b_rgb_norm[:, [1, 3]] /= 1024.0
            b_rgb_norm = np.clip(b_rgb_norm, 0.0, 1.0)
            boxes_list.append(b_rgb_norm.tolist())
            scores_list.append(p_rgb[:, 4].tolist())
            labels_list.append(p_rgb[:, 5].astype(int).tolist())
            weights.append(w_rgb)
            
        if len(boxes_list) == 0:
            fused_predictions[pair_id] = np.empty((0, 6), dtype=np.float32)
            continue
            
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=0.001
        )
        
        if len(f_boxes) > 0:
            # Unnormalize back to 1280x1024
            f_boxes[:, [0, 2]] *= 1280.0
            f_boxes[:, [1, 3]] *= 1024.0
            pred_arr = np.hstack([f_boxes, f_scores[:, None], f_labels[:, None]]).astype(np.float32)
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        fused_predictions[pair_id] = pred_arr
        
    return fused_predictions


def run_early_fusion_inference(model_path: Path, samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Run inference for E005 Early Fusion 4-Channel model."""
    m_e005 = build_4channel_yolo26(str(LLVIP_ROOT / "weights" / "yolo26n.pt"))
    m_e005.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m_e005.model.to(DEVICE).eval()
    
    predictions = {}
    
    for s in samples:
        pair_id = s["pair_id"]
        img_rgb = letterbox(image=cv2.imread(s["rgb_path"]))
        img_th = letterbox(image=cv2.imread(s["th_path"]))
        
        t_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_th = torch.from_numpy(img_th).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        t_4ch = torch.cat([t_rgb, t_th[:, 0:1, :, :]], dim=1)
        
        with torch.no_grad():
            preds = m_e005.model(t_4ch)
            nms_res = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.65, max_det=300)[0]
            
        if len(nms_res) > 0:
            boxes = nms_res[:, :4].clone()
            boxes_orig = scale_boxes((640, 640), boxes, (1024, 1280))
            pred_arr = torch.cat([boxes_orig, nms_res[:, 4:6]], dim=1).cpu().numpy()
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        predictions[pair_id] = pred_arr
        
    return predictions


def run_feature_fusion_inference(
    model_type: str,
    model_path: Path,
    samples: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """Run inference for E006 (Concat) and E007 (Asymmetric Gated Fusion) models."""
    base_weights = str(LLVIP_ROOT / "weights" / "yolo26n.pt")
    if model_type == "concat":
        model = DualStreamFeatureConcatYOLO26(base_weights)
    elif model_type == "asymmetric":
        model = AsymmetricGatedFusionYOLO26(base_weights)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    predictions = {}
    
    for s in samples:
        pair_id = s["pair_id"]
        img_rgb = letterbox(image=cv2.imread(s["rgb_path"]))
        img_th = letterbox(image=cv2.imread(s["th_path"]))
        
        t_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_th = torch.from_numpy(img_th).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        with torch.no_grad():
            preds = model(t_rgb, t_th)
            nms_res = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.65, max_det=300)[0]
            
        if len(nms_res) > 0:
            boxes = nms_res[:, :4].clone()
            boxes_orig = scale_boxes((640, 640), boxes, (1024, 1280))
            pred_arr = torch.cat([boxes_orig, nms_res[:, 4:6]], dim=1).cpu().numpy()
        else:
            pred_arr = np.empty((0, 6), dtype=np.float32)
            
        predictions[pair_id] = pred_arr
        
    return predictions


def save_predictions_json(exp_id: str, predictions: Dict[str, np.ndarray]):
    """Save serializable prediction dictionary."""
    serializable = {}
    for k, v in predictions.items():
        serializable[k] = v.tolist() if len(v) > 0 else []
        
    for p in [PRED_DIR / f"{exp_id}_predictions.json", AUDIT_DIR_ROOT / "predictions" / f"{exp_id}_predictions.json"]:
        with open(p, "w") as f:
            json.dump(serializable, f)


def main():
    print("================================================================================")
    print("STARTING UNIFIED VALIDATION EVALUATION AUDIT")
    print("================================================================================")
    manifest_path = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"
    samples, ground_truths = load_validation_ground_truths(manifest_path)
    
    checkpoints = {
        "E001_RGB_YOLO11N": LLVIP_ROOT / "experiments" / "E001_RGB_YOLO11N" / "weights" / "best.pt",
        "E002_RGB_YOLO26N": LLVIP_ROOT / "experiments" / "E002_RGB_YOLO26N" / "weights" / "best.pt",
        "E003_THERMAL_YOLO26N": LLVIP_ROOT / "experiments" / "E003_THERMAL_YOLO26N" / "weights" / "best.pt",
        "E005_EARLY_FUSION_4CH_YOLO26N": LLVIP_ROOT / "experiments" / "E005_EARLY_FUSION_4CH_YOLO26N" / "weights" / "best.pt",
        "E006_FEATURE_FUSION_CONCAT_YOLO26N": LLVIP_ROOT / "experiments" / "E006_FEATURE_FUSION_CONCAT_YOLO26N" / "weights" / "best.pt",
        "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N": LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt",
    }
    
    all_metrics = []
    
    # 1. E001 (YOLO11n RGB)
    print("\n[1/7] Evaluating E001 (YOLO11n RGB)...")
    preds_e001 = run_unimodal_inference(checkpoints["E001_RGB_YOLO11N"], samples, modality="rgb")
    save_predictions_json("E001_RGB_YOLO11N", preds_e001)
    m_e001 = evaluate_model_predictions("E001_RGB_YOLO11N", preds_e001, ground_truths)
    all_metrics.append(m_e001)
    print(f"--> E001: mAP50 = {m_e001['mAP50']:.2f}%, mAP50-95 = {m_e001['mAP50_95']:.2f}%, F1@0.25 = {m_e001['f1_at_conf025']:.2f}%")
    
    # 2. E002 (YOLO26n RGB)
    print("\n[2/7] Evaluating E002 (YOLO26n RGB)...")
    preds_e002 = run_unimodal_inference(checkpoints["E002_RGB_YOLO26N"], samples, modality="rgb")
    save_predictions_json("E002_RGB_YOLO26N", preds_e002)
    m_e002 = evaluate_model_predictions("E002_RGB_YOLO26N", preds_e002, ground_truths)
    all_metrics.append(m_e002)
    print(f"--> E002: mAP50 = {m_e002['mAP50']:.2f}%, mAP50-95 = {m_e002['mAP50_95']:.2f}%, F1@0.25 = {m_e002['f1_at_conf025']:.2f}%")
    
    # 3. E003 (YOLO26n Thermal)
    print("\n[3/7] Evaluating E003 (YOLO26n Thermal)...")
    preds_e003 = run_unimodal_inference(checkpoints["E003_THERMAL_YOLO26N"], samples, modality="thermal")
    save_predictions_json("E003_THERMAL_YOLO26N", preds_e003)
    m_e003 = evaluate_model_predictions("E003_THERMAL_YOLO26N", preds_e003, ground_truths)
    all_metrics.append(m_e003)
    print(f"--> E003: mAP50 = {m_e003['mAP50']:.2f}%, mAP50-95 = {m_e003['mAP50_95']:.2f}%, F1@0.25 = {m_e003['f1_at_conf025']:.2f}%")
    
    # 4. E004 (Late Fusion WBF)
    print("\n[4/7] Evaluating E004 (Late Fusion WBF with true 1280x1024 coordinates)...")
    preds_e004 = run_wbf_fusion(preds_e002, preds_e003, w_th=0.7, w_rgb=0.3, iou_thr=0.55)
    save_predictions_json("E004_LATE_FUSION_WBF_YOLO26N", preds_e004)
    m_e004 = evaluate_model_predictions("E004_LATE_FUSION_WBF_YOLO26N", preds_e004, ground_truths)
    all_metrics.append(m_e004)
    print(f"--> E004: mAP50 = {m_e004['mAP50']:.2f}%, mAP50-95 = {m_e004['mAP50_95']:.2f}%, F1@0.25 = {m_e004['f1_at_conf025']:.2f}%")
    
    # 5. E005 (Early Fusion 4CH)
    print("\n[5/7] Evaluating E005 (Early Fusion 4CH)...")
    preds_e005 = run_early_fusion_inference(checkpoints["E005_EARLY_FUSION_4CH_YOLO26N"], samples)
    save_predictions_json("E005_EARLY_FUSION_4CH_YOLO26N", preds_e005)
    m_e005 = evaluate_model_predictions("E005_EARLY_FUSION_4CH_YOLO26N", preds_e005, ground_truths)
    all_metrics.append(m_e005)
    print(f"--> E005: mAP50 = {m_e005['mAP50']:.2f}%, mAP50-95 = {m_e005['mAP50_95']:.2f}%, F1@0.25 = {m_e005['f1_at_conf025']:.2f}%")
    
    # 6. E006 (Feature Concatenation)
    print("\n[6/7] Evaluating E006 (Dual-Stream Feature Concatenation)...")
    preds_e006 = run_feature_fusion_inference("concat", checkpoints["E006_FEATURE_FUSION_CONCAT_YOLO26N"], samples)
    save_predictions_json("E006_FEATURE_FUSION_CONCAT_YOLO26N", preds_e006)
    m_e006 = evaluate_model_predictions("E006_FEATURE_FUSION_CONCAT_YOLO26N", preds_e006, ground_truths)
    all_metrics.append(m_e006)
    print(f"--> E006: mAP50 = {m_e006['mAP50']:.2f}%, mAP50-95 = {m_e006['mAP50_95']:.2f}%, F1@0.25 = {m_e006['f1_at_conf025']:.2f}%")
    
    # 7. E007 (Proposed Asymmetric Gated Fusion)
    print("\n[7/7] Evaluating E007 (Proposed Asymmetric Gated Fusion)...")
    preds_e007 = run_feature_fusion_inference("asymmetric", checkpoints["E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N"], samples)
    save_predictions_json("E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N", preds_e007)
    m_e007 = evaluate_model_predictions("E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N", preds_e007, ground_truths)
    all_metrics.append(m_e007)
    print(f"--> E007: mAP50 = {m_e007['mAP50']:.2f}%, mAP50-95 = {m_e007['mAP50_95']:.2f}%, F1@0.25 = {m_e007['f1_at_conf025']:.2f}%")
    
    # Save Consolidated Validation Results
    df_metrics = pd.DataFrame(all_metrics)
    csv_paths = [
        AUDIT_DIR / "unified_validation_metrics.csv",
        AUDIT_DIR_ROOT / "unified_validation_metrics.csv"
    ]
    for p in csv_paths:
        df_metrics.to_csv(p, index=False)
        
    json_paths = [
        AUDIT_DIR / "unified_validation_metrics.json",
        AUDIT_DIR_ROOT / "unified_validation_metrics.json"
    ]
    for p in json_paths:
        with open(p, "w") as f:
            json.dump(all_metrics, f, indent=4)
            
    # Write Evaluation Protocol Document
    protocol_md = """# Unified Validation Evaluation Protocol

## 1. Objectives & Scope
This protocol defines the standardized, harmonized evaluation benchmark executed during Phase 2 of the research audit. It replaces heterogeneous evaluation pipelines with a single common ground-truth matching and coordinate transformation framework.

---

## 2. Benchmark Dataset & Ground Truth
- **Partition**: Scene-aware validation split containing exactly **1,803 image pairs** (Scenes 2, 5, 7, 25).
- **Official Test Partition**: Strictly preserved and **100% untouched** (3,463 pairs).
- **Ground Truth Source**: Raw XML annotations parsed directly from `C:\\LLVIP\\LLVIP\\Annotations\\`.
- **Target Coordinate System**: Original native **1280 × 1024** pixel coordinate system.
- **Classes**: Exactly 1 class (`person`, class ID 0).

---

## 3. Transformation & Unletterboxing Invariant
- **Model Input Size**: $640 \\times 640$.
- **Letterbox Transform**: Downscales $1280 \\times 1024$ by scale factor $0.5$ ($640 \\times 512$) with symmetric padding of 64 pixels on top and bottom (`pad_x=0`, `pad_y=64`).
- **Coordinate Reversal**: All predicted bounding boxes are mapped back to $1280 \\times 1024$ via Ultralytics `scale_boxes`:
  $$x_{\\text{orig}} = \\frac{x_{\\text{box}} - \\text{pad}_x}{\\text{gain}_x}, \\quad y_{\\text{orig}} = \\frac{y_{\\text{box}} - \\text{pad}_y}{\\text{gain}_y}$$
- **WBF Aspect Ratio Handling**: In E004, predictions from E002 and E003 are normalized using their respective axes ($x / 1280.0$, $y / 1024.0$) before fusion, and unnormalized using the same factors, strictly avoiding square-distortion artifacts.

---

## 4. Detection Post-Processing & Metrics
- **Non-Maximum Suppression (NMS)**:
  - Confidence threshold: $\\text{conf} = 0.001$ (standard evaluation threshold to extract full precision-recall curves).
  - IoU threshold: $\\text{IoU}_{\\text{nms}} = 0.65$.
  - Maximum detections: $\\text{max\\_det} = 300$.
- **IoU Matching**: Standard COCO-style 10 IoU thresholds $\\text{IoU} \\in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]$ evaluated using Ultralytics `match_predictions` (descending IoU assignment).
- **Reported Metric Policies**:
  1. $\\text{mAP}_{50}$: Mean Average Precision at $\\text{IoU} = 0.50$ via 101-point interpolation.
  2. $\\text{mAP}_{50-95}$: Mean Average Precision averaged over all 10 IoU thresholds.
  3. Operating Point at standard $\\text{conf} = 0.25$: Precision, Recall, and F1.
  4. Optimal Operating Point (Max F1): Peak F1 score, precision, recall, and optimal confidence threshold.
"""
    for p in [AUDIT_DIR / "evaluation_protocol.md", AUDIT_DIR_ROOT / "evaluation_protocol.md"]:
        with open(p, "w") as f:
            f.write(protocol_md)
            
    print("\n================================================================================")
    print("UNIFIED VALIDATION EVALUATION COMPLETE!")
    print(f"Results saved to: {AUDIT_DIR / 'unified_validation_metrics.csv'}")
    print("================================================================================")

if __name__ == "__main__":
    main()
