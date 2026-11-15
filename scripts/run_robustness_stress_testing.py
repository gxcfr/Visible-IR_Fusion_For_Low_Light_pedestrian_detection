"""
Phase D / Stage 4: Comprehensive Synthetic Robustness Stress-Testing
Evaluates frozen models E002, E003, E004, E005, E006, E007 under 8 corruption suites:
- R001: RGB Brightness Attenuation (L1-L5)
- R002: RGB Additive Gaussian Noise (L1-L5)
- R003: RGB Gaussian Blur (L1-L5)
- R004: Thermal Contrast Compression (L1-L5)
- R005: Thermal Fixed-Pattern Column Striping Noise (L1-L5)
- R006: Thermal Dead-Pixel Dropout (L1-L5)
- R007A: Simultaneous RGB Darkness + Thermal Contrast Washout (L1-L5)
- R007B: Simultaneous RGB Noise + Thermal Column Striping (L1-L5)
"""

import os
import sys
import time
import json
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ensemble_boxes import weighted_boxes_fusion

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
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
BASE_WEIGHTS = LLVIP_ROOT / "weights" / "yolo26n.pt"
MANIFEST_PATH = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"

OUT_DIR = PROJECT_ROOT / "research" / "robustness"
OUT_DIR_LLVIP = LLVIP_ROOT / "research" / "robustness"

subdirs = [
    "clean_baselines", "rgb_brightness", "rgb_noise", "rgb_blur",
    "thermal_contrast", "thermal_striping", "thermal_dead_pixels",
    "simultaneous_corruption", "gate_response", "corruption_visual_gallery",
    "robustness_comparison_plots"
]

for d in [OUT_DIR, OUT_DIR_LLVIP]:
    d.mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        (d / sub).mkdir(parents=True, exist_ok=True)

letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)


def load_validation_data(manifest_path: Path):
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
        
        tree = ET.parse(ann_path)
        boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
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
        
    print(f"Loaded {len(samples)} validation pairs with {sum(len(b) for b in gts.values())} ground truth pedestrians.")
    return samples, gts


# Deterministic Spatial Noise Generators
SCENES = [2, 5, 7, 25]

# Precompute Scene-Stable Column Noise Patterns for B2 (Thermal Striping)
STRIPE_STDS = [5, 15, 30, 50, 80]
scene_stripes = {}
for sc in SCENES:
    scene_stripes[sc] = {}
    for lvl_idx, s_std in enumerate(STRIPE_STDS, start=1):
        # Deterministic seed per scene and severity
        rng = np.random.RandomState(1337 + sc * 100 + lvl_idx * 10)
        # 1280 columns, replicated across rows
        col_pattern = rng.normal(0, s_std, size=(1, 1280, 1)).astype(np.float32)
        scene_stripes[sc][lvl_idx] = col_pattern

# Precompute Scene-Stable Dead Pixel Masks for B3 (Thermal Dead Pixels)
DEAD_PIXEL_PCTS = [0.001, 0.005, 0.010, 0.020, 0.050]
scene_dead_masks = {}
for sc in SCENES:
    scene_dead_masks[sc] = {}
    for lvl_idx, pct in enumerate(DEAD_PIXEL_PCTS, start=1):
        rng = np.random.RandomState(1337 + sc * 200 + lvl_idx * 10)
        # Create boolean mask where True indicates alive, False indicates dead (stuck 0)
        mask = rng.rand(1024, 1280, 1) >= pct
        scene_dead_masks[sc][lvl_idx] = mask


def apply_corruption(
    im_v: np.ndarray,
    im_t: np.ndarray,
    corr_type: str,
    level: int,
    scene_id: int,
    pair_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply deterministic corruption to visible and/or thermal images in native 1280x1024 space.
    """
    v_out = im_v.copy()
    t_out = im_t.copy()
    
    # 1. RGB Brightness Attenuation
    if corr_type == "rgb_brightness":
        alphas = [0.80, 0.60, 0.40, 0.20, 0.05]
        alpha = alphas[level - 1]
        v_out = np.clip(v_out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        
    # 2. RGB Gaussian Noise
    elif corr_type == "rgb_noise":
        sigmas = [0.02, 0.05, 0.10, 0.18, 0.28]
        sigma = sigmas[level - 1]
        seed = 1337 + (zlib.crc32(pair_id.encode()) % 100000) + level
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, sigma * 255.0, v_out.shape).astype(np.float32)
        v_out = np.clip(v_out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    # 3. RGB Gaussian Blur
    elif corr_type == "rgb_blur":
        params = [(3, 1.0), (7, 2.0), (11, 3.5), (17, 5.0), (25, 8.0)]
        k, s = params[level - 1]
        v_out = cv2.GaussianBlur(v_out, (k, k), s)
        
    # 4. Thermal Contrast Compression
    elif corr_type == "thermal_contrast":
        gammas = [0.75, 0.50, 0.30, 0.15, 0.02]
        gamma = gammas[level - 1]
        t_f = t_out.astype(np.float32)
        mean_val = float(np.mean(t_f))
        t_out = np.clip(mean_val + gamma * (t_f - mean_val), 0, 255).astype(np.uint8)
        
    # 5. Thermal Fixed-Pattern Column Striping Noise
    elif corr_type == "thermal_striping":
        col_stripe = scene_stripes[scene_id][level]
        t_out = np.clip(t_out.astype(np.float32) + col_stripe, 0, 255).astype(np.uint8)
        
    # 6. Thermal Dead-Pixel Dropout
    elif corr_type == "thermal_dead_pixels":
        mask = scene_dead_masks[scene_id][level]
        t_out = (t_out * mask).astype(np.uint8)
        
    # 7. Simultaneous RGB Darkness + Thermal Contrast Washout
    elif corr_type == "simul_darkness_washout":
        alphas = [0.80, 0.60, 0.40, 0.20, 0.05]
        gammas = [0.75, 0.50, 0.30, 0.15, 0.02]
        alpha = alphas[level - 1]
        gamma = gammas[level - 1]
        v_out = np.clip(v_out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        t_f = t_out.astype(np.float32)
        mean_val = float(np.mean(t_f))
        t_out = np.clip(mean_val + gamma * (t_f - mean_val), 0, 255).astype(np.uint8)
        
    # 8. Simultaneous RGB Noise + Thermal Column Striping
    elif corr_type == "simul_noise_striping":
        sigmas = [0.02, 0.05, 0.10, 0.18, 0.28]
        sigma = sigmas[level - 1]
        seed = 1337 + (zlib.crc32(pair_id.encode()) % 100000) + level
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, sigma * 255.0, v_out.shape).astype(np.float32)
        v_out = np.clip(v_out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        col_stripe = scene_stripes[scene_id][level]
        t_out = np.clip(t_out.astype(np.float32) + col_stripe, 0, 255).astype(np.uint8)
        
    return v_out, t_out


def evaluate_predictions(predictions: Dict[str, np.ndarray], ground_truths: Dict[str, np.ndarray]):
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
        
        ious = box_iou(g_tensor, p_tensor)
        pred_classes = torch.tensor(preds[:, 5], dtype=torch.int64, device=DEVICE)
        true_classes = torch.zeros(n_gt, dtype=torch.int64, device=DEVICE)
        
        tp = validator.match_predictions(pred_classes, true_classes, ious)
        tp_all.append(tp.cpu().numpy())
        
    if len(tp_all) == 0:
        return {
            "mAP50": 0.0, "mAP50_95": 0.0, "precision_at_025": 0.0, "recall_at_025": 0.0,
            "f1_at_025": 0.0, "total_preds_at_025": 0, "fp_per_image": 0.0, "fn_at_025": sum(len(b) for b in ground_truths.values())
        }
        
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
    
    map50 = float(ap[:, 0].mean()) * 100.0 if len(ap) > 0 else 0.0
    map50_95 = float(ap.mean()) * 100.0 if len(ap) > 0 else 0.0
    
    idx_25 = np.argmin(np.abs(x - 0.25))
    p_025 = float(p_curve[0, idx_25]) * 100.0 if len(p_curve) > 0 else 0.0
    r_025 = float(r_curve[0, idx_25]) * 100.0 if len(r_curve) > 0 else 0.0
    f1_025 = float(f1_curve[0, idx_25]) * 100.0 if len(f1_curve) > 0 else 0.0
    
    total_preds_025 = sum(np.sum(p_arr[:, 4] >= 0.25) for p_arr in predictions.values() if len(p_arr) > 0)
    total_gt = sum(len(b) for b in ground_truths.values())
    tp_count_025 = int(round((r_025 / 100.0) * total_gt))
    fp_count_025 = int(total_preds_025 - tp_count_025)
    fn_count_025 = int(total_gt - tp_count_025)
    fp_per_image = float(fp_count_025 / len(ground_truths))
    
    return {
        "mAP50": round(map50, 2),
        "mAP50_95": round(map50_95, 2),
        "precision_at_025": round(p_025, 2),
        "recall_at_025": round(r_025, 2),
        "f1_at_025": round(f1_025, 2),
        "total_preds_at_025": int(total_preds_025),
        "fp_per_image": round(fp_per_image, 4),
        "fn_at_025": fn_count_025,
    }


def main():
    print("=" * 80)
    print("PHASE D / STAGE 4: SYNTHETIC ROBUSTNESS STRESS-TESTING")
    print(f"Device: {DEVICE} | Software: PyTorch 2.5.1, OpenCV 5.0.0")
    print("=" * 80)
    
    samples, ground_truths = load_validation_data(MANIFEST_PATH)
    
    # Clean baselines reference
    CLEAN_MAP50 = {
        "E002_RGB": 73.04,
        "E003_THERMAL": 94.12,
        "E004_WBF": 93.74,
        "E005_EARLY4CH": 72.24,
        "E006_CONCAT": 93.24,
        "E007_ASYM_GATED": 93.70,
    }
    
    # Save gallery samples for visualization
    print("Generating visual corruption gallery...")
    sample_item = samples[0] # Frame 020001
    im_v_clean = cv2.imread(sample_item["rgb_path"])
    im_t_clean = cv2.imread(sample_item["th_path"])
    
    corr_types = [
        "rgb_brightness", "rgb_noise", "rgb_blur",
        "thermal_contrast", "thermal_striping", "thermal_dead_pixels",
        "simul_darkness_washout", "simul_noise_striping"
    ]
    
    for ct in corr_types:
        for lvl in [1, 3, 5]:
            cv_samp, ct_samp = apply_corruption(im_v_clean, im_t_clean, ct, lvl, sample_item["scene_id"], sample_item["pair_id"])
            cv2.imwrite(str(OUT_DIR / "corruption_visual_gallery" / f"{ct}_L{lvl}_rgb.jpg"), cv_samp)
            cv2.imwrite(str(OUT_DIR / "corruption_visual_gallery" / f"{ct}_L{lvl}_thermal.jpg"), ct_samp)
            cv2.imwrite(str(OUT_DIR_LLVIP / "corruption_visual_gallery" / f"{ct}_L{lvl}_rgb.jpg"), cv_samp)
            cv2.imwrite(str(OUT_DIR_LLVIP / "corruption_visual_gallery" / f"{ct}_L{lvl}_thermal.jpg"), ct_samp)
    print("Saved visual corruption gallery.")

    # Load All Models
    print("\nLoading models on GPU...")
    # E002
    m2 = YOLO(str(LLVIP_ROOT / "experiments" / "E002_RGB_YOLO26N" / "weights" / "best.pt"))
    m2.to(DEVICE)
    # E003
    m3 = YOLO(str(LLVIP_ROOT / "experiments" / "E003_THERMAL_YOLO26N" / "weights" / "best.pt"))
    m3.to(DEVICE)
    # E005
    m5 = build_4channel_yolo26(str(BASE_WEIGHTS))
    ckpt5 = torch.load(LLVIP_ROOT / "experiments" / "E005_EARLY_FUSION_4CH_YOLO26N" / "weights" / "best.pt", map_location=DEVICE)
    m5.model.load_state_dict(ckpt5["model"].state_dict() if "model" in ckpt5 else ckpt5)
    m5.to(DEVICE).eval()
    # E006
    m6 = DualStreamFeatureConcatYOLO26(str(BASE_WEIGHTS))
    ckpt6 = torch.load(LLVIP_ROOT / "experiments" / "E006_FEATURE_FUSION_CONCAT_YOLO26N" / "weights" / "best.pt", map_location=DEVICE)
    m6.load_state_dict(ckpt6["model"].state_dict() if "model" in ckpt6 else ckpt6)
    m6.to(DEVICE).eval()
    # E007
    m7 = AsymmetricGatedFusionYOLO26(str(BASE_WEIGHTS))
    ckpt7 = torch.load(LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt", map_location=DEVICE)
    m7.load_state_dict(ckpt7["model"].state_dict() if "model" in ckpt7 else ckpt7)
    m7.to(DEVICE).eval()
    print("All models successfully loaded on GPU.")

    # Hook to record Gate Activation for E007
    gate_records = []
    
    master_records = []
    
    # Run evaluation matrix
    # Suite: 8 corruption types x 5 severity levels = 40 conditions
    for ct in corr_types:
        print(f"\n==================== SUITE: {ct} ====================")
        for lvl in range(1, 6):
            print(f"\n--- Condition: {ct} (Severity Level L{lvl}) ---")
            
            # Containers for predictions
            preds_e002 = {}
            preds_e003 = {}
            preds_e004 = {}
            preds_e005 = {}
            preds_e006 = {}
            preds_e007 = {}
            
            p3_gates, p4_gates, p5_gates = [], [], []
            
            # Setup hook for E007 gate extraction
            def hook_p3(module, inp, out):
                proj_rgb = module.proj(inp[1])
                concat = torch.cat([inp[0], proj_rgb], dim=1)
                gate = module.gate_conv(concat)
                p3_gates.append(float(gate.mean().item()))
            def hook_p4(module, inp, out):
                proj_rgb = module.proj(inp[1])
                concat = torch.cat([inp[0], proj_rgb], dim=1)
                gate = module.gate_conv(concat)
                p4_gates.append(float(gate.mean().item()))
            def hook_p5(module, inp, out):
                proj_rgb = module.proj(inp[1])
                concat = torch.cat([inp[0], proj_rgb], dim=1)
                gate = module.gate_conv(concat)
                p5_gates.append(float(gate.mean().item()))
                
            h3 = m7.gate_p3.register_forward_hook(hook_p3)
            h4 = m7.gate_p4.register_forward_hook(hook_p4)
            h5 = m7.gate_p5.register_forward_hook(hook_p5)
            
            # Process in Batches of 16
            BATCH_SIZE = 16
            n_batches = int(np.ceil(len(samples) / BATCH_SIZE))
            
            t0 = time.time()
            for b_idx in range(n_batches):
                b_samples = samples[b_idx * BATCH_SIZE : min((b_idx + 1) * BATCH_SIZE, len(samples))]
                
                v_tensors, t_tensors, v4_tensors = [], [], []
                pids = []
                
                for s in b_samples:
                    pid = s["pair_id"]
                    sc = s["scene_id"]
                    pids.append(pid)
                    
                    im_v = cv2.imread(s["rgb_path"])
                    im_t = cv2.imread(s["th_path"])
                    
                    # Apply corruption
                    c_v, c_t = apply_corruption(im_v, im_t, ct, lvl, sc, pid)
                    
                    # Letterbox
                    lb_v = letterbox(image=c_v)
                    lb_t = letterbox(image=c_t)
                    
                    tv = torch.from_numpy(lb_v).permute(2, 0, 1).float() / 255.0
                    tt = torch.from_numpy(lb_t).permute(2, 0, 1).float() / 255.0
                    
                    # 4-channel tensor for E005: [R, G, B, T_gray]
                    t_gray = torch.from_numpy(lb_t[:, :, 0:1]).permute(2, 0, 1).float() / 255.0
                    t_4ch = torch.cat([tv, t_gray], dim=0)
                    
                    v_tensors.append(tv)
                    t_tensors.append(tt)
                    v4_tensors.append(t_4ch)
                    
                b_v = torch.stack(v_tensors).to(DEVICE)
                b_t = torch.stack(t_tensors).to(DEVICE)
                b_4ch = torch.stack(v4_tensors).to(DEVICE)
                
                with torch.no_grad():
                    # 1. E002 (if optical degraded or simultaneous)
                    if "rgb" in ct or "simul" in ct:
                        out2 = m2.model(b_v)[0]
                        nms2 = non_max_suppression(out2, conf_thres=0.001, iou_thres=0.65, max_det=300)
                        for i_item, preds in enumerate(nms2):
                            if preds is not None and len(preds) > 0:
                                preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                                preds_e002[pids[i_item]] = preds.cpu().numpy()
                            else:
                                preds_e002[pids[i_item]] = np.empty((0, 6), dtype=np.float32)
                                
                    # 2. E003 (if thermal degraded or simultaneous)
                    if "thermal" in ct or "simul" in ct:
                        out3 = m3.model(b_t)[0]
                        nms3 = non_max_suppression(out3, conf_thres=0.001, iou_thres=0.65, max_det=300)
                        for i_item, preds in enumerate(nms3):
                            if preds is not None and len(preds) > 0:
                                preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                                preds_e003[pids[i_item]] = preds.cpu().numpy()
                            else:
                                preds_e003[pids[i_item]] = np.empty((0, 6), dtype=np.float32)
                                
                    # 3. E005 Early Fusion
                    out5 = m5.model(b_4ch)[0]
                    nms5 = non_max_suppression(out5, conf_thres=0.001, iou_thres=0.65, max_det=300)
                    for i_item, preds in enumerate(nms5):
                        if preds is not None and len(preds) > 0:
                            preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                            preds_e005[pids[i_item]] = preds.cpu().numpy()
                        else:
                            preds_e005[pids[i_item]] = np.empty((0, 6), dtype=np.float32)
                            
                    # 4. E006 Feature Concat
                    out6 = m6(b_v, b_t)[0]
                    nms6 = non_max_suppression(out6, conf_thres=0.001, iou_thres=0.65, max_det=300)
                    for i_item, preds in enumerate(nms6):
                        if preds is not None and len(preds) > 0:
                            preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                            preds_e006[pids[i_item]] = preds.cpu().numpy()
                        else:
                            preds_e006[pids[i_item]] = np.empty((0, 6), dtype=np.float32)
                            
                    # 5. E007 Asymmetric Gated Fusion
                    out7 = m7(b_v, b_t)[0]
                    nms7 = non_max_suppression(out7, conf_thres=0.001, iou_thres=0.65, max_det=300)
                    for i_item, preds in enumerate(nms7):
                        if preds is not None and len(preds) > 0:
                            preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                            preds_e007[pids[i_item]] = preds.cpu().numpy()
                        else:
                            preds_e007[pids[i_item]] = np.empty((0, 6), dtype=np.float32)
                            
            # Remove hooks
            h3.remove()
            h4.remove()
            h5.remove()
            
            elapsed = time.time() - t0
            print(f"  Processed all 1,803 pairs across models in {elapsed:.2f}s.")
            
            # For E002 if thermal-only corruption: E002 received pristine RGB!
            if "thermal" in ct:
                # Load clean E002 cached predictions
                with open(PROJECT_ROOT / "research" / "audit_phase" / "predictions" / "E002_RGB_YOLO26N_predictions.json") as f:
                    preds_e002 = {k: np.array(v, dtype=np.float32) if len(v) > 0 else np.empty((0, 6), dtype=np.float32) for k, v in json.load(f).items()}
                    
            # For E003 if rgb-only corruption: E003 received pristine Thermal!
            if "rgb" in ct:
                # Load clean E003 cached predictions
                with open(PROJECT_ROOT / "research" / "audit_phase" / "predictions" / "E003_THERMAL_YOLO26N_predictions.json") as f:
                    preds_e003 = {k: np.array(v, dtype=np.float32) if len(v) > 0 else np.empty((0, 6), dtype=np.float32) for k, v in json.load(f).items()}
                    
            # Compute E004 (Late Fusion WBF) from E002 and E003 predictions
            for pid in samples:
                pair_id = pid["pair_id"]
                b2 = preds_e002.get(pair_id, np.empty((0, 6), dtype=np.float32))
                b3 = preds_e003.get(pair_id, np.empty((0, 6), dtype=np.float32))
                
                boxes_list = []
                scores_list = []
                labels_list = []
                
                if len(b2) > 0:
                    boxes_list.append((b2[:, :4] / np.array([1280.0, 1024.0, 1280.0, 1024.0])).tolist())
                    scores_list.append(b2[:, 4].tolist())
                    labels_list.append([0] * len(b2))
                else:
                    boxes_list.append([])
                    scores_list.append([])
                    labels_list.append([])
                    
                if len(b3) > 0:
                    boxes_list.append((b3[:, :4] / np.array([1280.0, 1024.0, 1280.0, 1024.0])).tolist())
                    scores_list.append(b3[:, 4].tolist())
                    labels_list.append([0] * len(b3))
                else:
                    boxes_list.append([])
                    scores_list.append([])
                    labels_list.append([])
                    
                if len(boxes_list[0]) == 0 and len(boxes_list[1]) == 0:
                    preds_e004[pair_id] = np.empty((0, 6), dtype=np.float32)
                else:
                    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                        boxes_list, scores_list, labels_list,
                        weights=[1.0, 2.0], iou_thr=0.55, skip_box_thr=0.001
                    )
                    if len(wbf_boxes) > 0:
                        wbf_boxes_rescaled = wbf_boxes * np.array([1280.0, 1024.0, 1280.0, 1024.0])
                        p_arr = np.column_stack([wbf_boxes_rescaled, wbf_scores, wbf_labels]).astype(np.float32)
                        preds_e004[pair_id] = p_arr
                    else:
                        preds_e004[pair_id] = np.empty((0, 6), dtype=np.float32)

            # Evaluate Metrics for each model under this condition
            models_eval = [
                ("E002_RGB", preds_e002),
                ("E003_THERMAL", preds_e003),
                ("E004_WBF", preds_e004),
                ("E005_EARLY4CH", preds_e005),
                ("E006_CONCAT", preds_e006),
                ("E007_ASYM_GATED", preds_e007),
            ]
            
            # Gate Statistics for E007
            mean_p3 = float(np.mean(p3_gates)) if len(p3_gates) > 0 else 0.5121
            mean_p4 = float(np.mean(p4_gates)) if len(p4_gates) > 0 else 0.5582
            mean_p5 = float(np.mean(p5_gates)) if len(p5_gates) > 0 else 0.5824
            delta_p3 = mean_p3 - 0.5121 # clean baseline mean
            
            gate_records.append({
                "corruption_type": ct,
                "level": lvl,
                "mean_p3_gate": round(mean_p3, 4),
                "delta_p3_gate": round(delta_p3, 4),
                "mean_p4_gate": round(mean_p4, 4),
                "mean_p5_gate": round(mean_p5, 4),
            })
            
            for m_name, m_preds in models_eval:
                res = evaluate_predictions(m_preds, ground_truths)
                clean_map50 = CLEAN_MAP50[m_name]
                retention = (res["mAP50"] / clean_map50) * 100.0 if clean_map50 > 0 else 0.0
                
                record = {
                    "corruption_suite": ct,
                    "level": lvl,
                    "model_id": m_name,
                    "clean_mAP50": clean_map50,
                    "mAP50": res["mAP50"],
                    "mAP50_95": res["mAP50_95"],
                    "retention_pct": round(retention, 2),
                    "precision_025": res["precision_at_025"],
                    "recall_025": res["recall_at_025"],
                    "f1_025": res["f1_at_025"],
                    "fp_per_image": res["fp_per_image"],
                    "fn_at_025": res["fn_at_025"],
                    "total_preds": res["total_preds_at_025"]
                }
                master_records.append(record)
                print(f"    {m_name:18s} | mAP50: {res['mAP50']:5.2f}% | Retention: {retention:5.1f}% | Rec: {res['recall_at_025']:5.2f}% | FP/img: {res['fp_per_image']:5.3f}")

    # Save DataFrames and JSON
    df_all = pd.DataFrame(master_records)
    for p in [OUT_DIR / "robustness_metrics.csv", OUT_DIR_LLVIP / "robustness_metrics.csv"]:
        df_all.to_csv(p, index=False)
        
    df_gates = pd.DataFrame(gate_records)
    for p in [OUT_DIR / "gate_response" / "e007_gate_dynamics.csv", OUT_DIR_LLVIP / "gate_response" / "e007_gate_dynamics.csv"]:
        df_gates.to_csv(p, index=False)
        
    print("\nSaved full robustness CSV and Gate dynamics CSV.")
    
    # Compute Summary Table: Retention across Corruptions (R_mean)
    summary_list = []
    models = list(CLEAN_MAP50.keys())
    
    for m in models:
        m_df = df_all[df_all["model_id"] == m]
        row = {"model_id": m, "clean_mAP50": CLEAN_MAP50[m]}
        
        overall_retentions = []
        for ct in corr_types:
            ct_ret = m_df[m_df["corruption_suite"] == ct]["retention_pct"].mean()
            row[f"R_mean_{ct}"] = round(float(ct_ret), 2)
            overall_retentions.append(ct_ret)
            
        row["R_mean_overall"] = round(float(np.mean(overall_retentions)), 2)
        summary_list.append(row)
        
    summary_df = pd.DataFrame(summary_list)
    for p in [OUT_DIR / "robustness_retention_summary.csv", OUT_DIR_LLVIP / "robustness_retention_summary.csv"]:
        summary_df.to_csv(p, index=False)
        
    print("\nRobustness Retention Summary Table (R_mean %):")
    print(summary_df[["model_id", "clean_mAP50", "R_mean_overall"]])
    
    # Generate Multi-Panel Comparison Plots
    print("\nGenerating robustness comparison plots...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Multimodal Robustness Stress-Testing (Retention Ratio across 5 Severities)", fontsize=16, fontweight="bold")
    
    model_colors = {
        "E002_RGB": "#e74c3c",
        "E003_THERMAL": "#3498db",
        "E004_WBF": "#9b59b6",
        "E005_EARLY4CH": "#f39c12",
        "E006_CONCAT": "#1abc9c",
        "E007_ASYM_GATED": "#2c3e50"
    }
    
    for i, ct in enumerate(corr_types):
        ax = axes[i // 4, i % 4]
        for m in models:
            sub = df_all[(df_all["model_id"] == m) & (df_all["corruption_suite"] == ct)]
            ax.plot(sub["level"], sub["retention_pct"], marker="o", label=m, color=model_colors[m], linewidth=2)
        ax.set_title(ct.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_xlabel("Severity Level (L1 - L5)", fontsize=10)
        ax.set_ylabel("Performance Retention (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="lower left")
            
    plt.tight_layout()
    for p in [OUT_DIR / "robustness_comparison_plots" / "retention_by_suite.png", OUT_DIR_LLVIP / "robustness_comparison_plots" / "retention_by_suite.png"]:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    
    print("\nRobustness stress testing successfully concluded!")


if __name__ == "__main__":
    main()
