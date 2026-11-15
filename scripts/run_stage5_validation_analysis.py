import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
MANIFEST_PATH = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"
PRED_DIR = PROJECT_ROOT / "research" / "audit_phase" / "predictions"
OUT_DIR = PROJECT_ROOT / "research" / "final_analysis" / "validation_analysis"
OUT_DIR_LLVIP = LLVIP_ROOT / "research" / "final_analysis" / "validation_analysis"

for d in [OUT_DIR, OUT_DIR_LLVIP, OUT_DIR / "qualitative_panels", OUT_DIR_LLVIP / "qualitative_panels"]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(LLVIP_ROOT))
from ultralytics.utils.metrics import ap_per_class

MODELS = [
    ("E001_RGB_YOLO11N", "E001_RGB_YOLO11N_predictions.json"),
    ("E002_RGB_YOLO26N", "E002_RGB_YOLO26N_predictions.json"),
    ("E003_THERMAL_YOLO26N", "E003_THERMAL_YOLO26N_predictions.json"),
    ("E004_LATE_FUSION_WBF_YOLO26N", "E004_LATE_FUSION_WBF_YOLO26N_predictions.json"),
    ("E005_EARLY_FUSION_4CH_YOLO26N", "E005_EARLY_FUSION_4CH_YOLO26N_predictions.json"),
    ("E006_FEATURE_FUSION_CONCAT_YOLO26N", "E006_FEATURE_FUSION_CONCAT_YOLO26N_predictions.json"),
    ("E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N", "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N_predictions.json"),
]

SCENES = [2, 5, 7, 25]

def load_val_annotations():
    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df["partition"] == "val"].reset_index(drop=True)
    gts = {}
    pair_scenes = {}
    pair_paths = {}

    for _, row in val_df.iterrows():
        pair_id = str(row["pair_id"]).zfill(6)
        sc = int(row["scene_id"])
        pair_scenes[pair_id] = sc
        pair_paths[pair_id] = {
            "rgb": row["rgb_path"],
            "th": row["infrared_path"],
            "ann": row["annotation_path"]
        }

        tree = ET.parse(row["annotation_path"])
        boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            xmin = max(0.0, min(1280.0, float(bnd.find("xmin").text)))
            ymin = max(0.0, min(1024.0, float(bnd.find("ymin").text)))
            xmax = max(0.0, min(1280.0, float(bnd.find("xmax").text)))
            ymax = max(0.0, min(1024.0, float(bnd.find("ymax").text)))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        gts[pair_id] = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.empty((0, 4), dtype=np.float32)

    return gts, pair_scenes, pair_paths

def evaluate_subset(preds_dict: Dict[str, np.ndarray], gts_dict: Dict[str, np.ndarray], subset_pids: List[str]):
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    conf_eval = 0.25

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    tp_at_025 = 0
    fp_at_025 = 0
    fn_at_025 = 0
    total_preds_at_025 = 0
    total_gts = 0

    fn_scales = {"small": 0, "medium": 0, "large": 0}
    fp_categories = {"sky_upper": 0, "ground_lower": 0, "mid_body": 0, "duplicate_split": 0}

    for pid in subset_pids:
        gt_boxes = gts_dict.get(pid, np.empty((0, 4), dtype=np.float32))
        n_gt = len(gt_boxes)
        total_gts += n_gt
        all_target_cls.extend([0] * n_gt)

        preds = preds_dict.get(pid, np.empty((0, 6), dtype=np.float32))
        if len(preds) == 0:
            if n_gt > 0:
                fn_at_025 += n_gt
                for gb in gt_boxes:
                    area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    if area < 32**2: fn_scales["small"] += 1
                    elif area <= 96**2: fn_scales["medium"] += 1
                    else: fn_scales["large"] += 1
            continue

        pred_boxes = preds[:, :4]
        pred_scores = preds[:, 4]
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        mask_025 = pred_scores >= conf_eval
        boxes_025 = pred_boxes[mask_025]
        scores_025 = pred_scores[mask_025]
        total_preds_at_025 += len(boxes_025)

        if n_gt == 0:
            fp_at_025 += len(boxes_025)
            all_tp.append(np.zeros((len(pred_boxes), 10), dtype=bool))
            all_conf.append(pred_scores)
            all_pred_cls.append(np.zeros(len(pred_boxes), dtype=np.int64))
            for pb in boxes_025:
                yc = (pb[1] + pb[3]) / 2.0
                if yc < 200: fp_categories["sky_upper"] += 1
                elif yc > 800: fp_categories["ground_lower"] += 1
                else: fp_categories["mid_body"] += 1
            continue

        gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
        gt_areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        p_x1, p_y1, p_x2, p_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        p_areas = (p_x2 - p_x1) * (p_y2 - p_y1)

        xx1 = np.maximum(p_x1[:, None], gt_x1[None, :])
        yy1 = np.maximum(p_y1[:, None], gt_y1[None, :])
        xx2 = np.minimum(p_x2[:, None], gt_x2[None, :])
        yy2 = np.minimum(p_y2[:, None], gt_y2[None, :])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = p_areas[:, None] + gt_areas[None, :] - inter
        iou_matrix = np.where(union > 0, inter / union, 0.0)

        tp_matrix = np.zeros((len(pred_boxes), 10), dtype=bool)
        for i_idx, iou_th in enumerate(iou_thresholds):
            matched_gts = set()
            for p_i in range(len(pred_boxes)):
                best_gt = -1
                best_iou = iou_th
                for g_i in range(n_gt):
                    if g_i not in matched_gts:
                        if iou_matrix[p_i, g_i] >= best_iou:
                            best_iou = iou_matrix[p_i, g_i]
                            best_gt = g_i
                if best_gt >= 0:
                    matched_gts.add(best_gt)
                    tp_matrix[p_i, i_idx] = True

        all_tp.append(tp_matrix)
        all_conf.append(pred_scores)
        all_pred_cls.append(np.zeros(len(pred_boxes), dtype=np.int64))

        if len(boxes_025) > 0:
            matched_025 = set()
            unmatched_p = []
            for p_i in range(len(boxes_025)):
                best_gt = -1
                best_iou = 0.50
                for g_i in range(n_gt):
                    if g_i not in matched_025:
                        if iou_matrix[p_i, g_i] >= best_iou:
                            best_iou = iou_matrix[p_i, g_i]
                            best_gt = g_i
                if best_gt >= 0:
                    matched_025.add(best_gt)
                else:
                    unmatched_p.append(boxes_025[p_i])

            cur_tp = len(matched_025)
            cur_fp = len(boxes_025) - cur_tp
            cur_fn = n_gt - cur_tp

            tp_at_025 += cur_tp
            fp_at_025 += cur_fp
            fn_at_025 += cur_fn

            # Classify FNs by scale
            for g_i in range(n_gt):
                if g_i not in matched_025:
                    area = gt_areas[g_i]
                    if area < 32**2: fn_scales["small"] += 1
                    elif area <= 96**2: fn_scales["medium"] += 1
                    else: fn_scales["large"] += 1

            # Classify FPs by location
            for pb in unmatched_p:
                yc = (pb[1] + pb[3]) / 2.0
                # Check if it overlaps with any GT (duplicate split)
                p_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                overlaps_gt = False
                for gb in gt_boxes:
                    xx = max(0, min(pb[2], gb[2]) - max(pb[0], gb[0]))
                    yy = max(0, min(pb[3], gb[3]) - max(pb[1], gb[1]))
                    if xx * yy > 0.1 * p_area:
                        overlaps_gt = True
                        break
                if overlaps_gt:
                    fp_categories["duplicate_split"] += 1
                elif yc < 200:
                    fp_categories["sky_upper"] += 1
                elif yc > 800:
                    fp_categories["ground_lower"] += 1
                else:
                    fp_categories["mid_body"] += 1
        else:
            fn_at_025 += n_gt
            for gb in gt_boxes:
                area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                if area < 32**2: fn_scales["small"] += 1
                elif area <= 96**2: fn_scales["medium"] += 1
                else: fn_scales["large"] += 1

    if len(all_tp) > 0:
        tp_cat = np.concatenate(all_tp, axis=0)
        conf_cat = np.concatenate(all_conf, axis=0)
        pred_cls_cat = np.concatenate(all_pred_cls, axis=0)
        target_cls_cat = np.array(all_target_cls, dtype=np.int64)

        results = ap_per_class(
            tp_cat, conf_cat, pred_cls_cat, target_cls_cat,
            plot=False, save_dir=None, names={0: "person"}
        )
        ap = results[5]
        map50 = float(ap[0, 0] * 100.0) if ap.shape[1] > 0 else 0.0
        map50_95 = float(np.mean(ap[0, :]) * 100.0) if ap.shape[1] > 0 else 0.0
    else:
        map50, map50_95 = 0.0, 0.0

    prec_025 = float(tp_at_025 / total_preds_at_025 * 100.0) if total_preds_at_025 > 0 else 0.0
    rec_025 = float(tp_at_025 / total_gts * 100.0) if total_gts > 0 else 0.0
    f1_025 = float(2 * (prec_025 * rec_025) / (prec_025 + rec_025)) if (prec_025 + rec_025) > 0 else 0.0
    fp_per_img = float(fp_at_025 / len(subset_pids)) if len(subset_pids) > 0 else 0.0

    return {
        "mAP50": round(map50, 2),
        "mAP50_95": round(map50_95, 2),
        "precision_at_025": round(prec_025, 2),
        "recall_at_025": round(rec_025, 2),
        "f1_at_025": round(f1_025, 2),
        "tp_at_025": tp_at_025,
        "fp_at_025": fp_at_025,
        "fn_at_025": fn_at_025,
        "total_preds_at_025": total_preds_at_025,
        "total_gts": total_gts,
        "fp_per_image": round(fp_per_img, 3),
        "fn_small": fn_scales["small"],
        "fn_med": fn_scales["medium"],
        "fn_large": fn_scales["large"],
        "fp_sky": fp_categories["sky_upper"],
        "fp_ground": fp_categories["ground_lower"],
        "fp_mid": fp_categories["mid_body"],
        "fp_split": fp_categories["duplicate_split"],
    }

def draw_boxes(im: np.ndarray, boxes, color=(0, 255, 0), thickness=2, label=""):
    im_out = im.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(im_out, (x1, y1), (x2, y2), color, thickness)
        if len(b) > 4:
            conf = b[4]
            cv2.putText(im_out, f"{conf:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return im_out

def generate_qualitative_panels(all_preds: Dict[str, Dict], gts: Dict, pair_paths: Dict):
    sample_pairs = [
        ("020001", "Scene 02: High Thermal Contrast, Deep Visible Darkness"),
        ("050001", "Scene 05: Streetlight Illumination & Distant Pedestrians"),
        ("070001", "Scene 07: Dense Occlusion & Thermal Silhouette Clutter"),
        ("250001", "Scene 25: Headlight Glare & Complementary Cross-Modal Recovery"),
    ]

    models_to_plot = [
        ("Ground Truth", None, (0, 255, 0)),
        ("E002 (RGB)", "E002_RGB_YOLO26N", (0, 0, 255)),
        ("E003 (Thermal)", "E003_THERMAL_YOLO26N", (255, 128, 0)),
        ("E004 (Late WBF)", "E004_LATE_FUSION_WBF_YOLO26N", (255, 0, 255)),
        ("E006 (Concat)", "E006_FEATURE_FUSION_CONCAT_YOLO26N", (0, 255, 255)),
        ("E007 (Asym Gated)", "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N", (0, 200, 100)),
    ]

    for pid, title in sample_pairs:
        rgb_path = pair_paths[pid]["rgb"]
        th_path = pair_paths[pid]["th"]
        im_v = cv2.imread(rgb_path)
        im_t = cv2.imread(th_path)
        gt_b = gts[pid]

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f"{title} (Pair ID: {pid})", fontsize=16, fontweight="bold")

        # 1. Raw Visible
        axes[0, 0].imshow(cv2.cvtColor(im_v, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Visible Input (RGB)", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Raw Thermal
        axes[0, 1].imshow(cv2.cvtColor(im_t, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Thermal Input (IR)", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. Ground Truth overlaid on Thermal
        gt_vis = draw_boxes(im_t, gt_b, color=(0, 255, 0), thickness=3)
        axes[0, 2].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"Ground Truth ({len(gt_b)} pedestrians)", fontsize=12, fontweight="bold")
        axes[0, 2].axis("off")

        # 4. E002 RGB
        p2 = all_preds["E002_RGB_YOLO26N"].get(pid, np.empty((0, 6)))
        p2_filt = p2[p2[:, 4] >= 0.25] if len(p2) > 0 else []
        im2_vis = draw_boxes(im_v, p2_filt, color=(0, 0, 255), thickness=2)
        axes[0, 3].imshow(cv2.cvtColor(im2_vis, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title(f"E002 RGB (Detections: {len(p2_filt)})", fontsize=12, fontweight="bold")
        axes[0, 3].axis("off")

        # 5. E003 Thermal
        p3 = all_preds["E003_THERMAL_YOLO26N"].get(pid, np.empty((0, 6)))
        p3_filt = p3[p3[:, 4] >= 0.25] if len(p3) > 0 else []
        im3_vis = draw_boxes(im_t, p3_filt, color=(255, 128, 0), thickness=2)
        axes[1, 0].imshow(cv2.cvtColor(im3_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"E003 Thermal (Detections: {len(p3_filt)})", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")

        # 6. E004 Late WBF
        p4 = all_preds["E004_LATE_FUSION_WBF_YOLO26N"].get(pid, np.empty((0, 6)))
        p4_filt = p4[p4[:, 4] >= 0.25] if len(p4) > 0 else []
        im4_vis = draw_boxes(im_t, p4_filt, color=(255, 0, 255), thickness=2)
        axes[1, 1].imshow(cv2.cvtColor(im4_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"E004 Late WBF (Detections: {len(p4_filt)})", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")

        # 7. E006 Concat
        p6 = all_preds["E006_FEATURE_FUSION_CONCAT_YOLO26N"].get(pid, np.empty((0, 6)))
        p6_filt = p6[p6[:, 4] >= 0.25] if len(p6) > 0 else []
        im6_vis = draw_boxes(im_t, p6_filt, color=(0, 255, 255), thickness=2)
        axes[1, 2].imshow(cv2.cvtColor(im6_vis, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f"E006 Feature Concat (Detections: {len(p6_filt)})", fontsize=12, fontweight="bold")
        axes[1, 2].axis("off")

        # 8. E007 Proposed Asymmetric Gated Fusion
        p7 = all_preds["E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N"].get(pid, np.empty((0, 6)))
        p7_filt = p7[p7[:, 4] >= 0.25] if len(p7) > 0 else []
        im7_vis = draw_boxes(im_t, p7_filt, color=(0, 200, 100), thickness=2)
        axes[1, 3].imshow(cv2.cvtColor(im7_vis, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title(f"E007 Asym Gated (Detections: {len(p7_filt)})", fontsize=12, fontweight="bold")
        axes[1, 3].axis("off")

        plt.tight_layout()
        save_path = OUT_DIR / "qualitative_panels" / f"comparison_pair_{pid}.png"
        save_path_llvip = OUT_DIR_LLVIP / "qualitative_panels" / f"comparison_pair_{pid}.png"
        fig.savefig(save_path, dpi=300)
        fig.savefig(save_path_llvip, dpi=300)
        plt.close(fig)
        print(f"Saved qualitative comparison panel: {save_path.name}")

def main():
    gts, pair_scenes, pair_paths = load_val_annotations()
    print(f"Loaded {len(gts)} validation pairs.")

    all_preds = {}
    for m_id, f_name in MODELS:
        f_path = PRED_DIR / f_name
        with open(f_path, "r") as f:
            raw = json.load(f)
            preds = {k: np.array(v, dtype=np.float32) if len(v) > 0 else np.empty((0, 6), dtype=np.float32) for k, v in raw.items()}
        all_preds[m_id] = preds
        print(f"Loaded {len(preds)} predictions for {m_id}.")

    # Generate Qualitative Visual Panels
    if not (OUT_DIR / "qualitative_panels" / "comparison_pair_250001.png").exists():
        print("\nGenerating qualitative multi-scene comparison panels...")
        generate_qualitative_panels(all_preds, gts, pair_paths)
    else:
        print("\nQualitative panels already exist, skipping generation.")

    # Compute Per-Scene & Overall Metrics
    master_records = []
    error_records = []

    scene_subsets = {sc: [p for p, s in pair_scenes.items() if s == sc] for sc in SCENES}
    scene_subsets["overall"] = list(gts.keys())

    for m_id, _ in MODELS:
        preds = all_preds[m_id]
        for sc_key, pids in scene_subsets.items():
            sc_str = f"Scene_{str(sc_key).zfill(2)}" if sc_key != "overall" else "Overall_Val"
            metrics = evaluate_subset(preds, gts, pids)

            master_records.append({
                "model_id": m_id,
                "partition_scene": sc_str,
                "frame_count": len(pids),
                "total_gts": metrics["total_gts"],
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "precision_025": metrics["precision_at_025"],
                "recall_025": metrics["recall_at_025"],
                "f1_025": metrics["f1_at_025"],
                "tp_025": metrics["tp_at_025"],
                "fp_025": metrics["fp_at_025"],
                "fn_025": metrics["fn_at_025"],
                "fp_per_image": metrics["fp_per_image"],
                "total_preds_025": metrics["total_preds_at_025"]
            })

            error_records.append({
                "model_id": m_id,
                "partition_scene": sc_str,
                "fn_small": metrics["fn_small"],
                "fn_medium": metrics["fn_med"],
                "fn_large": metrics["fn_large"],
                "fp_sky_upper": metrics["fp_sky"],
                "fp_ground_lower": metrics["fp_ground"],
                "fp_mid_body": metrics["fp_mid"],
                "fp_duplicate_split": metrics["fp_split"]
            })

    df_metrics = pd.DataFrame(master_records)
    for p in [OUT_DIR / "per_scene_metrics.csv", OUT_DIR_LLVIP / "per_scene_metrics.csv"]:
        df_metrics.to_csv(p, index=False)
        print(f"Saved: {p}")

    df_errors = pd.DataFrame(error_records)
    for p in [OUT_DIR / "error_taxonomy_breakdown.csv", OUT_DIR_LLVIP / "error_taxonomy_breakdown.csv"]:
        df_errors.to_csv(p, index=False)
        print(f"Saved: {p}")

    print("\nStage 5 Validation Analysis successfully concluded!")

if __name__ == "__main__":
    main()
