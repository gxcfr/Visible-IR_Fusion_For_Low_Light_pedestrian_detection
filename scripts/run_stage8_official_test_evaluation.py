import os
import sys
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
sys.path.insert(0, str(LLVIP_ROOT))

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class

from src.models.baselines.early_fusion_4ch import build_4channel_yolo26
from src.models.fusion.feature_concat_yolo26 import DualStreamFeatureConcatYOLO26
from src.models.fusion.asymmetric_gated_fusion_yolo26 import AsymmetricGatedFusionYOLO26

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_WEIGHTS = LLVIP_ROOT / "weights" / "yolo26n.pt"
MANIFEST_PATH = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"

OUT_DIR = PROJECT_ROOT / "research" / "final_analysis" / "test_evaluation"
OUT_DIR_LLVIP = LLVIP_ROOT / "research" / "final_analysis" / "test_evaluation"
PRED_DIR = OUT_DIR / "predictions"
PRED_DIR_LLVIP = OUT_DIR_LLVIP / "predictions"

for d in [OUT_DIR, OUT_DIR_LLVIP, PRED_DIR, PRED_DIR_LLVIP]:
    d.mkdir(parents=True, exist_ok=True)

letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)

def load_test_data():
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["partition"] == "test"].reset_index(drop=True)
    samples = []
    gts = {}
    pair_scenes = {}

    for _, row in test_df.iterrows():
        pair_id = str(row["pair_id"]).zfill(6)
        rgb_path = row["rgb_path"]
        th_path = row["infrared_path"]
        ann_path = row["annotation_path"]
        scene_id = int(row["scene_id"])
        pair_scenes[pair_id] = scene_id

        tree = ET.parse(ann_path)
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
        samples.append({
            "pair_id": pair_id,
            "scene_id": scene_id,
            "rgb_path": rgb_path,
            "th_path": th_path,
        })

    print(f"Loaded {len(samples)} test pairs with {sum(len(b) for b in gts.values())} ground truth pedestrians across {len(set(pair_scenes.values()))} scenes.")
    return samples, gts, pair_scenes

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

    for pid in subset_pids:
        gt_boxes = gts_dict.get(pid, np.empty((0, 4), dtype=np.float32))
        n_gt = len(gt_boxes)
        total_gts += n_gt
        all_target_cls.extend([0] * n_gt)

        preds = preds_dict.get(pid, np.empty((0, 6), dtype=np.float32))
        if len(preds) == 0:
            if n_gt > 0:
                fn_at_025 += n_gt
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
            cur_tp = 0
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
                    cur_tp += 1
            tp_at_025 += cur_tp
            fp_at_025 += (len(boxes_025) - cur_tp)
            fn_at_025 += (n_gt - cur_tp)
        else:
            fn_at_025 += n_gt

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
    }

def main():
    samples, gts, pair_scenes = load_test_data()
    all_pids = [s["pair_id"] for s in samples]

    print("\nInitializing frozen models on GPU...")
    # E001
    m1 = YOLO(str(LLVIP_ROOT / "experiments" / "E001_RGB_YOLO11N" / "weights" / "best.pt")).to(DEVICE)
    # E002
    m2 = YOLO(str(LLVIP_ROOT / "experiments" / "E002_RGB_YOLO26N" / "weights" / "best.pt")).to(DEVICE)
    # E003
    m3 = YOLO(str(LLVIP_ROOT / "experiments" / "E003_THERMAL_YOLO26N" / "weights" / "best.pt")).to(DEVICE)
    # E005
    m5 = build_4channel_yolo26(str(BASE_WEIGHTS))
    ckpt5 = torch.load(LLVIP_ROOT / "experiments" / "E005_EARLY_FUSION_4CH_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m5.model.load_state_dict(ckpt5["model"].state_dict() if "model" in ckpt5 else ckpt5)
    m5.to(DEVICE).eval()
    # E006
    m6 = DualStreamFeatureConcatYOLO26(str(BASE_WEIGHTS))
    ckpt6 = torch.load(LLVIP_ROOT / "experiments" / "E006_FEATURE_FUSION_CONCAT_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m6.load_state_dict(ckpt6["model"].state_dict() if "model" in ckpt6 else ckpt6)
    m6.to(DEVICE).eval()
    # E007
    m7 = AsymmetricGatedFusionYOLO26(str(BASE_WEIGHTS))
    ckpt7 = torch.load(LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m7.load_state_dict(ckpt7["model"].state_dict() if "model" in ckpt7 else ckpt7)
    m7.to(DEVICE).eval()
    print("All frozen checkpoints successfully loaded on GPU.")

    BATCH_SIZE = 16
    n_batches = int(np.ceil(len(samples) / BATCH_SIZE))

    preds_e001 = {}
    preds_e002 = {}
    preds_e003 = {}
    preds_e004 = {}
    preds_e005 = {}
    preds_e006 = {}
    preds_e007 = {}

    print(f"\nBeginning single-pass inference across {len(samples)} test pairs ({n_batches} batches)...")
    t_start = time.time()

    for b_idx in range(n_batches):
        b_samples = samples[b_idx * BATCH_SIZE : min((b_idx + 1) * BATCH_SIZE, len(samples))]
        v_tensors, t_tensors, v4_tensors, pids = [], [], [], []

        for s in b_samples:
            pids.append(s["pair_id"])
            im_v = cv2.imread(s["rgb_path"])
            im_t = cv2.imread(s["th_path"])

            lb_v = letterbox(image=im_v)
            lb_t = letterbox(image=im_t)

            tv = torch.from_numpy(lb_v).permute(2, 0, 1).float() / 255.0
            tt = torch.from_numpy(lb_t).permute(2, 0, 1).float() / 255.0
            t_gray = torch.from_numpy(lb_t[:, :, 0:1]).permute(2, 0, 1).float() / 255.0
            t_4ch = torch.cat([tv, t_gray], dim=0)

            v_tensors.append(tv)
            t_tensors.append(tt)
            v4_tensors.append(t_4ch)

        b_v = torch.stack(v_tensors).to(DEVICE)
        b_t = torch.stack(t_tensors).to(DEVICE)
        b_4ch = torch.stack(v4_tensors).to(DEVICE)

        with torch.no_grad():
            # 1. E001
            out1 = m1.model(b_v)[0]
            nms1 = non_max_suppression(out1, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms1):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e001[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e001[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

            # 2. E002
            out2 = m2.model(b_v)[0]
            nms2 = non_max_suppression(out2, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms2):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e002[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e002[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

            # 3. E003
            out3 = m3.model(b_t)[0]
            nms3 = non_max_suppression(out3, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms3):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e003[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e003[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

            # 4. E005
            out5 = m5.model(b_4ch)[0]
            nms5 = non_max_suppression(out5, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms5):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e005[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e005[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

            # 5. E006
            out6 = m6(b_v, b_t)[0]
            nms6 = non_max_suppression(out6, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms6):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e006[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e006[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

            # 6. E007
            out7 = m7(b_v, b_t)[0]
            nms7 = non_max_suppression(out7, conf_thres=0.001, iou_thres=0.65, max_det=300)
            for i_item, p in enumerate(nms7):
                if p is not None and len(p) > 0:
                    p[:, :4] = scale_boxes((640, 640), p[:, :4], (1024, 1280))
                    preds_e007[pids[i_item]] = p.cpu().numpy()
                else:
                    preds_e007[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches:
            el = time.time() - t_start
            print(f"  Processed {min((b_idx + 1) * BATCH_SIZE, len(samples))}/{len(samples)} pairs ({el:.1f}s)...")

    # 7. E004 Late Fusion WBF
    print("\nComputing E004 (Late Fusion WBF) from E002 and E003 predictions...")
    for s in samples:
        pid = s["pair_id"]
        b2 = preds_e002.get(pid, np.empty((0, 6), dtype=np.float32))
        b3 = preds_e003.get(pid, np.empty((0, 6), dtype=np.float32))

        boxes_list, scores_list, labels_list = [], [], []
        if len(b2) > 0:
            boxes_list.append((b2[:, :4] / np.array([1280.0, 1024.0, 1280.0, 1024.0])).tolist())
            scores_list.append(b2[:, 4].tolist())
            labels_list.append([0] * len(b2))
        else:
            boxes_list.append([]); scores_list.append([]); labels_list.append([])

        if len(b3) > 0:
            boxes_list.append((b3[:, :4] / np.array([1280.0, 1024.0, 1280.0, 1024.0])).tolist())
            scores_list.append(b3[:, 4].tolist())
            labels_list.append([0] * len(b3))
        else:
            boxes_list.append([]); scores_list.append([]); labels_list.append([])

        if len(boxes_list[0]) == 0 and len(boxes_list[1]) == 0:
            preds_e004[pid] = np.empty((0, 6), dtype=np.float32)
        else:
            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, weights=[1.0, 2.0], iou_thr=0.55, skip_box_thr=0.001
            )
            if len(wbf_boxes) > 0:
                wbf_rescaled = wbf_boxes * np.array([1280.0, 1024.0, 1280.0, 1024.0])
                preds_e004[pid] = np.column_stack([wbf_rescaled, wbf_scores, wbf_labels]).astype(np.float32)
            else:
                preds_e004[pid] = np.empty((0, 6), dtype=np.float32)

    # Save Predictions JSON
    all_models_dict = {
        "E001_RGB_YOLO11N": preds_e001,
        "E002_RGB_YOLO26N": preds_e002,
        "E003_THERMAL_YOLO26N": preds_e003,
        "E004_LATE_FUSION_WBF_YOLO26N": preds_e004,
        "E005_EARLY_FUSION_4CH_YOLO26N": preds_e005,
        "E006_FEATURE_FUSION_CONCAT_YOLO26N": preds_e006,
        "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N": preds_e007,
    }

    print("\nSaving test predictions JSON files...")
    for m_id, m_preds in all_models_dict.items():
        json_dict = {k: v.tolist() for k, v in m_preds.items()}
        for p in [PRED_DIR / f"{m_id}_test_predictions.json", PRED_DIR_LLVIP / f"{m_id}_test_predictions.json"]:
            with open(p, "w") as f:
                json.dump(json_dict, f)
        print(f"  Saved {m_id}_test_predictions.json")

    # Evaluate Overall Official Test Metrics
    print("\n==================== OFFICIAL TEST RESULTS (3,463 Pairs) ====================")
    overall_records = []
    per_scene_records = []

    unique_scenes = sorted(list(set(pair_scenes.values())))
    scene_subsets = {sc: [p for p, s in pair_scenes.items() if s == sc] for sc in unique_scenes}
    scene_subsets["overall"] = all_pids

    for m_id, m_preds in all_models_dict.items():
        res = evaluate_subset(m_preds, gts, all_pids)
        rec = {
            "model_id": m_id,
            "partition": "official_test",
            "frame_count": len(all_pids),
            "total_gts": res["total_gts"],
            "mAP50": res["mAP50"],
            "mAP50_95": res["mAP50_95"],
            "precision_at_025": res["precision_at_025"],
            "recall_at_025": res["recall_at_025"],
            "f1_at_025": res["f1_at_025"],
            "tp_at_025": res["tp_at_025"],
            "fp_at_025": res["fp_at_025"],
            "fn_at_025": res["fn_at_025"],
            "fp_per_image": res["fp_per_image"],
            "total_preds_at_025": res["total_preds_at_025"]
        }
        overall_records.append(rec)
        print(f"{m_id:38s} | mAP50: {res['mAP50']:5.2f}% | mAP50-95: {res['mAP50_95']:5.2f}% | Rec: {res['recall_at_025']:5.2f}% | Prec: {res['precision_at_025']:5.2f}% | F1: {res['f1_at_025']:5.2f}% | FP/img: {res['fp_per_image']:5.3f}")

        # Per-Scene evaluation
        for sc in unique_scenes:
            sc_pids = scene_subsets[sc]
            res_sc = evaluate_subset(m_preds, gts, sc_pids)
            per_scene_records.append({
                "model_id": m_id,
                "scene_id": f"Scene_{str(sc).zfill(2)}",
                "frame_count": len(sc_pids),
                "total_gts": res_sc["total_gts"],
                "mAP50": res_sc["mAP50"],
                "mAP50_95": res_sc["mAP50_95"],
                "precision_at_025": res_sc["precision_at_025"],
                "recall_at_025": res_sc["recall_at_025"],
                "f1_at_025": res_sc["f1_at_025"],
                "tp_at_025": res_sc["tp_at_025"],
                "fp_at_025": res_sc["fp_at_025"],
                "fn_at_025": res_sc["fn_at_025"],
                "fp_per_image": res_sc["fp_per_image"]
            })

    # Save Overall Metrics CSV and JSON
    df_overall = pd.DataFrame(overall_records)
    for p in [OUT_DIR / "official_test_metrics.csv", OUT_DIR_LLVIP / "official_test_metrics.csv"]:
        df_overall.to_csv(p, index=False)
    for p in [OUT_DIR / "official_test_metrics.json", OUT_DIR_LLVIP / "official_test_metrics.json"]:
        df_overall.to_json(p, orient="records", indent=2)

    # Save Per-Scene Metrics CSV
    df_scenes = pd.DataFrame(per_scene_records)
    for p in [OUT_DIR / "test_per_scene_metrics.csv", OUT_DIR_LLVIP / "test_per_scene_metrics.csv"]:
        df_scenes.to_csv(p, index=False)

    print("\nOfficial test metrics successfully saved to CSV and JSON.")
    print(f"Total Stage 8 execution duration: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    main()

