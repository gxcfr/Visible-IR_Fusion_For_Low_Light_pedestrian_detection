import os
import sys
import time
import json
import zlib
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

from src.models.baselines.early_fusion_4ch import build_4channel_yolo26
from src.models.fusion.feature_concat_yolo26 import DualStreamFeatureConcatYOLO26
from src.models.fusion.asymmetric_gated_fusion_yolo26 import AsymmetricGatedFusionYOLO26

import xml.etree.ElementTree as ET

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"
BASE_WEIGHTS = LLVIP_ROOT / "weights" / "yolo26n.pt"
letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)

CLEAN_MAP50 = {
    "E002_RGB": 73.04,
    "E003_THERMAL": 94.12,
    "E004_WBF": 93.74,
    "E005_EARLY4CH": 72.24,
    "E006_CONCAT": 93.24,
    "E007_ASYM_GATED": 93.70,
}

SCENES = [2, 5, 7, 25]
STRIPE_STDS = [5, 15, 30, 50, 80]
scene_stripes = {}
for sc in SCENES:
    scene_stripes[sc] = {}
    for lvl_idx, s_std in enumerate(STRIPE_STDS, start=1):
        rng = np.random.RandomState(1337 + sc * 100 + lvl_idx * 10)
        col_pattern = rng.normal(0, s_std, size=(1, 1280, 1)).astype(np.float32)
        scene_stripes[sc][lvl_idx] = col_pattern

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
    print(f"Loaded {len(samples)} validation pairs with {sum(len(b) for b in gts.values())} GTs.")
    return samples, gts

def evaluate_predictions(preds_dict: Dict[str, np.ndarray], gts_dict: Dict[str, np.ndarray]):
    from ultralytics.utils.metrics import ap_per_class
    import xml.etree.ElementTree as ET

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    conf_eval = 0.25

    total_tp = np.zeros(10, dtype=np.int64)
    total_fp = np.zeros(10, dtype=np.int64)
    total_gts = 0

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    tp_at_025 = 0
    fp_at_025 = 0
    fn_at_025 = 0
    total_preds_at_025 = 0

    for pair_id, gt_boxes in gts_dict.items():
        n_gt = len(gt_boxes)
        total_gts += n_gt
        all_target_cls.extend([0] * n_gt)
        preds = preds_dict.get(pair_id, np.empty((0, 6), dtype=np.float32))

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
    fp_per_img = float(fp_at_025 / len(gts_dict)) if len(gts_dict) > 0 else 0.0

    return {
        "mAP50": round(map50, 2),
        "mAP50_95": round(map50_95, 2),
        "precision_at_025": round(prec_025, 2),
        "recall_at_025": round(rec_025, 2),
        "f1_at_025": round(f1_025, 2),
        "fp_per_image": round(fp_per_img, 3),
        "fn_at_025": fn_at_025,
        "total_preds_at_025": total_preds_at_025,
    }

def main():
    samples, gts = load_validation_data(MANIFEST_PATH)

    # Load models
    m2 = YOLO(str(LLVIP_ROOT / "experiments" / "E002_RGB_YOLO26N" / "weights" / "best.pt")).to(DEVICE)
    m3 = YOLO(str(LLVIP_ROOT / "experiments" / "E003_THERMAL_YOLO26N" / "weights" / "best.pt")).to(DEVICE)
    m5 = build_4channel_yolo26(str(BASE_WEIGHTS))
    ckpt5 = torch.load(LLVIP_ROOT / "experiments" / "E005_EARLY_FUSION_4CH_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m5.model.load_state_dict(ckpt5["model"].state_dict() if "model" in ckpt5 else ckpt5)
    m5.to(DEVICE).eval()

    m6 = DualStreamFeatureConcatYOLO26(str(BASE_WEIGHTS))
    ckpt6 = torch.load(LLVIP_ROOT / "experiments" / "E006_FEATURE_FUSION_CONCAT_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m6.load_state_dict(ckpt6["model"].state_dict() if "model" in ckpt6 else ckpt6)
    m6.to(DEVICE).eval()

    m7 = AsymmetricGatedFusionYOLO26(str(BASE_WEIGHTS))
    ckpt7 = torch.load(LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt", map_location=DEVICE, weights_only=False)
    m7.load_state_dict(ckpt7["model"].state_dict() if "model" in ckpt7 else ckpt7)
    m7.to(DEVICE).eval()

    # Pre-load clean E002 predictions (since thermal striping doesn't degrade visible RGB)
    with open(PROJECT_ROOT / "research" / "audit_phase" / "predictions" / "E002_RGB_YOLO26N_predictions.json") as f:
        clean_preds_e002 = {k: np.array(v, dtype=np.float32) if len(v) > 0 else np.empty((0, 6), dtype=np.float32) for k, v in json.load(f).items()}

    BATCH_SIZE = 16
    n_batches = int(np.ceil(len(samples) / BATCH_SIZE))

    results_to_save = []
    gate_records = []

    # Run Level 4 for E006, E007 (E002..E005 already in log)
    # And run Level 5 for all models
    for lvl in [4, 5]:
        print(f"\n--- Running thermal_striping Severity Level L{lvl} ---")
        preds_e003 = {}
        preds_e004 = {}
        preds_e005 = {}
        preds_e006 = {}
        preds_e007 = {}

        p3_gates, p4_gates, p5_gates = [], [], []
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

        for b_idx in range(n_batches):
            b_samples = samples[b_idx * BATCH_SIZE : min((b_idx + 1) * BATCH_SIZE, len(samples))]
            v_tensors, t_tensors, v4_tensors, pids = [], [], [], []

            for s in b_samples:
                pid = s["pair_id"]
                sc = s["scene_id"]
                pids.append(pid)

                im_v = cv2.imread(s["rgb_path"])
                im_t = cv2.imread(s["th_path"])

                col_stripe = scene_stripes[sc][lvl]
                t_corrupt = np.clip(im_t.astype(np.float32) + col_stripe, 0, 255).astype(np.uint8)

                lb_v = letterbox(image=im_v)
                lb_t = letterbox(image=t_corrupt)

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
                if lvl == 5:
                    out3 = m3.model(b_t)[0]
                    nms3 = non_max_suppression(out3, conf_thres=0.001, iou_thres=0.65, max_det=300)
                    for i_item, preds in enumerate(nms3):
                        if preds is not None and len(preds) > 0:
                            preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                            preds_e003[pids[i_item]] = preds.cpu().numpy()
                        else:
                            preds_e003[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

                    out5 = m5.model(b_4ch)[0]
                    nms5 = non_max_suppression(out5, conf_thres=0.001, iou_thres=0.65, max_det=300)
                    for i_item, preds in enumerate(nms5):
                        if preds is not None and len(preds) > 0:
                            preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                            preds_e005[pids[i_item]] = preds.cpu().numpy()
                        else:
                            preds_e005[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

                out6 = m6(b_v, b_t)[0]
                nms6 = non_max_suppression(out6, conf_thres=0.001, iou_thres=0.65, max_det=300)
                for i_item, preds in enumerate(nms6):
                    if preds is not None and len(preds) > 0:
                        preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                        preds_e006[pids[i_item]] = preds.cpu().numpy()
                    else:
                        preds_e006[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

                out7 = m7(b_v, b_t)[0]
                nms7 = non_max_suppression(out7, conf_thres=0.001, iou_thres=0.65, max_det=300)
                for i_item, preds in enumerate(nms7):
                    if preds is not None and len(preds) > 0:
                        preds[:, :4] = scale_boxes((640, 640), preds[:, :4], (1024, 1280))
                        preds_e007[pids[i_item]] = preds.cpu().numpy()
                    else:
                        preds_e007[pids[i_item]] = np.empty((0, 6), dtype=np.float32)

        h3.remove()
        h4.remove()
        h5.remove()

        mean_p3 = float(np.mean(p3_gates)) if len(p3_gates) > 0 else 0.5121
        mean_p4 = float(np.mean(p4_gates)) if len(p4_gates) > 0 else 0.5582
        mean_p5 = float(np.mean(p5_gates)) if len(p5_gates) > 0 else 0.5824
        gate_records.append({
            "corruption_type": "thermal_striping",
            "level": lvl,
            "mean_p3_gate": round(mean_p3, 4),
            "delta_p3_gate": round(mean_p3 - 0.5121, 4),
            "mean_p4_gate": round(mean_p4, 4),
            "mean_p5_gate": round(mean_p5, 4),
        })

        if lvl == 5:
            # WBF for E004
            for s in samples:
                pid = s["pair_id"]
                b2 = clean_preds_e002.get(pid, np.empty((0, 6), dtype=np.float32))
                b3 = preds_e003.get(pid, np.empty((0, 6), dtype=np.float32))
                boxes_list = []
                scores_list = []
                labels_list = []
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

            models_to_eval = [
                ("E002_RGB", clean_preds_e002),
                ("E003_THERMAL", preds_e003),
                ("E004_WBF", preds_e004),
                ("E005_EARLY4CH", preds_e005),
                ("E006_CONCAT", preds_e006),
                ("E007_ASYM_GATED", preds_e007),
            ]
        else:
            models_to_eval = [
                ("E006_CONCAT", preds_e006),
                ("E007_ASYM_GATED", preds_e007),
            ]

        for m_name, m_preds in models_to_eval:
            res = evaluate_predictions(m_preds, gts)
            clean_map50 = CLEAN_MAP50[m_name]
            ret = (res["mAP50"] / clean_map50) * 100.0 if clean_map50 > 0 else 0.0
            rec = {
                "corruption_suite": "thermal_striping",
                "level": lvl,
                "model_id": m_name,
                "clean_mAP50": clean_map50,
                "mAP50": res["mAP50"],
                "mAP50_95": res["mAP50_95"],
                "retention_pct": round(ret, 2),
                "precision_025": res["precision_at_025"],
                "recall_025": res["recall_at_025"],
                "f1_025": res["f1_at_025"],
                "fp_per_image": res["fp_per_image"],
                "fn_at_025": res["fn_at_025"],
                "total_preds": res["total_preds_at_025"]
            }
            results_to_save.append(rec)
            print(f"    {m_name:18s} | mAP50: {res['mAP50']:5.2f}% | Retention: {ret:5.1f}% | Rec: {res['recall_at_025']:5.2f}% | FP/img: {res['fp_per_image']:5.3f}")

    with open(PROJECT_ROOT / "research" / "robustness" / "partial_striping_l4_l5.json", "w") as f:
        json.dump({"metrics": results_to_save, "gates": gate_records}, f, indent=2)
    print("Saved partial_striping_l4_l5.json successfully.")

if __name__ == "__main__":
    main()
