import json
import xml.etree.ElementTree as ET
from pathlib import Path

def box_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

pairs = [
    ("020001", "Scene 02: Standard Urban"),
    ("050001", "Scene 05: Distant Small Pedestrians"),
    ("070001", "Scene 07: Occluded Cluster"),
    ("250001", "Scene 25: Thermal Contrast & Artificial Glare")
]

models = [
    ("E001", "E001_RGB_YOLO11N"),
    ("E002", "E002_RGB_YOLO26N"),
    ("E003", "E003_THERMAL_YOLO26N"),
    ("E004", "E004_LATE_FUSION_WBF_YOLO26N"),
    ("E005", "E005_EARLY_FUSION_4CH_YOLO26N"),
    ("E006", "E006_FEATURE_FUSION_CONCAT_YOLO26N"),
    ("E007", "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N")
]

qualitative_results = {}

for pair_id, desc in pairs:
    xml_path = f"C:/LLVIP/LLVIP/Annotations/{pair_id}.xml"
    tree = ET.parse(xml_path)
    gt_boxes = []
    for obj in tree.findall("object"):
        b = obj.find("bndbox")
        gt_boxes.append([
            float(b.find("xmin").text),
            float(b.find("ymin").text),
            float(b.find("xmax").text),
            float(b.find("ymax").text)
        ])
    
    print(f"\n================================================================================")
    print(f"PAIR {pair_id} ({desc}) - {len(gt_boxes)} Ground Truths")
    print(f"================================================================================")
    for g_idx, gbox in enumerate(gt_boxes):
        w = gbox[2] - gbox[0]
        h = gbox[3] - gbox[1]
        print(f"  GT {g_idx}: [{gbox[0]:.1f}, {gbox[1]:.1f}, {gbox[2]:.1f}, {gbox[3]:.1f}] area={w*h:.1f} (w={w:.1f}, h={h:.1f})")
        
    qualitative_results[pair_id] = {
        "description": desc,
        "gt_boxes": gt_boxes,
        "models": {}
    }
    
    for tag, m_name in models:
        pred_path = f"research/audit_phase/predictions/{m_name}_predictions.json"
        with open(pred_path) as f:
            preds = json.load(f)
            
        raw_list = preds.get(pair_id, [])
        img_preds = []
        for item in raw_list:
            if isinstance(item, list) and len(item) >= 5:
                x1, y1, x2, y2, conf = item[:5]
                if conf >= 0.25:
                    img_preds.append({"box": [x1, y1, x2, y2], "confidence": conf})
                    
        img_preds_sorted = sorted(img_preds, key=lambda x: x["confidence"], reverse=True)
        
        matched_gt = set()
        matched_preds = {}
        
        for p_idx, pr in enumerate(img_preds_sorted):
            pbox = pr["box"]
            best_iou = 0.0
            best_gt = -1
            for g_idx, gbox in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g_idx
            if best_iou >= 0.50:
                matched_gt.add(best_gt)
                matched_preds[p_idx] = (best_gt, best_iou)
                
        tp = len(matched_gt)
        fp = len(img_preds_sorted) - tp
        fn = len(gt_boxes) - tp
        
        qualitative_results[pair_id]["models"][tag] = {
            "model_name": m_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predictions": []
        }
        
        print(f"\n--- {tag} ({m_name}): TP={tp}, FP={fp}, FN={fn} (Total: {len(img_preds_sorted)}) ---")
        for p_idx, pr in enumerate(img_preds_sorted):
            pbox = pr["box"]
            conf = pr["confidence"]
            if p_idx in matched_preds:
                g_match, iou_val = matched_preds[p_idx]
                status = f"TP (matches GT {g_match}, IoU={iou_val:.3f})"
                is_tp = True
            else:
                status = "FP"
                is_tp = False
                iou_val = 0.0
                g_match = None
            print(f"  [{status}] conf={conf:.3f} box=[{pbox[0]:.1f}, {pbox[1]:.1f}, {pbox[2]:.1f}, {pbox[3]:.1f}]")
            
            qualitative_results[pair_id]["models"][tag]["predictions"].append({
                "box": pbox,
                "confidence": conf,
                "status": "TP" if is_tp else "FP",
                "matched_gt": g_match,
                "matched_iou": round(iou_val, 4)
            })

with open("research/audit_phase/final_reconciliation/qualitative_evaluation_detailed.json", "w") as f:
    json.dump(qualitative_results, f, indent=4)
print("\nSaved detailed qualitative analysis to research/audit_phase/final_reconciliation/qualitative_evaluation_detailed.json")
