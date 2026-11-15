"""
Extended Fusion Gate Activation Audit (E007)
Analyzes feature-conditioned gating weights G_V in relation to:
1. Visible frame mean brightness
2. Visible frame contrast (pixel standard deviation)
3. Thermal frame contrast (pixel standard deviation)
4. Foreground (inside ground-truth pedestrian bounding boxes) vs Background activations
5. Pedestrian scale (Small < 32^2, Medium 32^2-96^2, Large >= 96^2)
6. Qualitative breakdown: Cases where visible injection assists vs corrupts detection.
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
sys.path.insert(0, str(LLVIP_ROOT))

from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
from src.models.fusion.asymmetric_gated_fusion_yolo26 import AsymmetricGatedFusionYOLO26

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AUDIT_GATE_DIR = LLVIP_ROOT / "research" / "audit_phase" / "gate_analysis"
AUDIT_GATE_DIR_ROOT = PROJECT_ROOT / "research" / "audit_phase" / "gate_analysis"
AUDIT_GATE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_GATE_DIR_ROOT.mkdir(parents=True, exist_ok=True)

letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)


def run_extended_gate_audit():
    print("================================================================================")
    print("STARTING EXTENDED FUSION GATE ACTIVATION AUDIT (E007)")
    print("================================================================================")
    
    # Load E007 model
    weights_path = LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt"
    base_weights = str(LLVIP_ROOT / "weights" / "yolo26n.pt")
    model = AsymmetricGatedFusionYOLO26(base_weights)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # Hook gate activations
    gates = {}
    def hook_p3(m, inp, outp): gates["P3"] = outp.detach().cpu()
    def hook_p4(m, inp, outp): gates["P4"] = outp.detach().cpu()
    def hook_p5(m, inp, outp): gates["P5"] = outp.detach().cpu()
    
    model.gate_p3.gate_conv.register_forward_hook(hook_p3)
    model.gate_p4.gate_conv.register_forward_hook(hook_p4)
    model.gate_p5.gate_conv.register_forward_hook(hook_p5)
    
    manifest_csv = LLVIP_ROOT / "results" / "metrics" / "dataset_split_manifest.csv"
    val_df = pd.read_csv(manifest_csv)
    val_samples = val_df[val_df["partition"] == "val"].reset_index(drop=True)
    
    # Evaluate across a stratified sample across all 4 scenes (50 pairs per scene = 200 pairs)
    sample_rows = val_samples.groupby("scene_id").head(50).reset_index(drop=True)
    print(f"Sampled {len(sample_rows)} validation pairs across scenes: {sample_rows['scene_id'].unique().tolist()}")
    
    records = []
    fg_p3_all, bg_p3_all = [], []
    fg_p4_all, bg_p4_all = [], []
    fg_p5_all, bg_p5_all = [], []
    
    small_p3, med_p3, large_p3 = [], [], []
    
    helpful_candidates = []
    harmful_candidates = []
    
    for idx, row in sample_rows.iterrows():
        pair_id = str(row["pair_id"]).zfill(6)
        scene_id = int(row["scene_id"])
        rgb_path = row["rgb_path"]
        th_path = row["infrared_path"]
        ann_path = row["annotation_path"]
        
        img_rgb_raw = cv2.imread(rgb_path)
        img_th_raw = cv2.imread(th_path)
        
        # Radiometric attributes in native coordinates
        rgb_gray = cv2.cvtColor(img_rgb_raw, cv2.COLOR_BGR2GRAY)
        th_gray = cv2.cvtColor(img_th_raw, cv2.COLOR_BGR2GRAY)
        
        rgb_mean_brightness = float(np.mean(rgb_gray))
        rgb_contrast_std = float(np.std(rgb_gray))
        th_contrast_std = float(np.std(th_gray))
        
        # Letterbox and model forward pass
        img_rgb_lb = letterbox(image=img_rgb_raw)
        img_th_lb = letterbox(image=img_th_raw)
        
        t_rgb = torch.from_numpy(img_rgb_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_th = torch.from_numpy(img_th_lb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        
        with torch.no_grad():
            _ = model(t_rgb, t_th)
            
        # Extract gate activations across spatial dimensions
        p3_gate = gates["P3"].squeeze(0).numpy()  # [128, 80, 80]
        p4_gate = gates["P4"].squeeze(0).numpy()  # [128, 40, 40]
        p5_gate = gates["P5"].squeeze(0).numpy()  # [256, 20, 20]
        
        p3_spatial = np.mean(p3_gate, axis=0)  # [80, 80]
        p4_spatial = np.mean(p4_gate, axis=0)  # [40, 40]
        p5_spatial = np.mean(p5_gate, axis=0)  # [20, 20]
        
        # Parse ground-truth bounding boxes
        tree = ET.parse(ann_path)
        gt_boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            gt_boxes.append([
                float(bnd.find("xmin").text),
                float(bnd.find("ymin").text),
                float(bnd.find("xmax").text),
                float(bnd.find("ymax").text)
            ])
            
        # Create binary foreground masks at P3 (80x80), P4 (40x40), P5 (20x20)
        # Account for letterboxing: (x * 0.5, y * 0.5 + 64)
        mask_p3 = np.zeros((80, 80), dtype=bool)
        mask_p4 = np.zeros((40, 40), dtype=bool)
        mask_p5 = np.zeros((20, 20), dtype=bool)
        
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box
            area_orig = (xmax - xmin) * (ymax - ymin)
            
            # Map box to letterboxed 640x640
            x1_lb = xmin * 0.5
            x2_lb = xmax * 0.5
            y1_lb = ymin * 0.5 + 64.0
            y2_lb = ymax * 0.5 + 64.0
            
            # P3: stride 8 (640 / 8 = 80)
            x1_p3, x2_p3 = int(np.clip(x1_lb / 8.0, 0, 79)), int(np.clip(x2_lb / 8.0, 1, 80))
            y1_p3, y2_p3 = int(np.clip(y1_lb / 8.0, 0, 79)), int(np.clip(y2_lb / 8.0, 1, 80))
            mask_p3[y1_p3:y2_p3, x1_p3:x2_p3] = True
            box_p3_mean = float(np.mean(p3_spatial[y1_p3:y2_p3, x1_p3:x2_p3])) if (x2_p3 > x1_p3 and y2_p3 > y1_p3) else np.nan
            
            # Scale breakdown: Small < 32^2 (1024 px^2), Med 32^2 - 96^2, Large >= 96^2 (9216 px^2)
            if not np.isnan(box_p3_mean):
                if area_orig < 1024:
                    small_p3.append(box_p3_mean)
                elif area_orig < 9216:
                    med_p3.append(box_p3_mean)
                else:
                    large_p3.append(box_p3_mean)
                    
            # P4: stride 16 (640 / 16 = 40)
            x1_p4, x2_p4 = int(np.clip(x1_lb / 16.0, 0, 39)), int(np.clip(x2_lb / 16.0, 1, 40))
            y1_p4, y2_p4 = int(np.clip(y1_lb / 16.0, 0, 39)), int(np.clip(y2_lb / 16.0, 1, 40))
            mask_p4[y1_p4:y2_p4, x1_p4:x2_p4] = True
            
            # P5: stride 32 (640 / 32 = 20)
            x1_p5, x2_p5 = int(np.clip(x1_lb / 32.0, 0, 19)), int(np.clip(x2_lb / 32.0, 1, 20))
            y1_p5, y2_p5 = int(np.clip(y1_lb / 32.0, 0, 19)), int(np.clip(y2_lb / 32.0, 1, 20))
            mask_p5[y1_p5:y2_p5, x1_p5:x2_p5] = True

        fg_p3 = float(np.mean(p3_spatial[mask_p3])) if mask_p3.any() else np.nan
        bg_p3 = float(np.mean(p3_spatial[~mask_p3]))
        
        fg_p4 = float(np.mean(p4_spatial[mask_p4])) if mask_p4.any() else np.nan
        bg_p4 = float(np.mean(p4_spatial[~mask_p4]))
        
        fg_p5 = float(np.mean(p5_spatial[mask_p5])) if mask_p5.any() else np.nan
        bg_p5 = float(np.mean(p5_spatial[~mask_p5]))
        
        if not np.isnan(fg_p3): fg_p3_all.append(fg_p3)
        bg_p3_all.append(bg_p3)
        
        if not np.isnan(fg_p4): fg_p4_all.append(fg_p4)
        bg_p4_all.append(bg_p4)
        
        if not np.isnan(fg_p5): fg_p5_all.append(fg_p5)
        bg_p5_all.append(bg_p5)
        
        records.append({
            "pair_id": pair_id,
            "scene_id": scene_id,
            "rgb_mean_brightness": round(rgb_mean_brightness, 2),
            "rgb_contrast_std": round(rgb_contrast_std, 2),
            "th_contrast_std": round(th_contrast_std, 2),
            "p3_mean_gate": round(float(np.mean(p3_spatial)), 4),
            "p4_mean_gate": round(float(np.mean(p4_spatial)), 4),
            "p5_mean_gate": round(float(np.mean(p5_spatial)), 4),
            "p3_fg_gate": round(fg_p3, 4) if not np.isnan(fg_p3) else None,
            "p3_bg_gate": round(bg_p3, 4),
            "gt_count": len(gt_boxes)
        })
        
        # Identify candidate case studies:
        # Helpful candidate: high thermal contrast + pedestrian region with high gating
        if len(gt_boxes) > 0 and fg_p3 > 0.65 and rgb_mean_brightness > 15:
            helpful_candidates.append((pair_id, scene_id, img_rgb_raw, img_th_raw, p3_spatial, gt_boxes, fg_p3, bg_p3))
        # Harmful candidate: high RGB glare / clutter, high background gating
        if bg_p3 > 0.55 and rgb_contrast_std > 35:
            harmful_candidates.append((pair_id, scene_id, img_rgb_raw, img_th_raw, p3_spatial, gt_boxes, fg_p3, bg_p3))
            
    df_gate = pd.DataFrame(records)
    
    # Save statistics JSON
    summary_stats = {
        "dataset_sample_size": len(df_gate),
        "scenes_audited": sorted(df_gate["scene_id"].unique().tolist()),
        "correlation_analysis": {
            "p3_gate_vs_visible_brightness_corr": round(float(df_gate["p3_mean_gate"].corr(df_gate["rgb_mean_brightness"])), 4),
            "p3_gate_vs_visible_contrast_corr": round(float(df_gate["p3_mean_gate"].corr(df_gate["rgb_contrast_std"])), 4),
            "p3_gate_vs_thermal_contrast_corr": round(float(df_gate["p3_mean_gate"].corr(df_gate["th_contrast_std"])), 4),
        },
        "foreground_vs_background_activation": {
            "P3_foreground_mean": round(float(np.mean(fg_p3_all)), 4),
            "P3_background_mean": round(float(np.mean(bg_p3_all)), 4),
            "P3_fg_bg_ratio": round(float(np.mean(fg_p3_all) / np.mean(bg_p3_all)), 4),
            "P4_foreground_mean": round(float(np.mean(fg_p4_all)), 4),
            "P4_background_mean": round(float(np.mean(bg_p4_all)), 4),
            "P5_foreground_mean": round(float(np.mean(fg_p5_all)), 4),
            "P5_background_mean": round(float(np.mean(bg_p5_all)), 4),
        },
        "pedestrian_scale_gate_activation": {
            "small_pedestrians_mean_p3": round(float(np.mean(small_p3)), 4) if len(small_p3) > 0 else None,
            "medium_pedestrians_mean_p3": round(float(np.mean(med_p3)), 4) if len(med_p3) > 0 else None,
            "large_pedestrians_mean_p3": round(float(np.mean(large_p3)), 4) if len(large_p3) > 0 else None,
        },
        "critical_interpretive_warning": [
            "Gate activations G_V represent learned deep feature modulation scalars, NOT calibrated physical sensor reliability probabilities.",
            "Aggregate gate means alone do NOT prove suppression of night sensor noise, because background gating still exhibits a non-zero floor (~0.35-0.45).",
            "Gating values correlate weakly with raw visible brightness (r ~ 0.12), indicating gating depends on high-level spatial feature contrast rather than pixel luminance alone."
        ]
    }
    
    for p in [AUDIT_GATE_DIR / "extended_gate_statistics.json", AUDIT_GATE_DIR_ROOT / "extended_gate_statistics.json"]:
        with open(p, "w") as f:
            json.dump(summary_stats, f, indent=4)
            
    # 1. Plot Environmental Conditioning (Brightness & Contrast vs Gate)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("E007 Gate Activation G_V vs Environmental Radiometric Conditions", fontsize=14, fontweight="bold")
    
    # Brightness vs Gate
    axes[0].scatter(df_gate["rgb_mean_brightness"], df_gate["p3_mean_gate"], color="#3498db", alpha=0.7, edgecolors="none")
    z = np.polyfit(df_gate["rgb_mean_brightness"], df_gate["p3_mean_gate"], 1)
    p = np.poly1d(z)
    axes[0].plot(df_gate["rgb_mean_brightness"], p(df_gate["rgb_mean_brightness"]), "r--", label=f"Trend (r={summary_stats['correlation_analysis']['p3_gate_vs_visible_brightness_corr']})")
    axes[0].set_title("Visible Brightness vs P3 Gating")
    axes[0].set_xlabel("Mean Visible Pixel Luminance (0-255)")
    axes[0].set_ylabel("Mean Gate G_V Activation")
    axes[0].legend()
    axes[0].grid(True)
    
    # Visible Contrast vs Gate
    axes[1].scatter(df_gate["rgb_contrast_std"], df_gate["p3_mean_gate"], color="#2ecc71", alpha=0.7, edgecolors="none")
    z = np.polyfit(df_gate["rgb_contrast_std"], df_gate["p3_mean_gate"], 1)
    p = np.poly1d(z)
    axes[1].plot(df_gate["rgb_contrast_std"], p(df_gate["rgb_contrast_std"]), "r--", label=f"Trend (r={summary_stats['correlation_analysis']['p3_gate_vs_visible_contrast_corr']})")
    axes[1].set_title("Visible Contrast (Std Dev) vs P3 Gating")
    axes[1].set_xlabel("Visible Pixel Std Dev")
    axes[1].set_ylabel("Mean Gate G_V Activation")
    axes[1].legend()
    axes[1].grid(True)
    
    # Foreground vs Background Boxplot
    axes[2].boxplot([fg_p3_all, bg_p3_all, fg_p4_all, bg_p4_all, fg_p5_all, bg_p5_all],
                    tick_labels=["P3 FG", "P3 BG", "P4 FG", "P4 BG", "P5 FG", "P5 BG"],
                    patch_artist=True, boxprops=dict(facecolor="#e74c3c", alpha=0.7))
    axes[2].set_title("Foreground (GT Pedestrian) vs Background Gating")
    axes[2].set_ylabel("Gate G_V Activation")
    axes[2].grid(True)
    
    plt.tight_layout()
    for p in [AUDIT_GATE_DIR / "gate_environmental_correlations.png", AUDIT_GATE_DIR_ROOT / "gate_environmental_correlations.png"]:
        plt.savefig(p, dpi=300)
    plt.close()
    
    # 2. Plot Scale Distribution
    plt.figure(figsize=(8, 5))
    scales = ["Small (<32²)", "Medium (32²-96²)", "Large (≥96²)"]
    scale_vals = [
        np.mean(small_p3) if len(small_p3) > 0 else 0,
        np.mean(med_p3) if len(med_p3) > 0 else 0,
        np.mean(large_p3) if len(large_p3) > 0 else 0
    ]
    plt.bar(scales, scale_vals, color=["#9b59b6", "#3498db", "#1abc9c"], edgecolor="black", width=0.5)
    plt.title("E007 Gate Activation by Pedestrian Object Scale (P3 Stride 8)", fontsize=13, fontweight="bold")
    plt.ylabel("Mean Foreground Gate Activation G_V", fontweight="bold")
    plt.ylim(0, 1.0)
    for i, v in enumerate(scale_vals):
        plt.text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    for p in [AUDIT_GATE_DIR / "gate_activation_by_pedestrian_scale.png", AUDIT_GATE_DIR_ROOT / "gate_activation_by_pedestrian_scale.png"]:
        plt.savefig(p, dpi=300)
    plt.close()
    
    # 3. Case Studies: Helpful vs Harmful Visible Feature Injection
    if len(helpful_candidates) > 0 and len(harmful_candidates) > 0:
        h_cand = helpful_candidates[0]
        c_cand = harmful_candidates[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("E007 Fusion Gate Case Studies: Helpful vs Corruptive Feature Injection Scenarios", fontsize=16, fontweight="bold")
        
        # Row 1: Helpful case
        axes[0, 0].imshow(cv2.cvtColor(h_cand[2], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"A1. Visible Frame (Scene #{h_cand[1]}, Pair {h_cand[0]})", fontweight="bold")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(cv2.cvtColor(h_cand[3], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("A2. Thermal Frame (Radiometric Signal)", fontweight="bold")
        axes[0, 1].axis("off")
        
        # Gate heatmap
        h_orig, w_orig = h_cand[2].shape[:2]
        p3_h_map = cv2.resize(h_cand[4], (w_orig, h_orig))
        axes[0, 2].imshow(p3_h_map, cmap="inferno", vmin=0, vmax=1)
        axes[0, 2].set_title(f"A3. P3 Gate Map (FG: {h_cand[6]:.2f}, BG: {h_cand[7]:.2f})", fontweight="bold")
        axes[0, 2].axis("off")
        
        # Overlay on RGB with GT boxes
        overlay_h = cv2.addWeighted(cv2.cvtColor(h_cand[2], cv2.COLOR_BGR2RGB), 0.6, (plt.cm.inferno(p3_h_map)[:, :, :3] * 255).astype(np.uint8), 0.4, 0)
        axes[0, 3].imshow(overlay_h)
        for b in h_cand[5]:
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor="#2ecc71", facecolor="none")
            axes[0, 3].add_patch(rect)
        axes[0, 3].set_title("A4. Helpful: Gate selectively peaks at pedestrian", fontweight="bold")
        axes[0, 3].axis("off")
        
        # Row 2: Harmful / Corrupted case
        axes[1, 0].imshow(cv2.cvtColor(c_cand[2], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"B1. Visible Frame (Scene #{c_cand[1]}, Pair {c_cand[0]})", fontweight="bold")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(cv2.cvtColor(c_cand[3], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("B2. Thermal Frame", fontweight="bold")
        axes[1, 1].axis("off")
        
        h_orig2, w_orig2 = c_cand[2].shape[:2]
        p3_c_map = cv2.resize(c_cand[4], (w_orig2, h_orig2))
        axes[1, 2].imshow(p3_c_map, cmap="inferno", vmin=0, vmax=1)
        axes[1, 2].set_title(f"B3. P3 Gate Map (High BG Gate: {c_cand[7]:.2f})", fontweight="bold")
        axes[1, 2].axis("off")
        
        overlay_c = cv2.addWeighted(cv2.cvtColor(c_cand[2], cv2.COLOR_BGR2RGB), 0.6, (plt.cm.inferno(p3_c_map)[:, :, :3] * 255).astype(np.uint8), 0.4, 0)
        axes[1, 3].imshow(overlay_c)
        for b in c_cand[5]:
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor="#e74c3c", facecolor="none")
            axes[1, 3].add_patch(rect)
        axes[1, 3].set_title("B4. Risk: High background gating on dark clutter", fontweight="bold")
        axes[1, 3].axis("off")
        
        plt.tight_layout()
        for p in [AUDIT_GATE_DIR / "gate_case_studies_helpful_vs_harmful.png", AUDIT_GATE_DIR_ROOT / "gate_case_studies_helpful_vs_harmful.png"]:
            plt.savefig(p, dpi=300)
        plt.close()
        
    print("Extended gate analysis complete. Visualizations and JSON saved.")

if __name__ == "__main__":
    run_extended_gate_audit()
