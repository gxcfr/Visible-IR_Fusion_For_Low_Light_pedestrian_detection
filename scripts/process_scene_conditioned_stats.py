import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
OUTPUT_DIR = PROJECT_ROOT / "research" / "next_phase"
OUTPUT_DIR_LLVIP = LLVIP_ROOT / "research" / "next_phase"

df = pd.read_csv(OUTPUT_DIR / "scene_conditioned_gate_records_1803.csv")
print(f"Loaded {len(df)} records from CSV.")

scenes = [int(x) for x in sorted(df["scene_id"].unique())]

def fisher_ci(r, n):
    if abs(r) >= 1.0 or n <= 3:
        return [float(r), float(r)]
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

per_scene_stats = {}

for sc in scenes:
    sdf = df[df["scene_id"] == sc].copy()
    n_sc = len(sdf)
    
    r_b, p_b = stats.pearsonr(sdf["p3_mean_gate"], sdf["rgb_mean_brightness"])
    r_vc, p_vc = stats.pearsonr(sdf["p3_mean_gate"], sdf["rgb_contrast_std"])
    r_tc, p_tc = stats.pearsonr(sdf["p3_mean_gate"], sdf["th_contrast_std"])
    
    rho_b, sp_b = stats.spearmanr(sdf["p3_mean_gate"], sdf["rgb_mean_brightness"])
    rho_vc, sp_vc = stats.spearmanr(sdf["p3_mean_gate"], sdf["rgb_contrast_std"])
    rho_tc, sp_tc = stats.spearmanr(sdf["p3_mean_gate"], sdf["th_contrast_std"])
    
    fg_valid = sdf["p3_fg_gate"].dropna()
    
    per_scene_stats[f"scene_{str(sc).zfill(2)}"] = {
        "scene_id": int(sc),
        "frame_count": int(n_sc),
        "scene_means": {
            "rgb_mean_brightness": round(float(sdf["rgb_mean_brightness"].mean()), 2),
            "rgb_contrast_std": round(float(sdf["rgb_contrast_std"].mean()), 2),
            "th_contrast_std": round(float(sdf["th_contrast_std"].mean()), 2),
            "p3_gate_mean": round(float(sdf["p3_mean_gate"].mean()), 4),
            "p3_gate_std": round(float(sdf["p3_mean_gate"].std()), 4),
            "p3_fg_gate_mean": round(float(fg_valid.mean()), 4) if len(fg_valid) > 0 else None,
            "p3_bg_gate_mean": round(float(sdf["p3_bg_gate"].mean()), 4),
        },
        "correlations": {
            "visible_brightness": {
                "pearson_r": round(float(r_b), 4),
                "pearson_p": float(p_b),
                "pearson_ci95": [round(float(x), 4) for x in fisher_ci(r_b, n_sc)],
                "spearman_rho": round(float(rho_b), 4),
                "spearman_p": float(sp_b)
            },
            "visible_contrast": {
                "pearson_r": round(float(r_vc), 4),
                "pearson_p": float(p_vc),
                "pearson_ci95": [round(float(x), 4) for x in fisher_ci(r_vc, n_sc)],
                "spearman_rho": round(float(rho_vc), 4),
                "spearman_p": float(sp_vc)
            },
            "thermal_contrast": {
                "pearson_r": round(float(r_tc), 4),
                "pearson_p": float(p_tc),
                "pearson_ci95": [round(float(x), 4) for x in fisher_ci(r_tc, n_sc)],
                "spearman_rho": round(float(rho_tc), 4),
                "spearman_p": float(sp_tc)
            }
        }
    }

# Within-Scene Centered Analysis
df_centered = df.copy()
for col in ["rgb_mean_brightness", "rgb_contrast_std", "th_contrast_std", "p3_mean_gate"]:
    df_centered[f"{col}_centered"] = df.groupby("scene_id")[col].transform(lambda x: x - x.mean())

r_b_cent, p_b_cent = stats.pearsonr(df_centered["p3_mean_gate_centered"], df_centered["rgb_mean_brightness_centered"])
r_vc_cent, p_vc_cent = stats.pearsonr(df_centered["p3_mean_gate_centered"], df_centered["rgb_contrast_std_centered"])
r_tc_cent, p_tc_cent = stats.pearsonr(df_centered["p3_mean_gate_centered"], df_centered["th_contrast_std_centered"])

rho_b_cent, sp_b_cent = stats.spearmanr(df_centered["p3_mean_gate_centered"], df_centered["rgb_mean_brightness_centered"])
rho_vc_cent, sp_vc_cent = stats.spearmanr(df_centered["p3_mean_gate_centered"], df_centered["rgb_contrast_std_centered"])
rho_tc_cent, sp_tc_cent = stats.spearmanr(df_centered["p3_mean_gate_centered"], df_centered["th_contrast_std_centered"])

within_centered_stats = {
    "interpretation": "Evaluates correlation after subtracting each scene's mean, removing scene-level baseline shifts and cross-scene confounding (Simpson's paradox).",
    "visible_brightness": {
        "centered_pearson_r": round(float(r_b_cent), 4),
        "centered_pearson_p": float(p_b_cent),
        "centered_spearman_rho": round(float(rho_b_cent), 4),
        "centered_spearman_p": float(sp_b_cent)
    },
    "visible_contrast": {
        "centered_pearson_r": round(float(r_vc_cent), 4),
        "centered_pearson_p": float(p_vc_cent),
        "centered_spearman_rho": round(float(rho_vc_cent), 4),
        "centered_spearman_p": float(sp_vc_cent)
    },
    "thermal_contrast": {
        "centered_pearson_r": round(float(r_tc_cent), 4),
        "centered_pearson_p": float(p_tc_cent),
        "centered_spearman_rho": round(float(rho_tc_cent), 4),
        "centered_spearman_p": float(sp_tc_cent)
    }
}

# Block Bootstrap
block_size = 50
blocks = []
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    n_blocks = int(np.ceil(len(sdf) / block_size))
    for b_idx in range(n_blocks):
        b_data = sdf.iloc[b_idx * block_size : (b_idx + 1) * block_size]
        blocks.append(b_data)

n_total_blocks = len(blocks)
np.random.seed(42)
boot_r_b, boot_r_vc, boot_r_tc = [], [], []

for _ in range(1000):
    sampled_indices = np.random.choice(n_total_blocks, size=n_total_blocks, replace=True)
    boot_df = pd.concat([blocks[i] for i in sampled_indices], ignore_index=True)
    
    rb, _ = stats.pearsonr(boot_df["p3_mean_gate"], boot_df["rgb_mean_brightness"])
    rvc, _ = stats.pearsonr(boot_df["p3_mean_gate"], boot_df["rgb_contrast_std"])
    rtc, _ = stats.pearsonr(boot_df["p3_mean_gate"], boot_df["th_contrast_std"])
    
    boot_r_b.append(rb)
    boot_r_vc.append(rvc)
    boot_r_tc.append(rtc)

block_bootstrap_stats = {
    "block_size_frames": int(block_size),
    "total_blocks": int(n_total_blocks),
    "bootstrap_iterations": 1000,
    "block_bootstrap_ci95": {
        "p3_vs_visible_brightness": [round(float(np.percentile(boot_r_b, 2.5)), 4), round(float(np.percentile(boot_r_b, 97.5)), 4)],
        "p3_vs_visible_contrast": [round(float(np.percentile(boot_r_vc, 2.5)), 4), round(float(np.percentile(boot_r_vc, 97.5)), 4)],
        "p3_vs_thermal_contrast": [round(float(np.percentile(boot_r_tc, 2.5)), 4), round(float(np.percentile(boot_r_tc, 97.5)), 4)]
    },
    "standard_error_comparison": {
        "naive_iid_se": round(float(1.0 / np.sqrt(len(df) - 3)), 4),
        "block_bootstrap_se_brightness": round(float(np.std(boot_r_b)), 4),
        "block_bootstrap_se_visible_contrast": round(float(np.std(boot_r_vc)), 4),
        "block_bootstrap_se_thermal_contrast": round(float(np.std(boot_r_tc)), 4)
    }
}

# Cross-Scene Comparative Direction Summary
direction_summary = {
    "visible_brightness": {
        "scene_02": per_scene_stats["scene_02"]["correlations"]["visible_brightness"]["pearson_r"],
        "scene_05": per_scene_stats["scene_05"]["correlations"]["visible_brightness"]["pearson_r"],
        "scene_07": per_scene_stats["scene_07"]["correlations"]["visible_brightness"]["pearson_r"],
        "scene_25": per_scene_stats["scene_25"]["correlations"]["visible_brightness"]["pearson_r"],
        "pooled_r": round(float(stats.pearsonr(df["p3_mean_gate"], df["rgb_mean_brightness"])[0]), 4),
        "centered_r": within_centered_stats["visible_brightness"]["centered_pearson_r"]
    },
    "visible_contrast": {
        "scene_02": per_scene_stats["scene_02"]["correlations"]["visible_contrast"]["pearson_r"],
        "scene_05": per_scene_stats["scene_05"]["correlations"]["visible_contrast"]["pearson_r"],
        "scene_07": per_scene_stats["scene_07"]["correlations"]["visible_contrast"]["pearson_r"],
        "scene_25": per_scene_stats["scene_25"]["correlations"]["visible_contrast"]["pearson_r"],
        "pooled_r": round(float(stats.pearsonr(df["p3_mean_gate"], df["rgb_contrast_std"])[0]), 4),
        "centered_r": within_centered_stats["visible_contrast"]["centered_pearson_r"]
    },
    "thermal_contrast": {
        "scene_02": per_scene_stats["scene_02"]["correlations"]["thermal_contrast"]["pearson_r"],
        "scene_05": per_scene_stats["scene_05"]["correlations"]["thermal_contrast"]["pearson_r"],
        "scene_07": per_scene_stats["scene_07"]["correlations"]["thermal_contrast"]["pearson_r"],
        "scene_25": per_scene_stats["scene_25"]["correlations"]["thermal_contrast"]["pearson_r"],
        "pooled_r": round(float(stats.pearsonr(df["p3_mean_gate"], df["th_contrast_std"])[0]), 4),
        "centered_r": within_centered_stats["thermal_contrast"]["centered_pearson_r"]
    }
}

full_output = {
    "analysis_type": "Scene-Conditioned Gate Correlation and Cluster Uncertainty Analysis",
    "dataset": "LLVIP Validation Split (1,803 image pairs across 4 camera scenes)",
    "per_scene_breakdown": per_scene_stats,
    "within_scene_centered_correlations": within_centered_stats,
    "cross_scene_direction_comparison": direction_summary,
    "block_bootstrap_uncertainty_analysis": block_bootstrap_stats,
}

for p in [OUTPUT_DIR / "scene_conditioned_gate_analysis.json", OUTPUT_DIR_LLVIP / "scene_conditioned_gate_analysis.json"]:
    with open(p, "w") as f:
        json.dump(full_output, f, indent=4)
print(f"Saved complete JSON to {OUTPUT_DIR / 'scene_conditioned_gate_analysis.json'}")

# Generate Multi-Panel Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Scene-Conditioned Gate Activation G_V vs Environmental Attributes", fontsize=15, fontweight="bold")

colors = {2: "#3498db", 5: "#9b59b6", 7: "#e67e22", 25: "#e74c3c"}
scene_labels = {2: "Scene 02 (Standard Urban)", 5: "Scene 05 (Distant Small)", 7: "Scene 07 (Occluded Cluster)", 25: "Scene 25 (Thermal Glare)"}

# 1. Visible Brightness by Scene
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    r_val = per_scene_stats[f"scene_{str(sc).zfill(2)}"]["correlations"]["visible_brightness"]["pearson_r"]
    axes[0, 0].scatter(sdf["rgb_mean_brightness"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.4, s=20, label=f"Scene {str(sc).zfill(2)} (r={r_val})")
    z = np.polyfit(sdf["rgb_mean_brightness"], sdf["p3_mean_gate"], 1)
    axes[0, 0].plot(sdf["rgb_mean_brightness"], np.poly1d(z)(sdf["rgb_mean_brightness"]), color=colors[sc], linestyle="--", linewidth=1.5)
axes[0, 0].set_title("Visible Brightness vs P3 Gate (By Scene)")
axes[0, 0].set_xlabel("Mean Visible Pixel Luminance")
axes[0, 0].set_ylabel("P3 Gate G_V Activation")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.4)

# 2. Visible Contrast by Scene
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    r_val = per_scene_stats[f"scene_{str(sc).zfill(2)}"]["correlations"]["visible_contrast"]["pearson_r"]
    axes[0, 1].scatter(sdf["rgb_contrast_std"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.4, s=20, label=f"Scene {str(sc).zfill(2)} (r={r_val})")
    z = np.polyfit(sdf["rgb_contrast_std"], sdf["p3_mean_gate"], 1)
    axes[0, 1].plot(sdf["rgb_contrast_std"], np.poly1d(z)(sdf["rgb_contrast_std"]), color=colors[sc], linestyle="--", linewidth=1.5)
axes[0, 1].set_title("Visible Contrast (Std) vs P3 Gate (By Scene)")
axes[0, 1].set_xlabel("Visible Pixel Std Dev")
axes[0, 1].set_ylabel("P3 Gate G_V Activation")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.4)

# 3. Thermal Contrast by Scene
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    r_val = per_scene_stats[f"scene_{str(sc).zfill(2)}"]["correlations"]["thermal_contrast"]["pearson_r"]
    axes[1, 0].scatter(sdf["th_contrast_std"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.4, s=20, label=f"Scene {str(sc).zfill(2)} (r={r_val})")
    z = np.polyfit(sdf["th_contrast_std"], sdf["p3_mean_gate"], 1)
    axes[1, 0].plot(sdf["th_contrast_std"], np.poly1d(z)(sdf["th_contrast_std"]), color=colors[sc], linestyle="--", linewidth=1.5)
axes[1, 0].set_title("Thermal Contrast (Std) vs P3 Gate (By Scene)")
axes[1, 0].set_xlabel("Thermal Pixel Std Dev")
axes[1, 0].set_ylabel("P3 Gate G_V Activation")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.4)

# 4. Within-Scene Centered Scatter (All Scenes Superimposed)
axes[1, 1].scatter(df_centered["rgb_mean_brightness_centered"], df_centered["p3_mean_gate_centered"], color="#16a085", alpha=0.3, s=15, label=f"Brightness (r={r_b_cent:.3f})")
axes[1, 1].scatter(df_centered["rgb_contrast_std_centered"], df_centered["p3_mean_gate_centered"], color="#2980b9", alpha=0.3, s=15, label=f"Visible Contrast (r={r_vc_cent:.3f})")
axes[1, 1].scatter(df_centered["th_contrast_std_centered"], df_centered["p3_mean_gate_centered"], color="#e74c3c", alpha=0.3, s=15, label=f"Thermal Contrast (r={r_tc_cent:.3f})")
axes[1, 1].set_title("Within-Scene Centered Residuals (Scene Mean Subtracted)")
axes[1, 1].set_xlabel("Centered Feature Value (X - Mean_Scene)")
axes[1, 1].set_ylabel("Centered Gate Value (G_V - Mean_Scene)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.4)

plt.tight_layout()
for p in [OUTPUT_DIR / "SCENE_CONDITIONED_GATE_ANALYSIS.png", OUTPUT_DIR_LLVIP / "SCENE_CONDITIONED_GATE_ANALYSIS.png"]:
    plt.savefig(p, dpi=300)
plt.close()
print("Saved scene-conditioned plot.")
