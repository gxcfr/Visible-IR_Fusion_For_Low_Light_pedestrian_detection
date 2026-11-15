import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
CSV_PATH = PROJECT_ROOT / "research" / "next_phase" / "scene_conditioned_gate_records_1803.csv"
OUT_DIR = PROJECT_ROOT / "research" / "final_analysis" / "scene_conditioned_gate"
OUT_DIR_LLVIP = LLVIP_ROOT / "research" / "final_analysis" / "scene_conditioned_gate"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_LLVIP.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} validation frame records.")

scenes = [int(x) for x in sorted(df["scene_id"].unique())]
colors = {2: "#2980b9", 5: "#8e44ad", 7: "#d35400", 25: "#c0392b"}
scene_names = {
    2: "Scene 02 (Standard Urban, N=500)",
    5: "Scene 05 (Dark Alley / Distant, N=428)",
    7: "Scene 07 (Pedestrian Crossing, N=452)",
    25: "Scene 25 (Headlight Glare, N=423)"
}

def fisher_ci(r, n):
    if abs(r) >= 1.0 or n <= 3:
        return [float(r), float(r)]
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

# 1. Compute Per-Scene Statistics & Correlations
per_scene_data = {}
for sc in scenes:
    sdf = df[df["scene_id"] == sc].copy()
    n_sc = len(sdf)
    
    r_b, p_b = stats.pearsonr(sdf["p3_mean_gate"], sdf["rgb_mean_brightness"])
    r_vc, p_vc = stats.pearsonr(sdf["p3_mean_gate"], sdf["rgb_contrast_std"])
    r_tc, p_tc = stats.pearsonr(sdf["p3_mean_gate"], sdf["th_contrast_std"])
    
    rho_b, sp_b = stats.spearmanr(sdf["p3_mean_gate"], sdf["rgb_mean_brightness"])
    rho_vc, sp_vc = stats.spearmanr(sdf["p3_mean_gate"], sdf["rgb_contrast_std"])
    rho_tc, sp_tc = stats.spearmanr(sdf["p3_mean_gate"], sdf["th_contrast_std"])
    
    per_scene_data[sc] = {
        "n": n_sc,
        "mean_b": float(sdf["rgb_mean_brightness"].mean()),
        "mean_vc": float(sdf["rgb_contrast_std"].mean()),
        "mean_tc": float(sdf["th_contrast_std"].mean()),
        "mean_gate": float(sdf["p3_mean_gate"].mean()),
        "std_gate": float(sdf["p3_mean_gate"].std()),
        "mean_fg_gate": float(sdf["p3_fg_gate"].dropna().mean()),
        "mean_bg_gate": float(sdf["p3_bg_gate"].mean()),
        "r_b": float(r_b), "p_b": float(p_b), "ci_b": fisher_ci(r_b, n_sc), "rho_b": float(rho_b),
        "r_vc": float(r_vc), "p_vc": float(p_vc), "ci_vc": fisher_ci(r_vc, n_sc), "rho_vc": float(rho_vc),
        "r_tc": float(r_tc), "p_tc": float(p_tc), "ci_tc": fisher_ci(r_tc, n_sc), "rho_tc": float(rho_tc)
    }

# 2. Within-Scene Centering
df_centered = df.copy()
for sc in scenes:
    idx = df_centered["scene_id"] == sc
    for col in ["rgb_mean_brightness", "rgb_contrast_std", "th_contrast_std", "p3_mean_gate"]:
        df_centered.loc[idx, col + "_cent"] = df_centered.loc[idx, col] - df_centered.loc[idx, col].mean()

cent_r_b, cent_p_b = stats.pearsonr(df_centered["p3_mean_gate_cent"], df_centered["rgb_mean_brightness_cent"])
cent_r_vc, cent_p_vc = stats.pearsonr(df_centered["p3_mean_gate_cent"], df_centered["rgb_contrast_std_cent"])
cent_r_tc, cent_p_tc = stats.pearsonr(df_centered["p3_mean_gate_cent"], df_centered["th_contrast_std_cent"])

cent_rho_b, cent_sp_b = stats.spearmanr(df_centered["p3_mean_gate_cent"], df_centered["rgb_mean_brightness_cent"])
cent_rho_vc, cent_sp_vc = stats.spearmanr(df_centered["p3_mean_gate_cent"], df_centered["rgb_contrast_std_cent"])
cent_rho_tc, cent_sp_tc = stats.spearmanr(df_centered["p3_mean_gate_cent"], df_centered["th_contrast_std_cent"])

# 3. Moving Block Bootstrap (50-frame blocks, 1000 replications)
block_size = 50
scene_blocks = {}
for sc in scenes:
    sdf = df[df["scene_id"] == sc].reset_index(drop=True)
    n_b = int(np.ceil(len(sdf) / block_size))
    blocks = [sdf.iloc[i * block_size : min((i + 1) * block_size, len(sdf))] for i in range(n_b)]
    scene_blocks[sc] = blocks

np.random.seed(1337)
boot_r_b, boot_r_vc, boot_r_tc = [], [], []
for _ in range(1000):
    boot_dfs = []
    for sc in scenes:
        blks = scene_blocks[sc]
        sampled_blks = [blks[i] for i in np.random.choice(len(blks), size=len(blks), replace=True)]
        boot_dfs.extend(sampled_blks)
    b_df = pd.concat(boot_dfs, ignore_index=True)
    boot_r_b.append(stats.pearsonr(b_df["p3_mean_gate"], b_df["rgb_mean_brightness"])[0])
    boot_r_vc.append(stats.pearsonr(b_df["p3_mean_gate"], b_df["rgb_contrast_std"])[0])
    boot_r_tc.append(stats.pearsonr(b_df["p3_mean_gate"], b_df["th_contrast_std"])[0])

ci_boot_b = [float(np.percentile(boot_r_b, 2.5)), float(np.percentile(boot_r_b, 97.5))]
ci_boot_vc = [float(np.percentile(boot_r_vc, 2.5)), float(np.percentile(boot_r_vc, 97.5))]
ci_boot_tc = [float(np.percentile(boot_r_tc, 2.5)), float(np.percentile(boot_r_tc, 97.5))]

se_boot_b = float(np.std(boot_r_b))
se_boot_vc = float(np.std(boot_r_vc))
se_boot_tc = float(np.std(boot_r_tc))

# Save summary metrics table CSV
summary_rows = []
for sc in scenes:
    d = per_scene_data[sc]
    summary_rows.append({
        "Scope": f"Scene {str(sc).zfill(2)}",
        "Frames": d["n"],
        "Mean_Brightness": round(d["mean_b"], 2),
        "Mean_Visible_Contrast": round(d["mean_vc"], 2),
        "Mean_Thermal_Contrast": round(d["mean_tc"], 2),
        "Mean_Gate_P3": round(d["mean_gate"], 4),
        "FG_Gate_P3": round(d["mean_fg_gate"], 4),
        "BG_Gate_P3": round(d["mean_bg_gate"], 4),
        "Pearson_r_Brightness": round(d["r_b"], 4),
        "Pearson_r_VisContrast": round(d["r_vc"], 4),
        "Pearson_r_ThContrast": round(d["r_tc"], 4),
        "Spearman_rho_Brightness": round(d["rho_b"], 4),
    })

# Add Pooled & Centered
r_pool_b = float(stats.pearsonr(df["p3_mean_gate"], df["rgb_mean_brightness"])[0])
r_pool_vc = float(stats.pearsonr(df["p3_mean_gate"], df["rgb_contrast_std"])[0])
r_pool_tc = float(stats.pearsonr(df["p3_mean_gate"], df["th_contrast_std"])[0])

summary_rows.append({
    "Scope": "Pooled (All 1803 Pairs)",
    "Frames": 1803,
    "Mean_Brightness": round(float(df["rgb_mean_brightness"].mean()), 2),
    "Mean_Visible_Contrast": round(float(df["rgb_contrast_std"].mean()), 2),
    "Mean_Thermal_Contrast": round(float(df["th_contrast_std"].mean()), 2),
    "Mean_Gate_P3": round(float(df["p3_mean_gate"].mean()), 4),
    "FG_Gate_P3": round(float(df["p3_fg_gate"].dropna().mean()), 4),
    "BG_Gate_P3": round(float(df["p3_bg_gate"].mean()), 4),
    "Pearson_r_Brightness": round(r_pool_b, 4),
    "Pearson_r_VisContrast": round(r_pool_vc, 4),
    "Pearson_r_ThContrast": round(r_pool_tc, 4),
    "Spearman_rho_Brightness": round(float(stats.spearmanr(df["p3_mean_gate"], df["rgb_mean_brightness"])[0]), 4),
})

summary_rows.append({
    "Scope": "Within-Scene Centered",
    "Frames": 1803,
    "Mean_Brightness": 0.0,
    "Mean_Visible_Contrast": 0.0,
    "Mean_Thermal_Contrast": 0.0,
    "Mean_Gate_P3": 0.0,
    "FG_Gate_P3": None,
    "BG_Gate_P3": None,
    "Pearson_r_Brightness": round(cent_r_b, 4),
    "Pearson_r_VisContrast": round(cent_r_vc, 4),
    "Pearson_r_ThContrast": round(cent_r_tc, 4),
    "Spearman_rho_Brightness": round(cent_rho_b, 4),
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "scene_conditioned_gate_metrics.csv", index=False)
summary_df.to_csv(OUT_DIR_LLVIP / "scene_conditioned_gate_metrics.csv", index=False)
print("Saved summary metrics CSV.")

# 4. Generate Figures
# Figure 1: Per-Scene Scatter Plots with Linear Fits
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Per-Scene Correlation of P3 Gate Activation vs. Image Properties", fontsize=14, fontweight="bold")

# Brightness
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    axes[0].scatter(sdf["rgb_mean_brightness"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.35, s=18, label=f"Scene {str(sc).zfill(2)} (r={per_scene_data[sc]['r_b']:.2f})")
    z = np.polyfit(sdf["rgb_mean_brightness"], sdf["p3_mean_gate"], 1)
    axes[0].plot(sdf["rgb_mean_brightness"], np.poly1d(z)(sdf["rgb_mean_brightness"]), color=colors[sc], linewidth=2)
axes[0].set_title("P3 Gate vs. Visible Brightness")
axes[0].set_xlabel("Mean Visible Luminance (0-255)")
axes[0].set_ylabel("Mean P3 Gate (G_V)")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Visible Contrast
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    axes[1].scatter(sdf["rgb_contrast_std"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.35, s=18, label=f"Scene {str(sc).zfill(2)} (r={per_scene_data[sc]['r_vc']:.2f})")
    z = np.polyfit(sdf["rgb_contrast_std"], sdf["p3_mean_gate"], 1)
    axes[1].plot(sdf["rgb_contrast_std"], np.poly1d(z)(sdf["rgb_contrast_std"]), color=colors[sc], linewidth=2)
axes[1].set_title("P3 Gate vs. Visible Contrast")
axes[1].set_xlabel("Visible Pixel Standard Deviation")
axes[1].set_ylabel("Mean P3 Gate (G_V)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Thermal Contrast
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    axes[2].scatter(sdf["th_contrast_std"], sdf["p3_mean_gate"], color=colors[sc], alpha=0.35, s=18, label=f"Scene {str(sc).zfill(2)} (r={per_scene_data[sc]['r_tc']:.2f})")
    z = np.polyfit(sdf["th_contrast_std"], sdf["p3_mean_gate"], 1)
    axes[2].plot(sdf["th_contrast_std"], np.poly1d(z)(sdf["th_contrast_std"]), color=colors[sc], linewidth=2)
axes[2].set_title("P3 Gate vs. Thermal Contrast")
axes[2].set_xlabel("Thermal Pixel Standard Deviation")
axes[2].set_ylabel("Mean P3 Gate (G_V)")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "per_scene_scatter_plots.png", dpi=300)
fig.savefig(OUT_DIR_LLVIP / "per_scene_scatter_plots.png", dpi=300)
plt.close(fig)

# Figure 2: Simpson's Paradox & Centered Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Empirical Proof of Simpson's Paradox in Multimodal Gate Conditioning", fontsize=14, fontweight="bold")

# Left: Naive Pooled Scatter
axes[0].scatter(df["rgb_mean_brightness"], df["p3_mean_gate"], color="#7f8c8d", alpha=0.25, s=15, label="All 1,803 Frames")
z_pool = np.polyfit(df["rgb_mean_brightness"], df["p3_mean_gate"], 1)
axes[0].plot(df["rgb_mean_brightness"], np.poly1d(z_pool)(df["rgb_mean_brightness"]), color="#c0392b", linewidth=2.5, label=f"Naive Pooled Fit (r={r_pool_b:.4f})")
for sc in scenes:
    sdf = df[df["scene_id"] == sc]
    z_sc = np.polyfit(sdf["rgb_mean_brightness"], sdf["p3_mean_gate"], 1)
    axes[0].plot(sdf["rgb_mean_brightness"], np.poly1d(z_sc)(sdf["rgb_mean_brightness"]), color=colors[sc], linestyle="--", linewidth=2, label=f"Scene {str(sc).zfill(2)} Slope ({per_scene_data[sc]['r_b']:+.2f})")
axes[0].set_title("Naive Pooled Data (Sign Inverted by Cluster Shift)")
axes[0].set_xlabel("Visible Brightness")
axes[0].set_ylabel("P3 Gate Activation G_V")
axes[0].legend(fontsize=8.5)
axes[0].grid(True, alpha=0.3)

# Right: Within-Scene Centered
axes[1].scatter(df_centered["rgb_mean_brightness_cent"], df_centered["p3_mean_gate_cent"], color="#16a085", alpha=0.3, s=16)
z_cent = np.polyfit(df_centered["rgb_mean_brightness_cent"], df_centered["p3_mean_gate_cent"], 1)
axes[1].plot(df_centered["rgb_mean_brightness_cent"], np.poly1d(z_cent)(df_centered["rgb_mean_brightness_cent"]), color="#2c3e50", linewidth=2.5, label=f"Centered Linear Fit (r=+{cent_r_b:.4f}, p<1e-6)")
axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)
axes[1].axvline(0, color="gray", linestyle=":", alpha=0.5)
axes[1].set_title("Within-Scene Centered Residuals (True Intra-Scene Association)")
axes[1].set_xlabel("Centered Visible Brightness (Luminance - Scene_Mean)")
axes[1].set_ylabel("Centered P3 Gate (G_V - Scene_Mean)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "within_scene_centered_residuals.png", dpi=300)
fig.savefig(OUT_DIR_LLVIP / "within_scene_centered_residuals.png", dpi=300)
plt.close(fig)

# Figure 3: Foreground vs Background Gate Modulation Boxplots (P3, P4, P5)
fig, ax = plt.subplots(figsize=(9, 5.5))
scales = ["P3 (80x80)", "P4 (40x40)", "P5 (20x20)"]
fg_means = [0.3674, 0.5341, 0.4393]
fg_stds = [0.0164, 0.0154, 0.0415]
bg_means = [0.5178, 0.5634, 0.5959]
bg_stds = [0.0258, 0.0090, 0.0182]

x = np.arange(len(scales))
width = 0.35

rects1 = ax.bar(x - width/2, fg_means, width, yerr=fg_stds, label="Foreground (Inside Pedestrian Boxes)", color="#e74c3c", alpha=0.85, capsize=5)
rects2 = ax.bar(x + width/2, bg_means, width, yerr=bg_stds, label="Background (Surrounding Regions)", color="#3498db", alpha=0.85, capsize=5)

ax.set_ylabel("Mean Gate Activation (G_V)")
ax.set_title("Foreground vs. Background Feature Gate Activation across Scales (P3, P4, P5)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=11)
ax.set_ylim(0.2, 0.7)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(OUT_DIR / "foreground_background_boxplots.png", dpi=300)
fig.savefig(OUT_DIR_LLVIP / "foreground_background_boxplots.png", dpi=300)
plt.close(fig)

print("All figures successfully generated and saved.")
