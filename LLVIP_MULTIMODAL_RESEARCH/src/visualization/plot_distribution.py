"""
Distribution Plotting Engine for LLVIP Multimodal Research
Generates publication-quality charts for class, instance, split, and box scale distributions.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_dataset_distribution(
    manifest_path: str,
    output_path: str = "results/plots/dataset_distribution.png",
):
    """Plot comprehensive dataset statistics and distributions."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle("LLVIP Dataset Distribution & Pairing Audit", fontsize=16, fontweight="bold", y=0.98)

    # 1. Split Distribution (Train vs Test)
    split_counts = df["split"].value_counts()
    colors = ["#2b5c8f", "#d95f02"]
    axes[0, 0].bar(split_counts.index, split_counts.values, color=colors, edgecolor="black", width=0.5)
    axes[0, 0].set_title("Partition Split Counts (Image Pairs)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Pair Count")
    for i, v in enumerate(split_counts.values):
        axes[0, 0].text(i, v + 200, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0, 0].set_ylim(0, max(split_counts.values) * 1.15)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Persons per image distribution
    person_counts = df["person_instance_count"]
    bins = range(0, min(15, int(person_counts.max()) + 2))
    axes[0, 1].hist(person_counts, bins=bins, color="#31a354", edgecolor="black", align="left", rwidth=0.8)
    axes[0, 1].set_title("Pedestrian Instance Count per Image Pair", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Instances per Pair")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_xticks(range(0, 15))
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Pairing Status
    status_counts = df["pairing_status"].value_counts()
    status_colors = ["#756bb1" if s == "PAIRED" else "#de2d26" for s in status_counts.index]
    axes[1, 0].bar(status_counts.index, status_counts.values, color=status_colors, edgecolor="black", width=0.4)
    axes[1, 0].set_title("Modality Pairing Status", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Pair Count")
    for i, v in enumerate(status_counts.values):
        axes[1, 0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
    axes[1, 0].set_ylim(0, max(status_counts.values) * 1.15)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Instances by Split (Train vs Test)
    inst_by_split = df.groupby("split")["person_instance_count"].sum()
    axes[1, 1].pie(
        inst_by_split.values,
        labels=[f"{k.capitalize()} ({v:,})" for k, v in inst_by_split.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#386cb0", "#fdb462"],
        wedgeprops={"edgecolor": "black", "linewidth": 1.2},
    )
    axes[1, 1].set_title("Total Pedestrian Annotations by Split", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dataset distribution plot saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="results/metrics/dataset_pairing_manifest.csv")
    parser.add_argument("--output", type=str, default="results/plots/dataset_distribution.png")
    args = parser.parse_args()
    plot_dataset_distribution(args.manifest, args.output)
