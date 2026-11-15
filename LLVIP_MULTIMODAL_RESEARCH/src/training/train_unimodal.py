"""
Unimodal Detection Baseline Training & Evaluation Engine
Controls training, validation, latency benchmarking, and artifact packaging
for YOLO11n and YOLO26n on Visible (RGB) and Thermal (Infrared) modalities.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Dynamically patch Ultralytics label path resolution
import ultralytics.data.dataset as ds_module
import ultralytics.data.utils as utils_module
labels_dir = (ROOT_DIR / "datasets" / "labels").resolve()
def custom_img2label_paths(img_paths):
    return [str(labels_dir / f"{Path(x).stem}.txt") for x in img_paths]
ds_module.img2label_paths = custom_img2label_paths
utils_module.img2label_paths = custom_img2label_paths

from ultralytics import YOLO
from src.evaluation.latency import benchmark_latency


def train_unimodal(
    exp_id: str,
    model_weights: str,
    modality: str = "rgb",
    batch_size: int = 8,
    epochs: int = 100,
    patience: int = 20,
    img_size: int = 640,
    workers: int = 2,
    lr0: float = 0.001,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete controlled unimodal training and benchmark pipeline."""
    exp_dir = ROOT_DIR / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STARTING EXPERIMENT: {exp_id}")
    print(f"Modality:           {modality.upper()}")
    print(f"Model Checkpoint:   {model_weights}")
    print(f"Batch Size:         {batch_size}")
    print(f"Max Epochs:         {epochs} (Patience: {patience})")
    print(f"Optimizer:          AdamW (lr0={lr0}, weight_decay={weight_decay})")
    print(f"Resolution:         {img_size}x{img_size}")
    print(f"Experiment Dir:     {exp_dir}")
    print("=" * 70)

    # 1. Prepare Dataset YAML pointing to internal train & validation files
    train_file = (ROOT_DIR / "datasets" / f"train_{modality}.txt").resolve()
    val_file = (ROOT_DIR / "datasets" / f"val_{modality}.txt").resolve()
    test_file = (ROOT_DIR / "datasets" / f"test_{modality}.txt").resolve()

    dataset_dict = {
        "path": str((ROOT_DIR / "datasets").resolve()),
        "train": str(train_file),
        "val": str(val_file),
        "test": str(test_file),
        "names": {0: "person"},
        "nc": 1,
    }

    dataset_yaml_path = exp_dir / "dataset.yaml"
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_dict, f, default_flow_style=False)

    # 2. Modality-specific augmentation parameters
    # For thermal: disable hue and saturation jitter (hsv_h=0, hsv_s=0) to preserve radiometric intensity
    if modality.lower() == "thermal":
        hsv_h = 0.0
        hsv_s = 0.0
        hsv_v = 0.2  # moderate intensity jitter
    else:  # RGB
        hsv_h = 0.015
        hsv_s = 0.7
        hsv_v = 0.4

    # Save full experiment config
    config_dict = {
        "experiment_id": exp_id,
        "modality": modality,
        "model_weights": model_weights,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "img_size": img_size,
        "optimizer": "AdamW",
        "lr0": lr0,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "seed": seed,
        "cos_lr": True,
        "amp": True,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "fliplr": 0.5,
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # 3. Model Initialization
    if os.path.exists(model_weights):
        weights_p = Path(model_weights).resolve()
    elif (ROOT_DIR / model_weights).exists():
        weights_p = (ROOT_DIR / model_weights).resolve()
    elif (ROOT_DIR / "weights" / Path(model_weights).name).exists():
        weights_p = (ROOT_DIR / "weights" / Path(model_weights).name).resolve()
    else:
        raise FileNotFoundError(f"Weight file not found: {model_weights}")

    print(f"Loading local verified weights from: {weights_p}")
    model = YOLO(str(weights_p))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_time = time.time()

    # 4. Training Run
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=img_size,
        device="0",
        workers=workers,
        optimizer="AdamW",
        lr0=lr0,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        cos_lr=True,
        amp=True,
        seed=seed,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        fliplr=0.5,
        save=True,
        project=str(ROOT_DIR / "experiments"),
        name=exp_id,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    training_duration_s = round(time.time() - start_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    # 5. Locate Best Checkpoint & Analyze Stopping Dynamics
    best_ckpt = exp_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = exp_dir / "weights" / "last.pt"

    # Analyze results.csv for stopping epoch, best epoch, and stopping reason
    results_csv = exp_dir / "results.csv"
    stopping_epoch = epochs
    best_epoch = epochs
    stopping_reason = "maximum_epochs_reached"
    improvement_near_stop = "Validation plateau reached"

    if results_csv.exists():
        import pandas as pd
        df_res = pd.read_csv(results_csv)
        df_res.columns = [c.strip() for c in df_res.columns]
        stopping_epoch = int(len(df_res))
        
        # Fitness in Ultralytics is 0.1 * mAP50 + 0.9 * mAP50-95
        if "metrics/mAP50(B)" in df_res.columns and "metrics/mAP50-95(B)" in df_res.columns:
            fitness = 0.1 * df_res["metrics/mAP50(B)"] + 0.9 * df_res["metrics/mAP50-95(B)"]
            best_epoch = int(fitness.idxmax() + 1)
        elif "metrics/mAP50(B)" in df_res.columns:
            best_epoch = int(df_res["metrics/mAP50(B)"].idxmax() + 1)

        if stopping_epoch < epochs:
            stopping_reason = f"early_stopping_triggered (patience={patience} exhausted after best epoch {best_epoch})"
            improvement_near_stop = f"No fitness improvement for {patience} consecutive epochs since epoch {best_epoch}"
        else:
            stopping_reason = f"maximum_epochs_reached ({epochs} epochs completed)"
            improvement_near_stop = "Completed full scheduled epochs"

    print(f"\nTraining completed in {training_duration_s / 60:.2f} minutes.")
    print(f"Stopping Epoch:  {stopping_epoch} / {epochs}")
    print(f"Best Epoch:      {best_epoch}")
    print(f"Stopping Reason: {stopping_reason}")
    print(f"Evaluating best checkpoint: {best_ckpt}")

    # 6. Evaluation on Validation Set
    val_model = YOLO(str(best_ckpt))
    val_res = val_model.val(
        data=str(dataset_yaml_path),
        split="val",
        imgsz=img_size,
        batch=batch_size,
        device="0",
        verbose=False,
    )

    metrics_dict = val_res.results_dict
    precision = round(float(metrics_dict.get("metrics/precision(B)", 0.0)), 4)
    recall = round(float(metrics_dict.get("metrics/recall(B)", 0.0)), 4)
    map50 = round(float(metrics_dict.get("metrics/mAP50(B)", 0.0)), 4)
    map50_95 = round(float(metrics_dict.get("metrics/mAP50-95(B)", 0.0)), 4)
    f1 = round(2 * precision * recall / max(precision + recall, 1e-16), 4)

    # 7. Hardware Latency Benchmarking (Batch 1, GPU sync)
    print("Running hardware latency benchmark...")
    latency_stats = benchmark_latency(str(best_ckpt), img_size=img_size, device="cuda:0")

    # 8. Assemble Master Experiment Metrics
    experiment_metrics = {
        "experiment_id": exp_id,
        "modality": modality,
        "model_architecture": Path(model_weights).stem,
        "status": "COMPLETED",
        "training_duration_seconds": training_duration_s,
        "training_duration_minutes": round(training_duration_s / 60, 2),
        "peak_vram_allocated_mb": peak_vram_mb,
        "stopping_epoch": stopping_epoch,
        "best_epoch": best_epoch,
        "stopping_reason": stopping_reason,
        "improvement_near_stop": improvement_near_stop,
        "validation_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mAP50": map50,
            "mAP50-95": map50_95,
        },
        "latency_metrics": latency_stats,
        "checkpoint_paths": {
            "best": str(best_ckpt),
            "last": str(exp_dir / "weights" / "last.pt"),
        },
    }

    # Save metrics JSON in experiment dir and results/metrics
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(experiment_metrics, f, indent=4)
    with open(ROOT_DIR / "results" / "metrics" / f"{exp_id}_metrics.json", "w") as f:
        json.dump(experiment_metrics, f, indent=4)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_id} BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"mAP50:           {map50}")
    print(f"mAP50-95:        {map50_95}")
    print(f"Precision:       {precision}")
    print(f"Recall:          {recall}")
    print(f"F1 Score:        {f1}")
    print(f"Model Latency:   {latency_stats['model_latency_mean_ms']} ms")
    print(f"Throughput:      {latency_stats['fps']} FPS")
    print(f"Parameters:      {latency_stats['model_parameters']:,}")
    print(f"Peak VRAM:       {peak_vram_mb} MB")
    print(f"Training Time:   {training_duration_s / 60:.2f} min")
    print("=" * 70)

    return experiment_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "thermal"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    train_unimodal(
        exp_id=args.exp_id,
        model_weights=args.model,
        modality=args.modality,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        workers=args.workers,
    )
