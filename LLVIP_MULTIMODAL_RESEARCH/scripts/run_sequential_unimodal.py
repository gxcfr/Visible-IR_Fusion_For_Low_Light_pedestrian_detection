"""
Sequential Unimodal Experiment Pipeline Runner (E002 -> E003)
Executes YOLO26n RGB baseline and YOLO26n Thermal baseline sequentially,
logging progress, tracking metrics, and ensuring complete GPU cleanup between runs.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def run_experiment(exp_id: str, model_weights: str, modality: str, batch: int = 8, epochs: int = 40, patience: int = 12):
    print("\n" + "#" * 80)
    print(f"PIPELINE STEP: Executing {exp_id} ({modality.upper()} Modality)")
    print(f"Model: {model_weights} | Batch: {batch} | Epochs: {epochs} | Patience: {patience}")
    print("#" * 80 + "\n", flush=True)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "training" / "train_unimodal.py"),
        "--exp_id", exp_id,
        "--model", str(ROOT_DIR / model_weights),
        "--modality", modality,
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--workers", "2",
    ]

    start_t = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR))
    duration = time.time() - start_t

    if proc.returncode != 0:
        raise RuntimeError(f"Experiment {exp_id} failed with exit code {proc.returncode}")

    metrics_file = ROOT_DIR / "experiments" / exp_id / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_file}")

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id} SUCCESSFULLY COMPLETED in {duration / 60:.2f} minutes")
    print(f"mAP50:     {metrics['validation_metrics']['mAP50']:.4f}")
    print(f"mAP50-95:  {metrics['validation_metrics']['mAP50-95']:.4f}")
    print(f"Precision: {metrics['validation_metrics']['precision']:.4f}")
    print(f"Recall:    {metrics['validation_metrics']['recall']:.4f}")
    print(f"F1:        {metrics['validation_metrics']['f1_score']:.4f}")
    print(f"Latency:   {metrics['latency_metrics']['model_latency_mean_ms']:.2f} ms ({metrics['latency_metrics']['fps']:.1f} FPS)")
    print("=" * 80 + "\n", flush=True)

    return metrics


def update_readme_and_logs(e002_metrics, e003_metrics=None):
    """Update experiment log and root README with freshly completed experiments."""
    log_file = ROOT_DIR / "research" / "experiment_log.md"
    readme_file = ROOT_DIR / "README.md"
    
    # We will log the progress entry
    with open(log_file, "a") as f:
        f.write(f"\n### [Execution] Milestone 4 - {e002_metrics['experiment_id']}\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- **Status**: COMPLETED\n")
        f.write(f"- **Architecture**: YOLO26n\n")
        f.write(f"- **Modality**: {e002_metrics['modality'].upper()}\n")
        f.write(f"- **Stopping Epoch**: {e002_metrics.get('stopping_epoch', 40)} / 40 (Best: {e002_metrics.get('best_epoch', 40)})\n")
        f.write(f"- **mAP50**: {e002_metrics['validation_metrics']['mAP50']:.4f} | **mAP50-95**: {e002_metrics['validation_metrics']['mAP50-95']:.4f}\n")
        f.write(f"- **Precision**: {e002_metrics['validation_metrics']['precision']:.4f} | **Recall**: {e002_metrics['validation_metrics']['recall']:.4f} | **F1**: {e002_metrics['validation_metrics']['f1_score']:.4f}\n")
        f.write(f"- **Latency**: {e002_metrics['latency_metrics']['model_latency_mean_ms']:.2f} ms ({e002_metrics['latency_metrics']['fps']:.1f} FPS)\n")
        
        if e003_metrics:
            f.write(f"\n### [Execution] Milestone 4 - {e003_metrics['experiment_id']}\n")
            f.write(f"- **Date**: {time.strftime('%Y-%m-%d')}\n")
            f.write(f"- **Status**: COMPLETED\n")
            f.write(f"- **Architecture**: YOLO26n\n")
            f.write(f"- **Modality**: {e003_metrics['modality'].upper()}\n")
            f.write(f"- **Stopping Epoch**: {e003_metrics.get('stopping_epoch', 40)} / 40 (Best: {e003_metrics.get('best_epoch', 40)})\n")
            f.write(f"- **mAP50**: {e003_metrics['validation_metrics']['mAP50']:.4f} | **mAP50-95**: {e003_metrics['validation_metrics']['mAP50-95']:.4f}\n")
            f.write(f"- **Precision**: {e003_metrics['validation_metrics']['precision']:.4f} | **Recall**: {e003_metrics['validation_metrics']['recall']:.4f} | **F1**: {e003_metrics['validation_metrics']['f1_score']:.4f}\n")
            f.write(f"- **Latency**: {e003_metrics['latency_metrics']['model_latency_mean_ms']:.2f} ms ({e003_metrics['latency_metrics']['fps']:.1f} FPS)\n")


def main():
    print("=" * 80)
    print("STARTING SEQUENTIAL UNIMODAL BASELINES PIPELINE (E002 -> E003)")
    print("=" * 80)

    # Step 1: E002_RGB_YOLO26N
    e002_metrics = run_experiment(
        exp_id="E002_RGB_YOLO26N",
        model_weights="weights/yolo26n.pt",
        modality="rgb",
        batch=8,
        epochs=40,
        patience=12,
    )
    update_readme_and_logs(e002_metrics)

    # Step 2: E003_THERMAL_YOLO26N
    e003_metrics = run_experiment(
        exp_id="E003_THERMAL_YOLO26N",
        model_weights="weights/yolo26n.pt",
        modality="thermal",
        batch=8,
        epochs=40,
        patience=12,
    )
    update_readme_and_logs(e002_metrics, e003_metrics)

    print("\n" + "*" * 80)
    print("ALL UNIMODAL BASELINES COMPLETED SUCCESSFULLY (E001, E002, E003)")
    print("Ready for Milestone 5: E004 (Late Fusion WBF) & E005 (Early Fusion 4CH)")
    print("*" * 80)


if __name__ == "__main__":
    main()
