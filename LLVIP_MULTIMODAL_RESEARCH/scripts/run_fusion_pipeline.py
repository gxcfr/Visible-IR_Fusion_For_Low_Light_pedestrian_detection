"""
Sequential Multimodal Fusion Pipeline Runner (E006 -> E007)
Executes Feature Concatenation baseline (E006) and Proposed Asymmetric Gated Fusion (E007)
sequentially, tracking metrics, saving checkpoints, and updating documentation.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def run_step(exp_id: str, model_type: str, batch: int = 4, epochs: int = 40, patience: int = 12):
    print("\n" + "#" * 80)
    print(f"PIPELINE STEP: Executing {exp_id} ({model_type.upper()})")
    print(f"Batch: {batch} | Epochs: {epochs} | Patience: {patience}")
    print("#" * 80 + "\n", flush=True)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT_DIR / "src" / "training" / "train_fusion_model.py"),
        "--exp_id", exp_id,
        "--model_type", model_type,
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--patience", str(patience),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR))
    duration = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Experiment {exp_id} failed with exit code {proc.returncode}")

    metrics_p = ROOT_DIR / "experiments" / exp_id / "metrics.json"
    if not metrics_p.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_p}")

    with open(metrics_p, "r") as f:
        m = json.load(f)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id} COMPLETED in {duration/60:.2f} min")
    print(f"mAP50:     {m['validation_metrics']['mAP50']:.4f}")
    print(f"mAP50-95:  {m['validation_metrics']['mAP50-95']:.4f}")
    print(f"Precision: {m['validation_metrics']['precision']:.4f}")
    print(f"Recall:    {m['validation_metrics']['recall']:.4f}")
    print(f"F1:        {m['validation_metrics']['f1_score']:.4f}")
    print(f"Latency:   {m['latency_metrics']['model_latency_mean_ms']:.2f} ms ({m['latency_metrics']['fps']:.1f} FPS)")
    print("=" * 80 + "\n", flush=True)

    # Automatically commit deliverables
    try:
        subprocess.run(["git", "add", f"experiments/{exp_id}/", f"results/metrics/{exp_id}_metrics.json"], cwd=str(ROOT_DIR))
        commit_msg = f"feat: complete {exp_id} (mAP50={m['validation_metrics']['mAP50']:.4f}, mAP50-95={m['validation_metrics']['mAP50-95']:.4f})"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(ROOT_DIR))
    except Exception as e:
        print(f"Git commit notice: {e}")

    return m


def main():
    print("=" * 80)
    print("STARTING MULTIMODAL FEATURE FUSION PIPELINE (E006 -> E007)")
    print("=" * 80)

    # Step 1: E006 Feature Concatenation
    e006_m = run_step(
        exp_id="E006_FEATURE_FUSION_CONCAT_YOLO26N",
        model_type="concat",
        batch=4,
        epochs=40,
        patience=12,
    )

    # Step 2: E007 Proposed Asymmetric Gated Fusion
    e007_m = run_step(
        exp_id="E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N",
        model_type="asymmetric",
        batch=4,
        epochs=40,
        patience=12,
    )

    print("\n" + "*" * 80)
    print("ALL EXPERIMENTS E001 THROUGH E007 HAVE BEEN COMPLETED SUCCESSFULLY!")
    print("*" * 80)


if __name__ == "__main__":
    main()
