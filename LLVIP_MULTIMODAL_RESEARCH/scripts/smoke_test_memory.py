"""
VRAM and Memory Smoke Test for Milestone 4 Unimodal Baselines
Tests Batch Size 8 vs Batch Size 4 on RTX 4060 Laptop GPU (8GB VRAM)
Monitors peak VRAM, GPU temperature, and forward+backward pass stability.
"""

import sys
import os
import subprocess
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Patch ultralytics label resolver
import ultralytics.data.dataset as ds_module
import ultralytics.data.utils as utils_module
labels_dir = (ROOT_DIR / "datasets" / "labels").resolve()
def custom_img2label_paths(img_paths):
    return [str(labels_dir / f"{Path(x).stem}.txt") for x in img_paths]
ds_module.img2label_paths = custom_img2label_paths
utils_module.img2label_paths = custom_img2label_paths

from ultralytics import YOLO


def get_gpu_metrics():
    """Query nvidia-smi for current temperature and memory usage."""
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        temp, mem_used, mem_total = [float(x.strip()) for x in out.split(",")]
        return {"temperature_c": temp, "memory_used_mb": mem_used, "memory_total_mb": mem_total}
    except Exception as e:
        return {"temperature_c": None, "memory_used_mb": None, "memory_total_mb": None, "error": str(e)}


def run_smoke_test(batch_size: int = 8, model_name: str = "yolo11n.pt"):
    """Run a 1-epoch / 10-batch smoke test with explicit AdamW and AMP."""
    print("=" * 60)
    print(f"SMOKE TEST: {model_name} | Batch Size: {batch_size} | Device: cuda:0")
    print("=" * 60)

    pre_metrics = get_gpu_metrics()
    print(f"Pre-test GPU Temp:  {pre_metrics['temperature_c']} C")
    print(f"Pre-test VRAM Used: {pre_metrics['memory_used_mb']} MB / {pre_metrics['memory_total_mb']} MB")

    weights_path = ROOT_DIR / "weights" / model_name
    model = YOLO(str(weights_path))

    data_config = {
        "path": str((ROOT_DIR / "datasets").resolve()),
        "train": str((ROOT_DIR / "datasets" / "train_rgb.txt").resolve()),
        "val": str((ROOT_DIR / "datasets" / "val_rgb.txt").resolve()),
        "names": {0: "person"},
        "nc": 1,
    }

    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(data_config, f)
        temp_yaml = f.name

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    try:
        # Run short 1-epoch test with 10 batches to check peak VRAM
        results = model.train(
            data=temp_yaml,
            epochs=1,
            batch=batch_size,
            imgsz=640,
            device="0",
            workers=2,
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,
            amp=True,
            seed=42,
            val=False,
            plots=False,
            save=False,
            exist_ok=True,
            project=str(ROOT_DIR / "experiments" / "smoke_test"),
            name="smoke_run",
        )
        post_metrics = get_gpu_metrics()
        peak_torch_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        print("\n" + "=" * 60)
        print("SMOKE TEST RESULTS")
        print("=" * 60)
        print(f"Status:             SUCCESS (No OOM)")
        print(f"Peak PyTorch Alloc: {peak_torch_mb:.1f} MB")
        print(f"Peak PyTorch Reserv:{peak_reserved_mb:.1f} MB")
        print(f"Post-test GPU Temp: {post_metrics['temperature_c']} C")
        print(f"Post-test VRAM Used:{post_metrics['memory_used_mb']} MB / {post_metrics['memory_total_mb']} MB")
        print("=" * 60)
        return True, peak_torch_mb, post_metrics

    except torch.cuda.OutOfMemoryError as oom:
        print(f"OOM ERROR with Batch Size {batch_size}: {oom}")
        return False, None, get_gpu_metrics()
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e
    finally:
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    args = parser.parse_args()
    success, peak_vram, metrics = run_smoke_test(batch_size=args.batch, model_name=args.model)
