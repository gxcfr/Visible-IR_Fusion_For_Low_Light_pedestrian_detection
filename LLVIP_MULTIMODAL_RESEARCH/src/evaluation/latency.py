"""
Latency & Throughput Benchmark Engine for Pedestrian Detectors
Measures batch-1 inference latency with GPU synchronization and warmup.
Distinguishes pure model forward latency from end-to-end inference latency.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
from ultralytics import YOLO


def benchmark_latency(
    model_path: str,
    img_size: int = 640,
    device: str = "cuda:0",
    warmup_runs: int = 50,
    test_runs: int = 200,
) -> Dict[str, Any]:
    """
    Perform rigorous batch-1 latency benchmark on GPU.
    Args:
        model_path: path to .pt model checkpoint
        img_size: input spatial dimension (default 640)
        device: compute device
        warmup_runs: number of un-timed warmup passes
        test_runs: number of timed passes for statistics
    Returns:
        dict containing latency stats (mean, std, min, max, FPS)
    """
    if not torch.cuda.is_available() and "cuda" in device:
        device = "cpu"

    model = YOLO(model_path)
    # Extract underlying PyTorch nn.Module
    net = model.model.to(device)
    net.eval()

    # Dummy tensor for batch-1 inference
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Warmup passes
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = net(dummy_input)
            if "cuda" in device:
                torch.cuda.synchronize()

    # 2. Pure Model Forward Latency
    model_times_ms = []
    with torch.no_grad():
        for _ in range(test_runs):
            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = net(dummy_input)

            if "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model_times_ms.append((t1 - t0) * 1000.0)

    # 3. End-to-End Latency (using Ultralytics predict pipeline on dummy image)
    dummy_numpy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    e2e_times_ms = []
    # Warmup e2e
    for _ in range(10):
        _ = model.predict(dummy_numpy, device=device, verbose=False)
        if "cuda" in device:
            torch.cuda.synchronize()

    for _ in range(min(50, test_runs)):
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model.predict(dummy_numpy, device=device, verbose=False)

        if "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_times_ms.append((t1 - t0) * 1000.0)

    model_mean_ms = float(np.mean(model_times_ms))
    model_std_ms = float(np.std(model_times_ms))
    e2e_mean_ms = float(np.mean(e2e_times_ms))
    fps = round(1000.0 / model_mean_ms, 2) if model_mean_ms > 0 else 0.0

    # Model parameters & GFLOPs
    params_count = sum(p.numel() for p in net.parameters())

    return {
        "device": device,
        "input_resolution": f"{img_size}x{img_size}",
        "batch_size": 1,
        "model_parameters": params_count,
        "model_latency_mean_ms": round(model_mean_ms, 3),
        "model_latency_std_ms": round(model_std_ms, 3),
        "model_latency_min_ms": round(float(np.min(model_times_ms)), 3),
        "model_latency_max_ms": round(float(np.max(model_times_ms)), 3),
        "e2e_latency_mean_ms": round(e2e_mean_ms, 3),
        "fps": fps,
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    res = benchmark_latency(args.model, device=args.device)
    for k, v in res.items():
        print(f"{k:25s}: {v}")
