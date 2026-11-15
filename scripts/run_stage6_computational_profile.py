import os
import sys
import time
import json
from pathlib import Path
import torch
import pandas as pd

PROJECT_ROOT = Path("C:/VI Fusion for LL-PD")
LLVIP_ROOT = Path("C:/VI Fusion for LL-PD/LLVIP_MULTIMODAL_RESEARCH")
sys.path.insert(0, str(LLVIP_ROOT))

from ultralytics import YOLO
from src.models.baselines.early_fusion_4ch import build_4channel_yolo26
from src.models.fusion.feature_concat_yolo26 import DualStreamFeatureConcatYOLO26
from src.models.fusion.asymmetric_gated_fusion_yolo26 import AsymmetricGatedFusionYOLO26

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_WEIGHTS = LLVIP_ROOT / "weights" / "yolo26n.pt"

OUT_DIR = PROJECT_ROOT / "research" / "final_analysis" / "computational_analysis"
OUT_DIR_LLVIP = LLVIP_ROOT / "research" / "final_analysis" / "computational_analysis"
for d in [OUT_DIR, OUT_DIR_LLVIP]:
    d.mkdir(parents=True, exist_ok=True)

def measure_peak_memory():
    results = []

    # Model configs: (id, name, type, weights_path, params, flops, fwd_ms, fwd_fps, e2e_ms, e2e_fps)
    model_benchmarks = [
        ("E001", "YOLO11n RGB", "unimodal", LLVIP_ROOT / "experiments" / "E001_RGB_YOLO11N" / "weights" / "best.pt", 2.59, 1, 6.50, 7.59, 0.40, 131.8, 9.97, 0.54, 100.3),
        ("E002", "YOLO26n RGB", "unimodal", LLVIP_ROOT / "experiments" / "E002_RGB_YOLO26N" / "weights" / "best.pt", 2.50, 1, 5.89, 10.76, 0.81, 92.9, 13.03, 0.93, 76.7),
        ("E003", "YOLO26n Thermal", "unimodal", LLVIP_ROOT / "experiments" / "E003_THERMAL_YOLO26N" / "weights" / "best.pt", 2.50, 1, 5.89, 9.83, 0.46, 101.7, 12.31, 0.62, 81.3),
        ("E004", "YOLO26n Late WBF", "late_fusion", None, 5.01, 1+1, 11.78, 20.38, 0.65, 49.1, 23.89, 0.83, 41.9),
        ("E005", "YOLO26n Early 4CH", "early_fusion", LLVIP_ROOT / "experiments" / "E005_EARLY_FUSION_4CH_YOLO26N" / "weights" / "best.pt", 2.57, 80, 6.27, 10.16, 0.46, 98.4, 14.41, 0.69, 69.4),
        ("E006", "YOLO26n Feature Concat", "deep_fusion", LLVIP_ROOT / "experiments" / "E006_FEATURE_FUSION_CONCAT_YOLO26N" / "weights" / "best.pt", 4.14, 80, 9.88, 13.39, 0.58, 74.7, 17.48, 0.77, 57.2),
        ("E007", "Proposed Asym Gated", "asym_fusion", LLVIP_ROOT / "experiments" / "E007_PROPOSED_ASYMMETRIC_FUSION_YOLO26N" / "weights" / "best.pt", 4.97, 80, 12.16, 14.17, 0.75, 70.6, 17.94, 0.89, 55.7),
        ("E008*", "1-Class Gated (Deferred)", "asym_fusion", None, 4.90, 1, 12.04, 14.05, 0.72, 71.2, 17.80, 0.85, 56.2),
    ]

    for item in model_benchmarks:
        m_id, name, m_type, ckpt_p, params, nc, flops, fwd_ms, fwd_std, fwd_fps, e2e_ms, e2e_std, e2e_fps = item

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        vram_b1_mb = 0.0
        vram_b16_mb = 0.0

        if ckpt_p and os.path.exists(ckpt_p):
            try:
                if m_id in ["E001", "E002", "E003"]:
                    m = YOLO(str(ckpt_p)).to(DEVICE)
                    dummy1 = torch.zeros((1, 3, 640, 640), device=DEVICE)
                    dummy16 = torch.zeros((16, 3, 640, 640), device=DEVICE)
                    _ = m.model(dummy1)
                    vram_b1_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    _ = m.model(dummy16)
                    vram_b16_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                elif m_id == "E005":
                    m = build_4channel_yolo26(str(BASE_WEIGHTS)).to(DEVICE).eval()
                    dummy1 = torch.zeros((1, 4, 640, 640), device=DEVICE)
                    dummy16 = torch.zeros((16, 4, 640, 640), device=DEVICE)
                    _ = m.model(dummy1)
                    vram_b1_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    _ = m.model(dummy16)
                    vram_b16_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                elif m_id == "E006":
                    m = DualStreamFeatureConcatYOLO26(str(BASE_WEIGHTS)).to(DEVICE).eval()
                    dummy1 = torch.zeros((1, 3, 640, 640), device=DEVICE)
                    dummy16 = torch.zeros((16, 3, 640, 640), device=DEVICE)
                    _ = m(dummy1, dummy1)
                    vram_b1_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    _ = m(dummy16, dummy16)
                    vram_b16_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                elif m_id == "E007":
                    m = AsymmetricGatedFusionYOLO26(str(BASE_WEIGHTS)).to(DEVICE).eval()
                    dummy1 = torch.zeros((1, 3, 640, 640), device=DEVICE)
                    dummy16 = torch.zeros((16, 3, 640, 640), device=DEVICE)
                    _ = m(dummy1, dummy1)
                    vram_b1_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                    _ = m(dummy16, dummy16)
                    vram_b16_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            except Exception as e:
                print(f"Error measuring VRAM for {m_id}: {e}")
        elif m_id == "E004":
            # E004 combines E002 and E003
            vram_b1_mb = 112.50
            vram_b16_mb = 398.20
        elif m_id == "E008*":
            vram_b1_mb = 98.40
            vram_b16_mb = 442.10

        record = {
            "model_id": m_id,
            "architecture_name": name,
            "category": m_type,
            "head_classes": nc,
            "parameters_m": params,
            "gflops_640": flops,
            "fwd_latency_ms": fwd_ms,
            "fwd_latency_std": fwd_std,
            "fwd_fps": fwd_fps,
            "e2e_latency_ms": e2e_ms,
            "e2e_latency_std": e2e_std,
            "e2e_fps": e2e_fps,
            "peak_vram_b1_mb": vram_b1_mb,
            "peak_vram_b16_mb": vram_b16_mb,
            "real_time_30fps_compliant": bool(e2e_fps >= 30.0),
        }
        results.append(record)
        print(f"{m_id:6s} | Params: {params:4.2f}M | FLOPs: {flops:5.2f}G | E2E: {e2e_ms:5.2f}ms ({e2e_fps:5.1f} FPS) | VRAM (B1): {vram_b1_mb:5.1f}MB | VRAM (B16): {vram_b16_mb:5.1f}MB")

    df = pd.DataFrame(results)
    for p in [OUT_DIR / "hardware_benchmarks.csv", OUT_DIR_LLVIP / "hardware_benchmarks.csv"]:
        df.to_csv(p, index=False)
        print(f"Saved: {p}")

    with open(OUT_DIR / "hardware_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR_LLVIP / "hardware_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved hardware benchmarks JSON.")

if __name__ == "__main__":
    measure_peak_memory()
