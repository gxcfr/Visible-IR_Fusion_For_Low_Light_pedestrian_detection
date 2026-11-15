"""
Early Fusion 4-Channel YOLO26n Training & Evaluation Engine
Trains the 4-channel YOLO26n detector on synchronized [R, G, B, Thermal] inputs,
evaluating mAP50, mAP50-95, precision, recall, and hardware latency.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.paired_dataset import LLVIPPairedDataset
from src.models.baselines.early_fusion_4ch import build_4channel_yolo26
from src.evaluation.late_fusion_wbf import evaluate_detections
from ultralytics.utils.nms import non_max_suppression
from ultralytics.cfg import get_cfg


def early_fusion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate 4-channel tensors: RGB (3 channels) + Thermal (1 channel)."""
    four_channel_imgs = []
    batch_idx_list = []
    cls_list = []
    bbox_list = []

    for b_idx, item in enumerate(batch):
        # RGB is [3, H, W], thermal is [3, H, W] (grayscale duplicated across channels)
        rgb = item["rgb"]
        thermal_1ch = item["thermal"][0:1, :, :]  # [1, H, W]
        img_4ch = torch.cat([rgb, thermal_1ch], dim=0)  # [4, H, W]
        four_channel_imgs.append(img_4ch)

        targets = item["targets"]  # [num_boxes, 5] -> (cid, xc, yc, w, h)
        if len(targets) > 0:
            for tgt in targets:
                batch_idx_list.append(b_idx)
                cls_list.append(tgt[0].item())
                bbox_list.append(tgt[1:5].numpy())

    imgs = torch.stack(four_channel_imgs, dim=0)
    if len(bbox_list) > 0:
        batch_idx = torch.tensor(batch_idx_list, dtype=torch.int64)
        classes = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(1)
        bboxes = torch.tensor(np.array(bbox_list), dtype=torch.float32)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.int64)
        classes = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)

    return {
        "img": imgs,
        "batch_idx": batch_idx,
        "cls": classes,
        "bboxes": bboxes,
        "pair_ids": [item["pair_id"] for item in batch],
    }


def validate_4channel_model(model: nn.Module, val_loader: DataLoader, device: str = "cuda:0") -> Dict[str, float]:
    """Evaluate 4-channel model on validation set using ap_per_class."""
    model.eval()
    pred_boxes_list = []
    gt_boxes_list = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["img"].to(device)
            preds = model(imgs)

            # In eval mode YOLO returns (raw_preds, extra)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            nms_preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.7)

            batch_idx = batch["batch_idx"]
            classes = batch["cls"].squeeze(1)
            bboxes = batch["bboxes"]

            for i in range(len(imgs)):
                p = nms_preds[i].cpu().numpy()
                if len(p) > 0:
                    pred_boxes_list.append(p)
                else:
                    pred_boxes_list.append(np.zeros((0, 6), dtype=np.float32))

                mask = (batch_idx == i)
                if mask.sum() > 0:
                    b_cls = classes[mask].cpu().numpy()
                    b_box = bboxes[mask].cpu().numpy()
                    # convert xywh [0, 1] to xyxy [0, 640]
                    xc, yc, w, h = b_box[:, 0], b_box[:, 1], b_box[:, 2], b_box[:, 3]
                    x1 = (xc - w / 2.0) * 640.0
                    y1 = (yc - h / 2.0) * 640.0
                    x2 = (xc + w / 2.0) * 640.0
                    y2 = (yc + h / 2.0) * 640.0
                    gt_arr = np.column_stack([b_cls, x1, y1, x2, y2])
                    gt_boxes_list.append(gt_arr)
                else:
                    gt_boxes_list.append(np.zeros((0, 5), dtype=np.float32))

    return evaluate_detections(pred_boxes_list, gt_boxes_list)


def train_early_fusion(
    exp_id: str = "E005_EARLY_FUSION_4CH_YOLO26N",
    pretrained_weights: str = "weights/yolo26n.pt",
    batch_size: int = 8,
    epochs: int = 40,
    patience: int = 12,
    lr0: float = 0.001,
    weight_decay: float = 0.0005,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    exp_dir = ROOT_DIR / "experiments" / exp_id
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING EXPERIMENT: {exp_id}")
    print(f"Architecture:   YOLO26n 4-Channel Early Fusion")
    print(f"Pretrained:     {pretrained_weights}")
    print(f"Batch Size:     {batch_size} | Epochs: {epochs} | Patience: {patience}")
    print("=" * 80)

    # 1. Datasets & Loaders
    train_ds = LLVIPPairedDataset(partition="train", augment=True)
    val_ds = LLVIPPairedDataset(partition="val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=early_fusion_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=early_fusion_collate_fn
    )

    # 2. Build 4-channel model
    yolo_wrapper = build_4channel_yolo26(str(ROOT_DIR / pretrained_weights))
    model = yolo_wrapper.model.to(device)
    model.args = get_cfg()

    optimizer = AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0 * 0.01)
    scaler = torch.amp.GradScaler("cuda")

    best_fitness = 0.0
    best_epoch = 1
    patience_counter = 0
    history = []
    start_time = time.time()

    print(f"\nBeginning training across {len(train_loader)} batches per epoch...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_ep_start = time.time()

        for step, batch in enumerate(train_loader):
            batch["img"] = batch["img"].to(device, non_blocking=True)
            batch["batch_idx"] = batch["batch_idx"].to(device, non_blocking=True)
            batch["cls"] = batch["cls"].to(device, non_blocking=True)
            batch["bboxes"] = batch["bboxes"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss, loss_items = model.loss(batch)
                loss_scalar = loss.sum()

            scaler.scale(loss_scalar).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_scalar.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch {epoch:2d}/{epochs} | Step {step+1:4d}/{len(train_loader)} | "
                    f"Loss: {loss_scalar.item():.4f} (Avg: {epoch_loss/(step+1):.4f})",
                    flush=True
                )

        scheduler.step()
        ep_duration = time.time() - t_ep_start

        # Validation
        val_metrics = validate_4channel_model(model, val_loader, device=device)
        fitness = 0.1 * val_metrics["mAP50"] + 0.9 * val_metrics["mAP50-95"]

        row = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / len(train_loader), 4),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1_score"],
            "val_mAP50": val_metrics["mAP50"],
            "val_mAP50-95": val_metrics["mAP50-95"],
            "fitness": round(fitness, 4),
            "epoch_seconds": round(ep_duration, 1),
        }
        history.append(row)

        print(
            f"--> Epoch {epoch:2d} Done ({ep_duration:.1f}s) | "
            f"mAP50: {val_metrics['mAP50']:.4f} | mAP50-95: {val_metrics['mAP50-95']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1_score']:.4f}",
            flush=True
        )

        # Save last checkpoint
        torch.save(model.state_dict(), weights_dir / "last.pt")

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"    * New best model saved (fitness: {fitness:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Patience of {patience} epochs reached without fitness improvement.")
                break

    total_duration = time.time() - start_time
    pd.DataFrame(history).to_csv(exp_dir / "results.csv", index=False)

    # Benchmark Latency with Best Weights
    print("\nBenchmarking batch-1 latency with best checkpoint...")
    model.load_state_dict(torch.load(weights_dir / "best.pt"))
    model.eval()

    dummy_input = torch.zeros((1, 4, 640, 640), device=device)
    for _ in range(50):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    latency_mean = round(float(np.mean(times)), 2)
    fps = round(1000.0 / latency_mean, 2)
    params_count = sum(p.numel() for p in model.parameters())

    best_metrics = [h for h in history if h["epoch"] == best_epoch][0]

    experiment_metrics = {
        "experiment_id": exp_id,
        "strategy": "early_fusion_4ch",
        "architecture": "yolo26n_4ch",
        "status": "COMPLETED",
        "training_duration_seconds": round(total_duration, 2),
        "training_duration_minutes": round(total_duration / 60, 2),
        "stopping_epoch": len(history),
        "best_epoch": best_epoch,
        "validation_metrics": {
            "precision": best_metrics["val_precision"],
            "recall": best_metrics["val_recall"],
            "f1_score": best_metrics["val_f1"],
            "mAP50": best_metrics["val_mAP50"],
            "mAP50-95": best_metrics["val_mAP50-95"],
        },
        "latency_metrics": {
            "device": device,
            "input_resolution": "640x640",
            "batch_size": 1,
            "model_parameters": params_count,
            "model_latency_mean_ms": latency_mean,
            "fps": fps,
        },
        "checkpoint_paths": {
            "best": str(weights_dir / "best.pt"),
            "last": str(weights_dir / "last.pt"),
        },
    }

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(experiment_metrics, f, indent=4)
    with open(ROOT_DIR / "results" / "metrics" / f"{exp_id}_metrics.json", "w") as f:
        json.dump(experiment_metrics, f, indent=4)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_id} BENCHMARK COMPLETE")
    print(f"mAP50:     {best_metrics['val_mAP50']:.4f}")
    print(f"mAP50-95:  {best_metrics['val_mAP50-95']:.4f}")
    print(f"Precision: {best_metrics['val_precision']:.4f}")
    print(f"Recall:    {best_metrics['val_recall']:.4f}")
    print(f"F1:        {best_metrics['val_f1']:.4f}")
    print(f"Latency:   {latency_mean} ms ({fps} FPS)")
    print("=" * 80)

    return experiment_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    try:
        train_early_fusion(
            epochs=args.epochs,
            batch_size=args.batch,
            patience=args.patience,
        )
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"\nFATAL TRAINING ERROR:\n{err}", flush=True)
        with open(ROOT_DIR / "experiments" / "early_fusion_fatal_error.log", "w") as f:
            f.write(err)
        sys.exit(1)
