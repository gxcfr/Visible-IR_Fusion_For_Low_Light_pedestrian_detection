"""
Augmentation & DataLoader Batch Visualizer
Draws transformed bounding boxes onto augmented RGB and Thermal tensor pairs
to verify geometric synchronization and box alignment.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float32 tensor [0, 1] to (H, W, 3) BGR uint8."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_yolo_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    color=(0, 255, 0),
    thickness=2,
) -> np.ndarray:
    """Draw normalized [xc, yc, w, h] boxes onto an image."""
    img_draw = img.copy()
    h, w = img.shape[:2]
    for box in boxes:
        xc, yc, bw, bh = box
        xmin = int(round((xc - bw / 2.0) * w))
        ymin = int(round((yc - bh / 2.0) * h))
        xmax = int(round((xc + bw / 2.0) * w))
        ymax = int(round((yc + bh / 2.0) * h))

        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(
            img_draw,
            "person",
            (xmin, max(15, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return img_draw


def verify_batch_visualizations(
    output_path: str = None,
    num_samples: int = 4,
):
    """Fetch a training batch and visualize transformed boxes on augmented modalities."""
    from src.data.paired_dataset import create_paired_dataloader

    out_p = Path(output_path) if output_path else ROOT_DIR / "results" / "qualitative" / "augmentation_verification.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    loader = create_paired_dataloader(partition="train", batch_size=num_samples, shuffle=True)
    batch = next(iter(loader))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    plt.suptitle("Augmentation & Bounding-Box Synchronization Verification", fontsize=14, fontweight="bold", y=0.99)

    for i in range(num_samples):
        rgb_img = tensor_to_cv2(batch["rgb"][i])
        ir_img = tensor_to_cv2(batch["thermal"][i])
        pair_id = batch["pair_ids"][i]
        scene_id = batch["scene_ids"][i]

        # Extract targets for this batch item: targets[:, 0] == i
        mask = batch["targets"][:, 0] == i
        sample_targets = batch["targets"][mask]
        sample_boxes = sample_targets[:, 2:6].cpu().numpy() if len(sample_targets) > 0 else np.zeros((0, 4))

        # Overlay bounding boxes
        annot_rgb = draw_yolo_boxes(rgb_img, sample_boxes, color=(0, 255, 0), thickness=2)
        annot_ir = draw_yolo_boxes(ir_img, sample_boxes, color=(0, 255, 255), thickness=2)

        # Convert to RGB for matplotlib display
        annot_rgb_disp = cv2.cvtColor(annot_rgb, cv2.COLOR_BGR2RGB)
        annot_ir_disp = cv2.cvtColor(annot_ir, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(annot_rgb_disp)
        axes[i, 0].set_title(f"Sample {i+1}: RGB (Visible) | Pair {pair_id} (Scene {scene_id}) | Boxes: {len(sample_boxes)}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(annot_ir_disp)
        axes[i, 1].set_title(f"Sample {i+1}: Infrared (Thermal) | Pair {pair_id} (Scene {scene_id}) | Boxes: {len(sample_boxes)}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(str(out_p), dpi=300)
    plt.close()
    print(f"Augmentation verification figure saved to: {out_p}")


if __name__ == "__main__":
    verify_batch_visualizations()
