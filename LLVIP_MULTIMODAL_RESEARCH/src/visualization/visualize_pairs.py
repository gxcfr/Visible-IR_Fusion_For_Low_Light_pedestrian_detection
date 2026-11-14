"""
Visualizer for LLVIP Multimodal Image Pairs
Generates publication-quality side-by-side RGB and Infrared images with bounding boxes overlaid.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET


def draw_voc_boxes(image: np.ndarray, xml_path: str, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Overlay bounding boxes from Pascal VOC XML onto an image."""
    img_copy = image.copy()
    if not os.path.exists(xml_path):
        return img_copy

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name")
            label = name.text.strip() if name is not None and name.text else "person"
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                # Draw bounding box
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, thickness)
                # Draw text label tag
                text = f"{label}"
                cv2.putText(
                    img_copy,
                    text,
                    (xmin, max(20, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    except Exception as e:
        print(f"Error drawing boxes from {xml_path}: {e}")

    return img_copy


def create_paired_gallery(
    manifest_path: str,
    output_dir: str = "results/qualitative/paired_gallery",
    num_samples: int = 50,
    seed: int = 42,
):
    """Generate side-by-side RGB and Infrared pair gallery with overlaid bounding boxes."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    paired_df = df[df["pairing_status"] == "PAIRED"].copy()

    # Sample representative diverse pairs
    # E.g. various person counts, train and test
    random.seed(seed)
    
    # Sort or sample ensuring both train and test samples are included
    train_samples = paired_df[paired_df["split"] == "train"].sample(
        n=min(int(num_samples * 0.7), len(paired_df[paired_df["split"] == "train"])),
        random_state=seed,
    )
    test_samples = paired_df[paired_df["split"] == "test"].sample(
        n=min(int(num_samples * 0.3), len(paired_df[paired_df["split"] == "test"])),
        random_state=seed,
    )
    selected = pd.concat([train_samples, test_samples]).reset_index(drop=True)

    print(f"Generating {len(selected)} side-by-side gallery pairs...")

    for idx, row in selected.iterrows():
        pair_id = str(row["pair_id"]).zfill(6)
        split = row["split"]
        rgb_path = row["rgb_path"]
        ir_path = row["infrared_path"]
        xml_path = row["annotation_path"]
        instance_count = row["person_instance_count"]

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)

        if img_rgb is None or img_ir is None:
            continue

        # Draw annotations: Green (0, 255, 0) for RGB, Yellow (0, 255, 255) for IR
        annot_rgb = draw_voc_boxes(img_rgb, xml_path, color=(0, 255, 0), thickness=2)
        annot_ir = draw_voc_boxes(img_ir, xml_path, color=(0, 255, 255), thickness=2)

        # Add headers onto images
        h, w = annot_rgb.shape[:2]
        header_height = 40
        canvas = np.zeros((h + header_height, w * 2 + 10, 3), dtype=np.uint8)

        # Title bar
        cv2.putText(
            canvas,
            f"Visible (RGB) | Pair: {pair_id} | Split: {split} | Pedestrians: {instance_count}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Infrared (Thermal) | Pair: {pair_id} | Split: {split} | Pedestrians: {instance_count}",
            (w + 30, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        canvas[header_height : header_height + h, 0:w] = annot_rgb
        canvas[header_height : header_height + h, w + 10 : w * 2 + 10] = annot_ir

        save_name = out_path / f"pair_{idx+1:03d}_{split}_{pair_id}.jpg"
        cv2.imwrite(str(save_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Gallery generated: {len(selected)} images saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="results/metrics/dataset_pairing_manifest.csv")
    parser.add_argument("--output", type=str, default="results/qualitative/paired_gallery")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    create_paired_gallery(args.manifest, args.output, args.count)
