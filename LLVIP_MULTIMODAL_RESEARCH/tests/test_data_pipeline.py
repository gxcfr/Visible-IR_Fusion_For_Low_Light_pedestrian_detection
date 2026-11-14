"""
Unit Tests for Multimodal Data Pipeline, Augmentations, and DataLoader
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.augmentation import (
    letterbox_image,
    transform_boxes_letterbox,
    SynchronizedMultimodalAugmentor,
)
from src.data.paired_dataset import LLVIPPairedDataset, paired_collate_fn


def test_letterbox_dimensions():
    """Verify aspect-ratio preserving letterboxing outputs exact target shape."""
    img = np.zeros((1024, 1280, 3), dtype=np.uint8)
    target_shape = (640, 640)
    padded, r, (dw, dh) = letterbox_image(img, target_shape=target_shape)

    assert padded.shape == (640, 640, 3), f"Unexpected letterboxed shape: {padded.shape}"
    assert r == 640 / 1280, f"Expected ratio {640/1280}, got {r}"
    assert dw == 0.0, f"Expected dw=0 for 1280x1024 scaling to 640x640, got {dw}"
    assert dh == (640 - 1024 * r) / 2, f"Unexpected dh: {dh}"


def test_bounding_box_letterbox_transformation():
    """Verify normalized boxes transform accurately through letterbox."""
    orig_shape = (1024, 1280)
    target_shape = (640, 640)
    r = 640 / 1280
    pad = (0.0, (640 - 1024 * r) / 2)

    # Box centered in original image: [0.5, 0.5, 0.2, 0.2]
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    transformed = transform_boxes_letterbox(boxes, orig_shape, target_shape, r, pad)

    assert transformed.shape == (1, 4)
    # x-center should still be 0.5 because dw=0
    assert abs(transformed[0, 0] - 0.5) < 1e-4, f"Expected xc=0.5, got {transformed[0, 0]}"
    # y-center should still be 0.5 because padding is symmetric
    assert abs(transformed[0, 1] - 0.5) < 1e-4, f"Expected yc=0.5, got {transformed[0, 1]}"
    # All values must remain in [0, 1]
    assert (transformed >= 0.0).all() and (transformed <= 1.0).all()


def test_synchronized_horizontal_flip():
    """Verify synchronized horizontal flip reflects both modalities and flips xc."""
    augmentor = SynchronizedMultimodalAugmentor(target_size=(640, 640), hflip_prob=1.0, photometric_rgb=False)
    img_rgb = np.zeros((1024, 1280, 3), dtype=np.uint8)
    img_ir = np.zeros((1024, 1280, 3), dtype=np.uint8)
    boxes = np.array([[0.3, 0.5, 0.1, 0.2]], dtype=np.float32)

    _, _, flipped_boxes = augmentor(img_rgb, img_ir, boxes, augment=True)
    # xc should be flipped to 1.0 - 0.3 = 0.7
    assert abs(flipped_boxes[0, 0] - 0.7) < 1e-4, f"Expected xc=0.7 after hflip, got {flipped_boxes[0, 0]}"


def test_paired_dataset_loading():
    """Verify LLVIPPairedDataset item retrieval and modality alignment."""
    dataset = LLVIPPairedDataset(partition="val", target_size=(640, 640), augment=False)
    assert len(dataset) == 1803, f"Expected 1,803 validation samples, got {len(dataset)}"

    sample = dataset[0]
    assert "rgb" in sample and "thermal" in sample and "targets" in sample
    assert sample["rgb"].shape == (3, 640, 640), f"Unexpected RGB tensor shape: {sample['rgb'].shape}"
    assert sample["thermal"].shape == (3, 640, 640), f"Unexpected Thermal tensor shape: {sample['thermal'].shape}"
    assert sample["rgb"].dtype == torch.float32
    assert sample["thermal"].dtype == torch.float32
    assert sample["rgb"].min() >= 0.0 and sample["rgb"].max() <= 1.0
    assert sample["thermal"].min() >= 0.0 and sample["thermal"].max() <= 1.0


def test_paired_collate_fn():
    """Verify batch collation creates expected tensor dimensions and target batch indices."""
    dataset = LLVIPPairedDataset(partition="val", target_size=(640, 640), augment=False)
    batch = paired_collate_fn([dataset[0], dataset[1]])

    assert batch["rgb"].shape == (2, 3, 640, 640)
    assert batch["thermal"].shape == (2, 3, 640, 640)
    assert batch["batch_size"] == 2
    if len(batch["targets"]) > 0:
        assert batch["targets"].shape[1] == 6, f"Expected 6 columns (batch_idx, cid, xc, yc, w, h)"
        assert batch["targets"][:, 0].max() <= 1, "Batch index exceeds batch size"
