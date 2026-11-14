"""
Unit Tests for YOLO11n and YOLO26n Pretrained Weight Compatibility
"""

from pathlib import Path
import pytest
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_yolo11n_offline_loading():
    """Verify official YOLO11n checkpoint loads and executes forward pass."""
    w_path = ROOT_DIR / "weights" / "yolo11n.pt"
    assert w_path.exists(), f"Pretrained weight file not found: {w_path}"

    model = YOLO(str(w_path))
    assert model.model is not None, "Model failed to instantiate"

    dummy = torch.zeros(1, 3, 640, 640)
    out = model.model(dummy)
    assert out is not None, "Forward pass failed"


def test_yolo26n_offline_loading():
    """Verify official YOLO26n checkpoint loads and executes forward pass."""
    w_path = ROOT_DIR / "weights" / "yolo26n.pt"
    assert w_path.exists(), f"Pretrained weight file not found: {w_path}"

    model = YOLO(str(w_path))
    assert model.model is not None, "Model failed to instantiate"

    dummy = torch.zeros(1, 3, 640, 640)
    out = model.model(dummy)
    assert out is not None, "Forward pass failed"
