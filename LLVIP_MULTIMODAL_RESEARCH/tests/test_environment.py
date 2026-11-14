"""
Test Suite for Milestone 1: Environment & Hardware Verification
"""

import sys
import torch
import pytest


def test_python_version():
    """Ensure Python version is >= 3.10."""
    assert sys.version_info >= (3, 10), f"Python version {sys.version} is below required 3.10"


def test_cuda_availability():
    """Ensure CUDA is available for acceleration."""
    assert torch.cuda.is_available(), "CUDA must be available for GPU acceleration"


def test_gpu_device():
    """Verify NVIDIA GPU is detected with at least 6GB VRAM."""
    device_name = torch.cuda.get_device_name(0)
    assert "RTX" in device_name or "NVIDIA" in device_name, f"Unexpected GPU: {device_name}"
    
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / (1024 ** 3)
    assert total_memory_gb >= 6.0, f"Expected at least 6GB VRAM, got {total_memory_gb:.2f} GB"


def test_amp_support():
    """Verify CUDA device supports mixed precision (compute capability >= 7.0)."""
    cap = torch.cuda.get_device_capability(0)
    assert cap[0] >= 7, f"Compute capability {cap} does not meet requirements for Tensor Cores"


def test_ultralytics_import():
    """Ensure Ultralytics framework is importable."""
    import ultralytics
    assert hasattr(ultralytics, "__version__"), "Ultralytics version not accessible"


def test_opencv_import():
    """Ensure OpenCV is available."""
    import cv2
    assert hasattr(cv2, "__version__"), "OpenCV version not accessible"
