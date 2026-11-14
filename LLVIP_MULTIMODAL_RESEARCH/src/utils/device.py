"""
Hardware and Device Management Utility
Ensures reproducible GPU identification, VRAM monitoring, and AMP support.
"""

import sys
import torch
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Inspect and return current compute device details."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "compute_capability": None,
        "total_memory_mb": 0.0,
        "allocated_memory_mb": 0.0,
        "reserved_memory_mb": 0.0,
        "amp_supported": False,
    }

    if torch.cuda.is_available():
        device_idx = 0
        info["device_name"] = torch.cuda.get_device_name(device_idx)
        cap = torch.cuda.get_device_capability(device_idx)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        props = torch.cuda.get_device_properties(device_idx)
        info["total_memory_mb"] = round(props.total_memory / (1024 ** 2), 2)
        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated(device_idx) / (1024 ** 2), 2)
        info["reserved_memory_mb"] = round(torch.cuda.memory_reserved(device_idx) / (1024 ** 2), 2)
        info["amp_supported"] = cap[0] >= 7  # Tensor cores present

    return info


def print_device_summary():
    """Print clean summary table of system device capabilities."""
    info = get_device_info()
    print("=" * 60)
    print("DEVICE & ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python Version:       {info['python_version']}")
    print(f"PyTorch Version:      {info['torch_version']}")
    print(f"CUDA Available:       {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"Device Name:          {info['device_name']}")
        print(f"Compute Capability:   {info['compute_capability']}")
        print(f"Total VRAM:           {info['total_memory_mb']:.1f} MB (~{info['total_memory_mb']/1024:.2f} GB)")
        print(f"Allocated VRAM:       {info['allocated_memory_mb']:.1f} MB")
        print(f"Reserved VRAM:        {info['reserved_memory_mb']:.1f} MB")
        print(f"AMP / Tensor Cores:   {info['amp_supported']}")
    else:
        print("WARNING: CUDA is NOT available. Running on CPU.")
    print("=" * 60)


if __name__ == "__main__":
    print_device_summary()
