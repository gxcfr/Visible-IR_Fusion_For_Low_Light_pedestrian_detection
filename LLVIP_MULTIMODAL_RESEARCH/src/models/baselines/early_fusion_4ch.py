"""
Early Fusion 4-Channel YOLO26n Architecture
Adapts YOLO26n stem to take 4 input channels [R, G, B, Thermal],
transferring pretrained RGB weights to channels 0-2 and RGB-mean to channel 3.
"""

import copy
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def build_4channel_yolo26(pretrained_weights: str) -> YOLO:
    """
    Load YOLO26n model, replace first conv layer with 4-channel input,
    and preserve pretrained weight transfer.
    """
    base_model = YOLO(str(pretrained_weights))
    torch_model = base_model.model
    
    # Locate first conv module: torch_model.model[0].conv
    first_conv = torch_model.model[0].conv
    assert isinstance(first_conv, nn.Conv2d), f"Expected Conv2d, got {type(first_conv)}"
    
    old_weight = first_conv.weight.data  # [out_c, 3, k, k]
    out_c, in_c, k_h, k_w = old_weight.shape
    
    # Create new Conv2d with 4 input channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_c,
        kernel_size=(k_h, k_w),
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
    )
    
    # Transfer pretrained RGB weights to channels 0-2
    new_conv.weight.data[:, 0:3, :, :] = old_weight
    # Initialize channel 3 (Thermal) with the channel-wise average of RGB
    new_conv.weight.data[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)
    
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()
        
    torch_model.model[0].conv = new_conv
    base_model.model = torch_model
    
    print(f"Successfully modified YOLO26n stem to 4 input channels: {new_conv.weight.shape}")
    return base_model
