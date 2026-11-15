"""
Proposed Thermal-Primary Asymmetric Reliability-Aware Gated Fusion Architecture (E007)
Treats thermal infrared as the primary semantic stream while adaptively gating
visible feature injection based on dynamic cross-modal reliability assessment.
"""

import copy
import torch
import torch.nn as nn
from ultralytics import YOLO


class ReliabilityGatingModule(nn.Module):
    """
    Computes spatially and channel-wise dynamic gating weights G_V in [0, 1]
    to modulate the injection of visible features into the primary thermal backbone.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat_thermal: torch.Tensor, feat_rgb: torch.Tensor) -> torch.Tensor:
        proj_rgb = self.proj(feat_rgb)
        concat = torch.cat([feat_thermal, proj_rgb], dim=1)
        gate = self.gate_conv(concat)
        # Residual thermal injection modulated by reliability gate
        return feat_thermal + gate * proj_rgb


class AsymmetricGatedFusionYOLO26(nn.Module):
    def __init__(self, pretrained_weights: str):
        super().__init__()
        base_yolo = YOLO(pretrained_weights)
        m = base_yolo.model.model

        # Thermal backbone (Primary semantic stream)
        self.backbone_thermal = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11)])
        # RGB backbone (Auxiliary stream)
        self.backbone_rgb = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11)])

        # Multi-scale Reliability Gating Modules at P3, P4, P5
        self.gate_p3 = ReliabilityGatingModule(channels=128)
        self.gate_p4 = ReliabilityGatingModule(channels=128)
        self.gate_p5 = ReliabilityGatingModule(channels=256)

        # Neck and Head layers (layers 11 through 23)
        self.neck_and_head = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11, len(m))])

        self.loss = base_yolo.model.loss
        self.init_criterion = base_yolo.model.init_criterion
        self.args = base_yolo.model.args

    def extract_features(self, x: torch.Tensor, backbone: nn.ModuleList):
        p3, p4, p5 = None, None, None
        feat = x
        for i, layer in enumerate(backbone):
            feat = layer(feat)
            if i == 4:
                p3 = feat
            elif i == 6:
                p4 = feat
            elif i == 10:
                p5 = feat
        return p3, p4, p5

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor = None):
        if thermal is None and rgb.shape[1] == 6:
            thermal = rgb[:, 3:6, :, :]
            rgb = rgb[:, 0:3, :, :]

        p3_t, p4_t, p5_t = self.extract_features(thermal, self.backbone_thermal)
        p3_v, p4_v, p5_v = self.extract_features(rgb, self.backbone_rgb)

        # Reliability-gated asymmetric fusion
        fused_p3 = self.gate_p3(p3_t, p3_v)
        fused_p4 = self.gate_p4(p4_t, p4_v)
        fused_p5 = self.gate_p5(p5_t, p5_v)

        saved = {4: fused_p3, 6: fused_p4, 10: fused_p5}
        x = fused_p5

        for i, layer in enumerate(self.neck_and_head, start=11):
            if i == 11:
                x = layer(x)
            elif i == 12:
                x = torch.cat([x, saved[6]], dim=1)
            elif i == 13:
                x = layer(x)
                saved[13] = x
            elif i == 14:
                x = layer(x)
            elif i == 15:
                x = torch.cat([x, saved[4]], dim=1)
            elif i == 16:
                x = layer(x)
                saved[16] = x
            elif i == 17:
                x = layer(x)
            elif i == 18:
                x = torch.cat([x, saved[13]], dim=1)
            elif i == 19:
                x = layer(x)
                saved[19] = x
            elif i == 20:
                x = layer(x)
            elif i == 21:
                x = torch.cat([x, saved[10]], dim=1)
            elif i == 22:
                x = layer(x)
                saved[22] = x
            elif i == 23:
                x = layer([saved[16], saved[19], saved[22]])

        return x
