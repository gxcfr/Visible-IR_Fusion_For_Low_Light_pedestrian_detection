"""
Dual-Stream Intermediate Feature Concatenation Architecture (E006)
Extracts multi-scale features (P3, P4, P5) from parallel RGB and Thermal YOLO26n backbones,
concatenates cross-modal features, projects back to original neck dimensions via 1x1 convs,
and feeds into the standard YOLO26n detection neck and head.
"""

import copy
import torch
import torch.nn as nn
from ultralytics import YOLO


class DualStreamFeatureConcatYOLO26(nn.Module):
    def __init__(self, pretrained_weights: str):
        super().__init__()
        base_yolo = YOLO(pretrained_weights)
        m = base_yolo.model.model

        # Thermal backbone (layers 0 through 10)
        self.backbone_thermal = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11)])
        # RGB backbone (layers 0 through 10)
        self.backbone_rgb = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11)])

        # Fusion projections at P3 (128 ch), P4 (128 ch), P5 (256 ch)
        self.fuse_p3 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.fuse_p4 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.fuse_p5 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        # Neck and Head layers (layers 11 through 23)
        self.neck_and_head = nn.ModuleList([copy.deepcopy(m[i]) for i in range(11, len(m))])

        # Ultralytics loss & metadata delegation
        from ultralytics.cfg import get_cfg
        base_yolo.model.args = get_cfg()
        self.args = base_yolo.model.args
        self.init_criterion = base_yolo.model.init_criterion
        self.criterion = self.init_criterion()

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def extract_features(self, x: torch.Tensor, backbone: nn.ModuleList):
        """Extract multi-scale features P3, P4, P5 from backbone layers 0..10."""
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

        fused_p3 = self.fuse_p3(torch.cat([p3_t, p3_v], dim=1))
        fused_p4 = self.fuse_p4(torch.cat([p4_t, p4_v], dim=1))
        fused_p5 = self.fuse_p5(torch.cat([p5_t, p5_v], dim=1))

        # Pass through neck and head
        # In YOLO26: Layer 11: Upsample, Layer 12: Concat [P5_up, P4], Layer 13: C3k2, etc.
        saved = {4: fused_p3, 6: fused_p4, 10: fused_p5}
        x = fused_p5

        for i, layer in enumerate(self.neck_and_head, start=11):
            if i == 11:  # Upsample
                x = layer(x)
            elif i == 12:  # Concat [-1, 6]
                x = torch.cat([x, saved[6]], dim=1)
            elif i == 13:  # C3k2
                x = layer(x)
                saved[13] = x
            elif i == 14:  # Upsample
                x = layer(x)
            elif i == 15:  # Concat [-1, 4]
                x = torch.cat([x, saved[4]], dim=1)
            elif i == 16:  # C3k2 (P3 head input)
                x = layer(x)
                saved[16] = x
            elif i == 17:  # Conv stride 2
                x = layer(x)
            elif i == 18:  # Concat [-1, 13]
                x = torch.cat([x, saved[13]], dim=1)
            elif i == 19:  # C3k2 (P4 head input)
                x = layer(x)
                saved[19] = x
            elif i == 20:  # Conv stride 2
                x = layer(x)
            elif i == 21:  # Concat [-1, 10]
                x = torch.cat([x, saved[10]], dim=1)
            elif i == 22:  # C3k2 (P5 head input)
                x = layer(x)
                saved[22] = x
            elif i == 23:  # Detect Head [16, 19, 22]
                x = layer([saved[16], saved[19], saved[22]])

        return x
