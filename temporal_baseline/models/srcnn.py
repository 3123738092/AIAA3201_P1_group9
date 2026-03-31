"""SRCNN: Image Super-Resolution Using Deep Convolutional Networks (Dong et al., TPAMI 2015)."""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Classic 3-layer SRCNN.

    Architecture:
        1. Patch extraction:  Conv(3 -> 64, 9x9) + ReLU
        2. Non-linear mapping: Conv(64 -> 32, 1x1) + ReLU  (originally 5x5, 1x1 is lighter)
        3. Reconstruction:    Conv(32 -> 3, 5x5)

    Input:  Bicubic-upscaled LR image (B, 3, H, W)
    Output: Residual-refined HR image (B, 3, H, W)
    """

    def __init__(self, num_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)
