"""Single-modality 3D convolutional branch with optional Riemannian attention."""

import torch.nn as nn

from .attention import MultiModalRiemannianAttention


class CNN3DBranch(nn.Module):
    """
    3D CNN branch for one MRI modality (T2, ADC or HBV).

    Convolutional blocks extract hierarchical features. Optional attention
    modules sit at configurable positions and receive a shared SPD covariance
    computed externally by :class:`~model.cnn3d_multi_att.CNN3DMultiAtt`.

    Parameters
    ----------
    reduction : int
        Bottleneck reduction factor passed to each attention module.
    spd_depth : int
        DeepSPDNet depth used inside each attention module.
    att_position : sequence of int, optional
        Length-6 binary mask:
          - ``[0]`` pre-pool1  (after block1_conv, before pool1)
          - ``[1]`` post-pool1 (after pool1, before block2_conv)
          - ``[2]`` pre-pool2  (after block2_conv, before pool2)
          - ``[3]`` post-pool2 (after pool2, before block3_conv)
          - ``[4]`` pre-pool3  (after block3_conv, before classification)
          - ``[5]`` reserved / unused
    """

    def __init__(self, reduction: int = 2, spd_depth: int = 1, att_position=None):
        super().__init__()

        if att_position is None:
            att_position = [1, 0, 1, 0, 1, 0]
        if len(att_position) != 6:
            raise ValueError("att_position must contain 6 elements")
        self.att_position = [int(v) for v in att_position]

        # Block 1: (B, 1, 8, 64, 64) -> (B, 16, 8, 64, 64)
        self.block1_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        # Shared SPD size at this depth: 3 modalities × 16 channels = 48
        self.att1 = (
            MultiModalRiemannianAttention(16, 48, reduction=reduction, spd_depth=spd_depth)
            if self.att_position[0]
            else None
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att1_post = (
            MultiModalRiemannianAttention(16, 48, reduction=reduction, spd_depth=spd_depth)
            if self.att_position[1]
            else None
        )

        # Block 2: (B, 16, 4, 32, 32) -> (B, 32, 4, 32, 32)
        self.block2_conv = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # Shared SPD size: 3 × 32 = 96
        self.att2 = (
            MultiModalRiemannianAttention(32, 96, reduction=reduction, spd_depth=spd_depth)
            if self.att_position[2]
            else None
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att2_post = (
            MultiModalRiemannianAttention(32, 96, reduction=reduction, spd_depth=spd_depth)
            if self.att_position[3]
            else None
        )

        # Block 3: (B, 32, 2, 16, 16) -> (B, 64, 2, 16, 16)
        self.block3_conv = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        # Shared SPD size: 3 × 64 = 192
        self.att3 = (
            MultiModalRiemannianAttention(64, 192, reduction=reduction, spd_depth=spd_depth)
            if self.att_position[4]
            else None
        )
