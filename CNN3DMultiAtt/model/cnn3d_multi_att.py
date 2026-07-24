"""
CNN3DMultiAtt: multimodal 3D CNN with shared SPD Riemannian channel attention.

Architecture overview
---------------------
Three parallel convolutional branches process T2-weighted, ADC and high-b-value
(HBV) MRI patches. At configurable network depths, feature maps from all
modalities are concatenated and a single symmetric positive definite (SPD)
covariance matrix is estimated. Each branch then derives channel-wise attention
weights from that shared matrix via a DeepSPDNet mapping on the Riemannian
manifold, and reweights its own feature maps accordingly. Branch outputs are
finally concatenated and classified by a fully connected head.
"""

import torch
import torch.nn as nn
from spd_learn.functional import covariance
from spd_learn.modules import CovLayer

from .branch import CNN3DBranch
from .classifier import build_classifier_head


class CNN3DMultiAtt(nn.Module):
    """
    Multimodal 3D CNN for binary csPCa classification on prostate MRI.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 2, csPCa vs. non-csPCa).
    dropout_p : float
        Dropout probability in the classifier head.
    spd_depth : int
        Depth of the DeepSPDNet used inside each attention block.
        A value of 0 flattens the upper triangle of the covariance without
        Riemannian projection.
    reduction : int
        Channel-reduction factor in the attention MLP bottleneck.
    att_position : sequence of int, optional
        Length-6 binary mask indicating where attention is applied:
        ``[pre_pool1, post_pool1, pre_pool2, post_pool2, pre_pool3, unused]``.
        Defaults to ``[1, 0, 1, 0, 1, 0]``.
    """

    model_name = "CNN3DMultiAtt"

    def __init__(
        self,
        num_classes: int = 2,
        dropout_p: float = 0.0,
        spd_depth: int = 1,
        reduction: int = 1,
        att_position=None,
    ):
        super().__init__()

        if att_position is None:
            att_position = [1, 0, 1, 0, 1, 0]
        if len(att_position) != 6:
            raise ValueError(
                "att_position must contain 6 flags: "
                "[pre_pool1, post_pool1, pre_pool2, post_pool2, pre_pool3, unused]"
            )
        self.att_position = [int(v) for v in att_position]

        # Shared covariance estimator reused at every active attention site.
        self.cov_layer = CovLayer(covariance)

        self.branch0 = CNN3DBranch(
            reduction=reduction, spd_depth=spd_depth, att_position=self.att_position
        )
        self.branch1 = CNN3DBranch(
            reduction=reduction, spd_depth=spd_depth, att_position=self.att_position
        )
        self.branch2 = CNN3DBranch(
            reduction=reduction, spd_depth=spd_depth, att_position=self.att_position
        )

        # Output of block 3 per branch: 64 × 2 × 16 × 16; three branches concatenated.
        flattened_size = 64 * 2 * 16 * 16 * 3
        self.fc = build_classifier_head(
            input_dim=flattened_size,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    def _compute_cov(self, f0, f1, f2):
        """
        Estimate the shared SPD covariance from concatenated branch features.

        Parameters
        ----------
        f0, f1, f2 : Tensor
            Feature maps of shape ``(B, C, D, H, W)`` with matching ``C``.

        Returns
        -------
        Tensor
            Covariance of shape ``(B, 3C, 3C)``.
        """
        x_cat = torch.cat([f0, f1, f2], dim=1)
        x_cat = x_cat.view(x_cat.size(0), x_cat.size(1), -1)
        return self.cov_layer(x_cat)

    def forward(self, x):
        """
        Parameters
        ----------
        x : list of Tensor
            ``[T2, ADC, HBV]``, each of shape ``(B, 1, D, H, W)``.

        Returns
        -------
        Tensor
            Logits of shape ``(B, num_classes)``.
        """
        # -------------------- Block 1 --------------------
        f0 = self.branch0.block1_conv(x[0])
        f1 = self.branch1.block1_conv(x[1])
        f2 = self.branch2.block1_conv(x[2])

        if self.branch0.att1 is not None:
            cov = self._compute_cov(f0, f1, f2)
            f0 = self.branch0.att1(f0, cov)
            f1 = self.branch1.att1(f1, cov)
            f2 = self.branch2.att1(f2, cov)

        f0 = self.branch0.pool1(f0)
        f1 = self.branch1.pool1(f1)
        f2 = self.branch2.pool1(f2)

        if self.branch0.att1_post is not None:
            cov = self._compute_cov(f0, f1, f2)
            f0 = self.branch0.att1_post(f0, cov)
            f1 = self.branch1.att1_post(f1, cov)
            f2 = self.branch2.att1_post(f2, cov)

        # -------------------- Block 2 --------------------
        f0 = self.branch0.block2_conv(f0)
        f1 = self.branch1.block2_conv(f1)
        f2 = self.branch2.block2_conv(f2)

        if self.branch0.att2 is not None:
            cov = self._compute_cov(f0, f1, f2)
            f0 = self.branch0.att2(f0, cov)
            f1 = self.branch1.att2(f1, cov)
            f2 = self.branch2.att2(f2, cov)

        f0 = self.branch0.pool2(f0)
        f1 = self.branch1.pool2(f1)
        f2 = self.branch2.pool2(f2)

        if self.branch0.att2_post is not None:
            cov = self._compute_cov(f0, f1, f2)
            f0 = self.branch0.att2_post(f0, cov)
            f1 = self.branch1.att2_post(f1, cov)
            f2 = self.branch2.att2_post(f2, cov)

        # -------------------- Block 3 --------------------
        f0 = self.branch0.block3_conv(f0)
        f1 = self.branch1.block3_conv(f1)
        f2 = self.branch2.block3_conv(f2)

        if self.branch0.att3 is not None:
            cov = self._compute_cov(f0, f1, f2)
            f0 = self.branch0.att3(f0, cov)
            f1 = self.branch1.att3(f1, cov)
            f2 = self.branch2.att3(f2, cov)

        # -------------------- Classifier --------------------
        f0_f = f0.view(f0.size(0), -1)
        f1_f = f1.view(f1.size(0), -1)
        f2_f = f2.view(f2.size(0), -1)
        return self.fc(torch.cat((f0_f, f1_f, f2_f), dim=1))
