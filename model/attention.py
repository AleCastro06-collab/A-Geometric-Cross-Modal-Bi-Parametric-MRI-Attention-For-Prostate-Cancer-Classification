"""Riemannian channel attention driven by a shared multimodal SPD matrix."""

import torch
import torch.nn as nn

from .deep_spd import DeepSPDNet


class MultiModalRiemannianAttention(nn.Module):
    """
    Reweight a single-modality feature map using a shared SPD covariance.

    The covariance matrix is estimated once from the concatenation of all
    modality features (outside this module) and supplied as input. This module
    maps the SPD matrix to a vector on the tangent space (via DeepSPDNet, or
    by flattening the upper triangle when ``spd_depth=0``), then produces
    sigmoid channel gates for the target modality.

    Parameters
    ----------
    in_features : int
        Number of channels in the feature map to be reweighted.
    total_features : int
        Dimensionality of the shared SPD matrix (sum of channels across
        modalities at the current depth).
    reduction : int
        Reduction factor of the attention MLP bottleneck.
    spd_depth : int
        Number of BiMap–ReEig blocks in DeepSPDNet. If 0, the upper triangle
        of the covariance is used directly without Riemannian projection.
    use_minmax_norm : bool
        If True, rescale sigmoid gates to the interval ``[0.5, 1.5]`` to
        avoid fully silencing channels.
    """

    def __init__(
        self,
        in_features: int,
        total_features: int,
        reduction: int = 2,
        spd_depth: int = 1,
        use_minmax_norm: bool = False,
    ):
        super().__init__()
        self.spd_depth = spd_depth
        self.total_features = total_features
        self.use_minmax_norm = use_minmax_norm

        if spd_depth == 0:
            vec_dim = total_features * (total_features + 1) // 2
        else:
            spd_out_dim = max(total_features // 2, 1)
            vec_dim = spd_out_dim * (spd_out_dim + 1) // 2
            self.spdnet_module = DeepSPDNet(
                n_chans=total_features,
                n_outputs=vec_dim,
                input_type="cov",
                depth=spd_depth,
                threshold=1e-6,
                upper=True,
            ).double()

        bottleneck = max(in_features // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_features, bias=False),
            nn.Sigmoid(),
        )

    def _minmax_normalize(self, w: torch.Tensor) -> torch.Tensor:
        """Affine map of channel gates onto ``[0.5, 1.5]``."""
        w_min = w.amin(dim=0, keepdim=True)
        w_max = w.amax(dim=0, keepdim=True)
        w_range = torch.clamp(w_max - w_min, min=1e-8)
        return (w - w_min) / w_range + 0.5

    def forward(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Feature map of shape ``(B, C, D, H, W)``.
        cov : Tensor
            Shared SPD covariance of shape ``(B, C_total, C_total)``.

        Returns
        -------
        Tensor
            Reweighted feature map with the same shape as ``x``.
        """
        if self.spd_depth == 0:
            triu_indices = torch.triu_indices(
                self.total_features, self.total_features, device=cov.device
            )
            vec = cov[:, triu_indices[0], triu_indices[1]]
        else:
            vec = self.spdnet_module(cov.double()).float()

        w = self.fc(vec)
        if self.use_minmax_norm:
            w = self._minmax_normalize(w)
        w = w.view(x.size(0), x.size(1), 1, 1, 1)
        return x * w
