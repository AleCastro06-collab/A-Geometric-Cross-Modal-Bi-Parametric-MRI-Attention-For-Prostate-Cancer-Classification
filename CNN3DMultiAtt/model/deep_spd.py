"""Deep SPDNet: stacked BiMap–ReEig blocks on the SPD manifold."""

from warnings import warn

import torch
import torch.nn as nn
from spd_learn.functional import covariance
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig


class DeepSPDNet(nn.Module):
    """
    Riemannian network operating on symmetric positive definite (SPD) matrices.

    Optionally estimates a covariance from raw features, then applies a stack of
    bilinear mappings (BiMap) and eigenvalue rectifications (ReEig). A
    logarithmic eigenvalue map (LogEig) projects the result to the tangent space,
    followed by a linear layer that produces the output vector.

    Parameters
    ----------
    n_chans : int
        Input SPD matrix size (number of channels / covariance dimension).
    n_outputs : int
        Dimensionality of the Euclidean output vector.
    depth : int
        Number of BiMap–ReEig blocks.
    reduction_factor : float or int
        Factor by which the SPD subspace dimension is reduced at each BiMap.
    input_type : {"raw", "cov"}
        ``"raw"`` estimates covariance from feature sequences; ``"cov"`` expects
        precomputed SPD matrices.
    threshold : float
        Eigenvalue floor used by ReEig for numerical stability.
    upper : bool
        If True, LogEig vectorizes the upper triangle only.
    """

    def __init__(
        self,
        n_chans,
        n_outputs,
        depth=2,
        reduction_factor=2,
        input_type="raw",
        cov_method=covariance,
        threshold=1e-4,
        upper=True,
    ):
        super().__init__()

        if input_type == "raw":
            self.cov = CovLayer(method=cov_method)
        elif input_type == "cov":
            self.cov = nn.Identity()
        else:
            raise ValueError("input_type must be 'raw' or 'cov'")

        self.bimaps = nn.ModuleList()
        self.reeigs = nn.ModuleList()

        current_dim = n_chans
        for i in range(depth):
            next_dim = int(current_dim // reduction_factor)
            if next_dim < 2:
                warn(
                    f"Layer {i + 1}: computed subspace dimension {next_dim} is "
                    f"too small; clamping to 2.",
                    UserWarning,
                )
                next_dim = 2
            self.bimaps.append(BiMap(current_dim, next_dim))
            self.reeigs.append(ReEig(threshold))
            current_dim = next_dim

        self.logeig = LogEig(upper=upper)
        self.len_last_layer = (
            current_dim * (current_dim + 1) // 2 if upper else current_dim**2
        )
        self.classifier = nn.Linear(self.len_last_layer, n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.cov(X)
        for bimap, reeig in zip(self.bimaps, self.reeigs):
            X = bimap(X)
            X = reeig(X)
        X = self.logeig(X)
        return self.classifier(X)
