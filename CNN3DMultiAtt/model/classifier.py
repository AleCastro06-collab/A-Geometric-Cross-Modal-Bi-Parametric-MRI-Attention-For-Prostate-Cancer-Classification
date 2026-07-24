"""Fully connected classification head."""

import torch.nn as nn


def build_classifier_head(input_dim: int, num_classes: int = 2, dropout_p: float = 0.0):
    """
    Build an MLP classifier.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the flattened multimodal feature vector.
    num_classes : int
        Number of output logits.
    dropout_p : float
        Dropout probability after the first hidden layer.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0),
        nn.Linear(128, num_classes),
    )
