"""Experiment configuration: dataset paths and default hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset paths
# Adjust these constants to point to the local PI-CAI patch archive,
# per-case metadata, and official cross-validation splits.
# ---------------------------------------------------------------------------
PATH_DATA = "/data/patches8x64x64_pooch/"
PATH_JSON_INFO = "/data/studies_info8x64x64_pooch/"
PATH_FOLDS = "/data/picai_splits_all.json"

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "lr": 1e-3,
    "epochs": 15,
    "batch_size": 32,
    "num_classes": 2,
    "dropout_p": 0.0,
    "spd_depth": 1,
    "reduction": 1,
    "att_position": [1, 0, 1, 0, 1, 0],
    "train_ratio": 1.0,
    "n_folds": 5,
    "grad_clip": 1.0,
    "seed": 42,
    "suffix": "",
    # Experiment tracking (Weights & Biases). Disable with --no-wandb.
    "use_wandb": True,
    "wandb_entity": "alejandro2230045-universidad-industrial-de-santander",
    "wandb_project": "SPDra",
}
