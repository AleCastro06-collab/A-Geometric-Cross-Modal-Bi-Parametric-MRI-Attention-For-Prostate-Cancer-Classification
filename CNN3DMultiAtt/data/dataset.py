"""Dataset utilities for PI-CAI multimodal prostate MRI experiments."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

MODALITY_NAMES = ("T2", "ADC", "HBV")


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def experiment_name(
    lr: float,
    epochs: int,
    batch_size: int,
    reduction: int,
    spd_depth: int,
    suffix: str = "",
) -> str:
    """
    Construct a deterministic run identifier from key hyperparameters.

    The resulting string is used as the subdirectory name under ``outputs/``.
    """
    name = f"lr{lr}_epochs{epochs}_bz{batch_size}{suffix}"
    name += f"_red{reduction}_spd{spd_depth}"
    return name


def _label_from_info(info: dict) -> int:
    """Map PI-CAI case metadata to a binary csPCa label."""
    return 0 if info["info_picai"]["case_csPCa"] == "NO" else 1


def _load_case(base_path_data: Path, base_path_info: Path, idx) -> tuple | None:
    folder_path = base_path_data / str(idx)
    npy_files = sorted(folder_path.glob("*.npy"))
    info_path = base_path_info / f"{idx}.json"
    if (not npy_files) or (not info_path.exists()):
        return None
    x = np.load(npy_files[0])
    with open(info_path, "r") as f:
        info = json.load(f)
    return x, _label_from_info(info)


def _to_modality_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    modality=(1, 1, 1),
) -> dict:
    if len(modality) != 3:
        raise ValueError("modality must have three flags: [T2, ADC, HBV]")
    selected = [int(v) for v in modality]
    if sum(selected) == 0:
        raise ValueError("At least one modality must be enabled")

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    data_by_modality = {}

    for modality_idx, is_selected in enumerate(selected):
        if not is_selected:
            continue
        name = MODALITY_NAMES[modality_idx]
        x_tr = torch.tensor(x_train[:, modality_idx, np.newaxis, :, :], dtype=torch.float32)
        x_te = torch.tensor(x_test[:, modality_idx, np.newaxis, :, :], dtype=torch.float32)
        data_by_modality[name] = {
            "train_loader": DataLoader(TensorDataset(x_tr, y_train_t), batch_size=batch_size),
            "test_loader": DataLoader(TensorDataset(x_te, y_test_t), batch_size=batch_size),
        }
    return data_by_modality


def get_fold_loaders(
    fold: int = 0,
    path_data: str = "",
    path_json_info: str = "",
    path_folds: str = "",
    batch_size: int = 32,
    train_ratio: float = 1.0,
    modality=(1, 1, 1),
) -> dict:
    """
    Load train and validation DataLoaders for one PI-CAI cross-validation fold.

    Patches are expected as NumPy arrays of shape ``(3, D, H, W)`` per case,
    with channels ordered as T2, ADC and HBV. Labels are derived from the
    PI-CAI ``case_csPCa`` field.

    Parameters
    ----------
    fold : int
        Fold index (typically 0–4).
    path_data : str
        Root directory containing one subdirectory per case index.
    path_json_info : str
        Directory with per-case JSON metadata files ``{idx}.json``.
    path_folds : str
        JSON file defining train/validation indices per fold.
    batch_size : int
        Mini-batch size for both loaders.
    train_ratio : float
        Fraction of the official training split to retain (ordered prefix).
        Useful for learning-curve or low-data experiments.
    modality : sequence of int
        Binary flags ``[T2, ADC, HBV]`` selecting which modalities to load.

    Returns
    -------
    dict
        Mapping from modality name to
        ``{"train_loader": DataLoader, "test_loader": DataLoader}``.
    """
    with open(path_folds, "r") as f:
        idxs = json.load(f)

    train_indices = idxs[f"Fold_{fold}_train"]
    test_indices = idxs[f"Fold_{fold}_val"]

    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("train_ratio must lie in the interval (0, 1]")
    limit = int(len(train_indices) * train_ratio)
    train_indices = train_indices[:limit]

    base_path_data = Path(path_data)
    base_path_info = Path(path_json_info)

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for idx in train_indices:
        case = _load_case(base_path_data, base_path_info, idx)
        if case is None:
            continue
        x, y = case
        x_train_list.append(x)
        y_train_list.append(y)

    for idx in test_indices:
        case = _load_case(base_path_data, base_path_info, idx)
        if case is None:
            continue
        x, y = case
        x_test_list.append(x)
        y_test_list.append(y)

    return _to_modality_loaders(
        np.array(x_train_list),
        np.array(y_train_list),
        np.array(x_test_list),
        np.array(y_test_list),
        batch_size=batch_size,
        modality=modality,
    )
