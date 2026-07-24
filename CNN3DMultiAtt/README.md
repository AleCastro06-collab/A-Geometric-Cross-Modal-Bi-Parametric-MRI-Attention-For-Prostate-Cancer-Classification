# CNN3DMultiAtt

Multimodal 3D convolutional network for clinically significant prostate cancer
(csPCa) detection on the [PI-CAI](https://pi-cai.grand-challenge.org/) challenge.
Three modality-specific branches (T2-weighted, ADC and high-b-value MRI) share a
single symmetric positive definite (SPD) covariance matrix that drives Riemannian
channel attention.

## Repository layout

```
CNN3DMultiAtt/
├── config.py              # dataset paths and default hyperparameters
├── model/
│   ├── cnn3d_multi_att.py # CNN3DMultiAtt definition
│   ├── branch.py          # per-modality 3D CNN branch
│   ├── attention.py       # SPD-based channel attention
│   ├── deep_spd.py        # DeepSPDNet (BiMap–ReEig–LogEig)
│   └── classifier.py      # fully connected classification head
├── data/
│   └── dataset.py         # PI-CAI fold loaders
├── notebooks/
│   ├── 01_train.ipynb     # 5-fold cross-validation training
│   └── 02_evaluate.ipynb  # metrics and confusion matrices
└── outputs/               # checkpoints and logs (created at runtime)
```

## Method summary

1. Each MRI modality is processed by an independent 3D CNN branch.
2. At selected depths (`att_position`), feature maps from all three branches are
   concatenated along the channel axis.
3. A shared SPD covariance matrix is estimated once (`CovLayer`).
4. Each branch maps that matrix through a DeepSPDNet and an MLP to obtain
   sigmoid channel gates, which reweight its own features.
5. Flattened branch outputs are concatenated and classified by an MLP head.

Default attention placement: `[1, 0, 1, 0, 1, 0]`
(pre-pool1, pre-pool2 and pre-pool3).

## Installation

```bash
cd CNN3DMultiAtt
pip install -r requirements.txt
```

Riemannian layers (`BiMap`, `CovLayer`, `LogEig`, `ReEig`) are provided by
[`spd_learn`](https://pypi.org/project/spd_learn/).

Configure local dataset paths in `config.py`:

```python
PATH_DATA = "/path/to/patches8x64x64/"
PATH_JSON_INFO = "/path/to/studies_info/"
PATH_FOLDS = "/path/to/picai_splits_all.json"
```

Expected patch tensor per case: `(3, 8, 64, 64)` ordered as T2 / ADC / HBV.

## Usage

Training and evaluation are driven from the notebooks:

| Notebook | Description |
|----------|-------------|
| `notebooks/01_train.ipynb` | 5-fold training, checkpointing and optional W&B logging |
| `notebooks/02_evaluate.ipynb` | Fold-wise metrics, confusion matrices and shape check |

Open the notebooks with the working directory set to `notebooks/` (they prepend
the repository root to `sys.path`). Hyperparameters are edited in the
corresponding notebook cells.

Checkpoints are written to:

```
outputs/CNN3DMultiAtt/<run_name>/weights/fold_{0..4}.pth
```

Reported evaluation metrics (mean ± std over folds): accuracy, precision,
recall, F1, AUC-ROC, AUC-PR, FPR@95% TPR and specificity.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spd_depth` | `1` | Depth of DeepSPDNet in attention (`0` flattens the SPD upper triangle) |
| `reduction` | `1` | Bottleneck reduction in the attention MLP |
| `att_position` | `[1, 0, 1, 0, 1, 0]` | Binary mask selecting attention insertion points |
| `dropout_p` | `0.0` | Dropout in the classification head |
| `train_ratio` | `1.0` | Fraction of each fold's training split to use |
