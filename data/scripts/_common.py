"""Shared helpers for the dissertation analysis scripts (data/scripts/a01..a13).

These scripts live outside the `codes` package (by design — they are one-off
analysis, not pipeline code), but they must never re-derive contracts the
pipeline already owns: registry keys, checkpoint filenames, the leave-
subjects-out split. Re-deriving those by hand is exactly the UB-02 bug class
(checkpoint filenames drifted between trainer and benchmark because two
places encoded the same contract differently). This module is the one place
that bootstraps `codes.*` access and exposes the split/model-loading logic,
so a01..a13 all agree with each other and with the run that produced the
checkpoints.

`DATA_DIR` is the parent of this file's directory: the scripts live in
`data/scripts/`, while the artifacts they read and write live in `data/`,
`data/outputs/` and `data/img/`.

Run these scripts with the project's own environment, e.g.::

    cd /home/doga/Documents/UBench/data/scripts
    ../../.venv/bin/python a03_per_subject_wilcoxon_best.py

Not the conda `base` environment — its opencv-python build is linked against
a system libtiff that is missing symbols (`libjpeg12_write_raw_data`), which
is exactly the crash recorded for A3 in outputs.txt. The project's `.venv`
(uv-managed) has the pinned, working stack (torch 2.5.1+cu121, opencv 5.0,
pandas, torchmetrics, ...).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Analysis artifacts go here — NEVER under the repo's own outputs/ or logs/,
# which are the pipeline's run directories (writing there would pollute a
# real run with unrelated ad-hoc analysis files).
OUT_DIR = DATA_DIR / "outputs"
IMG_DIR = DATA_DIR / "img"
OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Single source of truth for (display name, registry key). The real run this
# analysis targets trained exactly this from-scratch trio (see
# logs/<run>/run_metadata.json: models_to_train = ["unet", "transunet",
# "swin"]) — 'swin' there is main_pipeline's CLI alias, NOT a valid
# model_registry key, so it must be resolved to 'swin_unet_plus_plus' before
# it ever reaches create_model() or checkpoint_path() (this is exactly the
# bug in the original a02/a03 drafts).
MODELS = [
    ("U-Net", "unet"),
    ("TransUNet", "transunet"),
    ("Swin-UNet++", "swin_unet_plus_plus"),
]

# Families that take img_size at construction (mirrors main_pipeline.py's
# train_model kwargs branch — the from-scratch trio's transformer members
# plus the pretrained pair, for completeness).
TRANSFORMER_FAMILY_KEYS = {
    "transunet", "swin_unet_plus_plus", "swin_pretrained", "transunet_pretrained",
}

_LANDMARK_FILE_RE = re.compile(r"^S\d+\.csv$")
_LANDMARK_COL_RE = re.compile(r"^[xy]\d+$")


def landmark_files() -> list[Path]:
    """The 10 per-subject raw landmark/environment CSVs (data/S{n}.csv).

    Excludes `S{n}_bounding_boxes.csv` — a naive `glob('S*.csv')` (the
    original A1 bug) also matches those, which have a completely different
    schema (min_x/min_y/max_x/max_y, no per-landmark x{i}/y{i} columns) and
    are not landmark files at all.
    """
    return sorted(p for p in DATA_DIR.glob("S*.csv") if _LANDMARK_FILE_RE.match(p.name))


def landmark_xy_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _LANDMARK_COL_RE.match(c)]


def classify_landmarks(path: Path) -> pd.DataFrame:
    """Per-image landmark bucket (43=perfil / 73=frontal) for one subject.

    Each `S{n}.csv` has a FIXED schema width of 73 landmark pairs (146 x/y
    columns) for every row — frontal AND profile images alike. What varies
    per row is how many of those 146 cells are actually filled: profile
    (perfil) shots only populate ~43 landmark pairs and leave the rest NaN.
    Counting schema columns (the original bug) is therefore constant (=73)
    for every landmark file and tells you nothing; the signal is the
    per-row non-null count.

    A handful of rows are off by one point (e.g. 72 or 44 filled — a single
    occluded landmark) rather than exactly 73/43; those are bucketed to
    their nearest documented category rather than left as a spurious third
    bucket, via the midpoint threshold ``(43 + n_schema) / 2``.
    """
    df = pd.read_csv(path)
    xy_cols = landmark_xy_columns(df)
    n_schema = len(xy_cols) // 2  # will be 73 for every file in this dataset
    filled_pairs = df[xy_cols].notna().sum(axis=1) // 2
    bucket = np.where(filled_pairs >= (43 + n_schema) / 2, n_schema, 43)
    subject = path.stem
    return pd.DataFrame({
        "sample_id": subject + "/" + df["ID"].astype(str),
        "subject": subject,
        "img_id": df["ID"],
        "n_landmarks_filled": filled_pairs,
        "landmark_bucket": bucket,
    })


def load_raw_subject_csv(subject: str) -> pd.DataFrame:
    """Read one subject's raw CSV (landmarks + Distance/env-temp/...).

    Adds `sample_id` in the same `S{n}/{ID}` form used by
    data/processed/metadata.csv so this can be merged with processed-data /
    per-image result tables on a common key.
    """
    path = DATA_DIR / f"{subject}.csv"
    df = pd.read_csv(path)
    df = df.copy()
    # Raw per-subject exports carry Excel artifacts confirmed across this
    # dataset: trailing all-blank rows (S6.csv has one extra row with ID=NaN
    # from stray exported columns) and '#DIV/0!' error strings inside
    # otherwise-numeric columns (S2.csv, S10.csv both have this in
    # `env-temp`). Left as-is, concatenating subjects turns the column
    # dtype to `object` (str+float mixed) and breaks idxmin/idxmax/qcut
    # downstream with a confusing TypeError far from the actual bad cell.
    df = df.dropna(subset=["ID"])
    for col in ("Distance", "env-temp", "RH", "Airflow", "Sensation"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sample_id"] = subject + "/" + df["ID"].astype(str)
    return df


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    """Build the pipeline's Config the same way main_pipeline.py does.

    Config()'s paths (data/, data/processed/, outputs/, logs/) are relative
    to the CWD, so we chdir to the repo root first — mirrors run.sh always
    invoking python from the repo root. Analysis outputs still go to the
    absolute OUT_DIR/IMG_DIR above, never to a relative 'outputs/'/'img/'
    path, so this chdir can never make an a*.py script write into the real
    pipeline's outputs/logs run directories.
    """
    os.chdir(REPO_ROOT)
    from codes.unified_data import Config
    return Config()


def _ensure_models_registered() -> None:
    import codes.unet_v2          # noqa: F401  registers 'unet'
    import codes.transunet        # noqa: F401  registers 'transunet'
    import codes.swin_unet_plus_plus  # noqa: F401  registers 'swin_unet_plus_plus'


def build_model(registry_key: str, config):
    """Instantiate a registered architecture with the pipeline's own kwargs.

    Mirrors main_pipeline.py:train_model's kwargs branch exactly (img_size
    only for the transformer family) — the ValueError from a bare
    create_model(name) with an unregistered/misspelled key (the original
    a02 bug: model modules were never imported, so the registry was empty)
    is reproduced here as a real error too, not swallowed.
    """
    from codes.model_registry import create_model
    _ensure_models_registered()
    kwargs = {"in_channels": 1, "num_classes": config.NUM_CLASSES}
    if registry_key in TRANSFORMER_FAMILY_KEYS:
        kwargs["img_size"] = config.IMAGE_SIZE[0]
    return create_model(registry_key, **kwargs)


def resolve_run_dir(config) -> Path:
    """Resolve which outputs/<run_id> directory to analyze.

    Defaults to outputs/latest (the pipeline's own symlink convention, kept
    up to date by every ./run.sh invocation); override with UBENCH_RUN_ID to
    pin a specific historical run once 'latest' has moved on.
    """
    run_id = os.environ.get("UBENCH_RUN_ID")
    run_dir = (config.OUTPUT_DIR / run_id) if run_id else (config.OUTPUT_DIR / "latest")
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run dir not found: {run_dir}. Set UBENCH_RUN_ID=<timestamp> to "
            f"pick a specific outputs/<run_id>."
        )
    # outputs/latest is a symlink; resolve it so callers see the real
    # timestamped run_id (e.g. via .name) instead of the literal 'latest'.
    return run_dir.resolve()


def load_fold_model(registry_key: str, fold: int, run_dir: Path, config, device):
    """Build the architecture and load its best-checkpoint weights for one fold.

    Uses codes.naming.checkpoint_path — the single filename authority (UB-02)
    — instead of hand-formatting f'best_{key}_fold_{fold}_model.pth', so this
    can never drift from what the trainer actually wrote.
    """
    import torch
    from codes.naming import checkpoint_path

    model = build_model(registry_key, config).to(device).eval()
    ckpt = checkpoint_path(run_dir, registry_key, fold, kind="best")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def load_fold_model_epoch(registry_key: str, fold: int, epoch: int, run_dir: Path,
                          config, device):
    """Load the weights saved at the END of a specific epoch (0-indexed).

    ``best_*.pth`` under ``models/`` is a bare state_dict chosen by val mIoU —
    the *secondary* reporting variant. The epoch checkpoints under
    ``checkpoints/`` are full resume checkpoints (optimizer, scaler, scheduler
    and metric history alongside the weights), so the weights must be pulled
    out of ``model_state_dict``.

    Verified on this run: ``epoch_0099.pth`` carries ``epoch == 99`` (0-indexed,
    i.e. the 100th and last epoch of the budget) and its recorded
    ``val_ious[-1]`` equals the ``final_val_iou`` in the fold's metrics JSON —
    so it is exactly the state that produced the **primary** (final-epoch)
    variant reported in the dissertation.
    """
    import torch
    from codes.naming import checkpoint_path

    model = build_model(registry_key, config).to(device).eval()
    ckpt_path = checkpoint_path(run_dir, registry_key, fold, kind="epoch", epoch=epoch)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Epoch checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if ckpt.get("epoch") != epoch:
        raise ValueError(
            f"{ckpt_path} records epoch={ckpt.get('epoch')}, expected {epoch}"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def fold_subject_splits(config):
    """Reproduce the EXACT leave-subjects-out split the training run used.

    Authoritative by construction: calls the same
    GroupKFold(groups=df['dataset']) the trainer calls
    (codes/unified_data.py:create_kfold_data_loaders), on the same
    data/processed/metadata.csv (untouched since the run — same mtime).
    GroupKFold has no shuffle/random_state; it is a deterministic function of
    (row order, groups), so this reproduces the training split exactly
    without needing a hand-copied subject/fold table (which is exactly the
    kind of second authority that drifts — see UB-02 in CLAUDE.md).

    Returns:
        List of dicts, one per fold (1-based ``fold``), each with
        ``train_df``, ``val_df``, ``train_subjects``, ``val_subjects``.
    """
    from sklearn.model_selection import GroupKFold
    from codes.unified_data import load_split_metadata, resolve_fold_count

    df = load_split_metadata(config)
    effective_k = resolve_fold_count(config.K_FOLDS, df["dataset"])
    gkf = GroupKFold(n_splits=effective_k)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["dataset"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        folds.append({
            "fold": fold_idx + 1,
            "train_df": train_df,
            "val_df": val_df,
            "train_subjects": sorted(train_df["dataset"].unique()),
            "val_subjects": sorted(val_df["dataset"].unique()),
        })
    return folds
