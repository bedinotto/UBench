"""
UB-02: train↔benchmark checkpoint path contract (CLAUDE.md §7.1).

``codes/naming.py`` is the single authority for every checkpoint filename
(R5: registry names for I/O, display names for humans).  These are pure
function properties — no training, no file I/O.

The negative assertions are UB-02's tombstone: the historical bad
derivations ``best_u_net_…`` and ``best_swin_unetplusplus_…`` (display
names pushed through ``_safe_filename``) must never be produced again.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from codes.naming import checkpoint_path, epoch_checkpoint_glob

REGISTRY_KEYS = ["unet", "transunet", "swin_unet_plus_plus"]
FOLDS = [1, 2]
KINDS = ["best", "epoch"]
_OUT = Path("outputs/2099-01-01_00-00-00")

# The two historical bad derivations (UB-02 root cause).
_FORBIDDEN_SUBSTRINGS = ["u_net", "swin_unetplusplus"]


def _path_for(key: str, fold: int, kind: str) -> Path:
    epoch = 0 if kind == "epoch" else None
    return checkpoint_path(_OUT, key, fold, kind, epoch=epoch)


def test_registry_key_embedded_verbatim():
    """Every produced filename contains the registry key verbatim."""
    for key, fold, kind in itertools.product(REGISTRY_KEYS, FOLDS, KINDS):
        path = _path_for(key, fold, kind)
        assert key in path.name, (
            f"registry key {key!r} not embedded in {path.name!r}"
        )
        assert path.suffix == ".pth"


def test_no_historical_bad_derivations():
    """UB-02 tombstone: display-name derivations never appear in any path."""
    for key, fold, kind in itertools.product(REGISTRY_KEYS, FOLDS, KINDS):
        path = str(_path_for(key, fold, kind))
        for bad in _FORBIDDEN_SUBSTRINGS:
            assert bad not in path, (
                f"historical bad derivation {bad!r} resurfaced in {path!r}"
            )


def test_paths_deterministic():
    """Same (key, fold, kind) → identical path on every call."""
    for key, fold, kind in itertools.product(REGISTRY_KEYS, FOLDS, KINDS):
        assert _path_for(key, fold, kind) == _path_for(key, fold, kind)


def test_paths_unique_across_keys_and_folds():
    """No two (key, fold, kind) combinations collide."""
    combos = list(itertools.product(REGISTRY_KEYS, FOLDS, KINDS))
    paths = {_path_for(key, fold, kind) for key, fold, kind in combos}
    assert len(paths) == len(combos)


def test_display_names_rejected():
    """Display names must never reach file I/O — hard error, not a path."""
    for display in ["U-Net", "TransUNet ", "Swin-UNet++", "u net", ""]:
        with pytest.raises(ValueError):
            checkpoint_path(_OUT, display, 1, "best")


def test_epoch_kind_requires_epoch_and_best_forbids_it():
    with pytest.raises(ValueError):
        checkpoint_path(_OUT, "unet", 1, "epoch")  # epoch missing
    with pytest.raises(ValueError):
        checkpoint_path(_OUT, "unet", 1, "best", epoch=3)  # epoch meaningless
    with pytest.raises(ValueError):
        checkpoint_path(_OUT, "unet", 1, "bestest")  # unknown kind


def test_epoch_glob_matches_only_its_own_series():
    """The resume/prune glob is key- and fold-scoped."""
    glob_pat = epoch_checkpoint_glob(_OUT, "unet", 1)
    own = _path_for("unet", 1, "epoch")
    other_fold = _path_for("unet", 2, "epoch")
    other_key = _path_for("transunet", 1, "epoch")
    import fnmatch

    assert fnmatch.fnmatch(str(own), glob_pat)
    assert not fnmatch.fnmatch(str(other_fold), glob_pat)
    assert not fnmatch.fnmatch(str(other_key), glob_pat)
