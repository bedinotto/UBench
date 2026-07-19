"""Processed-data schema manifest (T3.4/M7).

A standalone module (no dependency on ``unified_data`` — it is imported by both
``preprocess_data`` and ``unified_data``, so it must not create an import cycle).

``data/processed/preprocess_manifest.json`` records the schema version the
``.npy`` files were written under. Design (i) stores resized **Celsius**
(``preprocess_version`` 2); legacy data baked per-image-minmax [0,1]
(version 1, no manifest). The load-time guard rejects any data whose manifest
is missing or a version mismatch, because the stored values would otherwise be
silently misinterpreted (e.g. legacy [0,1] normalized as if it were Celsius).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# Bumped to 2 in T3.4 (design (i)): .npy stores resized Celsius, normalization
# applied at load time. Version 1 = legacy baked-normalized [0,1] (no manifest).
PREPROCESS_VERSION = 2
MANIFEST_NAME = "preprocess_manifest.json"

_REMEDY = ("Rebuild the processed data with `./run.sh --force-preprocess` "
           "(or `python codes/main_pipeline.py --force-preprocess`).")


def verify_preprocess_manifest(processed_dir) -> dict:
    """Load and validate the manifest next to ``metadata.csv`` (R4/M7).

    Raises:
        FileNotFoundError: if the manifest is absent (legacy/stale data).
        ValueError: if ``preprocess_version`` != the current code version.
    """
    manifest_path = Path(processed_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing — the processed data predates the "
            f"T3.4 preprocess manifest (or is stale). {_REMEDY}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("preprocess_version")
    if version != PREPROCESS_VERSION:
        raise ValueError(
            f"processed-data schema mismatch: manifest preprocess_version="
            f"{version}, code expects {PREPROCESS_VERSION}. {_REMEDY}"
        )
    return manifest


def write_preprocess_manifest(processed_dir, image_size: Sequence[int],
                              normalization: str,
                              fixed_range_celsius: Sequence[float],
                              num_samples: int) -> Path:
    """Write the schema manifest recording version + provenance.

    ``normalization``/``fixed_range_celsius`` are the config at preprocess time,
    recorded for provenance only — normalization is applied at load, so the
    *version* is what the guard enforces, not the mode.
    """
    manifest = {
        "preprocess_version": PREPROCESS_VERSION,
        "stored_unit": "celsius",
        "image_size": list(image_size),
        "normalization": normalization,
        "fixed_range_celsius": list(fixed_range_celsius),
        "num_samples": num_samples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(processed_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
