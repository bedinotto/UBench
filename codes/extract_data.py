"""
Data Extraction Utility
=======================
Automatically extracts all ZIP files from requirements/ into data/.
Handles nested ZIPs (e.g., Charlotte-ThermalFace.zip inside the outer ZIP).
Idempotent: skips extraction if data already exists.
"""

import os
import sys
import glob
import shutil
import zipfile
import tempfile
from pathlib import Path

try:
    from codes.logger import start_from_env as _start_log
except ImportError:
    _start_log = None  # graceful fallback if logger isn't available yet


# Resolve the project root (parent of codes/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"
DATA_DIR = PROJECT_ROOT / "data"


def _is_data_already_extracted() -> bool:
    """
    Check if data/ already contains at least one Sx directory with TIFF files.
    Returns True if extraction can be skipped.
    """
    if not DATA_DIR.exists():
        return False

    for entry in DATA_DIR.iterdir():
        if (entry.is_dir()
                and entry.name.startswith("S")
                and entry.name[1:].isdigit()):
            tiff_files = list(entry.glob("*.tiff"))
            if tiff_files:
                return True

    return False


def _is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer instead of the actual data."""
    if not file_path.is_file() or file_path.stat().st_size > 1024:
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(100)
            return "version https://git-lfs" in content
    except Exception:
        return False


def _discover_zip_files() -> list:
    """Discover all .zip files in requirements/."""
    if not REQUIREMENTS_DIR.exists():
        print(f"❌ ERROR: Requirements directory not found: {REQUIREMENTS_DIR}")
        return []

    zip_files = sorted(REQUIREMENTS_DIR.glob("*.zip"))
    if not zip_files:
        print(f"⚠️  No ZIP files found in {REQUIREMENTS_DIR}")
    else:
        print(f"✅ Found {len(zip_files)} ZIP file(s) in requirements/:")
        for zf in zip_files:
            if _is_lfs_pointer(zf):
                print(f"   - {zf.name} (Git LFS pointer file — missing actual content)")
            else:
                size_mb = zf.stat().st_size / (1024 * 1024)
                print(f"   - {zf.name} ({size_mb:.1f} MB)")

    return zip_files


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """
    Extract a single ZIP file to the destination directory.
    Returns True on success, False on failure.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Validate the archive
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"⚠️  Warning: Corrupt entry in {zip_path.name}: {bad_file}")

            zf.extractall(dest_dir)
        return True
    except zipfile.BadZipFile:
        print(f"❌ ERROR: {zip_path.name} is not a valid ZIP file (corrupted?)")
        return False
    except Exception as e:
        print(f"❌ ERROR extracting {zip_path.name}: {e}")
        return False


def _process_extracted_contents(extract_dir: Path):
    """
    Process the extracted contents: find CSVs and TIFF directories,
    move them to data/. Also handle nested ZIP files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Walk through extracted contents looking for nested ZIPs, CSVs, and Sx dirs
    for root, dirs, files in os.walk(extract_dir):
        root_path = Path(root)

        for f in files:
            file_path = root_path / f

            # Handle nested ZIP files (e.g., Charlotte-ThermalFace.zip)
            if f.endswith(".zip"):
                print(f"  → Found nested ZIP: {f}")
                nested_dir = extract_dir / f"_nested_{f.replace('.zip', '')}"
                nested_dir.mkdir(exist_ok=True)

                if _extract_zip(file_path, nested_dir):
                    # Recursively process nested contents
                    _process_extracted_contents(nested_dir)

            # Copy CSV files that match Sx.csv pattern
            elif f.endswith(".csv") and len(f) >= 4:
                name_no_ext = f[:-4]
                if (name_no_ext.startswith("S")
                        and name_no_ext[1:].isdigit()):
                    dest = DATA_DIR / f
                    if not dest.exists():
                        shutil.copy2(file_path, dest)
                        print(f"  → Copied {f} to data/")

        # Copy Sx directories containing TIFF files
        for d in dirs:
            if d.startswith("S") and d[1:].isdigit():
                src_dir = root_path / d
                dest_dir = DATA_DIR / d

                # Check if source directory contains TIFF files
                tiff_files = list(src_dir.glob("*.tiff"))
                if tiff_files:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    copied = 0
                    for tiff in tiff_files:
                        dest_file = dest_dir / tiff.name
                        if not dest_file.exists():
                            shutil.copy2(tiff, dest_file)
                            copied += 1
                    if copied > 0:
                        print(f"  → Copied {copied} TIFF files to data/{d}/")
                    else:
                        print(f"  → data/{d}/ already has all TIFF files")


def _generate_annotations():
    """
    Generate bounding boxes and polygonal masks from the CSV files
    using generate_boxes_polygons.py functions.
    """
    try:
        from codes.generate_boxes_polygons import (
            generate_bounding_boxes,
            generate_polygonal_masks,
        )
    except ImportError:
        print("⚠️  Could not import generate_boxes_polygons — "
              "skipping annotation generation")
        return

    csv_files = sorted(DATA_DIR.glob("S*.csv"))
    valid_files = [
        f for f in csv_files
        if f.stem.startswith("S") and f.stem[1:].isdigit()
    ]
    valid_files.sort(key=lambda x: int(x.stem[1:]))

    if not valid_files:
        print("⚠️  No Sx.csv files found in data/ — skipping annotation generation")
        return

    for csv_path in valid_files:
        dataset_name = csv_path.stem
        bbox_path = DATA_DIR / f"{dataset_name}_bounding_boxes.csv"
        polygon_path = DATA_DIR / f"{dataset_name}_polygonal_masks.json"

        if bbox_path.exists() and polygon_path.exists():
            print(f"  ✅ Annotations already exist for {dataset_name}")
            continue

        print(f"  → Generating annotations for {dataset_name}...")
        if not bbox_path.exists():
            generate_bounding_boxes(str(csv_path), str(bbox_path))
        if not polygon_path.exists():
            generate_polygonal_masks(str(csv_path), str(polygon_path))


def extract_all_data(force: bool = False) -> bool:
    """
    Main function: extract all ZIP data from requirements/ into data/.

    Args:
        force: If True, extract even if data already exists.

    Returns:
        True if data is ready (extracted or already present), False on errors.
    """
    print("\n" + "=" * 80)
    print("DATA EXTRACTION")
    print("=" * 80)

    # Idempotency check — skip ZIP extraction but always run annotation generation
    if not force and _is_data_already_extracted():
        print("✅ Data already extracted — skipping ZIP extraction (use --force to re-extract)")
    else:
        # Discover ZIP files
        zip_files = _discover_zip_files()
        if not zip_files:
            # No ZIPs but data might already exist in another form
            if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
                print("⚠️  No ZIP files found, but data/ is not empty — continuing")
            else:
                print("❌ No ZIP files and no existing data — cannot proceed")
                return False
        else:
            # Extract each ZIP to a temporary directory, then process
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            for zip_path in zip_files:
                if _is_lfs_pointer(zip_path):
                    print(f"\n❌ CRITICAL ERROR: {zip_path.name} is a Git LFS pointer!")
                    print("   The actual file has not been downloaded because Git LFS is not initialized.")
                    print("   Please run the following commands in your terminal to fix this:")
                    print("     git lfs install")
                    print("     git lfs pull")
                    print("")
                    return False

                print(f"\nExtracting {zip_path.name}...")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)

                    if not _extract_zip(zip_path, tmp_path):
                        print(f"❌ Failed to extract {zip_path.name}")
                        continue

                    _process_extracted_contents(tmp_path)

    # Always generate annotations — idempotent (skips datasets that already have them)
    print("\nGenerating annotations...")
    _generate_annotations()

    # Final validation
    if _is_data_already_extracted():
        print("\n✅ Data extraction completed successfully!")
        return True
    else:
        print("\n❌ Data extraction completed but no valid data found")
        return False


def check_data_status():
    """Print a summary of the current data status without extracting."""
    print("\n" + "=" * 80)
    print("DATA STATUS CHECK")
    print("=" * 80)

    # Check requirements/
    zip_files = _discover_zip_files()

    # Check data/
    if not DATA_DIR.exists():
        print(f"\n❌ data/ directory does not exist")
        return

    dataset_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    ], key=lambda x: int(x.name[1:]))

    if not dataset_dirs:
        print(f"\n❌ No dataset directories found in data/")
        return

    print(f"\n✅ Found {len(dataset_dirs)} dataset directories:")
    total_tiffs = 0
    for d in dataset_dirs:
        tiff_count = len(list(d.glob("*.tiff")))
        total_tiffs += tiff_count
        csv_exists = (DATA_DIR / f"{d.name}.csv").exists()
        bbox_exists = (DATA_DIR / f"{d.name}_bounding_boxes.csv").exists()
        poly_exists = (DATA_DIR / f"{d.name}_polygonal_masks.json").exists()
        status = "✅" if (tiff_count > 0 and csv_exists and bbox_exists and poly_exists) else "⚠️ "
        print(f"  {status} {d.name}: {tiff_count} TIFFs | "
              f"CSV: {'✅' if csv_exists else '❌'} | "
              f"BBox: {'✅' if bbox_exists else '❌'} | "
              f"Poly: {'✅' if poly_exists else '❌'}")

    print(f"\nTotal TIFF files: {total_tiffs}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract training data from ZIP files in requirements/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if data already exists",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check data status, do not extract",
    )

    args = parser.parse_args()

    # Mirror output to extract.log in the pipeline's log directory (if available)
    _logger = _start_log("extract.log") if _start_log else None

    if args.check:
        check_data_status()
    else:
        success = extract_all_data(force=args.force)
        if _logger:
            _logger.stop()
        sys.exit(0 if success else 1)

    if _logger:
        _logger.stop()
