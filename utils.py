import csv
import os
import random
import shutil
import re
from pathlib import Path
from typing import Tuple, Dict, List


def _sanitize_label(name: str) -> str:
    """Sanitize folder name (remove problematic filesystem chars)."""
    if name is None:
        return ""
    name = str(name)
    # remove control chars and: <>:"/\|?*
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", name).strip() or "unknown"


def structure_c_mnist_with_splits(
        data_root: str = "data",
        images_subdir: str = "c_mnist",
        csv_filename: str = "chinese_mnist.csv",
        out_root: str = "data",
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        class_by: str = "value_char",  # "value", "character", or "value_char"
        seed: int = 42,
        mode: str = "copy",  # "copy" or "symlink"
        manifest_name: str = "manifest.csv",
        verbose: bool = True
) -> Dict[str, object]:
    """
    Create train/val/test folders with per-class splits from c_mnist.

    Returns a dict with summary counts and manifest path:
      { "counts": {"train": N, "val": M, "test": K},
        "manifest": Path("data/manifest.csv"),
        "per_class_counts": { "label": {"train":a,"val":b,"test":c}, ... } }

    Notes:
      - The function will NOT delete or move files from data/c_mnist; it copies (or symlinks) them.
      - The CSV is expected to contain headers including at least:
            suite_id, sample_id, code
        and optionally value and character.
    """

    assert mode in ("copy", "symlink"), "mode must be 'copy' or 'symlink'"
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    data_root_p = Path(data_root)
    images_dir = data_root_p / images_subdir
    csv_path = data_root_p / csv_filename
    out_root_p = Path(out_root)

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 1) Read CSV and map labels -> list of source image paths
    label_to_files: Dict[str, List[Path]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        # Validate
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or without header.")
        for row in reader:
            # robust extraction
            suite = row.get("suite_id") or row.get("suite") or row.get("suiteid")
            sample = row.get("sample_id") or row.get("sample") or row.get("sampleid")
            code = row.get("code") or row.get("code_id") or row.get("codeid")
            value = row.get("value") or row.get("label")  # numeric label as string
            character = row.get("character") or row.get("char") or row.get("glyph")

            if suite is None or sample is None or code is None:
                # try to be permissive: perhaps columns are positional -> raise clear error
                raise ValueError("CSV must contain 'suite_id','sample_id','code' columns")

            suite = str(suite).strip()
            sample = str(sample).strip()
            code = str(code).strip()
            value = (str(value).strip() if value is not None else "")
            character = (str(character).strip() if character is not None else "")

            filename = f"input_{suite}_{sample}_{code}.jpg"
            src_path = images_dir / filename

            if class_by == "value":
                label = _sanitize_label(value or code)
            elif class_by == "character":
                label = _sanitize_label(character or value or code)
            else:  # value_char
                label = _sanitize_label(f"{value}_{character}")

            label_to_files.setdefault(label, []).append(src_path)

    # 2) For each label keep only existing files, warn about missing
    random.seed(seed)
    splits_names = ("train", "val", "test")
    per_split_lists: Dict[str, Dict[str, List[Path]]] = {k: {} for k in splits_names}

    for label, paths in label_to_files.items():
        existing = [p for p in paths if p.exists()]
        missing_count = len(paths) - len(existing)
        if missing_count > 0 and verbose:
            print(f"Warning: {missing_count} files missing for class '{label}'; {len(existing)} will be used.")
        if not existing:
            if verbose:
                print(f"Skipping label '{label}' because no files were found.")
            continue
        random.shuffle(existing)
        n = len(existing)
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])
        # ensure we don't exceed length; assign remainder to test
        i1 = n_train
        i2 = i1 + n_val
        train_files = existing[:i1]
        val_files = existing[i1:i2]
        test_files = existing[i2:]
        per_split_lists["train"][label] = train_files
        per_split_lists["val"][label] = val_files
        per_split_lists["test"][label] = test_files

    # 3) Create directories and copy/symlink files, build manifest
    manifest_rows = []
    per_class_counts: Dict[str, Dict[str, int]] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name in splits_names:
        split_root = out_root_p / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        for label, files in per_split_lists[split_name].items():
            if not files:
                continue
            class_dir = split_root / label
            class_dir.mkdir(parents=True, exist_ok=True)
            per_class_counts.setdefault(label, {"train": 0, "val": 0, "test": 0})
            for src in files:
                if not src.exists():
                    # skip safety
                    continue
                dst = class_dir / src.name
                if dst.exists():
                    # skip if already there (idempotent)
                    action = "exists"
                else:
                    try:
                        if mode == "copy":
                            shutil.copy2(src, dst)
                        else:
                            # try symlink (may fail on Windows without privileges)
                            os.symlink(src.resolve(), dst)
                        action = mode
                    except Exception as e:
                        # fallback to copy if symlink fails
                        if mode == "symlink":
                            if verbose:
                                print(f"Symlink failed for {src} -> {dst}: {e}. Falling back to copy.")
                            shutil.copy2(src, dst)
                            action = "copy"
                        else:
                            raise
                counts[split_name] += 1
                per_class_counts[label][split_name] += 1
                manifest_rows.append({
                    "src": str(src.resolve()),
                    "dst": str(dst.resolve()),
                    "split": split_name,
                    "label": label,
                    "filename": dst.name,
                    "action": action
                })

    # 4) Write manifest
    manifest_path = out_root_p / manifest_name
    with manifest_path.open("w", newline="", encoding="utf-8") as mfh:
        fieldnames = ["src", "dst", "split", "label", "filename", "action"]
        writer = csv.DictWriter(mfh, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)

    if verbose:
        print("Split complete.")
        print(f"Counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        print(f"Manifest written to: {manifest_path.resolve()}")

    return {
        "counts": counts,
        "manifest": manifest_path,
        "per_class_counts": per_class_counts
    }


def setup_paths():
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "c_mnist"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_paths()
    summary = structure_c_mnist_with_splits(
        data_root="data",
        images_subdir="c_mnist",
        csv_filename="chinese_mnist.csv",
        out_root="data",
        splits=(0.8, 0.1, 0.1),
        class_by="value_char",
        seed=123,
        mode="copy",
        manifest_name="manifest.csv",
        verbose=True
    )
    print(summary)
