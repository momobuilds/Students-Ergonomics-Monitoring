#!/usr/bin/env python3
"""
Builds the training datasets by merging the feature modules
(posture, blinking, heart) for the normal, sleepy and stress classes.

It writes TWO sets of files:

  data/supervised/      (one row per window, WITH a 'label' column)
      normal.csv  sleepy.csv  stress.csv

  data/unsupervised/    (same features, WITHOUT the label column)
      normal.csv          -> only normal windows (train an anomaly detector)
      anomalous.csv       -> sleepy + stress stacked together (for evaluation)

Logic:
  1. for each class merge the modules HORIZONTALLY, aligning them on
     window_start + video_source;
  2. a module is included only if it has BOTH key columns; modules that
     cannot be aligned (e.g. the current heart_*.csv, which use ';' and have
     no window_start/video_source) are skipped with a warning;
  3. the 'video_source' value is simplified to the person's name
     (Alessia / Alvaro / Beatrice / Mo);
  4. supervised files keep a single 'label' column at the end;
     unsupervised files drop it.

Usage:
    python build_training_dataset.py
(works from any folder: it walks up until it finds data/raw)
"""

import re
from pathlib import Path
import pandas as pd

# modules that carry window_start + video_source and merge on those keys
KEYED_MODULES = ["posture", "blinking"]
BASE_CLASSES = ["normal", "sleepy", "stress"]
KEY_COLS = ["window_start", "video_source"]   # used to align the modules

# heart_*.csv have no keys: they are ';'-separated, hold HR/RMMSD/Target and a
# row count that can be slightly larger. They are aligned BY POSITION and any
# extra rows are trimmed from the bottom. 'Target' (a label) is dropped.
HEART_DROP_COLS = ["Target"]

# Known participants. video_source is reduced to one of these names.
NAMES = ["Alessia", "Alvaro", "Beatrice", "Mo"]

# Some 'normal' clips don't carry the person's name in the filename, so we map
# them explicitly. Keys are the lowercased filename stem (without extension).
# NOTE: verify this mapping is correct for your recordings and flip if needed.
VIDEO_NAME_OVERRIDES = {
    "straight_video":   "Alessia",
    "straight_video_2": "Beatrice",
}


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "raw").is_dir():
            return p
    raise SystemExit("Could not find a data/raw folder walking up the directories.")


def simplify_name(video_source: str) -> str:
    """Reduce a video_source value to a single participant name."""
    stem = Path(str(video_source)).stem
    key = stem.lower()
    if key in VIDEO_NAME_OVERRIDES:
        return VIDEO_NAME_OVERRIDES[key]
    tokens = set(re.split(r"[^a-z0-9]+", key))
    for name in NAMES:
        if name.lower() in tokens:
            return name
    # no known name found -> keep the original stem so nothing is silently lost
    print(f"  [warning] could not map video_source '{video_source}' to a name")
    return stem


def find_heart_file(cls: str, raw_dir: Path) -> Path | None:
    """heart file naming is inconsistent (e.g. Heart_stressed.csv for stress)."""
    for path in sorted(raw_dir.glob(f"[Hh]eart_{cls}*.csv")):
        return path
    return None


def attach_heart(merged: pd.DataFrame, cls: str, raw_dir: Path) -> pd.DataFrame:
    """Align the keyless heart series by position; trim extra rows from bottom."""
    path = find_heart_file(cls, raw_dir)
    if path is None:
        print(f"  [warning] no heart file for '{cls}' -> heart features skipped")
        return merged
    heart = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    heart = heart.drop(columns=HEART_DROP_COLS, errors="ignore")
    print(f"  {path.name}: {heart.shape[0]} rows, {heart.shape[1]} columns")

    n = min(len(merged), len(heart))
    if len(heart) != len(merged):
        print(f"  [info] heart has {len(heart)} rows vs {len(merged)} -> "
              f"trimming both to {n} from the bottom")
    merged = pd.concat(
        [merged.reset_index(drop=True).iloc[:n],
         heart.reset_index(drop=True).iloc[:n]],
        axis=1,
    )
    return merged


def build_class(cls: str, raw_dir: Path) -> pd.DataFrame:
    """Merge the feature modules for one class. Returns features + 'label'."""
    merged = None
    for module in KEYED_MODULES:
        path = raw_dir / f"{module}_{cls}.csv"
        if not path.exists():
            print(f"  [warning] missing {path.name} -> skipped")
            continue
        df = pd.read_csv(path)
        # the per-module 'label' column is re-added below; drop it so the
        # horizontal merge doesn't produce label_blinking / label_heart junk
        df = df.drop(columns=["label"], errors="ignore")
        if not set(KEY_COLS).issubset(df.columns):
            print(f"  [warning] {path.name} has no {KEY_COLS} -> cannot align, skipped")
            continue
        print(f"  {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=KEY_COLS, how="inner",
                                  suffixes=("", f"_{module}"))
    if merged is None:
        raise SystemExit(f"No alignable files found for class '{cls}'.")

    merged = attach_heart(merged, cls, raw_dir)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged["video_source"] = merged["video_source"].map(simplify_name)
    merged["label"] = cls

    key_present = [c for c in KEY_COLS if c in merged.columns]
    feature_cols = [c for c in merged.columns if c not in KEY_COLS and c != "label"]
    merged = merged[key_present + feature_cols + ["label"]]
    print(f"  -> class '{cls}': {merged.shape[0]} rows after merge\n")
    return merged


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"File written: {path}  ({df.shape[0]} rows x {df.shape[1]} columns)")


def main():
    root = find_project_root(Path(__file__).resolve().parent)
    raw_dir = root / "data" / "raw"
    sup_dir = root / "data" / "supervised"
    uns_dir = root / "data" / "unsupervised"

    # 1) build every base class once (features + label)
    by_class = {}
    for cls in BASE_CLASSES:
        print(f"Class: {cls}")
        by_class[cls] = build_class(cls, raw_dir)

    # sleepy + stress stacked together (built once, reused below)
    anomalous = pd.concat(
        [by_class["sleepy"], by_class["stress"]], ignore_index=True
    )

    # 2) SUPERVISED: keep the label, one file per class
    print("== Supervised (with label) ==")
    for cls in BASE_CLASSES:
        write_csv(by_class[cls], sup_dir / f"{cls}.csv")
    # combined anomalous class: label becomes 'anomalous' (normal vs anomalous)
    anomalous_sup = anomalous.assign(label="anomalous")
    write_csv(anomalous_sup, sup_dir / "anomalous.csv")
    print()

    # 3) UNSUPERVISED: drop the label
    print("== Unsupervised (no label) ==")
    write_csv(by_class["normal"].drop(columns=["label"]), uns_dir / "normal.csv")
    write_csv(anomalous.assign(label="anomalous"), uns_dir / "anomalous.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
