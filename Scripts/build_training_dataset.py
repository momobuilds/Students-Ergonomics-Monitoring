#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Builds data/raw/training_<class>.csv by merging the feature modules
(posture, blinking, heart) for the normal, sleepy and stress classes.

Logic:
  1. for each class (normal/sleepy/stress) merge the modules HORIZONTALLY,
     aligning them on window_start + video_source;
  2. keep window_start + video_source as the leading columns (once);
  3. add a 'label' column with the class name.

The heart_*.csv files may not exist yet: they are skipped with a warning
and included automatically once they are present in data/raw/.

Usage:
    python build_training_dataset.py
(works from any folder: it walks up until it finds data/raw)
"""

from pathlib import Path
import pandas as pd

MODULES = ["posture", "blinking", "heart"]
CLASSES = ["normal", "sleepy", "stress"]
KEY_COLS = ["window_start", "video_source"]   # used to align the modules


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "raw").is_dir():
            return p
    raise SystemExit("Could not find a data/raw folder walking up the directories.")


def build_class(cls: str, raw_dir: Path) -> pd.DataFrame:
    merged = None
    for module in MODULES:
        path = raw_dir / f"{module}_{cls}.csv"
        if not path.exists():
            print(f"  [warning] missing {path.name} -> skipped")
            continue
        df = pd.read_csv(path)
        # the per-module 'label' column is re-added below; drop it so the
        # horizontal merge doesn't produce label_blinking / label_heart junk
        df = df.drop(columns=["label"], errors="ignore")
        print(f"  {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        if merged is None:
            merged = df
        else:
            keys = [k for k in KEY_COLS if k in df.columns and k in merged.columns]
            if keys:
                merged = merged.merge(df, on=keys, how="inner",
                                      suffixes=("", f"_{module}"))
            else:
                # fallback: no common keys -> concatenate side by side by position
                merged = pd.concat(
                    [merged.reset_index(drop=True), df.reset_index(drop=True)],
                    axis=1)
    if merged is None:
        raise SystemExit(f"No files found for class '{cls}'.")
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged["label"] = cls
    print(f"  -> class '{cls}': {merged.shape[0]} rows after merge\n")
    return merged


def main():
    root = find_project_root(Path(__file__).resolve().parent)
    raw_dir = root / "data" / "raw"

    for cls in CLASSES:
        print(f"Class: {cls}")
        df = build_class(cls, raw_dir)

        key_present = [c for c in KEY_COLS if c in df.columns]
        other_cols  = [c for c in df.columns if c not in KEY_COLS and c != "label"]
        cols = key_present + other_cols + ["label"]
        df = df[cols]

        out_path = raw_dir / f"training_{cls}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"File written: {out_path}")
        print(f"Shape:        {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Combined 'anomalous' set: stack sleepy + stress VERTICALLY, reading the
    # already-built per-class CSVs (no re-processing). Label becomes 'anomalous'.
    print("Class: anomalous (sleepy + stress)")
    parts = []
    for cls in ("sleepy", "stress"):
        path = raw_dir / f"training_{cls}.csv"
        if not path.exists():
            print(f"  [warning] missing {path.name} -> skipped")
            continue
        part = pd.read_csv(path)
        print(f"  {path.name}: {part.shape[0]} rows")
        parts.append(part)

    if parts:
        anomalous = pd.concat(parts, ignore_index=True)
        anomalous["label"] = "anomalous"
        out_path = raw_dir / "training_anomalous.csv"
        anomalous.to_csv(out_path, index=False)
        print(f"File written: {out_path}")
        print(f"Shape:        {anomalous.shape[0]} rows x {anomalous.shape[1]} columns\n")
    else:
        print("  [warning] no sleepy/stress CSV found -> training_anomalous.csv not written\n")

    print("Done.")


if __name__ == "__main__":
    main()