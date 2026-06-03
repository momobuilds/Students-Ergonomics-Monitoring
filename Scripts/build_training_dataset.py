#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Builds data/raw/training_total.csv by merging the feature modules
(posture, blinking, heart) for the normal and anomalous classes.

Logic:
  1. for each class (normal/anomalous) merge the modules HORIZONTALLY,
     aligning them on window_start + video_source;
  2. drop the window_start and video_source columns;
  3. add a 'label' column (normal / anomalous);
  4. stack normal + anomalous VERTICALLY.

The heart_*.csv files may not exist yet: they are skipped with a warning
and included automatically once they are present in data/raw/.

Usage:
    python build_training_dataset.py
(works from any folder: it walks up until it finds data/raw)
"""

from pathlib import Path
import pandas as pd

MODULES = ["posture", "blinking", "heart"]
CLASSES = ["normal", "anomalous"]
KEY_COLS = ["window_start", "video_source"]   # used to align, then dropped


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

        df = df.drop(columns=[c for c in KEY_COLS if c in df.columns], errors="ignore")

        cols = [c for c in df.columns if c != "label"] + ["label"]
        df = df[cols]

        out_path = raw_dir / f"training_{cls}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"File written: {out_path}")
        print(f"Shape:        {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Done.")


if __name__ == "__main__":
    main()