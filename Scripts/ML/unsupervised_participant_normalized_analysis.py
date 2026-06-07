#!/usr/bin/env python3
"""Participant-normalized unsupervised clustering experiments.

This script tests whether clustering improves when each row is represented as
its difference or z-score from that participant's normal baseline.

Labels are not used to fit clustering models. The normal labels are used only
to construct a participant-specific baseline representation, and labels are
then used after clustering for evaluation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from unsupervised_augmentation_analysis import (
    BASELINES,
    BINARY_LABELS,
    FEATURE_COLUMNS,
    GROUP_COL,
    THREECLASS_LABELS,
    augmentation_settings,
    build_binary_dataset,
    build_clustering_summary,
    build_threeclass_dataset,
    coerce_features,
    format_table,
    load_csvs,
    predictions_for,
    run_clustering_augmentation,
    save_confusion_matrix_png,
    select_best,
)


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run participant-normalized unsupervised clustering experiments."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/unsupervised"),
        help="Directory containing normal.csv, anomalous.csv, sleepy.csv, stress.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/unsupervised/participant_normalized"),
        help="Output directory for participant-normalized results.",
    )
    parser.add_argument(
        "--normalization",
        choices=["delta", "zscore", "both"],
        default="both",
        help="Participant-normalized representation to test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for stochastic models and jitter.",
    )
    return parser.parse_args()


def modes_from_arg(value: str) -> List[str]:
    if value == "both":
        return ["delta", "zscore"]
    return [value]


def participant_normal_stats(
    df: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal = df[df["true_label"] == "normal"].copy()
    if normal.empty:
        raise ValueError("Participant normalization requires normal rows.")
    means = normal.groupby(GROUP_COL)[list(features)].mean()
    stds = normal.groupby(GROUP_COL)[list(features)].std(ddof=0)
    stds = stds.replace(0, np.nan).fillna(1.0)
    return means, stds


def add_participant_normalized_features(
    df: pd.DataFrame,
    features: Sequence[str],
    mode: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add delta or z-score features based on each participant's normal rows."""
    means, stds = participant_normal_stats(df, features)
    missing_groups = sorted(set(df[GROUP_COL].astype(str)) - set(means.index.astype(str)))
    if missing_groups:
        raise ValueError(
            "Cannot participant-normalize groups with no normal baseline: "
            + ", ".join(missing_groups)
        )

    transformed = df.copy()
    new_features = []
    for feature in features:
        if mode == "delta":
            new_feature = f"pn_delta_{feature}"
        elif mode == "zscore":
            new_feature = f"pn_z_{feature}"
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        group_means = transformed[GROUP_COL].map(means[feature])
        if mode == "delta":
            transformed[new_feature] = transformed[feature] - group_means
        else:
            group_stds = transformed[GROUP_COL].map(stds[feature])
            transformed[new_feature] = (transformed[feature] - group_means) / group_stds
        new_features.append(new_feature)

    return transformed, new_features


def save_best_confusions(
    mode_dir: Path,
    task_prefix: str,
    labels: Sequence[str],
    results: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
) -> None:
    for method in sorted(results["method"].unique()):
        baseline_best = select_best(results, exclude_baseline=False, method=method)
        baseline_pred = predictions_for(
            predictions,
            method,
            baseline_best["augmentation_name"],
        )
        save_confusion_matrix_png(
            baseline_pred["y_true"],
            baseline_pred["y_pred"],
            labels,
            f"{task_prefix} {method} Participant-Normalized Baseline",
            mode_dir
            / f"{task_prefix.lower()}_{method.lower().replace(' ', '_')}_pn_baseline_confusion_matrix.png",
        )

        augmented_best = select_best(results, exclude_baseline=True, method=method)
        augmented_pred = predictions_for(
            predictions,
            method,
            augmented_best["augmentation_name"],
        )
        save_confusion_matrix_png(
            augmented_pred["y_true"],
            augmented_pred["y_pred"],
            labels,
            f"{task_prefix} {method} Participant-Normalized Best Augmented",
            mode_dir
            / f"{task_prefix.lower()}_{method.lower().replace(' ', '_')}_pn_best_augmented_confusion_matrix.png",
        )


def run_mode(
    mode: str,
    binary_df: pd.DataFrame,
    three_df: pd.DataFrame,
    results_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    mode_dir = results_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    settings = augmentation_settings()

    binary_norm, binary_features = add_participant_normalized_features(
        binary_df,
        FEATURE_COLUMNS,
        mode,
    )
    three_norm, three_features = add_participant_normalized_features(
        three_df,
        FEATURE_COLUMNS,
        mode,
    )

    binary_results, binary_predictions = run_clustering_augmentation(
        binary_norm,
        binary_features,
        BINARY_LABELS,
        2,
        f"binary_clustering_participant_{mode}",
        random_state,
        settings,
    )
    binary_results.to_csv(
        mode_dir / "binary_participant_normalized_clustering_results.csv",
        index=False,
    )
    (mode_dir / "binary_participant_normalized_clustering_summary.txt").write_text(
        build_clustering_summary(
            f"BINARY CLUSTERING PARTICIPANT-{mode.upper()} SUMMARY",
            binary_results,
            BINARY_LABELS,
        )
        + "\n",
        encoding="utf-8",
    )
    save_best_confusions(
        mode_dir,
        "binary",
        BINARY_LABELS,
        binary_results,
        binary_predictions,
    )

    three_results, three_predictions = run_clustering_augmentation(
        three_norm,
        three_features,
        THREECLASS_LABELS,
        3,
        f"threeclass_clustering_participant_{mode}",
        random_state,
        settings,
    )
    three_results.to_csv(
        mode_dir / "threeclass_participant_normalized_clustering_results.csv",
        index=False,
    )
    (mode_dir / "threeclass_participant_normalized_clustering_summary.txt").write_text(
        build_clustering_summary(
            f"3-CLASS CLUSTERING PARTICIPANT-{mode.upper()} SUMMARY",
            three_results,
            THREECLASS_LABELS,
        )
        + "\n",
        encoding="utf-8",
    )
    save_best_confusions(
        mode_dir,
        "threeclass",
        THREECLASS_LABELS,
        three_results,
        three_predictions,
    )

    rows = []
    for task, results in [
        ("binary_clustering", binary_results),
        ("threeclass_clustering", three_results),
    ]:
        raw_baseline = BASELINES[task]
        normalized_baseline = select_best(results, exclude_baseline=False)
        normalized_augmented = select_best(results, exclude_baseline=True)
        rows.append(
            {
                "task": task,
                "normalization": mode,
                "raw_baseline_method": raw_baseline["method"],
                "raw_baseline_macro_f1": raw_baseline["macro_f1"],
                "raw_baseline_balanced_accuracy": raw_baseline["balanced_accuracy"],
                "participant_normalized_baseline_method": normalized_baseline["method"],
                "participant_normalized_baseline_macro_f1": normalized_baseline[
                    "macro_f1"
                ],
                "participant_normalized_baseline_balanced_accuracy": normalized_baseline[
                    "balanced_accuracy"
                ],
                "baseline_macro_f1_change_vs_raw": normalized_baseline["macro_f1"]
                - raw_baseline["macro_f1"],
                "baseline_balanced_accuracy_change_vs_raw": normalized_baseline[
                    "balanced_accuracy"
                ]
                - raw_baseline["balanced_accuracy"],
                "best_augmented_method": normalized_augmented["method"],
                "best_augmented_name": normalized_augmented["augmentation_name"],
                "best_augmented_factor": int(normalized_augmented["augmentation_factor"]),
                "best_augmented_noise_level": float(normalized_augmented["noise_level"]),
                "best_augmented_macro_f1": normalized_augmented["macro_f1"],
                "best_augmented_balanced_accuracy": normalized_augmented[
                    "balanced_accuracy"
                ],
                "best_augmented_macro_f1_change_vs_raw": normalized_augmented[
                    "macro_f1"
                ]
                - raw_baseline["macro_f1"],
                "best_augmented_balanced_accuracy_change_vs_raw": normalized_augmented[
                    "balanced_accuracy"
                ]
                - raw_baseline["balanced_accuracy"],
            }
        )
    mode_summary = pd.DataFrame(rows)
    mode_summary.to_csv(mode_dir / "participant_normalized_summary.csv", index=False)
    return mode_summary


def build_final_summary(comparison: pd.DataFrame) -> str:
    lines = ["PARTICIPANT-NORMALIZED UNSUPERVISED CLUSTERING SUMMARY", "=" * 80]
    lines.append(
        "Each feature was transformed relative to that participant's normal "
        "baseline before clustering. Clustering models still did not use labels "
        "during fitting; labels were used afterward for evaluation."
    )
    lines.append(
        "This is a normal-reference feature representation, so it assumes a "
        "normal calibration segment is available for each participant."
    )
    lines.append("")
    lines.append("Comparison against raw unsupervised baselines:")
    lines.append(format_table(comparison))
    lines.append("")

    for row in comparison.itertuples(index=False):
        delta = row.baseline_macro_f1_change_vs_raw
        aug_delta = row.best_augmented_macro_f1_change_vs_raw
        lines.append(
            f"{row.task} with participant {row.normalization}: normalized baseline "
            f"macro F1 {row.raw_baseline_macro_f1:.3f} -> "
            f"{row.participant_normalized_baseline_macro_f1:.3f} "
            f"(delta {delta:+.3f}); best normalized+augmented macro F1 "
            f"{row.best_augmented_macro_f1:.3f} (delta {aug_delta:+.3f})."
        )

    best = comparison.sort_values(
        [
            "best_augmented_macro_f1_change_vs_raw",
            "best_augmented_macro_f1",
            "best_augmented_balanced_accuracy",
        ],
        ascending=False,
    ).iloc[0]
    lines.append("")
    lines.append(
        "Best participant-normalized result: "
        f"{best['task']} using {best['normalization']} with "
        f"{best['best_augmented_method']} and {best['best_augmented_name']} "
        f"(macro F1={best['best_augmented_macro_f1']:.3f}, balanced accuracy="
        f"{best['best_augmented_balanced_accuracy']:.3f})."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = results_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    datasets = coerce_features(load_csvs(data_dir))
    binary_df = build_binary_dataset(datasets).reset_index(drop=True)
    three_df = build_threeclass_dataset(datasets).reset_index(drop=True)

    summaries = []
    for mode in modes_from_arg(args.normalization):
        print(f"Running participant-normalized clustering mode: {mode}")
        summaries.append(
            run_mode(
                mode,
                binary_df,
                three_df,
                results_dir,
                args.random_state,
            )
        )

    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(
        results_dir / "participant_normalized_vs_raw_summary.csv",
        index=False,
    )
    final_summary = build_final_summary(comparison)
    (results_dir / "final_participant_normalized_summary.txt").write_text(
        final_summary + "\n",
        encoding="utf-8",
    )
    print(final_summary)
    print(f"Saved participant-normalized outputs to: {results_dir}")


if __name__ == "__main__":
    main()
