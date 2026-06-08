#!/usr/bin/env python3
"""Unsupervised LOSO sensor ablation: cluster subsets of the 3 sensor groups.

Sensors (same 18 features as the rest of the project):
    Posture (camera body)   = POSTURE_FEATURES  (9)
    Blink   (camera face)   = BLINK_FEATURES    (7)
    Heart   (wearable)      = HEART_FEATURES    (HR, RMMSD; 2)

For every kept-sensor configuration (3 pairs + 3 singles) we evaluate three
scenarios, mirroring Scripts/ML/supervised_sensor_ablation_experiment.py:
    * "full"   : fit and predict using all 18 features (reference baseline).
    * "clean"  : fit and predict using only the kept-sensor features. This is
                 the realistic "we just don't have that sensor" setup the
                 real-time GUI would actually face.
    * "masked" : fit with all 18 features on the training subjects, then set
                 the excluded-sensor columns to NaN on the held-out subject so
                 the training-fold SimpleImputer fills them with the training
                 median. Deployment-mismatch stress test, reported only for
                 analysis.

Evaluation: Leave-One-Subject-Out on GROUP_COL ("video_source"). Each fold
fits SimpleImputer(median) + StandardScaler + GaussianMixture only on the
training subjects, freezes a cluster -> label mapping by majority vote on
the training assignments, then applies that mapping when predicting the
held-out subject. This mirrors how the offline-trained GMM in the GUI labels
real-time data from subjects it has never seen.

Model: Gaussian Mixture only (best unsupervised baseline in this project).
Tasks: binary (normal vs anomalous, n_components=2) and direct_3class
(normal/sleepy/stress, n_components=3).

Outputs go under results/unsupervised/sensor_ablation/.
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message=".*physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from unsupervised_augmentation_analysis import (
    BINARY_LABELS,
    GROUP_COL,
    THREECLASS_LABELS,
    build_binary_dataset,
    build_threeclass_dataset,
    coerce_features,
    load_csvs,
    preprocessing_pipeline,
)
from unsupervised_wellbeing_analysis import (
    classification_metrics,
    clustering_models as binary_clustering_models,
    majority_vote_mapping,
    mapped_labels,
    save_confusion_matrix_png,
)
from unsupervised_3group_wellbeing_analysis import (
    clustering_models as threeclass_clustering_models,
)


RANDOM_STATE = 42

POSTURE_FEATURES = [
    "p_shoulder_mean", "p_shoulder_std", "p_shoulder_bad_pct",
    "p_head_tilt_mean", "p_head_tilt_std", "p_head_tilt_bad_pct",
    "p_fhp_mean", "p_fhp_std", "p_fhp_bad_pct",
]
BLINK_FEATURES = [
    "b_ear_mean", "b_ear_std", "b_perclos",
    "b_blink_count", "b_blink_freq", "b_ibi_mean", "b_ibi_std",
]
HEART_FEATURES = ["HR", "RMMSD"]
ALL_FEATURES = POSTURE_FEATURES + BLINK_FEATURES + HEART_FEATURES

SENSOR_GROUPS: Dict[str, List[str]] = {
    "posture": POSTURE_FEATURES,
    "blink": BLINK_FEATURES,
    "heart": HEART_FEATURES,
}

SENSOR_LABEL = {
    "posture": "Posture",
    "blink": "Blink",
    "heart": "Heart",
}

# 3 pairs (keep 2, drop 1) + 3 singles (keep 1, drop 2).
CONFIGURATIONS: List[Dict[str, object]] = [
    {"id": "keep_posture_blink", "keep": ["posture", "blink"]},
    {"id": "keep_posture_heart", "keep": ["posture", "heart"]},
    {"id": "keep_blink_heart",   "keep": ["blink", "heart"]},
    {"id": "keep_posture",       "keep": ["posture"]},
    {"id": "keep_blink",         "keep": ["blink"]},
    {"id": "keep_heart",         "keep": ["heart"]},
]

TASKS: List[Dict[str, object]] = [
    {
        "id": "binary",
        "labels": BINARY_LABELS,
        "models_fn": binary_clustering_models,
    },
    {
        "id": "direct_3class",
        "labels": THREECLASS_LABELS,
        "models_fn": threeclass_clustering_models,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unsupervised LOSO sensor ablation across Posture, Blink and Heart "
            "feature groups, using Gaussian Mixture clustering only."
        )
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
        default=Path("results/unsupervised/sensor_ablation"),
        help="Output directory for the unsupervised sensor ablation results.",
    )
    return parser.parse_args()


def config_keep_features(keep_sensors: Sequence[str]) -> List[str]:
    """Concatenate sensor feature lists in canonical ALL_FEATURES order."""
    kept = set()
    for sensor in keep_sensors:
        kept.update(SENSOR_GROUPS[sensor])
    return [feature for feature in ALL_FEATURES if feature in kept]


def config_excluded_features(keep_sensors: Sequence[str]) -> List[str]:
    """Features for excluded sensors (canonical ALL_FEATURES order)."""
    kept = set(config_keep_features(keep_sensors))
    return [feature for feature in ALL_FEATURES if feature not in kept]


def configuration_label(keep_sensors: Sequence[str]) -> str:
    kept = " + ".join(SENSOR_LABEL[s] for s in keep_sensors)
    dropped_keys = [s for s in SENSOR_GROUPS if s not in keep_sensors]
    if dropped_keys:
        dropped = " + ".join(SENSOR_LABEL[s] for s in dropped_keys)
        return f"Keep {kept}  /  Exclude {dropped}"
    return f"Keep {kept}"


def run_gmm_loso(
    df: pd.DataFrame,
    train_features: Sequence[str],
    mask_features: Sequence[str],
    labels: Sequence[str],
    models_fn,
    group_col: str,
    random_state: int,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Leave-One-Subject-Out Gaussian Mixture evaluation.

    Each fold fits SimpleImputer(median) + StandardScaler + GaussianMixture
    only on the training subjects, derives a cluster -> label mapping by
    majority vote on the training assignments (a component with no training
    rows falls back to the most frequent training label so test predictions
    never KeyError), and applies that frozen mapping when predicting the
    held-out subject.

    Optional `mask_features` are set to NaN in the test fold so the
    training-fold imputer fills them with the training median ("masked"
    scenario). With `mask_features=[]` and `train_features=keep_features`
    this is the "clean" scenario; with `train_features=ALL_FEATURES` it is
    "full".
    """
    train_features = list(train_features)
    mask_set = set(mask_features) if mask_features else set()
    groups = sorted(df[group_col].dropna().astype(str).unique().tolist())

    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for heldout in groups:
        train_mask = df[group_col].astype(str) != heldout
        test_mask = df[group_col].astype(str) == heldout
        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        pipeline = preprocessing_pipeline()
        X_train = pipeline.fit_transform(train_df.loc[:, train_features])

        X_test_raw = test_df.loc[:, train_features].copy()
        for feature in train_features:
            if feature in mask_set:
                X_test_raw[feature] = np.nan
        X_test = pipeline.transform(X_test_raw)

        model = clone(models_fn(random_state)["Gaussian Mixture"])
        train_clusters = np.asarray(model.fit_predict(X_train))
        y_train = train_df["true_label"].tolist()

        mapping = majority_vote_mapping(y_train, train_clusters)
        majority_label = str(pd.Series(y_train).value_counts().idxmax())
        for component_id in range(int(model.n_components)):
            mapping.setdefault(component_id, majority_label)

        test_clusters = np.asarray(model.predict(X_test))
        y_pred = mapped_labels(test_clusters, mapping)
        y_true = test_df["true_label"].tolist()

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    metrics = classification_metrics(y_true_all, y_pred_all, labels=list(labels))
    return metrics, y_pred_all, y_true_all


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def fmt_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def render_ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  "
    out_lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out_lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out_lines.append(sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out_lines)


def render_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    label_width = max(max(len(str(l)) for l in labels), len("true \\ pred"))
    cell_width = max(
        len(str(int(cm.max()))) if cm.size else 1,
        max(len(str(l)) for l in labels),
    )
    header = (
        "true \\ pred".ljust(label_width)
        + " | "
        + " | ".join(str(l).rjust(cell_width) for l in labels)
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for i, true_label in enumerate(labels):
        cells = " | ".join(str(int(cm[i, j])).rjust(cell_width) for j in range(len(labels)))
        rows.append(str(true_label).ljust(label_width) + " | " + cells)
    return "\n".join(rows)


def interpretation_lines(
    keep_sensors: Sequence[str],
    full_by_task: Mapping[str, Mapping[str, object]],
    clean_by_task: Mapping[str, Mapping[str, object]],
    masked_by_task: Mapping[str, Mapping[str, object]],
) -> List[str]:
    dropped_keys = [s for s in SENSOR_GROUPS if s not in keep_sensors]
    dropped_text = (
        " + ".join(SENSOR_LABEL[s] for s in dropped_keys) if dropped_keys else "nothing"
    )

    drops_meaningful: List[str] = []
    drops_minor: List[str] = []
    gains: List[str] = []
    for task in TASKS:
        task_id = str(task["id"])
        delta = (
            float(clean_by_task[task_id]["metrics"]["macro_f1"])
            - float(full_by_task[task_id]["metrics"]["macro_f1"])
        )
        if delta <= -0.05:
            drops_meaningful.append(f"{task_id} (delta {delta:+.3f})")
        elif delta < -0.005:
            drops_minor.append(f"{task_id} (delta {delta:+.3f})")
        elif delta > 0.005:
            gains.append(f"{task_id} (delta {delta:+.3f})")

    lines: List[str] = []
    if drops_meaningful:
        lines.append(
            f"Removing {dropped_text} hurts macro F1 most on: "
            + ", ".join(drops_meaningful) + "."
        )
    if drops_minor and not drops_meaningful:
        lines.append(
            f"Removing {dropped_text} only slightly lowers macro F1 on: "
            + ", ".join(drops_minor) + "."
        )
    elif drops_minor:
        lines.append("Smaller drops also on: " + ", ".join(drops_minor) + ".")
    if gains:
        lines.append("Surprisingly improves on: " + ", ".join(gains) + ".")
    if not lines:
        lines.append(
            f"Removing {dropped_text} leaves macro F1 essentially unchanged across both tasks."
        )

    # Clean vs masked comparison (deployment-mismatch test).
    clean_minus_masked: List[Tuple[str, float]] = []
    for task in TASKS:
        task_id = str(task["id"])
        delta = (
            float(clean_by_task[task_id]["metrics"]["macro_f1"])
            - float(masked_by_task[task_id]["metrics"]["macro_f1"])
        )
        clean_minus_masked.append((task_id, delta))
    better_clean = [t for t, d in clean_minus_masked if d > 0.005]
    better_masked = [t for t, d in clean_minus_masked if d < -0.005]
    if better_clean and not better_masked:
        lines.append(
            "Training only on the kept sensors (clean) beats train-with-all-then-mask on: "
            + ", ".join(better_clean) + "."
        )
    elif better_masked and not better_clean:
        lines.append(
            "Imputing the removed sensors at test time (masked) is better on: "
            + ", ".join(better_masked) + "."
        )
    elif better_clean and better_masked:
        lines.append(
            "Clean wins on " + ", ".join(better_clean)
            + "; masked wins on " + ", ".join(better_masked) + "."
        )
    else:
        lines.append("Clean and masked scenarios give essentially the same macro F1.")
    return lines


def build_report(
    full_by_task: Mapping[str, Mapping[str, object]],
    config_results: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> str:
    sections: List[str] = []
    sections.append("UNSUPERVISED LOSO SENSOR ABLATION REPORT")
    sections.append("=" * 80)
    sections.append("")
    sections.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(
        "Model: Gaussian Mixture (sklearn.mixture.GaussianMixture, covariance_type='full')."
    )
    sections.append(
        "Pipeline: SimpleImputer(median) + StandardScaler + Gaussian Mixture, "
        "fit ONLY on the training subjects of each LOSO fold."
    )
    sections.append(
        f"Cross-validation: Leave-One-Subject-Out on group column = {GROUP_COL}."
    )
    sections.append(
        "Tasks: binary (normal vs anomalous, n_components=2) and direct_3class "
        "(normal/sleepy/stress, n_components=3)."
    )
    sections.append(
        "Method: per fold, cluster the training subjects with Gaussian Mixture; "
        "freeze a cluster -> label mapping by majority vote on the training "
        "assignments (a component with no training rows falls back to the most "
        "frequent training label); then apply that frozen mapping when predicting "
        "the held-out subject. This mirrors how the offline-trained GMM in the "
        "GUI labels real-time data from subjects it has never seen."
    )
    sections.append(f"Random state: {RANDOM_STATE}.")
    sections.append("")
    sections.append("Sensor groups:")
    sections.append(
        f"  - Posture (camera body): {len(POSTURE_FEATURES)} features -> "
        + ", ".join(POSTURE_FEATURES)
    )
    sections.append(
        f"  - Blink   (camera face): {len(BLINK_FEATURES)} features -> "
        + ", ".join(BLINK_FEATURES)
    )
    sections.append(
        f"  - Heart   (wearable)   : {len(HEART_FEATURES)} features -> "
        + ", ".join(HEART_FEATURES)
    )
    sections.append("")
    sections.append("Scenarios:")
    sections.append(
        "  full   : fit and predict using all 18 features. Reference baseline per task."
    )
    sections.append(
        "  clean  : fit and predict using only the features of the kept sensors. "
        "The realistic 'we just don't have this sensor' setup the GUI would face. "
        "This is the trustworthy scenario for the ablation."
    )
    sections.append(
        "  masked : fit with all 18 features on the training subjects, then set "
        "the excluded-sensor columns to NaN on the held-out subject so the "
        "training-fold SimpleImputer fills them with the training median. "
        "Deployment-mismatch stress test, reported only for analysis/comparison."
    )
    sections.append("")
    sections.append("FULL reference (all 18 features) per task:")
    full_rows = []
    for task in TASKS:
        task_id = str(task["id"])
        m = full_by_task[task_id]["metrics"]  # type: ignore[index]
        full_rows.append([
            task_id,
            fmt(m["macro_f1"]),
            fmt(m["balanced_accuracy"]),
            fmt(m["accuracy"]),
            fmt(m["weighted_f1"]),
        ])
    sections.append(
        render_ascii_table(
            ["task", "macro_f1", "bal_acc", "accuracy", "weighted_f1"],
            full_rows,
        )
    )
    sections.append("")

    for entry in config_results:
        config_id = entry["id"]
        keep_sensors = entry["keep_sensors"]  # type: ignore[index]
        keep_features = entry["keep_features"]  # type: ignore[index]
        mask_features = entry["mask_features"]  # type: ignore[index]

        sections.append("=" * 80)
        sections.append(f"CONFIGURATION: {config_id}")
        sections.append(configuration_label(keep_sensors))  # type: ignore[arg-type]
        sections.append(
            f"Kept features ({len(keep_features)}): " + ", ".join(keep_features)
        )
        if mask_features:
            sections.append(
                f"Excluded features ({len(mask_features)}): "
                + ", ".join(mask_features)
            )
        else:
            sections.append("Excluded features: none.")
        sections.append("=" * 80)
        sections.append("")

        for task in TASKS:
            task_id = str(task["id"])
            labels = list(task["labels"])
            full = full_by_task[task_id]
            clean = entry["clean"][task_id]  # type: ignore[index]
            masked = entry["masked"][task_id]  # type: ignore[index]

            sections.append(f"-- TASK: {task_id} --")
            table_rows = [
                [
                    "FULL (18)",
                    fmt(full["metrics"]["macro_f1"]),  # type: ignore[index]
                    fmt(full["metrics"]["balanced_accuracy"]),  # type: ignore[index]
                    fmt(full["metrics"]["accuracy"]),  # type: ignore[index]
                    fmt(full["metrics"]["weighted_f1"]),  # type: ignore[index]
                    "-",
                    "-",
                ],
                [
                    f"CLEAN ({clean['n_features']})",
                    fmt(clean["metrics"]["macro_f1"]),
                    fmt(clean["metrics"]["balanced_accuracy"]),
                    fmt(clean["metrics"]["accuracy"]),
                    fmt(clean["metrics"]["weighted_f1"]),
                    fmt_signed(
                        clean["metrics"]["macro_f1"] - full["metrics"]["macro_f1"]
                    ),
                    fmt_signed(
                        clean["metrics"]["balanced_accuracy"]
                        - full["metrics"]["balanced_accuracy"]
                    ),
                ],
                [
                    "MASKED (18)",
                    fmt(masked["metrics"]["macro_f1"]),
                    fmt(masked["metrics"]["balanced_accuracy"]),
                    fmt(masked["metrics"]["accuracy"]),
                    fmt(masked["metrics"]["weighted_f1"]),
                    fmt_signed(
                        masked["metrics"]["macro_f1"] - full["metrics"]["macro_f1"]
                    ),
                    fmt_signed(
                        masked["metrics"]["balanced_accuracy"]
                        - full["metrics"]["balanced_accuracy"]
                    ),
                ],
            ]
            sections.append(
                render_ascii_table(
                    [
                        "scenario",
                        "macro_f1",
                        "bal_acc",
                        "accuracy",
                        "weighted_f1",
                        "dF1_vs_full",
                        "dBA_vs_full",
                    ],
                    table_rows,
                )
            )
            sections.append("")
            sections.append("Confusion matrix - FULL (reference):")
            sections.append(
                render_confusion_matrix(full["y_true"], full["y_pred"], labels)
            )
            sections.append("")
            sections.append("Confusion matrix - CLEAN:")
            sections.append(
                render_confusion_matrix(clean["y_true"], clean["y_pred"], labels)
            )
            sections.append("")
            sections.append("Confusion matrix - MASKED:")
            sections.append(
                render_confusion_matrix(masked["y_true"], masked["y_pred"], labels)
            )
            sections.append("")

        sections.append("Interpretation:")
        for line in interpretation_lines(
            keep_sensors,  # type: ignore[arg-type]
            full_by_task,
            entry["clean"],  # type: ignore[arg-type]
            entry["masked"],  # type: ignore[arg-type]
        ):
            sections.append("  - " + line)
        sections.append("")

    sections.append("=" * 80)
    sections.append("OVERALL SUMMARY")
    sections.append("=" * 80)
    sections.append("")
    rows = []
    for row in summary_rows:
        rows.append([
            row["configuration"],
            row["task"],
            row["scenario"],
            str(row["n_features"]),
            fmt(row["macro_f1"]),
            fmt(row["balanced_accuracy"]),
            fmt(row["accuracy"]),
            fmt_signed(row["macro_f1_delta_vs_full"]),
        ])
    sections.append(
        render_ascii_table(
            [
                "configuration",
                "task",
                "scenario",
                "n_feat",
                "macro_f1",
                "bal_acc",
                "accuracy",
                "dF1_vs_full",
            ],
            rows,
        )
    )
    sections.append("")
    return "\n".join(sections)


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
    task_dataframes: Dict[str, pd.DataFrame] = {
        "binary": binary_df,
        "direct_3class": three_df,
    }

    summary_rows: List[Dict[str, object]] = []
    full_by_task: Dict[str, Dict[str, object]] = {}

    # ----- FULL reference (all 18 features, no masking) -------------------
    full_dir = results_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_metric_rows: List[Dict[str, object]] = []
    for task in TASKS:
        task_id = str(task["id"])
        labels = list(task["labels"])
        base_df = task_dataframes[task_id]

        metrics, y_pred, y_true = run_gmm_loso(
            base_df,
            ALL_FEATURES,
            [],
            labels,
            task["models_fn"],
            GROUP_COL,
            RANDOM_STATE,
        )
        save_confusion_matrix_png(
            y_true,
            y_pred,
            labels,
            f"FULL LOSO Gaussian Mixture - {task_id}",
            full_dir / f"{task_id}_full_confusion_matrix.png",
        )
        full_by_task[task_id] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_features": len(ALL_FEATURES),
        }
        full_metric_rows.append({
            "task": task_id,
            "scenario": "full",
            "n_features": len(ALL_FEATURES),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "cluster_mapping": "LOSO (frozen per fold)",
        })
        summary_rows.append({
            "configuration": "full",
            "task": task_id,
            "scenario": "full",
            "n_features": len(ALL_FEATURES),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_f1_delta_vs_full": 0.0,
        })
        print(
            f"[FULL] task={task_id} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"bal_acc={metrics['balanced_accuracy']:.3f}"
        )
    pd.DataFrame(full_metric_rows).to_csv(full_dir / "metrics.csv", index=False)

    # ----- Per-configuration CLEAN + MASKED -------------------------------
    config_results: List[Dict[str, object]] = []
    for config in CONFIGURATIONS:
        config_id = str(config["id"])
        keep_sensors = list(config["keep"])  # type: ignore[arg-type]
        keep_features = config_keep_features(keep_sensors)
        mask_features = config_excluded_features(keep_sensors)

        config_dir = results_dir / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        entry: Dict[str, object] = {
            "id": config_id,
            "keep_sensors": keep_sensors,
            "keep_features": keep_features,
            "mask_features": mask_features,
            "clean": {},
            "masked": {},
        }
        metric_records: List[Dict[str, object]] = []

        for task in TASKS:
            task_id = str(task["id"])
            labels = list(task["labels"])
            base_df = task_dataframes[task_id]
            full_metrics = full_by_task[task_id]["metrics"]

            # --- clean: fit and predict using only the kept-sensor features.
            clean_metrics, clean_pred, clean_true = run_gmm_loso(
                base_df,
                keep_features,
                [],
                labels,
                task["models_fn"],
                GROUP_COL,
                RANDOM_STATE,
            )
            save_confusion_matrix_png(
                clean_true,
                clean_pred,
                labels,
                f"{config_id} CLEAN LOSO Gaussian Mixture - {task_id}",
                config_dir / f"{task_id}_clean_confusion_matrix.png",
            )
            entry["clean"][task_id] = {  # type: ignore[index]
                "metrics": clean_metrics,
                "y_true": clean_true,
                "y_pred": clean_pred,
                "n_features": len(keep_features),
            }

            # --- masked: train on all 18 features, mask excluded sensors at test.
            masked_metrics, masked_pred, masked_true = run_gmm_loso(
                base_df,
                ALL_FEATURES,
                mask_features,
                labels,
                task["models_fn"],
                GROUP_COL,
                RANDOM_STATE,
            )
            save_confusion_matrix_png(
                masked_true,
                masked_pred,
                labels,
                f"{config_id} MASKED LOSO Gaussian Mixture - {task_id}",
                config_dir / f"{task_id}_masked_confusion_matrix.png",
            )
            entry["masked"][task_id] = {  # type: ignore[index]
                "metrics": masked_metrics,
                "y_true": masked_true,
                "y_pred": masked_pred,
                "n_features": len(ALL_FEATURES),
            }

            metric_records.append({
                "configuration": config_id,
                "task": task_id,
                "scenario": "full",
                "n_features": full_by_task[task_id]["n_features"],
                "macro_f1": full_metrics["macro_f1"],
                "balanced_accuracy": full_metrics["balanced_accuracy"],
                "accuracy": full_metrics["accuracy"],
                "weighted_f1": full_metrics["weighted_f1"],
                "macro_f1_delta_vs_full": 0.0,
                "balanced_accuracy_delta_vs_full": 0.0,
                "cluster_mapping": "LOSO (frozen per fold)",
            })
            metric_records.append({
                "configuration": config_id,
                "task": task_id,
                "scenario": "clean",
                "n_features": len(keep_features),
                "macro_f1": clean_metrics["macro_f1"],
                "balanced_accuracy": clean_metrics["balanced_accuracy"],
                "accuracy": clean_metrics["accuracy"],
                "weighted_f1": clean_metrics["weighted_f1"],
                "macro_f1_delta_vs_full": (
                    clean_metrics["macro_f1"] - full_metrics["macro_f1"]
                ),
                "balanced_accuracy_delta_vs_full": (
                    clean_metrics["balanced_accuracy"]
                    - full_metrics["balanced_accuracy"]
                ),
                "cluster_mapping": "LOSO (frozen per fold)",
            })
            metric_records.append({
                "configuration": config_id,
                "task": task_id,
                "scenario": "masked",
                "n_features": len(ALL_FEATURES),
                "macro_f1": masked_metrics["macro_f1"],
                "balanced_accuracy": masked_metrics["balanced_accuracy"],
                "accuracy": masked_metrics["accuracy"],
                "weighted_f1": masked_metrics["weighted_f1"],
                "macro_f1_delta_vs_full": (
                    masked_metrics["macro_f1"] - full_metrics["macro_f1"]
                ),
                "balanced_accuracy_delta_vs_full": (
                    masked_metrics["balanced_accuracy"]
                    - full_metrics["balanced_accuracy"]
                ),
                "cluster_mapping": "LOSO (frozen per fold)",
            })

            for scenario_name, scenario_metrics, n_feats in [
                ("clean", clean_metrics, len(keep_features)),
                ("masked", masked_metrics, len(ALL_FEATURES)),
            ]:
                summary_rows.append({
                    "configuration": config_id,
                    "task": task_id,
                    "scenario": scenario_name,
                    "n_features": n_feats,
                    "macro_f1": scenario_metrics["macro_f1"],
                    "balanced_accuracy": scenario_metrics["balanced_accuracy"],
                    "accuracy": scenario_metrics["accuracy"],
                    "macro_f1_delta_vs_full": (
                        scenario_metrics["macro_f1"] - full_metrics["macro_f1"]
                    ),
                })

            print(
                f"[{config_id}] task={task_id} "
                f"clean={clean_metrics['macro_f1']:.3f} "
                f"(dF1={clean_metrics['macro_f1'] - full_metrics['macro_f1']:+.3f}) | "
                f"masked={masked_metrics['macro_f1']:.3f} "
                f"(dF1={masked_metrics['macro_f1'] - full_metrics['macro_f1']:+.3f})"
            )

        pd.DataFrame(metric_records).to_csv(
            config_dir / "metrics.csv", index=False
        )
        config_results.append(entry)

    report = build_report(full_by_task, config_results, summary_rows)
    report_path = results_dir / "sensor_ablation_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    pd.DataFrame(summary_rows).to_csv(
        results_dir / "sensor_ablation_summary.csv", index=False
    )

    print(f"\nSaved sensor ablation outputs to: {results_dir}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
