#!/usr/bin/env python3
"""Supervised LOSO sensor ablation: train/test combinations of the 3 sensor groups.

Sensors:
    Posture  (camera body)   = POSTURE_FEATURES  (9)
    Blink    (camera face)   = BLINK_FEATURES    (7)
    Heart    (wearable)      = HEART_FEATURES    (HR, RMMSD; 2)

For every kept-sensor configuration (3 pairs + 3 singles) we evaluate:
    * "clean":   train and test using only the kept-sensor features.
    * "masked":  train with all 18 features, then set the excluded-sensor columns
                 to NaN at test time so the training-fold imputer fills them with
                 the training median.

A single "full" reference (all 18 features, no masking) is computed per task and
used to report clean-vs-full and masked-vs-full deltas.

Model: Random Forest only (best baseline model in this project). For the
hierarchical task both stages use Random Forest, i.e. "Random Forest -> Random
Forest".

Outputs go under results/supervised/sensor_ablation/.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from wellbeing_hierarchical_classification import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    build_binary_dataset,
    build_direct_3class_dataset,
    build_hierarchical_test_dataset,
    build_subtype_dataset,
    coerce_feature_columns,
    compute_metrics,
    infer_group_column,
    load_datasets,
    model_factories,
    resolve_csv_dir,
    save_confusion_matrix_png,
    validate_features,
)


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

# Six configurations: 3 pairs (keep 2, drop 1) + 3 singles (keep 1, drop 2).
CONFIGURATIONS: List[Dict[str, object]] = [
    {"id": "keep_posture_blink", "keep": ["posture", "blink"]},
    {"id": "keep_posture_heart", "keep": ["posture", "heart"]},
    {"id": "keep_blink_heart", "keep": ["blink", "heart"]},
    {"id": "keep_posture", "keep": ["posture"]},
    {"id": "keep_blink", "keep": ["blink"]},
    {"id": "keep_heart", "keep": ["heart"]},
]

TASKS: List[Dict[str, object]] = [
    {"id": "binary", "labels": ["normal", "abnormal"]},
    {"id": "subtype", "labels": ["sleepy", "stress"]},
    {"id": "hierarchical", "labels": ["normal", "sleepy", "stress"]},
    {"id": "direct_3class", "labels": ["normal", "sleepy", "stress"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised LOSO sensor ablation across Posture, Blink and Heart "
            "feature groups, using Random Forest only."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Project root or data/supervised directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/supervised/sensor_ablation"),
        help="Output directory for the sensor ablation results.",
    )
    return parser.parse_args()


def config_keep_features(keep_sensors: Sequence[str]) -> List[str]:
    """Concatenate sensor feature lists in canonical FEATURE_COLUMNS order."""
    kept = set()
    for sensor in keep_sensors:
        kept.update(SENSOR_GROUPS[sensor])
    return [feature for feature in FEATURE_COLUMNS if feature in kept]


def config_mask_features(keep_sensors: Sequence[str]) -> List[str]:
    """Features for excluded sensors (in canonical order)."""
    kept_set = set(config_keep_features(keep_sensors))
    return [feature for feature in FEATURE_COLUMNS if feature not in kept_set]


def configuration_label(keep_sensors: Sequence[str]) -> str:
    kept = " + ".join(SENSOR_LABEL[s] for s in keep_sensors)
    dropped_keys = [s for s in SENSOR_GROUPS if s not in keep_sensors]
    if dropped_keys:
        dropped = " + ".join(SENSOR_LABEL[s] for s in dropped_keys)
        return f"Keep {kept}  /  Exclude {dropped}"
    return f"Keep {kept}"


def mask_features_at_test(X_test: pd.DataFrame, mask_features: Sequence[str]) -> pd.DataFrame:
    masked = X_test.copy()
    for feature in mask_features:
        if feature in masked.columns:
            masked[feature] = pd.Series(np.nan, index=masked.index, dtype="float64")
    return masked


def evaluate_flat_loso(
    task_id: str,
    df: pd.DataFrame,
    group_col: str,
    labels: Sequence[str],
    rf_pipeline,
    train_features: Sequence[str],
    mask_features: Sequence[str],
    scenario: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """LOSO evaluation of a single Random Forest pipeline on a flat task."""
    logo = LeaveOneGroupOut()
    X = df.loc[:, list(train_features)]
    y = df["target"]
    groups = df[group_col].astype(str)

    records: List[Dict[str, object]] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = mask_features_at_test(X.iloc[test_idx], mask_features)
        y_train = y.iloc[train_idx]
        y_true = y.iloc[test_idx].to_numpy()

        fitted = clone(rf_pipeline)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)

        heldout = ",".join(sorted(groups.iloc[test_idx].unique().tolist()))
        row: Dict[str, object] = {
            "task": task_id,
            "scenario": scenario,
            "row_type": "fold",
            "fold": fold,
            "heldout_group": heldout,
            "n_train_features": len(train_features),
            "train_features": ",".join(train_features),
            "masked_test_features": ",".join(mask_features),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
        row.update(compute_metrics(y_true, y_pred, labels))
        records.append(row)

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    overall: Dict[str, object] = {
        "task": task_id,
        "scenario": scenario,
        "row_type": "overall",
        "fold": 0,
        "heldout_group": "ALL",
        "n_train_features": len(train_features),
        "train_features": ",".join(train_features),
        "masked_test_features": ",".join(mask_features),
        "train_rows": int(len(df)),
        "test_rows": int(len(df)),
    }
    overall.update(compute_metrics(y_true_all, y_pred_all, labels))
    records.append(overall)

    return pd.DataFrame(records), y_true_all, y_pred_all


def evaluate_hierarchical_loso(
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    group_col: str,
    labels: Sequence[str],
    rf_pipeline,
    train_features: Sequence[str],
    mask_features: Sequence[str],
    scenario: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """LOSO hierarchical evaluation: RF -> RF (binary -> subtype)."""
    group_values = sorted(final_df[group_col].dropna().astype(str).unique().tolist())
    feature_list = list(train_features)
    records: List[Dict[str, object]] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for fold, heldout in enumerate(group_values, start=1):
        binary_train = binary_df[binary_df[group_col].astype(str) != heldout]
        subtype_train = subtype_df[subtype_df[group_col].astype(str) != heldout]
        test = final_df[final_df[group_col].astype(str) == heldout]

        binary_model = clone(rf_pipeline)
        subtype_model = clone(rf_pipeline)
        binary_model.fit(binary_train.loc[:, feature_list], binary_train["target"])
        subtype_model.fit(subtype_train.loc[:, feature_list], subtype_train["target"])

        X_test = mask_features_at_test(test.loc[:, feature_list], mask_features)
        binary_pred = binary_model.predict(X_test)
        final_pred = np.array(
            ["normal" if pred == "normal" else "" for pred in binary_pred],
            dtype=object,
        )
        abnormal_mask = binary_pred == "abnormal"
        if abnormal_mask.any():
            final_pred[abnormal_mask] = subtype_model.predict(X_test.loc[abnormal_mask])

        y_true = test["target"].to_numpy()
        row: Dict[str, object] = {
            "task": "hierarchical",
            "scenario": scenario,
            "row_type": "fold",
            "fold": fold,
            "heldout_group": heldout,
            "n_train_features": len(train_features),
            "train_features": ",".join(train_features),
            "masked_test_features": ",".join(mask_features),
            "train_binary_rows": int(len(binary_train)),
            "train_subtype_rows": int(len(subtype_train)),
            "test_rows": int(len(test)),
        }
        row.update(compute_metrics(y_true, final_pred, labels))
        records.append(row)

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(final_pred.tolist())

    overall: Dict[str, object] = {
        "task": "hierarchical",
        "scenario": scenario,
        "row_type": "overall",
        "fold": 0,
        "heldout_group": "ALL",
        "n_train_features": len(train_features),
        "train_features": ",".join(train_features),
        "masked_test_features": ",".join(mask_features),
        "train_binary_rows": int(len(binary_df)),
        "train_subtype_rows": int(len(subtype_df)),
        "test_rows": int(len(final_df)),
    }
    overall.update(compute_metrics(y_true_all, y_pred_all, labels))
    records.append(overall)

    return pd.DataFrame(records), y_true_all, y_pred_all


def evaluate_task(
    task_id: str,
    labels: Sequence[str],
    rf_pipeline,
    train_features: Sequence[str],
    mask_features: Sequence[str],
    scenario: str,
    group_col: str,
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    direct_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if task_id == "binary":
        return evaluate_flat_loso(
            "binary", binary_df, group_col, labels, rf_pipeline,
            train_features, mask_features, scenario,
        )
    if task_id == "subtype":
        return evaluate_flat_loso(
            "subtype", subtype_df, group_col, labels, rf_pipeline,
            train_features, mask_features, scenario,
        )
    if task_id == "direct_3class":
        return evaluate_flat_loso(
            "direct_3class", direct_df, group_col, labels, rf_pipeline,
            train_features, mask_features, scenario,
        )
    if task_id == "hierarchical":
        return evaluate_hierarchical_loso(
            binary_df, subtype_df, final_df, group_col, labels, rf_pipeline,
            train_features, mask_features, scenario,
        )
    raise ValueError(f"Unknown task: {task_id}")


def overall_row(results: pd.DataFrame) -> pd.Series:
    return results[results["row_type"] == "overall"].iloc[0]


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
    out_lines = []
    out_lines.append(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    out_lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out_lines.append(sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out_lines)


def render_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    label_width = max(len(str(label)) for label in labels)
    label_width = max(label_width, len("true \\ pred"))
    cell_width = max(len(str(int(cm.max()))) if cm.size else 1, max(len(l) for l in labels))
    header = "true \\ pred".ljust(label_width) + " | " + " | ".join(str(l).rjust(cell_width) for l in labels)
    sep = "-" * len(header)
    rows = [header, sep]
    for i, true_label in enumerate(labels):
        row_cells = " | ".join(str(int(cm[i, j])).rjust(cell_width) for j in range(len(labels)))
        rows.append(str(true_label).ljust(label_width) + " | " + row_cells)
    return "\n".join(rows)


def interpretation_lines(
    config_id: str,
    keep_sensors: Sequence[str],
    full_overall: Mapping[str, pd.Series],
    clean_overall: Mapping[str, pd.Series],
    masked_overall: Mapping[str, pd.Series],
) -> List[str]:
    dropped_keys = [s for s in SENSOR_GROUPS if s not in keep_sensors]
    dropped_text = " + ".join(SENSOR_LABEL[s] for s in dropped_keys) if dropped_keys else "nothing"

    lines: List[str] = []
    drops_meaningful: List[str] = []
    drops_minor: List[str] = []
    gains: List[str] = []
    for task_id in [t["id"] for t in TASKS]:
        delta_clean = float(clean_overall[task_id]["macro_f1"]) - float(full_overall[task_id]["macro_f1"])
        if delta_clean <= -0.05:
            drops_meaningful.append(f"{task_id} (delta {delta_clean:+.3f})")
        elif delta_clean < -0.005:
            drops_minor.append(f"{task_id} (delta {delta_clean:+.3f})")
        elif delta_clean > 0.005:
            gains.append(f"{task_id} (delta {delta_clean:+.3f})")

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
            f"Removing {dropped_text} leaves macro F1 essentially unchanged across all four tasks."
        )

    # Compare clean vs masked: tell whether train-with-all + mask is better than
    # only training on kept features.
    clean_minus_masked = []
    for task_id in [t["id"] for t in TASKS]:
        delta = float(clean_overall[task_id]["macro_f1"]) - float(masked_overall[task_id]["macro_f1"])
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
    full_overall: Mapping[str, pd.Series],
    full_predictions: Mapping[str, Tuple[List[str], List[str]]],
    config_results: Sequence[Dict[str, object]],
    group_col: str,
    full_summary_rows: Sequence[Dict[str, object]],
) -> str:
    sections: List[str] = []
    sections.append("SUPERVISED SENSOR ABLATION REPORT")
    sections.append("=" * 80)
    sections.append("")
    sections.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"Model: Random Forest (Pipeline: SimpleImputer median + RandomForestClassifier).")
    sections.append(
        "Hierarchical task: Random Forest -> Random Forest (binary stage then subtype stage)."
    )
    sections.append(f"Cross-validation: Leave-One-Subject-Out on group column = {group_col}.")
    sections.append(f"Random state: {RANDOM_STATE}.")
    sections.append("")
    sections.append("Sensor groups:")
    sections.append(f"  - Posture (camera body): {len(POSTURE_FEATURES)} features -> {', '.join(POSTURE_FEATURES)}")
    sections.append(f"  - Blink   (camera face): {len(BLINK_FEATURES)} features -> {', '.join(BLINK_FEATURES)}")
    sections.append(f"  - Heart   (wearable)   : {len(HEART_FEATURES)} features -> {', '.join(HEART_FEATURES)}")
    sections.append("")
    sections.append("Scenarios:")
    sections.append(
        "  full  : train AND test with all 18 features. Reference baseline per task."
    )
    sections.append(
        "  clean : train AND test using only the features of the kept sensors. "
        "This is the realistic 'we just don't have this sensor' setup."
    )
    sections.append(
        "  masked: train with all 18 features, then set the excluded-sensor columns "
        "to NaN at test time so the training-fold SimpleImputer fills them with the "
        "training median. Deployment-mismatch stress test."
    )
    sections.append("")
    sections.append("FULL reference (all 18 features) per task:")
    full_rows = [
        [
            task["id"],
            fmt(float(full_overall[task["id"]]["macro_f1"])),
            fmt(float(full_overall[task["id"]]["balanced_accuracy"])),
            fmt(float(full_overall[task["id"]]["accuracy"])),
            fmt(float(full_overall[task["id"]]["weighted_f1"])),
        ]
        for task in TASKS
    ]
    sections.append(
        render_ascii_table(
            ["task", "macro_f1", "bal_acc", "accuracy", "weighted_f1"],
            full_rows,
        )
    )
    sections.append("")

    for entry in config_results:
        keep_sensors: Sequence[str] = entry["keep"]  # type: ignore[assignment]
        config_id: str = entry["id"]  # type: ignore[assignment]
        keep_features: Sequence[str] = entry["keep_features"]  # type: ignore[assignment]
        mask_features: Sequence[str] = entry["mask_features"]  # type: ignore[assignment]
        clean_overall: Mapping[str, pd.Series] = entry["clean_overall"]  # type: ignore[assignment]
        masked_overall: Mapping[str, pd.Series] = entry["masked_overall"]  # type: ignore[assignment]
        predictions: Mapping[str, Dict[str, Tuple[List[str], List[str]]]] = entry["predictions"]  # type: ignore[assignment]

        sections.append("=" * 80)
        sections.append(f"CONFIGURATION: {config_id}")
        sections.append(configuration_label(keep_sensors))
        sections.append(
            f"Kept features ({len(keep_features)}): " + ", ".join(keep_features)
        )
        if mask_features:
            sections.append(
                f"Excluded features ({len(mask_features)}): " + ", ".join(mask_features)
            )
        else:
            sections.append("Excluded features: none.")
        sections.append("=" * 80)
        sections.append("")

        for task in TASKS:
            task_id: str = task["id"]  # type: ignore[assignment]
            labels: Sequence[str] = task["labels"]  # type: ignore[assignment]
            full_row = full_overall[task_id]
            clean_row = clean_overall[task_id]
            masked_row = masked_overall[task_id]

            sections.append(f"-- TASK: {task_id} --")
            table_rows = [
                [
                    "FULL (18)",
                    fmt(float(full_row["macro_f1"])),
                    fmt(float(full_row["balanced_accuracy"])),
                    fmt(float(full_row["accuracy"])),
                    fmt(float(full_row["weighted_f1"])),
                    "—",
                    "—",
                ],
                [
                    f"CLEAN ({len(keep_features)})",
                    fmt(float(clean_row["macro_f1"])),
                    fmt(float(clean_row["balanced_accuracy"])),
                    fmt(float(clean_row["accuracy"])),
                    fmt(float(clean_row["weighted_f1"])),
                    fmt_signed(float(clean_row["macro_f1"]) - float(full_row["macro_f1"])),
                    fmt_signed(float(clean_row["balanced_accuracy"]) - float(full_row["balanced_accuracy"])),
                ],
                [
                    "MASKED (18)",
                    fmt(float(masked_row["macro_f1"])),
                    fmt(float(masked_row["balanced_accuracy"])),
                    fmt(float(masked_row["accuracy"])),
                    fmt(float(masked_row["weighted_f1"])),
                    fmt_signed(float(masked_row["macro_f1"]) - float(full_row["macro_f1"])),
                    fmt_signed(float(masked_row["balanced_accuracy"]) - float(full_row["balanced_accuracy"])),
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

            full_y_true, full_y_pred = full_predictions[task_id]
            clean_y_true, clean_y_pred = predictions["clean"][task_id]
            masked_y_true, masked_y_pred = predictions["masked"][task_id]

            sections.append("Confusion matrix - FULL (reference):")
            sections.append(render_confusion_matrix(full_y_true, full_y_pred, labels))
            sections.append("")
            sections.append("Confusion matrix - CLEAN:")
            sections.append(render_confusion_matrix(clean_y_true, clean_y_pred, labels))
            sections.append("")
            sections.append("Confusion matrix - MASKED:")
            sections.append(render_confusion_matrix(masked_y_true, masked_y_pred, labels))
            sections.append("")

        sections.append("Interpretation:")
        for line in interpretation_lines(
            config_id, keep_sensors, full_overall, clean_overall, masked_overall
        ):
            sections.append("  - " + line)
        sections.append("")

    sections.append("=" * 80)
    sections.append("OVERALL SUMMARY")
    sections.append("=" * 80)
    sections.append("")
    summary_rows = []
    for row in full_summary_rows:
        summary_rows.append([
            row["configuration"],
            row["task"],
            row["scenario"],
            str(row["n_train_features"]),
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
            summary_rows,
        )
    )
    sections.append("")
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    csv_dir = resolve_csv_dir(args.data_dir)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    mpl_config = results_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    raw_datasets = load_datasets(csv_dir)
    validate_features(raw_datasets)
    datasets = coerce_feature_columns(raw_datasets, FEATURE_COLUMNS)
    group_col = infer_group_column(datasets)

    binary_df = build_binary_dataset(datasets, group_col)
    subtype_df = build_subtype_dataset(datasets, group_col)
    final_df = build_hierarchical_test_dataset(datasets, group_col)
    direct_df = build_direct_3class_dataset(datasets, group_col)

    rf_pipeline = model_factories(RANDOM_STATE)["Random Forest"]

    # ----- FULL reference -------------------------------------------------
    full_dir = results_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_overall: Dict[str, pd.Series] = {}
    full_predictions: Dict[str, Tuple[List[str], List[str]]] = {}
    full_metric_rows: List[pd.DataFrame] = []
    for task in TASKS:
        task_id = str(task["id"])
        labels = list(task["labels"])  # type: ignore[arg-type]
        print(f"[FULL] task={task_id}")
        df_metrics, y_true_all, y_pred_all = evaluate_task(
            task_id,
            labels,
            rf_pipeline,
            FEATURE_COLUMNS,
            [],
            "full",
            group_col,
            binary_df, subtype_df, final_df, direct_df,
        )
        full_metric_rows.append(df_metrics)
        full_overall[task_id] = overall_row(df_metrics)
        full_predictions[task_id] = (y_true_all, y_pred_all)
        save_confusion_matrix_png(
            y_true_all,
            y_pred_all,
            labels,
            f"FULL Random Forest LOSO Confusion Matrix - {task_id}",
            full_dir / f"{task_id}_full_confusion_matrix.png",
        )
    pd.concat(full_metric_rows, ignore_index=True).to_csv(
        full_dir / "metrics.csv", index=False
    )

    # ----- Configurations --------------------------------------------------
    config_results: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for task in TASKS:
        task_id = str(task["id"])
        summary_rows.append({
            "configuration": "full",
            "task": task_id,
            "scenario": "full",
            "n_train_features": int(full_overall[task_id]["n_train_features"]),
            "macro_f1": float(full_overall[task_id]["macro_f1"]),
            "balanced_accuracy": float(full_overall[task_id]["balanced_accuracy"]),
            "accuracy": float(full_overall[task_id]["accuracy"]),
            "macro_f1_delta_vs_full": 0.0,
        })

    for config in CONFIGURATIONS:
        config_id = str(config["id"])
        keep_sensors = list(config["keep"])  # type: ignore[arg-type]
        keep_features = config_keep_features(keep_sensors)
        mask_features = config_mask_features(keep_sensors)

        config_dir = results_dir / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        clean_overall: Dict[str, pd.Series] = {}
        masked_overall: Dict[str, pd.Series] = {}
        predictions: Dict[str, Dict[str, Tuple[List[str], List[str]]]] = {
            "clean": {},
            "masked": {},
        }
        scenario_metric_rows: List[pd.DataFrame] = []

        for task in TASKS:
            task_id = str(task["id"])
            labels = list(task["labels"])  # type: ignore[arg-type]

            # --- clean ---
            print(f"[{config_id}] clean task={task_id} n_train_features={len(keep_features)}")
            df_metrics_c, yt_c, yp_c = evaluate_task(
                task_id,
                labels,
                rf_pipeline,
                keep_features,
                [],
                "clean",
                group_col,
                binary_df, subtype_df, final_df, direct_df,
            )
            scenario_metric_rows.append(df_metrics_c)
            clean_overall[task_id] = overall_row(df_metrics_c)
            predictions["clean"][task_id] = (yt_c, yp_c)
            save_confusion_matrix_png(
                yt_c,
                yp_c,
                labels,
                f"{config_id} CLEAN Random Forest LOSO - {task_id}",
                config_dir / f"{task_id}_clean_confusion_matrix.png",
            )

            # --- masked ---
            print(f"[{config_id}] masked task={task_id} mask={mask_features}")
            df_metrics_m, yt_m, yp_m = evaluate_task(
                task_id,
                labels,
                rf_pipeline,
                FEATURE_COLUMNS,
                mask_features,
                "masked",
                group_col,
                binary_df, subtype_df, final_df, direct_df,
            )
            scenario_metric_rows.append(df_metrics_m)
            masked_overall[task_id] = overall_row(df_metrics_m)
            predictions["masked"][task_id] = (yt_m, yp_m)
            save_confusion_matrix_png(
                yt_m,
                yp_m,
                labels,
                f"{config_id} MASKED Random Forest LOSO - {task_id}",
                config_dir / f"{task_id}_masked_confusion_matrix.png",
            )

            for scenario_name, overall in [("clean", clean_overall[task_id]), ("masked", masked_overall[task_id])]:
                summary_rows.append({
                    "configuration": config_id,
                    "task": task_id,
                    "scenario": scenario_name,
                    "n_train_features": int(overall["n_train_features"]),
                    "macro_f1": float(overall["macro_f1"]),
                    "balanced_accuracy": float(overall["balanced_accuracy"]),
                    "accuracy": float(overall["accuracy"]),
                    "macro_f1_delta_vs_full": float(overall["macro_f1"]) - float(full_overall[task_id]["macro_f1"]),
                })

        pd.concat(scenario_metric_rows, ignore_index=True).to_csv(
            config_dir / "metrics.csv", index=False
        )

        config_results.append({
            "id": config_id,
            "keep": keep_sensors,
            "keep_features": keep_features,
            "mask_features": mask_features,
            "clean_overall": clean_overall,
            "masked_overall": masked_overall,
            "predictions": predictions,
        })

    # ----- Report ----------------------------------------------------------
    report = build_report(
        full_overall,
        full_predictions,
        config_results,
        group_col,
        summary_rows,
    )
    report_path = results_dir / "sensor_ablation_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"\nSaved sensor ablation outputs to: {results_dir}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
