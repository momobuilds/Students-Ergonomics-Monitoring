#!/usr/bin/env python3
"""Supervised LOSO experiments without HR/RMMSD at inference.

Two variants are evaluated:
1. no_hr_rmmsd_train_test:
   Train and test using only posture + eye/blink features.
2. all_features_train_hr_rmmsd_masked_test:
   Train using all features, then mask HR/RMMSD to NaN at test time so the
   training-fold imputer supplies train medians. This simulates a deployment
   mismatch where the heart-rate sensor is missing at inference.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from wellbeing_hierarchical_classification import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    best_predictions,
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
    select_best_overall,
    validate_features,
)


HEART_FEATURES = ["HR", "RMMSD"]
NO_HEART_FEATURES = [
    feature for feature in FEATURE_COLUMNS if feature not in HEART_FEATURES
]

REFERENCE_MACRO_F1 = {
    "binary": 0.952279,
    "subtype": 1.000000,
    "hierarchical": 0.957293,
    "direct_3class": 0.951070,
}

REFERENCE_BALANCED_ACCURACY = {
    "binary": 0.958450,
    "subtype": 1.000000,
    "hierarchical": 0.957007,
    "direct_3class": 0.950855,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supervised LOSO experiments without HR/RMMSD."
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
        default=Path("results/no_hr_rmmsd"),
        help="Output directory for no-HR/RMMSD experiment results.",
    )
    return parser.parse_args()


def experiment_variants() -> List[Dict[str, object]]:
    return [
        {
            "variant": "no_hr_rmmsd_train_test",
            "description": "Train and test with posture + eye/blink features only.",
            "train_features": NO_HEART_FEATURES,
            "mask_test_features": [],
        },
        {
            "variant": "all_features_train_hr_rmmsd_masked_test",
            "description": (
                "Train with all features, but set HR/RMMSD to NaN in test folds "
                "so the training-fold imputer fills them."
            ),
            "train_features": FEATURE_COLUMNS,
            "mask_test_features": HEART_FEATURES,
        },
    ]


def mask_test_features(
    X_test: pd.DataFrame,
    features_to_mask: Sequence[str],
) -> pd.DataFrame:
    masked = X_test.copy()
    for feature in features_to_mask:
        if feature in masked.columns:
            masked.loc[:, feature] = np.nan
    return masked


def prediction_key(variant_name: str, model_name: str) -> str:
    return f"{variant_name}|||{model_name}"


def evaluate_flat_variant(
    task_name: str,
    df: pd.DataFrame,
    group_col: str,
    labels: Sequence[str],
    models: Mapping[str, object],
    variant: Mapping[str, object],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    features = list(variant["train_features"])
    masked_features = list(variant["mask_test_features"])
    variant_name = str(variant["variant"])
    logo = LeaveOneGroupOut()
    X = df.loc[:, features]
    y = df["target"]
    groups = df[group_col].astype(str)
    records = []
    predictions = {}

    for model_name, model in models.items():
        key = prediction_key(variant_name, model_name)
        predictions[key] = {"y_true": [], "y_pred": [], "groups": []}

        for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups=groups), start=1
        ):
            fitted = clone(model)
            X_train = X.iloc[train_idx]
            X_test = mask_test_features(X.iloc[test_idx], masked_features)
            y_train = y.iloc[train_idx]
            y_true = y.iloc[test_idx].to_numpy()

            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            heldout = ",".join(sorted(groups.iloc[test_idx].unique().tolist()))

            row = {
                "task": task_name,
                "variant": variant_name,
                "variant_description": variant["description"],
                "row_type": "fold",
                "fold": fold,
                "heldout_group": heldout,
                "model": model_name,
                "prediction_key": key,
                "n_train_features": len(features),
                "train_features": ",".join(features),
                "masked_test_features": ",".join(masked_features),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            row.update(compute_metrics(y_true, y_pred, labels))
            records.append(row)

            predictions[key]["y_true"].extend(y_true.tolist())
            predictions[key]["y_pred"].extend(y_pred.tolist())
            predictions[key]["groups"].extend(groups.iloc[test_idx].tolist())

        y_true_all = predictions[key]["y_true"]
        y_pred_all = predictions[key]["y_pred"]
        row = {
            "task": task_name,
            "variant": variant_name,
            "variant_description": variant["description"],
            "row_type": "overall",
            "fold": 0,
            "heldout_group": "ALL",
            "model": model_name,
            "prediction_key": key,
            "n_train_features": len(features),
            "train_features": ",".join(features),
            "masked_test_features": ",".join(masked_features),
            "train_rows": int(len(df)),
            "test_rows": int(len(df)),
        }
        row.update(compute_metrics(y_true_all, y_pred_all, labels))
        records.append(row)

    return pd.DataFrame(records), predictions


def evaluate_hierarchical_variant(
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    group_col: str,
    labels: Sequence[str],
    binary_models: Mapping[str, object],
    subtype_models: Mapping[str, object],
    variant: Mapping[str, object],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    features = list(variant["train_features"])
    masked_features = list(variant["mask_test_features"])
    variant_name = str(variant["variant"])
    group_values = sorted(final_df[group_col].dropna().astype(str).unique().tolist())
    records = []
    predictions = {}

    for binary_name, subtype_name in itertools.product(binary_models, subtype_models):
        combo_name = f"{binary_name} -> {subtype_name}"
        key = prediction_key(variant_name, combo_name)
        predictions[key] = {"y_true": [], "y_pred": [], "groups": []}

        for fold, heldout in enumerate(group_values, start=1):
            binary_train = binary_df[binary_df[group_col].astype(str) != heldout]
            subtype_train = subtype_df[subtype_df[group_col].astype(str) != heldout]
            test = final_df[final_df[group_col].astype(str) == heldout]

            binary_model = clone(binary_models[binary_name])
            subtype_model = clone(subtype_models[subtype_name])
            binary_model.fit(binary_train.loc[:, features], binary_train["target"])
            subtype_model.fit(subtype_train.loc[:, features], subtype_train["target"])

            X_test = mask_test_features(test.loc[:, features], masked_features)
            binary_pred = binary_model.predict(X_test)
            final_pred = np.array(
                ["normal" if pred == "normal" else "" for pred in binary_pred],
                dtype=object,
            )
            abnormal_mask = binary_pred == "abnormal"
            if abnormal_mask.any():
                final_pred[abnormal_mask] = subtype_model.predict(
                    X_test.loc[abnormal_mask]
                )

            y_true = test["target"].to_numpy()
            row = {
                "task": "hierarchical",
                "variant": variant_name,
                "variant_description": variant["description"],
                "row_type": "fold",
                "fold": fold,
                "heldout_group": heldout,
                "binary_model": binary_name,
                "subtype_model": subtype_name,
                "model": combo_name,
                "prediction_key": key,
                "n_train_features": len(features),
                "train_features": ",".join(features),
                "masked_test_features": ",".join(masked_features),
                "train_binary_rows": int(len(binary_train)),
                "train_subtype_rows": int(len(subtype_train)),
                "test_rows": int(len(test)),
            }
            row.update(compute_metrics(y_true, final_pred, labels))
            records.append(row)

            predictions[key]["y_true"].extend(y_true.tolist())
            predictions[key]["y_pred"].extend(final_pred.tolist())
            predictions[key]["groups"].extend(test[group_col].astype(str).tolist())

        y_true_all = predictions[key]["y_true"]
        y_pred_all = predictions[key]["y_pred"]
        row = {
            "task": "hierarchical",
            "variant": variant_name,
            "variant_description": variant["description"],
            "row_type": "overall",
            "fold": 0,
            "heldout_group": "ALL",
            "binary_model": binary_name,
            "subtype_model": subtype_name,
            "model": combo_name,
            "prediction_key": key,
            "n_train_features": len(features),
            "train_features": ",".join(features),
            "masked_test_features": ",".join(masked_features),
            "train_binary_rows": int(len(binary_df)),
            "train_subtype_rows": int(len(subtype_df)),
            "test_rows": int(len(final_df)),
        }
        row.update(compute_metrics(y_true_all, y_pred_all, labels))
        records.append(row)

    return pd.DataFrame(records), predictions


def select_best_for_variant(results: pd.DataFrame, variant_name: str) -> pd.Series:
    overall = results[
        (results["row_type"] == "overall") & (results["variant"] == variant_name)
    ].copy()
    return select_best_overall(overall)


def save_task_confusion_matrices(
    results_dir: Path,
    task_name: str,
    results: pd.DataFrame,
    predictions: Mapping[str, Dict[str, List[str]]],
    labels: Sequence[str],
) -> None:
    for variant in sorted(results["variant"].unique()):
        best = select_best_for_variant(results, variant)
        selected = predictions[str(best["prediction_key"])]
        save_confusion_matrix_png(
            selected["y_true"],
            selected["y_pred"],
            labels,
            f"{task_name} {variant}: {best['model']}",
            results_dir / f"{task_name}_{variant}_best_confusion_matrix.png",
        )


def build_summary_table(task_results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task_name, results in task_results.items():
        for variant in sorted(results["variant"].unique()):
            best = select_best_for_variant(results, variant)
            rows.append(
                {
                    "task": task_name,
                    "variant": variant,
                    "best_model": best["model"],
                    "n_train_features": best["n_train_features"],
                    "masked_test_features": best["masked_test_features"],
                    "accuracy": best["accuracy"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "macro_f1": best["macro_f1"],
                    "weighted_f1": best["weighted_f1"],
                    "precision_macro": best["precision_macro"],
                    "recall_macro": best["recall_macro"],
                    "reference_all_feature_macro_f1": REFERENCE_MACRO_F1[task_name],
                    "macro_f1_delta_vs_all_feature": best["macro_f1"]
                    - REFERENCE_MACRO_F1[task_name],
                    "reference_all_feature_balanced_accuracy": REFERENCE_BALANCED_ACCURACY[
                        task_name
                    ],
                    "balanced_accuracy_delta_vs_all_feature": best[
                        "balanced_accuracy"
                    ]
                    - REFERENCE_BALANCED_ACCURACY[task_name],
                }
            )
    return pd.DataFrame(rows)


def build_final_summary(summary: pd.DataFrame) -> str:
    lines = ["NO HR/RMMSD SUPERVISED LOSO EXPERIMENT SUMMARY", "=" * 80]
    lines.append(
        "Two scenarios were tested. The realistic no-heart-sensor scenario trains "
        "and tests using only posture + eye/blink features. The masked-test "
        "scenario trains with all features but sets HR/RMMSD to missing in each "
        "held-out test fold, so the training-fold imputer fills them."
    )
    lines.append("")
    lines.append("Best model per task and scenario:")
    lines.append(summary.to_string(index=False))
    lines.append("")

    for row in summary.itertuples(index=False):
        direction = "improved" if row.macro_f1_delta_vs_all_feature > 0 else "dropped"
        if abs(row.macro_f1_delta_vs_all_feature) < 0.005:
            direction = "changed only slightly"
        lines.append(
            f"{row.task} / {row.variant}: macro F1 {direction} from "
            f"{row.reference_all_feature_macro_f1:.3f} to {row.macro_f1:.3f} "
            f"(delta {row.macro_f1_delta_vs_all_feature:+.3f})."
        )

    clean = summary[summary["variant"] == "no_hr_rmmsd_train_test"].copy()
    masked = summary[
        summary["variant"] == "all_features_train_hr_rmmsd_masked_test"
    ].copy()
    lines.append("")
    lines.append(
        "Scientific recommendation: use the no_hr_rmmsd_train_test scenario for "
        "claims about a missing heart-rate sensor. Training with HR/RMMSD and "
        "then masking them at test time is a deployment-mismatch stress test, "
        "not the preferred production setup."
    )
    lines.append("")
    lines.append("Clean no-heart-sensor summary:")
    lines.append(clean.to_string(index=False))
    lines.append("")
    lines.append("Masked-test stress-test summary:")
    lines.append(masked.to_string(index=False))
    return "\n".join(lines)


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

    models = model_factories(RANDOM_STATE)
    binary_labels = ["normal", "abnormal"]
    subtype_labels = ["sleepy", "stress"]
    three_labels = ["normal", "sleepy", "stress"]

    task_results: Dict[str, pd.DataFrame] = {}
    task_predictions: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for variant in experiment_variants():
        print(f"Running variant: {variant['variant']}")
        binary_results, binary_predictions = evaluate_flat_variant(
            "binary",
            binary_df,
            group_col,
            binary_labels,
            models,
            variant,
        )
        subtype_results, subtype_predictions = evaluate_flat_variant(
            "subtype",
            subtype_df,
            group_col,
            subtype_labels,
            models,
            variant,
        )
        hierarchical_results, hierarchical_predictions = evaluate_hierarchical_variant(
            binary_df,
            subtype_df,
            final_df,
            group_col,
            three_labels,
            models,
            models,
            variant,
        )
        direct_results, direct_predictions = evaluate_flat_variant(
            "direct_3class",
            direct_df,
            group_col,
            three_labels,
            models,
            variant,
        )

        for task_name, results in [
            ("binary", binary_results),
            ("subtype", subtype_results),
            ("hierarchical", hierarchical_results),
            ("direct_3class", direct_results),
        ]:
            task_results.setdefault(task_name, pd.DataFrame())
            task_results[task_name] = pd.concat(
                [task_results[task_name], results],
                ignore_index=True,
                sort=False,
            )

        for task_name, predictions in [
            ("binary", binary_predictions),
            ("subtype", subtype_predictions),
            ("hierarchical", hierarchical_predictions),
            ("direct_3class", direct_predictions),
        ]:
            task_predictions.setdefault(task_name, {})
            task_predictions[task_name].update(predictions)

    filenames = {
        "binary": "binary_no_hr_rmmsd_results.csv",
        "subtype": "subtype_no_hr_rmmsd_results.csv",
        "hierarchical": "hierarchical_no_hr_rmmsd_results.csv",
        "direct_3class": "direct_3class_no_hr_rmmsd_results.csv",
    }
    for task_name, results in task_results.items():
        results.to_csv(results_dir / filenames[task_name], index=False)

    save_task_confusion_matrices(
        results_dir,
        "binary",
        task_results["binary"],
        task_predictions["binary"],
        binary_labels,
    )
    save_task_confusion_matrices(
        results_dir,
        "subtype",
        task_results["subtype"],
        task_predictions["subtype"],
        subtype_labels,
    )
    save_task_confusion_matrices(
        results_dir,
        "hierarchical",
        task_results["hierarchical"],
        task_predictions["hierarchical"],
        three_labels,
    )
    save_task_confusion_matrices(
        results_dir,
        "direct_3class",
        task_results["direct_3class"],
        task_predictions["direct_3class"],
        three_labels,
    )

    summary = build_summary_table(task_results)
    summary.to_csv(results_dir / "no_hr_rmmsd_summary.csv", index=False)

    final_summary = build_final_summary(summary)
    (results_dir / "final_no_hr_rmmsd_summary.txt").write_text(
        final_summary + "\n",
        encoding="utf-8",
    )
    print(final_summary)
    print(f"Saved no-HR/RMMSD outputs to: {results_dir}")


if __name__ == "__main__":
    main()
