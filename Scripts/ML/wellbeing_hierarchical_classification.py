#!/usr/bin/env python3
"""Leave-one-subject-out wellbeing classification pipeline.

This script evaluates:
1. Binary classification: normal vs abnormal.
2. Abnormal subtype classification: sleepy vs stress.
3. End-to-end hierarchical classification: normal, sleepy, stress.
4. Direct 3-class classification: normal vs sleepy vs stress.

The participant/source column is inferred from the CSV schemas, with
``video_source`` preferred when no explicit participant column is present.
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42

FILES = OrderedDict(
    [
        ("normal", "normal.csv"),
        ("anomalous", "anomalous.csv"),
        ("sleepy", "sleepy.csv"),
        ("stress", "stress.csv"),
    ]
)

FEATURE_COLUMNS = [
    "p_shoulder_mean",
    "p_shoulder_std",
    "p_shoulder_bad_pct",
    "p_head_tilt_mean",
    "p_head_tilt_std",
    "p_head_tilt_bad_pct",
    "p_fhp_mean",
    "p_fhp_std",
    "p_fhp_bad_pct",
    "b_ear_mean",
    "b_ear_std",
    "b_perclos",
    "b_blink_count",
    "b_blink_freq",
    "b_ibi_mean",
    "b_ibi_std",
    "HR",
    "RMMSD",
]

IDENTIFIER_HINTS = [
    "participant",
    "subject",
    "person",
    "video_source",
    "source",
    "session",
    "filename",
    "file",
    "name",
    "user",
]

GROUP_PRIORITY = [
    "participant",
    "subject",
    "person",
    "video_source",
    "source",
    "session",
    "filename",
    "file",
    "name",
    "user",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised wellbeing classification with leave-one-subject-out CV."
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        type=Path,
        help=(
            "Directory containing the four CSV files, or a project root containing "
            "data/supervised. Default: ."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=Path("results/supervised/baseline"),
        type=Path,
        help="Directory for CSV, PNG, and summary outputs. Default: results/supervised/baseline",
    )
    parser.add_argument(
        "--augmentation-results-dir",
        default=Path("results/supervised/augmentation"),
        type=Path,
        help=(
            "Directory for participant-aware augmentation experiment outputs. "
            "Default: results/supervised/augmentation"
        ),
    )
    return parser.parse_args()


def prepare_results_dirs(results_dir: Path) -> Dict[str, Path]:
    """Create and return organized output directories."""
    result_dirs = {
        "root": results_dir,
        "binary": results_dir / "binary",
        "subtype": results_dir / "subtype",
        "hierarchical": results_dir / "hierarchical",
        "direct_3class": results_dir / "direct_3class",
        "comparison": results_dir / "comparison",
        "summary": results_dir / "summary",
    }
    for path in result_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return result_dirs


def resolve_csv_dir(data_dir: Path) -> Path:
    """Resolve either a direct CSV directory or a project root."""
    data_dir = data_dir.expanduser().resolve()
    direct = all((data_dir / filename).exists() for filename in FILES.values())
    if direct:
        return data_dir

    supervised = data_dir / "data" / "supervised"
    nested = all((supervised / filename).exists() for filename in FILES.values())
    if nested:
        return supervised

    expected = ", ".join(FILES.values())
    raise FileNotFoundError(
        f"Could not find {expected} in {data_dir} or {supervised}."
    )


def load_datasets(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    datasets = {}
    for name, filename in FILES.items():
        datasets[name] = pd.read_csv(csv_dir / filename)
    return datasets


def coerce_feature_columns(
    datasets: Mapping[str, pd.DataFrame], features: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """Return copies with feature columns coerced to numeric values."""
    coerced = {}
    for name, df in datasets.items():
        converted = df.copy()
        for feature in features:
            converted[feature] = pd.to_numeric(converted[feature], errors="coerce")
        coerced[name] = converted
    return coerced


def validate_features(datasets: Mapping[str, pd.DataFrame]) -> List[str]:
    missing_by_file = {}
    for name, df in datasets.items():
        missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
        if missing:
            missing_by_file[name] = missing

    if not missing_by_file:
        return FEATURE_COLUMNS

    # Support the scientifically correct spelling if a future file uses it.
    can_use_rmssd = all(
        "RMMSD" in missing and "RMSSD" in datasets[name].columns
        for name, missing in missing_by_file.items()
    )
    if can_use_rmssd:
        return ["RMSSD" if feature == "RMMSD" else feature for feature in FEATURE_COLUMNS]

    details = "; ".join(
        f"{name}: {', '.join(missing)}" for name, missing in missing_by_file.items()
    )
    raise ValueError(f"Missing required feature columns: {details}")


def candidate_group_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for column in df.columns:
        lowered = column.lower()
        if any(hint in lowered for hint in IDENTIFIER_HINTS):
            candidates.append(column)
    return candidates


def inspect_group_candidates(
    datasets: Mapping[str, pd.DataFrame]
) -> Dict[str, Dict[str, List[object]]]:
    inspection: Dict[str, Dict[str, List[object]]] = {}
    for name, df in datasets.items():
        inspection[name] = {}
        for column in candidate_group_columns(df):
            values = sorted(df[column].dropna().astype(str).unique().tolist())
            inspection[name][column] = values
    return inspection


def _column_is_group_like(datasets: Mapping[str, pd.DataFrame], column: str) -> bool:
    values = []
    for df in datasets.values():
        if column not in df.columns:
            return False
        values.extend(df[column].dropna().astype(str).tolist())
    unique_values = set(values)
    if len(unique_values) < 2:
        return False
    # A group column should identify repeated sources, not individual rows.
    return len(unique_values) <= max(50, int(len(values) * 0.5))


def infer_group_column(datasets: Mapping[str, pd.DataFrame]) -> str:
    all_columns = set.intersection(*(set(df.columns) for df in datasets.values()))
    lower_to_actual = {column.lower(): column for column in all_columns}

    for preferred in GROUP_PRIORITY:
        if preferred in lower_to_actual:
            actual = lower_to_actual[preferred]
            if _column_is_group_like(datasets, actual):
                return actual

    candidate_columns = sorted(
        column
        for column in all_columns
        if any(hint in column.lower() for hint in IDENTIFIER_HINTS)
    )
    for column in candidate_columns:
        if _column_is_group_like(datasets, column):
            return column

    raise ValueError(
        "Could not infer a clean participant/source column. "
        "Please add or specify a participant, subject, video_source, source, "
        "session, or filename column."
    )


def model_factories(random_state: int = RANDOM_STATE) -> OrderedDict[str, Pipeline]:
    return OrderedDict(
        [
            (
                "Logistic Regression",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=5000,
                                class_weight="balanced",
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "SVM",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Random Forest",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=500,
                                random_state=random_state,
                                class_weight="balanced",
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Gradient Boosting",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "classifier",
                            GradientBoostingClassifier(random_state=random_state),
                        ),
                    ]
                ),
            ),
        ]
    )


def build_binary_dataset(
    datasets: Mapping[str, pd.DataFrame], group_col: str
) -> pd.DataFrame:
    normal = datasets["normal"].copy()
    normal["target"] = "normal"
    anomalous = datasets["anomalous"].copy()
    anomalous["target"] = "abnormal"
    combined = pd.concat([normal, anomalous], ignore_index=True)
    return combined.dropna(subset=[group_col]).reset_index(drop=True)


def build_subtype_dataset(
    datasets: Mapping[str, pd.DataFrame], group_col: str
) -> pd.DataFrame:
    sleepy = datasets["sleepy"].copy()
    sleepy["target"] = "sleepy"
    stress = datasets["stress"].copy()
    stress["target"] = "stress"
    combined = pd.concat([sleepy, stress], ignore_index=True)
    return combined.dropna(subset=[group_col]).reset_index(drop=True)


def build_hierarchical_test_dataset(
    datasets: Mapping[str, pd.DataFrame], group_col: str
) -> pd.DataFrame:
    normal = datasets["normal"].copy()
    normal["target"] = "normal"
    sleepy = datasets["sleepy"].copy()
    sleepy["target"] = "sleepy"
    stress = datasets["stress"].copy()
    stress["target"] = "stress"
    combined = pd.concat([normal, sleepy, stress], ignore_index=True)
    return combined.dropna(subset=[group_col]).reset_index(drop=True)


def build_direct_3class_dataset(
    datasets: Mapping[str, pd.DataFrame], group_col: str
) -> pd.DataFrame:
    """Build direct normal/sleepy/stress data without anomalous.csv duplicates."""
    return build_hierarchical_test_dataset(datasets, group_col)


def safe_label(label: str) -> str:
    return (
        str(label)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for idx, label in enumerate(labels):
        key = safe_label(label)
        metrics[f"precision_{key}"] = precision[idx]
        metrics[f"recall_{key}"] = recall[idx]
        metrics[f"f1_{key}"] = f1[idx]
        metrics[f"support_{key}"] = int(support[idx])
    return metrics


def select_best_overall(results: pd.DataFrame) -> pd.Series:
    overall = results[results["row_type"] == "overall"].copy()
    return (
        overall.sort_values(
            ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
            ascending=False,
        )
        .iloc[0]
        .copy()
    )


def evaluate_flat_task(
    task_name: str,
    df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    labels: Sequence[str],
    models: Mapping[str, Pipeline],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, List[str]]]]:
    logo = LeaveOneGroupOut()
    records = []
    predictions = {
        model_name: {"y_true": [], "y_pred": [], "groups": []} for model_name in models
    }

    X = df.loc[:, features]
    y = df["target"]
    groups = df[group_col].astype(str)

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups=groups), start=1
        ):
            heldout_values = sorted(groups.iloc[test_idx].unique().tolist())
            heldout = ",".join(heldout_values)

            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = fitted.predict(X.iloc[test_idx])
            y_true = y.iloc[test_idx].to_numpy()

            row = {
                "task": task_name,
                "row_type": "fold",
                "fold": fold,
                "heldout_group": heldout,
                "model": model_name,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            row.update(compute_metrics(y_true, y_pred, labels))
            records.append(row)

            predictions[model_name]["y_true"].extend(y_true.tolist())
            predictions[model_name]["y_pred"].extend(y_pred.tolist())
            predictions[model_name]["groups"].extend(groups.iloc[test_idx].tolist())

        y_true_all = predictions[model_name]["y_true"]
        y_pred_all = predictions[model_name]["y_pred"]
        row = {
            "task": task_name,
            "row_type": "overall",
            "fold": 0,
            "heldout_group": "ALL",
            "model": model_name,
            "train_rows": int(len(df)),
            "test_rows": int(len(df)),
        }
        row.update(compute_metrics(y_true_all, y_pred_all, labels))
        records.append(row)

    results = pd.DataFrame(records)
    best = select_best_overall(results)
    return results, best, predictions


def direct_3class_classification(
    direct_df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    models: Mapping[str, Pipeline],
    labels: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, List[str]]]]:
    """Evaluate direct normal/sleepy/stress models with LOSO CV."""
    return evaluate_flat_task(
        "direct_3class",
        direct_df,
        features,
        group_col,
        labels,
        models,
    )


def evaluate_hierarchical_task(
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    binary_models: Mapping[str, Pipeline],
    subtype_models: Mapping[str, Pipeline],
    labels: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, List[str]]]]:
    records = []
    predictions: Dict[str, Dict[str, List[str]]] = {}
    group_values = sorted(final_df[group_col].dropna().astype(str).unique().tolist())

    for binary_name, subtype_name in itertools.product(binary_models, subtype_models):
        combo_name = f"{binary_name} -> {subtype_name}"
        predictions[combo_name] = {"y_true": [], "y_pred": [], "groups": []}

        for fold, heldout in enumerate(group_values, start=1):
            binary_train = binary_df[binary_df[group_col].astype(str) != heldout]
            subtype_train = subtype_df[subtype_df[group_col].astype(str) != heldout]
            test = final_df[final_df[group_col].astype(str) == heldout]

            binary_model = clone(binary_models[binary_name])
            subtype_model = clone(subtype_models[subtype_name])

            binary_model.fit(binary_train.loc[:, features], binary_train["target"])
            subtype_model.fit(subtype_train.loc[:, features], subtype_train["target"])

            binary_pred = binary_model.predict(test.loc[:, features])
            final_pred = np.array(
                ["normal" if pred == "normal" else "" for pred in binary_pred],
                dtype=object,
            )
            abnormal_mask = binary_pred == "abnormal"
            if abnormal_mask.any():
                final_pred[abnormal_mask] = subtype_model.predict(
                    test.loc[abnormal_mask, features]
                )

            y_true = test["target"].to_numpy()
            row = {
                "task": "hierarchical",
                "row_type": "fold",
                "fold": fold,
                "heldout_group": heldout,
                "binary_model": binary_name,
                "subtype_model": subtype_name,
                "model": combo_name,
                "train_binary_rows": int(len(binary_train)),
                "train_subtype_rows": int(len(subtype_train)),
                "test_rows": int(len(test)),
            }
            row.update(compute_metrics(y_true, final_pred, labels))
            records.append(row)

            predictions[combo_name]["y_true"].extend(y_true.tolist())
            predictions[combo_name]["y_pred"].extend(final_pred.tolist())
            predictions[combo_name]["groups"].extend(test[group_col].astype(str).tolist())

        y_true_all = predictions[combo_name]["y_true"]
        y_pred_all = predictions[combo_name]["y_pred"]
        row = {
            "task": "hierarchical",
            "row_type": "overall",
            "fold": 0,
            "heldout_group": "ALL",
            "binary_model": binary_name,
            "subtype_model": subtype_name,
            "model": combo_name,
            "train_binary_rows": int(len(binary_df)),
            "train_subtype_rows": int(len(subtype_df)),
            "test_rows": int(len(final_df)),
        }
        row.update(compute_metrics(y_true_all, y_pred_all, labels))
        records.append(row)

    results = pd.DataFrame(records)
    best = select_best_overall(results)
    return results, best, predictions


def augmentation_experiments() -> List[Dict[str, object]]:
    """Return the baseline and groupwise Gaussian jitter settings to evaluate."""
    return [
        {
            "augmentation_name": "baseline_no_augmentation",
            "augmentation_factor": 0,
            "noise_level": 0.0,
            "is_baseline": True,
        },
        {
            "augmentation_name": "groupwise_jitter_factor1_noise0.01",
            "augmentation_factor": 1,
            "noise_level": 0.01,
            "is_baseline": False,
        },
        {
            "augmentation_name": "groupwise_jitter_factor1_noise0.02",
            "augmentation_factor": 1,
            "noise_level": 0.02,
            "is_baseline": False,
        },
        {
            "augmentation_name": "groupwise_jitter_factor1_noise0.05",
            "augmentation_factor": 1,
            "noise_level": 0.05,
            "is_baseline": False,
        },
        {
            "augmentation_name": "groupwise_jitter_factor2_noise0.02",
            "augmentation_factor": 2,
            "noise_level": 0.02,
            "is_baseline": False,
        },
    ]


def augmentation_prediction_key(augmentation_name: str, model_name: str) -> str:
    return f"{augmentation_name}|||{model_name}"


def augment_training_data_groupwise(
    X_train,
    y_train,
    groups_train,
    feature_columns,
    augmentation_factor=1,
    noise_level=0.02,
    random_state=42,
):
    """
    Create augmented samples only within each participant and class.
    Never mix participants.
    Never use test data.
    Return augmented X, y, groups.
    """
    X_original = X_train.loc[:, feature_columns].copy().reset_index(drop=True)
    y_original = pd.Series(y_train, name="_target").reset_index(drop=True)
    groups_original = pd.Series(groups_train, name="_group").astype(str).reset_index(
        drop=True
    )

    X_with_flags = X_original.copy()
    X_with_flags["is_augmented"] = False
    if augmentation_factor <= 0 or noise_level <= 0:
        return X_with_flags, y_original, groups_original

    training_frame = X_original.copy()
    training_frame["_target"] = y_original
    training_frame["_group"] = groups_original

    lower_bounds = X_original.quantile(0.01).fillna(X_original.min())
    upper_bounds = X_original.quantile(0.99).fillna(X_original.max())

    rng = np.random.default_rng(random_state)
    X_parts = [X_with_flags]
    y_parts = [y_original]
    group_parts = [groups_original]

    for group_value, group_frame in training_frame.groupby("_group", sort=True):
        for class_value, subset in group_frame.groupby("_target", sort=True):
            subset_features = subset.loc[:, feature_columns].reset_index(drop=True)
            if subset_features.empty:
                continue
            feature_std = subset_features.std(axis=0, ddof=0).fillna(0.0)
            noise_scale = (noise_level * feature_std).to_numpy(dtype=float)

            for _ in range(augmentation_factor):
                noise = rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=subset_features.shape,
                )
                noisy_features = pd.DataFrame(
                    subset_features.to_numpy(dtype=float) + noise,
                    columns=feature_columns,
                )
                noisy_features = noisy_features.clip(
                    lower=lower_bounds,
                    upper=upper_bounds,
                    axis=1,
                )
                noisy_features["is_augmented"] = True
                X_parts.append(noisy_features)
                y_parts.append(
                    pd.Series([class_value] * len(noisy_features), name="_target")
                )
                group_parts.append(
                    pd.Series([group_value] * len(noisy_features), name="_group")
                )

    X_augmented = pd.concat(X_parts, ignore_index=True)
    y_augmented = pd.concat(y_parts, ignore_index=True)
    groups_augmented = pd.concat(group_parts, ignore_index=True).astype(str)
    return X_augmented, y_augmented, groups_augmented


def training_data_for_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    features: Sequence[str],
    experiment: Mapping[str, object],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if bool(experiment["is_baseline"]):
        X_fit = X_train.loc[:, features].copy().reset_index(drop=True)
        X_fit["is_augmented"] = False
        return (
            X_fit,
            pd.Series(y_train, name="_target").reset_index(drop=True),
            pd.Series(groups_train, name="_group").astype(str).reset_index(drop=True),
        )

    return augment_training_data_groupwise(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        feature_columns=features,
        augmentation_factor=int(experiment["augmentation_factor"]),
        noise_level=float(experiment["noise_level"]),
        random_state=random_state,
    )


def select_best_result(results: pd.DataFrame) -> pd.Series:
    return (
        results.sort_values(
            ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
            ascending=False,
        )
        .iloc[0]
        .copy()
    )


def evaluate_flat_task_with_augmentation(
    task_name: str,
    df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    labels: Sequence[str],
    models: Mapping[str, Pipeline],
    experiments: Sequence[Mapping[str, object]],
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    logo = LeaveOneGroupOut()
    records = []
    predictions: Dict[str, Dict[str, List[str]]] = {}

    X = df.loc[:, features]
    y = df["target"]
    groups = df[group_col].astype(str)

    for experiment_idx, experiment in enumerate(experiments):
        augmentation_name = str(experiment["augmentation_name"])
        for model_name, model in models.items():
            prediction_key = augmentation_prediction_key(augmentation_name, model_name)
            predictions[prediction_key] = {"y_true": [], "y_pred": [], "groups": []}

            for fold, (train_idx, test_idx) in enumerate(
                logo.split(X, y, groups=groups), start=1
            ):
                heldout_values = sorted(groups.iloc[test_idx].unique().tolist())
                heldout = ",".join(heldout_values)
                fold_seed = random_state + (experiment_idx * 10000) + (fold * 100)

                X_fit, y_fit, _ = training_data_for_experiment(
                    X_train=X.iloc[train_idx],
                    y_train=y.iloc[train_idx],
                    groups_train=groups.iloc[train_idx],
                    features=features,
                    experiment=experiment,
                    random_state=fold_seed,
                )

                fitted = clone(model)
                fitted.fit(X_fit.loc[:, features], y_fit)
                y_pred = fitted.predict(X.iloc[test_idx])
                y_true = y.iloc[test_idx].to_numpy()

                row = {
                    "task": task_name,
                    "row_type": "fold",
                    "fold": fold,
                    "heldout_group": heldout,
                    "model": model_name,
                    "prediction_key": prediction_key,
                    "augmentation_name": augmentation_name,
                    "augmentation_factor": int(experiment["augmentation_factor"]),
                    "noise_level": float(experiment["noise_level"]),
                    "is_baseline": bool(experiment["is_baseline"]),
                    "train_original_rows": int(len(train_idx)),
                    "train_augmented_rows": int(X_fit["is_augmented"].sum()),
                    "train_rows": int(len(X_fit)),
                    "test_rows": int(len(test_idx)),
                }
                row.update(compute_metrics(y_true, y_pred, labels))
                records.append(row)

                predictions[prediction_key]["y_true"].extend(y_true.tolist())
                predictions[prediction_key]["y_pred"].extend(y_pred.tolist())
                predictions[prediction_key]["groups"].extend(
                    groups.iloc[test_idx].tolist()
                )

            y_true_all = predictions[prediction_key]["y_true"]
            y_pred_all = predictions[prediction_key]["y_pred"]
            row = {
                "task": task_name,
                "row_type": "overall",
                "fold": 0,
                "heldout_group": "ALL",
                "model": model_name,
                "prediction_key": prediction_key,
                "augmentation_name": augmentation_name,
                "augmentation_factor": int(experiment["augmentation_factor"]),
                "noise_level": float(experiment["noise_level"]),
                "is_baseline": bool(experiment["is_baseline"]),
                "train_original_rows": int(len(df)),
                "train_augmented_rows": int(
                    len(df) * int(experiment["augmentation_factor"])
                ),
                "train_rows": int(
                    len(df) * (1 + int(experiment["augmentation_factor"]))
                ),
                "test_rows": int(len(df)),
            }
            row.update(compute_metrics(y_true_all, y_pred_all, labels))
            records.append(row)

    return pd.DataFrame(records), predictions


def evaluate_hierarchical_task_with_augmentation(
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    binary_models: Mapping[str, Pipeline],
    subtype_models: Mapping[str, Pipeline],
    labels: Sequence[str],
    experiments: Sequence[Mapping[str, object]],
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    records = []
    predictions: Dict[str, Dict[str, List[str]]] = {}
    group_values = sorted(final_df[group_col].dropna().astype(str).unique().tolist())

    for experiment_idx, experiment in enumerate(experiments):
        augmentation_name = str(experiment["augmentation_name"])
        for binary_name, subtype_name in itertools.product(binary_models, subtype_models):
            combo_name = f"{binary_name} -> {subtype_name}"
            prediction_key = augmentation_prediction_key(augmentation_name, combo_name)
            predictions[prediction_key] = {"y_true": [], "y_pred": [], "groups": []}

            for fold, heldout in enumerate(group_values, start=1):
                binary_train = binary_df[binary_df[group_col].astype(str) != heldout]
                subtype_train = subtype_df[subtype_df[group_col].astype(str) != heldout]
                test = final_df[final_df[group_col].astype(str) == heldout]

                binary_seed = (
                    random_state + (experiment_idx * 10000) + (fold * 100) + 1
                )
                subtype_seed = (
                    random_state + (experiment_idx * 10000) + (fold * 100) + 2
                )

                X_binary_fit, y_binary_fit, _ = training_data_for_experiment(
                    X_train=binary_train.loc[:, features],
                    y_train=binary_train["target"],
                    groups_train=binary_train[group_col],
                    features=features,
                    experiment=experiment,
                    random_state=binary_seed,
                )
                X_subtype_fit, y_subtype_fit, _ = training_data_for_experiment(
                    X_train=subtype_train.loc[:, features],
                    y_train=subtype_train["target"],
                    groups_train=subtype_train[group_col],
                    features=features,
                    experiment=experiment,
                    random_state=subtype_seed,
                )

                binary_model = clone(binary_models[binary_name])
                subtype_model = clone(subtype_models[subtype_name])

                binary_model.fit(X_binary_fit.loc[:, features], y_binary_fit)
                subtype_model.fit(X_subtype_fit.loc[:, features], y_subtype_fit)

                binary_pred = binary_model.predict(test.loc[:, features])
                final_pred = np.array(
                    ["normal" if pred == "normal" else "" for pred in binary_pred],
                    dtype=object,
                )
                abnormal_mask = binary_pred == "abnormal"
                if abnormal_mask.any():
                    final_pred[abnormal_mask] = subtype_model.predict(
                        test.loc[abnormal_mask, features]
                    )

                y_true = test["target"].to_numpy()
                row = {
                    "task": "hierarchical",
                    "row_type": "fold",
                    "fold": fold,
                    "heldout_group": heldout,
                    "binary_model": binary_name,
                    "subtype_model": subtype_name,
                    "model": combo_name,
                    "prediction_key": prediction_key,
                    "augmentation_name": augmentation_name,
                    "augmentation_factor": int(experiment["augmentation_factor"]),
                    "noise_level": float(experiment["noise_level"]),
                    "is_baseline": bool(experiment["is_baseline"]),
                    "train_binary_original_rows": int(len(binary_train)),
                    "train_binary_augmented_rows": int(
                        X_binary_fit["is_augmented"].sum()
                    ),
                    "train_binary_rows": int(len(X_binary_fit)),
                    "train_subtype_original_rows": int(len(subtype_train)),
                    "train_subtype_augmented_rows": int(
                        X_subtype_fit["is_augmented"].sum()
                    ),
                    "train_subtype_rows": int(len(X_subtype_fit)),
                    "test_rows": int(len(test)),
                }
                row.update(compute_metrics(y_true, final_pred, labels))
                records.append(row)

                predictions[prediction_key]["y_true"].extend(y_true.tolist())
                predictions[prediction_key]["y_pred"].extend(final_pred.tolist())
                predictions[prediction_key]["groups"].extend(
                    test[group_col].astype(str).tolist()
                )

            y_true_all = predictions[prediction_key]["y_true"]
            y_pred_all = predictions[prediction_key]["y_pred"]
            row = {
                "task": "hierarchical",
                "row_type": "overall",
                "fold": 0,
                "heldout_group": "ALL",
                "binary_model": binary_name,
                "subtype_model": subtype_name,
                "model": combo_name,
                "prediction_key": prediction_key,
                "augmentation_name": augmentation_name,
                "augmentation_factor": int(experiment["augmentation_factor"]),
                "noise_level": float(experiment["noise_level"]),
                "is_baseline": bool(experiment["is_baseline"]),
                "train_binary_original_rows": int(len(binary_df)),
                "train_binary_augmented_rows": int(
                    len(binary_df) * int(experiment["augmentation_factor"])
                ),
                "train_binary_rows": int(
                    len(binary_df) * (1 + int(experiment["augmentation_factor"]))
                ),
                "train_subtype_original_rows": int(len(subtype_df)),
                "train_subtype_augmented_rows": int(
                    len(subtype_df) * int(experiment["augmentation_factor"])
                ),
                "train_subtype_rows": int(
                    len(subtype_df) * (1 + int(experiment["augmentation_factor"]))
                ),
                "test_rows": int(len(final_df)),
            }
            row.update(compute_metrics(y_true_all, y_pred_all, labels))
            records.append(row)

    return pd.DataFrame(records), predictions


def best_overall_for_experiment(results: pd.DataFrame, augmentation_name: str) -> pd.Series:
    subset = results[
        (results["row_type"] == "overall")
        & (results["augmentation_name"] == augmentation_name)
    ]
    return select_best_result(subset)


def best_augmented_overall(results: pd.DataFrame) -> pd.Series:
    subset = results[(results["row_type"] == "overall") & (~results["is_baseline"])]
    return select_best_result(subset)


def prediction_for_best(
    best: pd.Series, predictions: Mapping[str, Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    return predictions[str(best["prediction_key"])]


def confusion_count(
    predictions: Mapping[str, List[str]],
    labels: Sequence[str],
    true_label: str,
    predicted_label: str,
) -> int:
    cm = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=labels)
    return int(cm[labels.index(true_label), labels.index(predicted_label)])


def best_fold_metric(
    results: pd.DataFrame,
    best: pd.Series,
    heldout_group: str,
    metric: str = "macro_f1",
) -> Optional[float]:
    subset = results[
        (results["row_type"] == "fold")
        & (results["prediction_key"] == best["prediction_key"])
        & (results["heldout_group"] == heldout_group)
    ]
    if subset.empty:
        return None
    return float(subset.iloc[0][metric])


def metric_change_sentence(
    task_name: str,
    baseline_value: float,
    augmented_value: float,
    metric_name: str = "macro F1",
    threshold: float = 0.005,
) -> str:
    delta = augmented_value - baseline_value
    if delta > threshold:
        direction = "improved"
    elif delta > 0:
        direction = "improved slightly"
    elif delta < -threshold:
        direction = "got worse"
    elif delta < 0:
        direction = "got slightly worse"
    else:
        direction = "did not change"
    return (
        f"{task_name} {direction}: {metric_name} "
        f"{baseline_value:.3f} -> {augmented_value:.3f} (delta {delta:+.3f})."
    )


def class_f1_regressions(
    baseline: pd.Series, augmented: pd.Series, labels: Sequence[str]
) -> List[str]:
    regressions = []
    for label in labels:
        key = f"f1_{safe_label(label)}"
        if key in baseline and key in augmented:
            delta = float(augmented[key]) - float(baseline[key])
            if delta < 0:
                regressions.append(
                    f"{label}: {float(baseline[key]):.3f} -> "
                    f"{float(augmented[key]):.3f} (delta {delta:+.3f})"
                )
    return regressions


def build_augmentation_summary_table(
    task_results: Mapping[str, pd.DataFrame],
    experiments: Sequence[Mapping[str, object]],
    reference_baselines: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for task_name, results in task_results.items():
        for experiment in experiments:
            augmentation_name = str(experiment["augmentation_name"])
            best = best_overall_for_experiment(results, augmentation_name)
            reference = reference_baselines.get(task_name)
            rows.append(
                {
                    "task": task_name,
                    "augmentation_name": augmentation_name,
                    "augmentation_factor": int(experiment["augmentation_factor"]),
                    "noise_level": float(experiment["noise_level"]),
                    "is_baseline": bool(experiment["is_baseline"]),
                    "best_model": best["model"],
                    "accuracy": best["accuracy"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "macro_f1": best["macro_f1"],
                    "weighted_f1": best["weighted_f1"],
                    "precision_macro": best["precision_macro"],
                    "recall_macro": best["recall_macro"],
                    "reference_baseline_macro_f1": reference,
                    "macro_f1_delta_vs_reference": (
                        None if reference is None else best["macro_f1"] - reference
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_best_augmentation_confusion_matrices(
    result_dir: Path,
    task_results: Mapping[str, pd.DataFrame],
    task_predictions: Mapping[str, Mapping[str, Dict[str, List[str]]]],
    task_labels: Mapping[str, Sequence[str]],
) -> None:
    for task_name in ["binary", "subtype", "hierarchical", "direct_3class"]:
        results = task_results[task_name]
        labels = task_labels[task_name]
        predictions = task_predictions[task_name]
        baseline_best = best_overall_for_experiment(results, "baseline_no_augmentation")
        augmented_best = best_augmented_overall(results)

        baseline_selected = prediction_for_best(baseline_best, predictions)
        augmented_selected = prediction_for_best(augmented_best, predictions)

        task_dir = result_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_png(
            baseline_selected["y_true"],
            baseline_selected["y_pred"],
            labels,
            f"{task_name} Baseline LOSO Confusion Matrix: {baseline_best['model']}",
            task_dir / f"{task_name}_baseline_confusion_matrix.png",
        )
        save_confusion_matrix_png(
            augmented_selected["y_true"],
            augmented_selected["y_pred"],
            labels,
            (
                f"{task_name} Best Augmented LOSO Confusion Matrix: "
                f"{augmented_best['model']} ({augmented_best['augmentation_name']})"
            ),
            task_dir / f"{task_name}_best_augmented_confusion_matrix.png",
        )


def build_final_augmentation_summary(
    task_results: Mapping[str, pd.DataFrame],
    task_predictions: Mapping[str, Mapping[str, Dict[str, List[str]]]],
    task_labels: Mapping[str, Sequence[str]],
    summary_table: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("DATA AUGMENTATION EXPERIMENT SUMMARY")
    lines.append("=" * 80)
    lines.append(
        "Augmentation was applied only inside each LOSO training fold. "
        "Synthetic rows were generated separately for each video_source/class group "
        "using Gaussian jitter, and held-out participant rows were never augmented."
    )
    lines.append("")
    lines.append("Best setting per task:")
    best_rows = []
    for task_name, results in task_results.items():
        baseline_best = best_overall_for_experiment(results, "baseline_no_augmentation")
        augmented_best = best_augmented_overall(results)
        best_rows.append(
            {
                "task": task_name,
                "baseline_model": baseline_best["model"],
                "baseline_macro_f1": baseline_best["macro_f1"],
                "best_augmented_model": augmented_best["model"],
                "best_augmented_setting": augmented_best["augmentation_name"],
                "best_augmented_macro_f1": augmented_best["macro_f1"],
                "delta": augmented_best["macro_f1"] - baseline_best["macro_f1"],
            }
        )
    lines.append(format_table(pd.DataFrame(best_rows)))
    lines.append("")

    binary_baseline = best_overall_for_experiment(
        task_results["binary"], "baseline_no_augmentation"
    )
    binary_augmented = best_augmented_overall(task_results["binary"])
    hierarchical_baseline = best_overall_for_experiment(
        task_results["hierarchical"], "baseline_no_augmentation"
    )
    hierarchical_augmented = best_augmented_overall(task_results["hierarchical"])
    direct_baseline = best_overall_for_experiment(
        task_results["direct_3class"], "baseline_no_augmentation"
    )
    direct_augmented = best_augmented_overall(task_results["direct_3class"])

    lines.append("Interpretation:")
    lines.append(
        "- "
        + metric_change_sentence(
            "Binary classification",
            float(binary_baseline["macro_f1"]),
            float(binary_augmented["macro_f1"]),
        )
    )
    lines.append(
        "- "
        + metric_change_sentence(
            "Hierarchical classification",
            float(hierarchical_baseline["macro_f1"]),
            float(hierarchical_augmented["macro_f1"]),
        )
    )
    lines.append(
        "- "
        + metric_change_sentence(
            "Direct 3-class classification",
            float(direct_baseline["macro_f1"]),
            float(direct_augmented["macro_f1"]),
        )
    )

    mo_lines = []
    for task_name in ["binary", "hierarchical", "direct_3class"]:
        baseline_best = best_overall_for_experiment(
            task_results[task_name], "baseline_no_augmentation"
        )
        augmented_best = best_augmented_overall(task_results[task_name])
        baseline_mo = best_fold_metric(task_results[task_name], baseline_best, "Mo")
        augmented_mo = best_fold_metric(task_results[task_name], augmented_best, "Mo")
        if baseline_mo is not None and augmented_mo is not None:
            mo_lines.append(
                f"{task_name}: {baseline_mo:.3f} -> {augmented_mo:.3f} "
                f"(delta {augmented_mo - baseline_mo:+.3f})"
            )
    lines.append(
        "- Weakest participant check, Mo macro F1: " + "; ".join(mo_lines) + "."
    )

    binary_baseline_predictions = prediction_for_best(
        binary_baseline, task_predictions["binary"]
    )
    binary_augmented_predictions = prediction_for_best(
        binary_augmented, task_predictions["binary"]
    )
    abnormal_to_normal_baseline = confusion_count(
        binary_baseline_predictions, task_labels["binary"], "abnormal", "normal"
    )
    abnormal_to_normal_augmented = confusion_count(
        binary_augmented_predictions, task_labels["binary"], "abnormal", "normal"
    )
    lines.append(
        "- Abnormal -> normal errors in binary classification: "
        f"{abnormal_to_normal_baseline} -> {abnormal_to_normal_augmented}."
    )

    hierarchical_baseline_predictions = prediction_for_best(
        hierarchical_baseline, task_predictions["hierarchical"]
    )
    hierarchical_augmented_predictions = prediction_for_best(
        hierarchical_augmented, task_predictions["hierarchical"]
    )
    direct_baseline_predictions = prediction_for_best(
        direct_baseline, task_predictions["direct_3class"]
    )
    direct_augmented_predictions = prediction_for_best(
        direct_augmented, task_predictions["direct_3class"]
    )
    hierarchical_stress_to_normal_baseline = confusion_count(
        hierarchical_baseline_predictions,
        task_labels["hierarchical"],
        "stress",
        "normal",
    )
    hierarchical_stress_to_normal_augmented = confusion_count(
        hierarchical_augmented_predictions,
        task_labels["hierarchical"],
        "stress",
        "normal",
    )
    direct_stress_to_normal_baseline = confusion_count(
        direct_baseline_predictions,
        task_labels["direct_3class"],
        "stress",
        "normal",
    )
    direct_stress_to_normal_augmented = confusion_count(
        direct_augmented_predictions,
        task_labels["direct_3class"],
        "stress",
        "normal",
    )
    lines.append(
        "- Stress -> normal errors: hierarchical "
        f"{hierarchical_stress_to_normal_baseline} -> "
        f"{hierarchical_stress_to_normal_augmented}; direct 3-class "
        f"{direct_stress_to_normal_baseline} -> {direct_stress_to_normal_augmented}."
    )

    regressions = []
    for task_name, labels in task_labels.items():
        baseline_best = best_overall_for_experiment(
            task_results[task_name], "baseline_no_augmentation"
        )
        augmented_best = best_augmented_overall(task_results[task_name])
        task_regressions = class_f1_regressions(
            baseline_best, augmented_best, labels
        )
        if task_regressions:
            regressions.append(f"{task_name}: " + "; ".join(task_regressions))
    if regressions:
        lines.append("- Class-level F1 got worse for " + " | ".join(regressions) + ".")
    else:
        lines.append("- No selected best augmented setting reduced class-level F1.")

    augmented_only = summary_table[~summary_table["is_baseline"]].copy()
    best_setting = augmented_only.sort_values(
        ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    lines.append(
        "- Best augmented setting overall by macro F1 was "
        f"{best_setting['augmentation_name']} on {best_setting['task']} "
        f"with {best_setting['best_model']} "
        f"(macro F1={best_setting['macro_f1']:.3f})."
    )

    meaningful_improvements = [
        row for row in best_rows if float(row["delta"]) > 0.005
    ]
    slight_improvements = [
        row for row in best_rows if 0 < float(row["delta"]) <= 0.005
    ]
    if meaningful_improvements:
        keep_text = (
            "Keep augmentation as an experimental comparison, but report it "
            "carefully because there are only 4 participants."
        )
    elif slight_improvements:
        keep_text = (
            "Augmentation improves only slightly; keep it as a robustness "
            "experiment, not as a strong modeling claim."
        )
    else:
        keep_text = (
            "Remove augmentation from the main result and keep the no-augmentation "
            "LOSO baseline as the primary model."
        )
    lines.append("- Recommendation: " + keep_text)
    lines.append("")
    lines.append("Full best-per-experiment comparison:")
    lines.append(format_table(summary_table))
    return "\n".join(lines)


def run_data_augmentation_experiments(
    result_dir: Path,
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    features: Sequence[str],
    group_col: str,
    models: Mapping[str, Pipeline],
    binary_labels: Sequence[str],
    subtype_labels: Sequence[str],
    hierarchical_labels: Sequence[str],
    direct_labels: Sequence[str],
) -> str:
    result_dir.mkdir(parents=True, exist_ok=True)
    experiments = augmentation_experiments()

    print("Running data augmentation experiment: binary LOSO...")
    binary_results, binary_predictions = evaluate_flat_task_with_augmentation(
        "binary",
        binary_df,
        features,
        group_col,
        binary_labels,
        models,
        experiments,
    )
    print("Running data augmentation experiment: sleepy vs stress LOSO...")
    subtype_results, subtype_predictions = evaluate_flat_task_with_augmentation(
        "subtype",
        subtype_df,
        features,
        group_col,
        subtype_labels,
        models,
        experiments,
    )
    print("Running data augmentation experiment: hierarchical LOSO...")
    hierarchical_results, hierarchical_predictions = (
        evaluate_hierarchical_task_with_augmentation(
            binary_df,
            subtype_df,
            final_df,
            features,
            group_col,
            models,
            models,
            hierarchical_labels,
            experiments,
        )
    )
    print("Running data augmentation experiment: direct 3-class LOSO...")
    direct_results, direct_predictions = evaluate_flat_task_with_augmentation(
        "direct_3class",
        direct_df,
        features,
        group_col,
        direct_labels,
        models,
        experiments,
    )

    for sub in ("binary", "subtype", "hierarchical", "direct_3class", "summary"):
        (result_dir / sub).mkdir(parents=True, exist_ok=True)
    binary_results.to_csv(
        result_dir / "binary" / "binary_augmentation_results.csv", index=False
    )
    subtype_results.to_csv(
        result_dir / "subtype" / "subtype_augmentation_results.csv", index=False
    )
    hierarchical_results.to_csv(
        result_dir / "hierarchical" / "hierarchical_augmentation_results.csv",
        index=False,
    )
    direct_results.to_csv(
        result_dir / "direct_3class" / "direct_3class_augmentation_results.csv",
        index=False,
    )

    task_results = {
        "binary": binary_results,
        "subtype": subtype_results,
        "hierarchical": hierarchical_results,
        "direct_3class": direct_results,
    }
    task_predictions = {
        "binary": binary_predictions,
        "subtype": subtype_predictions,
        "hierarchical": hierarchical_predictions,
        "direct_3class": direct_predictions,
    }
    task_labels = {
        "binary": binary_labels,
        "subtype": subtype_labels,
        "hierarchical": hierarchical_labels,
        "direct_3class": direct_labels,
    }
    reference_baselines = {
        "binary": 0.952,
        "subtype": 1.000,
        "hierarchical": 0.957,
        "direct_3class": 0.951,
    }
    summary_table = build_augmentation_summary_table(
        task_results,
        experiments,
        reference_baselines,
    )
    summary_table.to_csv(result_dir / "summary" / "augmentation_summary.csv", index=False)

    save_best_augmentation_confusion_matrices(
        result_dir,
        task_results,
        task_predictions,
        task_labels,
    )

    summary_text = build_final_augmentation_summary(
        task_results,
        task_predictions,
        task_labels,
        summary_table,
    )
    (result_dir / "summary" / "final_augmentation_summary.txt").write_text(
        summary_text + "\n",
        encoding="utf-8",
    )
    return summary_text


def validate_anomalous_combination(
    datasets: Mapping[str, pd.DataFrame], group_col: str
) -> Dict[str, object]:
    anomalous = datasets["anomalous"].copy()
    sleepy = datasets["sleepy"].copy()
    stress = datasets["stress"].copy()
    combo = pd.concat(
        [sleepy.assign(label="anomalous"), stress.assign(label="anomalous")],
        ignore_index=True,
    )

    common_columns = list(anomalous.columns)
    same_columns = common_columns == list(combo.columns)
    exact_ordered = False
    exact_unordered = False
    if same_columns:
        exact_ordered = anomalous[common_columns].equals(combo[common_columns])
        left = anomalous[common_columns].astype(str).sort_values(common_columns)
        right = combo[common_columns].astype(str).sort_values(common_columns)
        exact_unordered = left.reset_index(drop=True).equals(right.reset_index(drop=True))

    return {
        "anomalous_rows": int(len(anomalous)),
        "sleepy_rows": int(len(sleepy)),
        "stress_rows": int(len(stress)),
        "sleepy_plus_stress_rows": int(len(combo)),
        "same_columns": bool(same_columns),
        "exact_ordered_match_after_relabel": bool(exact_ordered),
        "exact_unordered_match_after_relabel": bool(exact_unordered),
        "anomalous_by_group": anomalous[group_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict(),
        "sleepy_plus_stress_by_group": combo[group_col]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict(),
    }


def pivot_counts(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        pd.crosstab(df[group_col].astype(str), df["target"], margins=True)
        .rename_axis(index=group_col, columns="class")
        .reset_index()
    )


def format_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def describe_missing(df: pd.DataFrame) -> str:
    missing = df.isna().sum()
    nonzero = missing[missing > 0]
    if nonzero.empty:
        return "none"
    return ", ".join(f"{column}={int(count)}" for column, count in nonzero.items())


def save_confusion_matrix_png(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    # Drop any "(...)" detail and wrap long titles so they stay readable.
    import re

    clean_title = re.sub(r"\s*\([^)]*\)", "", title).strip()
    ax.set_title("\n".join(textwrap.wrap(clean_title, width=34)), fontsize=11)
    threshold = cm.max() / 2.0 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_random_forest_feature_importance(
    df: pd.DataFrame,
    features: Sequence[str],
    out_path: Path,
    rf_pipeline: Pipeline,
    best_model_name: str,
) -> pd.DataFrame:
    fitted = clone(rf_pipeline)
    fitted.fit(df.loc[:, features], df["target"])
    classifier = fitted.named_steps["classifier"]
    importances = pd.DataFrame(
        {
            "feature": list(features),
            "importance": classifier.feature_importances_,
            "stage_best_model": best_model_name,
            "random_forest_is_stage_best": best_model_name == "Random Forest",
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(out_path, index=False)
    return importances


def save_feature_importance_png(
    importances: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 18,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2f6f9f")
    ax.set_xlabel("Random Forest importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_hierarchical_vs_direct_comparison(
    hierarchical_best: pd.Series, direct_best: pd.Series
) -> pd.DataFrame:
    rows = []
    for approach, best in [
        ("hierarchical", hierarchical_best),
        ("direct_3class", direct_best),
    ]:
        rows.append(
            {
                "approach": approach,
                "model": best["model"],
                "accuracy": best["accuracy"],
                "balanced_accuracy": best["balanced_accuracy"],
                "macro_f1": best["macro_f1"],
                "weighted_f1": best["weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def off_diagonal_confusions(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> List[str]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    details = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                details.append(f"{true_label}->{pred_label}: {int(cm[i, j])}")
    return details


def classification_report_text(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def best_predictions(
    best: pd.Series, predictions: Mapping[str, Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    return predictions[str(best["model"])]


def append_dataset_summary(
    lines: List[str],
    datasets: Mapping[str, pd.DataFrame],
    group_col: str,
    group_inspection: Mapping[str, Mapping[str, List[object]]],
    validation: Mapping[str, object],
    features: Sequence[str],
    binary_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    final_df: pd.DataFrame,
    direct_df: pd.DataFrame,
) -> None:
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Group column used: {group_col}")
    lines.append(
        "Reason: this column is present in all four files, has repeated values, "
        "and identifies the 4 participants/sources."
    )
    lines.append(
        "Note: the file column is named RMMSD; scientifically this HRV measure is "
        "usually referred to as RMSSD."
    )
    lines.append("")
    lines.append("Columns by file:")
    for name, df in datasets.items():
        lines.append(f"- {name}.csv ({len(df)} rows): {list(df.columns)}")
    lines.append("")
    lines.append("Possible participant/source columns and unique values:")
    for name, columns in group_inspection.items():
        if not columns:
            lines.append(f"- {name}.csv: none detected")
            continue
        for column, values in columns.items():
            lines.append(f"- {name}.csv {column}: {values}")
    lines.append("")
    lines.append("Rows, labels, groups, and missing values by file:")
    for name, df in datasets.items():
        label_counts = (
            df["label"].value_counts(dropna=False).sort_index().to_dict()
            if "label" in df.columns
            else {}
        )
        group_counts = df[group_col].astype(str).value_counts().sort_index().to_dict()
        lines.append(f"- {name}.csv rows: {len(df)}")
        lines.append(f"  label distribution: {label_counts}")
        lines.append(f"  {group_col} distribution: {group_counts}")
        lines.append(f"  missing values: {describe_missing(df)}")
    lines.append("")
    lines.append("Missing values in selected numerical features:")
    for name, df in datasets.items():
        feature_missing = df.loc[:, features].isna().sum()
        missing_text = ", ".join(
            f"{feature}={int(count)}" for feature, count in feature_missing.items()
        )
        lines.append(f"- {name}.csv: {missing_text}")
    lines.append("")
    lines.append("Selected feature columns:")
    lines.append(", ".join(features))
    lines.append("")
    lines.append("Excluded identifier/non-feature columns include:")
    lines.append("window_start, video_source, label, target, filename, participant, subject")
    lines.append("")
    lines.append("Rows per group per class - binary task:")
    lines.append(format_table(pivot_counts(binary_df, group_col)))
    lines.append("")
    lines.append("Rows per group per class - subtype task:")
    lines.append(format_table(pivot_counts(subtype_df, group_col)))
    lines.append("")
    lines.append("Rows per group per class - hierarchical final labels:")
    lines.append(format_table(pivot_counts(final_df, group_col)))
    lines.append("")
    lines.append("Rows per group per class - direct 3-class task:")
    lines.append(format_table(pivot_counts(direct_df, group_col)))
    lines.append(
        "Direct 3-class uses normal.csv + sleepy.csv + stress.csv only; "
        "anomalous.csv is excluded to avoid duplicating sleepy/stress rows."
    )
    lines.append("")
    lines.append("Anomalous.csv validation against sleepy.csv + stress.csv:")
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    lines.append("")


def append_model_summary(
    lines: List[str],
    title: str,
    results: pd.DataFrame,
    best: pd.Series,
    predictions: Mapping[str, Dict[str, List[str]]],
    labels: Sequence[str],
) -> None:
    lines.append(title)
    lines.append("=" * 80)
    overall_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_macro",
        "recall_macro",
    ]
    overall = (
        results[results["row_type"] == "overall"]
        .loc[:, overall_columns]
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    )
    lines.append("Overall model ranking:")
    lines.append(format_table(overall))
    lines.append("")
    lines.append("Per-fold metrics for the selected model:")
    folds = results[
        (results["row_type"] == "fold") & (results["model"] == best["model"])
    ].copy()
    fold_columns = [
        "fold",
        "heldout_group",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_macro",
        "recall_macro",
    ]
    lines.append(format_table(folds.loc[:, fold_columns]))
    lines.append("")
    selected_predictions = best_predictions(best, predictions)
    lines.append(f"Selected model: {best['model']}")
    lines.append(
        "Selection criterion: highest macro F1, then balanced accuracy, "
        "then weighted F1 and accuracy."
    )
    lines.append("Classification report for selected model:")
    lines.append(
        classification_report_text(
            selected_predictions["y_true"], selected_predictions["y_pred"], labels
        )
    )
    lines.append("Confusion matrix for selected model:")
    lines.append(
        str(
            confusion_matrix(
                selected_predictions["y_true"],
                selected_predictions["y_pred"],
                labels=labels,
            )
        )
    )
    lines.append("")


def append_interpretation(
    lines: List[str],
    binary_best: pd.Series,
    subtype_best: pd.Series,
    hierarchical_best: pd.Series,
    direct_best: pd.Series,
    hierarchical_predictions: Mapping[str, Dict[str, List[str]]],
    direct_predictions: Mapping[str, Dict[str, List[str]]],
    hierarchical_labels: Sequence[str],
    direct_labels: Sequence[str],
    hierarchical_results: pd.DataFrame,
    direct_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines.append("SHORT INTERPRETATION")
    lines.append("=" * 80)
    lines.append(
        "These results are proof-of-concept only. Leave-One-Subject-Out is the "
        "right evaluation choice here, but 4 participants is not enough for strong "
        "generalization claims."
    )
    lines.append(
        f"Binary normal vs abnormal worked best with {binary_best['model']} "
        f"(macro F1={binary_best['macro_f1']:.3f}, balanced accuracy="
        f"{binary_best['balanced_accuracy']:.3f})."
    )
    lines.append(
        f"Sleepy vs stress worked best with {subtype_best['model']} "
        f"(macro F1={subtype_best['macro_f1']:.3f}, balanced accuracy="
        f"{subtype_best['balanced_accuracy']:.3f})."
    )
    lines.append(
        f"The full hierarchical system worked best with {hierarchical_best['model']} "
        f"(macro F1={hierarchical_best['macro_f1']:.3f}, balanced accuracy="
        f"{hierarchical_best['balanced_accuracy']:.3f}, weighted F1="
        f"{hierarchical_best['weighted_f1']:.3f})."
    )
    lines.append(
        f"The direct 3-class system worked best with {direct_best['model']} "
        f"(macro F1={direct_best['macro_f1']:.3f}, balanced accuracy="
        f"{direct_best['balanced_accuracy']:.3f}, weighted F1="
        f"{direct_best['weighted_f1']:.3f})."
    )

    lines.append("")
    lines.append("Hierarchical vs direct 3-class comparison:")
    lines.append(format_table(comparison))

    if direct_best["macro_f1"] > hierarchical_best["macro_f1"]:
        better = "direct 3-class"
    elif direct_best["macro_f1"] < hierarchical_best["macro_f1"]:
        better = "hierarchical"
    else:
        better = "tie on macro F1"
    lines.append(
        "Based mainly on macro F1, the better approach was: "
        f"{better}."
    )

    hierarchical_selected = best_predictions(hierarchical_best, hierarchical_predictions)
    hierarchical_confusions = off_diagonal_confusions(
        hierarchical_selected["y_true"],
        hierarchical_selected["y_pred"],
        hierarchical_labels,
    )
    if hierarchical_confusions:
        lines.append(
            "Hierarchical class confusions: " + ", ".join(hierarchical_confusions)
        )
    else:
        lines.append("Hierarchical class confusions: none in selected LOSO predictions.")

    direct_selected = best_predictions(direct_best, direct_predictions)
    direct_confusions = off_diagonal_confusions(
        direct_selected["y_true"],
        direct_selected["y_pred"],
        direct_labels,
    )
    if direct_confusions:
        lines.append("Direct 3-class confusions: " + ", ".join(direct_confusions))
    else:
        lines.append("Direct 3-class confusions: none in selected LOSO predictions.")

    direct_normal_confusions = [
        item
        for item in direct_confusions
        if item.startswith("normal->") or "->normal:" in item
    ]
    if direct_normal_confusions:
        lines.append(
            "The direct 3-class model did confuse normal with sleepy/stress: "
            + ", ".join(direct_normal_confusions)
        )
    else:
        lines.append(
            "The direct 3-class model did not confuse normal with sleepy/stress "
            "in the selected LOSO predictions."
        )

    folds = hierarchical_results[
        (hierarchical_results["row_type"] == "fold")
        & (hierarchical_results["model"] == hierarchical_best["model"])
    ]
    macro_min = folds["macro_f1"].min()
    macro_max = folds["macro_f1"].max()
    bal_min = folds["balanced_accuracy"].min()
    bal_max = folds["balanced_accuracy"].max()
    lines.append(
        "Fold stability for the selected hierarchical system: "
        f"macro F1 ranged from {macro_min:.3f} to {macro_max:.3f}; "
        f"balanced accuracy ranged from {bal_min:.3f} to {bal_max:.3f}."
    )
    direct_folds = direct_results[
        (direct_results["row_type"] == "fold")
        & (direct_results["model"] == direct_best["model"])
    ]
    direct_macro_min = direct_folds["macro_f1"].min()
    direct_macro_max = direct_folds["macro_f1"].max()
    direct_bal_min = direct_folds["balanced_accuracy"].min()
    direct_bal_max = direct_folds["balanced_accuracy"].max()
    lines.append(
        "Fold stability for the selected direct 3-class system: "
        f"macro F1 ranged from {direct_macro_min:.3f} to {direct_macro_max:.3f}; "
        f"balanced accuracy ranged from {direct_bal_min:.3f} to "
        f"{direct_bal_max:.3f}."
    )
    lines.append(
        "More participants are needed before using the model for reliable claims "
        "about unseen students or deployment settings."
    )
    lines.append("")


def main() -> None:
    args = parse_args()
    csv_dir = resolve_csv_dir(args.data_dir)
    results_dir = args.results_dir.expanduser().resolve()
    augmentation_results_dir = args.augmentation_results_dir.expanduser()
    if augmentation_results_dir.is_absolute():
        augmentation_results_dir = augmentation_results_dir.resolve()
    else:
        project_root = (
            csv_dir.parent.parent
            if csv_dir.name == "supervised" and csv_dir.parent.name == "data"
            else Path.cwd()
        )
        augmentation_results_dir = (project_root / augmentation_results_dir).resolve()
    result_dirs = prepare_results_dirs(results_dir)

    mpl_config = results_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    raw_datasets = load_datasets(csv_dir)
    features = validate_features(raw_datasets)
    datasets = coerce_feature_columns(raw_datasets, features)
    group_inspection = inspect_group_candidates(datasets)
    group_col = infer_group_column(datasets)

    validation = validate_anomalous_combination(datasets, group_col)

    binary_df = build_binary_dataset(datasets, group_col)
    subtype_df = build_subtype_dataset(datasets, group_col)
    final_df = build_hierarchical_test_dataset(datasets, group_col)
    direct_df = build_direct_3class_dataset(datasets, group_col)

    models = model_factories()
    binary_labels = ["normal", "abnormal"]
    subtype_labels = ["sleepy", "stress"]
    hierarchical_labels = ["normal", "sleepy", "stress"]
    direct_labels = ["normal", "sleepy", "stress"]

    print(f"Using CSV directory: {csv_dir}")
    print(f"Using group column: {group_col}")
    print("Running binary normal vs abnormal LOSO evaluation...")
    binary_results, binary_best, binary_predictions = evaluate_flat_task(
        "binary",
        binary_df,
        features,
        group_col,
        binary_labels,
        models,
    )

    print("Running sleepy vs stress LOSO evaluation...")
    subtype_results, subtype_best, subtype_predictions = evaluate_flat_task(
        "subtype",
        subtype_df,
        features,
        group_col,
        subtype_labels,
        models,
    )

    print("Running full hierarchical LOSO evaluation...")
    hierarchical_results, hierarchical_best, hierarchical_predictions = (
        evaluate_hierarchical_task(
            binary_df,
            subtype_df,
            final_df,
            features,
            group_col,
            models,
            models,
            hierarchical_labels,
        )
    )

    print("Running direct 3-class LOSO evaluation...")
    direct_results, direct_best, direct_predictions = direct_3class_classification(
        direct_df,
        features,
        group_col,
        models,
        direct_labels,
    )

    binary_results.to_csv(result_dirs["binary"] / "binary_results.csv", index=False)
    subtype_results.to_csv(result_dirs["subtype"] / "subtype_results.csv", index=False)
    hierarchical_results.to_csv(
        result_dirs["hierarchical"] / "hierarchical_results.csv",
        index=False,
    )
    direct_results.to_csv(
        result_dirs["direct_3class"] / "direct_3class_results.csv",
        index=False,
    )
    direct_results[direct_results["row_type"] == "fold"].to_csv(
        result_dirs["direct_3class"] / "direct_3class_per_fold_results.csv",
        index=False,
    )

    binary_selected = best_predictions(binary_best, binary_predictions)
    subtype_selected = best_predictions(subtype_best, subtype_predictions)
    hierarchical_selected = best_predictions(hierarchical_best, hierarchical_predictions)
    direct_selected = best_predictions(direct_best, direct_predictions)

    save_confusion_matrix_png(
        binary_selected["y_true"],
        binary_selected["y_pred"],
        binary_labels,
        f"Binary LOSO Confusion Matrix: {binary_best['model']}",
        result_dirs["binary"] / "binary_confusion_matrix.png",
    )
    save_confusion_matrix_png(
        subtype_selected["y_true"],
        subtype_selected["y_pred"],
        subtype_labels,
        f"Subtype LOSO Confusion Matrix: {subtype_best['model']}",
        result_dirs["subtype"] / "subtype_confusion_matrix.png",
    )
    save_confusion_matrix_png(
        hierarchical_selected["y_true"],
        hierarchical_selected["y_pred"],
        hierarchical_labels,
        f"Hierarchical LOSO Confusion Matrix: {hierarchical_best['model']}",
        result_dirs["hierarchical"] / "hierarchical_confusion_matrix.png",
    )
    save_confusion_matrix_png(
        direct_selected["y_true"],
        direct_selected["y_pred"],
        direct_labels,
        f"Direct 3-Class LOSO Confusion Matrix: {direct_best['model']}",
        result_dirs["direct_3class"] / "direct_3class_confusion_matrix.png",
    )

    direct_report = [
        f"Selected direct 3-class model: {direct_best['model']}",
        (
            "Selection criterion: highest macro F1, then balanced accuracy, "
            "then weighted F1 and accuracy."
        ),
        "",
        "Classification report:",
        classification_report_text(
            direct_selected["y_true"], direct_selected["y_pred"], direct_labels
        ),
        "Confusion matrix:",
        str(
            confusion_matrix(
                direct_selected["y_true"],
                direct_selected["y_pred"],
                labels=direct_labels,
            )
        ),
    ]
    (result_dirs["direct_3class"] / "direct_3class_classification_report.txt").write_text(
        "\n".join(direct_report) + "\n",
        encoding="utf-8",
    )

    save_random_forest_feature_importance(
        binary_df,
        features,
        result_dirs["binary"] / "feature_importance_binary.csv",
        models["Random Forest"],
        str(binary_best["model"]),
    )
    save_random_forest_feature_importance(
        subtype_df,
        features,
        result_dirs["subtype"] / "feature_importance_subtype.csv",
        models["Random Forest"],
        str(subtype_best["model"]),
    )
    direct_importances = save_random_forest_feature_importance(
        direct_df,
        features,
        result_dirs["direct_3class"] / "direct_3class_feature_importance.csv",
        models["Random Forest"],
        str(direct_best["model"]),
    )
    save_feature_importance_png(
        direct_importances,
        result_dirs["direct_3class"] / "direct_3class_feature_importance.png",
        "Direct 3-Class Random Forest Feature Importance",
    )

    comparison = build_hierarchical_vs_direct_comparison(
        hierarchical_best,
        direct_best,
    )
    comparison.to_csv(
        result_dirs["comparison"] / "hierarchical_vs_direct_3class_comparison.csv",
        index=False,
    )

    summary_lines: List[str] = []
    append_dataset_summary(
        summary_lines,
        datasets,
        group_col,
        group_inspection,
        validation,
        features,
        binary_df,
        subtype_df,
        final_df,
        direct_df,
    )
    append_model_summary(
        summary_lines,
        "BINARY: NORMAL VS ABNORMAL",
        binary_results,
        binary_best,
        binary_predictions,
        binary_labels,
    )
    append_model_summary(
        summary_lines,
        "SUBTYPE: SLEEPY VS STRESS",
        subtype_results,
        subtype_best,
        subtype_predictions,
        subtype_labels,
    )
    append_model_summary(
        summary_lines,
        "HIERARCHICAL: NORMAL, SLEEPY, STRESS",
        hierarchical_results,
        hierarchical_best,
        hierarchical_predictions,
        hierarchical_labels,
    )
    append_model_summary(
        summary_lines,
        "DIRECT 3-CLASS: NORMAL, SLEEPY, STRESS",
        direct_results,
        direct_best,
        direct_predictions,
        direct_labels,
    )
    append_interpretation(
        summary_lines,
        binary_best,
        subtype_best,
        hierarchical_best,
        direct_best,
        hierarchical_predictions,
        direct_predictions,
        hierarchical_labels,
        direct_labels,
        hierarchical_results,
        direct_results,
        comparison,
    )

    summary_text = "\n".join(summary_lines)
    (result_dirs["summary"] / "final_summary.txt").write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    augmentation_summary_text = run_data_augmentation_experiments(
        augmentation_results_dir,
        binary_df,
        subtype_df,
        final_df,
        direct_df,
        features,
        group_col,
        models,
        binary_labels,
        subtype_labels,
        hierarchical_labels,
        direct_labels,
    )

    print("")
    print(summary_text)
    print(f"Saved outputs to: {results_dir}")
    print("")
    print(augmentation_summary_text)
    print(f"Saved data augmentation outputs to: {augmentation_results_dir}")


if __name__ == "__main__":
    main()
