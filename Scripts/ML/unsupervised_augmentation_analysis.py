#!/usr/bin/env python3
"""Participant-aware data augmentation experiments for unsupervised wellbeing ML.

Clustering augmentation is label-free and participant-wise: true labels are not
used when synthetic rows are generated. Labels are used only after clustering to
evaluate discovered clusters.

LOSO anomaly detection trains only on normal samples from training participants.
Augmentation is applied only to those normal training samples and never to the
held-out participant.
"""

from __future__ import annotations

import argparse
import ast
import os
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


RANDOM_STATE = 42
GROUP_COL = "video_source"
BINARY_LABELS = ["normal", "anomalous"]
THREECLASS_LABELS = ["normal", "sleepy", "stress"]

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

FEATURE_GROUPS = {
    "posture_only": [
        "p_shoulder_mean",
        "p_shoulder_std",
        "p_shoulder_bad_pct",
        "p_head_tilt_mean",
        "p_head_tilt_std",
        "p_head_tilt_bad_pct",
        "p_fhp_mean",
        "p_fhp_std",
        "p_fhp_bad_pct",
    ],
    "eye_blink_only": [
        "b_ear_mean",
        "b_ear_std",
        "b_perclos",
        "b_blink_count",
        "b_blink_freq",
        "b_ibi_mean",
        "b_ibi_std",
    ],
    "hr_hrv_only": ["HR", "RMMSD"],
    "all_features": FEATURE_COLUMNS,
}

BASELINES = {
    "binary_clustering": {
        "method": "Gaussian Mixture",
        "macro_f1": 0.810,
        "balanced_accuracy": 0.860,
    },
    "anomaly_detection_loso": {
        "method": "LocalOutlierFactor",
        "macro_f1": 0.806,
        "balanced_accuracy": 0.778,
    },
    "threeclass_clustering": {
        "method": "Gaussian Mixture",
        "macro_f1": 0.815,
        "balanced_accuracy": 0.822,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised participant-aware augmentation experiments."
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
        default=Path("results/unsupervised/augmentation"),
        help="Directory for augmentation experiment outputs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for stochastic models and jitter.",
    )
    return parser.parse_args()


def augmentation_settings() -> List[Dict[str, object]]:
    return [
        {
            "augmentation_name": "baseline",
            "augmentation_factor": 0,
            "noise_level": 0.0,
            "is_baseline": True,
        },
        {
            "augmentation_name": "aug_1",
            "augmentation_factor": 1,
            "noise_level": 0.01,
            "is_baseline": False,
        },
        {
            "augmentation_name": "aug_2",
            "augmentation_factor": 1,
            "noise_level": 0.02,
            "is_baseline": False,
        },
        {
            "augmentation_name": "aug_3",
            "augmentation_factor": 1,
            "noise_level": 0.05,
            "is_baseline": False,
        },
        {
            "augmentation_name": "aug_4",
            "augmentation_factor": 2,
            "noise_level": 0.02,
            "is_baseline": False,
        },
    ]


def safe_label(label: str) -> str:
    return str(label).replace(" ", "_").replace("/", "_").replace("-", "_").lower()


def load_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data_dir = data_dir.expanduser().resolve()
    required = ["normal", "anomalous", "sleepy", "stress"]
    datasets = {}
    missing = []
    for name in required:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            missing.append(path.name)
        else:
            datasets[name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError(
            f"Expected all unsupervised CSVs inside {data_dir}; missing: {missing}"
        )
    return datasets


def coerce_features(datasets: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    converted = {}
    for name, df in datasets.items():
        copy = df.copy()
        missing = [feature for feature in FEATURE_COLUMNS if feature not in copy.columns]
        if missing:
            raise ValueError(f"{name}.csv is missing required features: {missing}")
        if GROUP_COL not in copy.columns:
            raise ValueError(f"{name}.csv is missing group column {GROUP_COL!r}.")
        for feature in FEATURE_COLUMNS:
            copy[feature] = pd.to_numeric(copy[feature], errors="coerce")
        converted[name] = copy
    return converted


def prepare_frame(df: pd.DataFrame, true_label: str, source_file: str) -> pd.DataFrame:
    prepared = df.copy()
    if "label" in prepared.columns:
        prepared = prepared.rename(columns={"label": "original_label"})
    else:
        prepared["original_label"] = np.nan
    prepared["true_label"] = true_label
    prepared["source_file"] = source_file
    prepared["source_row"] = np.arange(len(prepared))
    prepared["is_augmented"] = False
    return prepared


def build_binary_dataset(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            prepare_frame(datasets["normal"], "normal", "normal.csv"),
            prepare_frame(datasets["anomalous"], "anomalous", "anomalous.csv"),
        ],
        ignore_index=True,
        sort=False,
    ).dropna(subset=[GROUP_COL])


def build_threeclass_dataset(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            prepare_frame(datasets["normal"], "normal", "normal.csv"),
            prepare_frame(datasets["sleepy"], "sleepy", "sleepy.csv"),
            prepare_frame(datasets["stress"], "stress", "stress.csv"),
        ],
        ignore_index=True,
        sort=False,
    ).dropna(subset=[GROUP_COL])


def preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def augment_participantwise_label_free(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    group_col: str,
    augmentation_factor: int,
    noise_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Gaussian jitter by participant only; true labels are not used."""
    original = df.copy().reset_index(drop=True)
    original["is_augmented"] = False
    if augmentation_factor <= 0 or noise_level <= 0:
        return original

    lower_bounds = original.loc[:, feature_columns].quantile(0.01)
    upper_bounds = original.loc[:, feature_columns].quantile(0.99)
    rng = np.random.default_rng(random_state)
    parts = [original]

    for group_value, group_frame in original.groupby(group_col, sort=True):
        group_features = group_frame.loc[:, feature_columns].reset_index(drop=True)
        feature_std = group_features.std(axis=0, ddof=0).fillna(0.0)
        noise_scale = (noise_level * feature_std).to_numpy(dtype=float)

        for _ in range(augmentation_factor):
            synthetic = group_frame.copy().reset_index(drop=True).astype(
                {feature: "float64" for feature in feature_columns}
            )
            noise = rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=group_features.shape,
            )
            noisy_features = pd.DataFrame(
                group_features.to_numpy(dtype=float) + noise,
                columns=feature_columns,
            ).clip(lower=lower_bounds, upper=upper_bounds, axis=1)
            synthetic.loc[:, feature_columns] = noisy_features.to_numpy()
            synthetic[group_col] = group_value
            synthetic["is_augmented"] = True
            parts.append(synthetic)

    return pd.concat(parts, ignore_index=True, sort=False)


def augment_normal_training_participantwise(
    normal_train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    group_col: str,
    augmentation_factor: int,
    noise_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Gaussian jitter for normal-only anomaly training, grouped by participant."""
    return augment_participantwise_label_free(
        normal_train_df,
        feature_columns,
        group_col,
        augmentation_factor,
        noise_level,
        random_state,
    )


def clustering_models(n_clusters: int, random_state: int) -> Dict[str, object]:
    return {
        "K-Means": KMeans(
            n_clusters=n_clusters,
            n_init=50,
            random_state=random_state,
        ),
        "Gaussian Mixture": GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        ),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
    }


def anomaly_models(random_state: int) -> Dict[str, object]:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=500,
            contamination=0.05,
            random_state=random_state,
            n_jobs=1,
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True,
        ),
    }


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, float]:
    metrics = {
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


def internal_cluster_metrics(
    X_scaled: np.ndarray, cluster_ids: Sequence[int]
) -> Dict[str, float]:
    unique = np.unique(cluster_ids)
    if len(unique) < 2 or len(unique) >= len(cluster_ids):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
    return {
        "silhouette": float(silhouette_score(X_scaled, cluster_ids)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, cluster_ids)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, cluster_ids)),
    }


def majority_vote_mapping(
    y_true: Sequence[str], cluster_ids: Sequence[int]
) -> Dict[int, str]:
    table = pd.crosstab(pd.Series(cluster_ids, name="cluster"), pd.Series(y_true))
    mapping = {}
    for cluster_id, row in table.iterrows():
        mapping[int(cluster_id)] = str(row.sort_values(ascending=False).index[0])
    return mapping


def mapped_labels(cluster_ids: Sequence[int], mapping: Mapping[int, str]) -> List[str]:
    return [mapping[int(cluster_id)] for cluster_id in cluster_ids]


def cluster_label_table_string(y_true: Sequence[str], cluster_ids: Sequence[int]) -> str:
    table = pd.crosstab(
        pd.Series(cluster_ids, name="cluster"),
        pd.Series(y_true, name="true_label"),
    )
    mapping = majority_vote_mapping(y_true, cluster_ids)
    table["mapped_label"] = [mapping[int(idx)] for idx in table.index]
    return table.to_string()


def evaluate_cluster_assignments(
    task: str,
    method_name: str,
    augmentation: Mapping[str, object],
    X_original_scaled: np.ndarray,
    cluster_ids_original: np.ndarray,
    y_true: Sequence[str],
    labels: Sequence[str],
) -> Tuple[Dict[str, object], List[str]]:
    mapping = majority_vote_mapping(y_true, cluster_ids_original)
    y_pred = mapped_labels(cluster_ids_original, mapping)
    row: Dict[str, object] = {
        "task": task,
        "method": method_name,
        "augmentation_name": augmentation["augmentation_name"],
        "augmentation_factor": int(augmentation["augmentation_factor"]),
        "noise_level": float(augmentation["noise_level"]),
        "is_baseline": bool(augmentation["is_baseline"]),
        "n_clusters": int(len(np.unique(cluster_ids_original))),
        "cluster_mapping": str(mapping),
        "cluster_label_table": cluster_label_table_string(
            y_true, cluster_ids_original
        ),
        "confusion_matrix": str(confusion_matrix(y_true, y_pred, labels=labels)),
    }
    row.update(internal_cluster_metrics(X_original_scaled, cluster_ids_original))
    row["adjusted_rand_index"] = adjusted_rand_score(y_true, cluster_ids_original)
    row["normalized_mutual_info"] = normalized_mutual_info_score(
        y_true, cluster_ids_original
    )
    if len(labels) == 3:
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            y_true, cluster_ids_original
        )
        row["homogeneity"] = homogeneity
        row["completeness"] = completeness
        row["v_measure"] = v_measure
    row.update(classification_metrics(y_true, y_pred, labels))
    return row, y_pred


def run_clustering_augmentation(
    df: pd.DataFrame,
    features: Sequence[str],
    labels: Sequence[str],
    n_clusters: int,
    task_name: str,
    random_state: int,
    settings: Sequence[Mapping[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, List[str]]]]]:
    rows = []
    predictions: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    y_true_all = df["true_label"].tolist()

    for setting_idx, setting in enumerate(settings):
        augmented_df = augment_participantwise_label_free(
            df,
            features,
            GROUP_COL,
            int(setting["augmentation_factor"]),
            float(setting["noise_level"]),
            random_state + setting_idx * 1000,
        )
        original_mask = ~augmented_df["is_augmented"].astype(bool)
        pipeline = preprocessing_pipeline()
        X_all_scaled = pipeline.fit_transform(augmented_df.loc[:, features])
        X_original_scaled = X_all_scaled[original_mask.to_numpy()]

        for method_name, model in clustering_models(n_clusters, random_state).items():
            fitted = clone(model)
            if method_name in {"K-Means", "Gaussian Mixture"}:
                fitted.fit(X_all_scaled)
                cluster_ids_original = fitted.predict(X_original_scaled)
            else:
                cluster_ids_all = fitted.fit_predict(X_all_scaled)
                cluster_ids_original = cluster_ids_all[original_mask.to_numpy()]

            row, y_pred = evaluate_cluster_assignments(
                task_name,
                method_name,
                setting,
                X_original_scaled,
                np.asarray(cluster_ids_original),
                y_true_all,
                labels,
            )
            row["original_rows_evaluated"] = int(original_mask.sum())
            row["augmented_rows_fit"] = int(augmented_df["is_augmented"].sum())
            row["total_rows_fit"] = int(len(augmented_df))
            rows.append(row)

            predictions.setdefault(method_name, {})[str(setting["augmentation_name"])] = {
                "y_true": y_true_all,
                "y_pred": y_pred,
                "cluster_ids": [int(value) for value in cluster_ids_original],
            }

    return pd.DataFrame(rows), predictions


def run_loso_anomaly_augmentation(
    full_df: pd.DataFrame,
    features: Sequence[str],
    random_state: int,
    settings: Sequence[Mapping[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, List[str]]]]]:
    rows = []
    prediction_rows = []
    predictions: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    groups = sorted(full_df[GROUP_COL].dropna().astype(str).unique().tolist())

    for setting_idx, setting in enumerate(settings):
        setting_name = str(setting["augmentation_name"])
        for method_name, model in anomaly_models(random_state).items():
            overall_true: List[str] = []
            overall_pred: List[str] = []

            for fold, heldout in enumerate(groups, start=1):
                train_mask = (
                    (full_df[GROUP_COL].astype(str) != heldout)
                    & (full_df["true_label"] == "normal")
                )
                test_mask = full_df[GROUP_COL].astype(str) == heldout
                train_normal = full_df.loc[train_mask].reset_index(drop=True)
                test_df = full_df.loc[test_mask].reset_index(drop=True)
                seed = random_state + setting_idx * 1000 + fold * 100

                augmented_train = augment_normal_training_participantwise(
                    train_normal,
                    features,
                    GROUP_COL,
                    int(setting["augmentation_factor"]),
                    float(setting["noise_level"]),
                    seed,
                )
                pipeline = preprocessing_pipeline()
                X_train = pipeline.fit_transform(augmented_train.loc[:, features])
                X_test = pipeline.transform(test_df.loc[:, features])

                fitted = clone(model)
                fitted.fit(X_train)
                raw_pred = fitted.predict(X_test)
                y_pred = ["normal" if value == 1 else "anomalous" for value in raw_pred]
                y_true = test_df["true_label"].tolist()

                row: Dict[str, object] = {
                    "task": "anomaly_detection_loso",
                    "row_type": "fold",
                    "method": method_name,
                    "augmentation_name": setting_name,
                    "augmentation_factor": int(setting["augmentation_factor"]),
                    "noise_level": float(setting["noise_level"]),
                    "is_baseline": bool(setting["is_baseline"]),
                    "fold": fold,
                    "held_out_group": heldout,
                    "train_normal_original_rows": int(len(train_normal)),
                    "train_normal_augmented_rows": int(
                        augmented_train["is_augmented"].sum()
                    ),
                    "train_normal_rows_fit": int(len(augmented_train)),
                    "test_rows": int(len(test_df)),
                    "test_normal_samples": int(
                        (test_df["true_label"] == "normal").sum()
                    ),
                    "test_anomalous_samples": int(
                        (test_df["true_label"] == "anomalous").sum()
                    ),
                    "confusion_matrix": str(
                        confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
                    ),
                }
                row.update(classification_metrics(y_true, y_pred, BINARY_LABELS))
                rows.append(row)

                overall_true.extend(y_true)
                overall_pred.extend(y_pred)

                metadata = test_df[
                    [
                        column
                        for column in [
                            "source_file",
                            "source_row",
                            "window_start",
                            GROUP_COL,
                            "original_label",
                        ]
                        if column in test_df.columns
                    ]
                ].copy()
                metadata["true_label"] = y_true
                metadata["prediction"] = y_pred
                metadata["method"] = method_name
                metadata["augmentation_name"] = setting_name
                metadata["augmentation_factor"] = int(setting["augmentation_factor"])
                metadata["noise_level"] = float(setting["noise_level"])
                metadata["fold"] = fold
                metadata["held_out_group"] = heldout
                prediction_rows.append(metadata)

            overall_row: Dict[str, object] = {
                "task": "anomaly_detection_loso",
                "row_type": "overall",
                "method": method_name,
                "augmentation_name": setting_name,
                "augmentation_factor": int(setting["augmentation_factor"]),
                "noise_level": float(setting["noise_level"]),
                "is_baseline": bool(setting["is_baseline"]),
                "fold": 0,
                "held_out_group": "ALL",
                "train_normal_original_rows": np.nan,
                "train_normal_augmented_rows": np.nan,
                "train_normal_rows_fit": np.nan,
                "test_rows": int(len(full_df)),
                "test_normal_samples": int((full_df["true_label"] == "normal").sum()),
                "test_anomalous_samples": int(
                    (full_df["true_label"] == "anomalous").sum()
                ),
                "confusion_matrix": str(
                    confusion_matrix(overall_true, overall_pred, labels=BINARY_LABELS)
                ),
                "classification_report": classification_report(
                    overall_true,
                    overall_pred,
                    labels=BINARY_LABELS,
                    zero_division=0,
                ),
            }
            overall_row.update(
                classification_metrics(overall_true, overall_pred, BINARY_LABELS)
            )
            rows.append(overall_row)

            predictions.setdefault(method_name, {})[setting_name] = {
                "y_true": overall_true,
                "y_pred": overall_pred,
            }

    return (
        pd.DataFrame(rows),
        pd.concat(prediction_rows, ignore_index=True),
        predictions,
    )


def save_confusion_matrix_png(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_best(
    results: pd.DataFrame,
    exclude_baseline: bool = False,
    method: Optional[str] = None,
) -> pd.Series:
    subset = results.copy()
    if "row_type" in subset.columns:
        subset = subset[subset["row_type"].fillna("overall") == "overall"].copy()
    if exclude_baseline:
        subset = subset[~subset["is_baseline"]].copy()
    if method is not None:
        subset = subset[subset["method"] == method].copy()
    return (
        subset.sort_values(
            ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"],
            ascending=False,
        )
        .iloc[0]
        .copy()
    )


def predictions_for(
    predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    method: str,
    augmentation_name: str,
) -> Mapping[str, List[str]]:
    return predictions[method][augmentation_name]


def save_best_method_confusions(
    results_dir: Path,
    binary_predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    binary_results: pd.DataFrame,
    three_predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    three_results: pd.DataFrame,
    anomaly_predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    anomaly_results: pd.DataFrame,
) -> None:
    binary_files = {
        "K-Means": "binary_kmeans_best_augmented_confusion_matrix.png",
        "Gaussian Mixture": "binary_gmm_best_augmented_confusion_matrix.png",
        "Agglomerative": "binary_agglomerative_best_augmented_confusion_matrix.png",
    }
    three_files = {
        "K-Means": "threeclass_kmeans_best_augmented_confusion_matrix.png",
        "Gaussian Mixture": "threeclass_gmm_best_augmented_confusion_matrix.png",
        "Agglomerative": "threeclass_agglomerative_best_augmented_confusion_matrix.png",
    }
    anomaly_files = {
        "IsolationForest": "isolation_forest_loso_best_augmented_confusion_matrix.png",
        "OneClassSVM": "oneclass_svm_loso_best_augmented_confusion_matrix.png",
        "LocalOutlierFactor": "lof_loso_best_augmented_confusion_matrix.png",
    }

    for method, filename in binary_files.items():
        best = select_best(binary_results, exclude_baseline=True, method=method)
        pred = predictions_for(binary_predictions, method, best["augmentation_name"])
        save_confusion_matrix_png(
            pred["y_true"],
            pred["y_pred"],
            BINARY_LABELS,
            f"Binary {method} Best Augmented Confusion Matrix",
            results_dir / filename,
        )

    for method, filename in three_files.items():
        best = select_best(three_results, exclude_baseline=True, method=method)
        pred = predictions_for(three_predictions, method, best["augmentation_name"])
        save_confusion_matrix_png(
            pred["y_true"],
            pred["y_pred"],
            THREECLASS_LABELS,
            f"3-Class {method} Best Augmented Confusion Matrix",
            results_dir / filename,
        )

    for method, filename in anomaly_files.items():
        best = select_best(anomaly_results, exclude_baseline=True, method=method)
        pred = predictions_for(anomaly_predictions, method, best["augmentation_name"])
        save_confusion_matrix_png(
            pred["y_true"],
            pred["y_pred"],
            BINARY_LABELS,
            f"LOSO {method} Best Augmented Confusion Matrix",
            results_dir / filename,
        )


def run_clustering_feature_ablation(
    task_name: str,
    df: pd.DataFrame,
    labels: Sequence[str],
    n_clusters: int,
    best_setting: Mapping[str, object],
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for feature_group, features in FEATURE_GROUPS.items():
        results, _ = run_clustering_augmentation(
            df,
            features,
            labels,
            n_clusters,
            task_name,
            random_state,
            [best_setting],
        )
        results["feature_group"] = feature_group
        results["n_features"] = len(features)
        rows.append(results)
    return pd.concat(rows, ignore_index=True)


def run_anomaly_feature_ablation(
    full_df: pd.DataFrame,
    best_setting: Mapping[str, object],
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for feature_group, features in FEATURE_GROUPS.items():
        results, _, _ = run_loso_anomaly_augmentation(
            full_df,
            features,
            random_state,
            [best_setting],
        )
        results["feature_group"] = feature_group
        results["n_features"] = len(features)
        rows.append(results)
    return pd.concat(rows, ignore_index=True)


def setting_from_row(row: pd.Series) -> Dict[str, object]:
    return {
        "augmentation_name": row["augmentation_name"],
        "augmentation_factor": int(row["augmentation_factor"]),
        "noise_level": float(row["noise_level"]),
        "is_baseline": bool(row["is_baseline"]),
    }


def build_baseline_comparison(
    binary_results: pd.DataFrame,
    three_results: pd.DataFrame,
    anomaly_results: pd.DataFrame,
) -> pd.DataFrame:
    task_to_results = {
        "binary_clustering": binary_results,
        "threeclass_clustering": three_results,
        "anomaly_detection_loso": anomaly_results,
    }
    rows = []
    for task, results in task_to_results.items():
        baseline = BASELINES[task]
        best_augmented = select_best(results, exclude_baseline=True)
        rows.append(
            {
                "task": task,
                "method": best_augmented["method"],
                "baseline_method": baseline["method"],
                "baseline_macro_f1": baseline["macro_f1"],
                "best_augmented_macro_f1": best_augmented["macro_f1"],
                "change_macro_f1": best_augmented["macro_f1"]
                - baseline["macro_f1"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "best_augmented_balanced_accuracy": best_augmented[
                    "balanced_accuracy"
                ],
                "change_balanced_accuracy": best_augmented["balanced_accuracy"]
                - baseline["balanced_accuracy"],
                "best_augmentation_factor": int(best_augmented["augmentation_factor"]),
                "best_noise_level": float(best_augmented["noise_level"]),
                "best_augmentation_name": best_augmented["augmentation_name"],
            }
        )
    return pd.DataFrame(rows)


def confusion_from_prediction(
    prediction: Mapping[str, Sequence[str]], labels: Sequence[str]
) -> np.ndarray:
    return confusion_matrix(prediction["y_true"], prediction["y_pred"], labels=labels)


def confusion_counts_for_best(
    results: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    labels: Sequence[str],
    exclude_baseline: bool,
) -> np.ndarray:
    best = select_best(results, exclude_baseline=exclude_baseline)
    pred = predictions_for(predictions, best["method"], best["augmentation_name"])
    return confusion_from_prediction(pred, labels)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False)


def change_sentence(task: str, baseline: float, augmented: float) -> str:
    delta = augmented - baseline
    if delta > 0.005:
        direction = "improved"
    elif delta > 0:
        direction = "improved slightly"
    elif delta < -0.005:
        direction = "got worse"
    elif delta < 0:
        direction = "got slightly worse"
    else:
        direction = "did not change"
    return f"{task} {direction}: macro F1 {baseline:.3f} -> {augmented:.3f} (delta {delta:+.3f})."


def build_clustering_summary(
    title: str,
    results: pd.DataFrame,
    labels: Sequence[str],
) -> str:
    lines = [title, "=" * 80]
    display_cols = [
        "method",
        "augmentation_name",
        "augmentation_factor",
        "noise_level",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    if len(labels) == 3:
        display_cols.extend(["homogeneity", "completeness", "v_measure"])
    lines.append(format_table(results.loc[:, display_cols]))
    lines.append("")
    for method in sorted(results["method"].unique()):
        best_aug = select_best(results, exclude_baseline=True, method=method)
        lines.append(
            f"Best augmented {method}: {best_aug['augmentation_name']} "
            f"(factor={int(best_aug['augmentation_factor'])}, "
            f"noise={best_aug['noise_level']:.2f}, macro F1="
            f"{best_aug['macro_f1']:.3f}, balanced accuracy="
            f"{best_aug['balanced_accuracy']:.3f})."
        )
        lines.append("Cluster vs true-label table for that setting:")
        lines.append(str(best_aug["cluster_label_table"]))
        lines.append("Confusion matrix:")
        lines.append(str(best_aug["confusion_matrix"]))
        lines.append("")
    return "\n".join(lines)


def build_anomaly_summary(results: pd.DataFrame) -> str:
    lines = ["LOSO ANOMALY DETECTION AUGMENTATION SUMMARY", "=" * 80]
    lines.append(
        "The anomaly detector was trained only on normal samples from training "
        "participants. Augmentation was applied only to normal training samples "
        "and never to the held-out participant."
    )
    lines.append("")
    display_cols = [
        "method",
        "augmentation_name",
        "augmentation_factor",
        "noise_level",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "recall_normal",
        "recall_anomalous",
    ]
    overall = results[results["row_type"] == "overall"].copy()
    lines.append("Overall results:")
    lines.append(format_table(overall.loc[:, display_cols]))
    lines.append("")
    for method in sorted(overall["method"].unique()):
        best_aug = select_best(results, exclude_baseline=True, method=method)
        lines.append(
            f"Best augmented {method}: {best_aug['augmentation_name']} "
            f"(factor={int(best_aug['augmentation_factor'])}, "
            f"noise={best_aug['noise_level']:.2f}, macro F1="
            f"{best_aug['macro_f1']:.3f}, balanced accuracy="
            f"{best_aug['balanced_accuracy']:.3f}, normal recall="
            f"{best_aug['recall_normal']:.3f}, anomalous recall="
            f"{best_aug['recall_anomalous']:.3f})."
        )
        lines.append("Confusion matrix:")
        lines.append(str(best_aug["confusion_matrix"]))
        lines.append("Classification report:")
        lines.append(str(best_aug["classification_report"]))
        lines.append("")
    return "\n".join(lines)


def build_final_summary(
    comparison: pd.DataFrame,
    binary_results: pd.DataFrame,
    binary_predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    three_results: pd.DataFrame,
    three_predictions: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    anomaly_results: pd.DataFrame,
) -> str:
    lines = ["FINAL UNSUPERVISED AUGMENTATION SUMMARY", "=" * 80]
    lines.append(
        "Clustering augmentation was performed label-free and participant-wise. "
        "Labels were used only after clustering to evaluate the discovered clusters."
    )
    lines.append(
        "The anomaly detector was trained only on normal samples from training "
        "participants. Augmentation was applied only to normal training samples "
        "and never to the held-out participant."
    )
    lines.append("")
    lines.append("Baseline vs best augmented comparison:")
    lines.append(format_table(comparison))
    lines.append("")

    by_task = comparison.set_index("task")
    lines.append(
        "- "
        + change_sentence(
            "Binary clustering",
            by_task.loc["binary_clustering", "baseline_macro_f1"],
            by_task.loc["binary_clustering", "best_augmented_macro_f1"],
        )
    )
    lines.append(
        "- "
        + change_sentence(
            "3-class clustering",
            by_task.loc["threeclass_clustering", "baseline_macro_f1"],
            by_task.loc["threeclass_clustering", "best_augmented_macro_f1"],
        )
    )
    lines.append(
        "- "
        + change_sentence(
            "LOSO anomaly detection",
            by_task.loc["anomaly_detection_loso", "baseline_macro_f1"],
            by_task.loc["anomaly_detection_loso", "best_augmented_macro_f1"],
        )
    )

    best_augmented = comparison.sort_values(
        ["best_augmented_macro_f1", "best_augmented_balanced_accuracy"],
        ascending=False,
    ).iloc[0]
    lines.append(
        "- Best augmented setting by macro F1 was "
        f"{best_augmented['best_augmentation_name']} for {best_augmented['task']} "
        f"with {best_augmented['method']}."
    )

    three_baseline_cm = confusion_counts_for_best(
        three_results, three_predictions, THREECLASS_LABELS, exclude_baseline=False
    )
    three_aug_cm = confusion_counts_for_best(
        three_results, three_predictions, THREECLASS_LABELS, exclude_baseline=True
    )
    baseline_mixing = int(three_baseline_cm.sum() - np.trace(three_baseline_cm))
    augmented_mixing = int(three_aug_cm.sum() - np.trace(three_aug_cm))
    lines.append(
        "- 3-class normal/sleepy/stress mixing changed from "
        f"{baseline_mixing} off-diagonal errors to {augmented_mixing}."
    )

    anomaly_baseline = select_best(anomaly_results, exclude_baseline=False)
    anomaly_augmented = select_best(anomaly_results, exclude_baseline=True)
    lines.append(
        "- Normal recall in LOSO anomaly detection changed from "
        f"{anomaly_baseline['recall_normal']:.3f} to "
        f"{anomaly_augmented['recall_normal']:.3f}."
    )

    worse_rows = comparison[
        (comparison["change_macro_f1"] < 0)
        | (comparison["change_balanced_accuracy"] < 0)
    ]
    if worse_rows.empty:
        lines.append(
            "- No task's best augmented setting was worse than the supplied "
            "baseline on macro F1 or balanced accuracy."
        )
    else:
        lines.append(
            "- Some results worsened versus the supplied baseline: "
            + "; ".join(
                f"{row.task} macro delta {row.change_macro_f1:+.3f}, "
                f"balanced accuracy delta {row.change_balanced_accuracy:+.3f}"
                for row in worse_rows.itertuples(index=False)
            )
            + "."
        )

    positive = comparison[comparison["change_macro_f1"] > 0.005]
    slight = comparison[
        (comparison["change_macro_f1"] > 0)
        & (comparison["change_macro_f1"] <= 0.005)
    ]
    if len(positive) >= 2:
        recommendation = (
            "Keep augmentation as an unsupervised robustness experiment, but "
            "present the no-augmentation baselines as the primary comparison "
            "because there are only 4 participants."
        )
    elif not slight.empty:
        recommendation = (
            "Augmentation helps only slightly; keep it as supplementary analysis, "
            "not as a strong claim."
        )
    else:
        recommendation = (
            "Do not keep augmentation as a main unsupervised analysis result."
        )
    lines.append("- Recommendation: " + recommendation)
    lines.append("")
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
    settings = augmentation_settings()

    print("Running binary clustering augmentation experiments...")
    binary_results, binary_predictions = run_clustering_augmentation(
        binary_df,
        FEATURE_COLUMNS,
        BINARY_LABELS,
        2,
        "binary_clustering",
        args.random_state,
        settings,
    )
    binary_results.to_csv(
        results_dir / "binary_clustering_augmentation_results.csv",
        index=False,
    )
    (results_dir / "binary_clustering_augmentation_summary.txt").write_text(
        build_clustering_summary(
            "BINARY CLUSTERING AUGMENTATION SUMMARY",
            binary_results,
            BINARY_LABELS,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Running 3-class clustering augmentation experiments...")
    three_results, three_predictions = run_clustering_augmentation(
        three_df,
        FEATURE_COLUMNS,
        THREECLASS_LABELS,
        3,
        "threeclass_clustering",
        args.random_state,
        settings,
    )
    three_results.to_csv(
        results_dir / "threeclass_clustering_augmentation_results.csv",
        index=False,
    )
    (results_dir / "threeclass_clustering_augmentation_summary.txt").write_text(
        build_clustering_summary(
            "3-CLASS CLUSTERING AUGMENTATION SUMMARY",
            three_results,
            THREECLASS_LABELS,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Running LOSO anomaly detection augmentation experiments...")
    anomaly_results, anomaly_prediction_rows, anomaly_predictions = (
        run_loso_anomaly_augmentation(
            binary_df,
            FEATURE_COLUMNS,
            args.random_state,
            settings,
        )
    )
    anomaly_results.to_csv(
        results_dir / "anomaly_detection_loso_augmentation_results.csv",
        index=False,
    )
    anomaly_prediction_rows.to_csv(
        results_dir / "anomaly_detection_loso_augmentation_predictions.csv",
        index=False,
    )
    (results_dir / "anomaly_detection_loso_augmentation_summary.txt").write_text(
        build_anomaly_summary(anomaly_results) + "\n",
        encoding="utf-8",
    )

    save_best_method_confusions(
        results_dir,
        binary_predictions,
        binary_results,
        three_predictions,
        three_results,
        anomaly_predictions,
        anomaly_results,
    )

    print("Running feature ablation with best augmentation settings...")
    best_binary_setting = setting_from_row(select_best(binary_results, True))
    best_three_setting = setting_from_row(select_best(three_results, True))
    best_anomaly_setting = setting_from_row(select_best(anomaly_results, True))
    ablation_parts = [
        run_clustering_feature_ablation(
            "binary_clustering",
            binary_df,
            BINARY_LABELS,
            2,
            best_binary_setting,
            args.random_state,
        ),
        run_clustering_feature_ablation(
            "threeclass_clustering",
            three_df,
            THREECLASS_LABELS,
            3,
            best_three_setting,
            args.random_state,
        ),
        run_anomaly_feature_ablation(
            binary_df,
            best_anomaly_setting,
            args.random_state,
        ),
    ]
    feature_ablation = pd.concat(ablation_parts, ignore_index=True)
    feature_ablation.to_csv(
        results_dir / "unsupervised_augmentation_feature_ablation.csv",
        index=False,
    )

    comparison = build_baseline_comparison(
        binary_results,
        three_results,
        anomaly_results,
    )
    comparison.to_csv(
        results_dir / "augmentation_vs_baseline_summary.csv",
        index=False,
    )
    final_summary = build_final_summary(
        comparison,
        binary_results,
        binary_predictions,
        three_results,
        three_predictions,
        anomaly_results,
    )
    (results_dir / "final_unsupervised_augmentation_summary.txt").write_text(
        final_summary,
        encoding="utf-8",
    )

    print(final_summary)
    print(f"Saved unsupervised augmentation outputs to: {results_dir}")


if __name__ == "__main__":
    main()
