#!/usr/bin/env python3
"""Unsupervised three-group wellbeing analysis (normal / sleepy / stress).

The clustering models in this script do not use labels while fitting. Labels
are added only afterward to evaluate whether the unsupervised structure
corresponds to the normal, sleepy, and stress states.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Avoid noisy joblib core-count warnings in restricted environments.
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TRUE_LABELS = ["normal", "sleepy", "stress"]

FILES = {
    "normal": "normal.csv",
    "sleepy": "sleepy.csv",
    "stress": "stress.csv",
}

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
        description="Unsupervised three-group (normal/sleepy/stress) clustering analysis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/unsupervised"),
        help="Directory containing normal.csv, sleepy.csv, and stress.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/unsupervised/three_class"),
        help="Output directory for three-group unsupervised results.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for stochastic algorithms.",
    )
    return parser.parse_args()


def load_inputs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data_dir = data_dir.expanduser().resolve()
    missing = [
        filename for filename in FILES.values() if not (data_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Expected {', '.join(FILES.values())} inside {data_dir}; "
            f"missing: {', '.join(missing)}."
        )
    return {name: pd.read_csv(data_dir / filename) for name, filename in FILES.items()}


def validate_features(datasets: Mapping[str, pd.DataFrame]) -> List[str]:
    missing_by_file = {}
    for name, df in datasets.items():
        missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
        if missing:
            missing_by_file[name] = missing
    if not missing_by_file:
        return FEATURE_COLUMNS

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


def coerce_feature_columns(
    datasets: Mapping[str, pd.DataFrame], features: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    converted = {}
    for name, df in datasets.items():
        copy = df.copy()
        for feature in features:
            copy[feature] = pd.to_numeric(copy[feature], errors="coerce")
        converted[name] = copy
    return converted


def candidate_group_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if any(hint in column.lower() for hint in IDENTIFIER_HINTS)
    ]


def inspect_group_candidates(
    datasets: Mapping[str, pd.DataFrame]
) -> Dict[str, Dict[str, List[str]]]:
    inspection: Dict[str, Dict[str, List[str]]] = {}
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
    return len(unique_values) <= max(50, int(len(values) * 0.5))


def infer_group_column(datasets: Mapping[str, pd.DataFrame]) -> Optional[str]:
    common_columns = set.intersection(*(set(df.columns) for df in datasets.values()))
    lower_to_actual = {column.lower(): column for column in common_columns}
    for preferred in GROUP_PRIORITY:
        if preferred in lower_to_actual:
            actual = lower_to_actual[preferred]
            if _column_is_group_like(datasets, actual):
                return actual
    for column in sorted(common_columns):
        if any(hint in column.lower() for hint in IDENTIFIER_HINTS):
            if _column_is_group_like(datasets, column):
                return column
    return None


def build_dataset(
    datasets: Mapping[str, pd.DataFrame], group_col: Optional[str]
) -> pd.DataFrame:
    frames = []
    for name in TRUE_LABELS:
        df = datasets[name].copy()
        if "label" in df.columns:
            df = df.rename(columns={"label": "original_label"})
        else:
            df["original_label"] = np.nan
        df["source_file"] = FILES[name]
        df["true_label"] = name
        df["source_row"] = np.arange(len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if group_col is not None:
        combined = combined.dropna(subset=[group_col]).reset_index(drop=True)
    return combined


def preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_full(df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, Pipeline]:
    pipeline = preprocessing_pipeline()
    X_scaled = pipeline.fit_transform(df.loc[:, features])
    return X_scaled, pipeline


def safe_silhouette(X_scaled: np.ndarray, cluster_ids: Sequence[int]) -> float:
    unique = np.unique(cluster_ids)
    if len(unique) < 2 or len(unique) >= len(cluster_ids):
        return np.nan
    return float(silhouette_score(X_scaled, cluster_ids))


def internal_cluster_metrics(
    X_scaled: np.ndarray, cluster_ids: Sequence[int]
) -> Tuple[float, float, float]:
    """Return (silhouette, davies_bouldin, calinski_harabasz) with safe guards."""
    unique = np.unique(cluster_ids)
    if len(unique) < 2 or len(unique) >= len(cluster_ids):
        return np.nan, np.nan, np.nan
    silhouette = float(silhouette_score(X_scaled, cluster_ids))
    davies_bouldin = float(davies_bouldin_score(X_scaled, cluster_ids))
    calinski_harabasz = float(calinski_harabasz_score(X_scaled, cluster_ids))
    return silhouette, davies_bouldin, calinski_harabasz


def majority_vote_mapping(
    y_true: Sequence[str], cluster_ids: Sequence[int]
) -> Dict[int, str]:
    table = pd.crosstab(pd.Series(cluster_ids, name="cluster"), pd.Series(y_true))
    mapping = {}
    for cluster_id, row in table.iterrows():
        sorted_counts = row.sort_values(ascending=False)
        mapping[int(cluster_id)] = str(sorted_counts.index[0])
    return mapping


def mapped_labels(cluster_ids: Sequence[int], mapping: Mapping[int, str]) -> List[str]:
    return [mapping[int(cluster_id)] for cluster_id in cluster_ids]


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = TRUE_LABELS
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
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
        metrics[f"precision_{label}"] = precision[idx]
        metrics[f"recall_{label}"] = recall[idx]
        metrics[f"f1_{label}"] = f1[idx]
        metrics[f"support_{label}"] = int(support[idx])
    return metrics


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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
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
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_clustering(
    method_name: str,
    cluster_ids: np.ndarray,
    X_scaled: np.ndarray,
    y_true: Sequence[str],
    results_dir: Path,
    table_filename: str,
    cm_filename: str,
) -> Tuple[Dict[str, object], pd.DataFrame, List[str]]:
    mapping = majority_vote_mapping(y_true, cluster_ids)
    y_pred = mapped_labels(cluster_ids, mapping)
    table = pd.crosstab(
        pd.Series(cluster_ids, name="cluster"),
        pd.Series(y_true, name="true_label"),
    )
    table["mapped_label"] = [mapping[int(idx)] for idx in table.index]
    table.to_csv(results_dir / table_filename)

    save_confusion_matrix_png(
        y_true,
        y_pred,
        TRUE_LABELS,
        f"{method_name} Confusion Matrix (majority-vote mapping)",
        results_dir / cm_filename,
    )

    silhouette, davies_bouldin, calinski_harabasz = internal_cluster_metrics(
        X_scaled, cluster_ids
    )
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_true, cluster_ids
    )
    row: Dict[str, object] = {
        "method": method_name,
        "n_clusters": int(len(np.unique(cluster_ids))),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "adjusted_rand_index": adjusted_rand_score(y_true, cluster_ids),
        "normalized_mutual_info": normalized_mutual_info_score(y_true, cluster_ids),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "cluster_mapping": str(mapping),
    }
    row.update(classification_metrics(y_true, y_pred))
    return row, table, y_pred


def clustering_models(random_state: int) -> Dict[str, object]:
    """Return reusable (unfitted) clustering estimators for the 3-group task.

    Single source of truth for the clustering model definitions, so other
    scripts (e.g. the real-time / test-video monitors) can import the exact
    same estimators instead of re-instantiating them.
    """
    return {
        "K-Means": KMeans(n_clusters=3, n_init=50, random_state=random_state),
        "Gaussian Mixture": GaussianMixture(
            n_components=3,
            covariance_type="full",
            random_state=random_state,
        ),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
    }


def fit_clustering_models(
    X_scaled: np.ndarray, random_state: int
) -> Dict[str, np.ndarray]:
    return {
        name: model.fit_predict(X_scaled)
        for name, model in clustering_models(random_state).items()
    }


def save_scatter_plot(
    coords: np.ndarray,
    labels: Sequence[object],
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_array = np.array([str(label) for label in labels])
    unique_labels = sorted(np.unique(labels_array))
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in unique_labels:
        mask = labels_array == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=20,
            alpha=0.78,
            label=label,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Label", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_embeddings(
    X_scaled: np.ndarray, random_state: int
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    tsne_coords = None
    if len(X_scaled) >= 5:
        perplexity = min(30, max(5, (len(X_scaled) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        tsne_coords = tsne.fit_transform(X_scaled)

    umap_coords = None
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        umap_coords = reducer.fit_transform(X_scaled)
    except ImportError:
        umap_coords = None

    return pca_coords, tsne_coords, umap_coords


def save_visualizations(
    X_scaled: np.ndarray,
    y_true: Sequence[str],
    cluster_assignments: Mapping[str, Sequence[int]],
    results_dir: Path,
    random_state: int,
) -> Dict[str, bool]:
    pca_coords, tsne_coords, umap_coords = compute_embeddings(X_scaled, random_state)
    save_scatter_plot(
        pca_coords,
        y_true,
        "PCA colored by true labels",
        results_dir / "pca_true_labels.png",
        "PC1",
        "PC2",
    )
    method_to_filename = {
        "K-Means": "pca_kmeans_clusters.png",
        "Gaussian Mixture": "pca_gmm_clusters.png",
        "Agglomerative": "pca_agglomerative_clusters.png",
    }
    for method_name, clusters in cluster_assignments.items():
        save_scatter_plot(
            pca_coords,
            clusters,
            f"PCA colored by {method_name} clusters",
            results_dir / method_to_filename[method_name],
            "PC1",
            "PC2",
        )
    if tsne_coords is not None:
        save_scatter_plot(
            tsne_coords,
            y_true,
            "t-SNE colored by true labels",
            results_dir / "tsne_true_labels.png",
            "t-SNE 1",
            "t-SNE 2",
        )
        save_scatter_plot(
            tsne_coords,
            cluster_assignments["K-Means"],
            "t-SNE colored by K-Means clusters",
            results_dir / "tsne_kmeans_clusters.png",
            "t-SNE 1",
            "t-SNE 2",
        )
    if umap_coords is not None:
        save_scatter_plot(
            umap_coords,
            y_true,
            "UMAP colored by true labels",
            results_dir / "umap_true_labels.png",
            "UMAP 1",
            "UMAP 2",
        )
        save_scatter_plot(
            umap_coords,
            cluster_assignments["K-Means"],
            "UMAP colored by K-Means clusters",
            results_dir / "umap_kmeans_clusters.png",
            "UMAP 1",
            "UMAP 2",
        )
    return {
        "pca_saved": True,
        "tsne_saved": tsne_coords is not None,
        "umap_saved": umap_coords is not None,
    }


def run_feature_ablation(full_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Re-run K-Means and GMM clustering per feature modality."""
    y_true = full_df["true_label"].tolist()
    rows = []
    for group_name, features in FEATURE_GROUPS.items():
        X_scaled, _ = preprocess_full(full_df, features)
        method_clusters = {
            "K-Means": KMeans(
                n_clusters=3,
                n_init=50,
                random_state=random_state,
            ).fit_predict(X_scaled),
            "Gaussian Mixture": GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=random_state,
            ).fit_predict(X_scaled),
        }
        for method_name, clusters in method_clusters.items():
            mapping = majority_vote_mapping(y_true, clusters)
            y_pred = mapped_labels(clusters, mapping)
            silhouette, davies_bouldin, calinski_harabasz = internal_cluster_metrics(
                X_scaled, clusters
            )
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                y_true, clusters
            )
            row: Dict[str, object] = {
                "feature_group": group_name,
                "method": method_name,
                "n_features": len(features),
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz,
                "adjusted_rand_index": adjusted_rand_score(y_true, clusters),
                "normalized_mutual_info": normalized_mutual_info_score(y_true, clusters),
                "homogeneity": homogeneity,
                "completeness": completeness,
                "v_measure": v_measure,
            }
            row.update(classification_metrics(y_true, y_pred))
            rows.append(row)
    return pd.DataFrame(rows)


def metadata_columns(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    preferred = [
        "source_file",
        "source_row",
        "window_start",
        "video_source",
        "participant",
        "subject",
        "person",
        "source",
        "session",
        "filename",
        "file",
        "name",
        "original_label",
    ]
    columns = []
    for column in preferred:
        if column in df.columns and column not in columns and column not in features:
            columns.append(column)
    return columns


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False)


def rows_per_group_table(df: pd.DataFrame, group_col: Optional[str]) -> Optional[pd.DataFrame]:
    if group_col is None:
        return None
    return (
        pd.crosstab(df[group_col].astype(str), df["true_label"], margins=True)
        .rename_axis(index=group_col, columns="true_label")
        .reset_index()
    )


def dataset_summary_lines(
    datasets: Mapping[str, pd.DataFrame],
    combined: pd.DataFrame,
    features: Sequence[str],
    group_col: Optional[str],
    group_inspection: Mapping[str, Mapping[str, List[str]]],
) -> List[str]:
    lines = []
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    for name in TRUE_LABELS:
        lines.append(f"{FILES[name]} rows: {len(datasets[name])}")
    lines.append(f"total rows: {len(combined)}")
    lines.append("")
    lines.append("Column names:")
    for name, df in datasets.items():
        lines.append(f"- {FILES[name]}: {list(df.columns)}")
    lines.append("")
    lines.append("Selected feature columns:")
    lines.append(", ".join(features))
    lines.append(
        "RMMSD is the column name in the CSV file; scientifically this refers to RMSSD."
    )
    lines.append(
        "Excluded from features: video_source, participant/source/name fields, "
        "filenames, window_start, true_label, original labels, and ID columns."
    )
    lines.append("")
    lines.append("Missing values in selected features:")
    for name, df in datasets.items():
        missing = df.loc[:, features].isna().sum()
        missing_text = ", ".join(
            f"{feature}={int(count)}" for feature, count in missing.items()
        )
        lines.append(f"- {FILES[name]}: {missing_text}")
    lines.append("")
    lines.append(f"True label distribution: {combined['true_label'].value_counts().to_dict()}")
    lines.append("")
    lines.append("Possible participant/source columns and unique values:")
    for name, columns in group_inspection.items():
        if not columns:
            lines.append(f"- {FILES[name]}: none detected")
        for column, values in columns.items():
            lines.append(f"- {FILES[name]} {column}: {values}")
    if group_col is not None:
        lines.append(f"Group column used for reporting: {group_col}")
        lines.append("Rows per participant/group per true label:")
        lines.append(format_table(rows_per_group_table(combined, group_col)))
    else:
        lines.append("No clean participant/source group column was detected.")
    lines.append("")
    return lines


def best_by_macro_balanced(results: pd.DataFrame) -> pd.Series:
    return (
        results.sort_values(
            ["macro_f1", "balanced_accuracy", "accuracy"],
            ascending=False,
        )
        .iloc[0]
        .copy()
    )


def analyze_ablation(ablation: pd.DataFrame, method: str) -> str:
    subset = ablation[ablation["method"] == method].copy()
    if subset.empty:
        return f"{method}: no ablation results."
    all_row = subset[subset["feature_group"] == "all_features"].iloc[0]
    single = subset[subset["feature_group"] != "all_features"]
    best_single = best_by_macro_balanced(single)
    if all_row["macro_f1"] > best_single["macro_f1"]:
        fusion = "all-feature fusion improved over the best single-modality feature set"
    elif all_row["macro_f1"] < best_single["macro_f1"]:
        fusion = "the best single-modality feature set outperformed all-feature fusion"
    else:
        fusion = "all-feature fusion tied the best single-modality feature set"
    hr_row = subset[subset["feature_group"] == "hr_hrv_only"].iloc[0]
    return (
        f"{method}: best feature group was {best_single['feature_group']} "
        f"(macro F1={best_single['macro_f1']:.3f}, balanced accuracy="
        f"{best_single['balanced_accuracy']:.3f}); {fusion}. "
        f"HR/HRV-only macro F1={hr_row['macro_f1']:.3f}."
    )


def build_summary(
    dataset_lines: Sequence[str],
    clustering_results: pd.DataFrame,
    cluster_tables: Mapping[str, pd.DataFrame],
    cluster_predictions: Mapping[str, List[str]],
    ablation_results: pd.DataFrame,
    full_df: pd.DataFrame,
    visualization_status: Mapping[str, bool],
) -> str:
    lines = list(dataset_lines)
    lines.append("SCIENTIFIC NOTE")
    lines.append("=" * 80)
    lines.append(
        "The models grouped the data into three clusters without labels, and "
        "labels were used only afterward to evaluate whether the clusters "
        "corresponded to the normal, sleepy, and stress states."
    )
    lines.append("")

    lines.append("CLUSTERING RESULTS")
    lines.append("=" * 80)
    display_cols = [
        "method",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "v_measure",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    lines.append(format_table(clustering_results.loc[:, display_cols]))
    for name, table in cluster_tables.items():
        lines.append("")
        lines.append(f"{name} cluster vs true-label table:")
        lines.append(table.to_string())
    lines.append("")
    y_true = full_df["true_label"].tolist()
    for method, y_pred in cluster_predictions.items():
        lines.append("")
        lines.append(f"{method} majority-vote classification report:")
        lines.append(
            classification_report(y_true, y_pred, labels=TRUE_LABELS, zero_division=0)
        )
    lines.append("")

    lines.append("FEATURE ABLATION")
    lines.append("=" * 80)
    ablation_cols = [
        "feature_group",
        "method",
        "n_features",
        "silhouette",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "v_measure",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    lines.append(format_table(ablation_results.loc[:, ablation_cols]))
    lines.append("")
    lines.append(analyze_ablation(ablation_results, "K-Means"))
    lines.append(analyze_ablation(ablation_results, "Gaussian Mixture"))
    lines.append("")

    best_cluster = best_by_macro_balanced(clustering_results)
    lines.append("FINAL INTERPRETATION")
    lines.append("=" * 80)
    if best_cluster["macro_f1"] >= 0.8:
        separation = (
            "Clustering showed strong alignment with the normal/sleepy/stress labels."
        )
    elif best_cluster["macro_f1"] >= 0.6:
        separation = (
            "Clustering showed partial alignment with the normal/sleepy/stress labels."
        )
    else:
        separation = (
            "Clustering did not cleanly separate normal, sleepy, and stress samples."
        )
    lines.append(separation)
    lines.append(
        f"Best clustering method: {best_cluster['method']} "
        f"(macro F1={best_cluster['macro_f1']:.3f}, balanced accuracy="
        f"{best_cluster['balanced_accuracy']:.3f}, ARI="
        f"{best_cluster['adjusted_rand_index']:.3f}, NMI="
        f"{best_cluster['normalized_mutual_info']:.3f}, V-measure="
        f"{best_cluster['v_measure']:.3f})."
    )
    lines.append(analyze_ablation(ablation_results, "K-Means"))
    lines.append(analyze_ablation(ablation_results, "Gaussian Mixture"))
    lines.append(
        "This is proof-of-concept only because the dataset contains samples from "
        "only 4 participants; more participants are needed for stronger "
        "generalization claims."
    )
    lines.append(
        "Saved dimensionality-reduction plots: PCA=True, "
        f"t-SNE={visualization_status['tsne_saved']}, "
        f"UMAP={visualization_status['umap_saved']}."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    direct_dir = results_dir / "direct_3class"
    comparison_dir = results_dir / "comparison"
    summary_dir = results_dir / "summary"
    for path in (direct_dir, comparison_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    mpl_config = results_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    raw_datasets = load_inputs(data_dir)
    features = validate_features(raw_datasets)
    datasets = coerce_feature_columns(raw_datasets, features)
    group_inspection = inspect_group_candidates(datasets)
    group_col = infer_group_column(datasets)
    full_df = build_dataset(datasets, group_col)

    y_true = full_df["true_label"].tolist()
    X_scaled, _ = preprocess_full(full_df, features)

    cluster_assignments = fit_clustering_models(X_scaled, args.random_state)
    table_filenames = {
        "K-Means": "kmeans_cluster_label_table.csv",
        "Gaussian Mixture": "gmm_cluster_label_table.csv",
        "Agglomerative": "agglomerative_cluster_label_table.csv",
    }
    cm_filenames = {
        "K-Means": "kmeans_confusion_matrix.png",
        "Gaussian Mixture": "gmm_confusion_matrix.png",
        "Agglomerative": "agglomerative_confusion_matrix.png",
    }

    clustering_rows = []
    cluster_tables = {}
    cluster_predictions = {}
    for method_name, cluster_ids in cluster_assignments.items():
        row, table, y_pred = evaluate_clustering(
            method_name,
            np.asarray(cluster_ids),
            X_scaled,
            y_true,
            direct_dir,
            table_filenames[method_name],
            cm_filenames[method_name],
        )
        clustering_rows.append(row)
        cluster_tables[method_name] = table
        cluster_predictions[method_name] = y_pred

    clustering_results = pd.DataFrame(clustering_rows)
    clustering_results.to_csv(direct_dir / "clustering_3group_results.csv", index=False)

    visualization_status = save_visualizations(
        X_scaled,
        y_true,
        cluster_assignments,
        direct_dir,
        args.random_state,
    )

    ablation_results = run_feature_ablation(full_df, args.random_state)
    ablation_results.to_csv(comparison_dir / "feature_ablation_3group.csv", index=False)

    predictions_df = full_df.loc[:, metadata_columns(full_df, features)].copy()
    predictions_df["true_label"] = full_df["true_label"]
    predictions_df["kmeans_cluster"] = cluster_assignments["K-Means"]
    predictions_df["kmeans_mapped_label"] = cluster_predictions["K-Means"]
    predictions_df["gmm_cluster"] = cluster_assignments["Gaussian Mixture"]
    predictions_df["gmm_mapped_label"] = cluster_predictions["Gaussian Mixture"]
    predictions_df["agglomerative_cluster"] = cluster_assignments["Agglomerative"]
    predictions_df["agglomerative_mapped_label"] = cluster_predictions["Agglomerative"]
    predictions_df.to_csv(
        direct_dir / "unsupervised_3group_predictions.csv", index=False
    )

    summary_lines = dataset_summary_lines(
        datasets,
        full_df,
        features,
        group_col,
        group_inspection,
    )
    summary_text = build_summary(
        summary_lines,
        clustering_results,
        cluster_tables,
        cluster_predictions,
        ablation_results,
        full_df,
        visualization_status,
    )
    (summary_dir / "final_unsupervised_3class_summary.txt").write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    print(summary_text)
    print(f"Saved three-group unsupervised outputs to: {results_dir}")


if __name__ == "__main__":
    main()
