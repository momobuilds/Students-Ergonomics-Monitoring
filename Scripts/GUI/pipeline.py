"""
pipeline.py - Bridge between the Streamlit GUI and the project's Scripts.

This module is the single place that wires the GUI to the existing analysis
code under ``Scripts/``. It does three things:

  1. Puts the feature folders on ``sys.path`` exactly like Scripts/App does, so
     their modules (``config``, ``blink_config``, ``posture_analyzer``, ...) can
     be imported by their bare names.
  2. Loads ``Scripts/App/main.realtime.py`` as a module (its filename contains a
     dot so it is not importable directly) and re-exports the pieces the GUI
     needs: the model catalog, the ``RealtimeModel`` wrapper, the wearable
     ``HeartRateMonitor`` and the feature-assembly helpers. Nothing is
     duplicated -- the GUI uses the very same code paths as the CLI scripts.
  3. Exposes small helpers to lazily import the per-frame analyzers and to push
     the GUI parameter values (window length, posture/blink thresholds) into the
     shared config modules at run time.

Folder map (everything is resolved relative to this file):

    Scripts/GUI          <- this file lives here
    Scripts/App          -> main.realtime.py  (model menu + realtime engine)
    Scripts/Posture      -> config.py, posture_analyzer.py, feature_extractor.py
    Scripts/Blinking     -> blink_config.py, blink_analyzer.py, blink_feature_extractor.py
    Scripts/HeartRate    -> cms50d.py (optional CMS50D wearable)
    Scripts/ML           -> the trained model definitions
    data/supervised      -> the labelled training windows
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

# =============================================================================
# PROJECT PATHS
# =============================================================================

GUI_DIR = Path(__file__).resolve().parent      # Scripts/GUI
SCRIPTS_DIR = GUI_DIR.parent                    # Scripts
PROJECT_ROOT = SCRIPTS_DIR.parent               # project root

APP_DIR = SCRIPTS_DIR / "App"
POSTURE_DIR = SCRIPTS_DIR / "Posture"
BLINKING_DIR = SCRIPTS_DIR / "Blinking"
HEARTRATE_DIR = SCRIPTS_DIR / "HeartRate"
ML_DIR = SCRIPTS_DIR / "ML"

SUPERVISED_DIR = PROJECT_ROOT / "data" / "supervised"
TEST_VIDEO_DIR = PROJECT_ROOT / "Test Video"
VIDEO_DIR = PROJECT_ROOT / "Video"
RESULTS_DIR = PROJECT_ROOT / "results"

METRIC_PRIORITY = ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"]

# Same import wiring as Scripts/App: the sub-modules import their config as a
# bare name, so their folders must be importable directly. Order matters --
# Posture must come before others so ``import config`` resolves to the posture
# config (Blinking uses the explicit ``blink_config`` alias, so no clash).
for _module_dir in (APP_DIR, POSTURE_DIR, BLINKING_DIR, HEARTRATE_DIR, ML_DIR):
    _path = str(_module_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# =============================================================================
# LOAD THE REAL-TIME ENGINE (Scripts/App/main.realtime.py)
# =============================================================================

def _load_module(name: str, path: Path):
    """Import a .py file by path (needed because 'main.realtime.py' has a dot)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# This pulls in: model_catalog, RealtimeModel, HeartRateMonitor,
# assemble_feature_row, landmarks_from_results, FEATURE_COLUMNS,
# POSTURE_FEATURES, BLINK_FEATURES, TASK_FILES, RANDOM_STATE, ...
# No heavy work runs at import time (cv2/mediapipe/ML are imported lazily).
engine = _load_module("gui_realtime_engine", APP_DIR / "main.realtime.py")

# Re-export the names the GUI uses, so app.py never reaches into Scripts itself.
RealtimeModel = engine.RealtimeModel
HeartRateMonitor = engine.HeartRateMonitor
model_catalog = engine.model_catalog
assemble_feature_row = engine.assemble_feature_row
landmarks_from_results = engine.landmarks_from_results
FEATURE_COLUMNS = engine.FEATURE_COLUMNS
POSTURE_FEATURES = engine.POSTURE_FEATURES
BLINK_FEATURES = engine.BLINK_FEATURES
# Heart is whatever is left after posture+blink (HR, RMMSD) -- derived so it
# stays in sync with the engine's FEATURE_COLUMNS definition.
HEART_FEATURES = [
    column
    for column in FEATURE_COLUMNS
    if column not in set(POSTURE_FEATURES) | set(BLINK_FEATURES)
]

# Canonical sensor -> feature-group map, mirrored from the offline sensor
# ablation experiment so the GUI can train on any sensor subset.
SENSOR_FEATURES = {
    "posture": list(POSTURE_FEATURES),
    "blink": list(BLINK_FEATURES),
    "heart": list(HEART_FEATURES),
}
ALL_SENSORS = ["posture", "blink", "heart"]
SENSOR_LABELS = {"posture": "Posture", "blink": "Blinking", "heart": "Heart"}


def resolve_feature_columns(sensors=None) -> list[str]:
    """Map a list of active sensors to feature columns in canonical order.

    Passing ``None`` or an empty/unknown selection falls back to every sensor,
    so callers that don't care about subsets keep the full 18-feature model.
    """
    if not sensors:
        return list(FEATURE_COLUMNS)
    kept: set[str] = set()
    for sensor in sensors:
        kept.update(SENSOR_FEATURES.get(sensor, []))
    columns = [column for column in FEATURE_COLUMNS if column in kept]
    return columns or list(FEATURE_COLUMNS)


# =============================================================================
# MODEL HELPERS
# =============================================================================

def available_models(task: str, learning: str):
    """Return the list of model names available for a task/learning combo."""
    return list(model_catalog(task, learning).keys())


def open_camera(preferred_index: int = 0, max_index: int = 4):
    """Open the preferred webcam, then fall back to other local camera indices."""
    import cv2

    indices = [preferred_index, *[i for i in range(max_index + 1) if i != preferred_index]]
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    for index in indices:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap, index
        cap.release()
    return None, None


def model_menu_options(task: str, learning: str) -> list[str]:
    """Return GUI model choices with the automatic best-model option first."""
    return ["Best Model", *available_models(task, learning)]


def resolve_model_selection(
    task: str,
    learning: str,
    menu_choice: str,
    hierarchical: bool = False,
    use_augmentation: bool = False,
) -> dict:
    """Resolve a GUI menu choice to model names, metrics and confusion matrix."""
    if menu_choice == "Best Model":
        selection = select_best_model(
            task,
            learning,
            hierarchical=hierarchical,
            use_augmentation=use_augmentation,
        )
        selection["use_best_model"] = True
        return selection

    selection = lookup_model_metrics(
        task,
        learning,
        menu_choice,
        hierarchical=hierarchical,
        use_augmentation=use_augmentation,
    )
    selection = dict(selection)
    selection.setdefault("model_name", menu_choice)
    selection.setdefault("binary_model", menu_choice)
    selection.setdefault("subtype_model", menu_choice)
    selection.setdefault("metrics", {})
    selection.setdefault("results_source", "")
    selection.setdefault("augmentation_name", "")
    selection.setdefault("augmentation_factor", None)
    selection.setdefault("noise_level", None)
    selection.setdefault(
        "confusion_matrix_path",
        _confusion_matrix_path(
            task,
            learning,
            menu_choice,
            hierarchical,
            use_augmentation,
        ),
    )
    selection["use_best_model"] = False
    return selection


def _bool_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _sort_by_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = frame.copy()
    metrics = [column for column in METRIC_PRIORITY if column in sortable.columns]
    if not metrics:
        return sortable
    for column in metrics:
        sortable[column] = pd.to_numeric(sortable[column], errors="coerce").fillna(-np.inf)
    return sortable.sort_values(metrics, ascending=False)


def _metrics_from_row(row: pd.Series) -> dict:
    metrics = {}
    for column in METRIC_PRIORITY:
        if column in row and pd.notna(row[column]):
            metrics[column] = float(row[column])
    return metrics


def _relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _candidate_rows(
    path: Path,
    available: set[str],
    use_augmentation: bool | None = None,
    task_value: str | None = None,
    hierarchical: bool = False,
) -> pd.DataFrame:
    """Read a results CSV and return rows usable for GUI model selection."""
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()

    if "row_type" in frame.columns:
        frame = frame[frame["row_type"].astype(str).str.lower() == "overall"]
    if task_value is not None and "task" in frame.columns:
        frame = frame[frame["task"].astype(str) == task_value]
    if use_augmentation is not None and "is_baseline" in frame.columns:
        is_baseline = _bool_values(frame["is_baseline"])
        frame = frame[~is_baseline] if use_augmentation else frame[is_baseline]

    if hierarchical:
        if {"binary_model", "subtype_model"}.issubset(frame.columns):
            frame = frame[
                frame["binary_model"].astype(str).isin(available)
                & frame["subtype_model"].astype(str).isin(available)
            ]
        return frame

    if "model" not in frame.columns:
        if "method" in frame.columns:
            frame = frame.rename(columns={"method": "model"})
        elif "best_model" in frame.columns:
            frame = frame.rename(columns={"best_model": "model"})
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str).isin(available)]
    return frame


def _selection_from_row(row: pd.Series, path: Path, hierarchical: bool) -> dict:
    if hierarchical:
        binary_model = str(row.get("binary_model", "Random Forest"))
        subtype_model = str(row.get("subtype_model", binary_model))
        model_name = str(row.get("model", f"{binary_model} -> {subtype_model}"))
    else:
        model_name = str(row.get("model", row.get("method", row.get("best_model", ""))))
        binary_model = model_name
        subtype_model = model_name

    return {
        "model_name": model_name,
        "binary_model": binary_model,
        "subtype_model": subtype_model,
        "metrics": _metrics_from_row(row),
        "results_source": _relative_source(path),
        "augmentation_name": str(row.get("augmentation_name", "")),
        "augmentation_factor": row.get("augmentation_factor", None),
        "noise_level": row.get("noise_level", None),
    }


def _fallback_selection(task: str, learning: str, hierarchical: bool) -> dict:
    if hierarchical:
        return {
            "model_name": "Random Forest -> Random Forest",
            "binary_model": "Random Forest",
            "subtype_model": "Random Forest",
            "metrics": {},
            "results_source": "fallback",
            "augmentation_name": "",
            "augmentation_factor": None,
            "noise_level": None,
        }

    fallback = "Random Forest"
    if learning == "unsupervised":
        fallback = "Gaussian Mixture"
    if task == "subtype":
        fallback = "Random Forest"
    return {
        "model_name": fallback,
        "binary_model": fallback,
        "subtype_model": fallback,
        "metrics": {},
        "results_source": "fallback",
        "augmentation_name": "",
        "augmentation_factor": None,
        "noise_level": None,
    }


def _result_paths(task: str, learning: str, hierarchical: bool, use_augmentation: bool):
    if learning == "supervised":
        if hierarchical:
            filename = "hierarchical_augmentation_results.csv"
            baseline = (
                RESULTS_DIR
                / "supervised"
                / "baseline"
                / "hierarchical"
                / "hierarchical_results.csv"
            )
            augmented = (
                RESULTS_DIR
                / "supervised"
                / "augmentation"
                / "hierarchical"
                / filename
            )
            return [(augmented if use_augmentation else baseline, "hierarchical")]

        task_key = "direct_3class" if task == "3class" else task
        baseline = (
            RESULTS_DIR
            / "supervised"
            / "baseline"
            / task_key
            / f"{task_key}_results.csv"
        )
        augmented = (
            RESULTS_DIR
            / "supervised"
            / "augmentation"
            / task_key
            / f"{task_key}_augmentation_results.csv"
        )
        return [(augmented if use_augmentation else baseline, task_key)]

    if task == "3class":
        baseline = (
            RESULTS_DIR
            / "unsupervised"
            / "three_class"
            / "direct_3class"
            / "clustering_3group_results.csv"
        )
        augmented = (
            RESULTS_DIR
            / "unsupervised"
            / "augmentation"
            / "direct_3class"
            / "threeclass_clustering_augmentation_results.csv"
        )
        return [(augmented if use_augmentation else baseline, None)]

    if use_augmentation:
        return [
            (
                RESULTS_DIR
                / "unsupervised"
                / "augmentation"
                / "binary"
                / "binary_clustering_augmentation_results.csv",
                None,
            ),
            (
                RESULTS_DIR
                / "unsupervised"
                / "augmentation"
                / "binary"
                / "anomaly_detection_loso_augmentation_results.csv",
                "anomaly_detection_loso",
            ),
        ]

    return [
        (
            RESULTS_DIR
            / "unsupervised"
            / "baseline"
            / "binary"
            / "clustering_results.csv",
            None,
        ),
        (
            RESULTS_DIR
            / "unsupervised"
            / "binary_loso"
            / "binary"
            / "anomaly_detection_loso_results.csv",
            None,
        ),
    ]


def _confusion_matrix_path(
    task: str,
    learning: str,
    model_name: str,
    hierarchical: bool,
    use_augmentation: bool,
) -> str | None:
    if learning == "supervised":
        if hierarchical:
            path = (
                RESULTS_DIR
                / "supervised"
                / "augmentation"
                / "hierarchical"
                / ("hierarchical_best_augmented_confusion_matrix.png"
                   if use_augmentation else "hierarchical_baseline_confusion_matrix.png")
            )
        else:
            task_key = "direct_3class" if task == "3class" else task
            if use_augmentation:
                path = (
                    RESULTS_DIR
                    / "supervised"
                    / "augmentation"
                    / task_key
                    / f"{task_key}_best_augmented_confusion_matrix.png"
                )
            else:
                path = (
                    RESULTS_DIR
                    / "supervised"
                    / "baseline"
                    / task_key
                    / f"{task_key}_confusion_matrix.png"
                )
        return str(path) if path.exists() else None

    slug_by_model = {
        "K-Means": "kmeans",
        "Gaussian Mixture": "gmm",
        "IsolationForest": "isolation_forest",
        "OneClassSVM": "oneclass_svm",
        "LocalOutlierFactor": "lof",
    }
    slug = slug_by_model.get(model_name)
    if slug is None:
        return None

    if use_augmentation:
        if task == "3class":
            path = (
                RESULTS_DIR
                / "unsupervised"
                / "augmentation"
                / "direct_3class"
                / f"threeclass_{slug}_best_augmented_confusion_matrix.png"
            )
        elif model_name in {"K-Means", "Gaussian Mixture"}:
            path = (
                RESULTS_DIR
                / "unsupervised"
                / "augmentation"
                / "binary"
                / f"binary_{slug}_best_augmented_confusion_matrix.png"
            )
        else:
            path = (
                RESULTS_DIR
                / "unsupervised"
                / "augmentation"
                / "binary"
                / f"{slug}_loso_best_augmented_confusion_matrix.png"
            )
    elif task == "3class":
        path = (
            RESULTS_DIR
            / "unsupervised"
            / "three_class"
            / "direct_3class"
            / f"{slug}_confusion_matrix.png"
        )
    elif model_name in {"K-Means", "Gaussian Mixture"}:
        # Binary clustering baselines: results/unsupervised/baseline/binary/
        # {kmeans,gmm}_confusion_matrix.png
        path = (
            RESULTS_DIR
            / "unsupervised"
            / "baseline"
            / "binary"
            / f"{slug}_confusion_matrix.png"
        )
    else:
        # Binary anomaly-detection baselines use the full estimator names.
        anomaly_names = {
            "IsolationForest": "isolation_forest",
            "OneClassSVM": "oneclass_svm",
            "LocalOutlierFactor": "local_outlier_factor",
        }
        anomaly_slug = anomaly_names.get(model_name)
        if anomaly_slug is None:
            return None
        path = (
            RESULTS_DIR
            / "unsupervised"
            / "baseline"
            / "binary"
            / f"{anomaly_slug}_confusion_matrix.png"
        )

    return str(path) if path.exists() else None


def select_best_model(
    task: str,
    learning: str,
    hierarchical: bool = False,
    use_augmentation: bool = False,
) -> dict:
    """Choose a model from saved metrics, prioritising macro F1 and balanced accuracy."""
    available = set(available_models("binary" if hierarchical else task, learning))
    candidates = []
    for path, task_value in _result_paths(task, learning, hierarchical, use_augmentation):
        rows = _candidate_rows(
            path,
            available,
            use_augmentation=use_augmentation if "augmentation" in path.name else None,
            task_value=task_value,
            hierarchical=hierarchical,
        )
        if rows.empty:
            continue
        rows = rows.copy()
        rows["__source_path__"] = str(path)
        candidates.append(rows)

    if not candidates:
        selection = _fallback_selection(task, learning, hierarchical)
    else:
        best = _sort_by_metrics(pd.concat(candidates, ignore_index=True)).iloc[0]
        selection = _selection_from_row(best, Path(best["__source_path__"]), hierarchical)

    selection["confusion_matrix_path"] = _confusion_matrix_path(
        task,
        learning,
        selection["model_name"] if not hierarchical else selection["binary_model"],
        hierarchical,
        use_augmentation,
    )
    return selection


def lookup_model_metrics(
    task: str,
    learning: str,
    model_name: str,
    hierarchical: bool = False,
    use_augmentation: bool = False,
) -> dict:
    """Return the best saved metrics for a manually selected model, if present."""
    available = set(available_models("binary" if hierarchical else task, learning))
    if model_name not in available:
        return {}

    candidates = []
    for path, task_value in _result_paths(task, learning, hierarchical, use_augmentation):
        rows = _candidate_rows(
            path,
            available,
            use_augmentation=use_augmentation if "augmentation" in path.name else None,
            task_value=task_value,
            hierarchical=hierarchical,
        )
        if rows.empty:
            continue
        if hierarchical:
            rows = rows[
                (rows["binary_model"].astype(str) == model_name)
                & (rows["subtype_model"].astype(str) == model_name)
            ]
        else:
            rows = rows[rows["model"].astype(str) == model_name]
        if rows.empty:
            continue
        rows = rows.copy()
        rows["__source_path__"] = str(path)
        candidates.append(rows)

    if not candidates:
        return {}
    best = _sort_by_metrics(pd.concat(candidates, ignore_index=True)).iloc[0]
    selection = _selection_from_row(best, Path(best["__source_path__"]), hierarchical)
    selection["confusion_matrix_path"] = _confusion_matrix_path(
        task,
        learning,
        selection["model_name"] if not hierarchical else selection["binary_model"],
        hierarchical,
        use_augmentation,
    )
    return selection


def _source_feature_columns(data: pd.DataFrame) -> list[str]:
    source_cols = []
    for feature in FEATURE_COLUMNS:
        if feature == "RMMSD" and "RMMSD" not in data.columns and "RMSSD" in data.columns:
            source_cols.append("RMSSD")
        else:
            source_cols.append(feature)
    missing = [column for column in source_cols if column not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing feature columns: {missing}")
    return source_cols


def _load_dataset_with_groups(task: str, feature_columns=None):
    """Load canonical features, labels, and the video_source participant column.

    ``feature_columns`` optionally restricts the returned X to a sensor subset
    (in canonical order); the supervised CSVs always carry every column, so the
    subset is just a selection, not a different file.
    """
    frames = []
    for filename, label in engine.TASK_FILES[task]:
        csv_path = SUPERVISED_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Training file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["__label__"] = label
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if "video_source" not in data.columns:
        raise ValueError(
            "The GUI augmentation pipeline requires a video_source participant column."
        )

    source_cols = _source_feature_columns(data)
    X = data[source_cols].apply(pd.to_numeric, errors="coerce")
    X.columns = FEATURE_COLUMNS
    if feature_columns:
        X = X.loc[:, [c for c in FEATURE_COLUMNS if c in set(feature_columns)]]
    y = data["__label__"].astype(str).reset_index(drop=True)
    groups = data["video_source"].astype(str).reset_index(drop=True)
    return X.reset_index(drop=True), y, groups


def _augment_training_data_by_group(
    X_train: pd.DataFrame,
    groups_train: pd.Series,
    feature_columns,
    augmentation_factor: int = 1,
    noise_level: float = 0.02,
    random_state: int = 42,
):
    """Label-free Gaussian jitter within each participant only."""
    X_original = X_train.loc[:, feature_columns].copy().reset_index(drop=True)
    groups_original = pd.Series(groups_train, name="_group").astype(str).reset_index(drop=True)

    X_with_flags = X_original.copy()
    X_with_flags["is_augmented"] = False
    if augmentation_factor <= 0 or noise_level <= 0:
        return X_with_flags, groups_original

    lower_bounds = X_original.quantile(0.01).fillna(X_original.min())
    upper_bounds = X_original.quantile(0.99).fillna(X_original.max())
    training_frame = X_original.copy()
    training_frame["_group"] = groups_original

    rng = np.random.default_rng(random_state)
    X_parts = [X_with_flags]
    group_parts = [groups_original]

    for group_value, group_frame in training_frame.groupby("_group", sort=True):
        subset_features = group_frame.loc[:, feature_columns].reset_index(drop=True)
        if subset_features.empty:
            continue
        feature_std = subset_features.std(axis=0, ddof=0).fillna(0.0)
        noise_scale = (noise_level * feature_std).to_numpy(dtype=float)

        for _ in range(augmentation_factor):
            noise = rng.normal(0.0, noise_scale, size=subset_features.shape)
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
            group_parts.append(pd.Series([group_value] * len(noisy_features), name="_group"))

    return pd.concat(X_parts, ignore_index=True), pd.concat(group_parts, ignore_index=True)


def _fit_supervised_model(
    model: "RealtimeModel",
    augmentation_factor: int,
    noise_level: float,
) -> "RealtimeModel":
    from wellbeing_hierarchical_classification import augment_training_data_groupwise

    feature_columns = model.feature_columns
    X, y, groups = _load_dataset_with_groups(model.task, feature_columns)
    X_fit, y_fit, _ = augment_training_data_groupwise(
        X,
        y,
        groups,
        feature_columns,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        random_state=engine.RANDOM_STATE,
    )
    model.estimator.fit(X_fit.loc[:, feature_columns], y_fit)
    model.classes = list(model.estimator.classes_)
    model.training_metadata = {
        "train_original_rows": int(len(X)),
        "train_augmented_rows": int(len(X_fit) - len(X)),
        "train_rows": int(len(X_fit)),
        "augmentation_used": True,
        "augmentation_factor": augmentation_factor,
        "noise_level": noise_level,
    }
    return model


def _unsupervised_preprocessing_pipeline():
    from unsupervised_wellbeing_analysis import preprocessing_pipeline

    return preprocessing_pipeline()


def _fit_unsupervised_model(
    model: "RealtimeModel",
    augmentation_factor: int,
    noise_level: float,
) -> "RealtimeModel":
    feature_columns = model.feature_columns
    X, y, groups = _load_dataset_with_groups(model.task, feature_columns)

    if model.kind == "anomaly":
        normal_mask = y == "normal"
        X_normal = X.loc[normal_mask].reset_index(drop=True)
        groups_normal = groups.loc[normal_mask].reset_index(drop=True)
        X_fit, _ = _augment_training_data_by_group(
            X_normal,
            groups_normal,
            feature_columns,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level,
            random_state=engine.RANDOM_STATE,
        )
        model.preprocessor = _unsupervised_preprocessing_pipeline()
        model.preprocessor.fit(X_normal.loc[:, feature_columns])
        X_scaled = model.preprocessor.transform(X_fit.loc[:, feature_columns])
        model.estimator.fit(X_scaled)
        model.classes = ["normal", "abnormal"]

    elif model.kind == "cluster":
        X_fit, _ = _augment_training_data_by_group(
            X,
            groups,
            feature_columns,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level,
            random_state=engine.RANDOM_STATE,
        )
        model.preprocessor = _unsupervised_preprocessing_pipeline()
        X_scaled = model.preprocessor.fit_transform(X_fit.loc[:, feature_columns])
        model.estimator.fit(X_scaled)

        original_mask = ~X_fit["is_augmented"].astype(bool).to_numpy()
        cluster_ids = model.estimator.predict(X_scaled[original_mask])
        table = pd.crosstab(
            pd.Series(cluster_ids, name="cluster"),
            pd.Series(list(y), name="label"),
        )
        model.cluster_map = {int(c): str(table.loc[c].idxmax()) for c in table.index}
        model.classes = sorted(set(model.cluster_map.values()))

    else:
        raise ValueError(f"Unsupported unsupervised model kind: {model.kind}")

    model.training_metadata = {
        "train_original_rows": int(len(X)),
        "train_augmented_rows": int(augmentation_factor * len(X))
        if model.kind == "cluster"
        else int(augmentation_factor * int((y == "normal").sum())),
        "augmentation_used": True,
        "augmentation_factor": augmentation_factor,
        "noise_level": noise_level,
        "preprocessing": "SimpleImputer(strategy='median') + StandardScaler",
    }
    return model


def build_and_train_model(
    task: str,
    learning: str,
    name: str,
    use_augmentation: bool = False,
    augmentation_factor: int = 1,
    noise_level: float = 0.02,
    feature_columns=None,
) -> "RealtimeModel":
    """Instantiate the chosen estimator and fit it on data/supervised.

    ``feature_columns`` optionally restricts training (and later inference) to a
    sensor subset; ``None`` trains on every sensor as before.

    Returns a ready-to-use RealtimeModel (same object the CLI scripts use).
    """
    catalog = model_catalog(task, learning)
    if name not in catalog:
        raise KeyError(f"Model '{name}' not available for {task}/{learning}. "
                       f"Choices: {list(catalog)}")
    kind, estimator = catalog[name]
    model = RealtimeModel(
        task, learning, name, kind, clone(estimator), feature_columns=feature_columns
    )

    if use_augmentation:
        if learning == "supervised":
            _fit_supervised_model(model, augmentation_factor, noise_level)
        else:
            _fit_unsupervised_model(model, augmentation_factor, noise_level)
    else:
        model.fit()
        model.training_metadata = {
            "augmentation_used": False,
            "augmentation_factor": 0,
            "noise_level": 0.0,
        }
        if learning == "unsupervised":
            model.training_metadata["preprocessing"] = (
                "SimpleImputer(strategy='median') + StandardScaler"
            )
    model.training_metadata["feature_columns"] = list(model.feature_columns)
    model.training_metadata["n_features"] = len(model.feature_columns)
    return model


# --- Hierarchical (binary -> subtype) -----------------------------------------

# Register a "subtype" supervised task (sleepy vs stress) on the engine so the
# existing load_dataset / RealtimeModel machinery can train the second stage
# exactly like the CLI hierarchical pipeline (Scripts/ML).
engine.TASK_FILES.setdefault(
    "subtype", [("sleepy.csv", "sleepy"), ("stress.csv", "stress")]
)


class HierarchicalModel:
    """Two-stage classifier with the same interface as RealtimeModel.

    Stage 1 (binary):  normal vs abnormal.
    Stage 2 (subtype): if abnormal, sleepy vs stress.

    So a single ``predict_row`` returns normal / sleepy / stress -- the binary
    decision refined into its subtype, like Scripts/ML's hierarchical task.
    """

    def __init__(self, binary_model, subtype_model, name):
        self.binary = binary_model
        self.subtype = subtype_model
        self.name = name
        self.task = "hierarchical"
        self.learning = "supervised"
        self.classes = ["normal", "sleepy", "stress"]

    def describe(self):
        return f"{self.name} (hierarchical: binary -> sleepy/stress subtype)"

    def predict_row(self, feature_row):
        binary_result = self.binary.predict_row(feature_row)
        if binary_result["label"] in ("abnormal", "anomalous"):
            subtype_result = self.subtype.predict_row(feature_row)
            return {
                **subtype_result,
                "binary_prediction": binary_result["label"],
                "subtype_prediction": subtype_result["label"],
                "binary_proba": binary_result.get("proba", {}),
                "subtype_proba": subtype_result.get("proba", {}),
            }
        return {
            **binary_result,
            "binary_prediction": binary_result["label"],
            "subtype_prediction": "",
            "binary_proba": binary_result.get("proba", {}),
            "subtype_proba": {},
        }


def build_hierarchical_model(
    binary_name: str,
    subtype_name: str | None = None,
    use_augmentation: bool = False,
    augmentation_factor: int = 1,
    noise_level: float = 0.02,
    feature_columns=None,
) -> "HierarchicalModel":
    """Train both stages (binary + sleepy/stress) of the hierarchical model."""
    subtype_name = subtype_name or binary_name
    binary_model = build_and_train_model(
        "binary",
        "supervised",
        binary_name,
        use_augmentation=use_augmentation,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        feature_columns=feature_columns,
    )
    subtype_model = build_and_train_model(
        "subtype",
        "supervised",
        subtype_name,
        use_augmentation=use_augmentation,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        feature_columns=feature_columns,
    )

    display_name = f"{binary_name} -> {subtype_name}"
    model = HierarchicalModel(binary_model, subtype_model, display_name)
    model.training_metadata = {
        "augmentation_used": use_augmentation,
        "augmentation_factor": augmentation_factor if use_augmentation else 0,
        "noise_level": noise_level if use_augmentation else 0.0,
        "binary_model": binary_name,
        "subtype_model": subtype_name,
        "binary_training": getattr(binary_model, "training_metadata", {}),
        "subtype_training": getattr(subtype_model, "training_metadata", {}),
    }
    return model


# =============================================================================
# LIVE CONFUSION MATRIX -- LOSO-evaluate the EXACT trained configuration
# =============================================================================
#
# The offline results under results/ were computed on all three sensors. When the
# GUI trains on a sensor subset (or any task/learning/augmentation combo), we
# recompute a Leave-One-Subject-Out confusion matrix here, on the very same
# feature columns and augmentation the model was trained with, and render that
# instead of a precomputed PNG. So the matrix is always the right one.

GUI_CM_DIR = RESULTS_DIR / "gui" / "confusion_matrix"

# Labels per task (the order the confusion matrix uses).
_TASK_LABELS = {
    "binary": ["normal", "abnormal"],
    "subtype": ["sleepy", "stress"],
    "3class": ["normal", "sleepy", "stress"],
    "hierarchical": ["normal", "sleepy", "stress"],
}

_GUI_METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")


def _augment_supervised_fold(X_train, y_train, groups_train, feature_columns,
                             augmentation_factor, noise_level):
    """Group-wise augmentation of a single training fold (labels preserved)."""
    from wellbeing_hierarchical_classification import augment_training_data_groupwise

    X_fit, y_fit, _ = augment_training_data_groupwise(
        X_train.reset_index(drop=True),
        pd.Series(y_train).reset_index(drop=True),
        pd.Series(groups_train).reset_index(drop=True),
        feature_columns,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        random_state=engine.RANDOM_STATE,
    )
    return X_fit.loc[:, feature_columns], y_fit


def _loso_flat(task, kind, estimator, X, y, groups, feature_columns, labels,
               use_augmentation, augmentation_factor, noise_level):
    """Return (y_true, y_pred) over all LOSO folds for a flat (non-hierarchical) task."""
    from sklearn.model_selection import LeaveOneGroupOut

    feature_columns = list(feature_columns)
    X = X.loc[:, feature_columns].reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    majority_label = y.value_counts().idxmax()

    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        y_true = y.iloc[test_idx].tolist()

        if kind == "classifier":
            if use_augmentation and augmentation_factor > 0 and noise_level > 0:
                X_fit, y_fit = _augment_supervised_fold(
                    X_train, y_train, groups_train, feature_columns,
                    augmentation_factor, noise_level,
                )
            else:
                X_fit, y_fit = X_train, y_train
            est = clone(estimator)
            est.fit(X_fit, y_fit)
            y_pred = [str(p) for p in est.predict(X_test)]

        elif kind == "cluster":
            if use_augmentation and augmentation_factor > 0 and noise_level > 0:
                X_aug, _ = _augment_training_data_by_group(
                    X_train, groups_train, feature_columns,
                    augmentation_factor=augmentation_factor,
                    noise_level=noise_level, random_state=engine.RANDOM_STATE,
                )
                X_aug = X_aug.loc[:, feature_columns]
            else:
                X_aug = X_train
            pre = _unsupervised_preprocessing_pipeline()
            est = clone(estimator)
            est.fit(pre.fit_transform(X_aug))
            # Map clusters -> labels by majority vote on the ORIGINAL train rows.
            train_clusters = est.predict(pre.transform(X_train))
            table = pd.crosstab(pd.Series(train_clusters, name="cluster"),
                                pd.Series(y_train.tolist(), name="label"))
            cmap = {int(c): str(table.loc[c].idxmax()) for c in table.index}
            test_clusters = est.predict(pre.transform(X_test))
            y_pred = [cmap.get(int(c), majority_label) for c in test_clusters]

        elif kind == "anomaly":
            normal_mask = (y_train == "normal").to_numpy()
            X_normal = X_train.loc[normal_mask]
            groups_normal = groups_train.loc[normal_mask]
            if use_augmentation and augmentation_factor > 0 and noise_level > 0:
                X_aug, _ = _augment_training_data_by_group(
                    X_normal, groups_normal, feature_columns,
                    augmentation_factor=augmentation_factor,
                    noise_level=noise_level, random_state=engine.RANDOM_STATE,
                )
                X_aug = X_aug.loc[:, feature_columns]
            else:
                X_aug = X_normal
            pre = _unsupervised_preprocessing_pipeline()
            pre.fit(X_normal)
            est = clone(estimator)
            est.fit(pre.transform(X_aug))
            raw = est.predict(pre.transform(X_test))
            y_pred = ["normal" if r == 1 else "abnormal" for r in raw]

        else:
            raise ValueError(f"Unsupported model kind for evaluation: {kind}")

        y_true_all.extend(str(t) for t in y_true)
        y_pred_all.extend(str(p) for p in y_pred)

    return y_true_all, y_pred_all


def _loso_hierarchical(binary_estimator, subtype_estimator, feature_columns,
                       use_augmentation, augmentation_factor, noise_level):
    """LOSO for the binary -> subtype hierarchy, returns (y_true, y_pred) as 3 classes."""
    from sklearn.model_selection import LeaveOneGroupOut  # noqa: F401 (kept for symmetry)

    feature_columns = list(feature_columns)
    Xb, yb, gb = _load_dataset_with_groups("binary", feature_columns)
    Xs, ys, gs = _load_dataset_with_groups("subtype", feature_columns)
    Xf, yf, gf = _load_dataset_with_groups("3class", feature_columns)
    gb, gs, gf = (g.astype(str) for g in (gb, gs, gf))

    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for heldout in sorted(gf.unique()):
        b_tr = gb != heldout
        s_tr = gs != heldout
        te = gf == heldout

        if use_augmentation and augmentation_factor > 0 and noise_level > 0:
            Xb_fit, yb_fit = _augment_supervised_fold(
                Xb.loc[b_tr], yb.loc[b_tr], gb.loc[b_tr], feature_columns,
                augmentation_factor, noise_level)
            Xs_fit, ys_fit = _augment_supervised_fold(
                Xs.loc[s_tr], ys.loc[s_tr], gs.loc[s_tr], feature_columns,
                augmentation_factor, noise_level)
        else:
            Xb_fit, yb_fit = Xb.loc[b_tr, feature_columns], yb.loc[b_tr]
            Xs_fit, ys_fit = Xs.loc[s_tr, feature_columns], ys.loc[s_tr]

        b_model = clone(binary_estimator)
        s_model = clone(subtype_estimator)
        b_model.fit(Xb_fit, yb_fit)
        s_model.fit(Xs_fit, ys_fit)

        X_test = Xf.loc[te, feature_columns]
        binary_pred = b_model.predict(X_test)
        final_pred = np.array(
            ["normal" if p == "normal" else "" for p in binary_pred], dtype=object
        )
        abnormal = binary_pred == "abnormal"
        if abnormal.any():
            final_pred[abnormal] = s_model.predict(X_test.loc[abnormal])

        y_true_all.extend(str(t) for t in yf.loc[te].tolist())
        y_pred_all.extend(str(p) for p in final_pred.tolist())

    return y_true_all, y_pred_all


def evaluate_model_confusion(
    task: str,
    learning: str,
    model_name: str,
    *,
    hierarchical: bool = False,
    binary_name: str | None = None,
    subtype_name: str | None = None,
    feature_columns=None,
    use_augmentation: bool = False,
    augmentation_factor: int = 1,
    noise_level: float = 0.02,
) -> dict:
    """LOSO-evaluate the exact GUI configuration and save a green confusion matrix.

    Returns ``{"confusion_matrix_path", "metrics", "labels"}`` where metrics holds
    the four GUI scores measured on this very feature subset / augmentation, so
    both the matrix and the numbers match what was actually trained.
    """
    from wellbeing_hierarchical_classification import (
        compute_metrics,
        save_confusion_matrix_png,
    )

    feature_columns = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)
    GUI_CM_DIR.mkdir(parents=True, exist_ok=True)

    if hierarchical:
        labels = _TASK_LABELS["hierarchical"]
        binary_name = binary_name or model_name
        subtype_name = subtype_name or binary_name
        binary_estimator = model_catalog("binary", "supervised")[binary_name][1]
        subtype_estimator = model_catalog("subtype", "supervised")[subtype_name][1]
        y_true, y_pred = _loso_hierarchical(
            binary_estimator, subtype_estimator, feature_columns,
            use_augmentation, augmentation_factor, noise_level,
        )
        title = f"{binary_name} -> {subtype_name} LOSO - hierarchical"
        name_for_slug = f"hier_{binary_name}_{subtype_name}"
    else:
        labels = _TASK_LABELS[task]
        kind, estimator = model_catalog(task, learning)[model_name]
        X, y, groups = _load_dataset_with_groups(task, feature_columns)
        y_true, y_pred = _loso_flat(
            task, kind, estimator, X, y, groups, feature_columns, labels,
            use_augmentation, augmentation_factor, noise_level,
        )
        title = f"{model_name} LOSO - {task}"
        name_for_slug = model_name

    sensors_slug = "_".join(
        s for s in ALL_SENSORS
        if set(SENSOR_FEATURES[s]).issubset(set(feature_columns))
    ) or "custom"
    filename = (
        f"{_slugify(task if not hierarchical else 'hierarchical')}__"
        f"{_slugify(learning)}__{_slugify(name_for_slug)}__{sensors_slug}__"
        f"{'aug' if use_augmentation else 'base'}.png"
    )
    out_path = GUI_CM_DIR / filename
    save_confusion_matrix_png(y_true, y_pred, labels, title, out_path)

    all_metrics = compute_metrics(y_true, y_pred, labels)
    metrics = {k: float(all_metrics[k]) for k in _GUI_METRIC_KEYS if k in all_metrics}

    return {
        "confusion_matrix_path": str(out_path),
        "metrics": metrics,
        "labels": labels,
    }


# =============================================================================
# LAZY ANALYZER IMPORTS (cv2 / mediapipe are heavy -- only load on demand)
# =============================================================================

def load_analyzers():
    """Import and return everything needed to process frames.

    Returns a dict with the config modules, the per-frame analyzers, the
    sliding-window feature extractors and the shared overlay drawing module.
    """
    import config as posture_config
    import blink_config
    from posture_analyzer import PostureAnalyzer
    from feature_extractor import FeatureExtractor
    from blink_analyzer import BlinkAnalyzer
    from blink_feature_extractor import BlinkFeatureExtractor
    import overlay_drawing as overlay

    return {
        "posture_config": posture_config,
        "blink_config": blink_config,
        "PostureAnalyzer": PostureAnalyzer,
        "FeatureExtractor": FeatureExtractor,
        "BlinkAnalyzer": BlinkAnalyzer,
        "BlinkFeatureExtractor": BlinkFeatureExtractor,
        "overlay": overlay,
    }


def default_thresholds() -> dict:
    """Read the default thresholds straight from the project's config files.

    Posture defaults come from Scripts/Posture/config.py and the blink defaults
    from Scripts/Blinking/blink_config.py, so the GUI uses the exact
    literature-based values used by the offline pipeline (not hard-coded copies).
    """
    import config as posture_config
    import blink_config

    return {
        "window_size": int(posture_config.WINDOW_SIZE),
        "shoulder_thr": float(posture_config.SHOULDER_ASYMMETRY_THRESHOLD),
        "head_tilt_thr": float(posture_config.HEAD_LATERAL_TILT_THRESHOLD),
        "fhp_thr": float(posture_config.FORWARD_HEAD_PROXY_THRESHOLD),
        "min_visibility": float(posture_config.MIN_VISIBILITY),
        "ear_thr": float(blink_config.EAR_CLOSED_THRESHOLD),
        "perclos_thr": float(blink_config.PERCLOS_BAD_THRESHOLD),
    }


def apply_settings(settings: dict):
    """Align the shared config modules with how the GUI consumes windows.

    Nothing is editable from the GUI anymore: the window length stays at the
    config default (60s, to match data/supervised) and the posture/blink
    thresholds are left at their config values. The only adjustment is forcing
    the window STEP to equal the window SIZE, so the live pipeline produces a
    single tumbling window per minute instead of the offline 10s sliding step.

    Changing thresholds or window length at inference time would create a
    train/inference mismatch, so they remain fixed in the GUI.
    """
    import config as posture_config
    import blink_config

    win = int(posture_config.WINDOW_SIZE)
    blink_config.WINDOW_SIZE = win
    # A single tumbling window per minute -> step == size keeps one window.
    posture_config.WINDOW_STEP = win
    blink_config.WINDOW_STEP = win
