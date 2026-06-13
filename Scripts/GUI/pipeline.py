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


def _load_dataset_with_groups(task: str):
    """Load canonical features, labels, and the video_source participant column."""
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

    X, y, groups = _load_dataset_with_groups(model.task)
    X_fit, y_fit, _ = augment_training_data_groupwise(
        X,
        y,
        groups,
        FEATURE_COLUMNS,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        random_state=engine.RANDOM_STATE,
    )
    model.estimator.fit(X_fit.loc[:, FEATURE_COLUMNS], y_fit)
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
    X, y, groups = _load_dataset_with_groups(model.task)

    if model.kind == "anomaly":
        normal_mask = y == "normal"
        X_normal = X.loc[normal_mask].reset_index(drop=True)
        groups_normal = groups.loc[normal_mask].reset_index(drop=True)
        X_fit, _ = _augment_training_data_by_group(
            X_normal,
            groups_normal,
            FEATURE_COLUMNS,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level,
            random_state=engine.RANDOM_STATE,
        )
        model.preprocessor = _unsupervised_preprocessing_pipeline()
        model.preprocessor.fit(X_normal.loc[:, FEATURE_COLUMNS])
        X_scaled = model.preprocessor.transform(X_fit.loc[:, FEATURE_COLUMNS])
        model.estimator.fit(X_scaled)
        model.classes = ["normal", "abnormal"]

    elif model.kind == "cluster":
        X_fit, _ = _augment_training_data_by_group(
            X,
            groups,
            FEATURE_COLUMNS,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level,
            random_state=engine.RANDOM_STATE,
        )
        model.preprocessor = _unsupervised_preprocessing_pipeline()
        X_scaled = model.preprocessor.fit_transform(X_fit.loc[:, FEATURE_COLUMNS])
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
) -> "RealtimeModel":
    """Instantiate the chosen estimator and fit it on data/supervised.

    Returns a ready-to-use RealtimeModel (same object the CLI scripts use).
    """
    catalog = model_catalog(task, learning)
    if name not in catalog:
        raise KeyError(f"Model '{name}' not available for {task}/{learning}. "
                       f"Choices: {list(catalog)}")
    kind, estimator = catalog[name]
    model = RealtimeModel(task, learning, name, kind, clone(estimator))

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
    )
    subtype_model = build_and_train_model(
        "subtype",
        "supervised",
        subtype_name,
        use_augmentation=use_augmentation,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
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
