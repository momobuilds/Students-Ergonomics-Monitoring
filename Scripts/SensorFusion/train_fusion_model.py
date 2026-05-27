"""Train and validate the ergonomics sensor-fusion wellbeing classifier.

Accuracy alone is not enough for this project: a model can look acceptable by
mostly predicting common classes while missing rare "Critical" cases. This
script therefore selects and reports models with macro F1 and Critical recall in
view, keeps the final test split subject-based, and tunes any decision threshold
only on validation subjects so the test set remains a true final check.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

MPL_CACHE = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.modules.setdefault("train_fusion_model", sys.modules[__name__])

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # noqa: E402
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

try:  # noqa: E402
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on installed scikit-learn.
    StratifiedGroupKFold = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STATE_ORDER = ["Good", "Warning", "High Risk", "Critical"]
SEVERITY_MAP = {"Good": 0, "Warning": 1, "High Risk": 2, "Critical": 3}
INVERSE_SEVERITY_MAP = {value: key for key, value in SEVERITY_MAP.items()}
SEVERITY_SAMPLE_WEIGHTS = {
    "Good": 1.0,
    "Warning": 1.2,
    "High Risk": 1.8,
    "Critical": 2.5,
}

GROUP_COLUMN = "subject_id"
CONFUSION_MATRIX_IMAGE = "confusion_matrix.png"
MODEL_COMPARISON_CSV = "model_comparison.csv"
ABLATION_RESULTS_CSV = "ablation_results.csv"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"

BLINK_FEATURES = [
    "blink_frequency_bpm",
    "perclos",
    "last_inter_blink_interval",
    "mean_inter_blink_interval",
    "low_blink_alert",
    "high_perclos_alert",
]

POSTURE_FEATURES = [
    "good_posture_pct",
    "avg_shoulder_asymmetry",
    "avg_head_lateral_tilt",
    "avg_forward_head_proxy",
    "alerts_in_period",
    "dominant_status",
]

STRESS_FEATURES = [
    "baseline_heart_rate",
    "current_heart_rate",
    "stress_index",
    "stress_tag",
]

# These are the original approved sensor-fusion inputs. No label-like fields are
# allowed here; final_state is used only as y.
REQUIRED_FEATURES = BLINK_FEATURES + POSTURE_FEATURES + STRESS_FEATURES
CATEGORICAL_FEATURES = ["stress_tag", "dominant_status"]
ORIGINAL_NUMERIC_FEATURES = [
    feature for feature in REQUIRED_FEATURES if feature not in CATEGORICAL_FEATURES
]

ENGINEERED_FEATURES = [
    "hr_delta",
    "hr_ratio",
    "posture_bad_pct",
    "blink_interval_ratio",
    "blink_fatigue_proxy",
    "posture_risk_proxy",
    "stress_hr_proxy",
    "fatigue_stress_interaction",
    "posture_stress_interaction",
    "posture_fatigue_interaction",
    "alert_interaction",
]
ALL_NUMERIC_FEATURES = ORIGINAL_NUMERIC_FEATURES + ENGINEERED_FEATURES
FULL_ENGINEERED_FEATURES = REQUIRED_FEATURES + ENGINEERED_FEATURES

FEATURE_GROUPS = {
    "blink-only": BLINK_FEATURES,
    "posture-only": POSTURE_FEATURES,
    "stress-only": STRESS_FEATURES,
    "full fusion": REQUIRED_FEATURES,
    "full fusion engineered": FULL_ENGINEERED_FEATURES,
}

OPTIMIZE_CHOICES = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "combined_score",
    "critical_recall",
]


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived sensor features without using final_state or any label copy."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SensorFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        baseline_hr = self._numeric(df, "baseline_heart_rate")
        current_hr = self._numeric(df, "current_heart_rate")
        good_posture_pct = self._numeric(df, "good_posture_pct")
        last_ibi = self._numeric(df, "last_inter_blink_interval")
        mean_ibi = self._numeric(df, "mean_inter_blink_interval")
        perclos = self._numeric(df, "perclos")
        high_perclos_alert = self._numeric(df, "high_perclos_alert")
        low_blink_alert = self._numeric(df, "low_blink_alert")
        alerts_in_period = self._numeric(df, "alerts_in_period")
        forward_head_proxy = self._numeric(df, "avg_forward_head_proxy")
        stress_index = self._numeric(df, "stress_index")

        hr_ratio = self._safe_divide(current_hr, baseline_hr)
        posture_bad_pct = 100 - good_posture_pct

        df["hr_delta"] = current_hr - baseline_hr
        df["hr_ratio"] = hr_ratio
        df["posture_bad_pct"] = posture_bad_pct
        df["blink_interval_ratio"] = self._safe_divide(last_ibi, mean_ibi)
        df["blink_fatigue_proxy"] = perclos * (1 + high_perclos_alert) + low_blink_alert
        df["posture_risk_proxy"] = (
            posture_bad_pct / 100 + alerts_in_period / 10 + forward_head_proxy
        )
        df["stress_hr_proxy"] = stress_index * hr_ratio
        df["fatigue_stress_interaction"] = perclos * stress_index
        df["posture_stress_interaction"] = posture_bad_pct * stress_index
        df["posture_fatigue_interaction"] = posture_bad_pct * perclos
        df["alert_interaction"] = alerts_in_period * high_perclos_alert

        return df

    @staticmethod
    def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.to_numeric(df[column], errors="coerce")

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.where(denominator.abs() > 1e-9, np.nan)
        return numerator / denominator


class OrdinalSeverityModel(BaseEstimator, ClassifierMixin):
    """Wrap a regression pipeline and map predicted severity to class labels."""

    def __init__(
        self,
        pipeline: Pipeline,
        thresholds: tuple[float, float, float] = (0.5, 1.5, 2.5),
    ) -> None:
        self.pipeline = pipeline
        self.thresholds = thresholds
        self.classes_ = np.array(STATE_ORDER, dtype=object)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "OrdinalSeverityModel":
        severity = pd.Series(y).map(SEVERITY_MAP).to_numpy()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight
        self.pipeline.fit(X, severity, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        severity = self.pipeline.predict(X)
        return severity_to_labels(severity, self.thresholds)


SensorFeatureEngineer.__module__ = "train_fusion_model"
OrdinalSeverityModel.__module__ = "train_fusion_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate sensor-fusion wellbeing classifiers."
    )
    parser.add_argument(
        "--data",
        default="realistic_synthetic_ergonomics_dataset.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target",
        default="final_state",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--model-out",
        default="fusion_model.joblib",
        help="Path where the best trained pipeline/model will be saved.",
    )
    parser.add_argument(
        "--metrics-out",
        default="fusion_metrics.txt",
        help="Path where text metrics will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of subjects/rows reserved for validation and final testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits, tuning, and model initialization.",
    )
    parser.add_argument(
        "--optimize-for",
        choices=OPTIMIZE_CHOICES,
        default="macro_f1",
        help="Validation metric used to select the best candidate model.",
    )
    parser.add_argument(
        "--critical-weight",
        type=float,
        default=0.35,
        help="Weight of Critical recall in the combined model-selection score.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 0.5:
        raise ValueError("--test-size must be between 0 and 0.5")
    if not 0 <= args.critical_weight <= 1:
        raise ValueError("--critical-weight must be between 0 and 1")


def validate_columns(df: pd.DataFrame, target_column: str) -> None:
    missing_features = [column for column in REQUIRED_FEATURES if column not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing required feature column(s): " + ", ".join(missing_features)
        )
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if df[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")


def ordered_labels(y: pd.Series) -> list[str]:
    observed = [label for label in STATE_ORDER if label in set(y)]
    extras = sorted(str(label) for label in set(y) if label not in STATE_ORDER)
    return observed + extras


def feature_types(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in feature_columns if feature in ALL_NUMERIC_FEATURES]
    categorical = [
        feature for feature in feature_columns if feature in CATEGORICAL_FEATURES
    ]
    return numeric, categorical


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features, categorical_features = feature_types(feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(feature_columns: list[str], estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", SensorFeatureEngineer()),
            ("preprocessor", build_preprocessor(feature_columns)),
            ("model", estimator),
        ]
    )


def sample_weights(y: pd.Series) -> np.ndarray:
    return y.map(SEVERITY_SAMPLE_WEIGHTS).fillna(1.0).to_numpy(dtype=float)


def severity_to_labels(
    values: np.ndarray | pd.Series, thresholds: tuple[float, float, float]
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    buckets = np.digitize(values, bins=np.asarray(thresholds, dtype=float))
    buckets = np.clip(buckets, 0, 3)
    return np.asarray([INVERSE_SEVERITY_MAP[int(bucket)] for bucket in buckets])


def combined_selection_score(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    critical_weight: float,
) -> float:
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    critical = recall_score(
        y_true,
        y_pred,
        labels=["Critical"],
        average="macro",
        zero_division=0,
    )
    return (1 - critical_weight) * macro + critical_weight * critical


def scorer_for(metric: str, labels: list[str], critical_weight: float):
    if metric == "accuracy":
        return "accuracy"
    if metric == "macro_f1":
        return "f1_macro"
    if metric == "weighted_f1":
        return "f1_weighted"
    if metric == "critical_recall":
        return make_scorer(
            lambda y_true, y_pred: recall_score(
                y_true,
                y_pred,
                labels=["Critical"],
                average="macro",
                zero_division=0,
            )
        )
    return make_scorer(
        lambda y_true, y_pred: combined_selection_score(
            y_true,
            y_pred,
            labels=labels,
            critical_weight=critical_weight,
        )
    )


def split_contains_labels(y: pd.Series, idx: np.ndarray, labels: list[str]) -> bool:
    return set(labels).issubset(set(y.iloc[idx]))


def make_splits(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    labels: list[str],
) -> dict[str, Any]:
    y = df[target_column]
    all_idx = np.arange(len(df))

    if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].nunique(dropna=True) >= 3:
        groups = df[GROUP_COLUMN].astype(str)

        for seed_offset in range(100):
            seed = random_state + seed_offset
            outer = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=seed,
            )
            train_val_idx, test_idx = next(outer.split(df, y, groups=groups))
            if not split_contains_labels(y, test_idx, labels):
                continue

            train_val_groups = groups.iloc[train_val_idx]
            inner = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=seed + 1000,
            )
            inner_train_pos, inner_val_pos = next(
                inner.split(
                    df.iloc[train_val_idx],
                    y.iloc[train_val_idx],
                    groups=train_val_groups,
                )
            )
            train_idx = train_val_idx[inner_train_pos]
            val_idx = train_val_idx[inner_val_pos]

            if not split_contains_labels(y, train_idx, labels):
                continue
            if not split_contains_labels(y, val_idx, labels):
                continue

            train_subjects = set(groups.iloc[train_idx])
            val_subjects = set(groups.iloc[val_idx])
            test_subjects = set(groups.iloc[test_idx])
            if train_subjects & val_subjects or train_subjects & test_subjects:
                continue
            if val_subjects & test_subjects:
                continue

            return {
                "method": "GroupShuffleSplit by subject_id",
                "warning": None,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "train_groups": groups.iloc[train_idx],
                "val_groups": groups.iloc[val_idx],
                "test_groups": groups.iloc[test_idx],
            }

        raise ValueError("Could not create subject-group splits with all target classes.")

    warning = (
        "WARNING: subject_id is missing or has fewer than 3 unique values; "
        "falling back to stratified row splits. Use subject_id for stronger validation."
    )
    print(warning)

    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=test_size,
        random_state=random_state + 1000,
        stratify=y.iloc[train_val_idx],
    )
    return {
        "method": "Stratified row split fallback",
        "warning": warning,
        "train_idx": np.asarray(train_idx),
        "val_idx": np.asarray(val_idx),
        "test_idx": np.asarray(test_idx),
        "train_groups": None,
        "val_groups": None,
        "test_groups": None,
    }


def make_cv(
    y_train: pd.Series,
    train_groups: pd.Series | None,
    random_state: int,
) -> tuple[Any, str, pd.Series | None]:
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(3, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough rows per class for cross-validation.")

    if train_groups is not None and StratifiedGroupKFold is not None:
        n_splits = min(n_splits, int(train_groups.nunique()))
        return (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            ),
            f"StratifiedGroupKFold ({n_splits} folds)",
            train_groups,
        )

    if train_groups is not None:
        warnings.warn(
            "StratifiedGroupKFold is unavailable; tuning uses StratifiedKFold. "
            "The final holdout split remains subject-based.",
            stacklevel=2,
        )

    return (
        StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ),
        f"StratifiedKFold ({n_splits} folds)",
        None,
    )


def classifier_specs(random_state: int) -> dict[str, dict[str, Any]]:
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
            "params": {"model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]},
            "n_iter": 6,
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [8, 14, None],
                "model__min_samples_leaf": [3, 6, 10],
                "model__max_features": ["sqrt", 0.7],
            },
            "n_iter": 5,
        },
        "ExtraTrees": {
            "estimator": ExtraTreesClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [8, 14, None],
                "model__min_samples_leaf": [3, 6, 10],
                "model__max_features": ["sqrt", 0.7],
            },
            "n_iter": 5,
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "params": {
                "model__max_iter": [80, 140, 220],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            "n_iter": 5,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "model__n_estimators": [80, 140],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [3, 8],
                "model__subsample": [0.75, 1.0],
            },
            "n_iter": 5,
        },
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    critical_weight: float,
) -> dict[str, float]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    critical_report = report.get("Critical", {})
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    critical_recall = float(critical_report.get("recall", 0.0))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "critical_precision": float(critical_report.get("precision", 0.0)),
        "critical_recall": critical_recall,
        "critical_f1": float(critical_report.get("f1-score", 0.0)),
        "combined_score": (1 - critical_weight) * macro_f1
        + critical_weight * critical_recall,
    }


def get_pipeline_classes(model: Any) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(label) for label in model.classes_]
    if hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "classes_"):
        return [str(label) for label in model.named_steps["model"].classes_]
    return STATE_ORDER


def predict_with_critical_threshold(
    model: Any,
    X: pd.DataFrame,
    threshold: float | None,
) -> np.ndarray:
    if threshold is None or not hasattr(model, "predict_proba"):
        return model.predict(X)

    classes = get_pipeline_classes(model)
    if "Critical" not in classes:
        return model.predict(X)

    probabilities = model.predict_proba(X)
    critical_idx = classes.index("Critical")
    non_critical_indices = [index for index, label in enumerate(classes) if label != "Critical"]
    predictions: list[str] = []

    for row in probabilities:
        if row[critical_idx] >= threshold:
            predictions.append("Critical")
        else:
            best_idx = non_critical_indices[int(np.argmax(row[non_critical_indices]))]
            predictions.append(classes[best_idx])

    return np.asarray(predictions, dtype=object)


def tune_critical_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
    critical_weight: float,
) -> dict[str, Any]:
    base_pred = model.predict(X_val)
    base_metrics = evaluate_predictions(y_val, base_pred, labels, critical_weight)
    best = {
        "threshold": None,
        "metrics": base_metrics,
        "predictions": base_pred,
        "note": "base argmax predictions",
    }

    if not hasattr(model, "predict_proba"):
        return best

    for threshold in np.round(np.arange(0.10, 0.601, 0.02), 2):
        tuned_pred = predict_with_critical_threshold(model, X_val, float(threshold))
        tuned_metrics = evaluate_predictions(y_val, tuned_pred, labels, critical_weight)
        if tuned_metrics["combined_score"] > best["metrics"]["combined_score"]:
            best = {
                "threshold": float(threshold),
                "metrics": tuned_metrics,
                "predictions": tuned_pred,
                "note": "critical probability threshold",
            }

    return best


def tune_classifier(
    name: str,
    spec: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series | None,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
    optimize_for: str,
    critical_weight: float,
    random_state: int,
) -> dict[str, Any]:
    cv, cv_name, cv_groups = make_cv(y_train, train_groups, random_state)
    pipeline = build_pipeline(FULL_ENGINEERED_FEATURES, spec["estimator"])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec["params"],
        n_iter=spec["n_iter"],
        scoring=scorer_for(optimize_for, labels, critical_weight),
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search_fit_kwargs = {"model__sample_weight": sample_weights(y_train)}
    if cv_groups is not None:
        search_fit_kwargs["groups"] = cv_groups

    search.fit(X_train, y_train, **search_fit_kwargs)
    best_model = search.best_estimator_
    threshold_summary = tune_critical_threshold(
        best_model,
        X_val,
        y_val,
        labels,
        critical_weight,
    )

    return {
        "name": name,
        "model": best_model,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_method": cv_name,
        "threshold": threshold_summary["threshold"],
        "threshold_note": threshold_summary["note"],
        "validation_metrics": threshold_summary["metrics"],
        "validation_predictions": threshold_summary["predictions"],
        "feature_columns": FULL_ENGINEERED_FEATURES,
        "model_kind": "classifier",
    }


def tune_ordinal_thresholds(
    severity_pred: np.ndarray,
    y_val: pd.Series,
    labels: list[str],
    critical_weight: float,
) -> tuple[tuple[float, float, float], dict[str, float], np.ndarray]:
    best_thresholds = (0.5, 1.5, 2.5)
    best_pred = severity_to_labels(severity_pred, best_thresholds)
    best_metrics = evaluate_predictions(y_val, best_pred, labels, critical_weight)

    for t1 in np.round(np.arange(0.25, 1.01, 0.10), 2):
        for t2 in np.round(np.arange(1.05, 2.01, 0.10), 2):
            for t3 in np.round(np.arange(2.05, 3.01, 0.10), 2):
                if not t1 < t2 < t3:
                    continue
                thresholds = (float(t1), float(t2), float(t3))
                pred = severity_to_labels(severity_pred, thresholds)
                metrics = evaluate_predictions(y_val, pred, labels, critical_weight)
                if metrics["combined_score"] > best_metrics["combined_score"]:
                    best_thresholds = thresholds
                    best_pred = pred
                    best_metrics = metrics

    return best_thresholds, best_metrics, best_pred


def train_ordinal_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
    critical_weight: float,
    random_state: int,
) -> dict[str, Any]:
    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )
    pipeline = build_pipeline(FULL_ENGINEERED_FEATURES, regressor)
    ordinal_model = OrdinalSeverityModel(pipeline=pipeline)
    ordinal_model.fit(X_train, y_train, sample_weight=sample_weights(y_train))

    severity_pred = ordinal_model.pipeline.predict(X_val)
    thresholds, metrics, predictions = tune_ordinal_thresholds(
        severity_pred,
        y_val,
        labels,
        critical_weight,
    )
    ordinal_model.thresholds = thresholds

    return {
        "name": "OrdinalSeverityRegressor",
        "model": ordinal_model,
        "best_params": {
            "model": "RandomForestRegressor",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "thresholds": thresholds,
        },
        "best_cv_score": np.nan,
        "cv_method": "validation threshold tuning",
        "threshold": thresholds,
        "threshold_note": "ordinal severity thresholds",
        "validation_metrics": metrics,
        "validation_predictions": predictions,
        "feature_columns": FULL_ENGINEERED_FEATURES,
        "model_kind": "ordinal_regression",
    }


def selected_score(candidate: dict[str, Any], optimize_for: str) -> float:
    return float(candidate["validation_metrics"][optimize_for])


def metrics_row(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_weighted_f1": metrics["weighted_f1"],
        f"{prefix}_macro_precision": metrics["macro_precision"],
        f"{prefix}_weighted_precision": metrics["weighted_precision"],
        f"{prefix}_macro_recall": metrics["macro_recall"],
        f"{prefix}_weighted_recall": metrics["weighted_recall"],
        f"{prefix}_critical_precision": metrics["critical_precision"],
        f"{prefix}_critical_recall": metrics["critical_recall"],
        f"{prefix}_critical_f1": metrics["critical_f1"],
        f"{prefix}_combined_score": metrics["combined_score"],
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
    critical_weight: float,
    threshold: float | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if isinstance(model, OrdinalSeverityModel):
        predictions = model.predict(X)
    else:
        prob_threshold = threshold if isinstance(threshold, float) else None
        predictions = predict_with_critical_threshold(model, X, prob_threshold)

    metrics = evaluate_predictions(y, predictions, labels, critical_weight)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "report": classification_report(y, predictions, labels=labels, zero_division=0),
        "report_dict": classification_report(
            y,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels),
    }


def run_model_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series | None,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
    optimize_for: str,
    critical_weight: float,
    random_state: int,
) -> list[dict[str, Any]]:
    candidates = []
    for offset, (name, spec) in enumerate(classifier_specs(random_state).items()):
        print(f"Training candidate: {name}", flush=True)
        candidates.append(
            tune_classifier(
                name=name,
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                train_groups=train_groups,
                X_val=X_val,
                y_val=y_val,
                labels=labels,
                optimize_for=optimize_for,
                critical_weight=critical_weight,
                random_state=random_state + offset,
            )
        )

    print("Training candidate: OrdinalSeverityRegressor", flush=True)
    candidates.append(
        train_ordinal_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            labels=labels,
            critical_weight=critical_weight,
            random_state=random_state + 100,
        )
    )
    return candidates


def run_ablation(
    name: str,
    feature_columns: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    critical_weight: float,
    random_state: int,
) -> dict[str, Any]:
    model = build_pipeline(
        feature_columns,
        ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    )
    model.fit(X_train, y_train, model__sample_weight=sample_weights(y_train))
    threshold_summary = tune_critical_threshold(
        model,
        X_val,
        y_val,
        labels,
        critical_weight,
    )
    test_result = evaluate_model(
        model,
        X_test,
        y_test,
        labels,
        critical_weight,
        threshold=threshold_summary["threshold"],
    )

    return {
        "model": name,
        "features": len(feature_columns),
        "threshold": threshold_summary["threshold"],
        **metrics_row("validation", threshold_summary["metrics"]),
        **metrics_row("test", test_result["metrics"]),
    }


def run_leakage_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
    critical_weight: float,
    random_state: int,
) -> dict[str, Any]:
    with_categorical = build_pipeline(
        REQUIRED_FEATURES,
        ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    )
    without_categorical_features = [
        feature for feature in REQUIRED_FEATURES if feature not in CATEGORICAL_FEATURES
    ]
    without_categorical = build_pipeline(
        without_categorical_features,
        ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state + 1,
            n_jobs=1,
        ),
    )

    fit_kwargs = {"model__sample_weight": sample_weights(y_train)}
    with_categorical.fit(X_train, y_train, **fit_kwargs)
    without_categorical.fit(X_train, y_train, **fit_kwargs)

    with_pred = with_categorical.predict(X_val)
    without_pred = without_categorical.predict(X_val)
    with_metrics = evaluate_predictions(y_val, with_pred, labels, critical_weight)
    without_metrics = evaluate_predictions(y_val, without_pred, labels, critical_weight)
    macro_drop = with_metrics["macro_f1"] - without_metrics["macro_f1"]
    warning = "No large categorical leakage signal detected."
    if macro_drop >= 0.10:
        warning = (
            "WARNING: Removing stress_tag and dominant_status greatly reduced "
            "validation macro F1. These categorical variables may be too close to "
            "the label."
        )

    return {
        "with_categorical_macro_f1": with_metrics["macro_f1"],
        "without_categorical_macro_f1": without_metrics["macro_f1"],
        "macro_f1_drop": macro_drop,
        "warning": warning,
    }


def run_label_shuffle_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    critical_weight: float,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    shuffled_train = pd.Series(
        rng.permutation(y_train.to_numpy()),
        index=y_train.index,
        name=y_train.name,
    )
    shuffled_test = pd.Series(
        rng.permutation(y_test.to_numpy()),
        index=y_test.index,
        name=y_test.name,
    )
    model = build_pipeline(
        FULL_ENGINEERED_FEATURES,
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
    )
    model.fit(X_train, shuffled_train, model__sample_weight=sample_weights(shuffled_train))
    predictions = model.predict(X_test)
    return evaluate_predictions(shuffled_test, predictions, labels, critical_weight)


def save_confusion_matrix_image(
    matrix: np.ndarray, class_labels: list[str], output_path: Path
) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Best Sensor Fusion Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def extract_feature_importance(model: Any) -> pd.DataFrame | None:
    pipeline = model.pipeline if isinstance(model, OrdinalSeverityModel) else model
    if not hasattr(pipeline, "named_steps"):
        return None

    final_estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None

    if hasattr(final_estimator, "feature_importances_"):
        importance = np.asarray(final_estimator.feature_importances_, dtype=float)
    elif hasattr(final_estimator, "coef_"):
        coef = np.asarray(final_estimator.coef_, dtype=float)
        importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    if len(feature_names) != len(importance):
        return None

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def format_metric_line(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name}: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"weighted_f1={metrics['weighted_f1']:.4f}, "
        f"precision={metrics['weighted_precision']:.4f}, "
        f"recall={metrics['weighted_recall']:.4f}, "
        f"critical_recall={metrics['critical_recall']:.4f}, "
        f"combined_score={metrics['combined_score']:.4f}"
    )


def format_metrics_report(
    args: argparse.Namespace,
    df: pd.DataFrame,
    split_info: dict[str, Any],
    labels: list[str],
    best_candidate: dict[str, Any],
    train_result: dict[str, Any],
    validation_result: dict[str, Any],
    test_result: dict[str, Any],
    model_comparison: pd.DataFrame,
    ablation_results: pd.DataFrame,
    leakage_check: dict[str, Any],
    shuffled_metrics: dict[str, float],
    feature_importance_path: Path | None,
    model_path: Path,
    metrics_path: Path,
    confusion_matrix_path: Path,
) -> str:
    matrix_df = pd.DataFrame(
        test_result["confusion_matrix"],
        index=labels,
        columns=labels,
    )
    critical = test_result["report_dict"].get("Critical", {})
    train_groups = split_info.get("train_groups")
    val_groups = split_info.get("val_groups")
    test_groups = split_info.get("test_groups")

    if train_groups is None:
        subject_summary = "Subject counts: unavailable because stratified fallback was used."
    else:
        subject_summary = (
            f"Train subjects: {train_groups.nunique()}\n"
            f"Validation subjects: {val_groups.nunique()}\n"
            f"Test subjects: {test_groups.nunique()}"
        )

    command = (
        "python train_fusion_model.py \\\n"
        f"  --data {shlex.quote(str(args.data))} \\\n"
        f"  --target {shlex.quote(str(args.target))} \\\n"
        f"  --model-out {shlex.quote(str(args.model_out))} \\\n"
        f"  --metrics-out {shlex.quote(str(args.metrics_out))} \\\n"
        f"  --test-size {args.test_size} \\\n"
        f"  --random-state {args.random_state} \\\n"
        f"  --optimize-for {args.optimize_for} \\\n"
        f"  --critical-weight {args.critical_weight}"
    )

    threshold = best_candidate["threshold"]
    threshold_text = "None" if threshold is None else str(threshold)
    feature_importance_text = (
        str(feature_importance_path)
        if feature_importance_path is not None
        else "Not available for selected model"
    )

    return (
        "Sensor Fusion Model Validation\n"
        "==============================\n\n"
        "Validation notes:\n"
        "- Macro F1 matters more than plain accuracy because each wellbeing class "
        "should count, even when Good and Warning are more common.\n"
        "- Critical recall matters because missing a Critical state is more costly "
        "than raising an occasional false alarm.\n"
        "- The final test split is subject-based so the model cannot benefit from "
        "seeing the same student in both training and testing.\n"
        "- Critical threshold tuning uses validation subjects only. The test set is "
        "used once for the final selected model.\n\n"
        "Dataset:\n"
        f"Data path: {args.data}\n"
        f"Target column: {args.target}\n"
        f"Rows: {len(df)}\n"
        f"Class distribution:\n{df[args.target].value_counts().to_string()}\n\n"
        "Split:\n"
        f"Method: {split_info['method']}\n"
        f"{split_info.get('warning') or 'No split warning.'}\n"
        f"{subject_summary}\n\n"
        "Best Model:\n"
        f"Name: {best_candidate['name']}\n"
        f"Optimized for: {args.optimize_for}\n"
        f"CV method: {best_candidate['cv_method']}\n"
        f"CV score: {best_candidate['best_cv_score']:.4f}\n"
        f"Validation selection score: {selected_score(best_candidate, args.optimize_for):.4f}\n"
        f"Critical threshold / ordinal thresholds: {threshold_text}\n"
        f"Best hyperparameters:\n{json.dumps(best_candidate['best_params'], indent=2, default=str)}\n\n"
        "Train / Validation / Final Test Scores:\n"
        f"{format_metric_line('Train', train_result['metrics'])}\n"
        f"{format_metric_line('Validation', validation_result['metrics'])}\n"
        f"{format_metric_line('Final test', test_result['metrics'])}\n\n"
        "Critical Final Test Metrics:\n"
        f"Precision: {critical.get('precision', 0.0):.4f}\n"
        f"Recall: {critical.get('recall', 0.0):.4f}\n"
        f"F1: {critical.get('f1-score', 0.0):.4f}\n"
        f"Support: {critical.get('support', 0.0):.0f}\n\n"
        "Final Test Classification Report:\n"
        f"{test_result['report']}\n\n"
        "Final Test Confusion Matrix:\n"
        f"{matrix_df.to_string()}\n\n"
        "Model Comparison:\n"
        f"{model_comparison.to_string(index=False, max_colwidth=70)}\n\n"
        "Ablation Comparison:\n"
        f"{ablation_results.to_string(index=False, max_colwidth=70)}\n\n"
        "Categorical Leakage Check:\n"
        f"Validation macro F1 with stress_tag and dominant_status: "
        f"{leakage_check['with_categorical_macro_f1']:.4f}\n"
        f"Validation macro F1 without stress_tag and dominant_status: "
        f"{leakage_check['without_categorical_macro_f1']:.4f}\n"
        f"Macro F1 drop: {leakage_check['macro_f1_drop']:.4f}\n"
        f"{leakage_check['warning']}\n\n"
        "Label-Shuffle Sanity Test:\n"
        f"{format_metric_line('Shuffled labels', shuffled_metrics)}\n"
        "A shuffled-label score close to the real-label score would suggest leakage "
        "or an evaluation bug.\n\n"
        "Artifacts:\n"
        f"Saved best model: {model_path}\n"
        f"Saved metrics: {metrics_path}\n"
        f"Saved confusion matrix image: {confusion_matrix_path}\n"
        f"Saved model comparison: {MODEL_COMPARISON_CSV}\n"
        f"Saved ablation results: {ABLATION_RESULTS_CSV}\n"
        f"Saved feature importance: {feature_importance_text}\n\n"
        "Exact command:\n"
        f"{command}\n"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    data_path = Path(args.data)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    confusion_matrix_path = Path(CONFUSION_MATRIX_IMAGE)
    model_comparison_path = Path(MODEL_COMPARISON_CSV)
    ablation_results_path = Path(ABLATION_RESULTS_CSV)
    feature_importance_path = Path(FEATURE_IMPORTANCE_CSV)

    df = pd.read_csv(data_path)
    validate_columns(df, args.target)
    labels = ordered_labels(df[args.target])

    split_info = make_splits(
        df=df,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        labels=labels,
    )
    train_idx = split_info["train_idx"]
    val_idx = split_info["val_idx"]
    test_idx = split_info["test_idx"]

    X = df[REQUIRED_FEATURES].copy()
    y = df[args.target].copy()
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    candidates = run_model_search(
        X_train=X_train,
        y_train=y_train,
        train_groups=split_info["train_groups"],
        X_val=X_val,
        y_val=y_val,
        labels=labels,
        optimize_for=args.optimize_for,
        critical_weight=args.critical_weight,
        random_state=args.random_state,
    )
    best_candidate = max(
        candidates,
        key=lambda candidate: selected_score(candidate, args.optimize_for),
    )
    best_model = best_candidate["model"]
    best_threshold = best_candidate["threshold"]

    train_result = evaluate_model(
        best_model,
        X_train,
        y_train,
        labels,
        args.critical_weight,
        threshold=best_threshold,
    )
    validation_result = evaluate_model(
        best_model,
        X_val,
        y_val,
        labels,
        args.critical_weight,
        threshold=best_threshold,
    )
    test_result = evaluate_model(
        best_model,
        X_test,
        y_test,
        labels,
        args.critical_weight,
        threshold=best_threshold,
    )

    comparison_rows = []
    for candidate in candidates:
        row = {
            "model": candidate["name"],
            "cv_method": candidate["cv_method"],
            "cv_score": candidate["best_cv_score"],
            "selection_metric": args.optimize_for,
            "selection_score": selected_score(candidate, args.optimize_for),
            "threshold": candidate["threshold"],
            "best_params": json.dumps(candidate["best_params"], default=str),
        }
        row.update(metrics_row("validation", candidate["validation_metrics"]))
        comparison_rows.append(row)
    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        "selection_score",
        ascending=False,
    )

    print("Running ablation tests", flush=True)
    ablation_results = pd.DataFrame(
        [
            run_ablation(
                name=name,
                feature_columns=features,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                labels=labels,
                critical_weight=args.critical_weight,
                random_state=args.random_state + index,
            )
            for index, (name, features) in enumerate(FEATURE_GROUPS.items())
        ]
    ).sort_values("test_combined_score", ascending=False)

    leakage_check = run_leakage_check(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        labels=labels,
        critical_weight=args.critical_weight,
        random_state=args.random_state,
    )
    shuffled_metrics = run_label_shuffle_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        critical_weight=args.critical_weight,
        random_state=args.random_state + 999,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    save_confusion_matrix_image(test_result["confusion_matrix"], labels, confusion_matrix_path)
    model_comparison.to_csv(model_comparison_path, index=False)
    ablation_results.to_csv(ablation_results_path, index=False)

    feature_importance = extract_feature_importance(best_model)
    saved_feature_importance_path: Path | None = None
    if feature_importance is not None:
        feature_importance.to_csv(feature_importance_path, index=False)
        saved_feature_importance_path = feature_importance_path
    elif feature_importance_path.exists():
        feature_importance_path.unlink()

    metrics_text = format_metrics_report(
        args=args,
        df=df,
        split_info=split_info,
        labels=labels,
        best_candidate=best_candidate,
        train_result=train_result,
        validation_result=validation_result,
        test_result=test_result,
        model_comparison=model_comparison,
        ablation_results=ablation_results,
        leakage_check=leakage_check,
        shuffled_metrics=shuffled_metrics,
        feature_importance_path=saved_feature_importance_path,
        model_path=model_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")
    print(metrics_text)


if __name__ == "__main__":
    main()
