"""Train a sensor-fusion wellbeing classifier for ergonomics monitoring."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Keep Matplotlib CLI-friendly before it initializes config/cache paths.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# This is the strict sensor-fusion input contract. Extra CSV columns are ignored.
NUMERIC_FEATURES = [
    "blink_frequency_bpm",
    "perclos",
    "last_inter_blink_interval",
    "mean_inter_blink_interval",
    "low_blink_alert",
    "high_perclos_alert",
    "good_posture_pct",
    "avg_shoulder_asymmetry",
    "avg_head_lateral_tilt",
    "avg_forward_head_proxy",
    "alerts_in_period",
    "baseline_heart_rate",
    "current_heart_rate",
    "stress_index",
]

CATEGORICAL_FEATURES = ["stress_tag", "dominant_status"]
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CONFUSION_MATRIX_IMAGE = "confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest sensor-fusion model for wellbeing state classification."
    )
    parser.add_argument(
        "--data",
        default="synthetic_ergonomics_dataset.csv",
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
        help="Path where the trained sklearn pipeline will be saved.",
    )
    parser.add_argument(
        "--metrics-out",
        default="fusion_metrics.txt",
        help="Path where text metrics will be saved.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, target_column: str) -> None:
    """Fail early if the dataset cannot support the required training schema."""
    missing_features = [column for column in REQUIRED_FEATURES if column not in df.columns]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Missing required feature column(s): {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")


def validate_target_for_stratification(target: pd.Series) -> None:
    class_counts = target.value_counts(dropna=False)
    classes_with_too_few_rows = class_counts[class_counts < 2]
    if not classes_with_too_few_rows.empty:
        details = ", ".join(
            f"{label}={count}" for label, count in classes_with_too_few_rows.items()
        )
        raise ValueError(
            "Stratified train/test split requires at least 2 rows per class. "
            f"Insufficient class count(s): {details}"
        )


def build_pipeline() -> Pipeline:
    # Numeric and categorical sensor streams use separate preprocessing paths.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def format_metrics(
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    report: str,
    matrix,
    class_labels: list[str],
) -> str:
    matrix_df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

    return (
        "Sensor Fusion Model Metrics\n"
        "===========================\n\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-score:  {f1_score:.4f}\n\n"
        "Classification Report:\n"
        f"{report}\n\n"
        "Confusion Matrix:\n"
        f"{matrix_df.to_string()}\n"
    )


def save_confusion_matrix_image(
    matrix, class_labels: list[str], output_path: Path
) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Sensor Fusion Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    confusion_matrix_path = Path(CONFUSION_MATRIX_IMAGE)

    df = pd.read_csv(data_path)
    validate_columns(df, args.target)

    X = df[REQUIRED_FEATURES].copy()
    y = df[args.target].copy()
    validate_target_for_stratification(y)

    # Stratification preserves the wellbeing-state distribution in validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    class_labels = list(model.named_steps["classifier"].classes_)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=class_labels)

    metrics_text = format_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        report=report,
        matrix=matrix,
        class_labels=class_labels,
    )

    print(metrics_text)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(metrics_text, encoding="utf-8")
    save_confusion_matrix_image(matrix, class_labels, confusion_matrix_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix image to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
