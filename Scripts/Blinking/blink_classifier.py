"""Optional ML classifier for blinking features."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import blink_config as config


class BlinkClassifier:
    """Small wrapper for dataset collection, training, and prediction."""

    feature_names = [
        "ear",
        "blink_frequency_bpm",
        "perclos",
        "last_inter_blink_interval",
        "mean_inter_blink_interval",
    ]

    def __init__(
        self,
        training_data_path: Path = config.TRAINING_DATA_PATH,
        model_path: Path = config.MODEL_PATH,
    ) -> None:
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.model: Any | None = None

    def append_sample(self, state: dict[str, Any], label: str) -> None:
        self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.training_data_path.exists()
        with self.training_data_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[*self.feature_names, "label"])
            if write_header:
                writer.writeheader()
            row = {name: self._numeric_or_zero(state.get(name)) for name in self.feature_names}
            row["label"] = label
            writer.writerow(row)

    def train(self) -> dict[str, Any]:
        try:
            import joblib
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVC
        except ImportError as exc:
            raise ImportError(
                "ML mode requires scikit-learn, pandas, and joblib. "
                "Install them with: pip install scikit-learn pandas joblib"
            ) from exc

        data = pd.read_csv(self.training_data_path)
        if data["label"].nunique() < 2:
            raise ValueError("Training needs at least two labels.")

        x = data[self.feature_names].fillna(0.0)
        y = data["label"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        candidates = {
            "svm": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            ),
        }
        scores: dict[str, float] = {}
        best_name = ""
        best_model = None
        best_score = -1.0

        for name, model in candidates.items():
            model.fit(x_train, y_train)
            score = accuracy_score(y_test, model.predict(x_test))
            scores[name] = float(score)
            if score > best_score:
                best_name = name
                best_model = model
                best_score = score

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_model, "feature_names": self.feature_names}, self.model_path)
        self.model = best_model
        return {"best_model": best_name, "scores": scores, "model_path": str(self.model_path)}

    def load(self) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Prediction requires joblib: pip install joblib") from exc

        payload = joblib.load(self.model_path)
        self.model = payload["model"]

    def predict(self, state: dict[str, Any]) -> str:
        if self.model is None:
            self.load()

        features = [[self._numeric_or_zero(state.get(name)) for name in self.feature_names]]
        return str(self.model.predict(features)[0])

    @staticmethod
    def _numeric_or_zero(value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

