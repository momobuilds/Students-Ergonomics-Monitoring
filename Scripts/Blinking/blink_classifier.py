"""
blink_classifier.py — ML-Based Blink Classifier (SVM / Random Forest)

Mirrors the PostureClassifier pipeline exactly:

1. DATA COLLECTION: Record feature vectors with labels ("Normal" / "Tired")
   while the user demonstrates each blinking state.
   - start_collection(label)  →  add_sample(state)  →  stop_collection()

2. TRAINING: Train BOTH SVM and Random Forest on the collected data,
   print comparison, save confusion matrices as PNG images in
   data/processed/blink, and automatically select the best model.

3. PREDICTION: Classify new frames using the trained model.

Feature vector (5 features from BlinkMonitor state):
    [ear, blink_frequency_bpm, perclos,
     last_inter_blink_interval, mean_inter_blink_interval]
"""

import csv
import os
import pickle
from typing import Any

import numpy as np

import blink_config as config


class BlinkClassifier:
    """
    ML classifier for blink-state classification.

    Usage:
        classifier = BlinkClassifier()

        # Collect training data
        classifier.start_collection("Normal")
        # ... call classifier.add_sample(state) each frame ...
        classifier.stop_collection()

        classifier.start_collection("Tired")
        # ... call classifier.add_sample(state) each frame ...
        classifier.stop_collection()

        # Train both models and pick the best
        result = classifier.train()

        # Predict
        prediction = classifier.predict(state)  # → {"label": str, "confidence": float}
    """

    FEATURE_NAMES = [
        "ear",
        "blink_frequency_bpm",
        "perclos",
        "last_inter_blink_interval",
        "mean_inter_blink_interval",
    ]

    def __init__(
        self,
        training_data_path: str = None,
        model_path: str = None,
    ) -> None:
        self.training_data_path = training_data_path or str(config.TRAINING_DATA_PATH)
        self.model_path = model_path or str(config.BEST_MODEL_PATH)

        # Trained model and scaler (None until trained or loaded)
        self._model: Any = None
        self._scaler: Any = None

        # Data collection state
        self._collecting: bool = False
        self._current_label: str | None = None
        self._collected_samples: list[list[float]] = []

        # Try to load an existing model
        self._load_model()

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def start_collection(self, label: str) -> None:
        """
        Start collecting training samples for a given label.

        Args:
            label: "Normal" or "Tired".
        """
        self._collecting = True
        self._current_label = label
        self._collected_samples = []
        print(f"[INFO] Started collecting blink data for label: '{label}'")

    def add_sample(self, state: dict[str, Any]) -> None:
        """
        Add one BlinkMonitor state dict to the current collection.

        Args:
            state: dict returned by BlinkMonitor.update().
        """
        if not self._collecting or state is None:
            return
        features = self._extract_features(state)
        if features is not None:
            self._collected_samples.append(features)

    def stop_collection(self) -> int:
        """
        Stop collecting and save samples to CSV.

        Returns:
            Number of samples saved.
        """
        if not self._collecting:
            return 0

        self._collecting = False
        count = len(self._collected_samples)
        if count > 0:
            self._save_training_data(self._collected_samples, self._current_label)

        print(
            f"[INFO] Stopped collecting. Saved {count} samples "
            f"with label '{self._current_label}'."
        )
        self._collected_samples = []
        self._current_label = None
        return count

    def is_collecting(self) -> bool:
        """Return True if data collection is active."""
        return self._collecting

    def get_collection_label(self) -> str | None:
        """Return the label currently being collected."""
        return self._current_label

    def get_collection_count(self) -> int:
        """Return the number of samples collected so far in this session."""
        return len(self._collected_samples)

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, model_type: str = "auto") -> dict | None:
        """
        Train classifier(s) on the collected training data.

        When model_type is "auto" (default), trains BOTH SVM and Random Forest
        on the same data, prints a comparison, saves confusion matrices as PNG
        images, and automatically selects the best model.

        When model_type is "svm" or "rf", trains only that model.

        Args:
            model_type: "auto" (both), "svm", or "rf".

        Returns:
            dict with training results of the best/selected model, or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:
            raise ImportError(
                "ML training requires scikit-learn, matplotlib.\n"
                "Install: pip install scikit-learn matplotlib"
            ) from exc

        X, y = self._load_training_data()

        if X is None or len(X) < 10:
            print("[ERROR] Not enough blink training data. Need at least 10 samples.")
            return None

        if len(set(y)) < 2:
            print("[ERROR] Need at least 2 different labels (e.g. Normal and Tired).")
            return None

        labels_order = sorted(set(y))

        # 80/20 split — same split for both models (fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardise features
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        models_to_train = ["svm", "rf"] if model_type == "auto" else [model_type]
        results: dict[str, dict] = {}

        for mtype in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"  Training: {mtype.upper()}")
            print(f"  Samples: {len(X)} | Features: {X.shape[1]}")
            print(f"{'=' * 60}")

            if mtype == "svm":
                model = SVC(
                    kernel="rbf", C=1.0, gamma="scale", probability=True
                )
            elif mtype == "rf":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
            else:
                print(f"[ERROR] Unknown model type: {mtype}")
                continue

            model.fit(X_train_sc, y_train)

            y_pred = model.predict(X_test_sc)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, target_names=labels_order
            )
            cm = confusion_matrix(y_test, y_pred, labels=labels_order)

            print(f"\n  Accuracy: {accuracy:.2%}")
            print(report)

            # Print confusion matrix to terminal
            print(f"  Confusion Matrix ({mtype.upper()}):")
            header = "".join(f"  Pred {lbl:>10}" for lbl in labels_order)
            print(f"  {'':>20}{header}")
            for i, row_label in enumerate(labels_order):
                row_str = "".join(f"  {cm[i, j]:>15}" for j in range(len(labels_order)))
                print(f"  Actual {row_label:>12}{row_str}")

            # Save confusion matrix image
            self._save_confusion_matrix_image(
                cm, labels_order, mtype, accuracy, plt
            )

            # Feature importance for Random Forest
            if mtype == "rf":
                importances = model.feature_importances_
                print(f"\n  Feature Importance (RF):")
                for fname, imp in sorted(
                    zip(self.FEATURE_NAMES, importances), key=lambda x: -x[1]
                ):
                    bar = "█" * int(imp * 40)
                    print(f"    {fname:>35s}: {imp:.3f} {bar}")

            # Save individual model to disk
            indiv_path = (
                str(config.SVM_MODEL_PATH)
                if mtype == "svm"
                else str(config.RF_MODEL_PATH)
            )
            self._dump_model(model, scaler, indiv_path)
            print(f"  Model saved: {indiv_path}")

            results[mtype] = {
                "model": model,
                "scaler": scaler,
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": cm,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "model_type": mtype,
            }

        if not results:
            return None

        # --- Select and save best model ---
        if len(results) == 1:
            best_type = list(results.keys())[0]
        else:
            print(f"\n{'=' * 60}")
            print(f"  MODEL COMPARISON")
            print(f"{'=' * 60}")
            for mtype, r in results.items():
                print(f"  {mtype.upper():>6s}: {r['accuracy']:.2%} accuracy")

            best_type = max(results, key=lambda k: results[k]["accuracy"])
            print(
                f"\n  >>> Best model: {best_type.upper()} "
                f"({results[best_type]['accuracy']:.2%})"
            )
            print(f"  >>> Saved as {self.model_path}")

        best = results[best_type]
        self._model = best["model"]
        self._scaler = best["scaler"]
        self._dump_model(self._model, self._scaler, self.model_path)
        print(f"[INFO] Best model ({best_type.upper()}) saved to {self.model_path}")

        return best

    def _save_confusion_matrix_image(
        self,
        cm: np.ndarray,
        labels: list[str],
        model_type: str,
        accuracy: float,
        plt: Any,
    ) -> None:
        """Save confusion matrix as a PNG image in data/processed/blink."""
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)

            ax.set(
                xticks=range(len(labels)),
                yticks=range(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                xlabel="Predicted",
                ylabel="Actual",
                title=(
                    f"Confusion Matrix — {model_type.upper()} "
                    f"(Accuracy: {accuracy:.2%})"
                ),
            )

            thresh = cm.max() / 2.0
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = "white" if cm[i, j] > thresh else "black"
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color=color, fontsize=20, fontweight="bold",
                    )

            plt.tight_layout()

            out_path = (
                str(config.CM_SVM_PATH)
                if model_type == "svm"
                else str(config.CM_RF_PATH)
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Confusion matrix saved: {out_path}")

        except Exception as exc:
            print(f"  [WARNING] Could not save confusion matrix image: {exc}")

    def is_trained(self) -> bool:
        """Return True if a trained model is loaded."""
        return self._model is not None and self._scaler is not None

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, state: dict[str, Any]) -> dict | None:
        """
        Classify a single BlinkMonitor state.

        Args:
            state: dict returned by BlinkMonitor.update().

        Returns:
            {"label": str, "confidence": float} or None.
        """
        if not self.is_trained() or state is None:
            return None

        features = self._extract_features(state)
        if features is None:
            return None

        X = np.array(features).reshape(1, -1)
        X_sc = self._scaler.transform(X)

        label = self._model.predict(X_sc)[0]
        proba = self._model.predict_proba(X_sc)[0]
        confidence = float(max(proba))

        return {"label": label, "confidence": round(confidence, 3)}

    # =========================================================================
    # DATA PERSISTENCE
    # =========================================================================

    def _extract_features(self, state: dict[str, Any]) -> list[float] | None:
        """Convert a BlinkMonitor state dict into a fixed-size feature vector."""
        try:
            return [
                float(state.get(name) or 0.0)
                for name in self.FEATURE_NAMES
            ]
        except (TypeError, ValueError):
            return None

    def _save_training_data(
        self, samples: list[list[float]], label: str
    ) -> None:
        """Append training samples to the CSV file."""
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        file_exists = os.path.exists(self.training_data_path)

        with open(self.training_data_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.FEATURE_NAMES + ["label"])
            for sample in samples:
                writer.writerow(list(sample) + [label])

    def _load_training_data(
        self,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        """Load training data from CSV."""
        if not os.path.exists(self.training_data_path):
            print(
                f"[ERROR] Blink training data not found at "
                f"{self.training_data_path}"
            )
            return None, None

        X, y = [], []
        with open(self.training_data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                X.append([float(v) for v in row[:-1]])
                y.append(row[-1])

        return np.array(X), y

    def _dump_model(self, model: Any, scaler: Any, path: str) -> None:
        """Pickle model + scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)

    def _load_model(self) -> None:
        """Load the best saved model from disk if available."""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            print(f"[INFO] Blink model loaded from {self.model_path}")
        except Exception as exc:
            print(f"[WARNING] Could not load blink model: {exc}")
