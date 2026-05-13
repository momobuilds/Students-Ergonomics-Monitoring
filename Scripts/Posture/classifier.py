"""
classifier.py — ML-Based Posture Classifier (SVM / Random Forest)

Provides a machine learning alternative to the fixed-threshold approach:

1. DATA COLLECTION: Record feature vectors (3 posture angles) with labels
   ("Good" / "Slouching") while the user demonstrates each posture.

2. TRAINING: Train an SVM or Random Forest on the collected data.

3. PREDICTION: Classify new frames using the trained model.

Feature vector (3 features from frontal camera):
    [shoulder_asymmetry, head_lateral_tilt, forward_head_proxy]
"""

import csv
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import config


class PostureClassifier:
    """
    ML classifier for posture classification.

    Usage:
        classifier = PostureClassifier()

        # Collect training data
        classifier.start_collection("Good")
        # ... call classifier.add_sample(features) each frame ...
        classifier.stop_collection()

        classifier.start_collection("Slouching")
        # ... call classifier.add_sample(features) each frame ...
        classifier.stop_collection()

        # Train
        results = classifier.train(model_type="svm")

        # Predict
        prediction = classifier.predict(features)
    """

    def __init__(self, training_data_path=None, model_path=None):
        """
        Initialize the classifier.

        Args:
            training_data_path (str): Path for the training data CSV.
            model_path (str): Path for saving/loading the trained model.
        """
        self.training_data_path = training_data_path or config.TRAINING_DATA_PATH
        self.model_path = model_path or config.MODEL_PATH

        # Trained model and scaler (None until trained or loaded)
        self._model = None
        self._scaler = None

        # Data collection state
        self._collecting = False
        self._current_label = None
        self._collected_samples = []

        # Try to load an existing model
        self._load_model()

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def start_collection(self, label):
        """
        Start collecting training samples for a given label.

        Args:
            label (str): "Good" or "Slouching".
        """
        self._collecting = True
        self._current_label = label
        self._collected_samples = []
        print(f"[INFO] Started collecting data for label: '{label}'")

    def add_sample(self, features):
        """
        Add a feature vector to the current collection.

        Args:
            features (list): Feature vector from PostureAnalyzer.get_feature_vector().
        """
        if not self._collecting or features is None:
            return
        self._collected_samples.append(features)

    def stop_collection(self):
        """
        Stop collecting and save samples to CSV.

        Returns:
            int: Number of samples saved.
        """
        if not self._collecting:
            return 0

        self._collecting = False
        count = len(self._collected_samples)
        self._save_training_data(self._collected_samples, self._current_label)

        print(f"[INFO] Stopped collecting. Saved {count} samples "
              f"with label '{self._current_label}'.")

        self._collected_samples = []
        self._current_label = None
        return count

    def is_collecting(self):
        """Check if data collection is active."""
        return self._collecting

    def get_collection_label(self):
        """Get the label being collected."""
        return self._current_label

    def get_collection_count(self):
        """Get the number of samples collected so far."""
        return len(self._collected_samples)

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, model_type="auto"):
        """
        Train classifier(s) on the collected training data.

        When model_type is "auto" (default), trains BOTH SVM and Random Forest
        on the same data, prints comparison, saves confusion matrices as images,
        and automatically selects the best model.

        When model_type is "svm" or "rf", trains only that model.

        Args:
            model_type (str): "auto" (both), "svm", or "rf".

        Returns:
            dict or None: Training results of the best/selected model.
        """
        X, y = self._load_training_data()

        if X is None or len(X) < 10:
            print("[ERROR] Not enough training data. Need at least 10 samples.")
            return None

        if len(set(y)) < 2:
            print("[ERROR] Need at least 2 different labels (Good and Slouching).")
            return None

        # Split 80/20 (same split for both models for fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Decide which models to train
        if model_type == "auto":
            models_to_train = ["svm", "rf"]
        else:
            models_to_train = [model_type]

        results = {}

        for mtype in models_to_train:
            print(f"\n{'='*60}")
            print(f"  Training: {mtype.upper()}")
            print(f"  Samples: {len(X)} | Features: {X.shape[1]}")
            print(f"{'='*60}")

            if mtype == "svm":
                model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
            elif mtype == "rf":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
            else:
                print(f"[ERROR] Unknown model type: {mtype}")
                continue

            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=["Good", "Slouching"])
            cm = confusion_matrix(y_test, y_pred, labels=["Good", "Slouching"])

            print(f"\n  Accuracy: {accuracy:.2%}")
            print(report)

            # Print confusion matrix to terminal
            print(f"  Confusion Matrix ({mtype.upper()}):")
            print(f"                 Predicted Good  Predicted Slouching")
            print(f"  Actual Good         {cm[0][0]:>5}          {cm[0][1]:>5}")
            print(f"  Actual Slouching    {cm[1][0]:>5}          {cm[1][1]:>5}")

            # Save confusion matrix as image
            self._save_confusion_matrix_image(cm, mtype, accuracy)

            # Feature importance for Random Forest
            if mtype == "rf":
                feature_names = ["Shoulder Asymmetry", "Head Lateral Tilt", "Forward Head Proxy"]
                importances = model.feature_importances_
                print(f"\n  Feature Importance ({mtype.upper()}):")
                for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
                    bar = "\u2588" * int(imp * 40)
                    print(f"    {fname:>25s}: {imp:.3f} {bar}")

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

        # --- Select best model ---
        if len(results) == 0:
            return None

        if len(results) == 1:
            best_type = list(results.keys())[0]
        else:
            # Compare and select best
            print(f"\n{'='*60}")
            print(f"  MODEL COMPARISON")
            print(f"{'='*60}")
            for mtype, r in results.items():
                print(f"  {mtype.upper():>6s}: {r['accuracy']:.2%} accuracy")

            best_type = max(results, key=lambda k: results[k]["accuracy"])
            print(f"\n  >>> Best model: {best_type.upper()} ({results[best_type]['accuracy']:.2%})")
            print(f"  >>> Saved as {self.model_path}")

        # Save the best model
        best = results[best_type]
        self._model = best["model"]
        self._scaler = best["scaler"]
        self._save_model()

        return best

    def _save_confusion_matrix_image(self, cm, model_type, accuracy):
        """
        Save a confusion matrix as a PNG image.

        Args:
            cm (np.array): 2x2 confusion matrix.
            model_type (str): "svm" or "rf".
            accuracy (float): Model accuracy.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))

            # Draw the matrix as a heatmap
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)

            labels = ["Good", "Slouching"]
            ax.set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=labels, yticklabels=labels,
                   xlabel="Predicted", ylabel="Actual",
                   title=f"Confusion Matrix \u2014 {model_type.upper()} (Accuracy: {accuracy:.2%})")

            # Write numbers in cells
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color=color, fontsize=20, fontweight="bold")

            plt.tight_layout()
            filename = os.path.join(config.DATA_DIR, f"confusion_matrix_{model_type}.png")
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"  Confusion matrix saved: {filename}")

        except Exception as e:
            print(f"  [WARNING] Could not save confusion matrix image: {e}")

    def is_trained(self):
        """Check if a trained model is available."""
        return self._model is not None and self._scaler is not None

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, features):
        """
        Classify a single feature vector.

        Args:
            features (list): Feature vector from PostureAnalyzer.get_feature_vector().

        Returns:
            dict or None: {"label": str, "confidence": float}
        """
        if not self.is_trained() or features is None:
            return None

        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        label = self._model.predict(X_scaled)[0]
        proba = self._model.predict_proba(X_scaled)[0]
        confidence = max(proba)

        return {
            "label": label,
            "confidence": round(confidence, 3),
        }

    # =========================================================================
    # DATA PERSISTENCE
    # =========================================================================

    def _save_training_data(self, samples, label):
        """Append training samples to CSV."""
        file_exists = os.path.exists(self.training_data_path)

        with open(self.training_data_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                n_features = len(samples[0]) if samples else 0
                header = [f"feature_{i}" for i in range(n_features)] + ["label"]
                writer.writerow(header)
            for sample in samples:
                writer.writerow(list(sample) + [label])

    def _load_training_data(self):
        """Load training data from CSV."""
        if not os.path.exists(self.training_data_path):
            print(f"[ERROR] Training data not found at {self.training_data_path}")
            return None, None

        X, y = [], []
        with open(self.training_data_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                X.append([float(v) for v in row[:-1]])
                y.append(row[-1])

        return np.array(X), y

    def _save_model(self):
        """Save model and scaler to disk."""
        data = {"model": self._model, "scaler": self._scaler}
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] Model saved to {self.model_path}")

    def _load_model(self):
        """Load model from disk if available."""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            print(f"[INFO] Model loaded from {self.model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
