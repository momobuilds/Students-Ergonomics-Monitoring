"""
main.realtime.py - Real-time wellbeing monitor (camera + wearable -> ML output).

This is the live, real-time counterpart of the offline pipeline:

    Scripts/Posture        -> 9 posture features per 60s window
    Scripts/Blinking       -> 7 blinking features per 60s window
    Scripts/HeartRate       -> HR + RMMSD from the CMS50D wearable
    Scripts/ML             -> classifies normal / sleepy / stress on the
                              SAME 18 features (see wellbeing_hierarchical_*)

What it does:
    1. Opens one webcam stream.
    2. For every frame it extracts the real eye (EAR) and posture values.
    3. In parallel it reads HR / heart-rate variability from the wearable.
    4. Every 60 seconds it builds the 18-feature vector exactly like the
       training data (data/supervised) and feeds it to the chosen model,
       printing and displaying the predicted state.

At startup a guided menu walks you through every choice, with no default:
    1. how many classes:  binary (normal vs abnormal)  or  3 classes
                          (normal / sleepy / stress)
    2. learning type:     supervised  or  unsupervised
    3. the exact model among those available for that combination, e.g.
       - supervised:     Logistic Regression, SVM, Random Forest, Gradient Boosting
       - unsupervised:   IsolationForest, OneClassSVM, LocalOutlierFactor (binary),
                         K-Means, Gaussian Mixture (clustering)
The chosen model is then trained on data/supervised. The --task / --learning /
--model flags pre-fill any step so the prompt can be skipped in scripts.

The wearable is optional: if no CMS50D is connected, the two heart features
are left missing and imputed by the model, so the camera pipeline can still
be tested on a machine without the device.

Usage:
    python main.realtime.py                                  # full guided menu
    python main.realtime.py --task 3class --learning supervised --model "Random Forest"
    python main.realtime.py --task binary --learning unsupervised
    python main.realtime.py --camera-index 1
    python main.realtime.py --port COM5                      # wearable on another serial port
    python main.realtime.py --port HID                       # wearable over HID
    python main.realtime.py --no-wearable                    # skip the wearable entirely

Keyboard controls (click the video window first):
    q = quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


# =============================================================================
# PATHS / MODULE WIRING
# =============================================================================

# This file lives in Scripts/App; the sibling feature folders and the project
# root are one level up from Scripts.
APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent
POSTURE_DIR = SCRIPTS_DIR / "Posture"
BLINKING_DIR = SCRIPTS_DIR / "Blinking"
HEARTRATE_DIR = SCRIPTS_DIR / "HeartRate"
ML_DIR = SCRIPTS_DIR / "ML"
PROJECT_ROOT = SCRIPTS_DIR.parent
SUPERVISED_DIR = PROJECT_ROOT / "data" / "supervised"

# The sub-modules import their config as a bare name ("config" / "blink_config"),
# so their folders must be importable directly. ML is added so we can reuse the
# exact model definitions (model_factories) from the evaluation pipeline.
# APP_DIR is added so we can import the shared overlay_drawing module (same folder).
for _module_dir in (APP_DIR, POSTURE_DIR, BLINKING_DIR, HEARTRATE_DIR, ML_DIR):
    _path = str(_module_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# =============================================================================
# FEATURE SCHEMA -- must match Scripts/ML and data/supervised exactly
# =============================================================================

POSTURE_FEATURES = [
    "p_shoulder_mean", "p_shoulder_std", "p_shoulder_bad_pct",
    "p_head_tilt_mean", "p_head_tilt_std", "p_head_tilt_bad_pct",
    "p_fhp_mean", "p_fhp_std", "p_fhp_bad_pct",
]
BLINK_FEATURES = [
    "b_ear_mean", "b_ear_std", "b_perclos",
    "b_blink_count", "b_blink_freq", "b_ibi_mean", "b_ibi_std",
]
HEART_FEATURES = ["HR", "RMMSD"]
FEATURE_COLUMNS = POSTURE_FEATURES + BLINK_FEATURES + HEART_FEATURES

RANDOM_STATE = 42
WINDOW_NAME = "Wellbeing Monitor (camera + wearable -> ML)"

# Which supervised CSVs and which label to assign for each task.
TASK_FILES = {
    "binary": [("normal.csv", "normal"), ("anomalous.csv", "abnormal")],
    "3class": [("normal.csv", "normal"), ("sleepy.csv", "sleepy"), ("stress.csv", "stress")],
}


# =============================================================================
# MODEL SELECTION -- guided menu: #classes -> supervised? -> exact model
# =============================================================================

def _select(prompt, options, preselected=None):
    """Ask the user to pick one option. No default unless preselected matches.

    Args:
        options: list of (key, label) tuples.
        preselected: a key/label to use non-interactively (from CLI), or None.
    """
    keys = [k for k, _ in options]
    if preselected:
        for key, label in options:
            if preselected.strip().lower() in (key.lower(), label.lower()):
                return key
        print(f"  [!] Unknown choice '{preselected}', please pick from the list.")

    print(f"\n{prompt}")
    for index, (key, label) in enumerate(options, start=1):
        print(f"  {index}) {label}")
    while True:
        choice = input(f"Select [1-{len(options)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return keys[int(choice) - 1]
        for key, label in options:
            if choice.lower() in (key.lower(), label.lower()):
                return key
        print("Invalid choice, try again.")


def supervised_models():
    """The 4 supervised classifier pipelines from the ML evaluation script."""
    from wellbeing_hierarchical_classification import model_factories
    return model_factories()


def model_catalog(task, learning):
    """Return {model_name: (kind, estimator)} for a task/learning combination.

    kind is one of 'classifier', 'anomaly', 'cluster'. Every estimator is
    imported from the ML scripts (supervised + unsupervised), so the model
    definitions live in a single place.
    """
    catalog = {}

    if learning == "supervised":
        for name, pipeline in supervised_models().items():
            catalog[name] = ("classifier", pipeline)
        return catalog

    # Unsupervised: anomaly detectors (binary only) + clustering, all from ML.
    if task == "binary":
        from unsupervised_wellbeing_analysis import anomaly_models, clustering_models
        for name, model in anomaly_models(RANDOM_STATE).items():
            catalog[name] = ("anomaly", model)
    else:
        from unsupervised_3group_wellbeing_analysis import clustering_models

    for name, estimator in clustering_models(RANDOM_STATE).items():
        # keep only clustering models that can assign a label to NEW samples
        # (e.g. Agglomerative has no predict, so it is skipped for live use)
        if hasattr(estimator, "predict"):
            catalog[name] = ("cluster", estimator)
    return catalog


def choose_model(args):
    """Walk the user through every choice down to the exact model.

    Returns an unfitted RealtimeModel. CLI flags (--task/--learning/--model)
    pre-fill a step; anything not provided is asked interactively.
    """
    task = _select(
        "How many classes should the model predict?",
        [("binary", "Binary  (normal vs abnormal)"),
         ("3class", "3 classes  (normal / sleepy / stress)")],
        preselected=args.task,
    )
    learning = _select(
        "Supervised or unsupervised?",
        [("supervised", "Supervised  (uses the labels)"),
         ("unsupervised", "Unsupervised  (anomaly detection / clustering)")],
        preselected=args.learning,
    )

    catalog = model_catalog(task, learning)
    name = _select(
        "Which exact model?",
        [(n, n) for n in catalog],
        preselected=args.model,
    )
    kind, estimator = catalog[name]
    return RealtimeModel(task, learning, name, kind, estimator)


def load_dataset(task):
    """Load the supervised CSVs for a task into (X canonical features, y labels)."""
    import pandas as pd

    frames = []
    for filename, label in TASK_FILES[task]:
        csv_path = SUPERVISED_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {csv_path}\n"
                f"Run Scripts/build_training_dataset.py first to create data/supervised."
            )
        df = pd.read_csv(csv_path)
        df["__label__"] = label
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # The HRV column is spelled 'RMMSD'; accept 'RMSSD' too, then normalise.
    source_cols = []
    for feature in FEATURE_COLUMNS:
        if feature == "RMMSD" and "RMMSD" not in data.columns and "RMSSD" in data.columns:
            source_cols.append("RMSSD")
        else:
            source_cols.append(feature)

    missing = [c for c in source_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Supervised data is missing feature columns: {missing}")

    X = data[source_cols].apply(pd.to_numeric, errors="coerce")
    X.columns = FEATURE_COLUMNS  # canonical names for inference
    y = data["__label__"].astype(str)
    return X, y


class RealtimeModel:
    """Wraps the chosen estimator and exposes a uniform fit / predict_row API.

    Handles three kinds:
      - 'classifier': supervised, predicts the label directly (with proba).
      - 'anomaly':    trained on normal samples only, flags normal/abnormal.
      - 'cluster':    clusters are mapped to labels by majority vote on the
                      training labels, then new samples are assigned a cluster.
    """

    def __init__(self, task, learning, name, kind, estimator):
        self.task = task
        self.learning = learning
        self.name = name
        self.kind = kind
        self.estimator = estimator
        self.preprocessor = None     # imputer + scaler for unsupervised models
        self.cluster_map = None      # cluster id -> label, for clustering
        self.classes = []

    def describe(self):
        return f"{self.name} [{self.task} / {self.learning}]"

    def fit(self):
        import pandas as pd
        X, y = load_dataset(self.task)

        if self.kind == "classifier":
            self.estimator.fit(X, y)
            self.classes = list(self.estimator.classes_)

        elif self.kind == "anomaly":
            from unsupervised_wellbeing_analysis import preprocessing_pipeline
            self.preprocessor = preprocessing_pipeline()
            X_normal = X[y == "normal"]
            X_scaled = self.preprocessor.fit_transform(X_normal)
            self.estimator.fit(X_scaled)
            self.classes = ["normal", "abnormal"]

        elif self.kind == "cluster":
            from unsupervised_wellbeing_analysis import preprocessing_pipeline
            self.preprocessor = preprocessing_pipeline()
            X_scaled = self.preprocessor.fit_transform(X)
            self.estimator.fit(X_scaled)
            cluster_ids = self.estimator.predict(X_scaled)
            table = pd.crosstab(pd.Series(cluster_ids, name="cluster"),
                                pd.Series(list(y), name="label"))
            self.cluster_map = {int(c): str(table.loc[c].idxmax()) for c in table.index}
            self.classes = sorted(set(self.cluster_map.values()))

        n_per = ", ".join(f"{c}={int((y == c).sum())}" for c in sorted(y.unique()))
        print(f"[ML] {self.describe()} trained on {len(X)} windows ({n_per}).")
        return self

    def predict_row(self, feature_row):
        """Predict for one assembled feature row -> {'label', 'proba'}."""
        import pandas as pd
        frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

        if self.kind == "classifier":
            label = str(self.estimator.predict(frame)[0])
            proba = {}
            if hasattr(self.estimator, "predict_proba"):
                probs = self.estimator.predict_proba(frame)[0]
                proba = {str(c): float(p) for c, p in zip(self.estimator.classes_, probs)}
            return {"label": label, "proba": proba}

        X_scaled = self.preprocessor.transform(frame)

        if self.kind == "anomaly":
            raw = self.estimator.predict(X_scaled)[0]
            label = "normal" if raw == 1 else "abnormal"
            return {"label": label, "proba": {}}

        # cluster
        cluster_id = int(self.estimator.predict(X_scaled)[0])
        label = self.cluster_map.get(cluster_id, str(cluster_id))
        return {"label": label, "proba": {}}


# =============================================================================
# HEART RATE -- thin wrapper around the CMS50D wearable (optional)
# =============================================================================

class HeartRateMonitor:
    """Reads HR and RR intervals from the CMS50D in the background.

    Per 60s window it exposes:
        HR    = mean pulse rate (bpm)
        RMMSD = root mean square of successive RR differences (HRV)

    If the device is absent or fails to connect, it stays in fallback mode and
    returns NaN for both features (imputed downstream by the model).
    """

    TARGET_FPS = 30
    MIN_HR_BPM = 40

    def __init__(self, port):
        self.port = port
        self.available = False
        self.monitor = None
        self._find_peaks = None

        self._wave = deque(maxlen=300)
        self._wave_t = deque(maxlen=300)
        self._last_peak_time = None

        self._hr_readings = []   # pulse rates collected in the current window
        self._rr_intervals = []  # RR intervals (s) collected in the current window
        self._last_hr = None

    def start(self):
        if self.port is None:
            print("[HR] Wearable disabled -> HR/RMMSD will be imputed by the model.")
            return
        try:
            from scipy.signal import find_peaks
            from cms50d import CMS50D

            self._find_peaks = find_peaks
            self.monitor = CMS50D(self.port)
            self.monitor.connect()
            self.monitor.start_live_acquisition()
            self.available = True
            print(f"[HR] CMS50D connected on {self.port}.")
        except Exception as exc:
            self.available = False
            self.monitor = None
            print(f"[HR] Could not connect to CMS50D on {self.port}: {exc}")
            print("[HR] Continuing without wearable; HR/RMMSD will be imputed.")

    def poll(self):
        """Drain the device queue; call frequently from the main loop."""
        if not self.available:
            return

        min_beat_distance = int(self.TARGET_FPS * 60 / self.MIN_HR_BPM)

        while True:
            data = self.monitor.get_latest_data()
            if data is None:
                break

            waveform = data.get("waveform")
            timestamp = data.get("timestamp")
            if waveform is not None and timestamp is not None:
                self._wave.append(waveform)
                self._wave_t.append(timestamp)

                if len(self._wave) >= 50:
                    signal = np.array(list(self._wave), dtype=float)
                    times = list(self._wave_t)
                    peaks, _ = self._find_peaks(
                        signal,
                        distance=min_beat_distance,
                        height=signal.mean() * 0.7,
                        prominence=signal.std() * 0.5,
                    )
                    if len(peaks) >= 2:
                        for i in range(1, len(peaks)):
                            t_prev = times[peaks[i - 1]]
                            t_curr = times[peaks[i]]
                            if self._last_peak_time is not None and t_prev <= self._last_peak_time:
                                continue  # already processed this pair
                            rr_sec = (t_curr - t_prev).total_seconds()
                            if 0.3 < rr_sec < 2.0:
                                self._rr_intervals.append(rr_sec)
                            self._last_peak_time = t_prev

            hr = data.get("pulse_rate")
            if hr is not None:
                self._hr_readings.append(hr)
                self._last_hr = hr

    def consume_window(self):
        """Return (hr_mean, rmssd) for the current window and reset it."""
        if self.available and self._hr_readings:
            hr_mean = float(np.mean(self._hr_readings))
        else:
            hr_mean = float("nan")

        if self.available and len(self._rr_intervals) >= 2:
            rr = np.array(self._rr_intervals, dtype=float)
            rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        else:
            rmssd = float("nan")

        self._hr_readings = []
        self._rr_intervals = []
        return hr_mean, rmssd

    @property
    def live_bpm(self):
        return self._last_hr

    def stop(self):
        if self.monitor is not None:
            try:
                self.monitor.stop_live_acquisition()
            except Exception:
                pass
            try:
                self.monitor.disconnect()
            except Exception:
                pass


# =============================================================================
# FEATURE ASSEMBLY
# =============================================================================

def landmarks_from_results(results):
    """Convert MediaPipe Pose results into the dict list PostureAnalyzer wants."""
    if results.pose_landmarks is None:
        return None
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in results.pose_landmarks.landmark
    ]


def assemble_feature_row(posture_rows, blink_rows, hr_mean, rmssd):
    """Combine the per-module window outputs into one 18-feature row.

    Posture is the only mandatory sensor: without it the model would have to
    median-impute every body feature, which collapses accuracy to chance, so
    the window is dropped (returns None).

    Blink and heart are optional. When their window is missing their features
    are left as NaN, so the model's median imputer fills them. As long as the
    posture is present this has a negligible accuracy cost.
    """
    if not posture_rows:
        return None

    posture = posture_rows[0]
    row = {name: posture[name] for name in POSTURE_FEATURES}

    if blink_rows:
        blink = blink_rows[0]
        row.update({name: blink[name] for name in BLINK_FEATURES})
    else:
        row.update({name: float("nan") for name in BLINK_FEATURES})

    row["HR"] = hr_mean
    row["RMMSD"] = rmssd
    return row


# =============================================================================
# OVERLAY
# =============================================================================

COL_WHITE = (255, 255, 255)
COL_DIM = (160, 160, 160)
COL_GREEN = (0, 200, 0)
COL_AMBER = (0, 200, 255)
COL_RED = (60, 60, 255)

STATE_COLORS = {
    "normal": COL_GREEN,
    "sleepy": COL_AMBER,
    "stress": COL_RED,
    "abnormal": COL_RED,
    "anomalous": COL_RED,
}


def _put(cv2, frame, text, x, y, scale, color, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_overlay(cv2, frame, *, remaining, posture_ok, ear, face_detected,
                 hr_live, hr_available, prediction, p_count, b_count):
    """Draw live readouts + last minute's prediction."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    _put(cv2, frame, "LIVE", 12, 24, 0.6, COL_WHITE, 2)
    _put(cv2, frame, f"Next reading in: {max(0.0, remaining):4.1f}s",
         12, 48, 0.5, COL_WHITE)

    ear_txt = f"{ear:.3f}" if ear is not None else "--"
    _put(cv2, frame, f"Eyes (EAR): {ear_txt}", 12, 70, 0.45,
         COL_GREEN if face_detected else COL_DIM)
    _put(cv2, frame, f"Posture: {'tracked' if posture_ok else 'not visible'}",
         12, 90, 0.45, COL_GREEN if posture_ok else COL_DIM)

    if hr_available:
        hr_txt = f"{hr_live} bpm" if hr_live is not None else "warming up"
        _put(cv2, frame, f"Wearable: {hr_txt}", 12, 110, 0.45, COL_GREEN)
    else:
        _put(cv2, frame, "Wearable: not connected", 12, 110, 0.45, COL_DIM)

    _put(cv2, frame, f"Samples this min: posture={p_count} blink={b_count}",
         12, 132, 0.42, COL_DIM)

    # Last minute's prediction -- big banner, bottom of the frame.
    h, w = frame.shape[:2]
    banner = frame.copy()
    cv2.rectangle(banner, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.6, frame, 0.4, 0, frame)

    if prediction is None:
        _put(cv2, frame, "STATE: waiting for first full minute...",
             12, h - 38, 0.7, COL_DIM, 2)
    else:
        label = prediction["label"]
        color = STATE_COLORS.get(label, COL_WHITE)
        _put(cv2, frame, f"STATE: {label.upper()}", 12, h - 38, 0.9, color, 2)
        if prediction["proba"]:
            proba_txt = "   ".join(
                f"{cls}:{prediction['proba'][cls]:.0%}"
                for cls in sorted(prediction["proba"])
            )
            _put(cv2, frame, proba_txt, 12, h - 12, 0.55, COL_WHITE)

    _put(cv2, frame, "q = quit", w - 110, 24, 0.5, COL_DIM)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time wellbeing monitor (camera + wearable -> ML)."
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Webcam index (default: 0).")
    parser.add_argument("--port", default="COM3",
                        help="CMS50D serial port, or 'HID' (default: COM3).")
    parser.add_argument("--no-wearable", action="store_true",
                        help="Run without the heart-rate wearable.")
    parser.add_argument("--task", default=None, choices=["binary", "3class"],
                        help="Skip the first prompt: 'binary' or '3class'.")
    parser.add_argument("--learning", default=None,
                        choices=["supervised", "unsupervised"],
                        help="Skip the second prompt: 'supervised' or 'unsupervised'.")
    parser.add_argument("--model", default=None,
                        help="Skip the model prompt and use this model by name "
                             "(e.g. 'Random Forest'). Any step left unset is "
                             "asked interactively at startup.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  REAL-TIME WELLBEING MONITOR")
    print("=" * 70)

    # --- Guided selection (#classes -> supervised? -> exact model) + train ---
    model = choose_model(args)
    model.fit()

    # --- Heavy imports only now (so --help stays fast) ---
    import cv2
    import mediapipe as mp

    import config as posture_config
    from posture_analyzer import PostureAnalyzer
    from feature_extractor import FeatureExtractor
    from blink_analyzer import BlinkAnalyzer
    from blink_feature_extractor import BlinkFeatureExtractor
    import overlay_drawing as overlay

    window_size = posture_config.WINDOW_SIZE  # 60s, same as the training windows

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=posture_config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=posture_config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=posture_config.MP_MIN_TRACKING_CONFIDENCE,
    )
    posture_analyzer = PostureAnalyzer()
    blink_analyzer = BlinkAnalyzer()

    # --- Wearable ---
    hr_monitor = HeartRateMonitor(None if args.no_wearable else args.port)
    hr_monitor.start()

    # --- Camera ---
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[FATAL] Could not open camera {args.camera_index}. Exiting.")
        hr_monitor.stop()
        return

    posture_fe = FeatureExtractor()
    blink_fe = BlinkFeatureExtractor()

    window_start = time.time()
    last_ear = None
    last_face = False
    last_posture_ok = False
    prediction = None

    print(f"\n[INFO] Window size: {window_size}s. A result is produced every minute.")
    print("[INFO] Click the video window, then press q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            t_rel = now - window_start

            # --- Posture (MediaPipe Pose -> 3 raw values) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_results = pose.process(rgb)
            landmarks = landmarks_from_results(pose_results)
            values = posture_analyzer.extract_values(landmarks)
            last_posture_ok = values is not None
            if values is not None:
                shoulder, head_tilt, fhp = values
                posture_fe.add_sample(t_rel, shoulder, head_tilt, fhp)

            # --- Blinking (EAR per frame) ---
            blink_result = blink_analyzer.analyze(frame)
            last_face = blink_result["face_detected"]
            last_ear = blink_result["ear"]
            if last_face and last_ear is not None:
                blink_fe.add_sample(t_rel, last_ear, blink_result["is_closed"])

            # --- Wearable ---
            hr_monitor.poll()

            # --- Body / eye overlays drawn on top of the person ---
            overlay.draw_posture_skeleton(cv2, frame, landmarks, values, posture_config)
            overlay.draw_eye_boxes(cv2, frame, blink_result.get("left_eye", []),
                                   blink_result.get("right_eye", []),
                                   blink_result.get("is_closed", False))

            # --- Info panels + prediction banner ---
            draw_overlay(
                cv2, frame,
                remaining=window_size - t_rel,
                posture_ok=last_posture_ok,
                ear=last_ear,
                face_detected=last_face,
                hr_live=hr_monitor.live_bpm,
                hr_available=hr_monitor.available,
                prediction=prediction,
                p_count=posture_fe.get_sample_count(),
                b_count=blink_fe.get_sample_count(),
            )
            cv2.imshow(WINDOW_NAME, frame)

            # --- End of a 60s window: build features, classify, output ---
            if t_rel >= window_size:
                posture_rows = posture_fe.compute_windows()
                blink_rows = blink_fe.compute_windows()
                hr_mean, rmssd = hr_monitor.consume_window()

                feature_row = assemble_feature_row(
                    posture_rows, blink_rows, hr_mean, rmssd
                )

                print("-" * 70)
                print(f"[{time.strftime('%H:%M:%S')}] Minute complete.")
                if feature_row is None:
                    print("  Posture not detected this minute "
                          "(body not in frame). No prediction.")
                else:
                    prediction = model.predict_row(feature_row)
                    hr_txt = "n/a" if np.isnan(hr_mean) else f"{hr_mean:.1f} bpm"
                    rmssd_txt = "n/a" if np.isnan(rmssd) else f"{rmssd:.4f}"
                    print(f"  HR={hr_txt}  RMMSD={rmssd_txt}")
                    print(f"  >>> STATE: {prediction['label'].upper()}")
                    if prediction["proba"]:
                        print("  probabilities: " + ", ".join(
                            f"{cls}={prediction['proba'][cls]:.0%}"
                            for cls in sorted(prediction["proba"])
                        ))

                posture_fe.reset()
                blink_fe.reset()
                window_start = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        pose.close()
        blink_analyzer.close()
        hr_monitor.stop()
        cv2.destroyAllWindows()
        print("[INFO] System shut down.")


if __name__ == "__main__":
    main()
