"""
main.TestVideo.py - Offline wellbeing analysis on a pre-recorded test video.

This is the file-based counterpart of main.realtime.py. Instead of a live
webcam + wearable, it works on recorded inputs placed in the "Test Video"
folder at the project root:

  - A video file (a few minutes long): the blink (EAR) and posture features are
    extracted from it MINUTE BY MINUTE -- one non-overlapping 60-second window
    per minute (using the same Scripts/Posture and Scripts/Blinking extractors,
    keeping only the windows that start at 0:00, 1:00, 2:00, ...).
  - Heart_TestVideo.csv : the synthetic heart signal (HR + RMMSD), one row per
    minute. It is aligned to the per-minute windows BY POSITION (row 0 -> first
    minute, row 1 -> second minute, ...), like Scripts/build_training_dataset.py.
    The CSV may also carry a 'Target' column (the true state of that minute):
    it is kept for reference only and is NOT used here -- the model predicts the
    state from the features, so Target is never read. It is left in place so it
    can later be used as ground truth to score predicted-vs-real accuracy.

For every minute it assembles the same 18-feature vector as the training data
(9 posture + 7 blinking + HR + RMMSD) and feeds it to the chosen model,
printing the predicted state for that minute of the video (normal / sleepy /
stress, or normal / abnormal in the binary tasks).

Model selection is identical to main.realtime.py: a guided menu (no default)
asks for  #classes -> supervised/unsupervised -> the exact model, all of which
can be pre-filled with --task / --learning / --model.

The heart CSV is optional: if it is missing (or --no-heart is given) the two
heart features are left missing and imputed by the model, so the video can
still be analysed from the camera features alone.

While it runs it opens a window showing the video with the body skeleton, the
measured posture parameters, boxes around the eyes, and the per-minute state.

Usage:
    python main.TestVideo.py                                  # guided menu, auto-pick the video
    python main.TestVideo.py --video "Test Video/session.mp4"
    python main.TestVideo.py --task 3class --learning supervised --model "Random Forest"
    python main.TestVideo.py --task binary --learning unsupervised
    python main.TestVideo.py --no-heart                       # ignore the heart CSV
    python main.TestVideo.py --no-show                        # console only, no window
"""

from __future__ import annotations

import argparse
import glob
import sys
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
ML_DIR = SCRIPTS_DIR / "ML"
PROJECT_ROOT = SCRIPTS_DIR.parent
SUPERVISED_DIR = PROJECT_ROOT / "data" / "supervised"

TEST_VIDEO_DIR = PROJECT_ROOT / "Test Video"
HEART_CSV = TEST_VIDEO_DIR / "Heart_TestVideo.csv"

# The sub-modules import their config as a bare name ("config" / "blink_config"),
# so their folders must be importable directly. ML is added so we can reuse the
# exact model definitions (model_factories / anomaly_models). APP_DIR is added
# so we can import the shared overlay_drawing module (same folder).
for _module_dir in (APP_DIR, POSTURE_DIR, BLINKING_DIR, ML_DIR):
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

# Which supervised CSVs and which label to assign for each task.
TASK_FILES = {
    "binary": [("normal.csv", "normal"), ("anomalous.csv", "abnormal")],
    "3class": [("normal.csv", "normal"), ("sleepy.csv", "sleepy"), ("stress.csv", "stress")],
}


# =============================================================================
# MODEL SELECTION -- guided menu: #classes -> supervised? -> exact model
# (identical to main.realtime.py)
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
# VIDEO INPUT + FEATURE EXTRACTION
# =============================================================================

VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def find_video(arg):
    """Resolve the video path from --video, or auto-pick one in 'Test Video'."""
    if arg:
        candidate = Path(arg)
        if not candidate.is_absolute():
            # try relative to cwd, then to the project root
            for base in (Path.cwd(), PROJECT_ROOT):
                if (base / candidate).exists():
                    return (base / candidate).resolve()
        return candidate.resolve()

    if not TEST_VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"Folder not found: {TEST_VIDEO_DIR}")

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(str(TEST_VIDEO_DIR / ext)))
    videos.sort()
    if not videos:
        raise FileNotFoundError(
            f"No video found in {TEST_VIDEO_DIR}. Drop a .mp4/.mov/.avi/.mkv there "
            f"or pass --video <path>."
        )
    return Path(videos[0]).resolve()


def landmarks_from_results(results):
    """Convert MediaPipe Pose results into the dict list PostureAnalyzer wants."""
    if results.pose_landmarks is None:
        return None
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in results.pose_landmarks.landmark
    ]


def draw_video_info(cv2, frame, *, t, duration, minute, ear, face, posture_ok,
                    hr_csv, rmssd_csv):
    """Compact top-left info panel for the test-video playback."""
    box = frame.copy()
    cv2.rectangle(box, (0, 0), (330, 150), (0, 0, 0), -1)
    cv2.addWeighted(box, 0.55, frame, 0.45, 0, frame)

    def put(text, x, y, scale, color, th=1):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, th, cv2.LINE_AA)

    green, dim, white = (90, 220, 90), (160, 160, 160), (255, 255, 255)
    put("TEST VIDEO", 12, 26, 0.6, white, 2)
    put(f"Time {fmt_time(t)} / {fmt_time(duration)}    Minute {minute}",
        12, 50, 0.5, white)
    ear_txt = f"{ear:.3f}" if ear is not None else "--"
    put(f"Eyes (EAR): {ear_txt}", 12, 72, 0.45, green if face else dim)
    put(f"Posture: {'tracked' if posture_ok else 'not visible'}",
        12, 92, 0.45, green if posture_ok else dim)
    if np.isnan(hr_csv):
        put("Heart (this min): imputed", 12, 112, 0.45, dim)
    else:
        put(f"Heart (this min): HR {hr_csv:.0f}  RMMSD {rmssd_csv:.3f}",
            12, 112, 0.45, green)
    put("q = quit", 12, 134, 0.42, dim)


def _print_minute(minute_index, start, hr, rmssd, prediction):
    span = f"{fmt_time(start)}-{fmt_time(start + 60)}"
    hr_txt = "n/a" if np.isnan(hr) else f"{hr:.1f}"
    rmssd_txt = "n/a" if np.isnan(rmssd) else f"{rmssd:.4f}"
    line = (f"  [{span}]  HR={hr_txt:>6}  RMMSD={rmssd_txt:>8}  "
            f"->  {prediction['label'].upper()}")
    if prediction["proba"]:
        line += "   (" + ", ".join(
            f"{cls}={prediction['proba'][cls]:.0%}"
            for cls in sorted(prediction["proba"])
        ) + ")"
    print(line)


def analyze_video(video_path, model, heart, show=True):
    """Play the video, draw the overlays, and predict the state once per minute.

    Mirrors the real-time loop (tumbling 60s windows, reset each minute) but
    reads frames from the file and the synthetic heart values from the CSV
    (aligned by position: minute 0 -> heart row 0, minute 1 -> row 1, ...).

    Returns a list of (minute_index, start_seconds, hr, rmssd, prediction).
    """
    import cv2
    import mediapipe as mp

    import config as posture_config
    from posture_analyzer import PostureAnalyzer
    from feature_extractor import FeatureExtractor
    from blink_analyzer import BlinkAnalyzer
    from blink_feature_extractor import BlinkFeatureExtractor
    import overlay_drawing as overlay

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    window_size = posture_config.WINDOW_SIZE
    print(f"[VIDEO] {video_path.name}: FPS={fps:.1f}, frames={total_frames}, "
          f"duration={duration:.1f}s")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=posture_config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=posture_config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=posture_config.MP_MIN_TRACKING_CONFIDENCE,
    )
    posture_analyzer = PostureAnalyzer()
    blink_analyzer = BlinkAnalyzer()
    posture_fe = FeatureExtractor()
    blink_fe = BlinkFeatureExtractor()

    window_name = "Test Video - Wellbeing Analysis"
    results = []
    last_pred = None
    minute_index = 0
    minute_start = 0.0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_index / fps if fps > 0 else 0.0
            t_rel = t - minute_start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            landmarks = landmarks_from_results(pose.process(rgb))
            values = posture_analyzer.extract_values(landmarks)
            if values is not None:
                posture_fe.add_sample(t_rel, *values)

            blink_result = blink_analyzer.analyze(frame)
            face = blink_result["face_detected"]
            ear = blink_result["ear"]
            if face and ear is not None:
                blink_fe.add_sample(t_rel, ear, blink_result["is_closed"])

            if show:
                hr_csv, rmssd_csv = (heart[minute_index]
                                     if minute_index < len(heart)
                                     else (float("nan"), float("nan")))
                overlay.draw_posture_skeleton(cv2, frame, landmarks, values, posture_config)
                overlay.draw_eye_boxes(cv2, frame, blink_result.get("left_eye", []),
                                       blink_result.get("right_eye", []),
                                       blink_result.get("is_closed", False))
                draw_video_info(cv2, frame, t=t, duration=duration,
                                minute=minute_index + 1, ear=ear, face=face,
                                posture_ok=values is not None,
                                hr_csv=float(hr_csv), rmssd_csv=float(rmssd_csv))
                overlay.draw_state_banner(cv2, frame, last_pred,
                                          waiting_text="analyzing first minute...")
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped by user.")
                    break
            elif frame_index % 500 == 0:
                pct = frame_index / total_frames * 100 if total_frames > 0 else 0
                print(f"  Frame {frame_index}/{total_frames} ({pct:.0f}%)")

            # one tumbling window per completed minute
            if t_rel >= window_size:
                posture_rows = posture_fe.compute_windows()
                blink_rows = blink_fe.compute_windows()
                if posture_rows and blink_rows:
                    p, b = posture_rows[0], blink_rows[0]
                    hr, rmssd = (heart[minute_index]
                                 if minute_index < len(heart)
                                 else (float("nan"), float("nan")))
                    hr, rmssd = float(hr), float(rmssd)
                    row = {name: p[name] for name in POSTURE_FEATURES}
                    row.update({name: b[name] for name in BLINK_FEATURES})
                    row["HR"], row["RMMSD"] = hr, rmssd
                    pred = model.predict_row(row)
                    last_pred = pred
                    results.append((minute_index, minute_start, hr, rmssd, pred))
                    _print_minute(minute_index, minute_start, hr, rmssd, pred)
                else:
                    print(f"  [minute {minute_index + 1}] not enough valid camera "
                          f"data -> skipped.")
                minute_index += 1
                posture_fe.reset()
                blink_fe.reset()
                minute_start = t
            frame_index += 1
    finally:
        cap.release()
        pose.close()
        blink_analyzer.close()
        if show:
            cv2.destroyAllWindows()

    return results


def load_heart_series(path, use_heart):
    """Load (HR, RMMSD) pairs from Heart_TestVideo.csv, aligned by position.

    Returns a list of (hr, rmssd) floats. Empty list if disabled/missing.
    """
    if not use_heart:
        print("[HEART] Disabled -> HR/RMMSD will be imputed by the model.")
        return []
    if not path.exists():
        print(f"[HEART] {path.name} not found -> HR/RMMSD will be imputed.")
        return []

    import pandas as pd

    # The training heart files are ';'-separated with a BOM; try that first,
    # then fall back to comma if only one column comes back.
    heart = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if heart.shape[1] == 1:
        heart = pd.read_csv(path, sep=",", encoding="utf-8-sig")

    # Only HR and RMMSD are used as features. A 'Target' column, if present,
    # is the true state of the minute and is intentionally ignored here (kept
    # in the CSV only as reference / future ground truth for accuracy scoring).
    columns = {c.strip().lower(): c for c in heart.columns}
    hr_col = columns.get("hr")
    rmssd_col = columns.get("rmmsd") or columns.get("rmssd")
    if hr_col is None or rmssd_col is None:
        print(f"[HEART] {path.name} has no HR/RMMSD columns (found {list(heart.columns)}) "
              f"-> HR/RMMSD will be imputed.")
        return []

    hr = pd.to_numeric(heart[hr_col], errors="coerce")
    rmssd = pd.to_numeric(heart[rmssd_col], errors="coerce")
    series = list(zip(hr.tolist(), rmssd.tolist()))
    print(f"[HEART] Loaded {len(series)} HR/RMMSD rows from {path.name}.")
    return series


def fmt_time(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline wellbeing analysis on a test video + heart CSV."
    )
    parser.add_argument("--video", default=None,
                        help="Path to the video. Default: first video in 'Test Video'.")
    parser.add_argument("--no-heart", action="store_true",
                        help="Ignore Heart_TestVideo.csv (impute HR/RMMSD).")
    parser.add_argument("--no-show", action="store_true",
                        help="Process without the video window (console only).")
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
    print("  TEST-VIDEO WELLBEING ANALYSIS")
    print("=" * 70)

    video_path = find_video(args.video)

    # --- Guided selection (#classes -> supervised? -> exact model) + train ---
    model = choose_model(args)
    model.fit()

    # --- Synthetic heart signal, one row per minute, aligned by position ---
    heart = load_heart_series(HEART_CSV, use_heart=not args.no_heart)

    # --- Play the video, draw overlays, classify each minute live ---
    print("\n" + "=" * 70)
    print(f"  PER-MINUTE PREDICTIONS  (model: {model.describe()})")
    print("=" * 70)
    results = analyze_video(video_path, model, heart, show=not args.no_show)

    if not results:
        print("\n[RESULT] No per-minute predictions were produced "
              "(video too short or face/body not visible). Nothing to summarise.")
        return

    # --- Summary ---
    counts = {}
    for _, _, _, _, prediction in results:
        counts[prediction["label"]] = counts.get(prediction["label"], 0) + 1

    print("\n" + "-" * 70)
    print(f"  SUMMARY over {len(results)} minutes:")
    for label in sorted(counts):
        share = counts[label] / len(results) * 100
        print(f"    {label:10s}: {counts[label]:3d} min ({share:.0f}%)")
    overall = max(counts, key=counts.get)
    print(f"  Most frequent state: {overall.upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
