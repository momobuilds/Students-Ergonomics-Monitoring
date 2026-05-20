"""
main2.0.py - Combined blinking + posture monitor.

This script uses one webcam stream and evaluates everything computed by:
- Scripts/Blinking/main.py
- Scripts/Posture/main.py

Examples:
    python main2.0.py                                        # Basic threshold mode
    python main2.0.py --calibrate                            # Calibrate both
    python main2.0.py --collect-posture-data                 # Collect posture training data
    python main2.0.py --collect-blink-data                   # Collect blink training data
    python main2.0.py --train                                # Train both modules (SVM+RF, auto-pick best)
    python main2.0.py --posture-mode ml --blink-mode ml      # Use ML for both

Keyboard controls during monitoring:
    q = quit
    c = recalibrate posture and blinking
    s = save posture and blinking CSV snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
POSTURE_DIR = SCRIPT_DIR / "Posture"
BLINKING_DIR = SCRIPT_DIR / "Blinking"

WINDOW_NAME = "Blinking + Posture Monitor"

cv2: Any = None
np: Any = None
posture_config: Any = None
PostureClassifier: Any = None
PoseExtractor: Any = None
PostureAnalyzer: Any = None
PostureMonitor: Any = None
BlinkAnalyzer: Any = None
BlinkClassifier: Any = None
BlinkMonitor: Any = None


def ensure_module_paths() -> None:
    # The existing modules import local files as plain names:
    # - Posture modules import "config"
    # - Blinking modules import "blink_config"
    for module_dir in [str(BLINKING_DIR), str(POSTURE_DIR)]:
        if module_dir in sys.path:
            sys.path.remove(module_dir)
    sys.path.insert(0, str(BLINKING_DIR))
    sys.path.insert(0, str(POSTURE_DIR))


def load_training_modules() -> None:
    """Import only modules needed for model training and data paths."""
    global posture_config
    global PostureClassifier
    global BlinkClassifier

    if PostureClassifier is not None and BlinkClassifier is not None:
        return

    ensure_module_paths()

    from Posture import config as posture_config_module
    from Posture.classifier import PostureClassifier as PostureClassifierClass
    from Blinking.blink_classifier import BlinkClassifier as BlinkClassifierClass

    posture_config = posture_config_module
    PostureClassifier = PostureClassifierClass
    BlinkClassifier = BlinkClassifierClass


def load_runtime_modules() -> None:
    """Import webcam modules only after argparse handled --help."""
    global cv2
    global np
    global PoseExtractor
    global PostureAnalyzer
    global PostureMonitor
    global BlinkAnalyzer
    global BlinkMonitor

    if cv2 is not None:
        return

    load_training_modules()

    import cv2 as cv2_module
    import numpy as np_module
    from Posture.pose_extractor import PoseExtractor as PoseExtractorClass
    from Posture.posture_analyzer import PostureAnalyzer as PostureAnalyzerClass
    from Posture.posture_monitor import PostureMonitor as PostureMonitorClass
    from Blinking.blink_analyzer import BlinkAnalyzer as BlinkAnalyzerClass
    from Blinking.blink_monitor import BlinkMonitor as BlinkMonitorClass

    cv2 = cv2_module
    np = np_module
    PoseExtractor = PoseExtractorClass
    PostureAnalyzer = PostureAnalyzerClass
    PostureMonitor = PostureMonitorClass
    BlinkAnalyzer = BlinkAnalyzerClass
    BlinkMonitor = BlinkMonitorClass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined blinking and posture monitor")
    parser.add_argument("--camera-index", type=int, default=0)

    parser.add_argument(
        "--posture-mode",
        choices=["thresholds", "ml"],
        default="thresholds",
        help="Posture classification mode.",
    )
    parser.add_argument(
        "--blink-mode",
        choices=["threshold", "ml"],
        default="threshold",
        help="Blinking classification mode.",
    )

    parser.add_argument("--calibrate", action="store_true", help="Calibrate posture and blinking.")
    parser.add_argument("--calibrate-posture", action="store_true", help="Calibrate only posture.")
    parser.add_argument("--calibrate-blink", action="store_true", help="Calibrate only blinking.")
    parser.add_argument("--blink-calibration-seconds", type=float, default=8.0)

    parser.add_argument("--train", action="store_true", help="Train posture and blinking models, then exit.")
    parser.add_argument("--train-posture", action="store_true", help="Train posture model, then exit.")
    parser.add_argument("--train-blink", action="store_true", help="Train blinking model, then exit.")

    parser.add_argument(
        "--collect-posture-data",
        action="store_true",
        help="Run guided posture data collection for Good and Slouching.",
    )
    parser.add_argument(
        "--collect-blink-data",
        action="store_true",
        help="Run guided blink data collection for Normal and Tired.",
    )
    return parser.parse_args()


def load_posture_calibration(analyzer: Any) -> None:
    try:
        with open(posture_config.CALIBRATION_PATH, "r", encoding="utf-8") as calibration_file:
            analyzer.set_baseline(json.load(calibration_file))
    except FileNotFoundError:
        return


def run_posture_calibration(
    extractor: Any,
    analyzer: Any,
) -> dict | None:
    print("\n[POSTURE CALIBRATION] Sit upright in your normal good posture.")
    print(f"[POSTURE CALIBRATION] Recording for {posture_config.CALIBRATION_DURATION} seconds.")
    print("[POSTURE CALIBRATION] Click the camera window, then press q to skip.\n")

    samples = []
    start = time.time()

    while time.time() - start < posture_config.CALIBRATION_DURATION:
        if not extractor.update():
            continue

        data = extractor.get_current_data()
        frame = data["frame"]
        landmarks = data["landmarks"]

        if landmarks is not None:
            sample = analyzer.compute_calibration_data(landmarks)
            if sample is not None:
                samples.append(sample)

        if frame is not None:
            remaining = posture_config.CALIBRATION_DURATION - (time.time() - start)
            cv2.putText(
                frame,
                f"Posture calibration: sit upright ({remaining:.1f}s, {len(samples)} samples)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                posture_config.COLOR_WARNING,
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[POSTURE CALIBRATION] Skipped.")
            return None

    if not samples:
        print("[POSTURE CALIBRATION] No data collected.")
        return None

    baseline = {}
    all_keys = set()
    for sample in samples:
        all_keys.update(sample.keys())

    for key in all_keys:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            baseline[key] = float(np.mean(values))

    with open(posture_config.CALIBRATION_PATH, "w", encoding="utf-8") as calibration_file:
        json.dump(baseline, calibration_file, indent=2)

    print(f"[POSTURE CALIBRATION] Baseline saved with {len(baseline)} parameters.")
    return baseline


def run_blink_calibration(
    extractor: Any,
    analyzer: Any,
    duration_seconds: float,
) -> None:
    print("\n[BLINK CALIBRATION] Keep your eyes naturally open and look at the screen.")
    print(f"[BLINK CALIBRATION] Recording for {duration_seconds:.1f} seconds.")
    print("[BLINK CALIBRATION] Click the camera window, then press q to skip.\n")

    samples: list[float] = []
    start = time.time()

    while time.time() - start < duration_seconds:
        if not extractor.update():
            continue

        frame = extractor.get_current_data()["frame"]
        if frame is None:
            continue

        analysis = analyzer.analyze(frame)
        if analysis["face_detected"] and analysis["ear"] is not None:
            samples.append(float(analysis["ear"]))

        # NOTE: do NOT call analyzer.draw_overlay() here — we use our own overlay
        remaining = duration_seconds - (time.time() - start)
        cv2.putText(
            frame,
            f"Blink calibration: keep eyes open ({remaining:.1f}s, {len(samples)} samples)",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[BLINK CALIBRATION] Skipped.")
            return

    payload = analyzer.calibrate_from_samples(samples)
    print(
        "[BLINK CALIBRATION] Saved. "
        f"Baseline EAR={payload['baseline_open_ear']:.3f}, "
        f"threshold={payload['ear_threshold']:.3f}"
    )


def run_posture_data_collection(
    extractor: Any,
    analyzer: Any,
    classifier: Any,
) -> None:
    for label in ["Good", "Slouching"]:
        print(f"\n[POSTURE DATA] Get into your '{label}' posture.")
        print(f"[POSTURE DATA] Recording for {posture_config.TRAINING_COLLECTION_DURATION} seconds.")
        input("[POSTURE DATA] Press Enter when ready...")

        classifier.start_collection(label)
        start = time.time()

        while time.time() - start < posture_config.TRAINING_COLLECTION_DURATION:
            if not extractor.update():
                continue

            data = extractor.get_current_data()
            frame = data["frame"]
            landmarks = data["landmarks"]

            if landmarks is not None:
                classifier.add_sample(analyzer.get_feature_vector(landmarks))

            if frame is not None:
                remaining = posture_config.TRAINING_COLLECTION_DURATION - (time.time() - start)
                count = classifier.get_collection_count()
                cv2.putText(
                    frame,
                    f"Collecting posture '{label}': {count} samples ({remaining:.1f}s left)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    posture_config.COLOR_WARNING,
                    2,
                )
                cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        classifier.stop_collection()

    print("\n[POSTURE DATA] Complete. Run with --train-posture to build the model.")


def _run_blink_data_collection(
    extractor: Any,
    analyzer: Any,
    monitor: Any,
    classifier: Any,
) -> None:
    """
    Guided blink data collection — mirrors run_posture_data_collection.
    Collects 'Normal' then 'Tired' sessions.
    """
    import blink_config

    for label in ["Normal", "Tired"]:
        print(f"\n[BLINK DATA] Get into your '{label}' blinking state.")
        print(f"[BLINK DATA] Recording for {blink_config.TRAINING_COLLECTION_DURATION}s.")
        input(f"[BLINK DATA] Press Enter when ready...")

        classifier.start_collection(label)
        start = time.time()

        while time.time() - start < blink_config.TRAINING_COLLECTION_DURATION:
            if not extractor.update():
                continue

            data = extractor.get_current_data()
            frame = data["frame"]

            if frame is None:
                continue

            analysis = analyzer.analyze(frame)
            state = monitor.update(analysis)

            # add_sample expects BlinkMonitor state dict
            classifier.add_sample(state)

            remaining = blink_config.TRAINING_COLLECTION_DURATION - (time.time() - start)
            count = classifier.get_collection_count()
            cv2.putText(
                frame,
                f"Collecting blink '{label}': {count} samples ({remaining:.1f}s left)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        saved = classifier.stop_collection()
        print(f"[BLINK DATA] Saved {saved} samples for '{label}'.")

    print("\n[BLINK DATA] Complete. Run with --train-blink to build the model.")


# =============================================================================
# OVERLAY — all drawing functions
# =============================================================================

FOOTER_H = 50          # px — two lines of text, readable
ALERT_LINE_H = 30      # px per alert row
PANEL_X = 10           # left margin
PANEL_W = 280          # panel width
LINE_H = 22            # row height inside panels
GAP = 8                # vertical gap between the two panels
FONT_SCALE = 0.45      # metric rows
HEADER_SCALE = 0.52    # section headers (POSTURE / BLINKING)

COL_WHITE = (255, 255, 255)
COL_GRAY = (200, 200, 200)
COL_DIM = (140, 140, 140)
COL_GREEN = (0, 200, 0)
COL_RED = (60, 60, 255)       # BGR red-ish
COL_ALERT_BG = (0, 0, 170)   # dark red background
COL_CYAN = (220, 220, 100)


def _transparent_rect(
    frame, x1: int, y1: int, x2: int, y2: int,
    color: tuple = (0, 0, 0), alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put(frame, text: str, x: int, y: int, scale: float,
         color: tuple, thickness: int = 1) -> None:
    """Shorthand for cv2.putText with LINE_AA."""
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Left-side panels: POSTURE on top, BLINKING below
# ---------------------------------------------------------------------------

def _draw_panels(frame, posture_state: dict, blink_state: dict) -> None:
    """Draw both metric panels stacked on the left."""

    # ── Build posture rows ──────────────────────────────────────────────
    p_rows: list[tuple[str, tuple]] = []

    status = posture_state.get("status", "Unknown")
    if status == "Good":
        sc = COL_GREEN
    elif status == "Slouching":
        sc = COL_RED
    else:
        sc = COL_GRAY
    p_rows.append((f"Status: {status}", sc))

    for name, metric in posture_state.get("metrics", {}).items():
        if metric is None:
            continue
        good = metric.get("is_good", True)
        color = COL_GREEN if good else COL_RED
        label = name.replace("_", " ").title()
        val = metric.get("value", "N/A")
        unit = metric.get("unit", "")
        p_rows.append((f"  {label}: {val}{unit}", color))

    ml_conf = posture_state.get("ml_confidence")
    if ml_conf is not None:
        p_rows.append((f"  ML conf: {ml_conf:.0%}", COL_CYAN))

    # ── Build blink rows ────────────────────────────────────────────────
    b_rows: list[tuple[str, tuple]] = [
        (f"  EAR:        {_format_float(blink_state.get('ear'), 3)}", COL_GRAY),
        (f"  Blink freq: {blink_state.get('blink_frequency_bpm', 0.0):.1f} bpm", COL_GRAY),
        (f"  PERCLOS:    {blink_state.get('perclos', 0.0) * 100:.1f}%", COL_GRAY),
        (f"  IBI last:   {_format_seconds(blink_state.get('last_inter_blink_interval'))}", COL_GRAY),
        (f"  IBI mean:   {_format_seconds(blink_state.get('mean_inter_blink_interval'))}", COL_GRAY),
        (f"  Blinks/win: {blink_state.get('blink_count_window', 0)}", COL_GRAY),
    ]

    # ── Compute sizes ───────────────────────────────────────────────────
    # Each section = 1 header row + N data rows
    posture_h = LINE_H * (1 + len(p_rows)) + 8
    blink_h = LINE_H * (1 + len(b_rows)) + 8
    total_h = posture_h + GAP + blink_h

    # ── Draw background rectangles ──────────────────────────────────────
    y_top = 8
    _transparent_rect(frame, PANEL_X - 4, y_top - 4,
                      PANEL_X + PANEL_W, y_top + posture_h)

    blink_top = y_top + posture_h + GAP
    _transparent_rect(frame, PANEL_X - 4, blink_top - 4,
                      PANEL_X + PANEL_W, blink_top + blink_h)

    # ── Draw posture section ────────────────────────────────────────────
    y = y_top + LINE_H
    _put(frame, "POSTURE", PANEL_X + 4, y, HEADER_SCALE, COL_WHITE, 2)
    y += LINE_H
    for text, color in p_rows:
        _put(frame, text, PANEL_X + 4, y, FONT_SCALE, color)
        y += LINE_H

    # ── Draw blinking section ───────────────────────────────────────────
    y = blink_top + LINE_H
    _put(frame, "BLINKING", PANEL_X + 4, y, HEADER_SCALE, COL_WHITE, 2)
    y += LINE_H
    for text, color in b_rows:
        _put(frame, text, PANEL_X + 4, y, FONT_SCALE, color)
        y += LINE_H


# ---------------------------------------------------------------------------
# Alert bar — full-width red bar above footer, describes what's wrong
# ---------------------------------------------------------------------------

def _build_alerts(posture_state: dict, blink_state: dict) -> list[str]:
    """Collect human-readable alert messages."""
    alerts: list[str] = []

    if posture_state.get("alert_active", False):
        tier = posture_state.get("alert_tier", "")
        status = posture_state.get("status", "Slouching")
        if tier == "all_bad":
            dur = posture_state.get("all_bad_duration", 0)
            alerts.append(
                f"POSTURE: all metrics bad for {dur:.0f}s  -  {status}"
            )
        else:
            dur = posture_state.get("bad_posture_duration", 0)
            alerts.append(
                f"POSTURE: bad posture for {dur:.0f}s  -  {status}"
            )

    if blink_state.get("low_blink_alert", False):
        alerts.append("BLINK: low blink frequency  -  risk of eye strain")

    if blink_state.get("high_perclos_alert", False):
        pct = blink_state.get("perclos", 0.0) * 100
        alerts.append(
            f"BLINK: high eye closure (PERCLOS {pct:.0f}%)  -  fatigue"
        )

    return alerts


def _draw_alerts(frame, alerts: list[str], footer_h: int) -> None:
    """Draw a red alert block above the footer bar."""
    if not alerts:
        return

    h, w = frame.shape[:2]
    bar_h = ALERT_LINE_H * len(alerts) + 12
    bar_top = h - footer_h - bar_h

    _transparent_rect(frame, 0, bar_top, w, h - footer_h,
                      color=COL_ALERT_BG, alpha=0.75)

    # Flash: alternate white / light-red every 0.5 s
    flash = int(time.time() * 2) % 2 == 0
    txt_col = COL_WHITE if flash else (180, 200, 255)

    ay = bar_top + ALERT_LINE_H
    for msg in alerts:
        display = f"!! {msg} !!"
        tw = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)[0][0]
        ax = max(10, (w - tw) // 2)
        _put(frame, display, ax, ay, 0.56, txt_col, 2)
        ay += ALERT_LINE_H


# ---------------------------------------------------------------------------
# Footer bar — two readable lines at the very bottom
# ---------------------------------------------------------------------------

def _draw_footer(
    frame,
    posture_state: dict,
    posture_mode: str,
    blink_mode: str,
    blink_prediction: str | None,
) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, h - FOOTER_H), (w, h), (0, 0, 0), -1)

    # Line 1: session stats
    session = posture_state.get("session_duration", 0)
    good_pct = posture_state.get("good_posture_percentage", 0)
    alert_count = posture_state.get("alert_count", 0)
    line1 = (
        f"Session: {session:.0f}s  |  Good posture: {good_pct:.1f}%  |  "
        f"Alerts: {alert_count}"
    )
    if blink_prediction is not None:
        line1 += f"  |  Blink ML: {blink_prediction}"

    _put(frame, line1, 10, h - 30, 0.44, COL_WHITE)

    # Line 2: modes + controls
    line2 = (
        f"Posture: {posture_mode.upper()}  |  Blink: {blink_mode.upper()}"
        f"          Q=Quit  C=Calibrate  S=Save CSVs"
    )
    _put(frame, line2, 10, h - 10, 0.40, COL_DIM)


# ---------------------------------------------------------------------------
# Main entry — called from the loop
# ---------------------------------------------------------------------------

def draw_combined_overlay(
    frame,
    posture_state: dict,
    blink_state: dict,
    posture_mode: str,
    blink_mode: str,
    blink_prediction: str | None = None,
) -> None:

    # 1. Left-side metric panels (posture on top, blinking below)
    _draw_panels(frame, posture_state, blink_state)

    # 2. Alert bar (red, full-width, above footer — only when alerts fire)
    alerts = _build_alerts(posture_state, blink_state)
    _draw_alerts(frame, alerts, FOOTER_H)

    # 3. Footer (two lines, black background)
    _draw_footer(frame, posture_state, posture_mode, blink_mode, blink_prediction)


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}s"


def train_and_exit(args: argparse.Namespace) -> bool:
    """Train models and exit. Prints results clearly to terminal."""
    if not (args.train or args.train_posture or args.train_blink):
        return False

    load_training_modules()

    print("\n" + "=" * 70)
    print("  MODEL TRAINING")
    print("=" * 70)

    if args.train or args.train_posture:
        print("\n--- POSTURE ---")
        posture_classifier = PostureClassifier()
        p_result = posture_classifier.train()  # trains SVM + RF, picks best
        if p_result:
            print(
                f"\n  [POSTURE] Best model: {p_result['model_type'].upper()}"
                f"  |  Accuracy: {p_result['accuracy']:.2%}"
                f"  |  Samples: {p_result['n_samples']}"
                f"  |  Features: {p_result['n_features']}"
            )
            print(f"  [POSTURE] Confusion matrices saved as PNG.")
        else:
            print("  [POSTURE] Training failed — not enough data or labels.")

    if args.train or args.train_blink:
        print("\n--- BLINKING ---")
        blink_classifier = BlinkClassifier()
        b_result = blink_classifier.train()  # trains SVM + RF, picks best
        if b_result:
            print(
                f"\n  [BLINK] Best model: {b_result['model_type'].upper()}"
                f"  |  Accuracy: {b_result['accuracy']:.2%}"
                f"  |  Samples: {b_result.get('n_samples', '?')}"
            )
            print(f"  [BLINK] Confusion matrices saved as PNG.")
        else:
            print("  [BLINK] Training failed — not enough data or labels.")

    # Summary
    print("\n" + "=" * 70)
    if args.train:
        p_type = p_result['model_type'].upper() if p_result else "NONE"
        b_type = b_result['model_type'].upper() if b_result else "NONE"
        print(f"  SUMMARY:  Posture → {p_type}  |  Blink → {b_type}")
        print(f"  Models saved. Run with --posture-mode ml --blink-mode ml to use them.")
    print("=" * 70 + "\n")

    return True


def main() -> None:
    args = parse_args()

    if train_and_exit(args):
        return

    load_runtime_modules()

    print("=" * 70)
    print("  COMBINED BLINKING + POSTURE MONITOR")
    print("=" * 70)

    extractor = PoseExtractor(camera_index=args.camera_index)
    if not extractor.start():
        print("[FATAL] Could not start camera. Exiting.")
        return

    posture_analyzer = PostureAnalyzer()
    load_posture_calibration(posture_analyzer)
    posture_classifier = PostureClassifier()

    blink_analyzer = BlinkAnalyzer()
    blink_monitor = BlinkMonitor()
    blink_classifier = BlinkClassifier()  # auto-loads saved model in __init__

    try:
        if args.collect_posture_data:
            run_posture_data_collection(extractor, posture_analyzer, posture_classifier)
            return

        # --- Blink data collection mode (guided, mirrors posture collection) ---
        if args.collect_blink_data:
            _run_blink_data_collection(
                extractor, blink_analyzer, blink_monitor, blink_classifier,
            )
            return

        should_calibrate_posture = args.calibrate or args.calibrate_posture
        should_calibrate_blink = args.calibrate or args.calibrate_blink

        if should_calibrate_posture:
            baseline = run_posture_calibration(extractor, posture_analyzer)
            if baseline:
                posture_analyzer.set_baseline(baseline)

        if should_calibrate_blink:
            run_blink_calibration(extractor, blink_analyzer, args.blink_calibration_seconds)

        if args.posture_mode == "ml" and not posture_classifier.is_trained():
            print("[WARNING] No trained posture model found. Falling back to thresholds.")
            args.posture_mode = "thresholds"

        if args.blink_mode == "ml" and not blink_classifier.is_trained():
            print("[WARNING] No trained blink model found. Falling back to threshold.")
            args.blink_mode = "threshold"

        posture_monitor = PostureMonitor()
        posture_monitor.start_session()

        # Print active configuration
        print(f"\n[INFO] Posture mode: {args.posture_mode}")
        print(f"[INFO] Blink mode:   {args.blink_mode}")
        print("[INFO] Monitoring started.")
        print("[INFO] Click the camera window before using keys: q=quit, c=calibrate, s=save CSVs.")
        print("[INFO] If the terminal is focused, use Ctrl+C to stop.\n")

        while True:
            if not extractor.update():
                continue

            data = extractor.get_current_data()
            frame = data["frame"]
            landmarks = data["landmarks"]
            timestamp = data["timestamp"]

            if frame is None:
                continue

            extractor.draw_landmarks_on_frame(frame, landmarks)

            posture_analysis = posture_analyzer.analyze(landmarks)
            if args.posture_mode == "ml" and landmarks is not None:
                features = posture_analyzer.get_feature_vector(landmarks)
                prediction = posture_classifier.predict(features)
                if prediction:
                    posture_analysis["status"] = prediction["label"]
                    posture_analysis["ml_confidence"] = prediction["confidence"]

            posture_state = posture_monitor.update(posture_analysis)

            blink_analysis = blink_analyzer.analyze(frame)
            blink_state = blink_monitor.update(blink_analysis, timestamp=timestamp)

            # ML predictions
            blink_prediction: dict | None = None
            if args.blink_mode == "ml" and blink_classifier.is_trained():
                blink_prediction = blink_classifier.predict(blink_state)

            # Extract label string for overlay display
            blink_pred_label: str | None = None
            if blink_prediction is not None:
                conf = blink_prediction.get("confidence", 0.0)
                blink_pred_label = f"{blink_prediction['label']} ({conf:.0%})"

            draw_combined_overlay(
                frame,
                posture_state,
                blink_state,
                args.posture_mode,
                args.blink_mode,
                blink_pred_label,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                baseline = run_posture_calibration(extractor, posture_analyzer)
                if baseline:
                    posture_analyzer.set_baseline(baseline)
                run_blink_calibration(extractor, blink_analyzer, args.blink_calibration_seconds)
            elif key == ord("s"):
                posture_rows = posture_monitor.save_csv_snapshot()
                blink_rows = blink_monitor.save_csv_snapshot()
                print(
                    "[INFO] CSV save requested: "
                    f"posture frames={posture_rows}, blink rows={blink_rows}."
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if "posture_monitor" in locals():
            posture_monitor.end_session()
        if "blink_monitor" in locals():
            blink_monitor.save_csv_snapshot()
        blink_analyzer.close()
        extractor.stop()
        cv2.destroyAllWindows()
        print("[INFO] System shut down.")


if __name__ == "__main__":
    main()