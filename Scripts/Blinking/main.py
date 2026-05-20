"""
main.py — Entry Point for the Blinking Monitor

Mirrors the posture main.py pipeline exactly.

Run modes:
    python main.py                      # Basic threshold mode
    python main.py --calibrate          # Personal EAR calibration
    python main.py --collect-data       # Collect ML training data
    python main.py --train              # Train SVM + RF, pick the best
    python main.py --mode ml            # Use ML classifier

Keyboard controls during monitoring:
    q = quit  |  s = save CSV snapshot
"""

from __future__ import annotations

import argparse
import time

import cv2

import blink_config as config
from blink_analyzer import BlinkAnalyzer
from blink_classifier import BlinkClassifier
from blink_monitor import BlinkMonitor


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blinking Monitor")
    parser.add_argument(
        "--mode", type=str, choices=["threshold", "ml"], default="threshold",
        help="Classification mode: 'threshold' (rule-based) or 'ml' (classifier)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run personal EAR calibration before monitoring",
    )
    parser.add_argument(
        "--calibration-seconds", type=float, default=8.0,
        help="Duration of calibration phase in seconds (default: 8)",
    )
    parser.add_argument(
        "--collect-data", action="store_true",
        help="Enter data-collection mode for ML training",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train both SVM and Random Forest, pick the best, then exit",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    return parser.parse_args()


# =============================================================================
# CALIBRATION
# =============================================================================

def run_calibration(
    analyzer: BlinkAnalyzer,
    camera: cv2.VideoCapture,
    duration_seconds: float,
) -> None:
    """
    Personal EAR calibration: user keeps eyes naturally open.
    Mirrors run_calibration() in posture/main.py.
    """
    print(
        f"\n[CALIBRATION] Keep your eyes naturally open and look at the screen."
    )
    print(f"[CALIBRATION] Recording baseline for {duration_seconds:.0f} seconds...")
    print(f"[CALIBRATION] Press 'q' to skip.\n")

    samples: list[float] = []
    start = time.time()

    while time.time() - start < duration_seconds:
        ok, frame = camera.read()
        if not ok:
           time.sleep(0.01)   # evita busy-loop infinito se la camera si inceppa
           continue

        analysis = analyzer.analyze(frame)
        if analysis["face_detected"] and analysis["ear"] is not None:
            samples.append(float(analysis["ear"]))

        analyzer.draw_overlay(frame, analysis)
        elapsed = time.time() - start
        remaining = max(0.0, duration_seconds - elapsed)
        cv2.putText(
            frame,
            f"CALIBRATING... Keep eyes open! ({remaining:.1f}s, {len(samples)} samples)",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 165, 255), 2,
        )
        cv2.imshow("Blinking Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[CALIBRATION] Skipped.")
            return

    if samples:
        payload = analyzer.calibrate_from_samples(samples)
        print(
            f"[CALIBRATION] Baseline saved. "
            f"EAR={payload['baseline_open_ear']:.3f}, "
            f"threshold={payload['ear_threshold']:.3f}"
        )
    else:
        print("[CALIBRATION] No data collected.")


# =============================================================================
# DATA COLLECTION
# =============================================================================

def run_data_collection(
    analyzer: BlinkAnalyzer,
    monitor: BlinkMonitor,
    classifier: BlinkClassifier,
    camera: cv2.VideoCapture,
) -> None:
    """
    Guide the user through collecting training data for the ML classifier.
    Mirrors run_data_collection() in posture/main.py exactly:
      - one timed session per label
      - user presses Enter to start each session
      - progress shown on-screen
    """
    for label in ["Normal", "Tired"]:
        print(f"\n[DATA COLLECTION] Get into your '{label}' blinking state.")
        print(
            f"[DATA COLLECTION] Recording for "
            f"{config.TRAINING_COLLECTION_DURATION}s."
        )
        input(f"[DATA COLLECTION] Press Enter when ready...")

        classifier.start_collection(label)
        start_time = time.time()

        while time.time() - start_time < config.TRAINING_COLLECTION_DURATION:
            ok, frame = camera.read()
            if not ok:
                continue

            analysis = analyzer.analyze(frame)
            state = monitor.update(analysis)

            # Hand state to classifier (mirrors add_sample in posture)
            classifier.add_sample(state)

            # On-screen progress
            elapsed = time.time() - start_time
            remaining = config.TRAINING_COLLECTION_DURATION - elapsed
            count = classifier.get_collection_count()

            analyzer.draw_overlay(frame, analysis)
            cv2.putText(
                frame,
                f"Collecting '{label}': {count} samples ({remaining:.1f}s left)",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 165, 255), 2,
            )
            cv2.imshow("Blinking Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        classifier.stop_collection()

    print("\n[DATA COLLECTION] Complete! Run with --train to build the model.")


# =============================================================================
# MONITORING OVERLAY
# =============================================================================

def draw_monitor_overlay(
    frame: "cv2.Mat",
    state: dict,
    mode: str,
    prediction: dict | None = None,
) -> None:
    """Draw blink metrics and ML result on the frame."""
    y = 100
    rows = [
        f"Blink freq:  {state['blink_frequency_bpm']:.1f} bpm",
        f"PERCLOS:     {state['perclos'] * 100:.1f}%",
        f"IBI last:    {_fmt(state['last_inter_blink_interval'])}",
        f"IBI mean:    {_fmt(state['mean_inter_blink_interval'])}",
        f"Mode:        {mode.upper()}",
    ]
    if prediction is not None:
        conf = prediction.get("confidence", 0.0)
        rows.append(f"ML class:    {prediction['label']} ({conf:.0%})")
    if state["low_blink_alert"]:
        rows.append("ALERT: low blink frequency")
    if state["high_perclos_alert"]:
        rows.append("ALERT: high eye closure")

    for row in rows:
        cv2.putText(
            frame, row, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
        )
        y += 28


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}s"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    print("=" * 60)
    print("  BLINKING MONITOR")
    print("  Students Ergonomics Monitoring Project")
    print("=" * 60)

    # --- Handle train-only mode (no camera needed) ---
    if args.train:
        classifier = BlinkClassifier()
        result = classifier.train()
        if result:
            print(
                f"\nBest model: {result['model_type'].upper()} "
                f"with {result['accuracy']:.2%} accuracy."
            )
            print("Confusion matrices saved as PNG images.")
        return

    # --- Open camera ---
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}"
        )

    analyzer = BlinkAnalyzer()
    monitor = BlinkMonitor()
    classifier = BlinkClassifier()

    try:
        # --- Optional calibration ---
        if args.calibrate:
            run_calibration(analyzer, camera, args.calibration_seconds)

        # --- Data collection mode ---
        if args.collect_data:
            run_data_collection(analyzer, monitor, classifier, camera)
            return

        # --- ML mode: check model is available ---
        if args.mode == "ml" and not classifier.is_trained():
            print(
                "[WARNING] No trained blink model found. "
                "Falling back to threshold mode."
            )
            print("[WARNING] Run with --collect-data and --train first.")
            args.mode = "threshold"

        print(f"\n[INFO] Mode: {args.mode}")
        print(f"[INFO] Monitoring started. Press 'q' to quit.\n")

        # --- Main monitoring loop ---
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                break

            analysis = analyzer.analyze(frame)
            state = monitor.update(analysis)

            prediction: dict | None = None
            if args.mode == "ml":
                prediction = classifier.predict(state)

            analyzer.draw_overlay(frame, analysis)
            draw_monitor_overlay(frame, state, args.mode, prediction)
            cv2.imshow("Blinking Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                monitor.save_csv_snapshot()

    finally:
        monitor.save_csv_snapshot()
        analyzer.close()
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Blinking monitor shut down.")


if __name__ == "__main__":
    main()
