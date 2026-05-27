"""
main.py — Entry Point for the Posture Monitoring System

Orchestrates all modules:
1. PoseExtractor: captures frames and extracts landmarks
2. PostureAnalyzer: computes angles and classifies posture
3. PostureMonitor: manages timing, alerts, and CSV logging
4. PostureClassifier: (optional) ML-based classification

Run modes:
    python main.py                     # Basic mode, fixed thresholds
    python main.py --calibrate         # With 5-second calibration
    python main.py --collect-data      # Collect training data for ML
    python main.py --train             # Train both SVM and RF, pick the best
    python main.py --mode ml           # Use ML classifier

Keyboard controls during monitoring:
    q = quit, c = recalibrate, s = save CSV snapshot

The PoseExtractor exposes the raw frame via get_current_data(), so other
modules (e.g., blinking detection) can access the same camera feed without
opening a second VideoCapture.
"""

import argparse
import cv2
import json
import numpy as np
import time

import config
from pose_extractor import PoseExtractor
from posture_analyzer import PostureAnalyzer
from posture_monitor import PostureMonitor
from classifier import PostureClassifier


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Posture Monitoring System for Students"
    )
    parser.add_argument(
        "--mode", type=str, choices=["thresholds", "ml"], default="thresholds",
        help="Classification mode: 'thresholds' (rule-based) or 'ml' (classifier)"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run initial calibration (sit upright for a few seconds)"
    )
    parser.add_argument(
        "--collect-data", action="store_true",
        help="Enter data collection mode for ML training"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train both SVM and Random Forest, pick the best, then exit"
    )
    return parser.parse_args()


def run_calibration(extractor, analyzer):
    """
    Run calibration: user sits upright for a few seconds, system records
    baseline values for all 3 parameters.

    Args:
        extractor (PoseExtractor): Camera and landmark extractor.
        analyzer (PostureAnalyzer): Posture analyzer.

    Returns:
        dict or None: Baseline data.
    """
    print(f"\n[CALIBRATION] Please sit upright in your normal good posture.")
    print(f"[CALIBRATION] Recording baseline for {config.CALIBRATION_DURATION} seconds...")
    print(f"[CALIBRATION] Press 'q' to skip.\n")

    calibration_samples = []
    start_time = time.time()

    while time.time() - start_time < config.CALIBRATION_DURATION:
        if not extractor.update():
            continue

        data = extractor.get_current_data()
        frame = data["frame"]
        landmarks = data["landmarks"]

        if landmarks is not None:
            sample = analyzer.compute_calibration_data(landmarks)
            if sample is not None:
                calibration_samples.append(sample)

        # Show progress
        if frame is not None:
            elapsed = time.time() - start_time
            remaining = config.CALIBRATION_DURATION - elapsed
            n = len(calibration_samples)
            cv2.putText(
                frame,
                f"CALIBRATING... Sit upright! ({remaining:.1f}s, {n} samples)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                config.COLOR_WARNING, 2
            )
            cv2.imshow("Posture Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[CALIBRATION] Skipped.")
            return None

    if calibration_samples:
        # Average all samples for each key
        baseline = {}
        all_keys = set()
        for sample in calibration_samples:
            all_keys.update(sample.keys())

        for key in all_keys:
            values = [s[key] for s in calibration_samples if key in s]
            if values:
                baseline[key] = float(np.mean(values))

        with open(config.CALIBRATION_PATH, "w") as f:
            json.dump(baseline, f, indent=2)

        print(f"[CALIBRATION] Baseline saved with {len(baseline)} parameters.")
        return baseline
    else:
        print("[CALIBRATION] No data collected.")
        return None


def run_data_collection(extractor, analyzer, classifier):
    """
    Guide the user through collecting training data for the ML classifier.

    Args:
        extractor (PoseExtractor): Camera and landmark extractor.
        analyzer (PostureAnalyzer): Posture analyzer.
        classifier (PostureClassifier): ML classifier.
    """
    for label in ["Good", "Slouching"]:
        print(f"\n[DATA COLLECTION] Get into your '{label}' posture.")
        print(f"[DATA COLLECTION] Recording for {config.TRAINING_COLLECTION_DURATION} seconds.")
        input(f"[DATA COLLECTION] Press Enter when ready...")

        classifier.start_collection(label)
        start_time = time.time()

        while time.time() - start_time < config.TRAINING_COLLECTION_DURATION:
            if not extractor.update():
                continue

            data = extractor.get_current_data()
            landmarks = data["landmarks"]

            if landmarks is not None:
                features = analyzer.get_feature_vector(landmarks)
                classifier.add_sample(features)

            # Show progress
            frame = data["frame"]
            if frame is not None:
                elapsed = time.time() - start_time
                remaining = config.TRAINING_COLLECTION_DURATION - elapsed
                count = classifier.get_collection_count()
                cv2.putText(
                    frame,
                    f"Collecting '{label}': {count} samples ({remaining:.1f}s left)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    config.COLOR_WARNING, 2
                )
                cv2.imshow("Posture Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        classifier.stop_collection()

    print("\n[DATA COLLECTION] Complete! Run with --train to build the model.")


def draw_overlay(frame, state, mode):
    """
    Draw the posture monitoring overlay on the video frame.

    Shows:
    - Top bar: real-time frame status (Good / Slouching) + countdown to next eval
    - Right panel: real-time metric values
    - Left panel: last 60-second evaluation verdict + per-parameter breakdown
    - Bottom bar: session stats

    Args:
        frame (np.array): The BGR video frame.
        state (dict): Current state from PostureMonitor.
        mode (str): Classification mode.

    Returns:
        np.array: Frame with overlay.
    """
    h, w, _ = frame.shape
    status = state.get("status", "Unknown")
    metrics = state.get("metrics", {})
    window_left = state.get("window_seconds_left", 0)
    last_eval = state.get("last_evaluation", None)

    # --- Status bar at top ---
    if status == "Good":
        bar_color = config.COLOR_GOOD
    elif status == "Slouching":
        bar_color = config.COLOR_WARNING
    else:
        bar_color = (100, 100, 100)

    cv2.rectangle(frame, (0, 0), (w, 60), bar_color, -1)

    # Status text with countdown
    status_text = f"STATUS: {status} | Next eval: {window_left:.0f}s"
    cv2.putText(
        frame, status_text,
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2
    )

    # Mode indicator
    mode_text = f"Mode: {mode.upper()}"
    cv2.putText(
        frame, mode_text,
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1
    )

    # --- Real-time metrics panel (right side) ---
    panel_x = w - 300
    panel_y = 70
    line_height = 25

    overlay_img = frame.copy()
    cv2.rectangle(overlay_img, (panel_x - 10, panel_y - 10),
                  (w - 5, panel_y + line_height * (len(metrics) + 1)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_img, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, "REAL-TIME:",
        (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1
    )
    panel_y += line_height

    for name, metric in metrics.items():
        if metric is None:
            continue
        value = metric.get("value", "N/A")
        unit = metric.get("unit", "")
        is_good = metric.get("is_good", True)
        color = config.COLOR_GOOD if is_good else config.COLOR_BAD
        display_name = name.replace("_", " ").title()
        text = f"{display_name}: {value}{unit}"
        cv2.putText(
            frame, text,
            (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
        panel_y += line_height

    # --- Last evaluation panel (left side) ---
    if last_eval is not None:
        eval_x = 10
        eval_y = 70
        verdict = last_eval["verdict"]
        params = last_eval["params"]

        # Background
        eval_overlay = frame.copy()
        eval_h = line_height * (len(params) + 2)
        cv2.rectangle(eval_overlay, (eval_x - 5, eval_y - 10),
                      (320, eval_y + eval_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(eval_overlay, 0.5, frame, 0.5, 0, frame)

        # Verdict header
        verdict_color = config.COLOR_BAD if verdict == "Bad Posture" else config.COLOR_GOOD
        cv2.putText(
            frame, f"LAST EVAL: {verdict}",
            (eval_x, eval_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, verdict_color, 2
        )
        eval_y += line_height

        # Per-parameter breakdown
        param_display = {
            "shoulder_asymmetry": ("Shoulders", "deg"),
            "head_lateral_tilt": ("Head Tilt", "deg"),
            "forward_head_proxy": ("Fwd Head", "%"),
        }

        for param_name, (label, unit) in param_display.items():
            p = params.get(param_name)
            if p is None:
                continue
            color = config.COLOR_BAD if p["is_bad"] else config.COLOR_GOOD
            text = f"{label}: bad {p['bad_pct']:.0f}% | avg {p['avg_value']:.1f}{unit}"
            cv2.putText(
                frame, text,
                (eval_x, eval_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1
            )
            eval_y += line_height

    # --- Session stats at bottom ---
    session_dur = state.get("session_duration", 0)
    good_pct = state.get("good_posture_percentage", 0)
    alert_count = state.get("alert_count", 0)

    stats_text = (f"Session: {session_dur:.0f}s | "
                  f"Good: {good_pct:.1f}% | "
                  f"Alerts: {alert_count}")

    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, stats_text,
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1
    )

    # Controls hint
    controls = "Q=Quit | C=Calibrate | S=Save CSV"
    cv2.putText(
        frame, controls,
        (w - 280, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
    )

    return frame


def main():
    """Main entry point."""
    args = parse_arguments()

    print("=" * 60)
    print("  POSTURE MONITORING SYSTEM")
    print("  Students Ergonomics Monitoring Project")
    print("=" * 60)

    # --- Initialize modules ---
    extractor = PoseExtractor()

    if not extractor.start():
        print("[FATAL] Could not start camera. Exiting.")
        return

    analyzer = PostureAnalyzer()

    # Load existing calibration if available
    try:
        with open(config.CALIBRATION_PATH, "r") as f:
            baseline = json.load(f)
        analyzer.set_baseline(baseline)
    except FileNotFoundError:
        pass

    classifier = PostureClassifier()

    # --- Handle special modes ---

    # Train only
    if args.train:
        results = classifier.train()  # Trains both SVM and RF, picks the best
        if results:
            print(f"\nBest model: {results['model_type'].upper()} "
                  f"with {results['accuracy']:.2%} accuracy.")
            print(f"Confusion matrices saved as PNG images.")
        extractor.stop()
        return

    # Data collection
    if args.collect_data:
        run_data_collection(extractor, analyzer, classifier)
        extractor.stop()
        cv2.destroyAllWindows()
        return

    # Calibration
    if args.calibrate:
        baseline = run_calibration(extractor, analyzer)
        if baseline:
            analyzer.set_baseline(baseline)

    # Check ML mode
    if args.mode == "ml" and not classifier.is_trained():
        print("[WARNING] No trained model found. Falling back to threshold mode.")
        print("[WARNING] Run with --collect-data and --train first.")
        args.mode = "thresholds"

    # Start monitoring
    monitor = PostureMonitor()
    monitor.start_session()

    print(f"\n[INFO] Mode: {args.mode}")
    print(f"[INFO] Alert thresholds: all bad = {config.ALL_BAD_TIME_THRESHOLD}s, "
          f"any bad = {config.ANY_BAD_TIME_THRESHOLD}s")
    print(f"[INFO] Monitoring started. Press 'q' to quit.\n")

    # --- Main loop ---
    try:
        while True:
            # 1. Capture and extract landmarks
            if not extractor.update():
                continue

            data = extractor.get_current_data()
            frame = data["frame"]
            landmarks = data["landmarks"]

            if frame is None:
                continue

            # 2. Draw landmarks
            extractor.draw_landmarks_on_frame(frame, landmarks)

            # 3. Analyze posture
            analysis = analyzer.analyze(landmarks)

            # Override with ML prediction if in ML mode
            if args.mode == "ml" and landmarks is not None:
                features = analyzer.get_feature_vector(landmarks)
                prediction = classifier.predict(features)
                if prediction:
                    analysis["status"] = prediction["label"]
                    analysis["ml_confidence"] = prediction["confidence"]

            # 4. Update monitor
            state = monitor.update(analysis)

            # 5. Draw overlay
            frame = draw_overlay(frame, state, args.mode)

            # 6. Show
            cv2.imshow("Posture Monitor", frame)

            # 7. Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                baseline = run_calibration(extractor, analyzer)
                if baseline:
                    analyzer.set_baseline(baseline)
            elif key == ord("s"):
                monitor.save_csv_snapshot()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        monitor.end_session()
        extractor.stop()
        cv2.destroyAllWindows()
        print("[INFO] System shut down.")


if __name__ == "__main__":
    main()
