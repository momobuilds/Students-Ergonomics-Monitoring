"""
main2.0.py - Combined blinking + posture monitor.

This script uses one webcam stream and evaluates everything computed by:
- Scripts/Blinking/main.py
- Scripts/Posture/main.py

Examples:
    python Scripts/main2.0.py
    python Scripts/main2.0.py --calibrate
    python Scripts/main2.0.py --posture-mode ml --blink-mode ml
    python Scripts/main2.0.py --train
    python Scripts/main2.0.py --collect-blink-data --blink-label Tired
    python Scripts/main2.0.py --collect-posture-data

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
        help="Append blinking samples during the live session.",
    )
    parser.add_argument("--blink-label", default="Normal", help="Label used with --collect-blink-data.")
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

        analyzer.draw_overlay(frame, analysis)
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


def draw_combined_overlay(
    frame,
    posture_state: dict,
    blink_state: dict,
    posture_mode: str,
    blink_mode: str,
    blink_prediction: str | None = None,
) -> None:
    height, width = frame.shape[:2]

    posture_status = posture_state.get("status", "Unknown")
    posture_alert = posture_state.get("alert_active", False)
    blink_alert = blink_state.get("low_blink_alert", False) or blink_state.get("high_perclos_alert", False)

    if posture_alert or blink_alert:
        bar_color = (0, 0, 200)
    elif posture_status == "Good":
        bar_color = (0, 160, 0)
    elif posture_status == "Slouching":
        bar_color = (0, 165, 255)
    else:
        bar_color = (90, 90, 90)

    cv2.rectangle(frame, (0, 0), (width, 68), bar_color, -1)
    header = f"Posture: {posture_status} | Eyes: {blink_state.get('eye_state', 'Unknown')}"
    cv2.putText(frame, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mode_text = f"Posture mode: {posture_mode.upper()} | Blink mode: {blink_mode.upper()}"
    if blink_prediction is not None:
        mode_text += f" | Blink ML: {blink_prediction}"
    cv2.putText(frame, mode_text, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    draw_posture_panel(frame, posture_state, width)
    draw_blink_panel(frame, blink_state)

    session = posture_state.get("session_duration", 0)
    good_pct = posture_state.get("good_posture_percentage", 0)
    alerts = posture_state.get("alert_count", 0)
    footer = f"Session: {session:.0f}s | Good posture: {good_pct:.1f}% | Posture alerts: {alerts}"
    controls = "Q=Quit | C=Calibrate | S=Save CSVs"
    cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, footer, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, controls, (width - 285, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def draw_posture_panel(frame, posture_state: dict, width: int) -> None:
    metrics = posture_state.get("metrics", {})
    panel_x = max(width - 315, 10)
    panel_y = 84
    line_height = 25
    panel_height = line_height * (len(metrics) + 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y - 10), (width - 5, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)

    cv2.putText(frame, "POSTURE", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y = panel_y + line_height
    for name, metric in metrics.items():
        if metric is None:
            continue
        color = (0, 210, 0) if metric.get("is_good", True) else (0, 0, 255)
        label = name.replace("_", " ").title()
        value = metric.get("value", "N/A")
        unit = metric.get("unit", "")
        cv2.putText(frame, f"{label}: {value}{unit}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += line_height


def draw_blink_panel(frame, blink_state: dict) -> None:
    x = 20
    y = 92
    line_height = 26
    rows = [
        "BLINKING",
        f"EAR: {_format_float(blink_state.get('ear'), 3)}",
        f"Blink freq: {blink_state.get('blink_frequency_bpm', 0.0):.1f} bpm",
        f"PERCLOS: {blink_state.get('perclos', 0.0) * 100:.1f}%",
        f"IBI last: {_format_seconds(blink_state.get('last_inter_blink_interval'))}",
        f"IBI mean: {_format_seconds(blink_state.get('mean_inter_blink_interval'))}",
        f"Blinks/window: {blink_state.get('blink_count_window', 0)}",
    ]
    if blink_state.get("low_blink_alert"):
        rows.append("ALERT: low blink frequency")
    if blink_state.get("high_perclos_alert"):
        rows.append("ALERT: high eye closure")

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y - 18), (315, y + line_height * len(rows)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)

    for index, row in enumerate(rows):
        color = (255, 255, 255)
        if row.startswith("ALERT"):
            color = (0, 0, 255)
        elif index == 0:
            color = (255, 255, 255)
        cv2.putText(frame, row, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_height


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}s"


def train_and_exit(args: argparse.Namespace) -> bool:
    if not (args.train or args.train_posture or args.train_blink):
        return False

    load_training_modules()

    if args.train or args.train_posture:
        posture_classifier = PostureClassifier()
        result = posture_classifier.train()
        if result:
            print(
                f"\n[POSTURE TRAINING] Best model: {result['model_type'].upper()} "
                f"with {result['accuracy']:.2%} accuracy."
            )

    if args.train or args.train_blink:
        blink_classifier = BlinkClassifier()
        result = blink_classifier.train()
        print(f"\n[BLINK TRAINING] Complete: {result}")

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
    blink_classifier = BlinkClassifier()
    if args.blink_mode == "ml":
        blink_classifier.load()

    try:
        if args.collect_posture_data:
            run_posture_data_collection(extractor, posture_analyzer, posture_classifier)
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

        posture_monitor = PostureMonitor()
        posture_monitor.start_session()

        print(f"\n[INFO] Posture mode: {args.posture_mode}")
        print(f"[INFO] Blink mode: {args.blink_mode}")
        if args.collect_blink_data:
            print(f"[INFO] Collecting blinking data with label: {args.blink_label}")
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
            blink_prediction = blink_classifier.predict(blink_state) if args.blink_mode == "ml" else None

            if args.collect_blink_data:
                blink_classifier.append_sample(blink_state, args.blink_label)

            draw_combined_overlay(
                frame,
                posture_state,
                blink_state,
                args.posture_mode,
                args.blink_mode,
                blink_prediction,
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
