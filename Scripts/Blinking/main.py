"""Entry point for the webcam blinking monitor.

to start with parameters: python Scripts/Blinking/main.py
with calibration: python Scripts/Blinking/main.py --calibrate 
ML: python Scripts/Blinking/main.py--mode ml

collect data normal case: python Scripts/Blinking/main.py --collect-data --label Normal
collect data tired case: python Scripts/Blinking/main.py --collect-data --label Tired
train model : python Scripts/Blinking/main.py --train

"""

from __future__ import annotations

import argparse
import time

import cv2

from blink_analyzer import BlinkAnalyzer
from blink_classifier import BlinkClassifier
from blink_monitor import BlinkMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blinking monitor")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--label", default="Normal", help="Label used with --collect-data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--mode", choices=["threshold", "ml"], default="threshold")
    parser.add_argument("--calibration-seconds", type=float, default=8.0)
    return parser.parse_args()


def run_calibration(
    analyzer: BlinkAnalyzer,
    camera: cv2.VideoCapture,
    duration_seconds: float,
) -> None:
    print("Calibration started. Keep your eyes naturally open and look at the screen.")
    samples: list[float] = []
    start = time.time()

    while time.time() - start < duration_seconds:
        ok, frame = camera.read()
        if not ok:
            continue

        analysis = analyzer.analyze(frame)
        if analysis["face_detected"] and analysis["ear"] is not None:
            samples.append(float(analysis["ear"]))
        analyzer.draw_overlay(frame, analysis)
        cv2.putText(
            frame,
            "Calibration: keep eyes open",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Blinking Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    payload = analyzer.calibrate_from_samples(samples)
    print(
        "Calibration saved. "
        f"Baseline EAR={payload['baseline_open_ear']:.3f}, "
        f"threshold={payload['ear_threshold']:.3f}"
    )


def draw_monitor_overlay(frame, state: dict, prediction: str | None = None) -> None:
    y = 100
    rows = [
        f"Blink freq: {state['blink_frequency_bpm']:.1f} bpm",
        f"PERCLOS: {state['perclos'] * 100:.1f}%",
        f"IBI last: {_format_seconds(state['last_inter_blink_interval'])}",
        f"IBI mean: {_format_seconds(state['mean_inter_blink_interval'])}",
    ]
    if prediction is not None:
        rows.append(f"ML: {prediction}")
    if state["low_blink_alert"]:
        rows.append("ALERT: low blink frequency")
    if state["high_perclos_alert"]:
        rows.append("ALERT: high eye closure")

    for row in rows:
        cv2.putText(frame, row, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}s"


def main() -> None:
    args = parse_args()
    classifier = BlinkClassifier()
    if args.train:
        result = classifier.train()
        print(f"Training complete: {result}")
        return

    analyzer = BlinkAnalyzer()
    monitor = BlinkMonitor()
    if args.mode == "ml":
        classifier.load()

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    try:
        if args.calibrate:
            run_calibration(analyzer, camera, args.calibration_seconds)

        print("Blinking monitor started. Press q to stop, s to save CSV snapshot.")
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                break

            analysis = analyzer.analyze(frame)
            state = monitor.update(analysis)
            prediction = classifier.predict(state) if args.mode == "ml" else None

            if args.collect_data:
                classifier.append_sample(state, args.label)

            analyzer.draw_overlay(frame, analysis)
            draw_monitor_overlay(frame, state, prediction)
            cv2.imshow("Blinking Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                monitor.save_csv_snapshot()
    finally:
        monitor.save_csv_snapshot()
        analyzer.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
