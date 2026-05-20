"""Frame-level blinking analysis using MediaPipe FaceMesh."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import blink_config as config

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime with clear message
    mp = None


@dataclass
class EyeMetrics:
    left_ear: float
    right_ear: float
    ear: float


class BlinkAnalyzer:
    """Runs FaceMesh and computes Eye Aspect Ratio for each frame."""

    def __init__(
        self,
        ear_threshold: float | None = None,
        calibration_path: Path = config.CALIBRATION_PATH,
        use_calibration: bool = True,
    ) -> None:
        if mp is None:
            raise ImportError(
                "MediaPipe is required for BlinkAnalyzer. Install it with: pip install mediapipe"
            )

        self.calibration_path = calibration_path
        self.ear_threshold = ear_threshold or config.EAR_CLOSED_THRESHOLD
        if use_calibration:
            self._load_calibration()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame: np.ndarray) -> dict[str, Any]:
        """Return EAR and open/closed classification for one BGR frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self.face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return {
                "face_detected": False,
                "left_ear": None,
                "right_ear": None,
                "ear": None,
                "ear_threshold": self.ear_threshold,
                "eye_state": "Unknown",
                "is_closed": False,
                "classification": "No face",
            }

        landmarks = result.multi_face_landmarks[0].landmark
        metrics = self._compute_eye_metrics(landmarks, frame.shape)
        is_closed = metrics.ear < self.ear_threshold

        return {
            "face_detected": True,
            "left_ear": metrics.left_ear,
            "right_ear": metrics.right_ear,
            "ear": metrics.ear,
            "ear_threshold": self.ear_threshold,
            "eye_state": "Closed" if is_closed else "Open",
            "is_closed": is_closed,
            "classification": "Bad" if is_closed else "Good",
        }

    def draw_overlay(
        self,
        frame: np.ndarray,
        analysis: dict[str, Any],
        origin: tuple[int, int] = (20, 40),
    ) -> None:
        """Draw compact blinking values on the frame."""
        x, y = origin
        ear = analysis.get("ear")
        ear_text = "EAR: --" if ear is None else f"EAR: {ear:.3f}"
        state = analysis.get("eye_state", "Unknown")
        color = (0, 0, 255) if state == "Closed" else (0, 180, 0)

        cv2.putText(frame, ear_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(
            frame,
            f"Eyes: {state}",
            (x, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def calibrate_from_samples(self, ear_samples: list[float]) -> dict[str, float]:
        """Save a personal open-eye baseline and derive a threshold."""
        valid_samples = [value for value in ear_samples if value > 0]
        if not valid_samples:
            raise ValueError("No valid EAR samples collected for calibration.")

        baseline_ear = float(np.median(valid_samples))
        threshold = baseline_ear * 0.75
        payload = {
            "baseline_open_ear": baseline_ear,
            "ear_threshold": threshold,
            "num_samples": len(valid_samples),
        }
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.ear_threshold = threshold
        return payload

    def _load_calibration(self) -> None:
        if not self.calibration_path.exists():
            return

        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        self.ear_threshold = float(payload.get("ear_threshold", self.ear_threshold))

    def _compute_eye_metrics(self, landmarks: Any, frame_shape: tuple[int, ...]) -> EyeMetrics:
        left = self._eye_aspect_ratio(landmarks, config.LEFT_EYE_INDICES, frame_shape)
        right = self._eye_aspect_ratio(landmarks, config.RIGHT_EYE_INDICES, frame_shape)
        return EyeMetrics(left_ear=left, right_ear=right, ear=(left + right) / 2.0)

    @staticmethod
    def _eye_aspect_ratio(
        landmarks: Any,
        indices: list[int],
        frame_shape: tuple[int, ...],
    ) -> float:
        height, width = frame_shape[:2]
        points = [
            np.array([landmarks[index].x * width, landmarks[index].y * height])
            for index in indices
        ]

        p1, p2, p3, p4, p5, p6 = points
        vertical_1 = math.dist(p2, p6)
        vertical_2 = math.dist(p3, p5)
        horizontal = math.dist(p1, p4)
        if horizontal == 0:
            return 0.0

        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

