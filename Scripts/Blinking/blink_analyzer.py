"""
blink_analyzer.py — Frame-level Eye Aspect Ratio (EAR) computation

Uses MediaPipe FaceMesh to detect face landmarks and compute the Eye
Aspect Ratio for each frame. Returns raw EAR value and open/closed state.

EAR formula (Soukupová & Čech, 2016):
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

where p1-p6 are the 6 eye landmark points.
"""

import math

import cv2
import numpy as np
import mediapipe as mp

import blink_config as config


class BlinkAnalyzer:
    """
    Computes Eye Aspect Ratio from a video frame.

    Usage:
        analyzer = BlinkAnalyzer()
        result = analyzer.analyze(frame)
        # result = {"ear": 0.28, "is_closed": False, "face_detected": True}
        analyzer.close()
    """

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame):
        """
        Compute EAR for one BGR frame.

        Args:
            frame (np.ndarray): BGR video frame.

        Returns:
            dict: {
                "face_detected": bool,
                "ear": float or None,
                "is_closed": bool,
            }
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"face_detected": False, "ear": None, "is_closed": False,
                    "left_eye": [], "right_eye": [],
                    "eye_visibility": 0.0, "eyes_visible": False}

        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_ear = self._eye_aspect_ratio(landmarks, config.LEFT_EYE_INDICES, w, h)
        right_ear = self._eye_aspect_ratio(landmarks, config.RIGHT_EYE_INDICES, w, h)
        ear = (left_ear + right_ear) / 2.0

        # Pixel coordinates of the eye landmarks, for drawing overlays.
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in config.LEFT_EYE_INDICES]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                     for i in config.RIGHT_EYE_INDICES]

        # Occlusion check: FaceMesh infers eye landmarks even behind sunglasses
        # or a hand, so we inspect the eye pixels directly. One clearly visible
        # eye is enough to trust the EAR for this frame.
        eye_visibility = max(
            self._eye_region_score(frame, left_eye),
            self._eye_region_score(frame, right_eye),
        )

        return {
            "face_detected": True,
            "ear": ear,
            "is_closed": ear < config.EAR_CLOSED_THRESHOLD,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "eye_visibility": eye_visibility,
            "eyes_visible": eye_visibility >= config.EYE_VISIBILITY_MIN_STD,
        }

    def close(self):
        """Release FaceMesh resources."""
        self.face_mesh.close()

    @staticmethod
    def _eye_region_score(frame, eye_points):
        """Grayscale std-dev over the eye bounding box (an occlusion proxy).

        A real, uncovered eye has high local contrast (white sclera, dark iris,
        lashes), so the std-dev is high. A dark lens or a hand covering the eye
        is comparatively uniform, so the std-dev drops. Returns 0.0 when the
        region is empty or off-frame.
        """
        if not eye_points:
            return 0.0
        xs = [p[0] for p in eye_points]
        ys = [p[1] for p in eye_points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # Eyes are wide and short; pad vertically more to include the lid/lashes.
        pad_x = int((x1 - x0) * 0.3) + 1
        pad_y = int((y1 - y0) * 0.8) + 2
        h, w = frame.shape[:2]
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(h, y1 + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    @staticmethod
    def _eye_aspect_ratio(landmarks, indices, width, height):
        """
        Compute EAR for one eye.

        Args:
            landmarks: FaceMesh landmarks.
            indices (list): 6 landmark indices for the eye.
            width (int): Frame width.
            height (int): Frame height.

        Returns:
            float: Eye aspect ratio.
        """
        points = [
            np.array([landmarks[i].x * width, landmarks[i].y * height])
            for i in indices
        ]

        p1, p2, p3, p4, p5, p6 = points
        vertical_1 = math.dist(p2, p6)
        vertical_2 = math.dist(p3, p5)
        horizontal = math.dist(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)
