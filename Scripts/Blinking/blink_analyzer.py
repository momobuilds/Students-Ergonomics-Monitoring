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
            return {"face_detected": False, "ear": None, "is_closed": False}

        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_ear = self._eye_aspect_ratio(landmarks, config.LEFT_EYE_INDICES, w, h)
        right_ear = self._eye_aspect_ratio(landmarks, config.RIGHT_EYE_INDICES, w, h)
        ear = (left_ear + right_ear) / 2.0

        return {
            "face_detected": True,
            "ear": ear,
            "is_closed": ear < config.EAR_CLOSED_THRESHOLD,
        }

    def close(self):
        """Release FaceMesh resources."""
        self.face_mesh.close()

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
