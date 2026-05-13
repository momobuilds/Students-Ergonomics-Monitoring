"""
pose_extractor.py — Camera Management and Pose Landmark Extraction

This module handles:
- Opening and reading from the frontal webcam
- Running MediaPipe Pose on each frame to extract 33 body landmarks
- Providing the raw frame for other modules (e.g., blinking detection)

The module does NOT know anything about posture analysis — it only
extracts raw landmarks and passes them forward.

MediaPipe Landmark Indices (relevant for posture):
    0  = nose
    7  = left ear
    8  = right ear
    11 = left shoulder
    12 = right shoulder
    23 = left hip
    24 = right hip

Each landmark has (x, y, z, visibility):
    x, y = normalized coordinates [0, 1] relative to image size
    z    = depth estimate (relative to hip midpoint, smaller = closer)
    visibility = confidence that the landmark is visible [0, 1]
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import config


class PoseExtractor:
    """
    Manages the webcam and extracts pose landmarks using MediaPipe.

    Usage:
        extractor = PoseExtractor()
        extractor.start()

        while True:
            if extractor.update():
                data = extractor.get_current_data()
                frame = data["frame"]
                landmarks = data["landmarks"]
                ...

        extractor.stop()
    """

    def __init__(self, camera_index=None):
        """
        Initialize the PoseExtractor.

        Args:
            camera_index (int): Index of the webcam.
                                Defaults to config.CAMERA_INDEX.
        """
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=config.MP_STATIC_IMAGE_MODE,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
        )

        # Video capture object
        self.cap = None

        # Latest data
        self._current_data = {
            "frame": None,
            "landmarks": None,
            "timestamp": 0,
        }

    def start(self):
        """
        Open the webcam and start capturing.

        Returns:
            bool: True if the camera opened successfully.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return False

        print(f"[INFO] Webcam opened (index {self.camera_index}).")
        return True

    def stop(self):
        """
        Stop capturing and release all resources.
        """
        if self.cap is not None:
            self.cap.release()
        self.pose.close()
        print("[INFO] PoseExtractor stopped.")

    def update(self):
        """
        Read one frame from the webcam, run MediaPipe, and store results.

        Returns:
            bool: True if a frame was successfully read and processed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Extract landmarks as a list of dictionaries
        landmarks = self._extract_landmarks(results)

        self._current_data = {
            "frame": frame,
            "landmarks": landmarks,
            "timestamp": time.time(),
        }

        return True

    def _extract_landmarks(self, results):
        """
        Convert MediaPipe results into a clean list of landmark dictionaries.

        Args:
            results: MediaPipe Pose results object.

        Returns:
            list or None: List of 33 landmark dicts with keys (x, y, z, visibility),
                          or None if no pose was detected.
        """
        if results.pose_landmarks is None:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,           # Normalized [0, 1], left-to-right
                "y": lm.y,           # Normalized [0, 1], top-to-bottom
                "z": lm.z,           # Depth relative to hips
                "visibility": lm.visibility,  # Confidence [0, 1]
            })

        return landmarks

    def get_current_data(self):
        """
        Get the most recent frame and landmarks.

        This is also the interface for other modules (e.g., blinking detection)
        that need access to the same camera frame without opening a second
        VideoCapture on the same device.

        Returns:
            dict: {
                "frame": np.array (BGR image),
                "landmarks": list of 33 dicts or None,
                "timestamp": float
            }
        """
        return self._current_data.copy()

    def draw_landmarks_on_frame(self, frame, landmarks):
        """
        Draw pose landmarks and connections on a frame for visualization.

        Args:
            frame (np.array): The BGR image to draw on (modified in place).
            landmarks (list): List of landmark dicts from _extract_landmarks.

        Returns:
            np.array: The frame with landmarks drawn on it.
        """
        if landmarks is None:
            return frame

        h, w, _ = frame.shape

        # Draw connections between key posture landmarks
        connections = [
            (11, 12),  # Left shoulder — Right shoulder
            (11, 13),  # Left shoulder — Left elbow
            (12, 14),  # Right shoulder — Right elbow
            (11, 23),  # Left shoulder — Left hip
            (12, 24),  # Right shoulder — Right hip
            (23, 24),  # Left hip — Right hip
            (0, 7),    # Nose — Left ear
            (0, 8),    # Nose — Right ear
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if start["visibility"] > 0.5 and end["visibility"] > 0.5:
                pt1 = (int(start["x"] * w), int(start["y"] * h))
                pt2 = (int(end["x"] * w), int(end["y"] * h))
                cv2.line(frame, pt1, pt2, config.COLOR_LANDMARK, config.LINE_THICKNESS)

        # Draw key landmark points
        key_indices = [0, 7, 8, 11, 12, 23, 24]
        for idx in key_indices:
            lm = landmarks[idx]
            if lm["visibility"] > 0.5:
                cx = int(lm["x"] * w)
                cy = int(lm["y"] * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        return frame
