"""
posture_analyzer.py — Postural Angle Computation

Receives raw landmarks from PoseExtractor and computes three postural
parameters from the frontal camera:

1. Shoulder Asymmetry: angle of shoulder line vs horizontal (degrees)
2. Head Lateral Tilt: lateral tilt of head relative to shoulders (degrees)
3. Forward Head Proxy: vertical distance nose-shoulder change (z-based)

Returns raw numerical values only — no classification, no thresholds.
"""

import math
import config


class PostureAnalyzer:
    """
    Computes posture metrics from MediaPipe landmarks.

    Usage:
        analyzer = PostureAnalyzer()
        values = analyzer.extract_values(landmarks)
        # values = (shoulder_asym, head_tilt, fhp) or None
    """

    def extract_values(self, landmarks):
        """
        Extract the 3 raw posture values from a single frame's landmarks.

        Args:
            landmarks (list or None): 33 MediaPipe landmarks.

        Returns:
            tuple or None: (shoulder_asymmetry, head_lateral_tilt, forward_head_proxy)
                           Returns None if landmarks are missing or insufficient.
        """
        if landmarks is None:
            return None

        shoulder_asym = self._compute_shoulder_asymmetry(landmarks)
        head_tilt = self._compute_head_lateral_tilt(landmarks)
        fhp = self._compute_forward_head_proxy(landmarks)

        # All three must be valid
        if any(v is None for v in [shoulder_asym, head_tilt, fhp]):
            return None

        return (shoulder_asym, head_tilt, fhp)

    # =========================================================================
    # METRIC COMPUTATIONS
    # =========================================================================

    def _compute_shoulder_asymmetry(self, landmarks):
        """
        Angle of the shoulder line relative to horizontal.
        0° = perfectly level shoulders.

        Landmarks: 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: Angle in degrees. Positive = right shoulder lower.
        """
        left = landmarks[11]
        right = landmarks[12]

        if left["visibility"] < config.MIN_VISIBILITY or right["visibility"] < config.MIN_VISIBILITY:
            return None

        dx = right["x"] - left["x"]
        dy = right["y"] - left["y"]

        if abs(dx) < 1e-6:
            return 0.0

        angle_rad = math.atan2(dy, abs(dx))
        return math.degrees(angle_rad)

    def _compute_head_lateral_tilt(self, landmarks):
        """
        Lateral tilt of the head: angle between ear midpoint → shoulder
        midpoint line and the vertical axis.
        0° = head perfectly centered above shoulders.

        Landmarks: 7 (left ear), 8 (right ear), 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: Angle in degrees. Positive = tilted right.
        """
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if any(lm["visibility"] < config.MIN_VISIBILITY
               for lm in [left_ear, right_ear, left_shoulder, right_shoulder]):
            return None

        ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
        ear_mid_y = (left_ear["y"] + right_ear["y"]) / 2
        shoulder_mid_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2

        dx = ear_mid_x - shoulder_mid_x
        dy = ear_mid_y - shoulder_mid_y

        angle_rad = math.atan2(dx, -dy)
        return math.degrees(angle_rad)

    def _compute_forward_head_proxy(self, landmarks):
        """
        Estimate forward head posture from frontal view using z-depth
        difference between nose and shoulder midpoint.

        Landmarks: 0 (nose), 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: z-depth difference * 100 (proxy value).
        """
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if any(lm["visibility"] < config.MIN_VISIBILITY
               for lm in [nose, left_shoulder, right_shoulder]):
            return None

        shoulder_mid_z = (left_shoulder["z"] + right_shoulder["z"]) / 2
        z_diff = nose["z"] - shoulder_mid_z

        return z_diff * 100
