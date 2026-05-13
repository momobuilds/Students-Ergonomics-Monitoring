"""
posture_analyzer.py — Postural Angle Computation and Classification

This module receives raw landmarks from PoseExtractor and computes
three postural parameters from the frontal webcam:

1. Shoulder Asymmetry: angle of the shoulder line vs horizontal
2. Head Lateral Tilt: lateral tilt of the head relative to shoulders
3. Forward Head Proxy: depth/vertical change between nose and shoulders

Each metric returns a value (degrees or percentage) and a boolean (good/bad).
The final classification combines all three metrics.

Supports two modes:
- Fixed thresholds (from literature): used without calibration
- Calibrated thresholds: personalized via a 5-second calibration phase

Angle computation uses arctan2, the same approach as the lab's curl counter.
"""

import math
import numpy as np
import config


class PostureAnalyzer:
    """
    Analyzes pose landmarks and classifies posture as "Good" or "Slouching".

    Usage:
        analyzer = PostureAnalyzer()
        analyzer.set_baseline(baseline_data)  # optional calibration

        result = analyzer.analyze(landmarks)
        print(result["status"])  # "Good" or "Slouching"
        print(result["metrics"])  # dict of all computed angles
    """

    def __init__(self):
        """
        Initialize the PostureAnalyzer.
        """
        # Calibration baselines — None means "use fixed thresholds from literature".
        # When calibrated, stores the user's "good posture" average values.
        self._baseline = None
        self._is_calibrated = False

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def set_baseline(self, baseline_data):
        """
        Set the calibration baseline from "good posture" measurements.

        Args:
            baseline_data (dict): Average values during calibration:
                - "shoulder_asymmetry": float (degrees)
                - "head_lateral_tilt": float (degrees)
                - "nose_shoulder_ratio": float (for forward head proxy)
        """
        if baseline_data:
            self._baseline = baseline_data
            self._is_calibrated = True
            print(f"[INFO] Calibration baseline set:")
            for key, value in baseline_data.items():
                print(f"       {key} = {value:.2f}")

    def analyze(self, landmarks):
        """
        Perform full posture analysis on the current frame's landmarks.

        Args:
            landmarks (list or None): 33 landmarks from the frontal camera.

        Returns:
            dict: {
                "status": "Good" or "Slouching",
                "metrics": {
                    "shoulder_asymmetry": {"value": float, "is_good": bool, "unit": "deg"},
                    "head_lateral_tilt": {"value": float, "is_good": bool, "unit": "deg"},
                    "forward_head_proxy": {"value": float, "is_good": bool, "unit": "%"},
                },
                "bad_count": int,
                "total_count": int,
                "all_bad": bool,
                "pose_detected": bool,
            }
        """
        result = {
            "status": "Unknown",
            "metrics": {},
            "bad_count": 0,
            "total_count": 0,
            "all_bad": False,
            "pose_detected": False,
        }

        if landmarks is None:
            return result

        result["pose_detected"] = True

        # --- 1. Shoulder Asymmetry ---
        shoulder_asym = self._compute_shoulder_asymmetry(landmarks)
        if shoulder_asym is not None:
            if self._is_calibrated and "shoulder_asymmetry" in self._baseline:
                threshold = abs(self._baseline["shoulder_asymmetry"]) + config.CALIBRATION_MARGIN_SHOULDER_ASYMMETRY
            else:
                threshold = config.SHOULDER_ASYMMETRY_THRESHOLD

            result["metrics"]["shoulder_asymmetry"] = {
                "value": round(shoulder_asym, 1),
                "is_good": abs(shoulder_asym) < threshold,
                "unit": "deg",
            }

        # --- 2. Head Lateral Tilt ---
        head_tilt = self._compute_head_lateral_tilt(landmarks)
        if head_tilt is not None:
            if self._is_calibrated and "head_lateral_tilt" in self._baseline:
                threshold = abs(self._baseline["head_lateral_tilt"]) + config.CALIBRATION_MARGIN_HEAD_LATERAL_TILT
            else:
                threshold = config.HEAD_LATERAL_TILT_THRESHOLD

            result["metrics"]["head_lateral_tilt"] = {
                "value": round(head_tilt, 1),
                "is_good": abs(head_tilt) < threshold,
                "unit": "deg",
            }

        # --- 3. Forward Head Proxy ---
        fhp = self._compute_forward_head_proxy(landmarks)
        if fhp is not None:
            result["metrics"]["forward_head_proxy"] = {
                "value": round(fhp, 1),
                "is_good": abs(fhp) < config.CALIBRATION_MARGIN_FORWARD_HEAD_PROXY,
                "unit": "%",
            }

        # --- FINAL CLASSIFICATION ---
        metrics_with_values = [m for m in result["metrics"].values() if m is not None]
        if metrics_with_values:
            bad_count = sum(1 for m in metrics_with_values if not m["is_good"])
            total_count = len(metrics_with_values)
            all_good = bad_count == 0
            all_bad = bad_count == total_count

            result["status"] = "Good" if all_good else "Slouching"
            result["bad_count"] = bad_count
            result["total_count"] = total_count
            result["all_bad"] = all_bad

        return result

    def compute_calibration_data(self, landmarks):
        """
        Compute baseline data for all parameters during calibration.
        Called on each frame while the user is sitting upright.

        Args:
            landmarks (list): 33 MediaPipe landmarks.

        Returns:
            dict or None: Dictionary with all measured values.
        """
        if landmarks is None:
            return None

        data = {}

        shoulder_asym = self._compute_shoulder_asymmetry(landmarks)
        if shoulder_asym is not None:
            data["shoulder_asymmetry"] = shoulder_asym

        head_tilt = self._compute_head_lateral_tilt(landmarks)
        if head_tilt is not None:
            data["head_lateral_tilt"] = head_tilt

        ratio = self._get_nose_shoulder_vertical_ratio(landmarks)
        if ratio is not None:
            data["nose_shoulder_ratio"] = ratio

        return data if data else None

    def get_feature_vector(self, landmarks):
        """
        Extract a feature vector from landmarks for ML classification.

        Args:
            landmarks (list): 33 MediaPipe landmarks.

        Returns:
            list or None: [shoulder_asym, head_tilt, fhp] or None if invalid.
        """
        if landmarks is None:
            return None

        shoulder_asym = self._compute_shoulder_asymmetry(landmarks)
        head_tilt = self._compute_head_lateral_tilt(landmarks)
        fhp = self._compute_forward_head_proxy(landmarks)

        features = [
            shoulder_asym if shoulder_asym is not None else 0.0,
            head_tilt if head_tilt is not None else 0.0,
            fhp if fhp is not None else 0.0,
        ]

        return features

    # =========================================================================
    # METRIC COMPUTATIONS
    # =========================================================================

    def _compute_shoulder_asymmetry(self, landmarks):
        """
        Compute the angle of the shoulder line relative to the horizontal.
        0° = perfectly level shoulders.

        Uses landmarks: 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: Angle in degrees. Positive = right shoulder lower.
        """
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if left_shoulder["visibility"] < 0.5 or right_shoulder["visibility"] < 0.5:
            return None

        dx = right_shoulder["x"] - left_shoulder["x"]
        dy = right_shoulder["y"] - left_shoulder["y"]

        if abs(dx) < 1e-6:
            return 0.0

        angle_rad = math.atan2(dy, abs(dx))
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _compute_head_lateral_tilt(self, landmarks):
        """
        Compute lateral tilt of the head: angle between ear midpoint → shoulder
        midpoint line and the vertical axis.
        0° = head perfectly centered above shoulders.

        Uses landmarks: 7 (left ear), 8 (right ear), 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: Angle in degrees. Positive = tilted right.
        """
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if any(lm["visibility"] < 0.5 for lm in [left_ear, right_ear, left_shoulder, right_shoulder]):
            return None

        ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
        ear_mid_y = (left_ear["y"] + right_ear["y"]) / 2
        shoulder_mid_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2

        dx = ear_mid_x - shoulder_mid_x
        dy = ear_mid_y - shoulder_mid_y

        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _compute_forward_head_proxy(self, landmarks):
        """
        Estimate forward head posture from the frontal view.

        With calibration: percentage deviation of nose-shoulder vertical distance
        from the calibrated baseline (head drops when leaning forward).

        Without calibration: raw z-depth difference between nose and shoulders
        (less reliable but still provides a signal).

        Uses landmarks: 0 (nose), 11 (left shoulder), 12 (right shoulder).

        Returns:
            float or None: Percentage deviation from baseline, or raw z-diff.
        """
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if any(lm["visibility"] < 0.5 for lm in [nose, left_shoulder, right_shoulder]):
            return None

        current_ratio = self._get_nose_shoulder_vertical_ratio(landmarks)
        if current_ratio is None:
            return None

        if self._is_calibrated and "nose_shoulder_ratio" in self._baseline and self._baseline["nose_shoulder_ratio"] != 0:
            deviation = ((current_ratio - self._baseline["nose_shoulder_ratio"])
                         / self._baseline["nose_shoulder_ratio"] * 100)
            return deviation
        else:
            shoulder_mid_z = (left_shoulder["z"] + right_shoulder["z"]) / 2
            z_diff = nose["z"] - shoulder_mid_z
            return z_diff * 100

    def _get_nose_shoulder_vertical_ratio(self, landmarks):
        """
        Compute vertical distance between nose and shoulder midpoint.
        Decreases when the user leans forward.

        Returns:
            float or None: Vertical distance in normalized coordinates.
        """
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        if any(lm["visibility"] < 0.5 for lm in [nose, left_shoulder, right_shoulder]):
            return None

        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        vertical_dist = shoulder_mid_y - nose["y"]

        return vertical_dist
