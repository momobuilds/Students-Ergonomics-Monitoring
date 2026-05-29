"""
feature_extractor.py — Sliding Window Feature Extraction for Posture

Replaces posture_monitor.py. Instead of real-time monitoring and alerts,
this module:

1. Collects raw frame-by-frame posture values from a video
2. Applies a sliding window (60s window, 10s step)
3. For each window, computes 9 features:
   - mean, std, bad_pct for shoulder_asymmetry
   - mean, std, bad_pct for head_lateral_tilt
   - mean, std, bad_pct for forward_head_proxy

bad_pct = percentage of frames where |value| exceeds the literature threshold.
"""

import numpy as np
import config


class FeatureExtractor:
    """
    Computes posture features over sliding windows.

    Usage:
        extractor = FeatureExtractor()

        # Add frame-by-frame values
        extractor.add_sample(timestamp, shoulder, head_tilt, fhp)
        ...

        # Compute all windows
        rows = extractor.compute_windows()
    """

    def __init__(self):
        self._samples = []  # list of (timestamp, shoulder, head_tilt, fhp)

    def reset(self):
        """Clear all accumulated samples."""
        self._samples = []

    def add_sample(self, timestamp, shoulder_asym, head_tilt, fhp):
        """
        Add one frame's values to the buffer.

        Args:
            timestamp (float): Time in seconds from start of video.
            shoulder_asym (float): Shoulder asymmetry in degrees.
            head_tilt (float): Head lateral tilt in degrees.
            fhp (float): Forward head proxy value.
        """
        self._samples.append((timestamp, shoulder_asym, head_tilt, fhp))

    def get_sample_count(self):
        """Return the number of samples collected."""
        return len(self._samples)

    def compute_windows(self):
        """
        Compute features for all sliding windows.

        Returns:
            list of dict: One dict per window, each containing the 9 features
                          plus window_start (in seconds).
                          Returns empty list if not enough data.
        """
        if not self._samples:
            return []

        samples = np.array(self._samples)  # shape: (N, 4)
        timestamps = samples[:, 0]
        values = samples[:, 1:]  # columns: shoulder, head_tilt, fhp

        video_duration = timestamps[-1]
        window_size = config.WINDOW_SIZE
        window_step = config.WINDOW_STEP

        # First window starts at 0, last window must fit entirely in the video
        if video_duration < window_size:
            print(f"[WARNING] Video duration ({video_duration:.1f}s) is shorter "
                  f"than window size ({window_size}s). No windows generated.")
            return []

        rows = []
        window_start = 0.0

        while window_start + window_size <= video_duration + 0.001:
            window_end = window_start + window_size

            # Select frames within this window
            mask = (timestamps >= window_start) & (timestamps < window_end)
            window_values = values[mask]  # shape: (n_frames, 3)

            if len(window_values) < 5:
                # Too few frames (e.g., landmark detection failed), skip
                window_start += window_step
                continue

            # Compute features for each parameter
            # Shoulder and head_tilt: abs value, bad = above threshold
            # FHP ratio: raw value, bad = below threshold (lower = more slouching)

            shoulder_vals = np.abs(window_values[:, 0])
            head_tilt_vals = np.abs(window_values[:, 1])
            fhp_vals = window_values[:, 2]  # no abs — ratio is always positive

            row = {
                "window_start": round(window_start, 1),
                "p_shoulder_mean": round(float(np.mean(shoulder_vals)), 4),
                "p_shoulder_std": round(float(np.std(shoulder_vals)), 4),
                "p_shoulder_bad_pct": round(float(np.mean(shoulder_vals > config.SHOULDER_ASYMMETRY_THRESHOLD) * 100), 2),
                "p_head_tilt_mean": round(float(np.mean(head_tilt_vals)), 4),
                "p_head_tilt_std": round(float(np.std(head_tilt_vals)), 4),
                "p_head_tilt_bad_pct": round(float(np.mean(head_tilt_vals > config.HEAD_LATERAL_TILT_THRESHOLD) * 100), 2),
                "p_fhp_mean": round(float(np.mean(fhp_vals)), 4),
                "p_fhp_std": round(float(np.std(fhp_vals)), 4),
                "p_fhp_bad_pct": round(float(np.mean(fhp_vals < config.FORWARD_HEAD_PROXY_THRESHOLD) * 100), 2),
            }

            rows.append(row)
            window_start += window_step

        return rows
