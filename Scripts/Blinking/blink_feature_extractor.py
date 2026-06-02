"""
blink_feature_extractor.py — Sliding Window Feature Extraction for Blinking

Collects frame-by-frame EAR values, detects blinks, and computes
7 features per sliding window (60s window, 10s step):

1. b_ear_mean      — mean EAR across frames
2. b_ear_std       — std of EAR (eye openness variability)
3. b_perclos       — fraction of frames with eyes closed
4. b_blink_count   — number of complete blinks
5. b_blink_freq    — blinks per minute
6. b_ibi_mean      — mean inter-blink interval (seconds)
7. b_ibi_std       — std of inter-blink interval (regularity)
"""

import numpy as np
import blink_config as config


class BlinkFeatureExtractor:
    """
    Accumulates per-frame eye data, detects blinks, computes window features.

    Usage:
        extractor = BlinkFeatureExtractor()
        extractor.add_sample(timestamp, ear, is_closed)
        ...
        rows = extractor.compute_windows()
    """

    def __init__(self):
        self._samples = []  # list of (timestamp, ear, is_closed)

    def reset(self):
        """Clear all accumulated samples."""
        self._samples = []

    def add_sample(self, timestamp, ear, is_closed):
        """
        Add one frame's data.

        Args:
            timestamp (float): Time in seconds from start of video.
            ear (float): Eye Aspect Ratio value.
            is_closed (bool): Whether eyes are classified as closed.
        """
        self._samples.append((timestamp, ear, is_closed))

    def get_sample_count(self):
        """Return the number of samples collected."""
        return len(self._samples)

    def compute_windows(self):
        """
        Detect blinks globally, then compute features for all sliding windows.

        Returns:
            list of dict: One dict per window with 7 features + window_start.
        """
        if not self._samples:
            return []

        samples = np.array(
            [(t, ear, float(closed)) for t, ear, closed in self._samples],
            dtype=float,
        )
        timestamps = samples[:, 0]
        ears = samples[:, 1]
        closed_flags = samples[:, 2].astype(bool)

        video_duration = timestamps[-1]
        window_size = config.WINDOW_SIZE
        window_step = config.WINDOW_STEP

        if video_duration < window_size:
            print(f"[WARNING] Video ({video_duration:.1f}s) shorter than "
                  f"window ({window_size}s). No windows generated.")
            return []

        # Detect all blinks globally
        blink_times = self._detect_blinks(timestamps, closed_flags)

        rows = []
        window_start = 0.0

        while window_start + window_size <= video_duration + 0.001:
            window_end = window_start + window_size

            # Select frames in this window
            mask = (timestamps >= window_start) & (timestamps < window_end)
            w_ears = ears[mask]
            w_closed = closed_flags[mask]

            if len(w_ears) < 5:
                window_start += window_step
                continue

            # EAR stats
            ear_mean = float(np.mean(w_ears))
            ear_std = float(np.std(w_ears))

            # PERCLOS
            perclos = float(np.mean(w_closed))

            # Blinks in this window
            w_blinks = [t for t in blink_times
                        if window_start <= t < window_end]
            blink_count = len(w_blinks)

            # Blink frequency (blinks per minute)
            window_duration_min = window_size / 60.0
            blink_freq = blink_count / window_duration_min

            # Inter-blink intervals within this window
            ibi_mean, ibi_std = self._compute_ibi(w_blinks)

            row = {
                "window_start": round(window_start, 1),
                "b_ear_mean": round(ear_mean, 4),
                "b_ear_std": round(ear_std, 4),
                "b_perclos": round(perclos, 4),
                "b_blink_count": blink_count,
                "b_blink_freq": round(blink_freq, 2),
                "b_ibi_mean": round(ibi_mean, 4) if ibi_mean is not None else 0.0,
                "b_ibi_std": round(ibi_std, 4) if ibi_std is not None else 0.0,
            }

            rows.append(row)
            window_start += window_step

        return rows

    def _detect_blinks(self, timestamps, closed_flags):
        """
        Detect complete blinks from the full sequence.

        A blink is: MIN_BLINK_FRAMES or more consecutive closed-eye frames,
        followed by an open-eye frame, with total closed duration <= MAX_BLINK_SECONDS.

        Args:
            timestamps (np.array): Frame timestamps.
            closed_flags (np.array): Boolean, True = eyes closed.

        Returns:
            list of float: Timestamp of each detected blink (at eye re-opening).
        """
        blink_times = []
        closed_start_idx = None
        closed_count = 0

        for i, is_closed in enumerate(closed_flags):
            if is_closed:
                if closed_start_idx is None:
                    closed_start_idx = i
                    closed_count = 0
                closed_count += 1
            else:
                if closed_start_idx is not None:
                    # Eyes just opened — check if this was a valid blink
                    closed_duration = timestamps[i] - timestamps[closed_start_idx]
                    if (closed_count >= config.MIN_BLINK_FRAMES
                            and closed_duration <= config.MAX_BLINK_SECONDS):
                        blink_times.append(float(timestamps[i]))

                    closed_start_idx = None
                    closed_count = 0

        return blink_times

    @staticmethod
    def _compute_ibi(blink_times):
        """
        Compute mean and std of inter-blink intervals.

        Args:
            blink_times (list of float): Sorted blink timestamps.

        Returns:
            tuple: (mean_ibi, std_ibi) or (None, None) if < 2 blinks.
        """
        if len(blink_times) < 2:
            return None, None

        ibis = [blink_times[i+1] - blink_times[i]
                for i in range(len(blink_times) - 1)]

        return float(np.mean(ibis)), float(np.std(ibis))
