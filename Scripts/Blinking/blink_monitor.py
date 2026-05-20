"""Temporal blinking metrics, alert logic, and CSV logging."""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Any

import blink_config as config


class BlinkMonitor:
    """Converts frame-level eye state into blinking indicators over time."""

    def __init__(
        self,
        csv_output_path: Path = config.CSV_OUTPUT_PATH,
        metrics_window_seconds: float = config.METRICS_WINDOW_SECONDS,
    ) -> None:
        self.csv_output_path = csv_output_path
        self.metrics_window_seconds = metrics_window_seconds
        self.samples: deque[tuple[float, bool, float | None]] = deque()
        self.blink_timestamps: deque[float] = deque()
        self.inter_blink_intervals: deque[float] = deque(maxlen=20)

        self.closed_start: float | None = None
        self.closed_frame_count = 0
        self.last_blink_timestamp: float | None = None
        self.low_blink_start: float | None = None
        self.high_perclos_start: float | None = None
        self.last_log_time = time.time()
        self._csv_buffer: list[dict[str, Any]] = []

        self.current_state: dict[str, Any] = {
            "ear": None,
            "eye_state": "Unknown",
            "blink_count_window": 0,
            "blink_frequency_bpm": 0.0,
            "perclos": 0.0,
            "last_inter_blink_interval": None,
            "mean_inter_blink_interval": None,
            "low_blink_alert": False,
            "high_perclos_alert": False,
        }
        self._init_csv()

    def update(self, analysis: dict[str, Any], timestamp: float | None = None) -> dict[str, Any]:
        now = timestamp or time.time()
        is_closed = bool(analysis.get("is_closed", False))
        ear = analysis.get("ear")

        self.samples.append((now, is_closed, ear))
        self._drop_old_samples(now)
        self._track_blink_event(is_closed, now)
        self._drop_old_blinks(now)

        blink_frequency = self._blink_frequency_bpm()
        perclos = self._perclos()
        low_blink_alert = self._persistent_condition(
            blink_frequency < config.MIN_BLINKS_PER_MINUTE,
            "low_blink_start",
            config.LOW_BLINK_ALERT_SECONDS,
            now,
        )
        high_perclos_alert = self._persistent_condition(
            perclos > config.PERCLOS_BAD_THRESHOLD,
            "high_perclos_start",
            config.HIGH_PERCLOS_ALERT_SECONDS,
            now,
        )

        self.current_state = {
            "timestamp": now,
            "face_detected": analysis.get("face_detected", False),
            "ear": ear,
            "ear_threshold": analysis.get("ear_threshold"),
            "eye_state": analysis.get("eye_state", "Unknown"),
            "blink_count_window": len(self.blink_timestamps),
            "blink_frequency_bpm": blink_frequency,
            "perclos": perclos,
            "last_inter_blink_interval": self._last_ibi(),
            "mean_inter_blink_interval": self._mean_ibi(),
            "low_blink_alert": low_blink_alert,
            "high_perclos_alert": high_perclos_alert,
        }
        self._csv_buffer.append(dict(self.current_state))

        if now - self.last_log_time >= config.CSV_LOG_INTERVAL_SECONDS:
            self.save_csv_snapshot()
            self.last_log_time = now

        return self.current_state

    def get_current_state(self) -> dict[str, Any]:
        return dict(self.current_state)

    def save_csv_snapshot(self) -> int:
        """Save all blink states collected since the previous CSV write."""
        if not self._csv_buffer:
            return 0

        rows_saved = len(self._csv_buffer)
        self._write_csv_rows(self._csv_buffer)
        self._csv_buffer = []
        print(f"[INFO] Blink CSV snapshot saved: {rows_saved} rows.")
        return rows_saved

    def _track_blink_event(self, is_closed: bool, now: float) -> None:
        if is_closed:
            if self.closed_start is None:
                self.closed_start = now
                self.closed_frame_count = 0
            self.closed_frame_count += 1
            return

        if self.closed_start is None:
            return

        closed_duration = now - self.closed_start
        is_valid_blink = (
            self.closed_frame_count >= config.MIN_BLINK_FRAMES
            and closed_duration <= config.MAX_BLINK_SECONDS
        )
        if is_valid_blink:
            self.blink_timestamps.append(now)
            if self.last_blink_timestamp is not None:
                self.inter_blink_intervals.append(now - self.last_blink_timestamp)
            self.last_blink_timestamp = now

        self.closed_start = None
        self.closed_frame_count = 0

    def _drop_old_samples(self, now: float) -> None:
        cutoff = now - self.metrics_window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def _drop_old_blinks(self, now: float) -> None:
        cutoff = now - self.metrics_window_seconds
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

    def _blink_frequency_bpm(self) -> float:
        if not self.samples:
            return 0.0
        elapsed = max(self.samples[-1][0] - self.samples[0][0], 1.0)
        return len(self.blink_timestamps) * 60.0 / elapsed

    def _perclos(self) -> float:
        if not self.samples:
            return 0.0
        closed = sum(1 for _, is_closed, _ in self.samples if is_closed)
        return closed / len(self.samples)

    def _last_ibi(self) -> float | None:
        if not self.inter_blink_intervals:
            return None
        return self.inter_blink_intervals[-1]

    def _mean_ibi(self) -> float | None:
        if not self.inter_blink_intervals:
            return None
        return sum(self.inter_blink_intervals) / len(self.inter_blink_intervals)

    def _persistent_condition(
        self,
        condition: bool,
        start_attribute: str,
        duration_seconds: float,
        now: float,
    ) -> bool:
        start = getattr(self, start_attribute)
        if condition:
            if start is None:
                setattr(self, start_attribute, now)
                return False
            return now - start >= duration_seconds

        setattr(self, start_attribute, None)
        return False

    def _init_csv(self) -> None:
        self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Start each run with a fresh session log, like the posture monitor.
        with self.csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "timestamp",
                    "ear",
                    "blink_frequency_bpm",
                    "perclos",
                    "last_inter_blink_interval",
                    "mean_inter_blink_interval",
                    "low_blink_alert",
                    "high_perclos_alert",
                ]
            )

    def _write_csv_rows(self, states: list[dict[str, Any]]) -> None:
        with self.csv_output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            for state in states:
                writer.writerow(
                    [
                        state.get("timestamp"),
                        state.get("ear"),
                        state.get("blink_frequency_bpm"),
                        state.get("perclos"),
                        state.get("last_inter_blink_interval"),
                        state.get("mean_inter_blink_interval"),
                        state.get("low_blink_alert"),
                        state.get("high_perclos_alert"),
                    ]
                )
