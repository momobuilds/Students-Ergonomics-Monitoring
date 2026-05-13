"""
posture_monitor.py — Temporal Logic, Alerts, and Data Logging

This module manages:
- Tracking how long the user has been in bad posture continuously
- Two-tier alert system (all bad 30s / any bad 60s)
- Periodic CSV logging for Streamlit dashboard and OpenClaw
- Exposing current state as a dictionary for real-time dashboard
"""

import csv
import time
import config


class PostureMonitor:
    """
    Monitors posture over time and manages alerts and data logging.

    Usage:
        monitor = PostureMonitor()
        monitor.start_session()

        # In the main loop:
        monitor.update(analysis_result)
        state = monitor.get_current_state()

        monitor.end_session()
    """

    def __init__(self, csv_path=None):
        """
        Initialize the PostureMonitor.

        Args:
            csv_path (str): Path for the CSV output file.
                            Defaults to config.CSV_OUTPUT_PATH.
        """
        self.csv_path = csv_path or config.CSV_OUTPUT_PATH

        # --- Timing state ---
        # Two-tier tracking: "any bad" streak and "all bad" streak
        self._any_bad_start = None
        self._all_bad_start = None
        self._total_bad_seconds = 0
        self._total_good_seconds = 0

        # --- Alert state ---
        self._alert_active = False
        self._alert_count = 0
        self._alert_tier = None

        # --- Session state ---
        self._session_start = None
        self._last_csv_save = None
        self._last_update_time = None

        # --- Current state (shared with dashboard) ---
        self._current_state = {
            "status": "Unknown",
            "metrics": {},
            "alert_active": False,
            "alert_tier": None,
            "bad_posture_duration": 0,
            "all_bad_duration": 0,
            "session_duration": 0,
            "total_bad_seconds": 0,
            "total_good_seconds": 0,
            "good_posture_percentage": 0,
            "alert_count": 0,
        }

        # --- CSV buffer ---
        self._metrics_buffer = []

    def start_session(self):
        """
        Start a new monitoring session. Initializes timers and creates the CSV.
        """
        self._session_start = time.time()
        self._last_csv_save = time.time()
        self._last_update_time = time.time()
        self._any_bad_start = None
        self._all_bad_start = None
        self._total_bad_seconds = 0
        self._total_good_seconds = 0
        self._alert_active = False
        self._alert_tier = None
        self._alert_count = 0
        self._metrics_buffer = []

        self._init_csv()
        print(f"[INFO] Monitoring session started. CSV output: {self.csv_path}")

    def end_session(self):
        """
        End the current session. Saves any remaining data to CSV.
        """
        if self._metrics_buffer:
            self._save_csv_summary()

        session_duration = time.time() - self._session_start if self._session_start else 0
        print(f"[INFO] Session ended. Duration: {session_duration:.0f}s, "
              f"Alerts: {self._alert_count}, "
              f"Good posture: {self._get_good_percentage():.1f}%")

    def update(self, analysis_result):
        """
        Update the monitor with the latest posture analysis result.
        Called every frame.

        Args:
            analysis_result (dict): Output from PostureAnalyzer.analyze().

        Returns:
            dict: The updated current state.
        """
        now = time.time()
        dt = now - self._last_update_time if self._last_update_time else 0
        self._last_update_time = now

        status = analysis_result.get("status", "Unknown")
        pose_detected = analysis_result.get("pose_detected", False)

        if not pose_detected:
            self._current_state["status"] = "No Pose Detected"
            return self._current_state

        # --- Update time counters ---
        if status == "Slouching":
            self._total_bad_seconds += dt
            all_bad = analysis_result.get("all_bad", False)

            # Track "any bad" streak
            if self._any_bad_start is None:
                self._any_bad_start = now

            # Track "all bad" streak
            if all_bad:
                if self._all_bad_start is None:
                    self._all_bad_start = now
            else:
                self._all_bad_start = None

            # --- Two-tier alert logic ---
            if not self._alert_active:
                all_bad_duration = (now - self._all_bad_start) if self._all_bad_start else 0
                any_bad_duration = (now - self._any_bad_start) if self._any_bad_start else 0

                if all_bad_duration >= config.ALL_BAD_TIME_THRESHOLD:
                    self._alert_active = True
                    self._alert_tier = "all_bad"
                    self._alert_count += 1
                    print(f"[ALERT] ALL metrics bad for {all_bad_duration:.0f}s! "
                          f"(Alert #{self._alert_count})")

                elif any_bad_duration >= config.ANY_BAD_TIME_THRESHOLD:
                    self._alert_active = True
                    self._alert_tier = "any_bad"
                    self._alert_count += 1
                    print(f"[ALERT] Bad posture for {any_bad_duration:.0f}s! "
                          f"(Alert #{self._alert_count})")

        elif status == "Good":
            self._total_good_seconds += dt
            self._any_bad_start = None
            self._all_bad_start = None
            self._alert_active = False
            self._alert_tier = None

        # --- Buffer metrics for CSV ---
        self._metrics_buffer.append({
            "timestamp": now,
            "status": status,
            "metrics": analysis_result.get("metrics", {}),
        })

        # --- Periodic CSV save ---
        if now - self._last_csv_save >= config.CSV_SAVE_INTERVAL:
            self._save_csv_summary()
            self._last_csv_save = now

        # --- Update shared state ---
        any_bad_streak = (now - self._any_bad_start) if self._any_bad_start else 0
        all_bad_streak = (now - self._all_bad_start) if self._all_bad_start else 0
        session_duration = now - self._session_start if self._session_start else 0

        self._current_state = {
            "status": status,
            "metrics": analysis_result.get("metrics", {}),
            "alert_active": self._alert_active,
            "alert_tier": self._alert_tier,
            "bad_posture_duration": round(any_bad_streak, 1),
            "all_bad_duration": round(all_bad_streak, 1),
            "session_duration": round(session_duration, 1),
            "total_bad_seconds": round(self._total_bad_seconds, 1),
            "total_good_seconds": round(self._total_good_seconds, 1),
            "good_posture_percentage": round(self._get_good_percentage(), 1),
            "alert_count": self._alert_count,
        }

        return self._current_state

    def get_current_state(self):
        """
        Get the current monitoring state. Main interface for Streamlit dashboard.

        Returns:
            dict: Current state including status, metrics, timers, and alerts.
        """
        return self._current_state.copy()

    # =========================================================================
    # CSV MANAGEMENT
    # =========================================================================

    def _init_csv(self):
        """
        Create the CSV file with headers.
        """
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "period_start",
                "period_end",
                "good_posture_pct",
                "avg_shoulder_asymmetry",
                "avg_head_lateral_tilt",
                "avg_forward_head_proxy",
                "alerts_in_period",
                "dominant_status",
            ])

    def _save_csv_summary(self):
        """
        Aggregate buffered metrics and write a summary row to CSV.
        """
        if not self._metrics_buffer:
            return

        metric_sums = {}
        metric_counts = {}
        status_counts = {"Good": 0, "Slouching": 0}

        for entry in self._metrics_buffer:
            status = entry["status"]
            if status in status_counts:
                status_counts[status] += 1

            for key, val in entry["metrics"].items():
                if val is not None and "value" in val:
                    if key not in metric_sums:
                        metric_sums[key] = 0
                        metric_counts[key] = 0
                    metric_sums[key] += val["value"]
                    metric_counts[key] += 1

        # Compute averages
        metric_avgs = {}
        for key in metric_sums:
            if metric_counts[key] > 0:
                metric_avgs[key] = round(metric_sums[key] / metric_counts[key], 2)
            else:
                metric_avgs[key] = None

        total = status_counts["Good"] + status_counts["Slouching"]
        good_pct = (status_counts["Good"] / total * 100) if total > 0 else 0
        dominant = "Good" if status_counts["Good"] >= status_counts["Slouching"] else "Slouching"

        period_start = self._metrics_buffer[0]["timestamp"]
        period_end = self._metrics_buffer[-1]["timestamp"]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%H:%M:%S", time.localtime(period_start)),
                time.strftime("%H:%M:%S", time.localtime(period_end)),
                round(good_pct, 1),
                metric_avgs.get("shoulder_asymmetry"),
                metric_avgs.get("head_lateral_tilt"),
                metric_avgs.get("forward_head_proxy"),
                self._alert_count,
                dominant,
            ])

        print(f"[INFO] CSV summary saved. Good posture: {good_pct:.1f}%, Status: {dominant}")
        self._metrics_buffer = []

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_good_percentage(self):
        """
        Percentage of time in good posture during the session.
        """
        total = self._total_good_seconds + self._total_bad_seconds
        if total == 0:
            return 0
        return (self._total_good_seconds / total) * 100
