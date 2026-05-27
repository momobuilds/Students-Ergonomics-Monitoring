"""
posture_monitor.py — Temporal Logic, Alerts, and Data Logging

This module manages:
- Saving RAW frame-by-frame data (timestamp + 3 metric values) to CSV
- Window-based evaluation every 60 seconds:
    For each of the 3 parameters, compute % of frames where it exceeded
    its threshold. If ANY parameter is bad for >= 60% of frames → "Bad Posture".
- Summary CSV with the verdict + per-parameter violation % + averages
- Exposing current state as a dictionary for real-time dashboard

Two CSV files:
    data/raw/posture/posture_raw.csv       — every frame, raw values only
    data/processed/posture/posture_log.csv  — one row per 60-second window
"""

import csv
import time
import config


class PostureMonitor:
    """
    Monitors posture over time using a window-based evaluation approach.

    Usage:
        monitor = PostureMonitor()
        monitor.start_session()

        # In the main loop:
        monitor.update(analysis_result)
        state = monitor.get_current_state()

        monitor.end_session()
    """

    def __init__(self, summary_csv_path=None, raw_csv_path=None):
        """
        Initialize the PostureMonitor.

        Args:
            summary_csv_path (str): Path for the summary CSV.
            raw_csv_path (str): Path for the raw frame-by-frame CSV.
        """
        self.summary_csv_path = summary_csv_path or config.CSV_OUTPUT_PATH
        self.raw_csv_path = raw_csv_path or config.RAW_CSV_OUTPUT_PATH

        # --- Window buffer ---
        # Collects per-frame data for the current 60-second window
        self._window_buffer = []
        self._window_start = None

        # --- Session counters ---
        self._session_start = None
        self._last_update_time = None
        self._total_good_seconds = 0
        self._total_bad_seconds = 0
        self._alert_count = 0

        # --- Latest window evaluation result ---
        self._last_evaluation = None

        # --- Current state (shared with dashboard / overlay) ---
        self._current_state = {
            "status": "Monitoring...",
            "metrics": {},
            "session_duration": 0,
            "total_good_seconds": 0,
            "total_bad_seconds": 0,
            "good_posture_percentage": 0,
            "alert_count": 0,
            "window_seconds_left": config.EVALUATION_WINDOW_SECONDS,
            "last_evaluation": None,
        }

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def start_session(self):
        """
        Start a new monitoring session. Creates both CSV files.
        """
        now = time.time()
        self._session_start = now
        self._last_update_time = now
        self._window_start = now
        self._window_buffer = []
        self._total_good_seconds = 0
        self._total_bad_seconds = 0
        self._alert_count = 0
        self._last_evaluation = None

        self._init_raw_csv()
        self._init_summary_csv()

        print(f"[INFO] Monitoring session started.")
        print(f"       Raw CSV:     {self.raw_csv_path}")
        print(f"       Summary CSV: {self.summary_csv_path}")

    def end_session(self):
        """
        End the current session. Evaluates any remaining data in the window.
        """
        # Evaluate whatever is left in the buffer (partial window)
        if self._window_buffer:
            self._evaluate_window()

        session_duration = time.time() - self._session_start if self._session_start else 0
        print(f"[INFO] Session ended. Duration: {session_duration:.0f}s, "
              f"Alerts: {self._alert_count}, "
              f"Good posture: {self._get_good_percentage():.1f}%")

    # =========================================================================
    # MAIN UPDATE (called every frame)
    # =========================================================================

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

        pose_detected = analysis_result.get("pose_detected", False)
        metrics = analysis_result.get("metrics", {})

        # --- Extract raw values ---
        shoulder_val = None
        shoulder_bad = None
        head_val = None
        head_bad = None
        fhp_val = None
        fhp_bad = None

        if pose_detected and metrics:
            sa = metrics.get("shoulder_asymmetry")
            if sa is not None:
                shoulder_val = sa["value"]
                shoulder_bad = not sa["is_good"]

            ht = metrics.get("head_lateral_tilt")
            if ht is not None:
                head_val = ht["value"]
                head_bad = not ht["is_good"]

            fhp = metrics.get("forward_head_proxy")
            if fhp is not None:
                fhp_val = fhp["value"]
                fhp_bad = not fhp["is_good"]

        # --- Save raw frame to CSV ---
        self._save_raw_frame(now, shoulder_val, head_val, fhp_val)

        # --- Add to window buffer ---
        self._window_buffer.append({
            "timestamp": now,
            "shoulder_asymmetry": shoulder_val,
            "head_lateral_tilt": head_val,
            "forward_head_proxy": fhp_val,
            "shoulder_bad": shoulder_bad,
            "head_bad": head_bad,
            "fhp_bad": fhp_bad,
            "pose_detected": pose_detected,
        })

        # --- Update session time counters ---
        if pose_detected:
            frame_status = analysis_result.get("status", "Unknown")
            if frame_status == "Good":
                self._total_good_seconds += dt
            elif frame_status == "Slouching":
                self._total_bad_seconds += dt

        # --- Check if window is complete ---
        window_elapsed = now - self._window_start if self._window_start else 0
        if window_elapsed >= config.EVALUATION_WINDOW_SECONDS:
            self._evaluate_window()
            self._window_start = now
            self._window_buffer = []

        # --- Build current state ---
        session_duration = now - self._session_start if self._session_start else 0
        window_left = max(0, config.EVALUATION_WINDOW_SECONDS - window_elapsed)

        self._current_state = {
            "status": analysis_result.get("status", "Unknown"),
            "metrics": metrics,
            "session_duration": round(session_duration, 1),
            "total_good_seconds": round(self._total_good_seconds, 1),
            "total_bad_seconds": round(self._total_bad_seconds, 1),
            "good_posture_percentage": round(self._get_good_percentage(), 1),
            "alert_count": self._alert_count,
            "window_seconds_left": round(window_left, 1),
            "last_evaluation": self._last_evaluation,
        }

        return self._current_state

    def get_current_state(self):
        """
        Get the current monitoring state.

        Returns:
            dict: Current state including status, metrics, timers, and evaluation.
        """
        return self._current_state.copy()

    def save_csv_snapshot(self):
        """
        Force-evaluate the current window (e.g., when user presses 's').
        """
        if self._window_buffer:
            self._evaluate_window()
            self._window_start = time.time()
            self._window_buffer = []
            print("[INFO] Manual CSV snapshot saved.")
        else:
            print("[INFO] No frames buffered for snapshot.")

    # =========================================================================
    # WINDOW EVALUATION (the core new logic)
    # =========================================================================

    def _evaluate_window(self):
        """
        Evaluate the current window buffer.

        For each of the 3 parameters, compute:
        - % of frames where it exceeded its threshold
        - average value across the window

        Verdict: if ANY parameter is bad for >= BAD_POSTURE_PCT_THRESHOLD %
        of frames → "Bad Posture", otherwise "Good Posture".
        """
        if not self._window_buffer:
            return

        # Count frames where each parameter was bad
        counts = {
            "shoulder_asymmetry": {"bad": 0, "total": 0, "sum": 0},
            "head_lateral_tilt":  {"bad": 0, "total": 0, "sum": 0},
            "forward_head_proxy": {"bad": 0, "total": 0, "sum": 0},
        }

        for frame in self._window_buffer:
            if not frame["pose_detected"]:
                continue

            # Shoulder asymmetry
            if frame["shoulder_asymmetry"] is not None:
                counts["shoulder_asymmetry"]["total"] += 1
                counts["shoulder_asymmetry"]["sum"] += abs(frame["shoulder_asymmetry"])
                if frame["shoulder_bad"]:
                    counts["shoulder_asymmetry"]["bad"] += 1

            # Head lateral tilt
            if frame["head_lateral_tilt"] is not None:
                counts["head_lateral_tilt"]["total"] += 1
                counts["head_lateral_tilt"]["sum"] += abs(frame["head_lateral_tilt"])
                if frame["head_bad"]:
                    counts["head_lateral_tilt"]["bad"] += 1

            # Forward head proxy
            if frame["forward_head_proxy"] is not None:
                counts["forward_head_proxy"]["total"] += 1
                counts["forward_head_proxy"]["sum"] += abs(frame["forward_head_proxy"])
                if frame["fhp_bad"]:
                    counts["forward_head_proxy"]["bad"] += 1

        # Compute per-parameter results
        param_results = {}
        any_param_bad = False
        threshold = config.BAD_POSTURE_PCT_THRESHOLD

        for param_name, c in counts.items():
            if c["total"] > 0:
                bad_pct = (c["bad"] / c["total"]) * 100
                avg_val = c["sum"] / c["total"]
                is_bad = bad_pct >= threshold
            else:
                bad_pct = 0
                avg_val = 0
                is_bad = False

            param_results[param_name] = {
                "bad_pct": round(bad_pct, 1),
                "avg_value": round(avg_val, 2),
                "is_bad": is_bad,
                "total_frames": c["total"],
                "bad_frames": c["bad"],
            }

            if is_bad:
                any_param_bad = True

        # Overall verdict
        verdict = "Bad Posture" if any_param_bad else "Good Posture"

        if any_param_bad:
            self._alert_count += 1

        # Build evaluation result
        period_start = self._window_buffer[0]["timestamp"]
        period_end = self._window_buffer[-1]["timestamp"]

        evaluation = {
            "verdict": verdict,
            "params": param_results,
            "period_start": period_start,
            "period_end": period_end,
            "total_frames": len(self._window_buffer),
        }

        self._last_evaluation = evaluation

        # Print to console
        self._print_evaluation(evaluation)

        # Save to summary CSV
        self._save_summary_row(evaluation)

    def _print_evaluation(self, evaluation):
        """
        Print the window evaluation result to the console.
        """
        verdict = evaluation["verdict"]
        params = evaluation["params"]

        print(f"\n{'='*60}")
        print(f"  POSTURE EVALUATION — {verdict.upper()}")
        print(f"  Window: {evaluation['total_frames']} frames")
        print(f"{'='*60}")

        for name, p in params.items():
            display_name = name.replace("_", " ").title()
            status = "BAD" if p["is_bad"] else "OK "
            unit = "%" if name == "forward_head_proxy" else "deg"
            print(f"  [{status}] {display_name:>25s}: "
                  f"bad {p['bad_pct']:5.1f}% of frames | "
                  f"avg {p['avg_value']:6.2f}{unit}")

        print(f"{'='*60}\n")

    # =========================================================================
    # RAW CSV (frame-by-frame)
    # =========================================================================

    def _init_raw_csv(self):
        """
        Create the raw CSV file with headers.
        """
        with open(self.raw_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "shoulder_asymmetry",
                "head_lateral_tilt",
                "forward_head_proxy",
            ])

    def _save_raw_frame(self, timestamp, shoulder_val, head_val, fhp_val):
        """
        Append one raw frame row to the raw CSV.
        """
        with open(self.raw_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                round(shoulder_val, 4) if shoulder_val is not None else "",
                round(head_val, 4) if head_val is not None else "",
                round(fhp_val, 4) if fhp_val is not None else "",
            ])

    # =========================================================================
    # SUMMARY CSV (one row per window)
    # =========================================================================

    def _init_summary_csv(self):
        """
        Create the summary CSV file with headers.
        """
        with open(self.summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "period_start",
                "period_end",
                "verdict",
                "shoulder_bad_pct",
                "head_tilt_bad_pct",
                "fhp_bad_pct",
                "avg_shoulder_asymmetry",
                "avg_head_lateral_tilt",
                "avg_forward_head_proxy",
                "alert_count",
            ])

    def _save_summary_row(self, evaluation):
        """
        Write one summary row to the summary CSV.
        """
        params = evaluation["params"]
        period_start = evaluation["period_start"]
        period_end = evaluation["period_end"]

        with open(self.summary_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%H:%M:%S", time.localtime(period_start)),
                time.strftime("%H:%M:%S", time.localtime(period_end)),
                evaluation["verdict"],
                params["shoulder_asymmetry"]["bad_pct"],
                params["head_lateral_tilt"]["bad_pct"],
                params["forward_head_proxy"]["bad_pct"],
                params["shoulder_asymmetry"]["avg_value"],
                params["head_lateral_tilt"]["avg_value"],
                params["forward_head_proxy"]["avg_value"],
                self._alert_count,
            ])

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
