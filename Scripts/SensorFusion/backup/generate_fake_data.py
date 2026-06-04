"""Generate simulated sensor readings for the ergonomics prototype.

Run from the repository root:
    python Scripts/SensorFusion/generate_fake_data.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import random


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "fake_sensor_data.csv"
SESSION_HOURS = 8
ROW_COUNT = SESSION_HOURS * 60
FIELDNAMES = [
    "timestamp",
    "heart_rate_bpm",
    "baseline_heart_rate_bpm",
    "blink_rate_per_min",
    "posture_status",
    "slouching_time_minutes",
    "temperature_c",
    "data_source",
]


def _normal_row(rng: random.Random, timestamp: datetime) -> dict:
    baseline_heart_rate = rng.randrange(66, 76)
    return {
        "timestamp": timestamp.isoformat(timespec="minutes"),
        "heart_rate_bpm": baseline_heart_rate + rng.randrange(-3, 8),
        "baseline_heart_rate_bpm": baseline_heart_rate,
        "blink_rate_per_min": rng.randrange(12, 20),
        "posture_status": "Good",
        "slouching_time_minutes": round(rng.uniform(0.0, 1.9), 1),
        "temperature_c": round(rng.uniform(36.2, 36.8), 1),
        "data_source": "simulated",
    }


def _warning_row(rng: random.Random, timestamp: datetime) -> dict:
    baseline_heart_rate = rng.randrange(66, 76)
    return {
        "timestamp": timestamp.isoformat(timespec="minutes"),
        "heart_rate_bpm": baseline_heart_rate + rng.randrange(10, 25),
        "baseline_heart_rate_bpm": baseline_heart_rate,
        "blink_rate_per_min": rng.randrange(7, 12),
        "posture_status": "Good",
        "slouching_time_minutes": round(rng.uniform(2.0, 5.0), 1),
        "temperature_c": round(rng.choice([35.8, 35.9, 37.1, 37.2]), 1),
        "data_source": "simulated",
    }


def _high_risk_row(rng: random.Random, timestamp: datetime) -> dict:
    baseline_heart_rate = rng.randrange(66, 76)
    return {
        "timestamp": timestamp.isoformat(timespec="minutes"),
        "heart_rate_bpm": baseline_heart_rate + rng.randrange(25, 40),
        "baseline_heart_rate_bpm": baseline_heart_rate,
        "blink_rate_per_min": rng.randrange(3, 7),
        "posture_status": "Slouching",
        "slouching_time_minutes": round(rng.uniform(5.1, 12.0), 1),
        "temperature_c": round(rng.choice([35.2, 35.3, 37.7, 37.9]), 1),
        "data_source": "simulated",
    }


def _row_for_minute(rng: random.Random, minute: int, timestamp: datetime) -> dict:
    """Simulate an 8-hour session with changing ergonomic conditions."""
    if minute < 90:
        return _normal_row(rng, timestamp)
    if minute < 150:
        return _warning_row(rng, timestamp)
    if minute < 210:
        return _high_risk_row(rng, timestamp)
    if minute < 270:
        return _warning_row(rng, timestamp)
    if minute < 360:
        return _normal_row(rng, timestamp)
    if minute < 420:
        return _warning_row(rng, timestamp)
    return _high_risk_row(rng, timestamp)


def generate_fake_data(output_path: Path | str = DEFAULT_OUTPUT_PATH) -> list[dict]:
    """Create an 8-hour minute-level simulated ergonomics session."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    start_time = datetime.now().replace(second=0, microsecond=0) - timedelta(
        minutes=ROW_COUNT - 1
    )

    rows = []
    for minute in range(ROW_COUNT):
        timestamp = start_time + timedelta(minutes=minute)
        rows.append(_row_for_minute(rng, minute, timestamp))

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    rows = generate_fake_data()
    print(f"Generated {len(rows)} simulated sensor rows.")
    print(f"Saved fake sensor data to: {DEFAULT_OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print("Data source: simulated")
    print(f"Session length: {SESSION_HOURS} hours at 1 row per minute.")
    print("Includes repeated Normal, Warning, and High Risk examples.")


if __name__ == "__main__":
    main()
