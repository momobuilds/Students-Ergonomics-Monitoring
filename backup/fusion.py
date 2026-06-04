"""Sensor fusion scoring for the simulated ergonomics prototype."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert sensor values to floats without failing on missing readings."""
    try:
        if pd.isna(value):
            return default
    except TypeError:
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if math.isfinite(number) else default


def _get_value(sensor_row: Any, key: str, default: Any = None) -> Any:
    if hasattr(sensor_row, "get"):
        return sensor_row.get(key, default)
    return default


def calculate_posture_score(posture_status, slouching_time_minutes) -> float:
    posture = str(posture_status or "").strip().lower()
    slouching_time = _to_float(slouching_time_minutes)

    if posture == "slouching" or slouching_time > 5:
        return 1.0
    if 2 <= slouching_time <= 5:
        return 0.5
    if posture == "good" and slouching_time < 2:
        return 0.0
    return 0.0


def calculate_blink_score(blink_rate_per_min) -> float:
    blink_rate = _to_float(blink_rate_per_min, default=12.0)

    if blink_rate >= 12:
        return 0.0
    if 7 <= blink_rate < 12:
        return 0.5
    return 1.0


def calculate_heart_score(heart_rate_bpm, baseline_heart_rate_bpm) -> float:
    heart_rate = _to_float(heart_rate_bpm)
    baseline = _to_float(baseline_heart_rate_bpm, default=heart_rate)
    increase = heart_rate - baseline

    if increase >= 25:
        return 1.0
    if 10 <= increase < 25:
        return 0.5
    return 0.0


def calculate_temperature_score(temperature_c, baseline_temperature_c=36.5) -> float:
    temperature = _to_float(temperature_c, default=baseline_temperature_c)
    baseline = _to_float(baseline_temperature_c, default=36.5)
    difference = abs(temperature - baseline)

    if difference > 1.0:
        return 1.0
    if 0.5 <= difference <= 1.0:
        return 0.5
    return 0.0


def calculate_fusion_result(sensor_row) -> dict:
    """Combine feature-level scores into the final decision-level risk status."""
    posture_status = _get_value(sensor_row, "posture_status")
    slouching_time = _get_value(sensor_row, "slouching_time_minutes")
    blink_rate = _get_value(sensor_row, "blink_rate_per_min")
    heart_rate = _get_value(sensor_row, "heart_rate_bpm")
    baseline_heart_rate = _get_value(sensor_row, "baseline_heart_rate_bpm")
    temperature = _get_value(sensor_row, "temperature_c")

    posture_score = calculate_posture_score(posture_status, slouching_time)
    blink_score = calculate_blink_score(blink_rate)
    heart_score = calculate_heart_score(heart_rate, baseline_heart_rate)
    temperature_score = calculate_temperature_score(temperature)

    final_risk_score = (
        0.35 * posture_score
        + 0.30 * blink_score
        + 0.25 * heart_score
        + 0.10 * temperature_score
    )

    if final_risk_score >= 0.60:
        status = "High Risk"
    elif final_risk_score >= 0.30:
        status = "Warning"
    else:
        status = "Normal"

    reasons = []
    posture = str(posture_status or "").strip().lower()
    slouching_time_value = _to_float(slouching_time)
    if slouching_time_value > 5:
        reasons.append("long slouching duration")
    elif posture == "slouching":
        reasons.append("slouching posture detected")
    elif posture_score == 0.5:
        reasons.append("moderate slouching duration")

    if blink_score == 1.0:
        reasons.append("low blink rate")
    elif blink_score == 0.5:
        reasons.append("reduced blink rate")

    if heart_score > 0:
        reasons.append("elevated heart rate")

    if temperature_score > 0:
        reasons.append("temperature outside baseline range")

    if not reasons:
        reasons.append("all sensor readings are near baseline")

    return {
        "final_risk_score": round(final_risk_score, 3),
        "status": status,
        "posture_score": posture_score,
        "blink_score": blink_score,
        "heart_score": heart_score,
        "temperature_score": temperature_score,
        "reasons": reasons,
    }


def apply_fusion_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add fusion scores, status, and reason text to every dataframe row."""
    if df.empty:
        return df.copy()

    results = df.apply(lambda row: pd.Series(calculate_fusion_result(row)), axis=1)
    if "reasons" in results.columns:
        results["reasons"] = results["reasons"].apply(
            lambda items: "; ".join(items) if isinstance(items, list) else str(items)
        )

    return pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
