"""Generate a more realistic synthetic ergonomics sensor-fusion dataset.

The old dataset was intentionally simple: features were sampled directly from
the final label. That makes validation too easy and can produce perfect scores.
This generator instead creates hidden fatigue, posture strain, and stress
levels first, derives noisy sensor readings from those levels, and only then
samples the final wellbeing state from a probabilistic combined risk score.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["Good", "Warning", "High Risk", "Critical"]
OUTPUT_COLUMNS = [
    "subject_id",
    "session_id",
    "timestamp",
    "blink_frequency_bpm",
    "perclos",
    "last_inter_blink_interval",
    "mean_inter_blink_interval",
    "low_blink_alert",
    "high_perclos_alert",
    "good_posture_pct",
    "avg_shoulder_asymmetry",
    "avg_head_lateral_tilt",
    "avg_forward_head_proxy",
    "alerts_in_period",
    "baseline_heart_rate",
    "current_heart_rate",
    "stress_index",
    "stress_tag",
    "dominant_status",
    "final_state",
]

NUMERIC_COLUMNS = [
    "blink_frequency_bpm",
    "perclos",
    "last_inter_blink_interval",
    "mean_inter_blink_interval",
    "low_blink_alert",
    "high_perclos_alert",
    "good_posture_pct",
    "avg_shoulder_asymmetry",
    "avg_head_lateral_tilt",
    "avg_forward_head_proxy",
    "alerts_in_period",
    "baseline_heart_rate",
    "current_heart_rate",
    "stress_index",
]

CATEGORICAL_COLUMNS = ["stress_tag", "dominant_status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic ergonomics sensor data."
    )
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows to create.")
    parser.add_argument(
        "--subjects",
        type=int,
        default=40,
        help="Number of distinct subjects with personal baselines.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=5,
        help="Number of sessions per subject.",
    )
    parser.add_argument(
        "--out",
        default="realistic_synthetic_ergonomics_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def sample_final_state(risk_score: float, rng: np.random.Generator) -> str:
    """Sample a label near the risk score instead of using hard thresholds."""
    centers = np.array([0.15, 0.42, 0.68, 0.88])
    sigma = 0.17
    logits = -((risk_score - centers) ** 2) / (2 * sigma**2)
    probabilities = softmax(logits)
    return str(rng.choice(STATES, p=probabilities))


def choose_noisy_bucket(
    score: float,
    buckets: list[tuple[float, str]],
    rng: np.random.Generator,
    noise_std: float = 0.14,
    random_flip_rate: float = 0.12,
) -> str:
    """Convert a latent score to an imperfect categorical sensor summary."""
    noisy_score = clip(score + rng.normal(0, noise_std), 0, 1)
    chosen = buckets[-1][1]
    for upper_bound, label in buckets:
        if noisy_score <= upper_bound:
            chosen = label
            break

    if rng.random() < random_flip_rate:
        labels = [label for _, label in buckets]
        chosen_index = labels.index(chosen)
        nearby = [
            labels[index]
            for index in range(max(0, chosen_index - 1), min(len(labels), chosen_index + 2))
        ]
        chosen = str(rng.choice(nearby))

    return chosen


def create_subject_profiles(subjects: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for index in range(subjects):
        rows.append(
            {
                "subject_id": f"S{index + 1:03d}",
                "baseline_heart_rate": clip(rng.normal(72, 8), 52, 96),
                "fatigue_bias": rng.normal(0, 0.09),
                "posture_bias": rng.normal(0, 0.11),
                "stress_bias": rng.normal(0, 0.10),
                "blink_bias": rng.normal(0, 1.8),
            }
        )
    return pd.DataFrame(rows)


def add_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject occasional sensor dropout while leaving IDs, timestamps, and labels intact."""
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        mask = rng.random(len(df)) < 0.025
        df.loc[mask, column] = np.nan

    for column in CATEGORICAL_COLUMNS:
        mask = rng.random(len(df)) < 0.015
        df.loc[mask, column] = np.nan

    return df


def generate_dataset(
    rows: int, subjects: int, sessions: int, rng: np.random.Generator
) -> pd.DataFrame:
    subject_profiles = create_subject_profiles(subjects, rng)
    session_loads = {
        (subject_id, session_index): rng.normal(0, 0.12)
        for subject_id in subject_profiles["subject_id"]
        for session_index in range(1, sessions + 1)
    }

    stress_buckets = [
        (0.22, "relaxed"),
        (0.42, "normal"),
        (0.60, "slightly stressed"),
        (0.78, "stressed"),
        (0.92, "highly stressed"),
        (1.00, "critically stressed"),
    ]
    posture_buckets = [
        (0.25, "Good"),
        (0.45, "Mild Slouching"),
        (0.65, "Slouching"),
        (0.82, "Forward Head"),
        (1.00, "Severe Slouching"),
    ]

    generated_rows = []
    start_time = datetime.now().replace(second=0, microsecond=0)

    for row_index in range(rows):
        subject = subject_profiles.iloc[int(rng.integers(0, len(subject_profiles)))]
        subject_id = str(subject["subject_id"])
        session_number = int(rng.integers(1, sessions + 1))
        session_id = f"{subject_id}_SESSION_{session_number:02d}"
        timestamp = start_time + timedelta(minutes=row_index)

        session_load = session_loads[(subject_id, session_number)]
        time_in_session = (row_index % 90) / 89
        workload = rng.beta(2.2, 3.4)

        fatigue_level = clip(
            0.14
            + 0.38 * workload
            + 0.18 * time_in_session
            + float(subject["fatigue_bias"])
            + 0.45 * session_load
            + rng.normal(0, 0.17),
            0,
            1,
        )
        posture_strain_level = clip(
            0.12
            + 0.28 * fatigue_level
            + 0.20 * workload
            + float(subject["posture_bias"])
            + 0.25 * session_load
            + rng.normal(0, 0.20),
            0,
            1,
        )
        stress_level = clip(
            0.12
            + 0.32 * workload
            + 0.12 * fatigue_level
            + float(subject["stress_bias"])
            + 0.55 * session_load
            + rng.normal(0, 0.22),
            0,
            1,
        )

        if rng.random() < 0.08:
            stress_level = clip(stress_level + rng.uniform(0.15, 0.38), 0, 1)
        if rng.random() < 0.06:
            fatigue_level = clip(fatigue_level + rng.uniform(0.12, 0.30), 0, 1)

        blink_frequency = clip(
            19.5 - 10.5 * fatigue_level - 2.2 * stress_level + float(subject["blink_bias"])
            + rng.normal(0, 3.8),
            2,
            35,
        )
        perclos = clip(
            0.045 + 0.34 * fatigue_level + 0.08 * stress_level + rng.normal(0, 0.075),
            0,
            0.85,
        )
        mean_ibi = clip(60 / max(blink_frequency, 1) + rng.normal(0, 1.0), 0.7, 30)
        last_ibi = clip(mean_ibi + rng.normal(0, max(0.6, mean_ibi * 0.35)), 0.5, 35)
        low_blink_alert = int(blink_frequency < rng.normal(8.5, 1.2) or rng.random() < 0.025)
        high_perclos_alert = int(perclos > rng.normal(0.31, 0.045) or rng.random() < 0.025)

        good_posture_pct = clip(
            93 - 58 * posture_strain_level - 10 * fatigue_level + rng.normal(0, 12),
            0,
            100,
        )
        shoulder_asymmetry = clip(
            1.8 + 11.5 * posture_strain_level + 3.5 * fatigue_level + rng.normal(0, 3.0),
            0,
            32,
        )
        head_lateral_tilt = clip(
            1.2 + 11.0 * posture_strain_level + rng.normal(0, 3.4),
            0,
            32,
        )
        forward_head_proxy = clip(
            0.08 + 0.62 * posture_strain_level + 0.10 * fatigue_level + rng.normal(0, 0.11),
            0,
            1,
        )
        alert_rate = max(0.05, 0.35 + 2.2 * posture_strain_level + 1.2 * fatigue_level)
        alerts_in_period = int(min(rng.poisson(alert_rate), 12))

        baseline_hr = clip(float(subject["baseline_heart_rate"]) + rng.normal(0, 1.6), 50, 100)
        current_hr = clip(
            baseline_hr + 2.5 + 23 * stress_level + 6.0 * fatigue_level + rng.normal(0, 8.0),
            45,
            160,
        )
        stress_index = clip(
            0.08 + 0.72 * stress_level + 0.10 * fatigue_level + rng.normal(0, 0.13),
            0,
            1,
        )

        stress_tag = choose_noisy_bucket(stress_index, stress_buckets, rng)
        dominant_status = choose_noisy_bucket(posture_strain_level, posture_buckets, rng)

        combined_risk = clip(
            0.36 * fatigue_level
            + 0.31 * posture_strain_level
            + 0.33 * stress_level
            + rng.normal(0, 0.12),
            0,
            1,
        )
        final_state = sample_final_state(combined_risk, rng)

        if rng.random() < 0.05:
            alternatives = [state for state in STATES if state != final_state]
            final_state = str(rng.choice(alternatives))

        generated_rows.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "timestamp": timestamp,
                "blink_frequency_bpm": round(blink_frequency, 2),
                "perclos": round(perclos, 3),
                "last_inter_blink_interval": round(last_ibi, 2),
                "mean_inter_blink_interval": round(mean_ibi, 2),
                "low_blink_alert": low_blink_alert,
                "high_perclos_alert": high_perclos_alert,
                "good_posture_pct": round(good_posture_pct, 2),
                "avg_shoulder_asymmetry": round(shoulder_asymmetry, 2),
                "avg_head_lateral_tilt": round(head_lateral_tilt, 2),
                "avg_forward_head_proxy": round(forward_head_proxy, 3),
                "alerts_in_period": alerts_in_period,
                "baseline_heart_rate": round(baseline_hr, 2),
                "current_heart_rate": round(current_hr, 2),
                "stress_index": round(stress_index, 3),
                "stress_tag": stress_tag,
                "dominant_status": dominant_status,
                "final_state": final_state,
            }
        )

    df = pd.DataFrame(generated_rows, columns=OUTPUT_COLUMNS)
    return add_missing_values(df, rng)


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.subjects < 2:
        raise ValueError("--subjects must be at least 2 for grouped validation")
    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")

    rng = np.random.default_rng(args.seed)
    df = generate_dataset(args.rows, args.subjects, args.sessions, rng)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created dataset: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print("Class distribution:")
    print(df["final_state"].value_counts().to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
