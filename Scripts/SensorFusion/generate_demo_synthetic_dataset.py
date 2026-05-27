"""Generate a controlled demo dataset for ergonomics sensor fusion.

The dataset is intentionally learnable but not perfectly separable. Class
patterns are centered around the four wellbeing states, neighboring classes
share a small boundary region, and a small amount of label noise prevents
perfect validation scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["Good", "Warning", "High Risk", "Critical"]
STATE_PROBS = np.array([0.36, 0.30, 0.22, 0.12])
STATE_CENTERS = {
    "Good": 0.10,
    "Warning": 0.36,
    "High Risk": 0.66,
    "Critical": 0.91,
}
STATE_SIGMAS = {
    "Good": 0.035,
    "Warning": 0.045,
    "High Risk": 0.045,
    "Critical": 0.035,
}
OVERLAP_PROBS = {
    "Good": [("Warning", 0.035)],
    "Warning": [("Good", 0.025), ("High Risk", 0.030)],
    "High Risk": [("Warning", 0.030), ("Critical", 0.035)],
    "Critical": [("High Risk", 0.035)],
}
LABEL_NOISE_RATE = 0.015
MISSING_RATE = 0.010
SESSIONS_PER_SUBJECT = 5

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
OUTPUT_COLUMNS = [
    "subject_id",
    "session_id",
    "timestamp",
    *NUMERIC_COLUMNS,
    *CATEGORICAL_COLUMNS,
    "final_state",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a demo synthetic ergonomics sensor-fusion dataset."
    )
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows to create.")
    parser.add_argument(
        "--subjects",
        type=int,
        default=40,
        help="Number of distinct student subjects.",
    )
    parser.add_argument(
        "--out",
        default="demo_synthetic_ergonomics_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible data generation.",
    )
    return parser.parse_args()


def clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def choose_boundary_state(state: str, rng: np.random.Generator) -> str | None:
    draw = rng.random()
    cumulative = 0.0
    for neighbor, probability in OVERLAP_PROBS[state]:
        cumulative += probability
        if draw < cumulative:
            return neighbor
    return None


def sample_latent_risk(state: str, rng: np.random.Generator) -> tuple[float, str | None]:
    """Sample a hidden risk score with small overlap into neighboring states."""
    overlap_state = choose_boundary_state(state, rng)
    if overlap_state is None:
        risk = rng.normal(STATE_CENTERS[state], STATE_SIGMAS[state])
        return clip(risk, 0, 1), None

    boundary_center = 0.42 * STATE_CENTERS[state] + 0.58 * STATE_CENTERS[overlap_state]
    risk = rng.normal(boundary_center, 0.045)
    return clip(risk, 0, 1), overlap_state


def create_subject_profiles(subjects: int, rng: np.random.Generator) -> pd.DataFrame:
    baselines = rng.normal(72, 7.5, size=subjects)
    rows = []
    for index, baseline in enumerate(baselines, start=1):
        rows.append(
            {
                "subject_id": f"S{index:03d}",
                "baseline_heart_rate": clip(baseline, 52, 94),
                "fatigue_bias": rng.normal(0, 0.030),
                "posture_bias": rng.normal(0, 0.032),
                "stress_bias": rng.normal(0, 0.032),
                "blink_bias": rng.normal(0, 0.8),
                "posture_pct_bias": rng.normal(0, 2.2),
            }
        )
    return pd.DataFrame(rows)


def assign_subjects(rows: int, subjects: int, rng: np.random.Generator) -> np.ndarray:
    subject_indices = np.resize(np.arange(subjects), rows)
    rng.shuffle(subject_indices)
    return subject_indices


def noisy_bucket(
    score: float,
    buckets: list[tuple[float, str]],
    rng: np.random.Generator,
    noise_std: float,
    adjacent_flip_rate: float,
    random_flip_rate: float,
) -> str:
    noisy_score = clip(score + rng.normal(0, noise_std), 0, 1)
    labels = [label for _, label in buckets]
    selected_index = len(buckets) - 1

    for index, (upper_bound, _) in enumerate(buckets):
        if noisy_score <= upper_bound:
            selected_index = index
            break

    if rng.random() < adjacent_flip_rate:
        low = max(0, selected_index - 1)
        high = min(len(labels), selected_index + 2)
        selected_index = int(rng.integers(low, high))

    if rng.random() < random_flip_rate:
        selected_index = int(rng.integers(0, len(labels)))

    return labels[selected_index]


def noisy_neighbor_label(label: str, rng: np.random.Generator) -> str:
    index = STATES.index(label)
    neighbors = []
    if index > 0:
        neighbors.append(STATES[index - 1])
    if index < len(STATES) - 1:
        neighbors.append(STATES[index + 1])
    return str(rng.choice(neighbors))


def generate_dataset(rows: int, subjects: int, rng: np.random.Generator) -> pd.DataFrame:
    profiles = create_subject_profiles(subjects, rng)
    subject_indices = assign_subjects(rows, subjects, rng)
    base_labels = rng.choice(STATES, size=rows, p=STATE_PROBS)

    stress_buckets = [
        (0.23, "relaxed"),
        (0.42, "normal"),
        (0.59, "slightly stressed"),
        (0.75, "stressed"),
        (0.89, "highly stressed"),
        (1.00, "critically stressed"),
    ]
    posture_buckets = [
        (0.27, "Good"),
        (0.47, "Mild Slouching"),
        (0.66, "Slouching"),
        (0.82, "Forward Head"),
        (1.00, "Severe Slouching"),
    ]

    start_time = datetime(2026, 1, 12, 9, 0)
    generated_rows = []

    for row_index, base_label in enumerate(base_labels):
        subject = profiles.iloc[int(subject_indices[row_index])]
        subject_id = str(subject["subject_id"])
        session_number = int(rng.integers(1, SESSIONS_PER_SUBJECT + 1))
        session_id = f"{subject_id}_SESSION_{session_number:02d}"
        timestamp = start_time + timedelta(minutes=row_index)

        risk, overlap_state = sample_latent_risk(str(base_label), rng)
        boundary_push = 0.010 if overlap_state is not None else 0.0

        fatigue = clip(
            risk
            + float(subject["fatigue_bias"])
            + rng.normal(0, 0.040 + boundary_push),
            0,
            1,
        )
        posture_strain = clip(
            risk
            + float(subject["posture_bias"])
            + rng.normal(0, 0.043 + boundary_push),
            0,
            1,
        )
        stress = clip(
            risk
            + float(subject["stress_bias"])
            + rng.normal(0, 0.045 + boundary_push),
            0,
            1,
        )

        blink_frequency = clip(
            20.0
            - 13.0 * fatigue
            - 1.3 * stress
            + float(subject["blink_bias"])
            + rng.normal(0, 1.4),
            2.0,
            34.0,
        )
        perclos = clip(
            0.035 + 0.430 * fatigue + 0.045 * stress + rng.normal(0, 0.025),
            0.010,
            0.760,
        )
        mean_ibi = clip(60.0 / max(blink_frequency, 1.0) + rng.normal(0, 0.40), 0.6, 28)
        last_ibi = clip(
            mean_ibi + rng.normal(0, max(0.35, mean_ibi * 0.18)),
            0.5,
            32,
        )

        low_blink_alert_probability = sigmoid((8.4 - blink_frequency) / 1.8 + 0.55 * fatigue - 0.25)
        high_perclos_alert_probability = sigmoid((perclos - 0.300) / 0.045 + 0.35 * fatigue - 0.20)
        low_blink_alert = int(rng.random() < low_blink_alert_probability)
        high_perclos_alert = int(rng.random() < high_perclos_alert_probability)

        good_posture_pct = clip(
            96.0
            - 63.0 * posture_strain
            - 5.0 * fatigue
            + float(subject["posture_pct_bias"])
            + rng.normal(0, 4.5),
            0,
            100,
        )
        shoulder_asymmetry = clip(
            1.8 + 13.8 * posture_strain + 1.4 * fatigue + rng.normal(0, 1.2),
            0,
            30,
        )
        head_lateral_tilt = clip(
            1.2 + 14.2 * posture_strain + rng.normal(0, 1.3),
            0,
            30,
        )
        forward_head_proxy = clip(
            0.055 + 0.710 * posture_strain + 0.040 * fatigue + rng.normal(0, 0.035),
            0,
            1,
        )
        alerts_rate = max(0.04, 0.22 + 3.6 * posture_strain + 1.35 * fatigue)
        alerts_in_period = int(min(rng.poisson(alerts_rate), 12))

        baseline_hr = float(subject["baseline_heart_rate"])
        current_hr = clip(
            baseline_hr + 1.5 + 25.0 * stress + 3.5 * fatigue + rng.normal(0, 3.8),
            45,
            158,
        )
        stress_index = clip(
            0.045 + 0.755 * stress + 0.060 * fatigue + rng.normal(0, 0.045),
            0,
            1,
        )

        stress_tag = noisy_bucket(
            stress_index,
            stress_buckets,
            rng,
            noise_std=0.135,
            adjacent_flip_rate=0.180,
            random_flip_rate=0.025,
        )
        dominant_status = noisy_bucket(
            posture_strain,
            posture_buckets,
            rng,
            noise_std=0.130,
            adjacent_flip_rate=0.190,
            random_flip_rate=0.025,
        )

        final_state = str(base_label)
        if rng.random() < LABEL_NOISE_RATE:
            final_state = noisy_neighbor_label(final_state, rng)

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


def add_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        mask = rng.random(len(df)) < MISSING_RATE
        df.loc[mask, column] = np.nan
    return df


def validate_args(args: argparse.Namespace) -> None:
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.subjects < 2:
        raise ValueError("--subjects must be at least 2")
    if args.subjects > args.rows:
        raise ValueError("--subjects must be less than or equal to --rows")


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.random_state)
    df = generate_dataset(args.rows, args.subjects, rng)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created dataset: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print("Class distribution:")
    print(df["final_state"].value_counts().reindex(STATES).to_string())
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
