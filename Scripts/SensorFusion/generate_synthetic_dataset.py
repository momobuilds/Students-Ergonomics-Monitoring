import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=10000)
parser.add_argument("--out", type=str, default="synthetic_ergonomics_dataset.csv")
args = parser.parse_args()

np.random.seed(42)

classes = ["Good", "Warning", "High Risk", "Critical"]
class_probs = [0.40, 0.30, 0.20, 0.10]

stress_tags = {
    "Good": ["relaxed", "normal"],
    "Warning": ["normal", "slightly stressed"],
    "High Risk": ["stressed", "highly stressed"],
    "Critical": ["critically stressed"]
}

dominant_statuses = {
    "Good": ["Good"],
    "Warning": ["Good", "Mild Slouching"],
    "High Risk": ["Slouching", "Forward Head"],
    "Critical": ["Severe Slouching", "Forward Head"]
}

def clip(x, low, high):
    return max(low, min(high, x))

rows = []
start_time = datetime.now().replace(second=0, microsecond=0)

for i in range(args.rows):
    final_state = np.random.choice(classes, p=class_probs)
    timestamp = start_time + timedelta(minutes=i)

    # Person-specific baseline heart rate
    baseline_hr = np.random.normal(72, 7)

    if final_state == "Good":
        blink_frequency = np.random.normal(17, 3)
        perclos = np.random.normal(0.08, 0.03)
        good_posture_pct = np.random.normal(90, 6)
        shoulder_asymmetry = np.random.normal(3, 1.5)
        head_lateral_tilt = np.random.normal(2, 1)
        forward_head_proxy = np.random.normal(0.15, 0.05)
        stress_index = np.random.normal(0.20, 0.08)
        hr_delta = np.random.normal(2, 3)
        alerts = np.random.poisson(0.3)

    elif final_state == "Warning":
        blink_frequency = np.random.normal(11, 3)
        perclos = np.random.normal(0.18, 0.06)
        good_posture_pct = np.random.normal(72, 10)
        shoulder_asymmetry = np.random.normal(6, 2)
        head_lateral_tilt = np.random.normal(5, 2)
        forward_head_proxy = np.random.normal(0.30, 0.08)
        stress_index = np.random.normal(0.45, 0.10)
        hr_delta = np.random.normal(8, 5)
        alerts = np.random.poisson(1.5)

    elif final_state == "High Risk":
        blink_frequency = np.random.normal(7, 2.5)
        perclos = np.random.normal(0.32, 0.08)
        good_posture_pct = np.random.normal(50, 12)
        shoulder_asymmetry = np.random.normal(10, 3)
        head_lateral_tilt = np.random.normal(9, 3)
        forward_head_proxy = np.random.normal(0.50, 0.10)
        stress_index = np.random.normal(0.70, 0.10)
        hr_delta = np.random.normal(16, 6)
        alerts = np.random.poisson(3)

    else:
        blink_frequency = np.random.normal(4, 2)
        perclos = np.random.normal(0.48, 0.10)
        good_posture_pct = np.random.normal(30, 10)
        shoulder_asymmetry = np.random.normal(15, 4)
        head_lateral_tilt = np.random.normal(14, 4)
        forward_head_proxy = np.random.normal(0.70, 0.12)
        stress_index = np.random.normal(0.90, 0.08)
        hr_delta = np.random.normal(25, 8)
        alerts = np.random.poisson(5)

    blink_frequency = clip(blink_frequency, 1, 35)
    perclos = clip(perclos, 0, 1)
    good_posture_pct = clip(good_posture_pct, 0, 100)
    shoulder_asymmetry = clip(shoulder_asymmetry, 0, 30)
    head_lateral_tilt = clip(head_lateral_tilt, 0, 30)
    forward_head_proxy = clip(forward_head_proxy, 0, 1)
    stress_index = clip(stress_index, 0, 1)

    current_hr = baseline_hr + hr_delta + np.random.normal(0, 2)
    current_hr = clip(current_hr, 45, 150)

    mean_ibi = 60 / blink_frequency
    last_ibi = np.random.normal(mean_ibi, mean_ibi * 0.25)
    last_ibi = clip(last_ibi, 0.5, 30)

    low_blink_alert = blink_frequency < 8
    high_perclos_alert = perclos > 0.30

    stress_tag = np.random.choice(stress_tags[final_state])
    dominant_status = np.random.choice(dominant_statuses[final_state])

    rows.append({
        "timestamp": timestamp,
        "period_start": timestamp,
        "period_end": timestamp + timedelta(minutes=1),

        "ear": round(np.random.normal(0.27, 0.04), 3),
        "blink_frequency_bpm": round(blink_frequency, 2),
        "perclos": round(perclos, 3),
        "last_inter_blink_interval": round(last_ibi, 2),
        "mean_inter_blink_interval": round(mean_ibi, 2),
        "low_blink_alert": int(low_blink_alert),
        "high_perclos_alert": int(high_perclos_alert),

        "good_posture_pct": round(good_posture_pct, 2),
        "avg_shoulder_asymmetry": round(shoulder_asymmetry, 2),
        "avg_head_lateral_tilt": round(head_lateral_tilt, 2),
        "avg_forward_head_proxy": round(forward_head_proxy, 3),
        "alerts_in_period": int(alerts),
        "dominant_status": dominant_status,

        "baseline_heart_rate": round(baseline_hr, 2),
        "current_heart_rate": round(current_hr, 2),
        "stress_index": round(stress_index, 3),
        "stress_tag": stress_tag,

        "final_state": final_state,
        "label_source": "synthetic_simulation"
    })

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)

print(f"Created dataset: {args.out}")
print(df["final_state"].value_counts())
print(df.head())
