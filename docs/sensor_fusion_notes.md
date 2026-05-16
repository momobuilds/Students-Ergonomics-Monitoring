# Sensor Fusion Notes

## Why Fake Data Is Used Now

The project does not have complete live sensor streams yet, so the current prototype uses simulated readings. This lets the team validate the data format, sensor fusion logic, dashboard layout, and LLM advisor interface before the hardware and computer vision modules are fully integrated.

At the current stage, simulated data is used to validate the sensor fusion module and the LLM advisor interface. When the real sensing modules are ready, the simulated CSV will be replaced by live data from the heart-rate sensor, webcam blink detector, posture detector, and optional temperature patch.

## Where Real Sensor Data Will Connect Later

The fake CSV currently provides these fields:

- `heart_rate_bpm`
- `baseline_heart_rate_bpm`
- `blink_rate_per_min`
- `posture_status`
- `slouching_time_minutes`
- `temperature_c`
- `data_source`

Later, teammates can replace `data/fake_sensor_data.csv` or the dashboard data loader with live values from the heart-rate wearable, webcam blink detector, posture detector, and optional temperature patch. The fusion module should keep receiving the same column names so the algorithm and dashboard do not need major changes.

## Feature-Level Fusion

Feature-level fusion converts each sensor input into a normalized risk score:

- Posture risk from posture status and slouching duration.
- Blink risk from blinks per minute.
- Heart risk from the difference between current and baseline heart rate.
- Temperature risk from the difference between current and baseline temperature.

Each feature score is `0.0`, `0.5`, or `1.0`, representing normal, warning, or high-risk sensor evidence.

## Decision-Level Fusion

Decision-level fusion combines the individual feature scores into one final risk score and status. This produces a single system decision that the dashboard can display and the advisor can explain.

## Weighted Risk-Score Formula

The final score is calculated as:

```text
final_risk_score =
0.35 * posture_score
+ 0.30 * blink_score
+ 0.25 * heart_score
+ 0.10 * temperature_score
```

The final status is:

- `Normal` when `final_risk_score < 0.30`
- `Warning` when `0.30 <= final_risk_score < 0.60`
- `High Risk` when `final_risk_score >= 0.60`

## Role of the Local LLM

The local LLM advisor is an explanation layer. It receives the already-calculated fusion result, turns the reasons into student-friendly language, and recommends practical ergonomic actions such as sitting upright, relaxing shoulders, blinking, or taking a short break.

## Why the LLM Comes After Sensor Fusion

The LLM must not replace the deterministic sensor fusion algorithm. Sensor fusion calculates the risk score and classification using fixed, inspectable rules. The LLM comes afterward so it can explain the decision without changing the decision. This makes the prototype easier to validate and safer to extend when real sensors are connected.
