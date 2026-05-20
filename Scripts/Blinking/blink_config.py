"""Configuration values for the blinking module."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "blink"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT_PATH = DATA_DIR / "blink_log.csv"
CALIBRATION_PATH = DATA_DIR / "blink_calibration.json"
TRAINING_DATA_PATH = DATA_DIR / "blink_training_data.csv"
MODEL_PATH = DATA_DIR / "blink_model.pkl"

# Eye Aspect Ratio (EAR) below this value is treated as closed eye.
EAR_CLOSED_THRESHOLD = 0.21

# A blink must last at least this many consecutive closed-eye frames.
MIN_BLINK_FRAMES = 2

# If the eyes stay closed longer than this, it is not counted as a normal blink.
MAX_BLINK_SECONDS = 1.0

# Rolling window used to compute blink frequency and PERCLOS.
METRICS_WINDOW_SECONDS = 60.0

# Percentage of time with closed eyes in the rolling window.
PERCLOS_BAD_THRESHOLD = 0.20

# Blinks per minute. A low value can indicate visual fatigue or staring.
MIN_BLINKS_PER_MINUTE = 10.0

# Alert is raised only if poor blinking metrics persist.
LOW_BLINK_ALERT_SECONDS = 120.0
HIGH_PERCLOS_ALERT_SECONDS = 30.0

# Periodic CSV logging interval.
CSV_LOG_INTERVAL_SECONDS = 60.0

# FaceMesh landmark indices commonly used for EAR.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

