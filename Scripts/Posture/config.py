"""
config.py — Configuration for Posture Feature Extraction

Parameters for extracting posture features from pre-recorded videos.
Sliding window approach: 60-second windows with 10-second step.

Output: 9 features per window (mean, std, bad_pct for each of 3 parameters).
"""

import os

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================

MP_MIN_DETECTION_CONFIDENCE = 0.7
MP_MIN_TRACKING_CONFIDENCE = 0.7
MP_STATIC_IMAGE_MODE = False
MP_MODEL_COMPLEXITY = 1

# =============================================================================
# SLIDING WINDOW SETTINGS
# =============================================================================

# Window duration in seconds
WINDOW_SIZE = 60

# Step between consecutive windows in seconds
WINDOW_STEP = 10

# =============================================================================
# POSTURAL THRESHOLDS (from literature, used to compute bad_pct)
# =============================================================================

# Shoulder Asymmetry — angle of shoulder line vs horizontal
# Sources: Scoliometer (ScienceDirect), Hustinawaty 2025
# <=5° normal, >=7° warrants attention
SHOULDER_ASYMMETRY_THRESHOLD = 7  # degrees

# Head Lateral Tilt — ear midpoint to shoulder midpoint vs vertical
# Sources: BMJ Open 2016 (CUELA norms)
# Neutral range: -10° to 10°
HEAD_LATERAL_TILT_THRESHOLD = 10  # degrees

# Forward Head Proxy — vertical distance nose-shoulder deviation
# Relative measure, no absolute clinical threshold
# Sources: PostureCV (GitHub)
FORWARD_HEAD_PROXY_THRESHOLD = 15  # percent deviation

# Minimum landmark visibility to consider a frame valid
MIN_VISIBILITY = 0.5

# =============================================================================
# FILE PATHS
# =============================================================================

# Base project directory (Scripts/Posture/../../ = project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Video input folders
VIDEO_NORMAL_DIR = os.path.join(PROJECT_ROOT, "Video", "Normal")
VIDEO_ANOMALOUS_DIR = os.path.join(PROJECT_ROOT, "Video", "Anomalous")

# CSV output folder and files
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

CSV_NORMAL_PATH = os.path.join(RAW_DATA_DIR, "posture_normal.csv")
CSV_ANOMALOUS_PATH = os.path.join(RAW_DATA_DIR, "posture_anomalous.csv")

# =============================================================================
# CSV COLUMN NAMES
# =============================================================================

FEATURE_COLUMNS = [
    "window_start",
    "video_source",
    "p_shoulder_mean",
    "p_shoulder_std",
    "p_shoulder_bad_pct",
    "p_head_tilt_mean",
    "p_head_tilt_std",
    "p_head_tilt_bad_pct",
    "p_fhp_mean",
    "p_fhp_std",
    "p_fhp_bad_pct",
]
