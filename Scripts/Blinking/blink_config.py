"""
blink_config.py — Configuration for Blinking Feature Extraction

Parameters for extracting blinking features from pre-recorded videos.
Sliding window approach: 60-second windows with 10-second step.

Output: 7 features per window.
"""

import os

# =============================================================================
# MEDIAPIPE FACEMESH — landmark indices for EAR
# =============================================================================

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# =============================================================================
# SLIDING WINDOW SETTINGS
# =============================================================================

WINDOW_SIZE = 60   # seconds
WINDOW_STEP = 10   # seconds

# =============================================================================
# EAR / BLINK DETECTION THRESHOLDS
# =============================================================================

# Eye Aspect Ratio below this → eye considered closed
EAR_CLOSED_THRESHOLD = 0.21

# A blink must last at least this many consecutive closed-eye frames
MIN_BLINK_FRAMES = 2

# If eyes stay closed longer than this, it is NOT a normal blink
MAX_BLINK_SECONDS = 1.0

# =============================================================================
# PERCLOS THRESHOLD (used for bad_pct equivalent)
# =============================================================================

# PERCLOS above this fraction → problematic (drowsiness/fatigue indicator)
PERCLOS_BAD_THRESHOLD = 0.20

# =============================================================================
# EYE OCCLUSION DETECTION (sunglasses / hand over the eyes)
# =============================================================================

# FaceMesh keeps inferring eye landmarks even when the eyes are covered, so the
# EAR alone cannot tell that the eyes are occluded. Instead we look at the
# actual pixels of the eye region: a real eye (sclera, iris, lashes) has high
# local contrast, while a dark lens or a hand is comparatively uniform.
#
# If the grayscale std-dev of the eye region is below this value, the eyes are
# considered NOT really visible and that frame is dropped from the blink signal
# (so the blink features get median-imputed instead of being trusted).
# Camera/lighting dependent — tune by reading the live "Eye signal" value.
EYE_VISIBILITY_MIN_STD = 18.0

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

VIDEO_NORMAL_DIR = os.path.join(PROJECT_ROOT, "Video", "Normal")
VIDEO_SLEEPY_DIR  = os.path.join(PROJECT_ROOT, "Video", "Sleepy")
VIDEO_STRESS_DIR  = os.path.join(PROJECT_ROOT, "Video", "Stress")

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

CSV_NORMAL_PATH = os.path.join(RAW_DATA_DIR, "blinking_normal.csv")
CSV_SLEEPY_PATH  = os.path.join(RAW_DATA_DIR, "blinking_sleepy.csv")
CSV_STRESS_PATH  = os.path.join(RAW_DATA_DIR, "blinking_stress.csv")

# =============================================================================
# CSV COLUMN NAMES
# =============================================================================

FEATURE_COLUMNS = [
    "window_start",
    "video_source",
    "b_ear_mean",
    "b_ear_std",
    "b_perclos",
    "b_blink_count",
    "b_blink_freq",
    "b_ibi_mean",
    "b_ibi_std",
    "label",
]
