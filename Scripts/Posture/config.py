"""
config.py — Configuration file for the Posture Monitoring System

This file contains all configurable parameters for the system:
- Postural angle thresholds (derived from literature review)
- Camera settings
- Calibration margins
- Timing settings
- File paths

All threshold values are backed by scientific literature.
See literature_review_posture_parameters.md for full references.
"""

# =============================================================================
# CAMERA SETTINGS
# =============================================================================

# Index of the laptop webcam (usually 0)
CAMERA_INDEX = 0

# Camera resolution (width, height)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Target frames per second for processing
TARGET_FPS = 15

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================

# Minimum detection confidence for MediaPipe Pose
MP_MIN_DETECTION_CONFIDENCE = 0.7

# Minimum tracking confidence for MediaPipe Pose
MP_MIN_TRACKING_CONFIDENCE = 0.7

# Whether to use the static image mode (False = video mode with tracking)
MP_STATIC_IMAGE_MODE = False

# Model complexity: 0 (lite), 1 (full), 2 (heavy)
# 1 is a good balance between speed and accuracy
MP_MODEL_COMPLEXITY = 1

# =============================================================================
# POSTURAL THRESHOLDS — FRONTAL VIEW
# =============================================================================
# These are the fixed thresholds from literature, used when calibration
# is NOT active. When calibrated, the system uses baseline + margin instead.

# Shoulder Asymmetry
# Measured as: angle of the shoulder line vs horizontal.
# Literature: <=5° is normal, >=7° warrants attention
# Sources: Scoliometer (ScienceDirect), Hustinawaty 2025
SHOULDER_ASYMMETRY_THRESHOLD = 7  # degrees — above this = bad posture

# Head Lateral Tilt
# Measured as: angle between ear-shoulder midpoints vs vertical, frontal plane.
# Literature: neutral is -10° to 10°, outside this range is problematic
# Sources: BMJ Open 2016 (CUELA norms)
HEAD_LATERAL_TILT_THRESHOLD = 10  # degrees — above this (absolute) = bad posture

# Forward Head Proxy (Frontal View)
# Measured as: percentage change in the vertical distance between nose and
# shoulder midpoint, relative to the calibrated baseline.
# No absolute clinical threshold exists — this is a relative measure.
# Sources: PostureCV (GitHub), ungermate
FORWARD_HEAD_PROXY_THRESHOLD = 15  # percent deviation from baseline

# =============================================================================
# CALIBRATION TOLERANCE MARGINS
# =============================================================================
# When calibration is active, the threshold for each parameter becomes:
#   baseline_value + margin
# These margins define how much deviation from YOUR baseline is acceptable.

CALIBRATION_MARGIN_SHOULDER_ASYMMETRY = 5   # degrees added to your baseline
CALIBRATION_MARGIN_HEAD_LATERAL_TILT = 7    # degrees added to your baseline
CALIBRATION_MARGIN_FORWARD_HEAD_PROXY = 15  # percent deviation from baseline

# =============================================================================
# TIMING SETTINGS
# =============================================================================

# Two-tier alert system:
#   - ALL metrics bad → alert after ALL_BAD_TIME_THRESHOLD seconds
#   - ANY metric bad  → alert after ANY_BAD_TIME_THRESHOLD seconds
ALL_BAD_TIME_THRESHOLD = 30   # seconds — all metrics red
ANY_BAD_TIME_THRESHOLD = 60   # seconds — at least one metric red

# How often to save data to the CSV file (in seconds).
# The project proposal specifies every 10 minutes for OpenClaw integration.
CSV_SAVE_INTERVAL = 60 # seconds (1 s)

# Duration of the initial calibration phase (in seconds).
# The user sits upright and the system records their baseline posture.
CALIBRATION_DURATION = 5  # seconds

# =============================================================================
# DATA COLLECTION SETTINGS (for ML classifier training)
# =============================================================================

# Duration of each data collection phase (good posture / bad posture)
TRAINING_COLLECTION_DURATION = 60  # seconds per class

# Window size for feature extraction (in frames)
# At 15 FPS, 15 frames = 1 second window
FEATURE_WINDOW_SIZE = 15

# =============================================================================
# FILE PATHS
# =============================================================================

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "posture")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(DATA_DIR, "posture_log.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "posture_model.pkl")
CALIBRATION_PATH = os.path.join(DATA_DIR, "calibration_baseline.json")

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Colors for OpenCV overlay (BGR format)
COLOR_GOOD = (0, 200, 0)       # Green
COLOR_BAD = (0, 0, 200)        # Red
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_TEXT = (255, 255, 255)    # White
COLOR_LANDMARK = (255, 0, 0)   # Blue

# Font scale for OpenCV text
FONT_SCALE = 0.6

# Thickness for OpenCV text and lines
LINE_THICKNESS = 2
