"""
pipeline.py - Bridge between the Streamlit GUI and the project's Scripts.

This module is the single place that wires the GUI to the existing analysis
code under ``Scripts/``. It does three things:

  1. Puts the feature folders on ``sys.path`` exactly like Scripts/App does, so
     their modules (``config``, ``blink_config``, ``posture_analyzer``, ...) can
     be imported by their bare names.
  2. Loads ``Scripts/App/main.realtime.py`` as a module (its filename contains a
     dot so it is not importable directly) and re-exports the pieces the GUI
     needs: the model catalog, the ``RealtimeModel`` wrapper, the wearable
     ``HeartRateMonitor`` and the feature-assembly helpers. Nothing is
     duplicated -- the GUI uses the very same code paths as the CLI scripts.
  3. Exposes small helpers to lazily import the per-frame analyzers and to push
     the GUI parameter values (window length, posture/blink thresholds) into the
     shared config modules at run time.

Folder map (everything is resolved relative to this file):

    Scripts/GUI          <- this file lives here
    Scripts/App          -> main.realtime.py  (model menu + realtime engine)
    Scripts/Posture      -> config.py, posture_analyzer.py, feature_extractor.py
    Scripts/Blinking     -> blink_config.py, blink_analyzer.py, blink_feature_extractor.py
    Scripts/HeartRate    -> cms50d.py (optional CMS50D wearable)
    Scripts/ML           -> the trained model definitions
    data/supervised      -> the labelled training windows
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

GUI_DIR = Path(__file__).resolve().parent      # Scripts/GUI
SCRIPTS_DIR = GUI_DIR.parent                    # Scripts
PROJECT_ROOT = SCRIPTS_DIR.parent               # project root

APP_DIR = SCRIPTS_DIR / "App"
POSTURE_DIR = SCRIPTS_DIR / "Posture"
BLINKING_DIR = SCRIPTS_DIR / "Blinking"
HEARTRATE_DIR = SCRIPTS_DIR / "HeartRate"
ML_DIR = SCRIPTS_DIR / "ML"

SUPERVISED_DIR = PROJECT_ROOT / "data" / "supervised"
TEST_VIDEO_DIR = PROJECT_ROOT / "Test Video"
VIDEO_DIR = PROJECT_ROOT / "Video"

# Same import wiring as Scripts/App: the sub-modules import their config as a
# bare name, so their folders must be importable directly. Order matters --
# Posture must come before others so ``import config`` resolves to the posture
# config (Blinking uses the explicit ``blink_config`` alias, so no clash).
for _module_dir in (APP_DIR, POSTURE_DIR, BLINKING_DIR, HEARTRATE_DIR, ML_DIR):
    _path = str(_module_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# =============================================================================
# LOAD THE REAL-TIME ENGINE (Scripts/App/main.realtime.py)
# =============================================================================

def _load_module(name: str, path: Path):
    """Import a .py file by path (needed because 'main.realtime.py' has a dot)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# This pulls in: model_catalog, RealtimeModel, HeartRateMonitor,
# assemble_feature_row, landmarks_from_results, FEATURE_COLUMNS,
# POSTURE_FEATURES, BLINK_FEATURES, TASK_FILES, RANDOM_STATE, ...
# No heavy work runs at import time (cv2/mediapipe/ML are imported lazily).
engine = _load_module("gui_realtime_engine", APP_DIR / "main.realtime.py")

# Re-export the names the GUI uses, so app.py never reaches into Scripts itself.
RealtimeModel = engine.RealtimeModel
HeartRateMonitor = engine.HeartRateMonitor
model_catalog = engine.model_catalog
assemble_feature_row = engine.assemble_feature_row
landmarks_from_results = engine.landmarks_from_results
FEATURE_COLUMNS = engine.FEATURE_COLUMNS
POSTURE_FEATURES = engine.POSTURE_FEATURES
BLINK_FEATURES = engine.BLINK_FEATURES


# =============================================================================
# MODEL HELPERS
# =============================================================================

def available_models(task: str, learning: str):
    """Return the list of model names available for a task/learning combo."""
    return list(model_catalog(task, learning).keys())


def build_and_train_model(task: str, learning: str, name: str) -> "RealtimeModel":
    """Instantiate the chosen estimator and fit it on data/supervised.

    Returns a ready-to-use RealtimeModel (same object the CLI scripts use).
    """
    catalog = model_catalog(task, learning)
    if name not in catalog:
        raise KeyError(f"Model '{name}' not available for {task}/{learning}. "
                       f"Choices: {list(catalog)}")
    kind, estimator = catalog[name]
    model = RealtimeModel(task, learning, name, kind, estimator)
    model.fit()
    return model


# --- Hierarchical (binary -> subtype) -----------------------------------------

# Register a "subtype" supervised task (sleepy vs stress) on the engine so the
# existing load_dataset / RealtimeModel machinery can train the second stage
# exactly like the CLI hierarchical pipeline (Scripts/ML).
engine.TASK_FILES.setdefault(
    "subtype", [("sleepy.csv", "sleepy"), ("stress.csv", "stress")]
)


class HierarchicalModel:
    """Two-stage classifier with the same interface as RealtimeModel.

    Stage 1 (binary):  normal vs abnormal.
    Stage 2 (subtype): if abnormal, sleepy vs stress.

    So a single ``predict_row`` returns normal / sleepy / stress -- the binary
    decision refined into its subtype, like Scripts/ML's hierarchical task.
    """

    def __init__(self, binary_model, subtype_model, name):
        self.binary = binary_model
        self.subtype = subtype_model
        self.name = name
        self.task = "hierarchical"
        self.learning = "supervised"
        self.classes = ["normal", "sleepy", "stress"]

    def describe(self):
        return f"{self.name} (gerarchico: binario -> sottotipo sleepy/stress)"

    def predict_row(self, feature_row):
        result = self.binary.predict_row(feature_row)
        if result["label"] in ("abnormal", "anomalous"):
            return self.subtype.predict_row(feature_row)
        return result


def build_hierarchical_model(name: str) -> "HierarchicalModel":
    """Train both stages (binary + sleepy/stress) of the hierarchical model."""
    binary_model = build_and_train_model("binary", "supervised", name)

    kind, estimator = model_catalog("subtype", "supervised")[name]
    subtype_model = RealtimeModel("subtype", "supervised", name, kind, estimator)
    subtype_model.fit()

    return HierarchicalModel(binary_model, subtype_model, name)


# =============================================================================
# LAZY ANALYZER IMPORTS (cv2 / mediapipe are heavy -- only load on demand)
# =============================================================================

def load_analyzers():
    """Import and return everything needed to process frames.

    Returns a dict with the config modules, the per-frame analyzers, the
    sliding-window feature extractors and the shared overlay drawing module.
    """
    import config as posture_config
    import blink_config
    from posture_analyzer import PostureAnalyzer
    from feature_extractor import FeatureExtractor
    from blink_analyzer import BlinkAnalyzer
    from blink_feature_extractor import BlinkFeatureExtractor
    import overlay_drawing as overlay

    return {
        "posture_config": posture_config,
        "blink_config": blink_config,
        "PostureAnalyzer": PostureAnalyzer,
        "FeatureExtractor": FeatureExtractor,
        "BlinkAnalyzer": BlinkAnalyzer,
        "BlinkFeatureExtractor": BlinkFeatureExtractor,
        "overlay": overlay,
    }


def default_thresholds() -> dict:
    """Read the default thresholds straight from the project's config files.

    Posture defaults come from Scripts/Posture/config.py and the blink defaults
    from Scripts/Blinking/blink_config.py, so the sidebar shows the exact
    literature-based values used by the offline pipeline (not hard-coded copies).
    """
    import config as posture_config
    import blink_config

    return {
        "window_size": int(posture_config.WINDOW_SIZE),
        "shoulder_thr": float(posture_config.SHOULDER_ASYMMETRY_THRESHOLD),
        "head_tilt_thr": float(posture_config.HEAD_LATERAL_TILT_THRESHOLD),
        "fhp_thr": float(posture_config.FORWARD_HEAD_PROXY_THRESHOLD),
        "min_visibility": float(posture_config.MIN_VISIBILITY),
        "ear_thr": float(blink_config.EAR_CLOSED_THRESHOLD),
        "perclos_thr": float(blink_config.PERCLOS_BAD_THRESHOLD),
    }


def apply_settings(settings: dict):
    """Push the GUI parameter values into the shared config modules.

    These are read by the feature extractors and the overlay, so changing them
    in the sidebar changes the live analysis without touching any script file.
    """
    import config as posture_config
    import blink_config

    win = int(settings.get("window_size", posture_config.WINDOW_SIZE))
    posture_config.WINDOW_SIZE = win
    blink_config.WINDOW_SIZE = win
    # A single tumbling window per minute -> step == size keeps one window.
    posture_config.WINDOW_STEP = win
    blink_config.WINDOW_STEP = win

    posture_config.SHOULDER_ASYMMETRY_THRESHOLD = settings["shoulder_thr"]
    posture_config.HEAD_LATERAL_TILT_THRESHOLD = settings["head_tilt_thr"]
    posture_config.FORWARD_HEAD_PROXY_THRESHOLD = settings["fhp_thr"]
    posture_config.MIN_VISIBILITY = settings["min_visibility"]

    blink_config.EAR_CLOSED_THRESHOLD = settings["ear_thr"]
    blink_config.PERCLOS_BAD_THRESHOLD = settings["perclos_thr"]
