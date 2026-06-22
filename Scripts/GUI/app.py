"""
app.py - Streamlit GUI for the Students Ergonomics & Wellbeing Monitor.

A five-page wizard that drives the project's existing analysis code through
``pipeline.py``. It guides the user from source selection and model training to
acquisition review and the final save decision.

Two sources are supported and share the same pipeline (Posture + Blinking +
optional Heart -> 18 features -> the chosen model):

  * Real-time : the webcam (and, optionally, the CMS50D wearable).
  * Video     : an uploaded recording (with an optional Heart CSV, one row
                per minute, aligned by position -- like Scripts/App).

Run it from the project root with:

    streamlit run ./scripts/GUI/app.py

(or use the helper script ./Scripts/GUI/run.sh).
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pipeline
from collections import deque


WINDOW_SECONDS = 10
PROCESSED_OUTPUT_DIR = pipeline.PROJECT_ROOT / "data" / "processed"


# =============================================================================
# PAGE CONFIG + GREEN THEME POLISH
# =============================================================================

st.set_page_config(
    page_title="SWEM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
      .stApp, [data-testid="stAppViewContainer"] { background: #ffffff; }
      .block-container { max-width: 1280px; padding-top: 4rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #1b5e20; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: #a5d6a7 !important;
          border-radius: 18px;
          background: #ffffff;
          box-shadow: 0 6px 24px rgba(27, 94, 32, 0.08);
      }
      div.stButton > button {
          border-radius: 10px;
          border: 1px solid #2e7d32;
          font-weight: 700;
      }
      div.stButton > button[kind="primary"] {
          background: #2e7d32;
          color: #ffffff;
      }
      div[data-baseweb="select"] { font-weight: 700; }
      .swem-step {
          color: #2e7d32; font-size: .78rem; font-weight: 800;
          letter-spacing: .12em; text-transform: uppercase;
      }
      .swem-subtitle {
          color: #1b5e20; font-size: 1.35rem; font-weight: 650;
          text-align: center; margin-top: .5rem;
      }

      .state-banner {
          padding: 1.0rem 1.4rem; border-radius: 14px; font-weight: 700;
          font-size: 1.6rem; text-align: center; color: white;
          letter-spacing: 0.04em; margin-bottom: 0.6rem;
      }
      .state-normal   { background: #2e7d32; }
      .state-sleepy   { background: #f39c12; }
      .state-stress   { background: #c0392b; }
      .state-abnormal { background: #c0392b; }
      .state-waiting  { background: #7f8c8d; }

      .metric-good { color: #2e7d32; font-weight: 600; }
      .metric-dim  { color: #95a5a6; font-weight: 600; }

    </style>
    """,
    unsafe_allow_html=True,
)
_ecg_buffer: deque = deque(maxlen=250 * 6)


# =============================================================================
# SESSION STATE
# =============================================================================

def _init_state():
    defaults = {
        "trained_model": None,
        "model_signature": None,   # task/model/options signature of the trained model
        "loaded_signature": None,  # signature of the loaded (not necessarily trained) model
        "model_info": None,        # metrics + training metadata for the active model
        "rt_running": False,       # webcam loop active?
        "history": [],             # list of per-window prediction dicts
        "window_metrics": [],      # list of full per-window feature dicts
        "processed_records": [],   # rows prepared for data/processed export
        "session_id": None,
        "session_video_source": None,
        "processed_output_path": None,
        "last_saved_path": None,
        "save_processed_data": True,
        "processed_window_seconds": 60,
        "page": 1,
        "source": None,
        "wearable_port": "",
        "use_wearable": False,
        "app_settings": None,
        "acquisition_complete": False,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "uploaded_heart_data": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename_part(value):
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe.strip("_") or "session"


def _remove_uploaded_video():
    path = st.session_state.get("uploaded_video_path")
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    st.session_state.uploaded_video_path = None
    st.session_state.uploaded_video_name = None
    st.session_state.uploaded_heart_data = []


def _persist_video_upload(video_file, heart_file):
    """Keep uploaded inputs available after uploader widgets leave the page."""
    _remove_uploaded_video()
    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.getvalue())
        st.session_state.uploaded_video_path = tmp.name
    st.session_state.uploaded_video_name = video_file.name
    st.session_state.uploaded_heart_data = load_heart_csv(heart_file)


def _resolve_output_dir(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        path = pipeline.PROJECT_ROOT / path
    return path


def _unique_processed_path(output_dir, session_id=None):
    output_dir = _resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_id = _safe_filename_part(session_id or _now_stamp())
    candidate = output_dir / f"processed_{base_id}.csv"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"processed_{base_id}_{counter}.csv"
        counter += 1
    return candidate


def _majority_or_latest(series):
    """Return the most common value, using the latest value to break ties."""
    clean = series.dropna()
    clean = clean[clean.astype(str) != ""]
    if clean.empty:
        return np.nan
    counts = clean.value_counts()
    tied = set(counts[counts == counts.max()].index)
    for value in reversed(clean.tolist()):
        if value in tied:
            return value
    return clean.iloc[-1]


def _aggregate_to_minutes(df, window_seconds):
    """Aggregate shorter analyzed windows into one-minute records.

    The live GUI currently emits one row per 60-second window. This helper is
    kept for future/offline inputs that may already be represented as 10-second
    windows: six consecutive rows are folded into one minute, continuous values
    are averaged, blink counts are summed, labels use majority vote with latest
    as a tie-breaker, and probabilities are averaged.
    """
    if df.empty or window_seconds <= 0 or window_seconds >= 60:
        return df

    rows_per_minute = int(round(60 / window_seconds))
    if rows_per_minute <= 1:
        return df

    work = df.copy()
    work["_minute_group"] = np.arange(len(work)) // rows_per_minute

    first_columns = {
        "session_id",
        "video_source",
        "participant",
        "model_type",
        "selected_model",
        "augmentation_used",
        "best_model_auto_selected",
        "true_label",
    }
    latest_columns = {"timestamp", "window_end"}
    first_window_columns = {"window_start"}
    count_columns = {"b_blink_count"}
    probability_columns = {
        "prediction_confidence",
        "probability_normal",
        "probability_abnormal",
        "probability_sleepy",
        "probability_stress",
    }
    label_columns = {
        "binary_prediction",
        "subtype_prediction",
        "final_prediction",
        "correct_prediction",
    }

    aggregations = {}
    for column in work.columns:
        if column == "_minute_group":
            continue
        if column in count_columns:
            aggregations[column] = "sum"
        elif column in probability_columns:
            aggregations[column] = "mean"
        elif column in pipeline.FEATURE_COLUMNS:
            aggregations[column] = "mean"
        elif column in first_columns:
            aggregations[column] = "first"
        elif column in first_window_columns:
            aggregations[column] = "first"
        elif column in latest_columns:
            aggregations[column] = "last"
        elif column in label_columns:
            aggregations[column] = _majority_or_latest
        else:
            aggregations[column] = "last"

    aggregated = work.groupby("_minute_group", sort=True).agg(aggregations).reset_index(drop=True)
    if {"true_label", "final_prediction"}.issubset(aggregated.columns):
        has_truth = aggregated["true_label"].notna() & (aggregated["true_label"].astype(str) != "")
        if has_truth.any():
            aggregated["correct_prediction"] = aggregated["correct_prediction"].astype(object)
            aggregated.loc[has_truth, "correct_prediction"] = (
                aggregated.loc[has_truth, "final_prediction"].astype(str)
                == aggregated.loc[has_truth, "true_label"].astype(str)
            )
    return aggregated


def save_processed_analysis(
    analyzed_df,
    output_dir="data/processed",
    session_id=None,
    aggregate_to_one_minute=True,
    window_seconds=WINDOW_SECONDS,
    output_path=None,
):
    """
    Save analyzed model outputs to data/processed.
    If aggregate_to_one_minute=True and input rows are shorter windows,
    aggregate rows into 1-minute records before saving.
    """
    output_dir = _resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = analyzed_df.copy()
    if aggregate_to_one_minute:
        processed = _aggregate_to_minutes(processed, window_seconds)

    path = Path(output_path) if output_path is not None else _unique_processed_path(
        output_dir, session_id=session_id
    )
    if not path.is_absolute():
        path = pipeline.PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)
    return path


def _begin_analysis_session(source, video_source=None):
    session_id = f"{source}_{_now_stamp()}"
    st.session_state.session_id = session_id
    st.session_state.session_video_source = video_source or source
    st.session_state.processed_records = []
    st.session_state.history = []
    st.session_state.window_metrics = []
    st.session_state.charts = None
    st.session_state.charts_show_hr = False
    st.session_state.last_saved_path = None
    st.session_state.processed_output_path = str(
        _unique_processed_path(PROCESSED_OUTPUT_DIR, session_id=session_id)
    )


def _relative_project_path(path):
    try:
        return str(Path(path).resolve().relative_to(pipeline.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _save_processed_records(final=False):
    if not st.session_state.get("save_processed_data", True):
        return None
    records = st.session_state.get("processed_records", [])
    if not records:
        return None

    output_path = st.session_state.get("processed_output_path")
    if output_path is None:
        output_path = str(
            _unique_processed_path(
                PROCESSED_OUTPUT_DIR,
                session_id=st.session_state.get("session_id"),
            )
        )
        st.session_state.processed_output_path = output_path

    saved_path = save_processed_analysis(
        pd.DataFrame(records),
        output_dir=PROCESSED_OUTPUT_DIR,
        session_id=st.session_state.get("session_id"),
        aggregate_to_one_minute=True,
        window_seconds=int(st.session_state.get("processed_window_seconds", 60)),
        output_path=output_path,
    )
    st.session_state.last_saved_path = str(saved_path)
    if final:
        st.session_state.processed_output_path = str(saved_path)
    return saved_path


def _start_realtime():
    _begin_analysis_session("realtime", video_source="webcam")
    st.session_state.acquisition_complete = False
    st.session_state.rt_running = True


def _stop_realtime():
    st.session_state.rt_running = False
    st.session_state.acquisition_complete = True


def _clear_data():
    """Drop the current acquisition's results (history, metrics, live charts)."""
    st.session_state.history = []
    st.session_state.window_metrics = []
    st.session_state.processed_records = []
    st.session_state.charts = None
    st.session_state.charts_show_hr = False
    st.session_state.last_saved_path = None
    st.session_state.processed_output_path = None
    st.session_state.session_id = None
    st.session_state.session_video_source = None
    st.session_state.acquisition_complete = False


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def model_configuration_controls() -> dict:
    """Render the progressive model choices and return the resolved settings."""
    task = st.radio(
        "Model output",
        ["binary", "3class"],
        format_func=lambda value: {
            "binary": "Binary: normal / anomalous",
            "3class": "3 classes: normal / sleepy / stress",
        }[value],
        horizontal=True,
        key="model_task",
    )
    hierarchical = False
    if task == "binary":
        hierarchical = st.toggle(
            "Show the anomalous subtype (sleepy / stress)",
            key="model_hierarchical",
        )
    else:
        st.caption("Subtype prediction is available only with the binary supervised model.")

    learning = st.radio(
        "Learning approach",
        ["supervised", "unsupervised"],
        format_func=str.title,
        horizontal=True,
        key="model_learning",
    )
    if learning == "unsupervised":
        hierarchical = False

    use_augmentation = st.toggle(
        "Perform data augmentation",
        key="model_augmentation",
    )
    options = pipeline.model_menu_options(task, learning)
    model_choice = st.selectbox(
        "Select model",
        options,
        format_func=lambda value: "BEST MODEL - recommended" if value == "Best Model" else value,
        help="Best Model selects the strongest saved confusion-matrix result.",
        key="model_menu_choice",
    )
    selection = pipeline.resolve_model_selection(
        task,
        learning,
        model_choice,
        hierarchical=hierarchical,
        use_augmentation=use_augmentation,
    )
    selected_name = selection.get("model_name") or model_choice
    binary_name = selection.get("binary_model") or selected_name
    subtype_name = selection.get("subtype_model") or selected_name

    factor = 0
    noise = 0.0
    if use_augmentation:
        try:
            factor = int(selection.get("augmentation_factor") or 0)
            noise = float(selection.get("noise_level") or 0.0)
        except (TypeError, ValueError):
            factor, noise = 0, 0.0
        if factor <= 0 or noise <= 0:
            st.caption("No saved augmentation setting was found; training will use original data.")

    signature = (
        task, learning, selected_name, binary_name, subtype_name, hierarchical,
        model_choice == "Best Model", use_augmentation, factor, noise,
    )
    is_trained = (
        st.session_state.trained_model is not None
        and st.session_state.model_signature == signature
    )
    settings = {
        "task": task,
        "learning": learning,
        "model_name": selected_name,
        "manual_model_name": None if model_choice == "Best Model" else model_choice,
        "use_best_model": model_choice == "Best Model",
        "use_augmentation": use_augmentation,
        "augmentation_factor": factor,
        "noise_level": noise,
        "save_processed_data": False,
        "hierarchical": hierarchical,
        "is_trained": is_trained,
        "is_loaded": True,
        "source": st.session_state.source,
        "camera_index": 0,
        "use_wearable": st.session_state.use_wearable,
        "wearable_port": st.session_state.wearable_port,
    }

    if st.button("Train the model", type="primary", use_container_width=True, disabled=is_trained):
        with st.spinner(f"Training '{selected_name}'..."):
            try:
                if hierarchical:
                    model = pipeline.build_hierarchical_model(
                        binary_name,
                        subtype_name,
                        use_augmentation=use_augmentation,
                        augmentation_factor=factor,
                        noise_level=noise,
                    )
                else:
                    model = pipeline.build_and_train_model(
                        task,
                        learning,
                        selected_name,
                        use_augmentation=use_augmentation,
                        augmentation_factor=factor,
                        noise_level=noise,
                    )
                st.session_state.trained_model = model
                st.session_state.model_signature = signature
                st.session_state.model_info = {
                    "task": "hierarchical" if hierarchical else task,
                    "learning": learning,
                    "model_name": selected_name,
                    "binary_model": binary_name,
                    "subtype_model": subtype_name if hierarchical else None,
                    "hierarchical": hierarchical,
                    "use_best_model": model_choice == "Best Model",
                    "use_augmentation": use_augmentation,
                    "augmentation_factor": factor,
                    "noise_level": noise,
                    "metrics": selection.get("metrics", {}),
                    "results_source": selection.get("results_source", ""),
                    "augmentation_name": selection.get("augmentation_name", ""),
                    "confusion_matrix_path": selection.get("confusion_matrix_path"),
                    "training_metadata": getattr(model, "training_metadata", {}),
                }
                settings["is_trained"] = True
                st.session_state.app_settings = settings
            except Exception as exc:  # noqa: BLE001
                st.session_state.trained_model = None
                st.session_state.model_signature = None
                st.error(f"Training error: {exc}")

    if is_trained:
        st.session_state.app_settings = settings
    return settings


# =============================================================================
# RENDERING HELPERS (central area)
# =============================================================================

_STATE_CLASS = {
    "normal": "state-normal",
    "sleepy": "state-sleepy",
    "stress": "state-stress",
    "abnormal": "state-abnormal",
    "anomalous": "state-abnormal",
}


def pretty_label(label):
    """Display name for a predicted label (the binary 'abnormal' -> 'anomalous')."""
    return "anomalous" if label == "abnormal" else label


def display_label(label):
    """Banner text for a label: 'normal' is always shown as 'Focused'."""
    if label == "normal":
        return "Focused"
    return pretty_label(label)


# Posture is the only mandatory sensor: when the body is not tracked the window
# is dropped, because median-imputing every body feature collapses the model to
# chance. Blink and heart are optional and median-imputed when missing.
SKIP_REASON_NO_POSTURE = "Posture not detected (body not in frame)"


def _is_missing(value):
    """True when a sensor value is absent (None or NaN)."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _imputation_note(blink_rows, hr_mean, rmssd):
    """Note for a kept window whose optional sensors were median-imputed."""
    imputed = []
    if not blink_rows:
        imputed.append("blink")
    if _is_missing(hr_mean) or _is_missing(rmssd):
        imputed.append("heart")
    return f"{', '.join(imputed)} imputed (median)" if imputed else ""


def _probability_from(proba, *labels):
    for label in labels:
        if label in proba:
            return float(proba[label])
    return np.nan


def _prediction_confidence(prediction):
    proba = prediction.get("proba") or {}
    if proba:
        return float(max(proba.values()))
    for key in ("subtype_proba", "binary_proba"):
        nested = prediction.get(key) or {}
        if nested:
            return float(max(nested.values()))
    return np.nan


def _processed_prediction_fields(prediction, settings):
    label = pretty_label(prediction.get("label", ""))
    binary_proba = prediction.get("binary_proba") or {}
    subtype_proba = prediction.get("subtype_proba") or {}
    proba = prediction.get("proba") or {}

    binary_prediction = prediction.get("binary_prediction")
    if binary_prediction is not None:
        binary_prediction = pretty_label(binary_prediction)
    elif settings["task"] == "binary":
        binary_prediction = label
    else:
        binary_prediction = ""

    subtype_prediction = prediction.get("subtype_prediction")
    if subtype_prediction is None and label in {"sleepy", "stress"}:
        subtype_prediction = label
    if subtype_prediction is None:
        subtype_prediction = ""

    return {
        "binary_prediction": binary_prediction,
        "subtype_prediction": subtype_prediction,
        "final_prediction": label,
        "prediction_confidence": _prediction_confidence(prediction),
        "probability_normal": _probability_from(binary_proba or proba, "normal"),
        "probability_abnormal": _probability_from(
            binary_proba or proba, "abnormal", "anomalous"
        ),
        "probability_sleepy": _probability_from(subtype_proba or proba, "sleepy"),
        "probability_stress": _probability_from(subtype_proba or proba, "stress"),
    }


def _append_processed_record(
    settings,
    prediction,
    feature_row,
    window_start,
    window_end,
    video_source=None,
    true_label=None,
):
    model_info = st.session_state.get("model_info") or {}
    final_prediction = pretty_label(prediction.get("label", ""))
    normalized_true_label = (
        pretty_label(str(true_label))
        if true_label is not None and str(true_label) != ""
        else None
    )
    correct_prediction = (
        final_prediction == normalized_true_label
        if normalized_true_label is not None
        else np.nan
    )

    record = {
        "session_id": st.session_state.get("session_id"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "window_start": window_start,
        "window_end": window_end,
        "video_source": video_source or st.session_state.get("session_video_source", ""),
        "participant": video_source or st.session_state.get("session_video_source", ""),
        "model_type": (
            "hierarchical supervised"
            if settings.get("hierarchical")
            else f"{settings.get('task')} {settings.get('learning')}"
        ),
        "selected_model": model_info.get("model_name", settings.get("model_name", "")),
        "augmentation_used": bool(settings.get("use_augmentation")),
        "best_model_auto_selected": bool(settings.get("use_best_model")),
    }
    for feature in pipeline.FEATURE_COLUMNS:
        record[feature] = feature_row.get(feature, np.nan)
    record.update(_processed_prediction_fields(prediction, settings))
    record["true_label"] = normalized_true_label or ""
    record["correct_prediction"] = correct_prediction

    st.session_state.processed_records.append(record)


def render_save_status():
    saved_path = st.session_state.get("last_saved_path")
    if saved_path and st.session_state.get("save_processed_data", True):
        st.success(
            "Processed analysis saved to "
            f"{_relative_project_path(saved_path)}"
        )


def _context_chip(col, label, value):
    """Compact label/value pair for the evaluation-context row.

    Smaller and word-wrapping than st.metric, so long names like
    "Gaussian Mixture" or "unsupervised" stay readable on two lines.
    """
    col.markdown(
        f'<div style="line-height:1.25;">'
        f'<div style="font-size:0.72rem;color:#5f6f5f;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.03em;">{label}</div>'
        f'<div style="font-size:1.0rem;font-weight:700;color:#1b5e20;'
        f'white-space:normal;word-break:break-word;">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_model_status(settings):
    """Show the active model, training options and saved evaluation metrics."""
    if not (settings["is_loaded"] or settings["is_trained"]):
        return
    info = st.session_state.get("model_info") or {}
    metrics = info.get("metrics") or {}

    with st.expander("Active model and evaluation context", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        _context_chip(c1, "Task", str(info.get("task", settings["task"])))
        _context_chip(c2, "Learning", str(info.get("learning", settings["learning"])))
        _context_chip(c3, "Model", str(info.get("model_name", settings["model_name"])))
        _context_chip(c4, "Auto-best", "on" if info.get("use_best_model") else "off")
        _context_chip(c5, "Augmentation", "on" if info.get("use_augmentation") else "off")

        if info.get("hierarchical"):
            st.write(
                "**Hierarchical stages:** "
                f"{info.get('binary_model')} -> {info.get('subtype_model')}"
            )

        # extra breathing room between the green chips and the metrics row
        st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

        if metrics:
            cols = st.columns(len(metrics))
            labels = {
                "accuracy": "Accuracy",
                "balanced_accuracy": "Balanced acc.",
                "macro_f1": "Macro F1",
                "weighted_f1": "Weighted F1",
            }
            for col, (key, value) in zip(cols, metrics.items()):
                col.metric(labels.get(key, key), f"{value:.3f}")
        else:
            st.caption("No saved evaluation metrics were found for this exact choice.")

        cm_path = info.get("confusion_matrix_path")
        if cm_path and Path(cm_path).exists():
            st.image(cm_path, caption="Saved confusion matrix", use_container_width=True)


def render_banner(placeholder, prediction):
    """Big coloured banner with the current predicted state."""
    if prediction is None:
        placeholder.markdown(
            '<div class="state-banner state-waiting">WAITING FOR THE FIRST WINDOW...</div>',
            unsafe_allow_html=True,
        )
        return
    label = prediction["label"]
    css = _STATE_CLASS.get(label, "state-waiting")
    text = display_label(label).upper()
    if prediction.get("proba"):
        text += "  -  " + "  ".join(
            f"{display_label(cls)} {prediction['proba'][cls]:.0%}"
            for cls in sorted(prediction["proba"])
        )
    placeholder.markdown(
        f'<div class="state-banner {css}">{text}</div>', unsafe_allow_html=True,
    )


# Per-window feature groups (keys match pipeline.assemble_feature_row output).
# label -> column key, grouped by signal so each table/chart stays on one scale.
_POSTURE_METRICS = {
    "Shoulder mean (deg)": "p_shoulder_mean",
    "Shoulder std (deg)": "p_shoulder_std",
    "Shoulder bad %": "p_shoulder_bad_pct",
    "Head tilt mean (deg)": "p_head_tilt_mean",
    "Head tilt std (deg)": "p_head_tilt_std",
    "Head tilt bad %": "p_head_tilt_bad_pct",
    "FHP mean (ratio)": "p_fhp_mean",
    "FHP std (ratio)": "p_fhp_std",
    "FHP bad %": "p_fhp_bad_pct",
}
_BLINK_METRICS = {
    "EAR mean": "b_ear_mean",
    "EAR std": "b_ear_std",
    "PERCLOS": "b_perclos",
    "Blink count": "b_blink_count",
    "Blink freq (/min)": "b_blink_freq",
    "IBI mean (s)": "b_ibi_mean",
    "IBI std (s)": "b_ibi_std",
}
_HEART_METRICS = {
    "HR (bpm)": "HR",
    "RMMSD": "RMMSD",
}


def _metrics_frame(records, label_map):
    """Build a per-window DataFrame (one row per window) for a metric group."""
    index = [r["window"] for r in records]
    data = {label: [r.get(key, np.nan) for r in records]
            for label, key in label_map.items()}
    return pd.DataFrame(data, index=index)


def render_window_metrics(placeholder):
    """End-of-acquisition tables: every per-window metric, by signal group."""
    records = st.session_state.get("window_metrics", [])
    if not records:
        return
    groups = [("Posture", _POSTURE_METRICS),
              ("Blinking", _BLINK_METRICS),
              ("Heart rate", _HEART_METRICS)]
    with placeholder.container():
        st.subheader("Per-window metrics")
        st.caption("All posture, blink and heart-rate features computed for "
                   "each analysis window (one row per window).")
        for name, label_map in groups:
            frame = _metrics_frame(records, label_map)
            if frame.dropna(how="all").empty:
                continue
            with st.expander(f"{name} ({len(label_map)} features)", expanded=True):
                st.dataframe(frame, use_container_width=True)


def render_history(placeholder):
    """Per-window prediction log + a small state summary."""
    history = st.session_state.history
    if not history:
        placeholder.info("No predictions yet. Start an acquisition or upload a video.")
        return
    df = pd.DataFrame(history).drop(columns=["_pred", "HR", "RMMSD"], errors="ignore")
    if "state" in df.columns:
        df["state"] = df["state"].apply(
            lambda s: "Focused" if s == "normal" else s
        )
    with placeholder.container():
        st.subheader("Per-window predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # State summary covers predicted windows only; skipped ones are reported
        # separately so they do not look like a wellbeing state.
        predicted = df[df["state"] != "skipped"]
        counts = predicted["state"].value_counts()
        if len(counts):
            cols = st.columns(len(counts))
            for col, (state, n) in zip(cols, counts.items()):
                col.metric(state.upper(), f"{n}", f"{n / len(predicted):.0%}")
        skipped = int((df["state"] == "skipped").sum())
        if skipped:
            st.caption(
                f"{skipped} window(s) skipped — {SKIP_REASON_NO_POSTURE.lower()}."
            )


class LiveCharts:
    """Per-sample buffers for the raw features, fed to streaming line charts.

    The model only emits a label once per window, but the underlying signals
    (EAR, posture angles, forward-head proxy, heart rate) are measured every
    frame. To keep the UI light, points are stored at a fixed cadence (not every
    frame) and pushed to the charts with ``add_rows`` (incremental append), so
    nothing is fully redrawn while streaming. Each signal that lives on a
    different scale gets its own chart, so e.g. the forward-head proxy (a
    ~0.7-1.2 ratio) is not flattened next to the posture angles in degrees.
    """

    def __init__(self):
        self.t = []
        self.ear = []
        self.shoulder = []
        self.tilt = []
        self.fhp = []
        self.hr = []

    def add(self, t, ear, posture_vals, hr):
        self.t.append(round(float(t), 2))
        self.ear.append(float(ear) if ear is not None else np.nan)
        if posture_vals is not None:
            shoulder, tilt, fhp = posture_vals
        else:
            shoulder = tilt = fhp = np.nan
        self.shoulder.append(float(shoulder))
        self.tilt.append(float(tilt))
        self.fhp.append(float(fhp))
        self.hr.append(float(hr) if hr is not None else np.nan)

    # --- frames for the four charts (full history) ---
    def eye_df(self):
        return pd.DataFrame({"EAR": self.ear}, index=self.t)

    def angles_df(self):
        return pd.DataFrame({"Shoulder (deg)": self.shoulder,
                             "Head tilt (deg)": self.tilt}, index=self.t)

    def fhp_df(self):
        return pd.DataFrame({"FHP (ratio)": self.fhp}, index=self.t)

    def hr_df(self):
        return pd.DataFrame({"HR (bpm)": self.hr}, index=self.t)

    def last_rows(self):
        """One-row frames for the just-added sample, for incremental add_rows."""
        i = [self.t[-1]]
        return {
            "ear": pd.DataFrame({"EAR": [self.ear[-1]]}, index=i),
            "angles": pd.DataFrame({"Shoulder (deg)": [self.shoulder[-1]],
                                    "Head tilt (deg)": [self.tilt[-1]]}, index=i),
            "fhp": pd.DataFrame({"FHP (ratio)": [self.fhp[-1]]}, index=i),
            "hr": pd.DataFrame({"HR (bpm)": [self.hr[-1]]}, index=i),
        }

    def has_data(self):
        return len(self.t) >= 1


def _empty_like(df):
    """An empty frame with the same columns, used to seed a streaming chart."""
    return df.iloc[0:0]


def make_chart_row(show_hr: bool, charts: "LiveCharts"):
    """Create the streaming charts (seeded empty) and return the chart objects."""
    st.markdown("#### Live features (model output at the end of each window)")
    specs = [("Eyes - EAR", charts.eye_df()),
             ("Posture angles (deg)", charts.angles_df()),
             ("Forward-head proxy (ratio)", charts.fhp_df())]
    if show_hr:
        specs.append(("Heart rate (bpm)", charts.hr_df()))
    cols = st.columns(len(specs))
    objs = []
    for col, (caption, frame) in zip(cols, specs):
        col.caption(caption)
        objs.append(col.line_chart(_empty_like(frame), height=200))
    if not show_hr:
        objs.append(None)  # keep a 4-slot shape (ear, angles, fhp, hr)
    return tuple(objs)


def push_charts(chart_objs, charts: "LiveCharts"):
    """Append only the newest sample to each streaming chart (cheap)."""
    ear_c, angles_c, fhp_c, hr_c = chart_objs
    rows = charts.last_rows()
    ear_c.add_rows(rows["ear"])
    angles_c.add_rows(rows["angles"])
    fhp_c.add_rows(rows["fhp"])
    if hr_c is not None:
        hr = rows["hr"]["HR (bpm)"].iloc[0]
        if 50 < hr < 120:
            hr_c.add_rows(rows["hr"])
        else:
            rows["hr"]["HR (bpm)"].iloc[0] = 50
            hr_c.add_rows(rows["hr"])


def show_persisted_charts():
    """Statically redraw the last acquisition's charts (kept in session_state)."""
    charts = st.session_state.get("charts")
    if charts is None or not charts.has_data():
        return
    show_hr = st.session_state.get("charts_show_hr", False)
    st.markdown("#### Live features (last acquisition)")
    specs = [("Eyes - EAR", charts.eye_df()),
             ("Posture angles (deg)", charts.angles_df()),
             ("Forward-head proxy (ratio)", charts.fhp_df())]
    if show_hr:
        specs.append(("Heart rate (bpm)", charts.hr_df()))
    cols = st.columns(len(specs))
    for col, (caption, frame) in zip(cols, specs):
        col.caption(caption)
        col.line_chart(frame, height=200)


def metrics_html(ear, posture_ok, face, hr_live, hr_on, p_count, b_count, remaining,
                 eye_vis=0.0, eyes_ok=True):
    good, dim = "metric-good", "metric-dim"
    ear_txt = f"{ear:.3f}" if ear is not None else "--"
    hr_txt = (f"{hr_live} bpm" if hr_live is not None else "warming up") if hr_on else "not connected"
    eye_txt = (f"visible ({eye_vis:.0f})" if eyes_ok else f"occluded? ({eye_vis:.0f})")
    return f"""
    <ul style="list-style:none;padding-left:0;line-height:1.9;font-size:1.05rem;">
      <li><b>Next reading in</b>: {max(0.0, remaining):.1f}s</li>
      <li><b>Eyes (EAR)</b>: <span class="{good if face else dim}">{ear_txt}</span></li>
      <li><b>Eye signal</b>: <span class="{good if (face and eyes_ok) else dim}">{eye_txt}</span></li>
      <li><b>Posture</b>: <span class="{good if posture_ok else dim}">
            {'tracked' if posture_ok else 'not visible'}</span></li>
      <li><b>Wearable</b>: <span class="{good if hr_on else dim}">{hr_txt}</span></li>
      <li><b>Samples this window</b>: posture={p_count} - blink={b_count}</li>
    </ul>
    """

def ecg_beat(n_samples: int) -> np.ndarray:
    t = np.linspace(0, 1, n_samples)
    r  =  1.00 * np.exp(-((t - 0.45) ** 2) / (2 * 0.018 ** 2))
    s  = -0.15 * np.exp(-((t - 0.52) ** 2) / (2 * 0.010 ** 2))
    tw =  0.25 * np.exp(-((t - 0.72) ** 2) / (2 * 0.045 ** 2))
    return r + s + tw

def bpm_to_ecg_chunk(bpm: float, sample_rate: int = 250) -> np.ndarray:
    rng = np.random.default_rng()
    samples_per_beat = int(sample_rate * 60 / bpm)
    chunk = np.zeros(sample_rate)
    pos = 0
    while pos < len(chunk):
        rr = int(samples_per_beat * rng.uniform(0.97, 1.03))
        if pos + rr > len(chunk):
            break
        chunk[pos:pos + rr] = ecg_beat(rr)
        pos += rr
    return chunk

# =============================================================================
# REAL-TIME ACQUISITION
# =============================================================================

def run_realtime(model, settings):
    """Webcam loop: process every frame, predict once per window.

    Stopping works through Streamlit's rerun mechanism: when the Stop button is
    pressed, Streamlit interrupts this loop at the next ``image()`` call and
    reruns the script with ``rt_running == False``.
    """
    import cv2
    import mediapipe as mp

    mods = pipeline.load_analyzers()
    pipeline.apply_settings(settings)
    posture_config = mods["posture_config"]
    overlay = mods["overlay"]
    window_size = posture_config.WINDOW_SIZE
    st.session_state.processed_window_seconds = int(window_size)
    if st.session_state.get("session_id") is None:
        _begin_analysis_session("realtime", video_source="webcam")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=posture_config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=posture_config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=posture_config.MP_MIN_TRACKING_CONFIDENCE,
    )
    posture_analyzer = mods["PostureAnalyzer"]()
    blink_analyzer = mods["BlinkAnalyzer"]()
    posture_fe = mods["FeatureExtractor"]()
    blink_fe = mods["BlinkFeatureExtractor"]()

    hr_monitor = pipeline.HeartRateMonitor(
        settings["wearable_port"] if settings["use_wearable"] else None
    )
    hr_monitor.start()

    cap, camera_index = pipeline.open_camera(settings["camera_index"])
    if cap is None:
        st.error(
            "Camera access is unavailable. On macOS, open **System Settings → "
            "Privacy & Security → Camera** and enable access for the terminal or "
            "application used to start Streamlit. Then close other apps using the "
            "camera, restart SWEM, and press Start again."
        )
        hr_monitor.stop()
        st.session_state.rt_running = False
        return
    settings["camera_index"] = camera_index

    # central layout: video on the left, metrics on the right
    col_video, col_side = st.columns([3, 2])
    video_ph = col_video.empty()
    banner_ph = col_side.empty()
    metrics_ph = col_side.empty()
    charts = LiveCharts()
    chart_objs = make_chart_row(show_hr=settings["use_wearable"], charts=charts)
    st.session_state.charts = charts      # kept so they stay visible after Stop
    st.session_state.charts_show_hr = settings["use_wearable"]

    window_start = time.time()
    t0 = window_start
    last_chart = 0.0
    prediction = st.session_state.history[-1]["_pred"] if st.session_state.history else None

    try:
        while st.session_state.rt_running:
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.time()
            t_rel = now - window_start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            landmarks = pipeline.landmarks_from_results(pose.process(rgb))
            values = posture_analyzer.extract_values(landmarks)
            posture_ok = values is not None
            if posture_ok:
                posture_fe.add_sample(t_rel, *values)

            blink_result = blink_analyzer.analyze(frame)
            face = blink_result["face_detected"]
            ear = blink_result["ear"]
            eyes_visible = blink_result.get("eyes_visible", True)
            # Drop frames where the eyes are occluded (sunglasses/hand): FaceMesh
            # still reports an EAR, but it is fabricated. Skipping them leaves the
            # window with too few blink samples, so the blink features get imputed.
            if face and ear is not None and eyes_visible:
                blink_fe.add_sample(t_rel, ear, blink_result["is_closed"])

            hr_monitor.poll()

            overlay.draw_posture_skeleton(cv2, frame, landmarks, values, posture_config)
            overlay.draw_eye_boxes(cv2, frame, blink_result.get("left_eye", []),
                                   blink_result.get("right_eye", []),
                                   blink_result.get("is_closed", False))

            # end of a window -> features + prediction
            if t_rel >= window_size:
                posture_rows = posture_fe.compute_windows()
                blink_rows = blink_fe.compute_windows()
                hr_mean, rmssd = hr_monitor.consume_window()
                feature_row = pipeline.assemble_feature_row(posture_rows, blink_rows, hr_mean, rmssd)
                stamp = time.strftime("%H:%M:%S")
                if feature_row is not None:
                    prediction = model.predict_row(feature_row)
                    st.session_state.history.append({
                        "time": stamp,
                        "state": pretty_label(prediction["label"]),
                        "note": _imputation_note(blink_rows, hr_mean, rmssd),
                        "HR": "n/a" if np.isnan(hr_mean) else round(hr_mean, 1),
                        "RMMSD": "n/a" if np.isnan(rmssd) else round(rmssd, 4),
                        "_pred": prediction,
                    })
                    st.session_state.window_metrics.append({
                        "window": stamp,
                        "state": pretty_label(prediction["label"]),
                        **feature_row,
                    })
                    _append_processed_record(
                        settings,
                        prediction,
                        feature_row,
                        window_start=round(window_start - t0, 2),
                        window_end=round(now - t0, 2),
                        video_source="webcam",
                    )
                else:
                    st.session_state.history.append({
                        "time": stamp,
                        "state": "skipped",
                        "note": SKIP_REASON_NO_POSTURE,
                        "HR": "n/a",
                        "RMMSD": "n/a",
                        "_pred": None,
                    })
                posture_fe.reset()
                blink_fe.reset()
                window_start = now

            # draw UI (these calls also let Streamlit interrupt on Stop)
            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                           use_container_width=True)
            render_banner(banner_ph, prediction)
            metrics_ph.markdown(
                metrics_html(ear, posture_ok, face, hr_monitor.live_bpm,
                             hr_monitor.available, posture_fe.get_sample_count(),
                             blink_fe.get_sample_count(), window_size - t_rel,
                             eye_vis=blink_result.get("eye_visibility", 0.0),
                             eyes_ok=eyes_visible),
                unsafe_allow_html=True,
            )
            if now - last_chart >= 0.4:   # store + stream one point (cheap add_rows)
                charts.add(now - t0, ear, values, hr_monitor.live_bpm)
                push_charts(chart_objs, charts)
                last_chart = now
    finally:
        cap.release()
        pose.close()
        blink_analyzer.close()
        hr_monitor.stop()


# =============================================================================
# VIDEO UPLOAD ANALYSIS
# =============================================================================

def load_heart_csv(uploaded_file):
    """Read an uploaded Heart CSV into a list of (HR, RMMSD), one row per minute."""
    if uploaded_file is None:
        return []
    try:
        heart = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig")
        if heart.shape[1] == 1:
            uploaded_file.seek(0)
            heart = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Heart CSV not readable ({exc}); HR/RMMSD will be imputed.")
        return []
    cols = {c.strip().lower(): c for c in heart.columns}
    hr_col = cols.get("hr")
    rmssd_col = cols.get("rmmsd") or cols.get("rmssd")
    if hr_col is None or rmssd_col is None:
        st.warning("The CSV has no HR/RMMSD columns; they will be imputed.")
        return []
    hr = pd.to_numeric(heart[hr_col], errors="coerce")
    rmssd = pd.to_numeric(heart[rmssd_col], errors="coerce")
    return list(zip(hr.tolist(), rmssd.tolist()))


def run_video(model, settings, video_path, heart):
    """Play an uploaded video, drawing overlays and predicting once per minute."""
    import cv2
    import mediapipe as mp

    mods = pipeline.load_analyzers()
    pipeline.apply_settings(settings)
    posture_config = mods["posture_config"]
    overlay = mods["overlay"]
    window_size = posture_config.WINDOW_SIZE
    st.session_state.processed_window_seconds = int(window_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open the video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=posture_config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=posture_config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=posture_config.MP_MIN_TRACKING_CONFIDENCE,
    )
    posture_analyzer = mods["PostureAnalyzer"]()
    blink_analyzer = mods["BlinkAnalyzer"]()
    posture_fe = mods["FeatureExtractor"]()
    blink_fe = mods["BlinkFeatureExtractor"]()

    col_video, col_side = st.columns([3, 2])
    video_ph = col_video.empty()
    progress = col_video.progress(0.0, text="Analysing...")
    banner_ph = col_side.empty()
    metrics_ph = col_side.empty()
    show_hr = bool(heart)
    charts = LiveCharts()
    chart_objs = make_chart_row(show_hr=show_hr, charts=charts)
    st.session_state.charts = charts      # kept so they stay visible after analysis
    st.session_state.charts_show_hr = show_hr

    minute_index = 0
    minute_start = 0.0
    frame_index = 0
    prediction = None

    try:
        while st.session_state.get("rt_running", False):
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_index / fps
            t_rel = t - minute_start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            landmarks = pipeline.landmarks_from_results(pose.process(rgb))
            values = posture_analyzer.extract_values(landmarks)
            posture_ok = values is not None
            if posture_ok:
                posture_fe.add_sample(t_rel, *values)

            blink_result = blink_analyzer.analyze(frame)
            face = blink_result["face_detected"]
            ear = blink_result["ear"]
            eyes_visible = blink_result.get("eyes_visible", True)
            if face and ear is not None and eyes_visible:
                blink_fe.add_sample(t_rel, ear, blink_result["is_closed"])

            cur_hr = (heart[minute_index][0]
                      if show_hr and minute_index < len(heart) else None)
            if frame_index % 10 == 0:   # stream one point every ~10 frames
                charts.add(t, ear, values, cur_hr)
                push_charts(chart_objs, charts)

            if t_rel >= window_size:
                posture_rows = posture_fe.compute_windows()
                blink_rows = blink_fe.compute_windows()
                hr, rmssd = (heart[minute_index] if minute_index < len(heart)
                             else (float("nan"), float("nan")))
                hr, rmssd = float(hr), float(rmssd)
                feature_row = pipeline.assemble_feature_row(posture_rows, blink_rows, hr, rmssd)
                if feature_row is not None:
                    prediction = model.predict_row(feature_row)
                    st.session_state.history.append({
                        "minute": minute_index + 1,
                        "state": pretty_label(prediction["label"]),
                        "note": _imputation_note(blink_rows, hr, rmssd),
                        "HR": "n/a" if np.isnan(hr) else round(hr, 1),
                        "RMMSD": "n/a" if np.isnan(rmssd) else round(rmssd, 4),
                        "_pred": prediction,
                    })
                    st.session_state.window_metrics.append({
                        "window": minute_index + 1,
                        "state": pretty_label(prediction["label"]),
                        **feature_row,
                    })
                    _append_processed_record(
                        settings,
                        prediction,
                        feature_row,
                        window_start=round(minute_start, 2),
                        window_end=round(t, 2),
                        video_source=st.session_state.get("session_video_source", "video"),
                    )
                else:
                    st.session_state.history.append({
                        "minute": minute_index + 1,
                        "state": "skipped",
                        "note": SKIP_REASON_NO_POSTURE,
                        "HR": "n/a",
                        "RMMSD": "n/a",
                        "_pred": None,
                    })
                minute_index += 1
                posture_fe.reset()
                blink_fe.reset()
                minute_start = t

            # draw every few frames to keep the UI responsive
            if frame_index % 3 == 0:
                overlay.draw_posture_skeleton(cv2, frame, landmarks, values, posture_config)
                overlay.draw_eye_boxes(cv2, frame, blink_result.get("left_eye", []),
                                       blink_result.get("right_eye", []),
                                       blink_result.get("is_closed", False))
                video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                               use_container_width=True)
                render_banner(banner_ph, prediction)
                metrics_ph.markdown(
                    metrics_html(ear, posture_ok, face, cur_hr, show_hr,
                                 posture_fe.get_sample_count(),
                                 blink_fe.get_sample_count(), window_size - t_rel,
                                 eye_vis=blink_result.get("eye_visibility", 0.0),
                                 eyes_ok=eyes_visible),
                    unsafe_allow_html=True,
                )
            if total_frames > 0:
                progress.progress(min(frame_index / total_frames, 1.0),
                                  text=f"Analysing... frame {frame_index}/{total_frames}")
            frame_index += 1
    finally:
        cap.release()
        pose.close()
        blink_analyzer.close()
    progress.progress(1.0, text="Analysis complete")
    st.session_state.rt_running = False
    st.session_state.acquisition_complete = True


# =============================================================================
# FIVE-PAGE WIZARD
# =============================================================================

def _set_page(page):
    st.session_state.page = page


def _go_home():
    st.session_state.rt_running = False
    _clear_data()
    _remove_uploaded_video()
    st.session_state.page = 1
    st.session_state.source = None
    st.session_state.app_settings = None
    st.session_state.trained_model = None
    st.session_state.model_signature = None
    st.session_state.model_info = None
    st.session_state.save_processed_data = False


def _page_header(title, step):
    st.markdown(f'<div class="swem-step">Step {step} of 5</div>', unsafe_allow_html=True)
    st.title(title)


def _bottom_navigation(step):
    st.write("")
    st.divider()
    left, spacer, right = st.columns([1, 4, 1])
    left.button("Back", on_click=_set_page, args=(max(1, step - 1),),
                use_container_width=True, key=f"bottom_back_{step}")
    right.button("Home", on_click=_go_home, use_container_width=True,
                 key=f"bottom_home_{step}")


def _page_home():
    logo = pipeline.PROJECT_ROOT / "docs" / "proposal" / "SWEM.png"
    _, center, _ = st.columns([1, 4, 1])
    with center:
        st.image(str(logo), use_container_width=True)
        st.markdown(
            '<div class="swem-subtitle">Student Wellbeing Evaluation &amp; Monitoring</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("Start", type="primary", use_container_width=True,
                  on_click=_set_page, args=(2,))


def _page_source():
    _page_header("Choose the acquisition source", 2)
    with st.container(border=True):
        choice = st.radio(
            "How would you like to provide the data?",
            ["video", "realtime"],
            format_func=lambda value: {
                "video": "Upload a video and an optional heart-rate CSV",
                "realtime": "Start a real-time acquisition",
            }[value],
            horizontal=True,
            key="source_choice",
        )
        if choice == "video":
            col_video, col_hr = st.columns(2)
            video_file = col_video.file_uploader(
                "Video file", type=["mp4", "mov", "avi", "mkv"], key="video_upload"
            )
            heart_file = col_hr.file_uploader(
                "Heart-rate CSV (optional)", type=["csv"], key="heart_upload",
                help="The file may contain HR and RMMSD columns, one row per minute.",
            )
            persisted_video = st.session_state.get("uploaded_video_path")
            has_persisted_video = bool(persisted_video and Path(persisted_video).exists())
            if video_file is None and has_persisted_video:
                st.success(
                    f"Video ready: {st.session_state.get('uploaded_video_name', 'uploaded video')}"
                )
            st.caption("All three sensors remain part of model training. Missing HR data is imputed.")
            if st.button("Continue", type="primary", use_container_width=True,
                         disabled=video_file is None and not has_persisted_video):
                if video_file is not None:
                    _persist_video_upload(video_file, heart_file)
                st.session_state.source = "video"
                st.session_state.use_wearable = False
                _set_page(3)
                st.rerun()
        else:
            port = st.text_input(
                "PPG serial port (optional)",
                value=st.session_state.wearable_port,
                placeholder="Example: COM3 or /dev/ttyUSB0",
            )
            st.caption("Leave the port empty or skip PPG to continue with camera-based sensors only.")
            use_col, skip_col = st.columns(2)
            if use_col.button("Continue with PPG", type="primary", use_container_width=True,
                              disabled=not port.strip()):
                _remove_uploaded_video()
                st.session_state.source = "realtime"
                st.session_state.wearable_port = port.strip()
                st.session_state.use_wearable = True
                _set_page(3)
                st.rerun()
            if skip_col.button("Skip PPG", use_container_width=True):
                _remove_uploaded_video()
                st.session_state.source = "realtime"
                st.session_state.wearable_port = ""
                st.session_state.use_wearable = False
                _set_page(3)
                st.rerun()
    _bottom_navigation(2)


def _page_model():
    _page_header("Configure and train the model", 3)
    with st.container(border=True):
        settings = model_configuration_controls()
    if settings["is_trained"]:
        st.success(f"'{settings['model_name']}' is trained and ready.")
        render_model_status(settings)
        st.button("Continue to acquisition", type="primary", use_container_width=True,
                  on_click=_set_page, args=(4,))
    _bottom_navigation(3)


def _start_video_analysis():
    st.session_state.rt_running = True
    st.session_state.acquisition_complete = False


def _page_acquisition():
    _page_header("Data acquisition", 4)
    settings = st.session_state.get("app_settings")
    model = st.session_state.get("trained_model")
    if settings is None or model is None:
        st.warning("Train a model before starting the acquisition.")
        st.button("Return to model configuration", on_click=_set_page, args=(3,))
        _bottom_navigation(4)
        return

    start_col, stop_col = st.columns(2)
    if settings["source"] == "realtime":
        start_col.button("Start", type="primary", on_click=_start_realtime,
                         use_container_width=True, disabled=st.session_state.rt_running)
    else:
        start_col.button("Start", type="primary", on_click=_start_video_analysis,
                         use_container_width=True, disabled=st.session_state.rt_running)
    stop_col.button("Stop", on_click=_stop_realtime, use_container_width=True,
                    disabled=not st.session_state.rt_running)

    if st.session_state.rt_running:
        if settings["source"] == "realtime":
            run_realtime(model, settings)
        else:
            video_path = st.session_state.get("uploaded_video_path")
            video_name = st.session_state.get("uploaded_video_name")
            if not video_path or not Path(video_path).exists():
                st.error("The uploaded video is no longer available. Return to the source page.")
                st.session_state.rt_running = False
                _bottom_navigation(4)
                return
            _begin_analysis_session("video", video_source=video_name)
            run_video(
                model,
                settings,
                video_path,
                st.session_state.get("uploaded_heart_data", []),
            )
    else:
        show_persisted_charts()

    if st.session_state.acquisition_complete:
        st.divider()
        st.subheader("Acquisition review")
        render_history(st.container())
        render_window_metrics(st.container())
        redo_col, next_col = st.columns(2)
        if redo_col.button("Redo acquisition", use_container_width=True):
            _clear_data()
            st.rerun()
        next_col.button("Continue", type="primary", use_container_width=True,
                        on_click=_set_page, args=(5,))
    _bottom_navigation(4)


def _finish_without_saving():
    st.session_state.save_processed_data = False
    _go_home()


def _save_and_finish():
    st.session_state.save_processed_data = True
    _save_processed_records(final=True)
    _go_home()


def _page_save():
    _page_header("Save acquisition data?", 5)
    with st.container(border=True):
        st.write("Would you like to save the reviewed acquisition as a processed CSV file?")
        save_col, discard_col = st.columns(2)
        save_col.button("Save and return home", type="primary", use_container_width=True,
                        on_click=_save_and_finish)
        discard_col.button("Do not save", use_container_width=True,
                           on_click=_finish_without_saving)
    _bottom_navigation(5)


def main():
    st.session_state.save_processed_data = False
    pages = {
        1: _page_home,
        2: _page_source,
        3: _page_model,
        4: _page_acquisition,
        5: _page_save,
    }
    pages.get(st.session_state.page, _page_home)()


if __name__ == "__main__":
    main()
