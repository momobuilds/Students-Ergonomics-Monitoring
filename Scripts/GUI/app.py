"""
app.py - Streamlit GUI for the Students Ergonomics & Wellbeing Monitor.

A single dashboard that drives the project's existing analysis code (under
``Scripts/``) through the bridge in ``pipeline.py``:

  * LEFT SIDEBAR  -> all the parameters: which model to use (number of classes,
    supervised/unsupervised, exact estimator), the camera / wearable settings,
    the analysis window length and the posture / blink thresholds.
  * CENTRAL AREA  -> the live (or video) feed with the body skeleton and eye
    overlays, the current predicted state, the live metrics, live charts of the
    raw features and a per-window log of predictions.

Two sources are supported and share the same pipeline (Posture + Blinking +
optional Heart -> 18 features -> the chosen model):

  * Real-time : the webcam (and, optionally, the CMS50D wearable).
  * Video     : an uploaded recording (with an optional Heart CSV, one row
                per minute, aligned by position -- like Scripts/App).

Run it from the project root with:

    streamlit run GUI/app.py

(or use the helper script  ./GUI/run.sh).
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


WINDOW_SECONDS = 10
PROCESSED_OUTPUT_DIR = pipeline.PROJECT_ROOT / "data" / "processed"


# =============================================================================
# PAGE CONFIG + GREEN THEME POLISH
# =============================================================================

st.set_page_config(
    page_title="Wellbeing Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little extra green styling on top of .streamlit/config.toml.
st.markdown(
    """
    <style>
      /* green sidebar */
      section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%);
      }
      /* labels / headings / captions sit on the green bg -> light text */
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] .stMarkdown,
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
          color: #f1f8e9 !important;
      }
      /* widgets with a white/light background -> dark text so they stay readable */
      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] textarea,
      section[data-testid="stSidebar"] [data-baseweb="select"] *,
      section[data-testid="stSidebar"] [data-baseweb="input"] *,
      section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
      section[data-testid="stSidebar"] [data-testid="stExpander"] label,
      section[data-testid="stSidebar"] [data-testid="stExpander"] p,
      section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stWidgetLabel"],
      section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
          color: #13311f !important;
      }
      /* keep the expander body on a light panel so dark text reads well */
      section[data-testid="stSidebar"] [data-testid="stExpander"] {
          background-color: #f1f8e9 !important;
          border-radius: 10px;
      }
      /* the train button is always solid green with white text, so it is
         readable in both its idle and trained states */
      section[data-testid="stSidebar"] div.stButton > button {
          background-color: #114d18 !important;
          color: #ffffff !important;
          border: 1px solid #a5d6a7 !important;
          font-weight: 700;
      }
      section[data-testid="stSidebar"] div.stButton > button:hover {
          background-color: #2e7d32 !important;
          color: #ffffff !important;
      }

      /* main title accent */
      h1, h2, h3 { color: #1b5e20; }

      /* prediction banner */
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

      div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

def _init_state():
    defaults = {
        "trained_model": None,
        "model_signature": None,   # task/model/options signature
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
    st.session_state.rt_running = True


def _stop_realtime():
    st.session_state.rt_running = False
    _save_processed_records(final=True)


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


# =============================================================================
# SIDEBAR -- ALL THE PARAMETERS
# =============================================================================

def sidebar_controls() -> dict:
    st.sidebar.title("Parameters")

    # ---- 1. MODEL OPTIONS -------------------------------------------------
    st.sidebar.header("Model")
    task = st.sidebar.radio(
        "Number of classes",
        options=["binary", "3class"],
        format_func=lambda k: {"binary": "Binary (normal / anomalous)",
                               "3class": "3 classes (normal / sleepy / stress)"}[k],
    )
    learning = st.sidebar.radio(
        "Learning type",
        options=["supervised", "unsupervised"],
        format_func=lambda k: {"supervised": "Supervised",
                               "unsupervised": "Unsupervised"}[k],
    )

    # Hierarchical refinement is only defined for binary + supervised:
    # after the normal/abnormal decision, a second model says sleepy vs stress.
    hierarchical = False
    if task == "binary" and learning == "supervised":
        hierarchical = st.sidebar.checkbox(
            "If anomalous, predict the subtype (sleepy / stress)",
            value=False,
            help="Hierarchical classification: a binary stage (normal/anomalous) "
                 "and, only for anomalous windows, a second sleepy-vs-stress model.",
        )

    use_best_model = st.sidebar.checkbox(
        "Use best model automatically",
        value=True,
        help=(
            "Selects from saved evaluation results using macro F1 first, then "
            "balanced accuracy, weighted F1 and accuracy."
        ),
    )

    st.sidebar.header("Training options")
    use_augmentation = st.sidebar.checkbox(
        "Use data augmentation during training",
        value=False,
        help=(
            "Supervised augmentation is participant-wise and class-wise. "
            "Unsupervised clustering augmentation is label-free and participant-wise."
        ),
    )
    augmentation_factor = 1
    noise_level = 0.02
    if use_augmentation:
        with st.sidebar.expander("Augmentation settings", expanded=True):
            augmentation_factor = st.number_input(
                "augmentation_factor",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                help="Number of jittered copies per original training row.",
            )
            noise_level = st.number_input(
                "noise_level",
                min_value=0.001,
                max_value=0.10,
                value=0.02,
                step=0.01,
                format="%.3f",
                help="Gaussian noise as a fraction of the participant/group feature std.",
            )
            st.caption(
                f"Using factor={augmentation_factor}, noise={noise_level:.3f}. "
                "Test/held-out data is never augmented."
            )

    try:
        model_names = pipeline.available_models(task, learning)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read the models: {exc}")
        model_names = []

    best_selection = {}
    if model_names:
        try:
            best_selection = pipeline.select_best_model(
                task,
                learning,
                hierarchical=hierarchical,
                use_augmentation=use_augmentation,
            )
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not read saved model results: {exc}")
            best_selection = {}

    auto_model_name = best_selection.get("model_name")
    auto_dropdown_name = (
        best_selection.get("binary_model")
        if hierarchical
        else best_selection.get("model_name")
    )
    default_index = 0
    if auto_dropdown_name in model_names:
        default_index = model_names.index(auto_dropdown_name)

    manual_model_name = st.sidebar.selectbox(
        "Model",
        options=model_names,
        index=default_index,
        disabled=use_best_model,
        help="Disabled when automatic best-model selection is enabled.",
    ) if model_names else None

    if use_best_model and best_selection:
        st.sidebar.success(
            "Best model selected automatically based on macro F1 and balanced accuracy."
        )
        if auto_model_name:
            st.sidebar.caption(f"Automatic choice: **{auto_model_name}**")

    selected_display_name = auto_model_name if use_best_model else manual_model_name
    binary_model_name = (
        best_selection.get("binary_model") if use_best_model else manual_model_name
    )
    subtype_model_name = (
        best_selection.get("subtype_model") if use_best_model else manual_model_name
    )

    if not selected_display_name:
        selected_display_name = manual_model_name
    if not binary_model_name:
        binary_model_name = manual_model_name
    if not subtype_model_name:
        subtype_model_name = manual_model_name

    effective_augmentation_factor = int(augmentation_factor) if use_augmentation else 0
    effective_noise_level = float(noise_level) if use_augmentation else 0.0
    if use_best_model and use_augmentation and best_selection:
        best_factor = best_selection.get("augmentation_factor")
        best_noise = best_selection.get("noise_level")
        if pd.notna(best_factor) and pd.notna(best_noise):
            try:
                best_factor = int(best_factor)
                best_noise = float(best_noise)
                if best_factor > 0 and best_noise > 0:
                    effective_augmentation_factor = best_factor
                    effective_noise_level = best_noise
                    st.sidebar.caption(
                        "Automatic best-result augmentation setting: "
                        f"factor={best_factor}, noise={best_noise:.3f}."
                    )
            except (TypeError, ValueError):
                pass

    signature = (
        task,
        learning,
        selected_display_name,
        binary_model_name,
        subtype_model_name,
        hierarchical,
        use_best_model,
        use_augmentation,
        effective_augmentation_factor,
        effective_noise_level,
    )
    is_trained = (st.session_state.trained_model is not None
                  and st.session_state.model_signature == signature)

    train_label = "Model ready" if is_trained else "Train / load model"
    if st.sidebar.button(
        train_label,
        use_container_width=True,
        disabled=selected_display_name is None,
    ):
        with st.spinner(f"Training '{selected_display_name}' on data/supervised..."):
            try:
                model_metrics = best_selection if use_best_model else pipeline.lookup_model_metrics(
                    task,
                    learning,
                    manual_model_name,
                    hierarchical=hierarchical,
                    use_augmentation=use_augmentation,
                )
                if hierarchical:
                    model = pipeline.build_hierarchical_model(
                        binary_model_name,
                        subtype_model_name,
                        use_augmentation=use_augmentation,
                        augmentation_factor=effective_augmentation_factor,
                        noise_level=effective_noise_level,
                    )
                else:
                    model = pipeline.build_and_train_model(
                        task,
                        learning,
                        selected_display_name,
                        use_augmentation=use_augmentation,
                        augmentation_factor=effective_augmentation_factor,
                        noise_level=effective_noise_level,
                    )
                st.session_state.trained_model = model
                st.session_state.model_signature = signature
                st.session_state.model_info = {
                    "task": "hierarchical" if hierarchical else task,
                    "learning": learning,
                    "model_name": selected_display_name,
                    "binary_model": binary_model_name,
                    "subtype_model": subtype_model_name if hierarchical else None,
                    "hierarchical": hierarchical,
                    "use_best_model": use_best_model,
                    "use_augmentation": use_augmentation,
                    "augmentation_factor": effective_augmentation_factor,
                    "noise_level": effective_noise_level,
                    "metrics": model_metrics.get("metrics", {}),
                    "results_source": model_metrics.get("results_source", ""),
                    "augmentation_name": model_metrics.get("augmentation_name", ""),
                    "confusion_matrix_path": model_metrics.get("confusion_matrix_path"),
                    "training_metadata": getattr(model, "training_metadata", {}),
                }
                st.sidebar.success(f"'{selected_display_name}' trained.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.trained_model = None
                st.session_state.model_signature = None
                st.session_state.model_info = None
                st.sidebar.error(f"Training error: {exc}")

    if is_trained:
        mode_txt = ("hierarchical (binary -> sleepy/stress)"
                    if hierarchical else f"{task} - {learning}")
        st.sidebar.caption(f"Active: **{selected_display_name}** - {mode_txt}")
    elif st.session_state.trained_model is not None:
        st.sidebar.warning("The selection changed: retrain the model.")

    # ---- 2. SOURCE --------------------------------------------------------
    st.sidebar.header("Source")
    source = st.sidebar.radio(
        "What do you want to analyse?",
        options=["realtime", "video"],
        format_func=lambda k: {"realtime": "Real-time acquisition (webcam)",
                               "video": "Video upload"}[k],
    )

    camera_index = 0   # default webcam, not exposed in the UI
    use_wearable = False
    wearable_port = "COM3"
    if source == "realtime":
        use_wearable = st.sidebar.checkbox("Use CMS50D wearable (HR/HRV)", value=False)
        if use_wearable:
            wearable_port = st.sidebar.text_input("Wearable serial port", value="COM3")

    # ---- 3. ANALYSIS WINDOW (fixed) --------------------------------------
    # The window length is fixed at 60s to match the training data: every
    # feature in data/supervised is computed over 60s windows, so a prediction
    # is produced once per minute. Not editable from the GUI.
    st.sidebar.header("Analysis window")
    st.sidebar.caption(
        "Each window is **60 seconds** (one prediction per minute), fixed to "
        "match the training data (data/supervised)."
    )

    # ---- 4. THRESHOLDS (fixed) -------------------------------------------
    # The posture/blink thresholds are intentionally NOT editable. The model is
    # trained on data/supervised, whose features (bad_pct, PERCLOS, blink
    # detection) were computed with these exact literature values. Changing them
    # at inference time would no longer match the training data, so they are
    # shown read-only and kept at their config defaults.
    dft = pipeline.default_thresholds()
    with st.sidebar.expander("Posture / blinking thresholds (fixed)", expanded=False):
        rows = [
            ("Shoulder asymmetry", f"{dft['shoulder_thr']:.0f}&deg;"),
            ("Head lateral tilt", f"{dft['head_tilt_thr']:.0f}&deg;"),
            ("Forward-head proxy", f"{dft['fhp_thr']:.2f}"),
            ("Min landmark visibility", f"{dft['min_visibility']:.2f}"),
            ("EAR closed-eye", f"{dft['ear_thr']:.2f}"),
            ("PERCLOS problematic", f"{dft['perclos_thr']:.2f}"),
        ]
        items = "".join(
            f'<li style="margin:0.55rem 0;display:flex;align-items:center;'
            f'justify-content:space-between;gap:1.2rem;width:100%;">'
            f'<span style="font-weight:600;flex:1;">{name}</span>'
            f'<span style="background:#66bb6a;color:#fff;border-radius:6px;'
            f'padding:0.05rem 0.5rem;font-weight:700;font-size:0.85rem;'
            f'min-width:3.2rem;text-align:center;flex:none;">'
            f'{value}</span></li>'
            for name, value in rows
        )
        st.markdown(
            f'<ul style="list-style:none;padding-left:0;margin:0;color:#000;">'
            f'{items}</ul>',
            unsafe_allow_html=True,
        )

    # ---- 5. DATA ----------------------------------------------------------
    st.sidebar.header("Data")
    save_processed_data = st.sidebar.checkbox(
        "Save processed analysis data",
        value=True,
        help="Save each analyzed one-minute window to data/processed/ as a CSV.",
    )
    st.session_state.save_processed_data = save_processed_data

    has_data = bool(st.session_state.history or st.session_state.window_metrics)
    st.sidebar.button(
        "Clear results", on_click=_clear_data, use_container_width=True,
        disabled=st.session_state.rt_running or not has_data,
        help="Discard the current predictions, per-window metrics and live "
             "charts so they are not kept for the next acquisition.",
    )

    return {
        "task": task,
        "learning": learning,
        "model_name": selected_display_name,
        "manual_model_name": manual_model_name,
        "use_best_model": use_best_model,
        "use_augmentation": use_augmentation,
        "augmentation_factor": effective_augmentation_factor,
        "noise_level": effective_noise_level,
        "save_processed_data": save_processed_data,
        "hierarchical": hierarchical,
        "is_trained": is_trained,
        "source": source,
        "camera_index": int(camera_index),
        "use_wearable": use_wearable,
        "wearable_port": wearable_port,
    }


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
    if not settings.get("save_processed_data", True):
        return

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
    _save_processed_records(final=False)


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
    if not settings["is_trained"]:
        return
    info = st.session_state.get("model_info") or {}
    metrics = info.get("metrics") or {}

    with st.expander("Active model and evaluation context", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        _context_chip(c1, "Task", str(info.get("task", settings["task"])))
        _context_chip(c2, "Learning", str(info.get("learning", settings["learning"])))
        _context_chip(c3, "Model", str(info.get("model_name", settings["model_name"])))
        _context_chip(c4, "Auto-best", "on" if info.get("use_best_model") else "off")

        aug_text = "on" if info.get("use_augmentation") else "off"
        if info.get("use_augmentation"):
            aug_text += (
                f" (factor={info.get('augmentation_factor')}, "
                f"noise={info.get('noise_level'):.3f})"
            )
        st.write(f"**Data augmentation during training:** {aug_text}")

        if info.get("hierarchical"):
            st.write(
                "**Hierarchical stages:** "
                f"{info.get('binary_model')} -> {info.get('subtype_model')}"
            )

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

        st.warning(
            "Proof-of-concept caution: the dataset contains only 4 participants, "
            "so results should be interpreted with Leave-One-Subject-Out validation "
            "and not as a finished clinical model."
        )


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
    text = pretty_label(label).upper()
    if prediction.get("proba"):
        text += "  -  " + "  ".join(
            f"{pretty_label(cls)} {prediction['proba'][cls]:.0%}"
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
    df = pd.DataFrame(history).drop(columns=["_pred"], errors="ignore")
    with placeholder.container():
        st.subheader("Per-window predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)
        counts = df["state"].value_counts()
        cols = st.columns(len(counts))
        for col, (state, n) in zip(cols, counts.items()):
            col.metric(state.upper(), f"{n}", f"{n / len(df):.0%}")


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
        objs.append(None)   # keep a 4-slot shape (ear, angles, fhp, hr)
    return tuple(objs)


def push_charts(chart_objs, charts: "LiveCharts"):
    """Append only the newest sample to each streaming chart (cheap)."""
    ear_c, angles_c, fhp_c, hr_c = chart_objs
    rows = charts.last_rows()
    ear_c.add_rows(rows["ear"])
    angles_c.add_rows(rows["angles"])
    fhp_c.add_rows(rows["fhp"])
    if hr_c is not None:
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


def metrics_html(ear, posture_ok, face, hr_live, hr_on, p_count, b_count, remaining):
    good, dim = "metric-good", "metric-dim"
    ear_txt = f"{ear:.3f}" if ear is not None else "--"
    hr_txt = (f"{hr_live} bpm" if hr_live is not None else "warming up") if hr_on else "not connected"
    return f"""
    <ul style="list-style:none;padding-left:0;line-height:1.9;font-size:1.05rem;">
      <li><b>Next reading in</b>: {max(0.0, remaining):.1f}s</li>
      <li><b>Eyes (EAR)</b>: <span class="{good if face else dim}">{ear_txt}</span></li>
      <li><b>Posture</b>: <span class="{good if posture_ok else dim}">
            {'tracked' if posture_ok else 'not visible'}</span></li>
      <li><b>Wearable</b>: <span class="{good if hr_on else dim}">{hr_txt}</span></li>
      <li><b>Samples this window</b>: posture={p_count} - blink={b_count}</li>
    </ul>
    """


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

    cap = cv2.VideoCapture(settings["camera_index"])
    if not cap.isOpened():
        st.error(f"Could not open the webcam (index {settings['camera_index']}).")
        hr_monitor.stop()
        st.session_state.rt_running = False
        return

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
            if face and ear is not None:
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
                if feature_row is not None:
                    prediction = model.predict_row(feature_row)
                    stamp = time.strftime("%H:%M:%S")
                    st.session_state.history.append({
                        "time": stamp,
                        "state": pretty_label(prediction["label"]),
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
                             blink_fe.get_sample_count(), window_size - t_rel),
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
        while True:
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
            if face and ear is not None:
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
                                 blink_fe.get_sample_count(), window_size - t_rel),
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


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    settings = sidebar_controls()

    st.title("Wellbeing Monitor")
    st.caption("Posture - Blinking - Heart rate -> state classification "
               "(normal / sleepy / stress) with the project models.")

    model = st.session_state.trained_model
    if not settings["is_trained"]:
        st.info("Choose the model options in the sidebar and press "
                "**Train / load model** to begin.")
    else:
        render_model_status(settings)
    render_save_status()

    # --- REAL-TIME ---------------------------------------------------------
    if settings["source"] == "realtime":
        st.subheader("Real-time acquisition")
        c1, c2, _ = st.columns([1, 1, 4])
        c1.button("Start", on_click=_start_realtime, use_container_width=True,
                  disabled=not settings["is_trained"] or st.session_state.rt_running)
        c2.button("Stop", on_click=_stop_realtime, use_container_width=True,
                  disabled=not st.session_state.rt_running)

        if st.session_state.rt_running and model is not None:
            run_realtime(model, settings)
        else:
            show_persisted_charts()

    # --- VIDEO UPLOAD ------------------------------------------------------
    else:
        st.subheader("Video analysis")
        up_col, csv_col = st.columns(2)
        video_file = up_col.file_uploader(
            "Upload a video", type=["mp4", "mov", "avi", "mkv"],
        )
        heart_file = csv_col.file_uploader(
            "Heart CSV (optional, 1 row/minute)", type=["csv"],
            help="HR and RMMSD columns, aligned by position to the video minutes.",
        )

        if st.button("Analyse video", use_container_width=True,
                     disabled=not settings["is_trained"] or video_file is None):
            _begin_analysis_session("video", video_source=video_file.name)
            heart = load_heart_csv(heart_file)
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=Path(video_file.name).suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            run_video(model, settings, tmp_path, heart)
            _save_processed_records(final=True)
            render_save_status()
        else:
            show_persisted_charts()

    # --- HISTORY (always shown below) -------------------------------------
    st.divider()
    render_history(st.container())
    render_window_metrics(st.container())


if __name__ == "__main__":
    main()
