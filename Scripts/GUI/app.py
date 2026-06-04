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
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pipeline


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
        "model_signature": None,   # (task, learning, name, hierarchical)
        "rt_running": False,       # webcam loop active?
        "history": [],             # list of per-window prediction dicts
        "window_metrics": [],      # list of full per-window feature dicts
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _start_realtime():
    st.session_state.rt_running = True


def _stop_realtime():
    st.session_state.rt_running = False


def _clear_data():
    """Drop the current acquisition's results (history, metrics, live charts)."""
    st.session_state.history = []
    st.session_state.window_metrics = []
    st.session_state.charts = None
    st.session_state.charts_show_hr = False


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

    try:
        model_names = pipeline.available_models(task, learning)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read the models: {exc}")
        model_names = []

    model_name = st.sidebar.selectbox(
        "Model", options=model_names,
        help="The options change according to the two choices above.",
    ) if model_names else None

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

    signature = (task, learning, model_name, hierarchical)
    is_trained = (st.session_state.trained_model is not None
                  and st.session_state.model_signature == signature)

    train_label = "Model ready" if is_trained else "Train / load model"
    if st.sidebar.button(train_label, use_container_width=True, disabled=model_name is None):
        with st.spinner(f"Training '{model_name}' on data/supervised..."):
            try:
                if hierarchical:
                    model = pipeline.build_hierarchical_model(model_name)
                else:
                    model = pipeline.build_and_train_model(task, learning, model_name)
                st.session_state.trained_model = model
                st.session_state.model_signature = signature
                st.sidebar.success(f"'{model_name}' trained.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.trained_model = None
                st.session_state.model_signature = None
                st.sidebar.error(f"Training error: {exc}")

    if is_trained:
        mode_txt = ("hierarchical (binary -> sleepy/stress)"
                    if hierarchical else f"{task} - {learning}")
        st.sidebar.caption(f"Active: **{model_name}** - {mode_txt}")
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

    # ---- 3. ANALYSIS WINDOW ----------------------------------------------
    st.sidebar.header("Analysis window")
    window_size = st.sidebar.slider(
        "Window length (seconds)", min_value=10, max_value=120, value=60, step=5,
        help="A prediction is produced at every completed window. "
             "60s = same as the training data; shorter values = quicker demos.",
    )

    # ---- 4. THRESHOLDS ----------------------------------------------------
    # Defaults are read from Scripts/Posture/config.py and
    # Scripts/Blinking/blink_config.py (literature-based values).
    dft = pipeline.default_thresholds()
    with st.sidebar.expander("Posture / blinking thresholds", expanded=False):
        st.caption("Defaults from Scripts/Posture/config.py and "
                   "Scripts/Blinking/blink_config.py (literature-based).")
        shoulder_thr = st.slider("Shoulder asymmetry (deg)", 1.0, 20.0, dft["shoulder_thr"], 0.5)
        head_tilt_thr = st.slider("Head lateral tilt (deg)", 1.0, 30.0, dft["head_tilt_thr"], 0.5)
        fhp_thr = st.slider("Forward-head proxy (threshold)", 0.3, 1.5, dft["fhp_thr"], 0.05)
        min_visibility = st.slider("Minimum landmark visibility", 0.1, 0.9, dft["min_visibility"], 0.05)
        ear_thr = st.slider("EAR closed-eye", 0.10, 0.35, dft["ear_thr"], 0.01)
        perclos_thr = st.slider("PERCLOS problematic", 0.05, 0.6, dft["perclos_thr"], 0.05)

    # ---- 5. DATA ----------------------------------------------------------
    st.sidebar.header("Data")
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
        "model_name": model_name,
        "hierarchical": hierarchical,
        "is_trained": is_trained,
        "source": source,
        "camera_index": int(camera_index),
        "use_wearable": use_wearable,
        "wearable_port": wearable_port,
        "window_size": window_size,
        "shoulder_thr": shoulder_thr,
        "head_tilt_thr": head_tilt_thr,
        "fhp_thr": fhp_thr,
        "min_visibility": min_visibility,
        "ear_thr": ear_thr,
        "perclos_thr": perclos_thr,
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
            st.session_state.history = []  # fresh log for this video
            st.session_state.window_metrics = []  # fresh per-window metrics
            heart = load_heart_csv(heart_file)
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=Path(video_file.name).suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            run_video(model, settings, tmp_path, heart)
        else:
            show_persisted_charts()

    # --- HISTORY (always shown below) -------------------------------------
    st.divider()
    render_history(st.container())
    render_window_metrics(st.container())


if __name__ == "__main__":
    main()
