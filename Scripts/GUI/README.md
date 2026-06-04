# GUI — Wellbeing Monitor (Streamlit)

A green-themed Streamlit dashboard that drives the project's existing analysis
code (the sibling folders under [`Scripts/`](../)) without duplicating it.

## Layout

- **Left sidebar** — all parameters:
  - **Model**: number of classes (`binary` / `3class`), learning type
    (`supervised` / `unsupervised`) and the exact estimator. The model list
    updates automatically from [`Scripts/ML`](../ML/) via the same
    `model_catalog` used by the CLI. For `binary` + `supervised` there is an
    optional hierarchical mode (if anomalous, predict sleepy vs stress).
  - **Source**: real-time webcam (+ optional CMS50D wearable) or video upload.
  - **Analysis window**: window length in seconds (60 s = same as training).
  - **Thresholds**: posture / blink thresholds, read from the config files and
    pushed live into the config modules.
- **Central area** — the live (or video) feed with the body skeleton and eye
  overlays, the current predicted state, live feature charts (EAR, posture
  angles, forward-head proxy, heart rate) and a per-window prediction log.

## How it connects to the project

[`pipeline.py`](pipeline.py) is the only bridge. It wires `sys.path` exactly
like [`Scripts/App`](../App/) and loads
[`Scripts/App/main.realtime.py`](../App/main.realtime.py) as a module,
re-exporting `model_catalog`, `RealtimeModel`, `HeartRateMonitor` and the
feature-assembly helpers. The same 18-feature pipeline
(9 posture + 7 blinking + HR + RMMSD) and the same models are used as in the CLI
scripts; training data is read from [`data/supervised`](../../data/supervised/).

## Run it

```bash
# from the project root
./Scripts/GUI/run.sh
# or, equivalently
streamlit run Scripts/GUI/app.py
```

Then in the browser:

1. In the sidebar choose the model options and press **Train / load model**.
2. Pick the source:
   - **Real-time** → press **Start** (and **Stop** to end).
   - **Video** → upload a recording (optionally a Heart CSV with `HR`/`RMMSD`,
     one row per minute) and press **Analyse video**.

> The webcam mode needs a local camera, so run Streamlit on the machine with the
> camera attached. The wearable (CMS50D) is optional; without it HR/RMMSD are
> imputed by the model.
