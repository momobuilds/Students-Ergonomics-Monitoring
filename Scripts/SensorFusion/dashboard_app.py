"""Streamlit dashboard for the simulated sensor fusion prototype.

Run from the repository root:
    streamlit run Scripts/SensorFusion/dashboard_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "fake_sensor_data.csv"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fusion import apply_fusion_to_dataframe, calculate_fusion_result
from generate_fake_data import generate_fake_data
from llm_advisor import (
    answer_data_question,
    check_ollama_available,
    generate_advice,
)


@st.cache_data
def load_sensor_data() -> pd.DataFrame:
    """Load simulated data, generating it first if the CSV is missing."""
    if not DATA_PATH.exists():
        generate_fake_data(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _status_message(status: str) -> None:
    if status == "High Risk":
        st.error("High Risk")
    elif status == "Warning":
        st.warning("Warning")
    else:
        st.success("Normal")


def main() -> None:
    st.set_page_config(
        page_title="Sensor Fusion Dashboard",
        layout="wide",
    )

    st.title("Sensor Fusion Dashboard")
    st.warning(
        "This dashboard is currently using simulated data. Real sensor data will be integrated later."
    )

    df = load_sensor_data()
    df_with_fusion = apply_fusion_to_dataframe(df)

    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Monitoring")
        if st.button("Start Monitoring", use_container_width=True):
            st.session_state.monitoring_active = True
        if st.button("Stop Monitoring", use_container_width=True):
            st.session_state.monitoring_active = False

        monitoring_label = "Running" if st.session_state.monitoring_active else "Stopped"
        st.caption(f"Status: {monitoring_label}")

        selected_index = st.slider(
            "Selected row",
            min_value=0,
            max_value=max(len(df) - 1, 0),
            value=max(len(df) - 1, 0),
        )

        timestamp_options = [
            timestamp.strftime("%Y-%m-%d %H:%M")
            for timestamp in df["timestamp"].dt.to_pydatetime()
        ]
        selected_timestamp = st.selectbox(
            "Timestamp",
            options=timestamp_options,
            index=selected_index,
        )
        selected_index = timestamp_options.index(selected_timestamp)

        st.header("Local LLM")
        use_local_llm = st.checkbox("Use Local LLM with Ollama", value=False)
        ollama_model = st.text_input("Ollama model", value="llama3.2:1b")
        ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
        ollama_available = check_ollama_available(ollama_host)
        if ollama_available:
            st.success("Ollama appears available.")
        else:
            st.warning("Ollama is not available.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []

    selected_row = df.iloc[selected_index]
    fusion_result = calculate_fusion_result(selected_row)
    effective_use_local_llm = use_local_llm and ollama_available
    if use_local_llm and not ollama_available:
        st.warning("Local LLM is enabled, but Ollama is unavailable. Using mock advice.")

    advice = generate_advice(
        fusion_result,
        selected_row,
        use_local_llm=effective_use_local_llm,
        model=ollama_model,
        ollama_host=ollama_host,
    )

    st.subheader("Current Sensor Readings")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Heart Rate", f"{selected_row['heart_rate_bpm']} bpm")
    metric_columns[1].metric("Blink Rate", f"{selected_row['blink_rate_per_min']}/min")
    metric_columns[2].metric("Posture", str(selected_row["posture_status"]))
    metric_columns[3].metric(
        "Slouching Time", f"{selected_row['slouching_time_minutes']} min"
    )
    metric_columns[4].metric("Temperature", f"{selected_row['temperature_c']} C")

    st.subheader("Fusion Result")
    result_columns = st.columns([1, 1, 2])
    result_columns[0].metric(
        "Final Risk Score", f"{fusion_result['final_risk_score']:.3f}"
    )
    with result_columns[1]:
        st.metric("Status", fusion_result["status"])
        _status_message(fusion_result["status"])
    with result_columns[2]:
        st.write("Fusion reasons")
        for reason in fusion_result["reasons"]:
            st.write(f"- {reason}")

    score_columns = st.columns(4)
    score_columns[0].metric("Posture Score", fusion_result["posture_score"])
    score_columns[1].metric("Blink Score", fusion_result["blink_score"])
    score_columns[2].metric("Heart Score", fusion_result["heart_score"])
    score_columns[3].metric("Temperature Score", fusion_result["temperature_score"])

    st.subheader("AI Ergonomic Assistant")
    st.metric("Alert Level", advice["alert_level"])
    st.info(advice["explanation"])
    st.write(advice["recommendation"])

    st.subheader("Chat with Local Ergonomic Assistant")
    st.caption(
        "The chat uses the full simulated dataset plus the current selected minute. No cloud API is used."
    )
    example_questions = [
        "Why is my risk high?",
        "What changed over time?",
        "What should I improve first?",
        "Is this data real or simulated?",
        "Summarize my current session.",
    ]
    st.caption("Example questions: " + " | ".join(example_questions))

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask about the current sensor and fusion data")
    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.write(user_question)

        answer = answer_data_question(
            user_question,
            df_with_fusion,
            selected_row=df_with_fusion.iloc[selected_index],
            fusion_result=fusion_result,
            use_local_llm=effective_use_local_llm,
            model=ollama_model,
            ollama_host=ollama_host,
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.write(answer)

    chart_data = df_with_fusion.set_index("timestamp")
    st.subheader("Sensor Trends")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.write("Heart rate over time")
        st.line_chart(chart_data[["heart_rate_bpm"]])
    with chart_columns[1]:
        st.write("Blink rate over time")
        st.line_chart(chart_data[["blink_rate_per_min"]])
    with chart_columns[0]:
        st.write("Slouching time over time")
        st.line_chart(chart_data[["slouching_time_minutes"]])
    with chart_columns[1]:
        st.write("Final risk score over time")
        st.line_chart(chart_data[["final_risk_score"]])

    st.subheader("Dataset With Fusion Outputs")
    st.dataframe(df_with_fusion, use_container_width=True)


if __name__ == "__main__":
    main()
