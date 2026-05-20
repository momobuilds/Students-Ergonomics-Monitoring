"""Local LLM advisor interface for explaining fusion results.

The LLM layer does not calculate risk. It only explains the deterministic
sensor fusion output in student-friendly language.
"""

from __future__ import annotations

import json
from urllib import error, request
from urllib.parse import urljoin

import pandas as pd


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
MAX_ROWS_IN_CHAT_RESPONSE = 25
DEFAULT_LUNCH_HOUR = 12


def _reason_text(fusion_result: dict) -> str:
    reasons = fusion_result.get("reasons") or []
    if isinstance(reasons, str):
        return reasons
    return ", ".join(reasons)


def _row_get(sensor_row, key: str, default="Unknown"):
    if hasattr(sensor_row, "get"):
        value = sensor_row.get(key, default)
        return default if value is None else value
    return default


def _normalize_host(ollama_host: str) -> str:
    return (ollama_host or DEFAULT_OLLAMA_HOST).rstrip("/") + "/"


def check_ollama_available(ollama_host=DEFAULT_OLLAMA_HOST) -> bool:
    """Return True when the local Ollama server is reachable."""
    try:
        url = urljoin(_normalize_host(ollama_host), "api/tags")
        with request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (error.URLError, TimeoutError, OSError):
        return False


def call_ollama_chat(
    messages,
    model=DEFAULT_OLLAMA_MODEL,
    ollama_host=DEFAULT_OLLAMA_HOST,
    timeout=30,
) -> str | None:
    """Call Ollama's local chat endpoint and return assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    url = urljoin(_normalize_host(ollama_host), "api/chat")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    message = response_data.get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def parse_advice_response(text, fallback_status="Unknown") -> dict:
    """Parse LLM JSON advice and guarantee the expected response shape."""
    default_advice = {
        "alert_level": fallback_status,
        "explanation": text or "The local LLM did not return an explanation.",
        "recommendation": (
            "Review the dashboard values and follow the system alert guidance."
        ),
    }

    if not text:
        return default_advice

    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.lower().startswith("json"):
            cleaned_text = cleaned_text[4:].strip()

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return default_advice

    if not isinstance(parsed, dict):
        return default_advice

    return {
        "alert_level": parsed.get("alert_level") or fallback_status,
        "explanation": parsed.get("explanation") or default_advice["explanation"],
        "recommendation": parsed.get("recommendation")
        or default_advice["recommendation"],
    }


def build_llm_prompt(fusion_result, sensor_row) -> str:
    """Build a strict prompt for local-only ergonomic advice."""
    reasons = _reason_text(fusion_result) or "no major risk signals"
    return (
        "You are an ergonomic assistant for students. You must not recalculate "
        "the risk score. Only explain the provided deterministic sensor fusion "
        "output. Do not make a medical diagnosis. Keep the advice short, "
        "practical, and student-friendly. Return valid JSON only with exactly "
        "these keys: alert_level, explanation, recommendation.\n\n"
        f"Fusion status: {fusion_result.get('status', 'Unknown')}\n"
        f"Final risk score: {fusion_result.get('final_risk_score', 'Unknown')}\n"
        f"Fusion reasons: {reasons}\n"
        f"Heart rate: {_row_get(sensor_row, 'heart_rate_bpm')} bpm\n"
        f"Baseline heart rate: {_row_get(sensor_row, 'baseline_heart_rate_bpm')} bpm\n"
        f"Blink rate: {_row_get(sensor_row, 'blink_rate_per_min')} blinks/min\n"
        f"Posture: {_row_get(sensor_row, 'posture_status')}\n"
        f"Slouching time: {_row_get(sensor_row, 'slouching_time_minutes')} minutes\n"
        f"Temperature: {_row_get(sensor_row, 'temperature_c')} C\n"
        f"Data source: {_row_get(sensor_row, 'data_source', 'simulated')}\n\n"
        "JSON format:\n"
        "{\n"
        '  "alert_level": "...",\n'
        '  "explanation": "...",\n'
        '  "recommendation": "..."\n'
        "}"
    )


def generate_mock_advice(fusion_result, sensor_row) -> dict:
    """Generate deterministic advice for the prototype dashboard."""
    status = fusion_result.get("status", "Unknown")
    reasons = _reason_text(fusion_result) or "the readings are close to baseline"

    explanation = (
        f"Your risk level is {status} because the system detected {reasons}."
    )

    if status == "High Risk":
        recommendation = (
            "Sit upright, relax your shoulders, look away from the screen for "
            "20 seconds, and take a short break."
        )
    elif status == "Warning":
        recommendation = (
            "Adjust your posture, blink intentionally, stretch your shoulders, "
            "and consider a brief pause if the warning continues."
        )
    elif status == "Normal":
        recommendation = (
            "Keep your current posture, continue taking regular screen breaks, "
            "and check in again after a few minutes."
        )
    else:
        recommendation = (
            "Review the sensor readings and make sure all modules are reporting "
            "valid data."
        )

    return {
        "alert_level": status,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def generate_advice_with_local_llm(
    prompt,
    model=DEFAULT_OLLAMA_MODEL,
    ollama_host=DEFAULT_OLLAMA_HOST,
    fallback_status="Unknown",
) -> dict | None:
    """Generate advice through local Ollama, returning None on any failure."""
    messages = [
        {
            "role": "system",
            "content": (
                "You explain deterministic ergonomics fusion results. "
                "Do not diagnose, do not recalculate scores, and return JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    response_text = call_ollama_chat(messages, model=model, ollama_host=ollama_host)
    if response_text is None:
        return None

    return parse_advice_response(response_text, fallback_status=fallback_status)


def generate_advice(
    fusion_result,
    sensor_row,
    use_local_llm=False,
    model=DEFAULT_OLLAMA_MODEL,
    ollama_host=DEFAULT_OLLAMA_HOST,
) -> dict:
    """Return a student-friendly explanation and recommendation."""
    if use_local_llm:
        prompt = build_llm_prompt(fusion_result, sensor_row)
        local_response = generate_advice_with_local_llm(
            prompt,
            model=model,
            ollama_host=ollama_host,
            fallback_status=fusion_result.get("status", "Unknown"),
        )
        if isinstance(local_response, dict):
            local_response["alert_level"] = (
                local_response.get("alert_level") or fusion_result.get("status", "Unknown")
            )
            return local_response

    return generate_mock_advice(fusion_result, sensor_row)


def _safe_mean(df, column: str):
    if column not in df.columns:
        return None
    try:
        return df[column].dropna().astype(float).mean()
    except (TypeError, ValueError):
        return None


def _safe_first_last(df, column: str):
    if column not in df.columns or df.empty:
        return None
    try:
        values = df[column].dropna()
        if values.empty:
            return None
        return values.iloc[0], values.iloc[-1]
    except (IndexError, TypeError, ValueError):
        return None


def _safe_min_max(df, column: str):
    if column not in df.columns:
        return None
    try:
        values = df[column].dropna().astype(float)
        if values.empty:
            return None
        return values.min(), values.max()
    except (TypeError, ValueError):
        return None


def _highest_risk_summary(df) -> str | None:
    if df.empty or "final_risk_score" not in df.columns:
        return None

    try:
        highest_index = df["final_risk_score"].astype(float).idxmax()
        row = df.loc[highest_index]
    except (TypeError, ValueError):
        return None

    parts = [
        f"score {_row_get(row, 'final_risk_score')}",
        f"status {_row_get(row, 'status')}",
    ]
    timestamp = _row_get(row, "timestamp", None)
    if timestamp is not None:
        parts.append(f"timestamp {timestamp}")
    reasons = _row_get(row, "reasons", None)
    if reasons is not None:
        parts.append(f"reasons {reasons}")

    return ", ".join(parts)


def _question_mentions_current_row(user_question: str) -> bool:
    question = (user_question or "").lower()
    current_terms = [
        "current",
        "selected",
        "this row",
        "this minute",
        "now",
        "right now",
        "my risk",
        "why is my risk",
    ]
    return any(term in question for term in current_terms)


def _requested_status(user_question: str) -> str | None:
    question = (user_question or "").lower()
    if "warning" in question:
        return "Warning"
    if "high risk" in question or "high-risk" in question or "highrisk" in question:
        return "High Risk"
    if "normal" in question:
        return "Normal"
    return None


def _question_requests_rows(user_question: str) -> bool:
    question = (user_question or "").lower()
    row_terms = [
        "row",
        "rows",
        "list",
        "show",
        "which",
        "when",
        "timestamps",
        "timestamp",
    ]
    status = _requested_status(question)
    return status is not None and any(term in question for term in row_terms)


def _format_status_rows(df, status: str, limit: int = MAX_ROWS_IN_CHAT_RESPONSE) -> str:
    if "status" not in df.columns:
        return "The dataframe does not include a fusion status column yet."

    matches = df[df["status"].astype(str).str.lower() == status.lower()]
    total = len(matches)
    if total == 0:
        return f"I could not find any rows with status {status} in the full dataset."

    columns = [
        column
        for column in [
            "timestamp",
            "heart_rate_bpm",
            "blink_rate_per_min",
            "posture_status",
            "slouching_time_minutes",
            "temperature_c",
            "final_risk_score",
            "reasons",
        ]
        if column in matches.columns
    ]

    lines = [
        f"I found {total} rows with status {status} in the full dataset.",
        f"Showing the first {min(limit, total)} rows:",
    ]

    for index, row in matches.head(limit).iterrows():
        row_number = index + 1
        values = ", ".join(f"{column}: {_row_get(row, column)}" for column in columns)
        lines.append(f"- Row {row_number}: {values}")

    if total > limit:
        lines.append(
            f"There are {total - limit} more {status} rows in the table below the charts."
        )

    lines.append("The current prototype data is simulated.")
    return "\n".join(lines)


def _metric_from_question(user_question: str):
    question = (user_question or "").lower()
    metrics = [
        ("heart_rate_bpm", "heart rate", "bpm", ["heart rate", "heartrate", "heart"]),
        (
            "blink_rate_per_min",
            "blink rate",
            "blinks/min",
            ["blink rate", "blink", "blinks"],
        ),
        (
            "slouching_time_minutes",
            "slouching time",
            "minutes",
            ["slouching", "slouch"],
        ),
        (
            "final_risk_score",
            "final risk score",
            "",
            ["risk score", "final risk", "risk"],
        ),
        ("temperature_c", "temperature", "C", ["temperature", "temp"]),
    ]

    for column, label, unit, aliases in metrics:
        if any(alias in question for alias in aliases):
            return column, label, unit
    return None


def _question_requests_average(user_question: str) -> bool:
    question = (user_question or "").lower()
    average_terms = ["average", "avg", "mean"]
    return any(term in question for term in average_terms)


def _time_filter_from_question(user_question: str):
    question = (user_question or "").lower()
    before_terms = ["before", "befor", "pre"]
    after_terms = ["after", "post"]

    mentions_lunch = "lunch" in question
    mentions_noon = "noon" in question or "midday" in question
    mentions_morning = "morning" in question
    mentions_afternoon = "afternoon" in question

    if (mentions_lunch or mentions_noon or mentions_morning) and any(
        term in question for term in before_terms
    ):
        return "before", DEFAULT_LUNCH_HOUR, "before lunch/noon"

    if (mentions_lunch or mentions_noon or mentions_afternoon) and any(
        term in question for term in after_terms
    ):
        return "after", DEFAULT_LUNCH_HOUR, "after lunch/noon"

    if mentions_morning:
        return "before", DEFAULT_LUNCH_HOUR, "in the morning"

    if mentions_afternoon:
        return "after", DEFAULT_LUNCH_HOUR, "in the afternoon"

    return None


def _average_metric_answer(user_question: str, df) -> str | None:
    metric = _metric_from_question(user_question)
    if not _question_requests_average(user_question) or metric is None:
        return None

    column, label, unit = metric
    if column not in df.columns:
        return f"The dataset does not include `{column}`, so I cannot calculate that average."

    filtered_df = df
    filter_description = "for the full dataset"
    assumption = ""
    time_filter = _time_filter_from_question(user_question)

    if time_filter is not None:
        if "timestamp" not in df.columns:
            return "The dataset does not include timestamps, so I cannot filter by lunch time."

        direction, hour, filter_description = time_filter
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if direction == "before":
            filtered_df = df[timestamps.dt.hour < hour]
        else:
            filtered_df = df[timestamps.dt.hour >= hour]
        assumption = f" I treated lunch/noon as {hour:02d}:00."

    if filtered_df.empty:
        return (
            f"I could not find any rows {filter_description}, so I cannot calculate "
            f"the average {label}.{assumption}"
        )

    try:
        values = filtered_df[column].dropna().astype(float)
    except (TypeError, ValueError):
        return f"The `{column}` values are not numeric enough to calculate an average."

    if values.empty:
        return f"There are no valid `{column}` values {filter_description}."

    average_value = values.mean()
    unit_text = f" {unit}" if unit else ""
    time_range = ""
    if "timestamp" in filtered_df.columns:
        time_range = (
            f" Time range used: {filtered_df['timestamp'].min()} to "
            f"{filtered_df['timestamp'].max()}."
        )

    return (
        f"The average {label} {filter_description} is {average_value:.2f}{unit_text} "
        f"based on {len(values)} rows.{assumption}{time_range} "
        "The current prototype data is simulated."
    )


def summarize_available_data(df, selected_row=None, fusion_result=None) -> str:
    """Create a compact, safe summary for local LLM chat."""
    lines = [
        "FULL SESSION DATA SUMMARY:",
        f"- Number of rows: {len(df)}",
    ]

    if "timestamp" in df.columns and not df.empty:
        try:
            lines.append(
                f"- Time range: {df['timestamp'].min()} to {df['timestamp'].max()}"
            )
        except (TypeError, ValueError):
            pass

    average_fields = [
        ("Average heart rate", "heart_rate_bpm", "bpm"),
        ("Average blink rate", "blink_rate_per_min", "blinks/min"),
        ("Average slouching time", "slouching_time_minutes", "minutes"),
        ("Average final risk score", "final_risk_score", ""),
    ]
    for label, column, unit in average_fields:
        mean_value = _safe_mean(df, column)
        if mean_value is not None:
            suffix = f" {unit}" if unit else ""
            lines.append(f"- {label}: {mean_value:.2f}{suffix}")

    trend_fields = [
        ("Heart rate", "heart_rate_bpm", "bpm"),
        ("Blink rate", "blink_rate_per_min", "blinks/min"),
        ("Slouching time", "slouching_time_minutes", "minutes"),
        ("Final risk score", "final_risk_score", ""),
    ]
    for label, column, unit in trend_fields:
        first_last = _safe_first_last(df, column)
        min_max = _safe_min_max(df, column)
        if first_last is not None:
            first_value, last_value = first_last
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"- {label} first-to-last: {first_value}{suffix} to {last_value}{suffix}"
            )
        if min_max is not None:
            min_value, max_value = min_max
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"- {label} range: {min_value:.2f}{suffix} to {max_value:.2f}{suffix}"
            )

    if "status" in df.columns:
        try:
            status_counts = df["status"].value_counts(dropna=True).to_dict()
            lines.append(f"- Fusion status counts: {status_counts}")
        except (TypeError, ValueError):
            pass

    if "data_source" in df.columns:
        try:
            sources = sorted(str(value) for value in df["data_source"].dropna().unique())
            lines.append(f"- Data sources: {', '.join(sources)}")
        except (TypeError, ValueError):
            pass

    highest_risk = _highest_risk_summary(df)
    if highest_risk is not None:
        lines.append(f"- Highest-risk row in full session: {highest_risk}")

    if selected_row is not None:
        lines.append("")
        lines.append("CURRENT SELECTED MINUTE ONLY:")
        selected_fields = [
            "timestamp",
            "heart_rate_bpm",
            "baseline_heart_rate_bpm",
            "blink_rate_per_min",
            "posture_status",
            "slouching_time_minutes",
            "temperature_c",
            "data_source",
        ]
        for field in selected_fields:
            value = _row_get(selected_row, field, None)
            if value is not None:
                lines.append(f"  - {field}: {value}")

    if fusion_result:
        lines.append("")
        lines.append("CURRENT SELECTED MINUTE FUSION RESULT:")
        lines.extend(
            [
                f"- Current fusion status: {fusion_result.get('status', 'Unknown')}",
                f"- Current final risk score: {fusion_result.get('final_risk_score', 'Unknown')}",
                f"- Current fusion reasons: {_reason_text(fusion_result)}",
            ]
        )

    lines.append(
        "- Note: the current prototype data is simulated until real sensors are integrated."
    )
    return "\n".join(lines)


def build_chat_prompt(user_question, data_summary) -> str:
    """Build a local chat prompt grounded only in the dashboard data summary."""
    return (
        "You are a local ergonomic assistant for a student monitoring dashboard. "
        "Answer only using the available data summary below. Do not invent sensor "
        "values. Be honest that the current data is simulated or incomplete when "
        "that is relevant. Do not make medical claims or diagnoses. Keep the answer "
        "short, practical, and student-friendly. Important: use the FULL SESSION "
        "DATA SUMMARY for questions about trends, changes over time, the session, "
        "general performance, lists of rows, or status counts. Do not answer from "
        "the selected minute unless the user explicitly asks about now, current, "
        "selected, or this row.\n\n"
        f"{data_summary}\n\n"
        f"User question: {user_question}"
    )


def _fallback_chat_answer(fusion_result=None) -> str:
    if fusion_result:
        status = fusion_result.get("status", "Unknown")
        score = fusion_result.get("final_risk_score", "Unknown")
        reasons = _reason_text(fusion_result) or "the available sensor readings"
        return (
            "The local LLM is not available right now. Based on the current fusion "
            f"result, the status is {status} with a risk score of {score}. "
            f"The main reasons are: {reasons}. The data is simulated in this prototype."
        )

    return (
        "The local LLM is not available right now. Based on the current fusion result, "
        "review the dashboard values and follow the system alert guidance. The data is "
        "simulated in this prototype."
    )


def answer_data_question(
    user_question,
    df,
    selected_row=None,
    fusion_result=None,
    use_local_llm=True,
    model=DEFAULT_OLLAMA_MODEL,
    ollama_host=DEFAULT_OLLAMA_HOST,
) -> str:
    """Answer dashboard questions using only local data and local Ollama."""
    calculated_answer = _average_metric_answer(user_question, df)
    if calculated_answer is not None:
        return calculated_answer

    if _question_requests_rows(user_question):
        return _format_status_rows(df, _requested_status(user_question))

    include_selected_context = _question_mentions_current_row(user_question)

    if not use_local_llm:
        if include_selected_context:
            return _fallback_chat_answer(fusion_result)
        return summarize_available_data(df)

    try:
        data_summary = summarize_available_data(
            df,
            selected_row if include_selected_context else None,
            fusion_result if include_selected_context else None,
        )
        prompt = build_chat_prompt(user_question, data_summary)
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about a local ergonomics dashboard. "
                    "Use only supplied data. Do not diagnose."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response_text = call_ollama_chat(
            messages,
            model=model,
            ollama_host=ollama_host,
        )
    except Exception:
        return _fallback_chat_answer(fusion_result)

    if response_text:
        return response_text
    return _fallback_chat_answer(fusion_result)
