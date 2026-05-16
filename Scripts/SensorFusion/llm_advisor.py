"""Local LLM advisor interface for explaining fusion results.

The LLM layer does not calculate risk. It only explains the deterministic
sensor fusion output in student-friendly language.
"""

from __future__ import annotations


def _reason_text(fusion_result: dict) -> str:
    reasons = fusion_result.get("reasons") or []
    if isinstance(reasons, str):
        return reasons
    return ", ".join(reasons)


def build_llm_prompt(fusion_result, sensor_row) -> str:
    """Build a prompt that can later be sent to a local LLM such as Ollama."""
    reasons = _reason_text(fusion_result) or "no major risk signals"
    return (
        "You are an ergonomic assistant for students. Explain the sensor fusion "
        "result clearly and give brief, practical advice. Do not recalculate the "
        "risk score.\n\n"
        f"Fusion status: {fusion_result.get('status', 'Unknown')}\n"
        f"Final risk score: {fusion_result.get('final_risk_score', 'Unknown')}\n"
        f"Fusion reasons: {reasons}\n"
        f"Heart rate: {sensor_row.get('heart_rate_bpm', 'Unknown')} bpm\n"
        f"Baseline heart rate: {sensor_row.get('baseline_heart_rate_bpm', 'Unknown')} bpm\n"
        f"Blink rate: {sensor_row.get('blink_rate_per_min', 'Unknown')} blinks/min\n"
        f"Posture: {sensor_row.get('posture_status', 'Unknown')}\n"
        f"Slouching time: {sensor_row.get('slouching_time_minutes', 'Unknown')} minutes\n"
        f"Temperature: {sensor_row.get('temperature_c', 'Unknown')} C\n\n"
        "Return an alert level, one explanation sentence, and one recommendation."
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


def generate_advice_with_local_llm(prompt):
    """Placeholder for a local-only LLM integration.

    Later, this function can call Ollama, llama.cpp, or another local model
    running on the same machine. It intentionally does not call any cloud API.
    """
    # Example future integration point:
    # response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    # return parse_local_llm_response(response)
    return None


def generate_advice(fusion_result, sensor_row, use_local_llm=False) -> dict:
    """Return a student-friendly explanation and recommendation."""
    if use_local_llm:
        prompt = build_llm_prompt(fusion_result, sensor_row)
        local_response = generate_advice_with_local_llm(prompt)
        if isinstance(local_response, dict):
            return local_response

    return generate_mock_advice(fusion_result, sensor_row)
