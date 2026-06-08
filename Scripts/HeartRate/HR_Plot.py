import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def simulate_hr_realtime(
    hr_values: list[float],
    sample_rate: int = 250,
    window_sec: float = 6,
    playback_speed: float = 1.0
) -> None:
    def ecg_beat(n_samples):
        t = np.linspace(0, 1, n_samples)
        # R peak — slightly wider sigma so it's less needle-like
        r = 1.0  * np.exp(-((t - 0.45) ** 2) / (2 * 0.018 ** 2))
        # S dip
        s = -0.15 * np.exp(-((t - 0.52) ** 2) / (2 * 0.010 ** 2))
        # T wave
        tw = 0.25 * np.exp(-((t - 0.72) ** 2) / (2 * 0.045 ** 2))
        return r + s + tw

    rng = np.random.default_rng(42)
    segments = []

    for bpm in hr_values:
        minute_samples = 60 * sample_rate
        base_rr = sample_rate * 60 / bpm

        beat_signal = np.zeros(minute_samples)
        pos = 0
        while pos < minute_samples:
            rr = int(base_rr * rng.uniform(0.97, 1.03))
            if pos + rr > minute_samples:
                rr = minute_samples - pos
            beat = ecg_beat(rr)
            beat_signal[pos:pos + rr] = beat
            pos += rr

        segments.append(beat_signal)

    signal = np.concatenate(segments)

    window_samples = int(window_sec * sample_rate)
    time_axis = np.linspace(0, window_sec, window_samples)

    fig, ax = plt.subplots(figsize=(12, 4))
    line, = ax.plot([], [], linewidth=1.2)
    bpm_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=13, fontweight="bold")
    min_text = ax.text(0.98, 0.90, "", transform=ax.transAxes, fontsize=10, ha="right", color="gray")

    ax.set_xlim(0, window_sec)
    ax.set_ylim(-0.4, 1.3)
    ax.set_xlabel("Time (s, rolling window)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live Heart Rate Simulation")

    interval_ms = max(1, int(1000 / (sample_rate * playback_speed)))
    step = max(1, int(sample_rate * playback_speed / 60))
    frame_idx = [0]

    def update(_):
        idx = frame_idx[0]
        frame_idx[0] = min(idx + step, len(signal) - 1)
        end_idx = frame_idx[0] + 1

        start_idx = max(0, end_idx - window_samples)
        chunk = signal[start_idx:end_idx]
        pad = window_samples - len(chunk)
        y_win = np.concatenate([np.full(pad, np.nan), chunk])

        line.set_data(time_axis, y_win)

        current_minute = min(idx // (60 * sample_rate), len(hr_values) - 1)
        bpm_text.set_text(f"♥  {hr_values[current_minute]:.0f} bpm")
        min_text.set_text(f"min {current_minute + 1} / {len(hr_values)}")

        if frame_idx[0] >= len(signal) - 1:
            ani.event_source.stop()

        return line, bpm_text, min_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(signal) // step,
        interval=interval_ms,
        blit=True,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

simulate_hr_realtime([65, 72])