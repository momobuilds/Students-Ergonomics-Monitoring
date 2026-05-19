import time
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import deque
from scipy.signal import find_peaks
from cms50d import CMS50D

def HR_Calibration(port="COM3", duration = 60):
    target_fps = 30
    monitor = CMS50D(port)
    monitor.connect()
    monitor.start_live_acquisition()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))

    window_seconds = 10
    frame_interval = 1.0 / target_fps

    xdata = deque()
    ydata = deque()

    line, = ax.plot([], [], '-m', label='Pulse Waveform')

    text_hr = ax.text(0.02, 0.95, 'HR: -- bpm', transform=ax.transAxes, color='red')

    ax.set_ylim(0, 128)
    ax.set_ylabel("Waveform")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.title("CMS50D Live Data")
    plt.legend()

    last_hr = None
    last_draw_time = 0.0
    min_hr_bpm = 40
    min_beat_distance = int(target_fps * 60 / min_hr_bpm)

    hr_values = []
    wave_buffer = deque(maxlen=300)
    time_buffer = deque(maxlen=300)
    rr_intervals = deque(maxlen=50)

    start_time = time.perf_counter()

    last_logged_hr = None
    last_peak_time = None

    try:
        while True:

            got_data = False
            while True:
                data = monitor.get_latest_data()
                if data is None:
                    break

                got_data = True
                now = data["timestamp"]
                waveform = data["waveform"]

                if waveform is not None:
                    xdata.append(now)
                    ydata.append(waveform)

                    wave_buffer.append(waveform)
                    time_buffer.append(now)

                    if len(wave_buffer) >= 50:
                        signal = np.array(list(wave_buffer))
                        times = list(time_buffer)

                        peaks, _ = find_peaks(
                            signal,
                            distance=min_beat_distance,
                            height=signal.mean() * 0.7,
                            prominence=signal.std() * 0.5
                        )

                        if len(peaks) >= 2:
                            for i in range(1, len(peaks)):
                                t_prev = times[peaks[i - 1]]
                                t_curr = times[peaks[i]]
                                if last_peak_time is not None and t_prev <= last_peak_time:
                                    continue  # already processed this pair
                                rr_sec = (t_curr - t_prev).total_seconds()
                                print("Beat-to-beat interval: " + str(rr_sec))
                                if 0.3 < rr_sec < 2.0:
                                    rr_intervals.append(rr_sec)
                                last_peak_time = t_prev

                hr = data.get("pulse_rate")

                if hr is not None:
                    last_hr = hr

                    if hr != last_logged_hr:
                        hr_values.append(hr)
                        last_logged_hr = hr

            if xdata:
                cutoff = xdata[-1] - datetime.timedelta(seconds=window_seconds)
                while xdata and xdata[0] < cutoff:
                    xdata.popleft()
                    ydata.popleft()

            current_time = time.perf_counter()
            if current_time - last_draw_time >= frame_interval and xdata:

                line.set_data(xdata, ydata)

                ax.set_xlim(xdata[0], xdata[-1])

                text_hr.set_text(f"HR: {last_hr if last_hr else '--'} bpm")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                last_draw_time = current_time

            plt.pause(0.001 if got_data else 0.005)

            exec_time = current_time - start_time
            if exec_time >= duration:
                break

    finally:
        monitor.stop_live_acquisition()
        monitor.disconnect()
        plt.close(fig)

        avg_hr = sum(hr_values) / len(hr_values) if hr_values else None

        if len(rr_intervals) >= 3:
            rr_arr = np.array(rr_intervals)
            avg_rmssd = float(np.sqrt(np.median(np.diff(rr_arr) ** 2)))
        else:
            avg_rmssd = None
        print(f"Calibration completed. Mean HR: {avg_hr:.2f} bpm; Mean RMSSD: {avg_rmssd:.2f}")

        result = {
            "HR_avg": round(avg_hr, 2) if avg_hr is not None else None,
            "RMSSD_avg": round(avg_rmssd, 2) if avg_rmssd is not None else None
        }

        with open("hr_result.json", "w") as f:
            json.dump(result, f, indent=4)

