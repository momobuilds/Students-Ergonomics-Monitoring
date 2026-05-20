import time
import json
import numpy as np

from collections import deque
from scipy.signal import find_peaks
from cms50d import CMS50D

def get_baseline():
    with open("hr_result.json", "r") as f:
        result = json.load(f)
        return result


def HR_Variation(buffer, baseline):
    avg_HR = np.mean(buffer)
    print(avg_HR)
    index = (avg_HR - baseline)*2/ baseline
    print("HR index: " + str(index))
    return index

def HRV_Variation(buffer, baseline):
    if len(buffer) >= 3:
        rr_arr = np.array(buffer)
        avg_rmssd = float(np.sqrt(np.mean(np.diff(rr_arr) ** 2)))
        print(avg_rmssd)
        index = 2 * baseline / avg_rmssd
        print("HRV index: " + str(index))
        return index
    else:
        print("no data")
        return 0.0

def get_stress_level(stress_index):
    if stress_index < 0.5:
        return "Relaxed"
    elif stress_index < 1:
        return "Slightly stressed"
    elif stress_index < 1.5:
        return "Stressed"
    elif stress_index < 2.0:
        return "Critically stressed"
    else:
        return "Extreme stress"  # edge case, good to handle explicitly



def HR_Measurement(port="COM3", duration = 0):

    monitor = CMS50D(port)
    monitor.connect()
    monitor.start_live_acquisition()

    target_fps = 30
    window_time = 10
    k=0.3 #Weight of HR to estimate the stress index. the Weight of HRV is (1-k)
    result = get_baseline()
    baseline_hr = result["HR_avg"]
    baseline_rmssd = result["RMSSD_avg"]

    min_hr_bpm = 40
    min_beat_distance = int(target_fps * 60 / min_hr_bpm)

    hr_intervals = deque(maxlen=50)
    wave_buffer = deque(maxlen=300)
    time_buffer = deque(maxlen=300)
    rr_intervals = deque(maxlen=50)

    start_time = time.perf_counter()
    window_start = start_time

    last_logged_hr = None
    last_peak_time = None



    try:
        while True:
            while True:
                data = monitor.get_latest_data()
                if data is None:
                    break

                now = data["timestamp"]
                waveform = data["waveform"]

                if waveform is not None:
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
                                if 0.3 < rr_sec < 2.0:
                                    rr_intervals.append(rr_sec)
                                last_peak_time = t_prev

                hr = data.get("pulse_rate")

                hr_intervals.append(hr)

            current_time = time.perf_counter()
            elpased_window_time = current_time - window_start
            if elpased_window_time > window_time:
                hr_index = HR_Variation(hr_intervals, baseline_hr)
                hrv_index = HRV_Variation(rr_intervals, baseline_rmssd)
                stress_index = k*hr_index + (1-k)*hrv_index
                print("Stress index: " + str(stress_index))
                print(get_stress_level(stress_index))
                window_start = current_time

            exec_time = current_time - start_time
            if exec_time >= duration > 0:
                break
    finally:
        monitor.stop_live_acquisition()
        monitor.disconnect()