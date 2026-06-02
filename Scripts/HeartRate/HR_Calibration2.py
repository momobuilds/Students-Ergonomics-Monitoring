import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from scipy.signal import find_peaks
from cms50d import CMS50D

def get_HR(hr_values):
    return sum(hr_values) / len(hr_values) if hr_values else None

def get_HRV(rr_intervals):
    if len(rr_intervals) >= 3:
        rr_arr = np.array(rr_intervals)
        avg_rmssd = float(np.sqrt(np.median(np.diff(rr_arr) ** 2)))
    else:
        avg_rmssd = None
    return avg_rmssd


def HR_Calibration2(port="COM3", duration = 600):
    #Variable initialization
    target_fps = 60
    window_seconds = 60
    hop_seconds = 10
    hr_max = 150
    min_beat_distance = int(target_fps * 60 / hr_max)
    print("_______________________")
    print(min_beat_distance)
    last_peak_time = None
    hop_time = time.perf_counter()
    start_hop = False

    #Arrays
    hr_values = deque()
    rr_intervals = deque(maxlen=50)
    timestamp = deque()
    amplitude = deque()
    wave_buffer = deque(maxlen=300) #This is used for the HRV
    time_buffer = deque(maxlen=300) #This is used for the HRV
    peaks = []

    #Results
    HR = []
    RMSSD = []

    #Start of the measurement
    monitor = CMS50D(port)
    monitor.connect()
    monitor.start_live_acquisition()

    start_time = time.perf_counter()

    #Plotting test
    plt.ion()

    fig, ax = plt.subplots()

    line_signal, = ax.plot([], [], label="Signal")
    line_peaks, = ax.plot([], [], "rx", label="Peaks")

    ax.legend()

    print("Starting HR calibration...")
    try:
        #While we measure
        while True:

            #Every time we receive a new data
            while True:
                data = monitor.get_latest_data()
                if data is None:
                    break


                now = data["timestamp"]
                waveform = data["waveform"]

                # Get and set the data from the sensor
                if waveform is not None:
                    timestamp.append(now)
                    amplitude.append(waveform)
                    wave_buffer.append(waveform)
                    time_buffer.append(now)

                    #Estimate the time between peaks (rr_interval)
                    if len(wave_buffer) >= 50:
                        signal = np.array(list(wave_buffer))
                        times = list(time_buffer)

                        #Finding the peaks in the current wavefrom
                        peaks, _ = find_peaks(
                            signal,
                            distance=min_beat_distance,
                            height=signal.mean() * 0.7,
                            prominence=signal.std() * 0.4
                        )


                        #If more than one peak found, for each consecutive pair we calculate the time between them
                        if len(peaks) >= 2:
                            for i in range(1, len(peaks)):
                                t_prev = times[peaks[i - 1]]
                                t_curr = times[peaks[i]]
                                if last_peak_time is not None and t_prev <= last_peak_time:
                                    continue  # already processed this pair, ignored
                                rr_sec = (t_curr - t_prev).total_seconds()
                                if 0.3 < rr_sec < 1.6 :
                                    print("Beat-to-beat interval: " + str(rr_sec))
                                    rr_intervals.append(rr_sec)
                                last_peak_time = t_prev

                #We get and set the HR
                hr = data.get("pulse_rate")
                if hr is not None:
                    hr_values.append(hr)


            #Testing plot
            signal = np.array(list(wave_buffer))
            x = np.arange(len(signal))

            line_signal.set_data(x, signal)
            line_peaks.set_data(peaks, signal[peaks])

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            #Keeping the data just in the window size
            if timestamp:
                cutoff = timestamp[-1] - datetime.timedelta(seconds=window_seconds)
                while timestamp and timestamp[0] < cutoff:
                    start_hop = True
                    timestamp.popleft()
                    amplitude.popleft()
                    hr_values.popleft()

            current_time = time.perf_counter()

            #If
            if start_hop:
                if current_time - hop_time >= hop_seconds:
                    print("Starting hop")
                    print(len(rr_intervals))
                    print(get_HR(hr_values))
                    print(get_HRV(rr_intervals))
                    HR.append(get_HR(hr_values))
                    RMSSD.append(get_HRV(rr_intervals))
                    hop_time = current_time

            #Finish if the time exceed the duration
            exec_time = current_time - start_time
            if exec_time >= duration:
                break

    finally:
        monitor.stop_live_acquisition()
        monitor.disconnect()
        avg_hr = np.mean(HR)
        avg_rmssd = np.mean(RMSSD)

        print(f"Calibration completed. Mean HR: {avg_hr:.2f} bpm; Mean RMSSD: {avg_rmssd:.2f}")

        df = pd.DataFrame({"HR": HR, "RMSSD": RMSSD})
        df.to_csv("hr_result.csv", index=False)

