import threading
import queue
import time
import datetime
import serial

try:
    import hid
except ImportError:
    hid = None


class CMS50D:
    def __init__(self, port, baudrate=115200, timeout=1, sample_rate_hz=60.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.realtime_streaming = False
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        self.data_queue = queue.Queue(maxsize=50)
        self.thread = None

        self.mode = "hid" if str(port).upper() == "HID" else "serial"

        # HID configuration
        self.vendor_id = 0x28E9
        self.product_id = 0x028A
        self.last_hr = None
        self.last_spo2 = None

        # Fixed-rate timestamp generation for HID waveform samples
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_period = datetime.timedelta(seconds=1.0 / self.sample_rate_hz)
        self.sample_timestamp = None

    def connect(self):
        if self.mode == "serial":
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                xonxoff=1
            )
        else:
            self._connect_hid()

    def disconnect(self):
        if self.mode == "serial":
            if self.connection and self.connection.is_open:
                self.connection.close()
        else:
            if self.connection is not None:
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.connection = None

    def send_command(self, command):
        def encode_package(cmd):
            package_type = 0x7D
            data = [cmd] + [0x00] * 6
            high_byte = 0x80
            for i in range(len(data)):
                high_byte |= (data[i] & 0x80) >> (7 - i)
                data[i] |= 0x80  # Set sync bit
            package_type &= 0x7F  # Clear sync bit
            return [package_type, high_byte] + data

        package = encode_package(command)
        self.connection.write(bytes(package))
        self.connection.flush()

    def send_keepalive(self):
        now = datetime.datetime.now()
        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xAF)  # keepalive
            self.keepalive_timestamp = now

    def start_live_acquisition(self):
        if self.mode == "serial":
            self.connection.reset_input_buffer()
            self.send_command(0xA1)  # Start real-time data
            self.realtime_streaming = True

            self.thread = threading.Thread(target=self._collect_data)
            self.thread.daemon = True
            self.thread.start()
        else:
            self.realtime_streaming = True
            self.sample_timestamp = datetime.datetime.now()

            self.thread = threading.Thread(target=self._collect_data_hid)
            self.thread.daemon = True
            self.thread.start()

    def stop_live_acquisition(self):
        self.realtime_streaming = False

        if self.mode == "serial":
            try:
                self.send_command(0xA2)  # Stop real-time data
            except Exception:
                pass

    def _collect_data(self):
        while self.realtime_streaming:
            packet = self._read_packet()
            if packet:
                package_type, data = self._decode_packet(packet)
                if package_type == 0x01 and len(data) == 7:
                    signal_strength = data[0] & 0x0F
                    pulse_beep = (data[0] & 0x40) >> 6
                    probe_error = (data[0] & 0x80) >> 7
                    pulse_waveform = data[1] & 0x7F
                    pulse_rate = data[3]
                    spO2 = data[4]

                    if not self.data_queue.full():
                        self.data_queue.put({
                            "timestamp": datetime.datetime.now(),
                            "pulse_rate": None if pulse_rate == 0xFF else pulse_rate,
                            "spO2": None if spO2 == 0x7F else spO2,
                            "waveform": pulse_waveform,
                            "signal_strength": signal_strength,
                            "pulse_beep": pulse_beep,
                            "probe_error": probe_error,
                            "mode": "serial",
                        })

    def _read_packet(self):
        while True:
            self.send_keepalive()
            byte = self.connection.read()
            if not byte:
                print("Serial timeout or no data received.")
                return None
            if not byte:
                return None
            if not (byte[0] & 0x80):
                packet = byte + self.connection.read(8)
                if len(packet) == 9:
                    return list(packet)

    def _decode_packet(self, packet):
        package_type = packet[0]
        high_byte = packet[1]
        data = list(packet[2:])
        for i in range(len(data)):
            data[i] = (data[i] & 0x7F) | ((high_byte << (7 - i)) & 0x80)
        return package_type, data

    def get_latest_data(self):
        """
        Fetch the latest data from the queue.
        Returns None if no data is available.
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    # =========================
    # HID-only part
    # =========================

    def _connect_hid(self):
        if hid is None:
            raise ImportError("The 'hid' module is not installed. Install it with: pip install hidapi")

        self.connection = hid.device()
        self.connection.open(self.vendor_id, self.product_id)
        self.connection.set_nonblocking(False)

    def _collect_data_hid(self):
        while self.realtime_streaming:
            try:
                raw = self.connection.read(64, timeout_ms=int(self.timeout * 1000))
                if not raw:
                    continue

                samples = self._parse_hid_packet(raw)
                for sample in samples:
                    if not self.data_queue.full():
                        self.data_queue.put(sample)

            except Exception as e:
                print(f"HID read error: {e}")
                time.sleep(0.2)

    def _parse_hid_packet(self, raw):
        """
        Parse 64-byte HID packets made of 6-byte records.

        Observed record format:
        [235, record_type, status, value1, value2, value3]

        record_type == 0 -> waveform record, waveform = value1
        record_type == 1 -> vital record, HR = value1, SpO2 = value2

        HID waveform timestamps are generated assuming a fixed sample rate.
        """
        samples = []

        for i in range(0, len(raw) - 5, 6):
            chunk = raw[i:i + 6]

            if len(chunk) != 6:
                continue

            if chunk[0] != 235:
                continue

            rec_type = chunk[1]

            if rec_type == 0:
                waveform = chunk[3]

                if self.sample_timestamp is None:
                    self.sample_timestamp = datetime.datetime.now()

                samples.append({
                    "timestamp": self.sample_timestamp,
                    "pulse_rate": self.last_hr,
                    "spO2": self.last_spo2,
                    "waveform": waveform,
                    "signal_strength": None,
                    "pulse_beep": None,
                    "probe_error": None,
                    "mode": "hid",
                    "raw_chunk": chunk,
                })

                self.sample_timestamp += self.sample_period

            elif rec_type == 1:
                self.last_hr = chunk[3]
                self.last_spo2 = chunk[4]

        return samples