import sys
import threading
import time

if sys.platform == "win32":
    import winsound
else:
    winsound = None
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        np = None
        sd = None


def _beep_once(freq=2500, duration_ms=800):
    if winsound is not None:
        winsound.Beep(freq, duration_ms)
        return

    if sd is not None and np is not None:
        sample_rate = 44100
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(wave, sample_rate)
        sd.wait()
        return

    print("\a", end="", flush=True)
    time.sleep(duration_ms / 1000.0)


class Alarm:
    def __init__(self):
        self.alarm_active = False
        self.thread = None

    def play_alarm_loop(self):
        while self.alarm_active:
            try:
                _beep_once(2500, 800)  # frequency=2500 Hz, duration=800 ms
                time.sleep(0.2)  # small gap between beeps
            except Exception as e:
                print(f"Alarm error: {e}")
                time.sleep(1)

    def start(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.thread = threading.Thread(target=self.play_alarm_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.alarm_active = False
        if self.thread:
            self.thread.join(timeout=0.1)
            self.thread = None
