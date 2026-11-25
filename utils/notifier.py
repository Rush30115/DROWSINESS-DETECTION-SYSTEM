import os
import threading
import time
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()


class TwilioNotifier:
    """Sends SMS alerts via Twilio with cooldown rate-limiting."""

    REQUIRED_VARS = (
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "TWILIO_TO_NUMBER",
    )

    def __init__(self, cooldown_seconds=None, alert_after=None):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
        self.to_number = os.getenv("TWILIO_TO_NUMBER", "").strip()
        self.message_template = os.getenv("TWILIO_MESSAGE_BODY", "").strip()
        env_cooldown = os.getenv("TWILIO_COOLDOWN_SECONDS", "60").strip()
        self.cooldown_seconds = float(cooldown_seconds if cooldown_seconds is not None else env_cooldown or 60)
        env_alert_after = os.getenv("TWILIO_ALERT_AFTER", "5").strip()
        self.alert_after = max(1, int(alert_after if alert_after is not None else env_alert_after or 5))
        self._lock = threading.Lock()
        self._last_sent = 0.0
        self._was_drowsy = False
        self.detection_count = 0

    def missing_fields(self):
        mapping = {
            "TWILIO_ACCOUNT_SID": self.account_sid,
            "TWILIO_AUTH_TOKEN": self.auth_token,
            "TWILIO_FROM_NUMBER": self.from_number,
            "TWILIO_TO_NUMBER": self.to_number,
        }
        return [k for k, v in mapping.items() if not v]

    def is_configured(self):
        return len(self.missing_fields()) == 0

    def _send_sms(self, body):
        client = Client(self.account_sid, self.auth_token)
        msg = client.messages.create(
            body=body,
            from_=self.from_number,
            to=self.to_number,
        )
        return msg.sid

    def send_drowsiness_alert(self, ear_value, on_result=None):
        """Send an alert SMS in a background thread if configured and outside cooldown."""
        if not self.is_configured():
            return False

        now = time.monotonic()
        with self._lock:
            if self._last_sent > 0 and (now - self._last_sent) < self.cooldown_seconds:
                return False
            self._last_sent = now

        body = self.message_template or f"ALERT: Driver drowsiness detected! Current EAR: {ear_value:.3f}"

        def _worker():
            try:
                sid = self._send_sms(body)
                if on_result:
                    on_result(True, f"Twilio SMS sent (SID: {sid})")
            except Exception as exc:
                with self._lock:
                    self._last_sent = 0.0
                if on_result:
                    on_result(False, f"Twilio SMS failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def handle_detection(self, is_drowsy, ear_value, on_result=None):
        """Count distinct drowsiness episodes and send SMS after `alert_after` detections."""
        should_send = False
        with self._lock:
            if is_drowsy and not self._was_drowsy:
                self.detection_count += 1
                if self.detection_count >= self.alert_after:
                    should_send = True
                    self.detection_count = 0
            self._was_drowsy = bool(is_drowsy)

        if should_send:
            return self.send_drowsiness_alert(ear_value, on_result=on_result)
        return False

    def send_test_message(self):
        """Send a synchronous test SMS and return (ok, detail)."""
        missing = self.missing_fields()
        if missing:
            return False, f"Missing Twilio config: {', '.join(missing)}"
        try:
            body = self.message_template or "Test alert from Driver Sleep Alert System."
            sid = self._send_sms(body)
            return True, f"Test SMS sent (SID: {sid})"
        except Exception as exc:
            return False, f"Twilio error: {exc}"
