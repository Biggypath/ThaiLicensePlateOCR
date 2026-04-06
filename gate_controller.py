"""
gate_controller.py — Send gate commands to ESP32 via HTTP.

The ESP32-CAM should expose endpoints:
  GET /gate?action=open    → open the barrier
  GET /gate?action=close   → close the barrier (or let it auto-close)

This module is called from the RabbitMQ ACK consumer thread
when the backend replies with ALLOWED / REJECTED / OK / ERROR.
"""

import os
import urllib.request
import urllib.error
import urllib.parse

# Derive the ESP32 gate base URL from ESP32_URL (same IP, port 80)
# or allow an explicit override via ESP32_GATE_URL.
_GATE_URL: str | None = None


def _get_gate_url() -> str:
    global _GATE_URL
    if _GATE_URL is not None:
        return _GATE_URL

    explicit = os.getenv("ESP32_GATE_URL")
    if explicit:
        _GATE_URL = explicit.rstrip("/")
        return _GATE_URL

    # Fallback: extract IP from ESP32_URL (the MJPEG stream URL)
    stream_url = os.getenv("ESP32_URL", "http://172.20.10.4:81/stream")
    parsed = urllib.parse.urlparse(stream_url)
    _GATE_URL = f"http://{parsed.hostname}"
    return _GATE_URL


def send_gate_command(action: str, timeout: float = 3.0) -> bool:
    """
    Send a gate command to the ESP32.

    action: "open" or "close"
    Returns True if the ESP32 responded with 2xx, False otherwise.
    """
    base = _get_gate_url()
    url = f"{base}/gate?action={urllib.parse.quote(action)}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            print(f"[GATE] {action.upper()} → ESP32 responded {status}")
            return 200 <= status < 300
    except urllib.error.URLError as exc:
        print(f"[GATE] {action.upper()} → ESP32 unreachable: {exc.reason}")
        return False
    except Exception as exc:
        print(f"[GATE] {action.upper()} → error: {exc}")
        return False


def notify_entry_ack(ack: dict) -> None:
    """React to an entry ACK from the backend."""
    status = ack.get("status", "")
    if status == "ALLOWED":
        send_gate_command("open")
    else:
        send_gate_command("close")


def notify_exit_ack(ack: dict) -> None:
    """React to an exit ACK from the backend."""
    status = ack.get("status", "")
    if status == "OK":
        send_gate_command("open")
    else:
        send_gate_command("close")
