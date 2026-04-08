"""
gate_controller.py — HTTP commands to the CAM board servo only.

v3 — matches firmware v7.2 centralized architecture:
  - Gate board controlled by backend via MQTT
  - Python only mirrors state to cam board servo via HTTP /open /close
  - Keep-alive thread sends /keep to cam board every 30s
  - Departure polling removed — gate board publishes MQTT DEPARTED
"""

import urllib.request
import urllib.error
import threading
import time
from typing import Optional

_GATE_URL: str = ""
_CAM_URL:  str = ""

_NOTIFY_TIMEOUT = 2.0
_HTTP_TIMEOUT   = 3.0
_KEEP_INTERVAL  = 30.0

_keep_thread: Optional[threading.Thread] = None
_keep_running = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def set_gate_url(url: str) -> None:
    global _GATE_URL
    _GATE_URL = url.rstrip("/")
    print(f"[GateController] Gate board URL: {_GATE_URL} (MQTT-controlled)")


def set_cam_url(url: str) -> None:
    global _CAM_URL
    _CAM_URL = url.rstrip("/")
    print(f"[GateController] Cam board URL: {_CAM_URL}")


def set_departure_callback(fn) -> None:
    """No-op — departure is MQTT-driven via backend."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _post(url: str, timeout: float = _HTTP_TIMEOUT) -> bool:
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.getcode() < 300
    except urllib.error.URLError as exc:
        print(f"[GateController] POST {url} → unreachable: {exc.reason}")
        return False
    except Exception as exc:
        print(f"[GateController] POST {url} → error: {exc}")
        return False


def _cmd_gate(endpoint: str) -> bool:
    """Gate board is MQTT-controlled — no HTTP call."""
    print(f"[GateController] Gate {endpoint} → skipped (backend sends MQTT)")
    return True


def _cmd_cam(endpoint: str) -> bool:
    """POST an endpoint on the cam board servo."""
    if not _CAM_URL:
        return False
    ok = _post(f"{_CAM_URL}{endpoint}", timeout=_NOTIFY_TIMEOUT)
    print(f"[GateController] Cam {endpoint} → {'OK' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# KEEP-ALIVE
# ═══════════════════════════════════════════════════════════════════════════

def _keep_alive_loop():
    global _keep_running
    while _keep_running:
        time.sleep(_KEEP_INTERVAL)
        if _CAM_URL:
            _post(f"{_CAM_URL}/keep", timeout=2.0)


def start_keep_alive() -> None:
    global _keep_thread, _keep_running
    if _keep_thread is not None and _keep_thread.is_alive():
        return
    _keep_running = True
    _keep_thread = threading.Thread(target=_keep_alive_loop, daemon=True,
                                    name="CamKeepAlive")
    _keep_thread.start()
    print(f"[GateController] Keep-alive thread started → every {_KEEP_INTERVAL:.0f}s")


def stop_keep_alive() -> None:
    global _keep_running
    _keep_running = False


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY / EXIT ACK HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

def notify_entry_ack(ack: dict) -> None:
    status = ack.get("status", "")
    reg    = ack.get("registration", "?")
    slot   = ack.get("slotId", "?")
    if status == "ALLOWED":
        print(f"[GateController] ENTRY ALLOWED: {reg}  slot={slot}  → cam servo open")
        _cmd_cam("/open")
    else:
        reason = ack.get("reason", "")
        print(f"[GateController] ENTRY {status}: {reg}  reason={reason}  → cam servo close")
        _cmd_cam("/close")


def notify_exit_ack(ack: dict) -> None:
    status = ack.get("status", "")
    reg    = ack.get("registration", "?")
    fee    = ack.get("totalFee", "?")
    mins   = ack.get("durationMinutes", "?")
    if status == "OK":
        print(f"[GateController] EXIT OK: {reg}  fee={fee}  {mins}min → cam servo open")
        _cmd_cam("/open")
    else:
        print(f"[GateController] EXIT {status}: {reg}  → no cam action")


# ═══════════════════════════════════════════════════════════════════════════
# DEPARTURE POLLING — removed, gate board publishes MQTT DEPARTED
# ═══════════════════════════════════════════════════════════════════════════

def start_departure_polling(cam_id: str, publish_exit_fn=None,
                            rmq_channel_getter=None) -> None:
    print(f"[GateController] Departure polling disabled — "
          f"gate board publishes MQTT DEPARTED for {cam_id}")


def stop_departure_polling() -> None:
    stop_keep_alive()