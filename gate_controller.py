"""
gate_controller.py — Send gate/cam commands to ESP32 boards via HTTP.

FLOW SUMMARY
─────────────
ENTRY:
  Backend ACK ALLOWED → notify_entry_ack() →
    POST /open on gate board (arm DOWN, slot open)
    POST /open on cam board  (cam servo DOWN, mirrors gate)

DURING PARKING (handled entirely by gate board ultrasonic):
  Gate board watches ≤ 50 cm → carParked.
  Arm stays DOWN while car is there.
  Car leaves → gate board starts 60-second timer.
    Car returns within 60 s → timer reset, stays DOWN.
    After 60 s clear        → gate board POSTs /notify_clear to cam board.
                               Cam board sets notifyClearPending = true.

EXIT (triggered by Python polling cam board /status):
  Python sees notify_clear_pending = true →
    publish_exit_event() to RabbitMQ.
    POST /exit_sent to cam board (clears pending flag).
  Backend ACK OK → notify_exit_ack() →
    POST /open on gate board (arm DOWN briefly for car to exit).
    POST /open on cam board  (mirrors).

REJECTED:
  notify_entry_ack() status != ALLOWED →
    POST /close on both boards (keep/raise arm).
"""

import urllib.request
import urllib.error
import urllib.parse
import threading
import time
from typing import Optional

_GATE_URL: str = ""
_CAM_URL: str  = ""

_HTTP_TIMEOUT = 3.0
_NOTIFY_TIMEOUT = 2.0

# How often Python polls cam board for departure signal (seconds)
_POLL_INTERVAL = 5.0

# Callback set by main.py — called when cam board reports car departed.
# Signature: on_departure(cam_id: str) → None
_on_departure_callback = None

_poll_thread: Optional[threading.Thread] = None
_poll_running = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def set_gate_url(url: str) -> None:
    """Set the gate board (motor+ultrasonic) base URL."""
    global _GATE_URL
    _GATE_URL = url.rstrip("/")
    print(f"[GateController] Gate board URL: {_GATE_URL}")


def set_cam_url(url: str) -> None:
    """Set the cam board base URL (for polling notify_clear_pending)."""
    global _CAM_URL
    _CAM_URL = url.rstrip("/")
    print(f"[GateController] Cam board URL: {_CAM_URL}")


def set_departure_callback(fn) -> None:
    """Register a callback called when car departure is confirmed by ultrasonic."""
    global _on_departure_callback
    _on_departure_callback = fn


# ═══════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _post(url: str, timeout: float = _HTTP_TIMEOUT) -> bool:
    """
    Fire-and-forget HTTP POST.
    Returns True on 2xx, False on any error.
    """
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            return 200 <= status < 300
    except urllib.error.URLError as exc:
        print(f"[GateController] POST {url} → unreachable: {exc.reason}")
        return False
    except Exception as exc:
        print(f"[GateController] POST {url} → error: {exc}")
        return False


def _get_json(url: str, timeout: float = _HTTP_TIMEOUT) -> Optional[dict]:
    """GET a JSON endpoint. Returns parsed dict or None on error."""
    import json
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except Exception as exc:
        print(f"[GateController] GET {url} → error: {exc}")
        return None


def _cmd_gate(endpoint: str) -> bool:
    """POST an endpoint on the gate board."""
    if not _GATE_URL:
        print("[GateController] Gate URL not configured")
        return False
    ok = _post(f"{_GATE_URL}{endpoint}", timeout=_NOTIFY_TIMEOUT)
    print(f"[GateController] Gate {endpoint} → {'OK' if ok else 'FAIL'}")
    return ok


def _cmd_cam(endpoint: str) -> bool:
    """POST an endpoint on the cam board."""
    if not _CAM_URL:
        return False
    ok = _post(f"{_CAM_URL}{endpoint}", timeout=_NOTIFY_TIMEOUT)
    print(f"[GateController] Cam {endpoint} → {'OK' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY ACK HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def notify_entry_ack(ack: dict) -> None:
    """
    Called from RabbitMQ consumer when backend sends entry ACK.

    ALLOWED → open both gate board and cam board (arm DOWN).
    Any other status → keep blocked (arm UP / already blocked).
    """
    status = ack.get("status", "")
    reg    = ack.get("registration", "?")
    slot   = ack.get("slotId", "?")

    if status == "ALLOWED":
        print(f"[GateController] ENTRY ALLOWED: {reg}  slot={slot}  → opening gate + cam")
        _cmd_gate("/open")
        _cmd_cam("/open")
    else:
        reason = ack.get("reason", "")
        print(f"[GateController] ENTRY {status}: {reg}  reason={reason}  → keeping blocked")
        # Boards boot blocked; no action needed, but explicitly close if somehow open
        _cmd_gate("/close")
        _cmd_cam("/close")


# ═══════════════════════════════════════════════════════════════════════════
# EXIT ACK HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def notify_exit_ack(ack: dict) -> None:
    """
    Called from RabbitMQ consumer when backend sends exit ACK.

    OK → open gate board arm briefly so car can drive out.
         Cam board mirrors open.
    Any other status → log only (car already left the slot physically).
    """
    status = ack.get("status", "")
    reg    = ack.get("registration", "?")
    fee    = ack.get("totalFee", "?")
    mins   = ack.get("durationMinutes", "?")

    if status == "OK":
        print(f"[GateController] EXIT OK: {reg}  fee={fee}  duration={mins}min → opening for exit")
        _cmd_gate("/open")   # gate board opens briefly then auto-closes (EXIT_OPEN_MS timer)
        _cmd_cam("/open")    # mirror
    else:
        print(f"[GateController] EXIT {status}: {reg}  → no gate action")


# ═══════════════════════════════════════════════════════════════════════════
# DEPARTURE POLLING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def _poll_for_departure(cam_id: str, publish_exit_fn, rmq_channel_getter):
    """
    Background thread: poll cam board /status every _POLL_INTERVAL seconds.
    When notify_clear_pending = true:
      1. Call publish_exit_fn(channel, registration, province, lot_id, cam_id).
      2. POST /exit_sent to cam board to clear the flag.
    
    rmq_channel_getter: a callable that returns the current (conn, channel).
    """
    global _poll_running
    print(f"[GateController] Departure poll thread started for {cam_id}")

    while _poll_running:
        time.sleep(_POLL_INTERVAL)

        if not _CAM_URL:
            continue

        status = _get_json(f"{_CAM_URL}/status", timeout=_HTTP_TIMEOUT)
        if status is None:
            continue

        if status.get("notify_clear_pending", False):
            print(f"[GateController] notify_clear_pending = true — car departed!")

            # Fire the departure callback in main.py
            if _on_departure_callback is not None:
                try:
                    _on_departure_callback(cam_id)
                except Exception as exc:
                    print(f"[GateController] departure callback error: {exc}")

            # Tell cam board we handled it
            _cmd_cam("/exit_sent")

    print(f"[GateController] Departure poll thread stopped for {cam_id}")


def start_departure_polling(cam_id: str, publish_exit_fn=None,
                            rmq_channel_getter=None) -> None:
    """
    Start background polling for cam board departure signal.
    Call once at startup from main.py.
    """
    global _poll_thread, _poll_running
    if _poll_thread is not None and _poll_thread.is_alive():
        return  # already running

    _poll_running = True
    _poll_thread = threading.Thread(
        target=_poll_for_departure,
        args=(cam_id, publish_exit_fn, rmq_channel_getter),
        daemon=True,
    )
    _poll_thread.start()


def stop_departure_polling() -> None:
    global _poll_running
    _poll_running = False