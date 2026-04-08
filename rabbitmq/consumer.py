"""
rabbitmq/consumer.py — Consume ACK messages from the backend in a daemon thread.

  ocr.entry.ack  →  { camId, lotId, registration, province, status, slotId?, reason?, timestamp }
  ocr.exit.ack   →  { camId, lotId, registration, province, status, totalFee?, durationMinutes?, timestamp }

v2 changes:
  - Removed gate_controller.notify_entry_ack / notify_exit_ack calls.
    The backend now controls the gate board DIRECTLY via MQTT
    (barrier/commands/<camId>). Python no longer needs to call HTTP
    /open or /close on the gate board.
  - consumer.py only pushes ACKs into AckStore for main.py to read.
    main.py uses ACKs to update SlotState and call /open /close on
    the CAM board servo (which mirrors the gate but is separate).
"""

import json
import threading
from collections import deque
from typing import Optional

import pika

from .connection import connect


class AckStore:
    """Thread-safe store that keeps the most recent ACKs for the main loop to read."""

    def __init__(self, maxlen: int = 256):
        self._entry_acks: deque[dict] = deque(maxlen=maxlen)
        self._exit_acks:  deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ── writers (called from consumer thread) ──────────────────────────
    def push_entry_ack(self, ack: dict) -> None:
        with self._lock:
            self._entry_acks.append(ack)

    def push_exit_ack(self, ack: dict) -> None:
        with self._lock:
            self._exit_acks.append(ack)

    # ── readers (called from main thread) ──────────────────────────────
    def pop_entry_ack(self) -> Optional[dict]:
        with self._lock:
            return self._entry_acks.popleft() if self._entry_acks else None

    def pop_exit_ack(self) -> Optional[dict]:
        with self._lock:
            return self._exit_acks.popleft() if self._exit_acks else None


def _on_entry_ack(ch, method, _properties, body, store: AckStore):
    try:
        ack    = json.loads(body)
        status = ack.get("status", "?")
        reg    = ack.get("registration", "?")
        if status == "ALLOWED":
            slot = ack.get("slotId", "?")
            print(f"[ACK-ENTRY] {reg}  → ALLOWED  slot={slot}")
        else:
            reason = ack.get("reason", "")
            print(f"[ACK-ENTRY] {reg}  → {status}  reason={reason}")
        # Push to store — main.py drains this and calls _cmd_cam(/open) if needed
        store.push_entry_ack(ack)
    except Exception as exc:
        print(f"[ACK-ENTRY] parse error: {exc}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def _on_exit_ack(ch, method, _properties, body, store: AckStore):
    try:
        ack    = json.loads(body)
        status = ack.get("status", "?")
        reg    = ack.get("registration", "?")
        fee    = ack.get("totalFee", "?")
        mins   = ack.get("durationMinutes", "?")
        print(f"[ACK-EXIT]  {reg}  → {status}  fee={fee}  duration={mins}min")
        store.push_exit_ack(ack)
    except Exception as exc:
        print(f"[ACK-EXIT]  parse error: {exc}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def _consumer_loop(store: AckStore):
    """Blocking loop that runs in a daemon thread — auto-reconnects."""
    while True:
        conn = None
        try:
            conn, ch = connect()
            ch.basic_qos(prefetch_count=1)

            ch.basic_consume(
                queue="ocr.entry.ack",
                on_message_callback=lambda ch, m, p, b: _on_entry_ack(ch, m, p, b, store),
            )
            ch.basic_consume(
                queue="ocr.exit.ack",
                on_message_callback=lambda ch, m, p, b: _on_exit_ack(ch, m, p, b, store),
            )

            print("[RMQ-Consumer] Listening on ocr.entry.ack & ocr.exit.ack ...")
            ch.start_consuming()

        except pika.exceptions.AMQPConnectionError as exc:
            print(f"[RMQ-Consumer] connection lost: {exc} — reconnecting in 5s")
        except Exception as exc:
            print(f"[RMQ-Consumer] unexpected error: {exc} — reconnecting in 5s")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

        import time
        time.sleep(5)


def start_ack_consumer() -> AckStore:
    """Spawn a daemon thread that consumes ACK queues. Returns an AckStore."""
    store = AckStore()
    t = threading.Thread(target=_consumer_loop, args=(store,), daemon=True)
    t.start()
    return store