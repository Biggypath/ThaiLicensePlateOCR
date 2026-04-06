"""
rabbitmq/consumer.py — Consume ACK messages from the backend in a daemon thread.

  ocr.entry.ack  →  { camId, lotId, registration, province, status, slotId?, reason?, timestamp }
  ocr.exit.ack   →  { camId, lotId, registration, province, status, totalFee?, durationMinutes?, timestamp }
"""

import json
import threading
from collections import deque
from typing import Optional

import pika

from .connection import connect
from gate_controller import notify_entry_ack, notify_exit_ack


class AckStore:
    """Thread-safe store that keeps the most recent ACKs for the main loop to read."""

    def __init__(self, maxlen: int = 256):
        self._entry_acks: deque[dict] = deque(maxlen=maxlen)
        self._exit_acks: deque[dict] = deque(maxlen=maxlen)
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
        ack = json.loads(body)
        store.push_entry_ack(ack)
        status = ack.get("status", "?")
        reg = ack.get("registration", "?")
        if status == "ALLOWED":
            slot = ack.get("slotId", "?")
            print(f"[ACK-ENTRY] {reg}  → ALLOWED  slot={slot}")
        else:
            reason = ack.get("reason", "")
            print(f"[ACK-ENTRY] {reg}  → {status}  reason={reason}")
        notify_entry_ack(ack)
    except Exception as exc:
        print(f"[ACK-ENTRY] parse error: {exc}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def _on_exit_ack(ch, method, _properties, body, store: AckStore):
    try:
        ack = json.loads(body)
        store.push_exit_ack(ack)
        status = ack.get("status", "?")
        reg = ack.get("registration", "?")
        fee = ack.get("totalFee", "?")
        mins = ack.get("durationMinutes", "?")
        print(f"[ACK-EXIT]  {reg}  → {status}  fee={fee}  duration={mins}min")
        notify_exit_ack(ack)
    except Exception as exc:
        print(f"[ACK-EXIT]  parse error: {exc}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


def _consumer_loop(store: AckStore):
    """Blocking loop that runs in a daemon thread — auto-reconnects."""
    while True:
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
