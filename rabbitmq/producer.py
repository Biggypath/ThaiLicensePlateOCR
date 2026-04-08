"""
rabbitmq/producer.py — Publish OCR events to the backend.

  ocr.entry.events  →  { registration, province, lotId, slotId, camId }
  ocr.exit.events   →  { registration, province, lotId, slotId, camId }
"""

import json
import pika


def _publish(channel: pika.adapters.blocking_connection.BlockingChannel,
             queue: str, payload: dict) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
        ),
    )


def publish_entry_event(channel, registration: str, province: str,
                        lot_id: str, slot_id: str, cam_id: str) -> None:
    _publish(channel, "ocr.entry.events", {
        "registration": registration,
        "province": province,
        "lotId": lot_id,
        "slotId": slot_id,
        "camId": cam_id,
    })


def publish_exit_event(channel, registration: str, province: str,
                       lot_id: str, slot_id: str, cam_id: str) -> None:
    _publish(channel, "ocr.exit.events", {
        "registration": registration,
        "province": province,
        "lotId": lot_id,
        "slotId": slot_id,
        "camId": cam_id,
    })