"""
rabbitmq/connection.py — Connect to RabbitMQ and assert the SmartPark OCR queues.

Queues:
  ocr.entry.events   OCR → Backend   (plate read at entrance)
  ocr.exit.events    OCR → Backend   (plate read at exit)

ACKs (entry/exit) are sent by the backend directly to ESP32 via MQTT plugin.
"""

import os
import pika

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672")

QUEUES = [
    "ocr.entry.events",
    "ocr.exit.events",
]


def connect() -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    """Create a blocking connection and assert all queues. Returns (connection, channel)."""
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    for q in QUEUES:
        ch.queue_declare(queue=q, durable=True)
    return conn, ch
