"""
rabbitmq/connection.py — Connect to RabbitMQ and assert the 4 SmartPark queues.

Queues:
  ocr.entry.events   OCR → Backend   (plate read at entrance)
  ocr.exit.events    OCR → Backend   (plate read at exit)
  ocr.entry.ack      Backend → OCR   (ALLOWED / REJECTED)
  ocr.exit.ack       Backend → OCR   (OK / ERROR + fee)
"""

import os
import pika

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672")

QUEUES = [
    "ocr.entry.events",
    "ocr.exit.events",
    "ocr.entry.ack",
    "ocr.exit.ack",
]


def connect() -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    """Create a blocking connection and assert all queues. Returns (connection, channel)."""
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    for q in QUEUES:
        ch.queue_declare(queue=q, durable=True)
    return conn, ch
