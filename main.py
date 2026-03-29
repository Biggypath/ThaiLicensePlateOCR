"""
main.py — Thai licence-plate ALPR  (ESP32-CAM → YOLO → OCR → RabbitMQ)

v6 changes:
  1. Province detection now passes the EasyOCR reader to
     extract_province_from_crop() so EasyOCR is used as primary
     (Tesseract alone was failing on blurry province text).
  2. preprocess_plate_image called with number_zone_only=True (default)
     so province text at the bottom of the crop no longer bleeds
     into the number OCR, eliminating 'ต', 'ว' false positives.
  3. PlateStabilityGate integrated — OCR is skipped while box is moving.
  4. Published payload includes province fields.
"""

import cv2
import json
import os
import glob
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr
import pika

from plate_utils import (
    extract_best_plate_read,
    should_send_plate,
    preprocess_plate_image,
    deskew_plate,
    run_easyocr,
    run_tesseract,
    extract_province_from_crop,
    PlateMajorityVoter,
    PlateStabilityGate,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
ESP32_URL            = "http://172.20.10.3:81/stream"

# Mirror/flip correction for ESP32-CAM:
#   1  = flip horizontally (most common — text appears reversed)
#   0  = flip vertically
#  -1  = flip both axes
#  None = no flip
ESP32_FLIP_CODE      = 1

RMQ_HOST             = "localhost"
RMQ_PORT             = 5672
RMQ_USER             = "guest"
RMQ_PASS             = "guest"
RMQ_QUEUE            = "alpr_queue"

COOLDOWN_TIME        = 10
FRAME_SKIP           = 3
YOLO_CONF_THRESHOLD  = 0.35
STREAM_RETRY_DELAY   = 2

PREPROCESS_SCALE     = 3
VOTE_WINDOW          = 8
VOTE_MIN_VOTES       = 3
OCR_MIN_CONF         = 0.10

# Stability gate
STABILITY_MIN_FRAMES = 4
STABILITY_MAX_DRIFT  = 6.0

# Debug
DEBUG_SAVE_CROPS     = True
DEBUG_MAX_SAVES      = 10
_debug_save_count    = 0

HF_REPO_ID           = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME    = "HurricaneOD_beta.pt"


# ==========================================
# 2. INITIALIZATION
# ==========================================
def cleanup_old_debug_files(pattern: str = "debug_plate_*.jpg", keep: int = 0):
    files = sorted(glob.glob(pattern))
    for f in files[: max(0, len(files) - keep)]:
        try:
            os.remove(f)
        except OSError:
            pass


print("Cleaning up old debug images...")
cleanup_old_debug_files()

print("Loading EasyOCR (Thai/English)...")
reader = easyocr.Reader(['th', 'en'], gpu=False)

print("Downloading/Loading YOLO model...")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
model = YOLO(model_path)

print("Connecting to RabbitMQ...")


def connect_rabbitmq():
    creds  = pika.PlainCredentials(RMQ_USER, RMQ_PASS)
    params = pika.ConnectionParameters(RMQ_HOST, RMQ_PORT, '/', creds)
    conn   = pika.BlockingConnection(params)
    ch     = conn.channel()
    ch.queue_declare(queue=RMQ_QUEUE, durable=True)
    return conn, ch


def publish_plate(channel, payload_bytes):
    channel.basic_publish(
        exchange    = '',
        routing_key = RMQ_QUEUE,
        body        = payload_bytes,
        properties  = pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
        ),
    )


try:
    rmq_conn, rmq_channel = connect_rabbitmq()
    print("RabbitMQ connected!")
except Exception as exc:
    print(f"RabbitMQ failed: {exc}")
    raise SystemExit(1)


# ==========================================
# 3. STREAM HELPERS
# ==========================================
def open_stream(url):
    cap = cv2.VideoCapture(url)
    return cap if cap.isOpened() else None


def reconnect_stream(old_cap, url):
    if old_cap:
        old_cap.release()
    time.sleep(STREAM_RETRY_DELAY)
    return open_stream(url)


def save_debug(tag: str, raw_crop, preprocessed):
    global _debug_save_count
    if not DEBUG_SAVE_CROPS or _debug_save_count >= DEBUG_MAX_SAVES:
        return
    ts = int(time.time() * 1000)
    cv2.imwrite(f"debug_plate_{ts}_{tag}_raw.jpg",  raw_crop)
    cv2.imwrite(f"debug_plate_{ts}_{tag}_prep.jpg", preprocessed)
    _debug_save_count += 1


# ==========================================
# 4. MAIN LOOP
# ==========================================
cap              = None
frame_count      = 0
last_seen_plates = {}
voter            = PlateMajorityVoter(window=VOTE_WINDOW, min_votes=VOTE_MIN_VOTES)
stability        = PlateStabilityGate(
    min_stable_frames=STABILITY_MIN_FRAMES,
    max_pixel_drift=STABILITY_MAX_DRIFT,
)

while True:
    # ── Stream connection ──────────────────────────────────────────────────
    if cap is None or not cap.isOpened():
        print("Connecting to ESP32 stream...")
        cap = open_stream(ESP32_URL)
        if cap is None:
            print("Stream unavailable, retrying...")
            time.sleep(STREAM_RETRY_DELAY)
            continue
        print("Stream connected.")

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame grab failed — reconnecting...")
        cap = reconnect_stream(cap, ESP32_URL)
        continue

    # ── Flip correction ────────────────────────────────────────────────────
    if ESP32_FLIP_CODE is not None:
        frame = cv2.flip(frame, ESP32_FLIP_CODE)

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("ESP32-CAM ALPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── YOLO detection ─────────────────────────────────────────────────────
    try:
        results = model(frame, verbose=False)[0]
    except Exception as exc:
        print(f"YOLO error: {exc}")
        continue

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_conf       = float(box.conf[0])

        if yolo_conf < YOLO_CONF_THRESHOLD:
            continue

        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # ── Stability gate ─────────────────────────────────────────────────
        if not stability.is_stable(x1, y1, x2, y2):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 210), 1)
            cv2.putText(frame, "stabilising...", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 210), 1,
                        cv2.LINE_AA)
            continue

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        # ── Step 1: deskew ─────────────────────────────────────────────────
        plate_crop = deskew_plate(plate_crop)

        # ── Step 2: province detection (raw colour crop + EasyOCR reader) ──
        # Must happen BEFORE preprocess_plate_image which slices the crop.
        province_info = extract_province_from_crop(
            plate_crop,
            easyocr_reader=reader,   # ← pass the reader for EasyOCR primary
        )
        if province_info:
            print(f"  Province: {province_info['thai']} "
                  f"({province_info['en']}, {province_info['region']}) "
                  f"score={province_info['score']:.2f} "
                  f"strip={province_info.get('strip','?')}")
        else:
            print("  Province: unknown")

        # ── Step 3: preprocess (number zone only — excludes province strip) ─
        try:
            preprocessed = preprocess_plate_image(
                plate_crop,
                scale=PREPROCESS_SCALE,
                number_zone_only=True,   # ← strips province text row
            )
        except Exception as exc:
            print(f"  Preprocess error: {exc}")
            continue

        # ── Step 4: EasyOCR on number zone ────────────────────────────────
        ocr_results = run_easyocr(reader, preprocessed)
        print(f"  EasyOCR ({yolo_conf:.2f}): "
              f"{[(r[1], round(float(r[2]), 2)) for r in ocr_results]}")

        candidate = extract_best_plate_read(ocr_results,
                                            min_confidence=OCR_MIN_CONF)

        # ── Step 5: Tesseract fallback ─────────────────────────────────────
        if candidate is None:
            tess_results = run_tesseract(preprocessed)
            if tess_results:
                print(f"  Tesseract: "
                      f"{[(r[1], round(float(r[2]), 2)) for r in tess_results]}")
                candidate = extract_best_plate_read(tess_results,
                                                    min_confidence=OCR_MIN_CONF)

        print(f"  Candidate: {candidate}")

        if candidate is not None:
            save_debug("hit", plate_crop, preprocessed)

        # ── Step 6: majority vote ──────────────────────────────────────────
        winner = voter.update(candidate)
        if winner is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            continue

        plate_text, ocr_conf = winner
        current_time = time.time()

        if not should_send_plate(plate_text, last_seen_plates,
                                 current_time, COOLDOWN_TIME):
            print(f"  [{plate_text}] in cooldown.")
            voter.reset()
            stability.reset()
            continue

        # ── Step 7: build & publish payload ───────────────────────────────
        province_thai   = province_info["thai"]   if province_info else None
        province_en     = province_info["en"]     if province_info else None
        province_region = province_info["region"] if province_info else None

        print(f"✅  {plate_text}  "
              f"province={province_thai} ({province_en})  "
              f"YOLO={yolo_conf:.2f}  OCR≈{ocr_conf:.2f}")

        last_seen_plates[plate_text] = current_time
        voter.reset()
        stability.reset()

        payload = {
            "timestamp"        : int(current_time),
            "plate_text"       : plate_text,
            "province_thai"    : province_thai,
            "province_en"      : province_en,
            "province_region"  : province_region,
            "yolo_confidence"  : yolo_conf,
            "ocr_confidence"   : float(ocr_conf),
            "camera_id"        : "esp32_cam_gate_1",
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            publish_plate(rmq_channel, json_payload)
            print(f"  Published: {json.dumps(payload, ensure_ascii=False)}")
        except Exception:
            print("RabbitMQ lost — reconnecting...")
            try:
                if rmq_conn and rmq_conn.is_open:
                    rmq_conn.close()
            except Exception:
                pass
            rmq_conn, rmq_channel = connect_rabbitmq()
            publish_plate(rmq_channel, json_payload)

        # Green box + label on screen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (f"{plate_text}  {province_en}"
                 if province_en else plate_text)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow("ESP32-CAM ALPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# 5. CLEANUP
# ==========================================
if cap:
    cap.release()
cv2.destroyAllWindows()
try:
    if rmq_conn and rmq_conn.is_open:
        rmq_conn.close()
except Exception:
    pass