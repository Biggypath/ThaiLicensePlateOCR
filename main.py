"""
main.py — Smart Parking Gate ALPR  v11.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hardware layout:
  ESP32-CAM #1  (Entrance)  172.20.10.6  → Gate ESP32 #1  172.20.10.8
  ESP32-CAM #2  (Exit)      172.20.10.7  → Gate ESP32 #2  172.20.10.9

What's new in v11.0 (vs v10.3):
  [v11-1]  DUAL CAMERA THREADS — each camera runs in its own daemon
           thread; both process video concurrently and independently.

  [v11-2]  GateController CLASS — all gate state (mutex, timer, HTTP)
           is encapsulated per camera-gate pair. Replaces the global
           _gate_* variables from v10.3.

  [v11-3]  LOW-LIGHT DETECTION — detect_low_light() measures mean
           frame brightness.  If below LOW_LIGHT_THRESHOLD the gate
           opens automatically for safety and /dark is sent to ESP32.
           Returns to normal via /light when brightness recovers.

  [v11-4]  CAR PRESENCE TRACKING — checks whether YOLO detects a plate
           in the current frame and sends /car or /clear to the ESP32,
           keeping its own smart-close logic in sync.

  [v11-5]  CROSS-CAMERA DUPLICATE PREVENTION — a shared plate registry
           (with lock) prevents the same vehicle triggering both gates.

  [v11-6]  PARKING OCCUPANCY COUNTER — entrance detections increment,
           exit detections decrement a shared counter displayed on the
           OpenCV overlay.

  [v11-7]  FULL HTTP ENDPOINT SET — open, close, keep, car, clear,
           dark, light are all called at the appropriate moments.

  [v11-8]  ENTRANCE / EXIT ROLE — entrance camera checks ALLOWED_PLATES
           and increments occupancy; exit camera decrements and always
           opens (any recognised plate means exit).
"""

import cv2
import json
import os
import glob
import time
import math
import threading
import requests
import numpy as np
from collections import deque, defaultdict, Counter
from enum import Enum
from typing import Optional
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr
import pika

from plate_utils import (
    extract_best_plate_read,
    fuse_ocr_results,
    adaptive_min_confidence,
    should_send_plate,
    preprocess_plate_image,
    deskew_plate,
    run_easyocr,
    run_easyocr_raw,
    run_tesseract,
    extract_province_from_crop,
    extract_province_from_ocr_tokens,
    normalize_plate_text,
    PlateTracker,
    PlateMajorityVoter,
    PlateStabilityGate,
)
from plate_corrector import is_valid_plate
from llm_plate_corrector import LLMCorrector as LLMOCRFallback


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

CAMERA_CONFIGS = [
    {
        "camera_id": "cam_entrance",
        "url":       "http://172.20.10.6:81/stream",
        "role":      "entrance",            # entrance: check ALLOWED_PLATES
        "gate_ip":   "http://172.20.10.8",  # gate ESP32 #1
        "cam_ip":    "http://172.20.10.6",  # for flash control
    },
    {
        "camera_id": "cam_exit",
        "url":       "http://172.20.10.7:81/stream",
        "role":      "exit",                # exit: any valid plate opens gate
        "gate_ip":   "http://172.20.10.9",  # gate ESP32 #2
        "cam_ip":    "http://172.20.10.7",
    },
]

ESP32_FLIP_CODE       = 1           # 1=horizontal, 0=vertical, -1=both, None=no flip

# ── RabbitMQ ────────────────────────────────────────────────────────────
RMQ_HOST              = "localhost"
RMQ_PORT              = 5672
RMQ_USER              = "guest"
RMQ_PASS              = "guest"
RMQ_QUEUE             = "alpr_queue"

# ── Detection ────────────────────────────────────────────────────────────
COOLDOWN_TIME         = 15          # seconds before the same plate re-fires
FRAME_SKIP            = 3           # process every Nth frame
YOLO_CONF_THRESHOLD   = 0.35
STREAM_RETRY_DELAY    = 2

# ── OCR / preprocessing ──────────────────────────────────────────────────
PREPROCESS_SCALE      = 3
VOTE_WINDOW           = 8
VOTE_MIN_VOTES        = 3
OCR_SKIP_DRIFT        = 3.0         # px: skip OCR if plate box barely moved
STABILITY_MIN_FRAMES  = 4
STABILITY_MAX_DRIFT   = 6.0
MAX_CORRECTION_SCORE  = 0.65
LLM_CANDIDATE_HISTORY = 20

# ── Gate timing ──────────────────────────────────────────────────────────
GATE_OPEN_DURATION    = 120.0       # seconds (2 minutes)
GATE_HTTP_TIMEOUT     = 3.0
GATE_RETRY_COUNT      = 3
GATE_RETRY_DELAY      = 0.5
MAX_LOCK_TIME         = GATE_OPEN_DURATION + 10.0
DETECTION_LINGER      = 3.0         # sec: delay close if detection was recent

# ── Low-light detection  [v11-3] ─────────────────────────────────────────
LOW_LIGHT_THRESHOLD      = 55       # mean gray pixel value (0–255)
LOW_LIGHT_HYSTERESIS     = 70       # threshold to return to "normal"
LOW_LIGHT_CHECK_INTERVAL = 5.0      # seconds between light-condition updates
DARK_OPEN_GATE           = True     # open gate automatically when dark

# ── Car presence  [v11-4] ────────────────────────────────────────────────
CAR_CHECK_INTERVAL    = 2.0         # seconds between /car or /clear updates
CAR_ABSENT_FRAMES     = 15          # consecutive no-detection frames = car gone

# ── Parking occupancy  [v11-6] ───────────────────────────────────────────
TOTAL_PARKING_SLOTS   = 20

# ── Allowed plates (entrance only)  [v11-8] ──────────────────────────────
ALLOWED_PLATES: set[str] = {
    "1กข1234",
    "3กบ7744",
    # Add more plates here or load from a database
}

# ── Debug ────────────────────────────────────────────────────────────────
DEBUG_SAVE_CROPS      = False
DEBUG_MAX_SAVES       = 10

# ── YOLO model ───────────────────────────────────────────────────────────
HF_REPO_ID            = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME     = "HurricaneOD_beta.pt"


# ══════════════════════════════════════════════════════════════════════════
# 2. GATE STATE MACHINE  [v11-2]
# ══════════════════════════════════════════════════════════════════════════

class GateState(Enum):
    IDLE             = "IDLE"
    OPEN             = "OPEN"
    WAITING_CAR_PASS = "WAITING_CAR_PASS"


class GateController:
    """
    [v11-2] Encapsulates gate state machine + all HTTP communication
    for one camera–gate pair.  Thread-safe.
    """

    def __init__(
        self,
        camera_id:     str,
        gate_ip:       str,
        open_duration: float = GATE_OPEN_DURATION,
    ):
        self.camera_id     = camera_id
        self.gate_ip       = gate_ip
        self.open_duration = open_duration

        self._mutex              = threading.Lock()
        self._state              = GateState.IDLE
        self._locked             = False
        self._locked_until       = 0.0
        self._release_timer: Optional[threading.Timer] = None

        # Tracking state for smart decisions
        self._last_detection_ts  = 0.0
        self._car_present        = False
        self._low_light          = False

        # Timestamps for rate-limiting HTTP updates
        self._light_update_ts    = 0.0
        self._car_update_ts      = 0.0

    # ── Public queries ────────────────────────────────────────────────────

    @property
    def state(self) -> GateState:
        with self._mutex:
            return self._state

    def is_idle(self) -> bool:
        with self._mutex:
            return not self._locked

    # ── Gate acquisition ──────────────────────────────────────────────────

    def try_acquire(self) -> bool:
        """Attempt to acquire gate lock. Returns True if this call owns the gate."""
        with self._mutex:
            now = time.monotonic()

            # [v11-2] Stale-lock guard
            if self._locked:
                stale_at = self._locked_until + (MAX_LOCK_TIME - self.open_duration)
                if now > stale_at:
                    print(f"  [Gate {self.camera_id} FORCED] Stale lock released")
                    self._locked = False
                    self._state  = GateState.IDLE
                    if self._release_timer:
                        self._release_timer.cancel()
                        self._release_timer = None

            if self._locked:
                remaining = max(0.0, self._locked_until - now)
                print(f"  [Gate {self.camera_id}] DENIED — "
                      f"locked ({remaining:.1f}s remaining)")
                return False

            # Acquire
            self._locked       = True
            self._state        = GateState.OPEN
            self._locked_until = now + self.open_duration
            self._release_timer = threading.Timer(
                self.open_duration, self._auto_release)
            self._release_timer.daemon = True
            self._release_timer.start()

            print(f"  [Gate {self.camera_id}] ACQUIRED — "
                  f"auto-release in {self.open_duration:.0f}s")
            return True

    def record_detection(self):
        """Call whenever a plate is confirmed. Used for smart-close delay."""
        self._last_detection_ts = time.time()

    def _auto_release(self):
        """Background callback: smart close with DETECTION_LINGER delay."""
        # [v11-4] Wait while a recent detection suggests car still present
        while True:
            since = time.time() - self._last_detection_ts
            if since < DETECTION_LINGER:
                print(f"  [Gate {self.camera_id}] Auto-release deferred "
                      f"({since:.1f}s since last detection)")
                time.sleep(DETECTION_LINGER)
            else:
                break

        print(f"  [Gate {self.camera_id}] Auto-release: "
              f"OPEN → WAITING_CAR_PASS")
        with self._mutex:
            self._state = GateState.WAITING_CAR_PASS

        self._do_http_sync("close")
        time.sleep(1.0)
        self._do_http_sync("clear")

        with self._mutex:
            self._state         = GateState.IDLE
            self._locked        = False
            self._release_timer = None

        print(f"  [Gate {self.camera_id}] State → IDLE")

    def force_release(self):
        """Immediate manual override — closes gate and resets state."""
        with self._mutex:
            if self._release_timer:
                self._release_timer.cancel()
                self._release_timer = None
            self._state  = GateState.IDLE
            self._locked = False
        print(f"  [Gate {self.camera_id}] Force-released → IDLE")

    # ── HTTP calls ────────────────────────────────────────────────────────

    def _do_http_sync(self, endpoint: str) -> bool:
        """Blocking HTTP POST with retry. Used inside background threads."""
        url = f"{self.gate_ip}/{endpoint}"
        for attempt in range(1, GATE_RETRY_COUNT + 1):
            try:
                resp = requests.post(url, timeout=GATE_HTTP_TIMEOUT)
                if resp.status_code == 200:
                    print(f"  [Gate HW {self.camera_id}] ✅ "
                          f"/{endpoint} OK (attempt {attempt})")
                    return True
                print(f"  [Gate HW {self.camera_id}] ⚠️  "
                      f"/{endpoint} HTTP {resp.status_code} (attempt {attempt})")
            except requests.exceptions.ConnectionError as exc:
                print(f"  [Gate RETRY {self.camera_id}] "
                      f"/{endpoint} ConnectionError attempt {attempt}: {exc}")
            except requests.exceptions.Timeout:
                print(f"  [Gate RETRY {self.camera_id}] "
                      f"/{endpoint} Timeout attempt {attempt}")
            except Exception as exc:
                print(f"  [Gate ERROR {self.camera_id}] "
                      f"/{endpoint} unexpected: {exc}")
            if attempt < GATE_RETRY_COUNT:
                time.sleep(GATE_RETRY_DELAY)
        print(f"  [Gate ERROR {self.camera_id}] "
              f"/{endpoint} ALL {GATE_RETRY_COUNT} attempts FAILED")
        return False

    def _fire_async(self, endpoint: str):
        """Non-blocking: launch _do_http_sync in a daemon thread."""
        t = threading.Thread(
            target=self._do_http_sync,
            args=(endpoint,),
            daemon=True,
            name=f"gate-{endpoint}-{self.camera_id}",
        )
        t.start()

    def open(self):    self._fire_async("open")
    def close(self):   self._fire_async("close")
    def keep_open(self): self._fire_async("keep")

    # ── Car presence updates  [v11-4] ─────────────────────────────────────
    def update_car_presence(self, present: bool):
        now = time.time()
        if now - self._car_update_ts < CAR_CHECK_INTERVAL:
            return
        changed = present != self._car_present
        self._car_present   = present
        self._car_update_ts = now
        if changed:
            self._fire_async("car" if present else "clear")
            label = "🚗 CAR" if present else "❌ CLEAR"
            print(f"  [CarPresence {self.camera_id}] {label}")

    # ── Light condition updates  [v11-3] ──────────────────────────────────
    def update_light_condition(self, is_dark: bool):
        now = time.time()
        if now - self._light_update_ts < LOW_LIGHT_CHECK_INTERVAL:
            return
        changed = is_dark != self._low_light
        self._low_light      = is_dark
        self._light_update_ts = now
        if changed:
            self._fire_async("dark" if is_dark else "light")
            print(f"  [Light {self.camera_id}] "
                  f"{'🌙 DARK — gate opening for safety' if is_dark else '☀️  LIGHT — normal'}")

    # ── Overlay label ─────────────────────────────────────────────────────
    def overlay_label(self) -> tuple:
        """Returns (label_str, bgr_color) for cv2.putText."""
        with self._mutex:
            state     = self._state
            locked_until = self._locked_until
            locked    = self._locked
        remaining = max(0.0, locked_until - time.monotonic()) if locked else 0
        light_tag = " [DARK]" if self._low_light else ""

        if state == GateState.OPEN:
            return (f"Gate {self.camera_id}: OPEN  {remaining:.0f}s{light_tag}",
                    (0, 255, 0))
        elif state == GateState.WAITING_CAR_PASS:
            return (f"Gate {self.camera_id}: CLOSING…{light_tag}", (0, 165, 255))
        else:
            return (f"Gate {self.camera_id}: IDLE{light_tag}", (180, 180, 180))


# ══════════════════════════════════════════════════════════════════════════
# 3. SHARED GLOBALS
# ══════════════════════════════════════════════════════════════════════════

# Plate cooldown — shared across all cameras  [v11-5]
_plates_lock             = threading.Lock()
_last_seen_plates: dict  = {}     # plate_text → timestamp
_plate_source:     dict  = {}     # plate_text → camera_id

# Parking occupancy  [v11-6]
_occupancy_lock          = threading.Lock()
_parking_occupied        = 0

# RabbitMQ — single connection with lock
_rmq_lock                = threading.Lock()
rmq_conn                 = None
rmq_channel              = None

# Debug
_debug_save_count        = 0
_debug_lock              = threading.Lock()

# EasyOCR semaphore (limit concurrency on CPU)
_ocr_semaphore           = threading.Semaphore(1)


# ══════════════════════════════════════════════════════════════════════════
# 4. INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

def _cleanup_debug_files():
    for f in glob.glob("debug_plate_*.jpg"):
        try:
            os.remove(f)
        except OSError:
            pass


print("Cleaning up old debug images…")
_cleanup_debug_files()

print("Loading EasyOCR (Thai/English)…")
reader = easyocr.Reader(['th', 'en'], gpu=False)

print("Downloading/loading YOLO model…")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
model      = YOLO(model_path)

print("Connecting to RabbitMQ…")


def _connect_rabbitmq():
    creds  = pika.PlainCredentials(RMQ_USER, RMQ_PASS)
    params = pika.ConnectionParameters(RMQ_HOST, RMQ_PORT, '/', creds)
    conn   = pika.BlockingConnection(params)
    ch     = conn.channel()
    ch.queue_declare(queue=RMQ_QUEUE, durable=True)
    return conn, ch


try:
    rmq_conn, rmq_channel = _connect_rabbitmq()
    print("RabbitMQ connected!")
except Exception as exc:
    print(f"RabbitMQ failed: {exc}")
    raise SystemExit(1)


def _publish(payload_bytes: bytes):
    global rmq_conn, rmq_channel
    with _rmq_lock:
        try:
            rmq_channel.basic_publish(
                exchange='',
                routing_key=RMQ_QUEUE,
                body=payload_bytes,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
            )
        except Exception:
            print("RabbitMQ lost — reconnecting…")
            try:
                if rmq_conn and rmq_conn.is_open:
                    rmq_conn.close()
            except Exception:
                pass
            rmq_conn, rmq_channel = _connect_rabbitmq()
            rmq_channel.basic_publish(
                exchange='',
                routing_key=RMQ_QUEUE,
                body=payload_bytes,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
            )


print("Initialising LLM OCR fallback…")
llm_fallback = LLMOCRFallback()
print("LLM fallback ready." if llm_fallback._api_key else
      "LLM fallback DISABLED (no API key).")

# Build gate controllers: one per camera config
gate_controllers: dict[str, GateController] = {
    cfg["camera_id"]: GateController(
        camera_id     = cfg["camera_id"],
        gate_ip       = cfg["gate_ip"],
        open_duration = GATE_OPEN_DURATION,
    )
    for cfg in CAMERA_CONFIGS
}


# ══════════════════════════════════════════════════════════════════════════
# 5. LOW-LIGHT DETECTION  [v11-3]
# ══════════════════════════════════════════════════════════════════════════

def detect_low_light(frame: np.ndarray, is_currently_dark: bool) -> bool:
    """
    Returns True if the frame indicates low-light conditions.
    Uses hysteresis: higher threshold to leave dark state than to enter it,
    preventing rapid oscillation at the boundary.
    """
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    if is_currently_dark:
        # Hysteresis: need to be clearly bright to leave dark mode
        return mean_brightness < LOW_LIGHT_HYSTERESIS
    else:
        return mean_brightness < LOW_LIGHT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════
# 6. ACCESS CONTROL  [v11-8]
# ══════════════════════════════════════════════════════════════════════════

def is_plate_allowed(plate_text: str, role: str) -> bool:
    """
    Entrance: plate must be in ALLOWED_PLATES.
    Exit:     any valid plate is allowed (they're leaving).
    """
    norm = normalize_plate_text(plate_text)
    if role == "exit":
        return True     # any valid plate can exit
    allowed = norm in ALLOWED_PLATES
    status  = "✅ ALLOWED" if allowed else "🚫 DENIED"
    print(f"  [AccessControl] {norm}  role={role}  →  {status}")
    return allowed


def update_occupancy(role: str, plate_text: str):
    """[v11-6] Increment on entrance, decrement on exit."""
    global _parking_occupied
    with _occupancy_lock:
        if role == "entrance":
            _parking_occupied = min(_parking_occupied + 1, TOTAL_PARKING_SLOTS)
        elif role == "exit":
            _parking_occupied = max(_parking_occupied - 1, 0)
    print(f"  [Occupancy] {_parking_occupied}/{TOTAL_PARKING_SLOTS} "
          f"({'entry' if role == 'entrance' else 'exit'}: {plate_text})")


# ══════════════════════════════════════════════════════════════════════════
# 7. CROSS-CAMERA DUPLICATE PREVENTION  [v11-5]
# ══════════════════════════════════════════════════════════════════════════

def _check_and_register_plate(
    plate_text: str,
    camera_id:  str,
    current_time: float,
) -> bool:
    """
    Returns True if this plate should be processed (no cooldown, no
    duplicate from another camera in the last COOLDOWN_TIME seconds).
    Registers the plate on success.
    """
    with _plates_lock:
        last_time = _last_seen_plates.get(plate_text, 0.0)
        if current_time - last_time < COOLDOWN_TIME:
            source = _plate_source.get(plate_text, "?")
            print(f"  [Duplicate] {plate_text} in cooldown "
                  f"(last seen by {source}, {current_time - last_time:.1f}s ago)")
            return False
        _last_seen_plates[plate_text] = current_time
        _plate_source[plate_text]     = camera_id
        return True


# ══════════════════════════════════════════════════════════════════════════
# 8. STREAM HELPERS
# ══════════════════════════════════════════════════════════════════════════

def open_stream(url: str):
    cap = cv2.VideoCapture(url)
    return cap if cap.isOpened() else None


def reconnect_stream(old_cap, url: str):
    if old_cap:
        old_cap.release()
    time.sleep(STREAM_RETRY_DELAY)
    return open_stream(url)


def save_debug(tag: str, raw_crop, preprocessed):
    global _debug_save_count
    if not DEBUG_SAVE_CROPS:
        return
    with _debug_lock:
        if _debug_save_count >= DEBUG_MAX_SAVES:
            return
        ts = int(time.time() * 1000)
        cv2.imwrite(f"debug_plate_{ts}_{tag}_raw.jpg",  raw_crop)
        cv2.imwrite(f"debug_plate_{ts}_{tag}_prep.jpg", preprocessed)
        _debug_save_count += 1


# ══════════════════════════════════════════════════════════════════════════
# 9. PUBLISH + GATE TRIGGER
# ══════════════════════════════════════════════════════════════════════════

def publish_and_trigger(
    frame,
    tid:            int,
    x1: int, y1: int, x2: int, y2: int,
    plate_text:     str,
    ocr_conf:       float,
    yolo_conf:      float,
    province_info,
    camera_id:      str,
    role:           str,
    gate:           GateController,
    source:         str = "voter",
):
    """
    Publishes plate event to RabbitMQ and optionally opens the gate.
    Also updates car presence and occupancy counter.
    """
    gate.record_detection()  # update smart-close timestamp

    province_thai   = province_info["thai"]   if province_info else None
    province_en     = province_info["en"]     if province_info else None
    province_region = province_info["region"] if province_info else None

    gate_triggered      = False
    gate_denied_reason  = None

    with _occupancy_lock:
        occupied = _parking_occupied

    # Gate decision
    if not gate.is_idle():
        gate_denied_reason = f"gate locked ({gate.state.value})"
        print(f"  [Gate] {camera_id} suppressed — {gate_denied_reason}")
    elif not is_plate_allowed(plate_text, role):
        gate_denied_reason = "plate not in allowed list"
    elif role == "entrance" and occupied >= TOTAL_PARKING_SLOTS:
        gate_denied_reason = f"parking full ({occupied}/{TOTAL_PARKING_SLOTS})"
    else:
        if gate.try_acquire():
            gate.open()
            gate_triggered = True
            update_occupancy(role, plate_text)
        else:
            gate_denied_reason = "gate race condition"

    # Console
    status_str = ("🟢 GATE OPEN" if gate_triggered
                  else f"🔴 NO GATE ({gate_denied_reason})")
    print(f"✅  [#{tid}] {plate_text}  cam={camera_id}  role={role}  "
          f"source={source}  province={province_thai} ({province_en})  "
          f"YOLO={yolo_conf:.2f}  OCR≈{ocr_conf:.2f}  "
          f"occ={occupied}/{TOTAL_PARKING_SLOTS}  {status_str}")

    # RabbitMQ payload
    payload = {
        "timestamp":          int(time.time()),
        "plate_text":         plate_text,
        "province_thai":      province_thai,
        "province_en":        province_en,
        "province_region":    province_region,
        "yolo_confidence":    round(yolo_conf, 4),
        "ocr_confidence":     round(float(ocr_conf), 4),
        "camera_id":          camera_id,
        "role":               role,
        "track_id":           tid,
        "ocr_source":         source,
        "gate_triggered":     gate_triggered,
        "gate_denied_reason": gate_denied_reason,
        "parking_occupied":   occupied,
        "parking_total":      TOTAL_PARKING_SLOTS,
    }
    _publish(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    # OpenCV overlay colour
    if gate_triggered:
        colour = (0, 255, 0)
    elif gate_denied_reason and "not in allowed" in str(gate_denied_reason):
        colour = (0, 0, 255)
    elif source == "llm_fallback":
        colour = (255, 180, 0)
    else:
        colour = (0, 165, 255)

    label = f"#{tid} {plate_text}"
    if province_en:
        label += f"  {province_en}"
    if gate_triggered:
        label += " [GATE OPEN]"
    elif gate_denied_reason and "not in allowed" in str(gate_denied_reason):
        label += " [DENIED]"
    if source == "llm_fallback":
        label += " [LLM]"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════
# 10. PER-CAMERA PROCESSING THREAD  [v11-1]
# ══════════════════════════════════════════════════════════════════════════

def camera_thread(camera_cfg: dict):
    """
    [v11-1] Runs in a dedicated daemon thread for each camera.
    Owns its own: tracker, voter pool, stability gates, OCR history,
    LLM fallback instance, and gate controller reference.
    """
    camera_id  = camera_cfg["camera_id"]
    stream_url = camera_cfg["url"]
    role       = camera_cfg["role"]
    gate       = gate_controllers[camera_id]

    # ── Per-camera state ────────────────────────────────────────────────
    tracker = PlateTracker()

    _voters:      dict[int, PlateMajorityVoter] = {}
    _stability:   dict[int, PlateStabilityGate] = {}
    _ocr_history: dict[int, deque]               = {}

    llm = LLMOCRFallback()

    def _voter(tid):
        if tid not in _voters:
            _voters[tid] = PlateMajorityVoter(
                window=VOTE_WINDOW, min_votes=VOTE_MIN_VOTES,
                llm_corrector=llm, track_id=tid)
        return _voters[tid]

    def _stab(tid):
        if tid not in _stability:
            _stability[tid] = PlateStabilityGate(
                min_stable_frames=STABILITY_MIN_FRAMES,
                max_pixel_drift=STABILITY_MAX_DRIFT)
        return _stability[tid]

    def _hist(tid) -> deque:
        if tid not in _ocr_history:
            _ocr_history[tid] = deque(maxlen=LLM_CANDIDATE_HISTORY)
        return _ocr_history[tid]

    def _add_history(tid, results):
        h = _hist(tid)
        seen = set(h)
        for item in results:
            if len(item) >= 2:
                txt = str(item[1]).strip()
                if txt and txt not in seen:
                    h.append(txt)
                    seen.add(txt)

    def _cleanup_track(tid):
        _voters.pop(tid, None)
        _stability.pop(tid, None)
        _ocr_history.pop(tid, None)
        llm.invalidate_track(tid)

    # ── Stream state ─────────────────────────────────────────────────────
    cap         = None
    frame_count = 0
    last_n_det  = deque(maxlen=CAR_ABSENT_FRAMES)  # recent detection bools

    print(f"[{camera_id}] Thread started — role={role} gate={gate.gate_ip}")

    while True:
        # ── Stream connection ───────────────────────────────────────────
        if cap is None or not cap.isOpened():
            print(f"[{camera_id}] Connecting to {stream_url} …")
            cap = open_stream(stream_url)
            if cap is None:
                print(f"[{camera_id}] Stream unavailable — retrying in {STREAM_RETRY_DELAY}s")
                time.sleep(STREAM_RETRY_DELAY)
                continue
            print(f"[{camera_id}] Stream connected.")

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[{camera_id}] Frame grab failed — reconnecting…")
            cap = reconnect_stream(cap, stream_url)
            continue

        if ESP32_FLIP_CODE is not None:
            frame = cv2.flip(frame, ESP32_FLIP_CODE)

        frame_count += 1

        # ── [v11-3] Low-light detection (every frame, rate-limited send) ─
        is_dark = detect_low_light(frame, gate._low_light)
        gate.update_light_condition(is_dark)

        # If dark and safety-open is enabled, open the gate immediately
        if is_dark and DARK_OPEN_GATE and gate.is_idle():
            print(f"  [{camera_id}] 🌙 Dark frame — safety open triggered")
            if gate.try_acquire():
                gate.open()

        # ── Frame skip ───────────────────────────────────────────────────
        if frame_count % FRAME_SKIP != 0:
            gate_label, gate_colour = gate.overlay_label()
            cv2.putText(frame, gate_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_colour, 2, cv2.LINE_AA)
            # Occupancy overlay
            with _occupancy_lock:
                occ = _parking_occupied
            cv2.putText(frame, f"Parking: {occ}/{TOTAL_PARKING_SLOTS}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 80), 1, cv2.LINE_AA)
            cv2.imshow(f"ALPR [{camera_id}]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            continue

        # ── YOLO inference ───────────────────────────────────────────────
        try:
            results = model(frame, verbose=False)[0]
        except Exception as exc:
            print(f"[{camera_id}] YOLO error: {exc}")
            continue

        fh, fw     = frame.shape[:2]
        raw_boxes  = []
        yolo_confs = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yc              = float(box.conf[0])
            if yc < YOLO_CONF_THRESHOLD:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            raw_boxes.append((x1, y1, x2, y2))
            yolo_confs.append(yc)

        # [v11-4] Car presence: YOLO detected anything?
        any_detected = len(raw_boxes) > 0
        last_n_det.append(any_detected)
        # Car is present if any detection in recent window
        car_in_frame = any(last_n_det)
        gate.update_car_presence(car_in_frame)

        track_assignments = tracker.update(raw_boxes)

        def _get_yolo_conf(box):
            try:
                idx = raw_boxes.index(box)
                return yolo_confs[idx]
            except (ValueError, IndexError):
                return YOLO_CONF_THRESHOLD

        for tid, (x1, y1, x2, y2) in track_assignments:
            yolo_conf = _get_yolo_conf((x1, y1, x2, y2))
            stab      = _stab(tid)
            voter     = _voter(tid)

            # ── Stability gate ────────────────────────────────────────────
            if not stab.is_stable(x1, y1, x2, y2):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 210), 1)
                cv2.putText(frame, f"#{tid} stabilising…", (x1, max(y1-6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 210), 1, cv2.LINE_AA)
                continue

            # ── [PF-1] Skip OCR if plate barely moved ─────────────────────
            last_box = tracker.get_last_ocr_box(tid)
            cx, cy   = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if last_box is not None:
                lx1, ly1, lx2, ly2 = last_box
                drift = math.hypot(cx - (lx1 + lx2) / 2,
                                   cy - (ly1 + ly2) / 2)
                if drift < OCR_SKIP_DRIFT:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    # Still check LLM cache while skipping OCR
                    llm_result = llm.query(
                        track_id         = tid,
                        ocr_candidates   = list(_hist(tid)),
                        prior_consonants = voter._consonant_candidates,
                    )
                    if llm_result is not None:
                        plate_text, llm_conf = llm_result
                        now = time.time()
                        if (is_valid_plate(plate_text)
                                and _check_and_register_plate(plate_text, camera_id, now)):
                            province_info = tracker.get_province_cache(tid)
                            if province_info is False:
                                province_info = None
                            voter.reset()
                            stab.reset()
                            tracker.reset_track(tid)
                            _cleanup_track(tid)
                            publish_and_trigger(
                                frame, tid, x1, y1, x2, y2,
                                plate_text, llm_conf, yolo_conf,
                                province_info, camera_id, role, gate,
                                source="llm_fallback",
                            )
                    continue

            tracker.set_last_ocr_box(tid, (x1, y1, x2, y2))

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # ── Step 1: deskew ────────────────────────────────────────────
            plate_crop = deskew_plate(plate_crop)

            # ── Step 2: preprocess ────────────────────────────────────────
            try:
                preprocessed = preprocess_plate_image(
                    plate_crop, scale=PREPROCESS_SCALE, number_zone_only=True)
            except Exception as exc:
                print(f"  [#{tid}] Preprocess error: {exc}")
                continue

            # ── Step 3: three OCR paths ───────────────────────────────────
            with _ocr_semaphore:
                easy_prep = run_easyocr(reader, preprocessed)
                easy_raw  = run_easyocr_raw(reader, plate_crop)
                tess      = run_tesseract(preprocessed)

            _add_history(tid, easy_prep + easy_raw + tess)

            min_conf = adaptive_min_confidence(yolo_conf)
            fused    = fuse_ocr_results(
                easy_prep + easy_raw, tess, min_confidence=min_conf)

            print(f"  [#{tid}@{camera_id}] "
                  f"EasyOCR-prep: {[(r[1], round(float(r[2]),2)) for r in easy_prep]}")
            print(f"  [#{tid}@{camera_id}] "
                  f"Tesseract:    {[(r[1], round(float(r[2]),2)) for r in tess]}")

            # ── Step 4: province detection ────────────────────────────────
            province_info = tracker.get_province_cache(tid)
            if province_info is None:
                all_toks      = easy_prep + easy_raw + tess
                province_info = extract_province_from_ocr_tokens(all_toks)
                if province_info is None:
                    province_info = extract_province_from_crop(
                        plate_crop, easyocr_reader=reader)
                tracker.set_province_cache(tid, province_info or False)
            elif province_info is False:
                province_info = None

            # ── Step 5: best plate read ───────────────────────────────────
            candidate = extract_best_plate_read(
                fused, min_confidence=min_conf,
                max_correction_score=MAX_CORRECTION_SCORE)
            print(f"  [#{tid}@{camera_id}] Candidate: {candidate}")

            if candidate is not None:
                save_debug(f"{camera_id}_tid{tid}", plate_crop, preprocessed)

            # ── Step 6: weighted majority vote ────────────────────────────
            winner = voter.update(candidate, yolo_conf=yolo_conf)

            if winner is not None:
                plate_text, ocr_conf = winner
                now = time.time()

                if not is_valid_plate(plate_text):
                    print(f"  [#{tid}] Voter winner invalid: {plate_text!r}")
                    continue

                if not _check_and_register_plate(plate_text, camera_id, now):
                    voter.reset()
                    stab.reset()
                    tracker.reset_track(tid)
                    continue

                prov_pub = tracker.get_province_cache(tid)
                if prov_pub is False:
                    prov_pub = None
                voter.reset()
                stab.reset()
                tracker.reset_track(tid)
                _cleanup_track(tid)

                publish_and_trigger(
                    frame, tid, x1, y1, x2, y2,
                    plate_text, ocr_conf, yolo_conf,
                    prov_pub, camera_id, role, gate,
                    source="voter",
                )

            else:
                # ── LLM fallback ──────────────────────────────────────────
                llm_result = llm.query(
                    track_id         = tid,
                    ocr_candidates   = list(_hist(tid)),
                    prior_consonants = voter._consonant_candidates,
                )
                if llm_result is not None:
                    plate_text, llm_conf = llm_result
                    now = time.time()

                    if not is_valid_plate(plate_text):
                        print(f"  [#{tid}] LLM invalid: {plate_text!r}")
                    elif not _check_and_register_plate(plate_text, camera_id, now):
                        voter.reset()
                        stab.reset()
                        tracker.reset_track(tid)
                        _cleanup_track(tid)
                    else:
                        prov_pub = tracker.get_province_cache(tid)
                        if prov_pub is False:
                            prov_pub = None
                        voter.reset()
                        stab.reset()
                        tracker.reset_track(tid)
                        _cleanup_track(tid)

                        publish_and_trigger(
                            frame, tid, x1, y1, x2, y2,
                            plate_text, llm_conf, yolo_conf,
                            prov_pub, camera_id, role, gate,
                            source="llm_fallback",
                        )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # ── Overlay: gate state + occupancy ──────────────────────────────
        gate_label, gate_colour = gate.overlay_label()
        cv2.putText(frame, gate_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_colour, 2, cv2.LINE_AA)
        with _occupancy_lock:
            occ = _parking_occupied
        cv2.putText(frame, f"Parking: {occ}/{TOTAL_PARKING_SLOTS}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 80), 1, cv2.LINE_AA)

        # Low-light indicator
        if is_dark:
            cv2.putText(frame, "🌙 LOW LIGHT", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2, cv2.LINE_AA)

        cv2.imshow(f"ALPR [{camera_id}]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ══════════════════════════════════════════════════════════════════════════
# 11. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

print("Starting camera threads…")
threads = []
for cfg in CAMERA_CONFIGS:
    t = threading.Thread(
        target   = camera_thread,
        args     = (cfg,),
        daemon   = True,
        name     = f"cam-{cfg['camera_id']}",
    )
    t.start()
    threads.append(t)
    print(f"  ✅ Thread started: {cfg['camera_id']} ({cfg['role']})")

print(f"\n🅿️  Smart Parking Gate ALPR running — {len(threads)} camera(s) active")
print("   Press Q in any camera window to stop all threads.")

# Keep main thread alive; wait for threads (they run forever unless Q pressed)
try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("\n[Main] Interrupted — shutting down…")

# ── Cleanup ───────────────────────────────────────────────────────────────
print("[Main] Releasing all gates…")
for gate in gate_controllers.values():
    gate.force_release()

cv2.destroyAllWindows()

try:
    with _rmq_lock:
        if rmq_conn and rmq_conn.is_open:
            rmq_conn.close()
except Exception:
    pass

print("[Main] Shutdown complete.")