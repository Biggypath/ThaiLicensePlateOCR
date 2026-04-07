"""
main.py — Thai licence-plate ALPR  (ESP32-CAM → YOLO → OCR → RabbitMQ)

FLOW (v11 — overlay: gate status, ultrasonic distance, detection timing)
──────────────────────────────────────────────────────────────────────────

Changes vs v10:
  [UI-1] StatusPoller thread polls gate board GET /status every
         POLL_INTERVAL_S seconds.  camUrl is optional — if absent,
         ultrasonic is read from gateUrl/status only.
  [UI-2] draw_overlay() paints a semi-transparent HUD on every frame:
           • Gate status  (OPEN / CLOSED / UNKNOWN)
           • Slot status  (OCCUPIED plate / FREE)
           • Ultrasonic   (cm value from gate or cam board)
           • Last detection timing (YOLO ms / OCR ms / Total ms)
  [UI-3] Per-iteration timing: t_yolo, t_ocr measured and stored
         in DetectionTimer; shown in overlay even when no plate is won.
  [UI-4] Thai-text rendering via Pillow (PIL) so ก–ฮ characters display
         correctly instead of "??????". Falls back gracefully if no Thai
         font is found (prints a warning at startup).
  All v10 logic retained unchanged.
"""

import cv2
import json
import os
import sys
import argparse
import glob
import time
import math
import threading
import requests
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr
import numpy as np

# ── Pillow Thai-text support [UI-4] ──────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("[UI-4] Pillow not installed — Thai text will show as '?'. "
          "Run: pip install pillow")

load_dotenv()

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
    PlateTracker,
    PlateMajorityVoter,
    PlateStabilityGate,
)
from plate_corrector import is_valid_plate
from rabbitmq import connect, publish_entry_event, publish_exit_event, start_ack_consumer
from camera_registry import get_camera, load_cameras
from slot_presence import SlotState
import gate_controller


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Thai licence-plate ALPR")
parser.add_argument("--camera", "-c", default=None,
                    help="Camera ID from cameras.json (e.g. cam-01)")
args = parser.parse_args()

if args.camera:
    cam_cfg = get_camera(args.camera)
else:
    cameras = load_cameras()
    if not cameras:
        print("No cameras defined in cameras.json")
        sys.exit(1)
    cam_cfg = cameras[0]
    print(f"No --camera specified, defaulting to '{cam_cfg.cam_id}'")

ESP32_URL            = cam_cfg.stream_url
ESP32_FLIP_CODE      = cam_cfg.flip_code
LOT_ID               = cam_cfg.lot_id
CAM_ID               = cam_cfg.cam_id
SLOT_ID              = cam_cfg.slot_id
GATE_URL             = cam_cfg.gate_url                      # gate board HTTP base
CAM_URL              = getattr(cam_cfg, "cam_url", "")       # cam board HTTP base (optional)

COOLDOWN_TIME        = 10
FRAME_SKIP           = 3
YOLO_CONF_THRESHOLD  = 0.35
STREAM_RETRY_DELAY   = 2

PREPROCESS_SCALE     = 3
VOTE_WINDOW          = 8
VOTE_MIN_VOTES       = 3

OCR_SKIP_DRIFT       = 3.0
STABILITY_MIN_FRAMES = 4
STABILITY_MAX_DRIFT  = 6.0
MAX_CORRECTION_SCORE = 0.65

DEBUG_SAVE_CROPS     = True
DEBUG_MAX_SAVES      = 10
_debug_save_count    = 0

HF_REPO_ID           = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME    = "HurricaneOD_beta.pt"

# Status polling
POLL_INTERVAL_S      = 2.0
STATUS_HTTP_TIMEOUT  = 1.5


# ══════════════════════════════════════════════════════════════════════════
# 2. THAI FONT LOADER  [UI-4]
# ══════════════════════════════════════════════════════════════════════════

# Search order: place a font file next to main.py first for portability.
_THAI_FONT_CANDIDATES = [
    # ── drop a font here next to main.py (highest priority) ──────────────
    os.path.join(os.path.dirname(__file__), "NotoSansThai-Regular.ttf"),
    os.path.join(os.path.dirname(__file__), "Tahoma.ttf"),
    os.path.join(os.path.dirname(__file__), "thai_font.ttf"),
    # ── macOS system ──────────────────────────────────────────────────────
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/Library/Fonts/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # ── Linux / Raspberry Pi (apt install fonts-thai-tlwg) ───────────────
    "/usr/share/fonts/truetype/tlwg/TlwgTypo.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]

_thai_font_path: Optional[str] = None
for _fp in _THAI_FONT_CANDIDATES:
    if os.path.isfile(_fp):
        _thai_font_path = _fp
        print(f"[UI-4] Thai font: {_fp}")
        break

if _thai_font_path is None:
    print(
        "[UI-4] WARNING: No Thai font found — Thai characters will render as boxes.\n"
        "  Quick fix (macOS): cp /System/Library/Fonts/Supplemental/Tahoma.ttf .\n"
        "  Quick fix (Linux): sudo apt install fonts-thai-tlwg\n"
        "  Or drop NotoSansThai-Regular.ttf next to main.py."
    )

_pil_font_cache: dict[int, object] = {}


def _get_pil_font(size: int):
    if not _PIL_OK:
        return None
    if size not in _pil_font_cache:
        if _thai_font_path:
            _pil_font_cache[size] = ImageFont.truetype(_thai_font_path, size)
        else:
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]


def put_thai_text(img_bgr: np.ndarray,
                  text: str,
                  pos: tuple,
                  font_size: int = 16,
                  color_bgr: tuple = (240, 240, 240)) -> np.ndarray:
    """
    [UI-4] Render Unicode/Thai text onto a BGR numpy array via PIL.
    Returns the (possibly modified) array.  Falls back to cv2.putText
    for ASCII-only strings or when PIL is unavailable.
    """
    # Fast path: if every character is ASCII, cv2 handles it fine
    if all(ord(c) < 128 for c in text):
        cv2.putText(img_bgr, text, (pos[0], pos[1] + font_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0,
                    color_bgr, 1, cv2.LINE_AA)
        return img_bgr

    font = _get_pil_font(font_size)
    if font is None:
        # PIL unavailable — best effort ASCII fallback
        cv2.putText(img_bgr, text.encode("ascii", "replace").decode(),
                    (pos[0], pos[1] + font_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0,
                    color_bgr, 1, cv2.LINE_AA)
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    # PIL fill = RGB
    draw.text(pos, text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════
# 3. STATUS CACHE  [UI-1]
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class StatusCache:
    """Thread-safe cache for gate board / cam board /status responses."""
    _lock:           threading.Lock = field(default_factory=threading.Lock, repr=False)

    gate_open:       Optional[bool]  = None   # True=OPEN, False=CLOSED, None=unknown
    gate_ultrasonic: Optional[float] = None   # cm — gate board
    cam_ultrasonic:  Optional[float] = None   # cm — cam board (optional)
    notify_pending:  Optional[bool]  = None

    @staticmethod
    def _extract_us(data: dict) -> Optional[float]:
        """Try multiple common key names for ultrasonic distance."""
        for key in ("ultrasonic_cm", "distance_cm", "ultrasonic", "us_cm", "dist"):
            raw = data.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        return None

    def update_gate(self, data: dict):
        with self._lock:
            raw = data.get("gate_open")
            if raw is not None:
                self.gate_open = bool(raw)
            us = self._extract_us(data)
            if us is not None:
                self.gate_ultrasonic = us

    def update_cam(self, data: dict):
        with self._lock:
            us = self._extract_us(data)
            if us is not None:
                self.cam_ultrasonic = us
            raw = data.get("notify_clear_pending")
            if raw is not None:
                self.notify_pending = bool(raw)

    def snapshot(self):
        with self._lock:
            return (
                self.gate_open,
                self.gate_ultrasonic,
                self.cam_ultrasonic,
                self.notify_pending,
            )


status_cache = StatusCache()
_stop_poller = threading.Event()


def _status_poller():
    """[UI-1] Background thread — polls /status endpoints."""
    while not _stop_poller.is_set():
        if GATE_URL:
            try:
                r = requests.get(f"{GATE_URL.rstrip('/')}/status",
                                 timeout=STATUS_HTTP_TIMEOUT)
                if r.ok:
                    status_cache.update_gate(r.json())
            except Exception:
                pass

        if CAM_URL:
            try:
                r = requests.get(f"{CAM_URL.rstrip('/')}/status",
                                 timeout=STATUS_HTTP_TIMEOUT)
                if r.ok:
                    status_cache.update_cam(r.json())
            except Exception:
                pass

        _stop_poller.wait(POLL_INTERVAL_S)


_poller_thread = threading.Thread(target=_status_poller, daemon=True, name="StatusPoller")
_poller_thread.start()
print(f"[UI] Status poller started  gate={GATE_URL}  "
      f"cam={CAM_URL or '(none)'}  interval={POLL_INTERVAL_S}s")


# ══════════════════════════════════════════════════════════════════════════
# 4. DETECTION TIMER  [UI-3]
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionTimer:
    yolo_ms:  float = 0.0
    ocr_ms:   float = 0.0
    total_ms: float = 0.0
    plate:    str   = ""
    province: str   = ""

    def record(self, yolo_ms: float, ocr_ms: float,
               plate: str = "", province: str = ""):
        self.yolo_ms  = yolo_ms
        self.ocr_ms   = ocr_ms
        self.total_ms = yolo_ms + ocr_ms
        self.plate    = plate
        self.province = province


det_timer = DetectionTimer()


# ══════════════════════════════════════════════════════════════════════════
# 5. HUD OVERLAY  [UI-2]
# ══════════════════════════════════════════════════════════════════════════

_PAD        = 8
_LINE_H     = 24
_FONT_SZ    = 16   # PIL font size — keep in sync with _LINE_H

# BGR colour palette
_C_BG      = (20,  20,  20)
_C_WHITE   = (240, 240, 240)
_C_GREEN   = ( 50, 220,  50)
_C_RED     = ( 50,  50, 210)
_C_YELLOW  = ( 30, 210, 210)
_C_CYAN    = (200, 180,  20)
_C_GREY    = (150, 150, 150)
_C_ORANGE  = ( 30, 140, 220)

# Fixed column widths (px)
_LABEL_W   = 72
_VALUE_X_OFFSET = _PAD + _LABEL_W


def draw_overlay(frame: np.ndarray, slot: SlotState) -> np.ndarray:
    """
    [UI-2] Semi-transparent HUD — top-left corner.
    All text rendered via put_thai_text so Thai chars display correctly.
    """
    gate_open, gate_us, cam_us, _ = status_cache.snapshot()

    # ── Build rows: (label_str, value_str, value_colour) ─────────────────
    rows: list[tuple[str, str, tuple]] = []

    rows.append(("CAM", f"{CAM_ID}  slot {SLOT_ID}", _C_CYAN))

    if gate_open is None:
        rows.append(("Gate", "UNKNOWN", _C_GREY))
    elif gate_open:
        rows.append(("Gate", "OPEN", _C_GREEN))
    else:
        rows.append(("Gate", "CLOSED", _C_RED))

    rows.append((
        "US gate",
        f"{gate_us:.1f} cm" if gate_us is not None else "-.-- cm",
        _C_YELLOW if gate_us is not None else _C_GREY,
    ))

    # Cam ultrasonic row only if cam_url is set
    if CAM_URL:
        rows.append((
            "US cam",
            f"{cam_us:.1f} cm" if cam_us is not None else "-.-- cm",
            _C_YELLOW if cam_us is not None else _C_GREY,
        ))

    if slot.is_occupied:
        rows.append(("Slot", slot.plate, _C_GREEN))
    else:
        rows.append(("Slot", "FREE", _C_GREY))

    rows.append(("", "-- detection --", _C_GREY))

    rows.append(("YOLO",  f"{det_timer.yolo_ms:.1f} ms",  _C_WHITE))
    rows.append(("OCR",   f"{det_timer.ocr_ms:.1f} ms",   _C_WHITE))
    rows.append(("Total", f"{det_timer.total_ms:.1f} ms", _C_WHITE))

    if det_timer.plate:
        prov_suffix = f"  {det_timer.province}" if det_timer.province else ""
        rows.append(("Plate", det_timer.plate + prov_suffix, _C_GREEN))

    # ── Panel size ────────────────────────────────────────────────────────
    n_rows  = len(rows)
    panel_w = 260
    panel_h = n_rows * _LINE_H + _PAD * 2
    x0, y0  = 8, 8

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  _C_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # ── Draw each row ─────────────────────────────────────────────────────
    for i, (lbl, val, col) in enumerate(rows):
        row_top = y0 + _PAD + i * _LINE_H   # top of this row's text cell

        # Label (ASCII — cv2 is fine)
        if lbl:
            cv2.putText(frame, lbl,
                        (x0 + _PAD, row_top + _FONT_SZ - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, _C_GREY, 1, cv2.LINE_AA)

        # Value — may contain Thai, use PIL [UI-4]
        vx = x0 + _VALUE_X_OFFSET if lbl else x0 + _PAD
        frame = put_thai_text(frame, val,
                              pos=(vx, row_top),
                              font_size=_FONT_SZ,
                              color_bgr=col)

    return frame


# ══════════════════════════════════════════════════════════════════════════
# 6. SLOT STATE
# ══════════════════════════════════════════════════════════════════════════

slot = SlotState()

print("Connecting to RabbitMQ...")
try:
    rmq_conn, rmq_channel = connect()
    print(f"RabbitMQ connected!  lot={LOT_ID}  cam={CAM_ID}  slot={SLOT_ID}")
except Exception as exc:
    print(f"RabbitMQ failed: {exc}")
    raise SystemExit(1)


def _reconnect_rmq():
    global rmq_conn, rmq_channel
    try:
        if rmq_conn and rmq_conn.is_open:
            rmq_conn.close()
    except Exception:
        pass
    rmq_conn, rmq_channel = connect()
    return rmq_conn, rmq_channel


def _publish_exit(plate: str, province: str) -> None:
    global rmq_conn, rmq_channel
    try:
        publish_exit_event(rmq_channel, plate, province, LOT_ID, CAM_ID)
        print(f"  Published EXIT: {plate}  province={province}")
    except Exception:
        print("RabbitMQ lost during exit publish — reconnecting...")
        _reconnect_rmq()
        publish_exit_event(rmq_channel, plate, province, LOT_ID, CAM_ID)


def on_departure(cam_id: str) -> None:
    """[FLOW-3] Called by gate_controller poll thread on 60-s clear."""
    if not slot.is_occupied:
        print(f"[Departure] {cam_id} signal but slot not occupied — ignoring")
        return
    departed_plate    = slot.plate
    departed_province = slot.province or ""
    print(f"Car departed (ultrasonic confirmed): {departed_plate}")
    _publish_exit(departed_plate, departed_province)
    slot.clear()


# ══════════════════════════════════════════════════════════════════════════
# 7. INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

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
model      = YOLO(model_path)

print("Starting gate controller...")
gate_controller.set_gate_url(GATE_URL)
gate_controller.set_cam_url(CAM_URL)
gate_controller.set_departure_callback(on_departure)

ack_store = start_ack_consumer()

gate_controller.start_departure_polling(
    cam_id=CAM_ID,
    publish_exit_fn=None,
    rmq_channel_getter=None,
)
print(f"[FLOW] Departure polling started → {CAM_URL or GATE_URL}")


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
    if not DEBUG_SAVE_CROPS or _debug_save_count >= DEBUG_MAX_SAVES:
        return
    ts = int(time.time() * 1000)
    cv2.imwrite(f"debug_plate_{ts}_{tag}_raw.jpg",  raw_crop)
    cv2.imwrite(f"debug_plate_{ts}_{tag}_prep.jpg", preprocessed)
    _debug_save_count += 1


# ══════════════════════════════════════════════════════════════════════════
# 9. PER-TRACK STATE
# ══════════════════════════════════════════════════════════════════════════

_track_voters:    dict[int, PlateMajorityVoter] = {}
_track_stability: dict[int, PlateStabilityGate] = {}


def _get_voter(tid: int) -> PlateMajorityVoter:
    if tid not in _track_voters:
        _track_voters[tid] = PlateMajorityVoter(
            window=VOTE_WINDOW, min_votes=VOTE_MIN_VOTES)
    return _track_voters[tid]


def _get_stability(tid: int) -> PlateStabilityGate:
    if tid not in _track_stability:
        _track_stability[tid] = PlateStabilityGate(
            min_stable_frames=STABILITY_MIN_FRAMES,
            max_pixel_drift=STABILITY_MAX_DRIFT,
        )
    return _track_stability[tid]


def _get_yolo_conf(raw_boxes, yolo_confs, box):
    try:
        return yolo_confs[raw_boxes.index(box)]
    except (ValueError, IndexError):
        return YOLO_CONF_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════
# 10. MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

cap              = None
frame_count      = 0
last_seen_plates: dict[str, float] = {}
tracker          = PlateTracker()

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

    if ESP32_FLIP_CODE is not None:
        frame = cv2.flip(frame, ESP32_FLIP_CODE)

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        frame = draw_overlay(frame, slot)
        cv2.imshow("ESP32-CAM ALPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── YOLO ───────────────────────────────────────────────────────────────
    try:
        t_y0    = time.perf_counter()
        results = model(frame, verbose=False)[0]
        yolo_ms = (time.perf_counter() - t_y0) * 1000.0
    except Exception as exc:
        print(f"YOLO error: {exc}")
        frame = draw_overlay(frame, slot)
        cv2.imshow("ESP32-CAM ALPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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

    track_assignments = tracker.update(raw_boxes)
    ocr_ms_frame      = 0.0

    for tid, (x1, y1, x2, y2) in track_assignments:
        yolo_conf = _get_yolo_conf(raw_boxes, yolo_confs, (x1, y1, x2, y2))
        stability = _get_stability(tid)
        voter     = _get_voter(tid)

        if not stability.is_stable(x1, y1, x2, y2):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 210), 1)
            cv2.putText(frame, f"#{tid} stabilising...", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 210), 1, cv2.LINE_AA)
            continue

        last_box = tracker.get_last_ocr_box(tid)
        cx, cy   = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if last_box is not None:
            lx1, ly1, lx2, ly2 = last_box
            if math.hypot(cx - (lx1 + lx2) / 2,
                          cy - (ly1 + ly2) / 2) < OCR_SKIP_DRIFT:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                continue

        tracker.set_last_ocr_box(tid, (x1, y1, x2, y2))

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        plate_crop    = deskew_plate(plate_crop)
        province_info = tracker.get_province_cache(tid)
        if province_info is None:
            province_info = extract_province_from_crop(
                plate_crop, easyocr_reader=reader)
            tracker.set_province_cache(tid, province_info or False)
        elif province_info is False:
            province_info = None

        try:
            preprocessed = preprocess_plate_image(
                plate_crop, scale=PREPROCESS_SCALE, number_zone_only=True)
        except Exception as exc:
            print(f"  [#{tid}] Preprocess error: {exc}")
            continue

        # ── OCR (timed) ───────────────────────────────────────────────────
        t_o0              = time.perf_counter()
        easy_results_prep = run_easyocr(reader, preprocessed)
        easy_results_raw  = run_easyocr_raw(reader, plate_crop)
        tess_results      = run_tesseract(preprocessed)
        ocr_ms_frame     += (time.perf_counter() - t_o0) * 1000.0

        min_conf = adaptive_min_confidence(yolo_conf)

        fused = fuse_ocr_results(
            easy_results_prep + easy_results_raw,
            tess_results,
            min_confidence=min_conf,
        )

        candidate = extract_best_plate_read(
            fused,
            min_confidence=min_conf,
            max_correction_score=MAX_CORRECTION_SCORE,
        )

        if candidate is not None:
            save_debug(f"tid{tid}", plate_crop, preprocessed)

        winner = voter.update(candidate, yolo_conf=yolo_conf)
        if winner is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            continue

        plate_text, ocr_conf = winner
        current_time = time.time()

        if not is_valid_plate(plate_text):
            print(f"  [#{tid}] Voter winner failed validation: {plate_text!r}")
            continue

        if not should_send_plate(plate_text, last_seen_plates,
                                 current_time, COOLDOWN_TIME):
            print(f"  [#{tid}] [{plate_text}] in cooldown (ENTRY guard).")
            voter.reset()
            stability.reset()
            tracker.reset_track(tid)
            continue

        province_thai = province_info["thai"] if province_info else None
        province_en   = province_info["en"]   if province_info else None
        province_str  = province_thai or ""

        print(f"  [#{tid}] {plate_text}  "
              f"province={province_thai} ({province_en})  "
              f"YOLO={yolo_conf:.2f}  OCR={ocr_conf:.2f}  "
              f"t_yolo={yolo_ms:.0f}ms  t_ocr={ocr_ms_frame:.0f}ms")

        # [UI-3] Record timing
        det_timer.record(yolo_ms=yolo_ms, ocr_ms=ocr_ms_frame,
                         plate=plate_text, province=province_thai or "")

        last_seen_plates[plate_text] = current_time
        voter.reset()
        stability.reset()
        tracker.reset_track(tid)

        # ── Slot ENTRY logic [FLOW-1] ─────────────────────────────────────
        if not slot.is_occupied:
            slot.park(plate_text, province_str, current_time)
            gate_controller._cmd_cam("/car")
            try:
                publish_entry_event(rmq_channel, plate_text,
                                    province_str, LOT_ID, CAM_ID)
                print(f"  Published ENTRY: {plate_text}  province={province_str}")
            except Exception:
                print("RabbitMQ lost — reconnecting...")
                _reconnect_rmq()
                publish_entry_event(rmq_channel, plate_text,
                                    province_str, LOT_ID, CAM_ID)

        elif slot.plate == plate_text:
            slot.see(current_time)

        else:
            print(f"  [#{tid}] Ignoring {plate_text} "
                  f"(slot occupied by {slot.plate})")

        # Bounding box + label (Thai via PIL) [UI-4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (f"#{tid} {plate_text}  {province_en}"
                 if province_en else f"#{tid} {plate_text}")
        frame = put_thai_text(frame, label,
                              pos=(x1, max(0, y1 - _FONT_SZ - 2)),
                              font_size=_FONT_SZ + 2,
                              color_bgr=(0, 255, 0))

    # Keep YOLO timing visible even when no plate is found
    if det_timer.yolo_ms == 0.0:
        det_timer.yolo_ms = yolo_ms

    # ── HUD then display ──────────────────────────────────────────────────
    frame = draw_overlay(frame, slot)
    cv2.imshow("ESP32-CAM ALPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ══════════════════════════════════════════════════════════════════════════
# 11. CLEANUP
# ══════════════════════════════════════════════════════════════════════════

_stop_poller.set()
gate_controller.stop_departure_polling()

if cap:
    cap.release()
cv2.destroyAllWindows()
try:
    if rmq_conn and rmq_conn.is_open:
        rmq_conn.close()
        print("RabbitMQ connection closed.")
except Exception:
    pass