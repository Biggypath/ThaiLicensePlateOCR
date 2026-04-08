"""
main.py — Thai licence-plate ALPR  (ESP32-CAM → YOLO → OCR → RabbitMQ)

FLOW (v15 — centralized architecture, global plate dedup fixed)
────────────────────────────────────────────────────────────────

  [DD-1]  Global plate dedup — check BEFORE slot.park() so the second
          camera never parks the plate locally either. The plate is added
          to the global set at the moment slot.park() is called, not
          inside _publish_entry(). On REJECTED or EXIT OK the plate is
          removed from the global set so the slot becomes available again.

  [FW-1]  Departure detection removed from Python. Gate board publishes
          MQTT slotState="DEPARTED". Backend drives exit flow.

  [FW-2]  Cam board HTTP: only /open, /close, /keep are called.

  [FW-3]  Keep-alive thread sends /keep to cam board every 30 s.

  [TE-1]  Tesseract data path set from TESSDATA_PREFIX env var if present,
          suppressing repeated error messages when tessdata is missing.
"""

import cv2
import os
import sys
import argparse
import glob
import time
import math
import threading
import queue
import requests
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("[UI-4] Pillow not installed — Thai text will show as '?'.")

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
from camera_registry import (
    get_camera, load_cameras,
    update_camera_plate, reset_all_plates,
)
from slot_presence import SlotState
import gate_controller as _gc_module


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

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
POLL_INTERVAL_S      = 2.0
STATUS_HTTP_TIMEOUT  = 1.5

HF_REPO_ID        = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME = "HurricaneOD_beta.pt"

# [DD-1] Global set — plates currently parked across ALL cameras
_globally_parked_plates: set[str] = set()
_global_plates_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# THAI FONT
# ══════════════════════════════════════════════════════════════════════════

_THAI_FONT_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "NotoSansThai-Regular.ttf"),
    os.path.join(os.path.dirname(__file__), "Tahoma.ttf"),
    os.path.join(os.path.dirname(__file__), "thai_font.ttf"),
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/Library/Fonts/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/tlwg/TlwgTypo.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifThai-Regular.ttf",
]

_thai_font_path: Optional[str] = None
for _fp in _THAI_FONT_CANDIDATES:
    if os.path.isfile(_fp):
        _thai_font_path = _fp
        print(f"[UI-4] Thai font: {_fp}")
        break

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


def put_thai_text(img_bgr: np.ndarray, text: str, pos: tuple,
                  font_size: int = 16,
                  color_bgr: tuple = (240, 240, 240)) -> np.ndarray:
    if all(ord(c) < 128 for c in text):
        cv2.putText(img_bgr, text, (pos[0], pos[1] + font_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0,
                    color_bgr, 1, cv2.LINE_AA)
        return img_bgr
    font = _get_pil_font(font_size)
    if font is None:
        cv2.putText(img_bgr, text.encode("ascii", "replace").decode(),
                    (pos[0], pos[1] + font_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 28.0,
                    color_bgr, 1, cv2.LINE_AA)
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════
# STATUS CACHE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class StatusCache:
    _lock:                threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    gate_open:            Optional[bool]  = None
    slot_state_str:       Optional[str]   = None
    gate_ultrasonic:      Optional[float] = None
    confirm_remaining_ms: Optional[int]   = None

    @staticmethod
    def _extract_us(data: dict) -> Optional[float]:
        for key in ("ultra_distance_cm", "ultrasonic_cm", "distance_cm",
                    "ultrasonic", "us_cm", "dist"):
            raw = data.get(key)
            if raw is not None:
                try:
                    val = float(raw)
                    if val > 0:
                        return val
                except (TypeError, ValueError):
                    pass
        return None

    def update_gate(self, data: dict) -> None:
        with self._lock:
            raw = data.get("arm_down")
            if raw is not None:
                self.gate_open = bool(raw)
            ss = data.get("slot_state")
            if ss is not None:
                self.slot_state_str = str(ss)
            us = self._extract_us(data)
            if us is not None:
                self.gate_ultrasonic = us
            cr = data.get("confirm_remaining_ms")
            if cr is not None:
                try:
                    self.confirm_remaining_ms = int(cr)
                except (TypeError, ValueError):
                    pass

    def snapshot(self):
        with self._lock:
            return (
                self.gate_open,
                self.slot_state_str,
                self.gate_ultrasonic,
                self.confirm_remaining_ms,
            )


# ══════════════════════════════════════════════════════════════════════════
# DETECTION TIMER
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


# ══════════════════════════════════════════════════════════════════════════
# HUD OVERLAY
# ══════════════════════════════════════════════════════════════════════════

_PAD       = 8
_LINE_H    = 24
_FONT_SZ   = 16

_C_BG     = (20,  20,  20)
_C_WHITE  = (240, 240, 240)
_C_GREEN  = ( 50, 220,  50)
_C_RED    = ( 50,  50, 210)
_C_YELLOW = ( 30, 210, 210)
_C_CYAN   = (200, 180,  20)
_C_GREY   = (150, 150, 150)
_C_ORANGE = ( 30, 140, 220)
_C_PURPLE = (200,  80, 200)
_C_PINK   = ( 80, 120, 220)

_LABEL_W        = 84
_VALUE_X_OFFSET = _PAD + _LABEL_W

_SLOT_STATE_COLORS = {
    "BLOCKED":    _C_RED,
    "OPEN":       _C_GREEN,
    "OCCUPIED":   _C_ORANGE,
    "CONFIRMING": _C_YELLOW,
    "EXITING":    _C_PURPLE,
    "DEPARTED":   _C_PINK,
}


def draw_overlay(frame: np.ndarray,
                 slot: SlotState,
                 status_cache: StatusCache,
                 det_timer: DetectionTimer,
                 cam_id: str,
                 slot_id: str) -> np.ndarray:
    gate_open, slot_state_str, gate_us, confirm_rem_ms = status_cache.snapshot()

    rows: list[tuple[str, str, tuple]] = []
    rows.append(("CAM", f"{cam_id}  slot {slot_id}", _C_CYAN))

    if gate_open is None:
        rows.append(("Gate arm", "UNKNOWN", _C_GREY))
    elif gate_open:
        rows.append(("Gate arm", "DOWN (open)", _C_GREEN))
    else:
        rows.append(("Gate arm", "UP (blocked)", _C_RED))

    if slot_state_str:
        fsmcol = _SLOT_STATE_COLORS.get(slot_state_str, _C_WHITE)
        if slot_state_str == "CONFIRMING" and confirm_rem_ms is not None:
            secs = max(0, confirm_rem_ms // 1000)
            fsm_label = f"CONFIRMING  {secs}s"
        else:
            fsm_label = slot_state_str
        rows.append(("Slot FSM", fsm_label, fsmcol))
    else:
        rows.append(("Slot FSM", "---", _C_GREY))

    rows.append((
        "US gate",
        f"{gate_us:.1f} cm" if gate_us is not None else "-.-- cm",
        _C_YELLOW if gate_us is not None else _C_GREY,
    ))

    if slot.is_occupied:
        rows.append(("Py slot", slot.plate, _C_GREEN))
    else:
        rows.append(("Py slot", "FREE", _C_GREY))

    rows.append(("", "-- detection --", _C_GREY))
    rows.append(("YOLO",  f"{det_timer.yolo_ms:.1f} ms",  _C_WHITE))
    rows.append(("OCR",   f"{det_timer.ocr_ms:.1f} ms",   _C_WHITE))
    rows.append(("Total", f"{det_timer.total_ms:.1f} ms", _C_WHITE))

    if det_timer.plate:
        prov_suffix = f"  {det_timer.province}" if det_timer.province else ""
        rows.append(("Plate", det_timer.plate + prov_suffix, _C_GREEN))

    n_rows  = len(rows)
    panel_w = 300
    panel_h = n_rows * _LINE_H + _PAD * 2
    x0, y0  = 8, 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  _C_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for i, (lbl, val, col) in enumerate(rows):
        row_top = y0 + _PAD + i * _LINE_H
        if lbl:
            cv2.putText(frame, lbl,
                        (x0 + _PAD, row_top + _FONT_SZ - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_GREY, 1,
                        cv2.LINE_AA)
        vx = x0 + _VALUE_X_OFFSET if lbl else x0 + _PAD
        frame = put_thai_text(frame, val, pos=(vx, row_top),
                              font_size=_FONT_SZ, color_bgr=col)
    return frame


# ══════════════════════════════════════════════════════════════════════════
# SHARED MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

def cleanup_image_files(directory: str = ".", keep: int = 0) -> int:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in image_exts:
            files.append(path)

    files.sort(key=os.path.getmtime)

    removed = 0
    for path in files[: max(0, len(files) - keep)]:
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:
            print(f"[WARN] Could not remove image file {path}: {exc}")

    return removed


print("Cleaning up old image files...")
_removed_at_start = cleanup_image_files()
if _removed_at_start:
    print(f"Removed {_removed_at_start} image file(s) at startup.")

print("[RS-1] Resetting all currentPlate values in cameras.json ...")
reset_all_plates()

print("Loading EasyOCR (Thai/English)...")
_shared_reader = easyocr.Reader(['th', 'en'], gpu=False)

print("Downloading/Loading YOLO model...")
_model_path   = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
_shared_model = YOLO(_model_path)


# ══════════════════════════════════════════════════════════════════════════
# STREAM HELPERS
# ══════════════════════════════════════════════════════════════════════════

def open_stream(url: str):
    cap = cv2.VideoCapture(url)
    return cap if cap.isOpened() else None


def reconnect_stream(old_cap, url: str):
    if old_cap:
        old_cap.release()
    time.sleep(STREAM_RETRY_DELAY)
    return open_stream(url)


# ══════════════════════════════════════════════════════════════════════════
# CAMERA WORKER
# ══════════════════════════════════════════════════════════════════════════

class CameraWorker:

    def __init__(self, cam_cfg, model, reader):
        self.cfg      = cam_cfg
        self.model    = model
        self.reader   = reader

        self.cam_id    = cam_cfg.cam_id
        self.lot_id    = cam_cfg.lot_id
        self.slot_id   = cam_cfg.slot_id
        self.esp32_url = cam_cfg.stream_url
        self.flip_code = cam_cfg.flip_code
        self.gate_url  = cam_cfg.gate_url
        self.cam_url   = cam_cfg.cam_url

        self.status_cache = StatusCache()
        self.det_timer    = DetectionTimer()
        self.slot         = SlotState()

        self._stop_poller  = threading.Event()
        self._debug_count  = 0
        self._last_seen: dict[str, float] = {}

        self._tracker:     PlateTracker = PlateTracker()
        self._voters:      dict[int, PlateMajorityVoter] = {}
        self._stabilities: dict[int, PlateStabilityGate] = {}

        self.rmq_conn    = None
        self.rmq_channel = None

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.running = True

    # ── RabbitMQ ──────────────────────────────────────────────────────────

    def _connect_rmq(self):
        try:
            if self.rmq_conn and self.rmq_conn.is_open:
                self.rmq_conn.close()
        except Exception:
            pass
        self.rmq_conn, self.rmq_channel = connect()

    def _publish_entry_rmq(self, plate: str, province: str):
        """Publish to RabbitMQ — global dedup already checked before calling."""
        try:
            publish_entry_event(self.rmq_channel, plate, province,
                                self.lot_id, self.slot_id, self.cam_id)
            print(f"  [{self.cam_id}] Published ENTRY: {plate}  province={province}")
        except Exception:
            print(f"  [{self.cam_id}] RabbitMQ lost — reconnecting...")
            self._connect_rmq()
            publish_entry_event(self.rmq_channel, plate, province,
                                self.lot_id, self.slot_id, self.cam_id)

    def _publish_exit_rmq(self, plate: str, province: str):
        try:
            publish_exit_event(self.rmq_channel, plate, province,
                               self.lot_id, self.slot_id, self.cam_id)
            print(f"  [{self.cam_id}] Published EXIT: {plate}  province={province}")
        except Exception:
            print(f"  [{self.cam_id}] RabbitMQ lost — reconnecting...")
            self._connect_rmq()
            publish_exit_event(self.rmq_channel, plate, province,
                               self.lot_id, self.slot_id, self.cam_id)

    # ── Status poller ─────────────────────────────────────────────────────

    def _status_poller(self):
        while not self._stop_poller.is_set():
            if self.gate_url:
                try:
                    r = requests.get(f"{self.gate_url.rstrip('/')}/status",
                                     timeout=STATUS_HTTP_TIMEOUT)
                    if r.ok:
                        self.status_cache.update_gate(r.json())
                except Exception:
                    pass
            self._stop_poller.wait(POLL_INTERVAL_S)

    # ── Cam board HTTP ─────────────────────────────────────────────────────

    def _cmd_cam(self, endpoint: str) -> bool:
        import urllib.request
        if not self.cam_url:
            return False
        url = f"{self.cam_url.rstrip('/')}{endpoint}"
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                ok = 200 <= resp.getcode() < 300
            print(f"  [{self.cam_id}] Cam {endpoint} → {'OK' if ok else 'FAIL'}")
            return ok
        except Exception as exc:
            print(f"  [{self.cam_id}] Cam {endpoint} → error: {exc}")
            return False

    # ── ACK handlers ──────────────────────────────────────────────────────

    def _handle_entry_ack(self, ack: dict):
        status  = ack.get("status", "")
        reg     = ack.get("registration", "?")
        slot_id = ack.get("slotId", "?")

        if status == "ALLOWED":
            print(f"  [{self.cam_id}] ENTRY ALLOWED: {reg}  slot={slot_id}")
            self._cmd_cam("/open")
        else:
            reason = ack.get("reason", "")
            print(f"  [{self.cam_id}] ENTRY {status}: {reg}  reason={reason}")
            self._cmd_cam("/close")
            # [DD-1] Remove from global set — entry rejected
            with _global_plates_lock:
                _globally_parked_plates.discard(reg)
            # Clear local slot state
            if self.slot.is_occupied and self.slot.plate == reg:
                self.slot.clear()
                update_camera_plate(self.cam_id, None)

    def _handle_exit_ack(self, ack: dict):
        status = ack.get("status", "")
        reg    = ack.get("registration", "?")
        fee    = ack.get("totalFee", "?")
        mins   = ack.get("durationMinutes", "?")

        if status == "OK":
            print(f"  [{self.cam_id}] EXIT OK: {reg}  fee={fee}  {mins}min")
            self._cmd_cam("/open")
            # Clear local slot state
            if self.slot.is_occupied and self.slot.plate == reg:
                self.slot.clear()
                update_camera_plate(self.cam_id, None)
            # [DD-1] Remove from global set
            with _global_plates_lock:
                _globally_parked_plates.discard(reg)
        else:
            print(f"  [{self.cam_id}] EXIT {status}: {reg}")

    # ── Per-track helpers ─────────────────────────────────────────────────

    def _get_voter(self, tid: int) -> PlateMajorityVoter:
        if tid not in self._voters:
            self._voters[tid] = PlateMajorityVoter(
                window=VOTE_WINDOW, min_votes=VOTE_MIN_VOTES)
        return self._voters[tid]

    def _get_stability(self, tid: int) -> PlateStabilityGate:
        if tid not in self._stabilities:
            self._stabilities[tid] = PlateStabilityGate(
                min_stable_frames=STABILITY_MIN_FRAMES,
                max_pixel_drift=STABILITY_MAX_DRIFT)
        return self._stabilities[tid]

    def _save_debug(self, tag: str, raw_crop, preprocessed):
        if not DEBUG_SAVE_CROPS or self._debug_count >= DEBUG_MAX_SAVES:
            return
        ts = int(time.time() * 1000)
        cv2.imwrite(f"debug_{self.cam_id}_{ts}_{tag}_raw.jpg",  raw_crop)
        cv2.imwrite(f"debug_{self.cam_id}_{ts}_{tag}_prep.jpg", preprocessed)
        self._debug_count += 1

    def _push_frame(self, frame: np.ndarray):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    # ── Main detection loop ───────────────────────────────────────────────

    def run(self):
        print(f"[{self.cam_id}] Connecting to RabbitMQ...")
        self._connect_rmq()
        print(f"[{self.cam_id}] RabbitMQ connected.")

        ack_store = start_ack_consumer()

        poller = threading.Thread(target=self._status_poller, daemon=True,
                                  name=f"StatusPoller-{self.cam_id}")
        poller.start()

        _gc_module.set_gate_url(self.gate_url)
        _gc_module.set_cam_url(self.cam_url)
        _gc_module.start_keep_alive()
        _gc_module.start_departure_polling(cam_id=self.cam_id)

        cap       = None
        frame_cnt = 0

        while self.running:
            # Stream connection
            if cap is None or not cap.isOpened():
                print(f"[{self.cam_id}] Connecting to ESP32 stream...")
                cap = open_stream(self.esp32_url)
                if cap is None:
                    print(f"[{self.cam_id}] Stream unavailable, retrying...")
                    time.sleep(STREAM_RETRY_DELAY)
                    continue
                print(f"[{self.cam_id}] Stream connected.")

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[{self.cam_id}] Frame grab failed — reconnecting...")
                cap = reconnect_stream(cap, self.esp32_url)
                continue

            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)

            frame_cnt += 1

            # Drain ACK store for this camera
            while True:
                ack = ack_store.pop_entry_ack()
                if ack is None:
                    break
                if ack.get("camId") == self.cam_id:
                    self._handle_entry_ack(ack)
            while True:
                ack = ack_store.pop_exit_ack()
                if ack is None:
                    break
                if ack.get("camId") == self.cam_id:
                    self._handle_exit_ack(ack)

            if frame_cnt % FRAME_SKIP != 0:
                frame = draw_overlay(frame, self.slot, self.status_cache,
                                     self.det_timer, self.cam_id, self.slot_id)
                self._push_frame(frame)
                continue

            # YOLO
            try:
                t_y0    = time.perf_counter()
                results = self.model(frame, verbose=False)[0]
                yolo_ms = (time.perf_counter() - t_y0) * 1000.0
            except Exception as exc:
                print(f"[{self.cam_id}] YOLO error: {exc}")
                frame = draw_overlay(frame, self.slot, self.status_cache,
                                     self.det_timer, self.cam_id, self.slot_id)
                self._push_frame(frame)
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

            track_assignments = self._tracker.update(raw_boxes)
            ocr_ms_frame      = 0.0

            for tid, (x1, y1, x2, y2) in track_assignments:
                try:
                    yolo_conf = yolo_confs[raw_boxes.index((x1, y1, x2, y2))]
                except (ValueError, IndexError):
                    yolo_conf = YOLO_CONF_THRESHOLD

                stability = self._get_stability(tid)
                voter     = self._get_voter(tid)

                if not stability.is_stable(x1, y1, x2, y2):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 210), 1)
                    cv2.putText(frame, f"#{tid} stabilising...", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 210, 210), 1, cv2.LINE_AA)
                    continue

                last_box = self._tracker.get_last_ocr_box(tid)
                cx, cy   = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if last_box is not None:
                    lx1, ly1, lx2, ly2 = last_box
                    if math.hypot(cx - (lx1 + lx2) / 2,
                                  cy - (ly1 + ly2) / 2) < OCR_SKIP_DRIFT:
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 165, 255), 1)
                        continue

                self._tracker.set_last_ocr_box(tid, (x1, y1, x2, y2))

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plate_crop    = deskew_plate(plate_crop)
                province_info = self._tracker.get_province_cache(tid)
                if province_info is None:
                    province_info = extract_province_from_crop(
                        plate_crop, easyocr_reader=self.reader)
                    self._tracker.set_province_cache(
                        tid, province_info or False)
                elif province_info is False:
                    province_info = None

                try:
                    preprocessed = preprocess_plate_image(
                        plate_crop, scale=PREPROCESS_SCALE,
                        number_zone_only=True)
                except Exception as exc:
                    print(f"  [{self.cam_id} #{tid}] Preprocess error: {exc}")
                    continue

                t_o0              = time.perf_counter()
                easy_results_prep = run_easyocr(self.reader, preprocessed)
                easy_results_raw  = run_easyocr_raw(self.reader, plate_crop)
                tess_results      = run_tesseract(preprocessed)
                ocr_ms_frame     += (time.perf_counter() - t_o0) * 1000.0

                min_conf = adaptive_min_confidence(yolo_conf)
                fused    = fuse_ocr_results(
                    easy_results_prep + easy_results_raw,
                    tess_results, min_confidence=min_conf)
                candidate = extract_best_plate_read(
                    fused, min_confidence=min_conf,
                    max_correction_score=MAX_CORRECTION_SCORE)

                if candidate is not None:
                    self._save_debug(f"tid{tid}", plate_crop, preprocessed)

                winner = voter.update(candidate, yolo_conf=yolo_conf)
                if winner is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 165, 255), 2)
                    continue

                plate_text, ocr_conf = winner
                current_time = time.time()

                if not is_valid_plate(plate_text):
                    continue

                if not should_send_plate(plate_text, self._last_seen,
                                         current_time, COOLDOWN_TIME):
                    voter.reset()
                    stability.reset()
                    self._tracker.reset_track(tid)
                    continue

                province_thai = province_info["thai"] if province_info else None
                province_en   = province_info["en"]   if province_info else None
                province_str  = province_thai or ""

                print(f"  ✅ [{self.cam_id} #{tid}] {plate_text}  "
                      f"province={province_thai}  "
                      f"YOLO={yolo_conf:.2f}  OCR={ocr_conf:.2f}  "
                      f"t_yolo={yolo_ms:.0f}ms  t_ocr={ocr_ms_frame:.0f}ms")

                self.det_timer.record(
                    yolo_ms=yolo_ms, ocr_ms=ocr_ms_frame,
                    plate=plate_text, province=province_thai or "")

                self._last_seen[plate_text] = current_time
                voter.reset()
                stability.reset()
                self._tracker.reset_track(tid)

                # ── Slot ENTRY logic  [DD-1] ──────────────────────────────
                if not self.slot.is_occupied:
                    # Check global dedup BEFORE parking locally
                    with _global_plates_lock:
                        if plate_text in _globally_parked_plates:
                            print(f"  [{self.cam_id}] {plate_text} already parked "
                                  f"globally — ignoring")
                            continue
                        # Reserve the plate globally NOW
                        _globally_parked_plates.add(plate_text)

                    # Park locally and publish
                    self.slot.park(plate_text, province_str, current_time)
                    update_camera_plate(self.cam_id, plate_text)
                    self._publish_entry_rmq(plate_text, province_str)

                elif self.slot.plate == plate_text:
                    self.slot.see(current_time)
                else:
                    print(f"  [{self.cam_id} #{tid}] Ignoring {plate_text} "
                          f"(slot occupied by {self.slot.plate})")

                # Bounding box label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = (f"#{tid} {plate_text}  {province_en}"
                         if province_en else f"#{tid} {plate_text}")
                frame = put_thai_text(
                    frame, label,
                    pos=(x1, max(0, y1 - _FONT_SZ - 2)),
                    font_size=_FONT_SZ + 2, color_bgr=(0, 255, 0))

            if self.det_timer.yolo_ms == 0.0:
                self.det_timer.yolo_ms = yolo_ms

            frame = draw_overlay(frame, self.slot, self.status_cache,
                                 self.det_timer, self.cam_id, self.slot_id)
            self._push_frame(frame)

        # Cleanup
        self._stop_poller.set()
        _gc_module.stop_departure_polling()
        if cap:
            cap.release()
        try:
            if self.rmq_conn and self.rmq_conn.is_open:
                self.rmq_conn.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Thai licence-plate ALPR")
parser.add_argument("--camera", "-c", default=None)
parser.add_argument("--cameras", nargs="+", default=None)
args = parser.parse_args()

all_cfgs = load_cameras()

if args.camera:
    cam_cfgs = [get_camera(args.camera)]
elif args.cameras:
    cam_cfgs = [get_camera(cid) for cid in args.cameras]
else:
    cam_cfgs = all_cfgs
    print(f"No --camera specified — running ALL {len(cam_cfgs)} camera(s): "
          f"{[c.cam_id for c in cam_cfgs]}")

if not cam_cfgs:
    print("No cameras to run.")
    sys.exit(1)

workers = [CameraWorker(cfg, _shared_model, _shared_reader)
           for cfg in cam_cfgs]

threads = []
for w in workers:
    t = threading.Thread(target=w.run, daemon=True,
                         name=f"CamThread-{w.cam_id}")
    t.start()
    threads.append(t)
    print(f"Started thread for {w.cam_id}")

print("Display loop running on main thread (press Q to quit all)")
while True:
    for w in workers:
        win_name = f"ALPR — {w.cam_id}"
        try:
            frame = w.frame_queue.get_nowait()
            cv2.imshow(win_name, frame)
        except queue.Empty:
            pass

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        for w in workers:
            w.running = False
        break

    if not any(t.is_alive() for t in threads):
        break

cv2.destroyAllWindows()
print("All cameras stopped.")

_removed_at_exit = cleanup_image_files()
if _removed_at_exit:
    print(f"Cleaned up {_removed_at_exit} image file(s) before exit.")