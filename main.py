"""
main.py — Smart Parking Gate ALPR  v15.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fixes from v15.0:
  [BUG-1]  plate = winner.plate  (was wrongly winner.cam_id)
  [BUG-2]  _on_registry_win removed — _wire_registry_callbacks is the
           single clean path; no dead/stub code remains.

Features (unchanged from v15.0):
  [FIX-1]  Best-Camera-Wins challenge window (CLAIM_WINDOW_SEC)
  [FIX-2]  ROI-masked motion + blob filter (waving trees / rain)
  [FIX-3]  TrackedLock + DeadlockWatchdog + strict lock-ordering
"""

import cv2
import json
import math
import os
import glob
import time
import sqlite3
import threading
import traceback
import requests
import numpy as np
from collections import deque, defaultdict, Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr

from plate_utils import (
    extract_best_plate_read,
    fuse_ocr_results,
    adaptive_min_confidence,
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
from llm_plate_corrector import LLMCorrector


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# ── Cameras & Gates ───────────────────────────────────────────────────────
# (stream_url, gate_ip, flip_code, label)
CAMERAS: List[Tuple[str, str, Optional[int], str]] = [
    ("http://172.20.10.4:81/stream", "http://172.20.10.6", 1, "CAM-1"),
    ("http://172.20.10.5:81/stream", "http://172.20.10.7", 1, "CAM-2"),
]

# ── [FIX-2] Per-camera parking spot ROI ──────────────────────────────────
# Pixel rectangle (x1, y1, x2, y2) of the actual parking spot.
# Frame-diff is computed ONLY inside this box — trees / shadows outside
# the box cannot trigger anything.
# Set to None to use the full frame (less precise, easier to start).
#
# To find coordinates: set SHOW_ROI_HELPER = True, run, click two corners
# in each camera window, copy the printed values here, set back to False.
SHOW_ROI_HELPER = False

CAMERA_ROI: Dict[str, Optional[Tuple[int, int, int, int]]] = {
    "CAM-1": None,   # e.g. (120, 80, 520, 400)
    "CAM-2": None,
}

# ── [FIX-2] Per-camera motion thresholds ─────────────────────────────────
# movement_frac : fraction of ROI pixels that must change → car ARRIVED
# exit_frac     : fraction of ROI pixels that must change → car LEFT
# Increase for windy / outdoor / tree cameras.  Decrease for indoor.
CAMERA_MOTION: Dict[str, dict] = {
    "CAM-1": {"movement_frac": 0.04, "exit_frac": 0.06},
    "CAM-2": {"movement_frac": 0.04, "exit_frac": 0.06},
}

# ── [FIX-2] Blob filter ───────────────────────────────────────────────────
# Ignore changed-pixel blobs smaller than this fraction of ROI area.
# Kills raindrops, insects, single-pixel noise.
MIN_BLOB_AREA_FRAC = 0.005   # 0.5 % of ROI

# ── [FIX-1] Best-Camera-Wins challenge window ─────────────────────────────
# All cameras have this many seconds to submit their scored bid.
# After the window the highest composite score wins.
CLAIM_WINDOW_SEC = 1.5

# ── Access control ────────────────────────────────────────────────────────
ALLOW_ALL_PLATES = False   # NEVER True in production

ALLOWED_PLATES: set = {
    # "กข1234",
    # "3กบ7744",
}
ALLOWED_PLATES_DB = "allowed_plates.db"   # "" to disable

# ── Detection ─────────────────────────────────────────────────────────────
COOLDOWN_TIME       = 15
FRAME_SKIP          = 3
YOLO_CONF_THRESHOLD = 0.35
STREAM_RETRY_DELAY  = 2

# ── OCR confidence gate ───────────────────────────────────────────────────
BASE_OCR_CONF  = 0.40
ADAPTIVE_SCALE = 0.20

# ── Stable-detection gate ─────────────────────────────────────────────────
REQUIRED_STABLE_DETECTIONS = 3

# ── OCR pipeline ─────────────────────────────────────────────────────────
PREPROCESS_SCALE      = 3
VOTE_WINDOW           = 8
VOTE_MIN_VOTES        = 3
OCR_SKIP_DRIFT        = 3.0
STABILITY_MIN_FRAMES  = 4
STABILITY_MAX_DRIFT   = 6.0
MAX_CORRECTION_SCORE  = 0.65
LLM_CANDIDATE_HISTORY = 20

# ── Gate timing ───────────────────────────────────────────────────────────
GATE_OPEN_DURATION = 120.0
GATE_HTTP_TIMEOUT  = 3.0
GATE_RETRY_COUNT   = 3
GATE_RETRY_DELAY   = 0.5
MAX_LOCK_TIME      = GATE_OPEN_DURATION + 10.0
DETECTION_LINGER   = 3.0

# ── Low-light ────────────────────────────────────────────────────────────
LOW_LIGHT_THRESHOLD      = 55
LOW_LIGHT_HYSTERESIS     = 70
LOW_LIGHT_CHECK_INTERVAL = 5.0
DARK_OPEN_GATE           = False   # gate never opens without a plate

# ── Car presence ──────────────────────────────────────────────────────────
CAR_CHECK_INTERVAL = 2.0
CAR_ABSENT_FRAMES  = 15

# ── Motion frame-diff ────────────────────────────────────────────────────
MOVEMENT_BLUR_K       = 7     # must be odd
MOVEMENT_THRESH_PIX   = 30
MOTION_CONFIRM_FRAMES = 3

# ── Parking timing ────────────────────────────────────────────────────────
PARK_CONFIRM_WAIT = 60.0
MIN_PARK_SECONDS  = 10.0

# ── YOLO model ────────────────────────────────────────────────────────────
HF_REPO_ID        = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME = "HurricaneOD_beta.pt"

# ── [FIX-3] Lock debug ────────────────────────────────────────────────────
LOG_LOCK_DEBUG     = False
DEADLOCK_TIMEOUT_S = 5.0

# ── Logging ───────────────────────────────────────────────────────────────
EVENT_LOG_FILE = "parking_events.jsonl"

# ── Debug ─────────────────────────────────────────────────────────────────
DEBUG_SAVE_CROPS = False
DEBUG_MAX_SAVES  = 10

# ── RabbitMQ (optional) ───────────────────────────────────────────────────
USE_RABBITMQ = False
RMQ_HOST = "localhost"; RMQ_PORT = 5672
RMQ_USER = "guest";     RMQ_PASS = "guest"
RMQ_QUEUE = "alpr_queue"

PANEL_W = 640


# ══════════════════════════════════════════════════════════════════════════
# 2. INSTRUMENTED LOCK  [FIX-3]
# ══════════════════════════════════════════════════════════════════════════

class TrackedLock:
    """Lock that records holder + acquisition time for the deadlock watchdog."""

    def __init__(self, name: str):
        self._name    = name
        self._lock    = threading.Lock()
        self._held_by: Optional[str] = None
        self._held_at: float         = 0.0
        self._meta    = threading.Lock()

    def acquire(self, blocking=True, timeout=-1):
        caller = threading.current_thread().name
        if LOG_LOCK_DEBUG:
            print(f"  [LOCK] {caller} → ACQUIRE {self._name}")
        result = self._lock.acquire(blocking, timeout)
        if result:
            with self._meta:
                self._held_by = caller
                self._held_at = time.monotonic()
            if LOG_LOCK_DEBUG:
                print(f"  [LOCK] {caller} ✓ HELD {self._name}")
        return result

    def release(self):
        with self._meta:
            self._held_by = None
            self._held_at = 0.0
        if LOG_LOCK_DEBUG:
            print(f"  [LOCK] {threading.current_thread().name}"
                  f" ↑ RELEASE {self._name}")
        self._lock.release()

    def __enter__(self):  self.acquire(); return self
    def __exit__(self, *_): self.release()

    def held_duration(self) -> float:
        with self._meta:
            return (time.monotonic() - self._held_at) if self._held_at else 0.0

    def holder(self) -> Optional[str]:
        with self._meta: return self._held_by


# ══════════════════════════════════════════════════════════════════════════
# 3. DEADLOCK WATCHDOG  [FIX-3]
# ══════════════════════════════════════════════════════════════════════════

_tracked_locks: List[TrackedLock] = []
_tl_meta = threading.Lock()


def _register_lock(lk: TrackedLock) -> TrackedLock:
    with _tl_meta: _tracked_locks.append(lk)
    return lk


def _deadlock_watchdog():
    while True:
        time.sleep(1.0)
        with _tl_meta: locks = list(_tracked_locks)
        for lk in locks:
            dur = lk.held_duration()
            if dur > DEADLOCK_TIMEOUT_S:
                msg = (f"⚠️  [WATCHDOG] Lock '{lk._name}' held by "
                       f"'{lk.holder()}' for {dur:.1f}s")
                print(msg)
                log_event("deadlock_warning", lock=lk._name,
                          holder=lk.holder(), duration=round(dur, 1))


# ══════════════════════════════════════════════════════════════════════════
# 4. EVENT LOGGER
# ══════════════════════════════════════════════════════════════════════════

_log_lock = threading.Lock()


def log_event(event_type: str, **kw):
    rec  = {"ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
             "event": event_type, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    print(f"  [EVENT] {line}")
    if not EVENT_LOG_FILE:
        return
    with _log_lock:
        try:
            with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"  [LOG] {e}")


# ══════════════════════════════════════════════════════════════════════════
# 5. ACCESS CONTROL
# ══════════════════════════════════════════════════════════════════════════

_allowed_cache: Optional[set] = None
_allowed_lock  = threading.Lock()


def _load_db_plates() -> set:
    if not ALLOWED_PLATES_DB or not Path(ALLOWED_PLATES_DB).exists():
        return set()
    try:
        with sqlite3.connect(ALLOWED_PLATES_DB, timeout=5) as con:
            rows = con.execute("SELECT plate FROM allowed_plates").fetchall()
        return {normalize_plate_text(r[0]) for r in rows}
    except Exception as e:
        print(f"  [DB] {e}"); return set()


def reload_allowed_plates():
    global _allowed_cache
    db = _load_db_plates()
    with _allowed_lock:
        _allowed_cache = ALLOWED_PLATES | db
    print(f"  [Access] Whitelist: {len(_allowed_cache)} plates")


def is_plate_allowed(plate: str) -> bool:
    if ALLOW_ALL_PLATES: return True
    norm = normalize_plate_text(plate)
    with _allowed_lock:
        ok = norm in (_allowed_cache or set())
    if not ok:
        print(f"  [Access] {norm} — NOT in whitelist")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# 6. PLATE CANDIDATE  (data class)
# ══════════════════════════════════════════════════════════════════════════

class PlateCandidate:
    """Everything a camera knows about a detected plate at claim time."""

    __slots__ = ("cam_id", "plate", "score", "yolo_conf", "ocr_conf",
                 "correction_score", "x1", "y1", "x2", "y2",
                 "prov_info", "source", "frame")

    def __init__(self, cam_id: str, plate: str, score: float,
                 yolo_conf: float, ocr_conf: float, correction_score: float,
                 x1: int, y1: int, x2: int, y2: int,
                 prov_info: Optional[dict], source: str,
                 frame: np.ndarray):
        self.cam_id           = cam_id
        self.plate            = plate            # ← actual plate string
        self.score            = score
        self.yolo_conf        = yolo_conf
        self.ocr_conf         = ocr_conf
        self.correction_score = correction_score
        self.x1, self.y1     = x1, y1
        self.x2, self.y2     = x2, y2
        self.prov_info        = prov_info
        self.source           = source
        self.frame            = frame            # snapshot for drawing


# ══════════════════════════════════════════════════════════════════════════
# 7. GLOBAL PLATE REGISTRY — Best-Camera-Wins  [FIX-1]
# ══════════════════════════════════════════════════════════════════════════

class GlobalPlateRegistry:
    """
    [FIX-1] Best-Camera-Wins coordinator.

    Flow:
      1. submit_claim(plate, candidate)
           → starts a CLAIM_WINDOW_SEC timer (first submission only).
           → subsequent cameras add their bid to the same window.
      2. After the window, _resolve(plate) fires in a Timer thread:
           → picks max(candidates, key=score)
           → calls the winner's registered open-gate callback.
           → lock is RELEASED before calling the callback [FIX-3].

    [FIX-3] ONE TrackedLock guards all mutable state.
            External callbacks (open_gate) are never called while locked.
    """

    def __init__(self):
        self._lock = _register_lock(TrackedLock("Registry"))

        # plate_norm → [PlateCandidate, ...]   (during challenge window)
        self._pending:  Dict[str, List[PlateCandidate]] = {}
        # plate_norm → {"cam_id": str, "ts": float}  (gate currently open)
        self._active:   Dict[str, dict] = {}
        # plate_norm → timestamp of last fire  (global cooldown)
        self._cooldown: Dict[str, float] = {}
        # plate_norm → threading.Timer
        self._timers:   Dict[str, threading.Timer] = {}
        # cam_id → callable(PlateCandidate)
        self._open_cbs: Dict[str, callable] = {}

    def register_open_callback(self, cam_id: str, fn: callable):
        with self._lock:
            self._open_cbs[cam_id] = fn

    def submit_claim(self, candidate: PlateCandidate) -> bool:
        """
        Returns True if the bid was accepted.
        Returns False if plate is in cooldown or the gate is already open.
        """
        now  = time.time()
        norm = normalize_plate_text(candidate.plate)

        with self._lock:
            # Global cooldown
            last = self._cooldown.get(norm, 0.0)
            if now - last < COOLDOWN_TIME:
                rem = COOLDOWN_TIME - (now - last)
                print(f"  [Registry] {norm} cooldown {rem:.1f}s "
                      f"({candidate.cam_id})")
                return False

            # Gate already open for this plate
            if norm in self._active:
                owner = self._active[norm]["cam_id"]
                print(f"  [Registry] {norm} gate already open by {owner}")
                return False

            # Open challenge window on first bid
            if norm not in self._pending:
                self._pending[norm] = []
                t = threading.Timer(CLAIM_WINDOW_SEC,
                                    self._resolve, args=(norm,))
                t.daemon = True
                t.start()
                self._timers[norm] = t
                print(f"  [Registry] {norm} challenge window opened "
                      f"({CLAIM_WINDOW_SEC}s)")

            self._pending[norm].append(candidate)
            print(f"  [Registry] {norm} bid from {candidate.cam_id} "
                  f"score={candidate.score:.3f}")
            return True

    def _resolve(self, norm: str):
        """
        Timer callback — runs in its own thread.
        [FIX-3] Acquires lock → copies data → releases lock → calls callback.
        No external code is ever called while the lock is held.
        """
        with self._lock:
            candidates = self._pending.pop(norm, [])
            self._timers.pop(norm, None)
            if not candidates:
                return

            # Best score wins  [FIX-1]
            winner = max(candidates, key=lambda c: c.score)

            self._active[norm]   = {"cam_id": winner.cam_id, "ts": time.time()}
            self._cooldown[norm] = time.time()

            # Copy callback ref — must NOT call while locked
            cb      = self._open_cbs.get(winner.cam_id)
            losers  = [c.cam_id for c in candidates
                       if c.cam_id != winner.cam_id]

        # ── Lock released ─────────────────────────────────────────────────
        print(f"  [Registry] {norm} WINNER={winner.cam_id} "
              f"score={winner.score:.3f}  losers={losers}")
        log_event("plate_winner", plate=norm,
                  winner=winner.cam_id, score=round(winner.score, 4),
                  losers=losers,
                  yolo=round(winner.yolo_conf, 4),
                  ocr=round(winner.ocr_conf, 4))

        if cb:
            try:
                cb(winner)   # winner.plate is the real plate string [BUG-1 fixed]
            except Exception as e:
                print(f"  [Registry] callback error for {winner.cam_id}: {e}")
                log_event("registry_cb_error", plate=norm, error=str(e))

    def release(self, plate: str, cam_id: str):
        norm = normalize_plate_text(plate)
        with self._lock:
            if self._active.get(norm, {}).get("cam_id") == cam_id:
                del self._active[norm]
                print(f"  [Registry] {norm} RELEASED by {cam_id}")
                log_event("plate_released", plate=norm, camera=cam_id)

    def active_plates(self) -> dict:
        with self._lock:
            return dict(self._active)


# ══════════════════════════════════════════════════════════════════════════
# 8. ROI-MASKED MOTION DETECTION  [FIX-2]
# ══════════════════════════════════════════════════════════════════════════

def _frame_diff_frac(f1: np.ndarray, f2: np.ndarray,
                     roi: Optional[Tuple[int, int, int, int]]) -> float:
    """
    Blur + binary mask inside ROI only.
    Blob filter removes components smaller than MIN_BLOB_AREA_FRAC of ROI.
    """
    h, w = f1.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        c1, c2 = f1[y1:y2, x1:x2], f2[y1:y2, x1:x2]
        roi_area = max((y2-y1) * (x2-x1), 1)
    else:
        c1, c2   = f1, f2
        roi_area = max(h * w, 1)

    if c1.size == 0 or c2.size == 0:
        return 0.0

    g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY)
    b1 = cv2.GaussianBlur(g1, (MOVEMENT_BLUR_K, MOVEMENT_BLUR_K), 0)
    b2 = cv2.GaussianBlur(g2, (MOVEMENT_BLUR_K, MOVEMENT_BLUR_K), 0)
    _, mask = cv2.threshold(cv2.absdiff(b1, b2),
                            MOVEMENT_THRESH_PIX, 255, cv2.THRESH_BINARY)

    # Morphological close: merge nearby changed pixels into blobs
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    # Blob filter: remove small components (raindrops, insects)
    min_px = int(roi_area * MIN_BLOB_AREA_FRAC)
    if min_px > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        filtered = np.zeros_like(mask)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
                filtered[labels == lbl] = 255
        mask = filtered

    return float(np.sum(mask > 0)) / roi_area


# ══════════════════════════════════════════════════════════════════════════
# 9. PARKING MONITOR
# ══════════════════════════════════════════════════════════════════════════

class ParkingMonitor:
    class Phase(Enum):
        IDLE    = "IDLE"
        WAITING = "WAITING"
        PARKED  = "PARKED"
        EXITING = "EXITING"

    def __init__(self, label: str,
                 movement_frac: float, exit_frac: float,
                 roi: Optional[Tuple[int, int, int, int]]):
        self.label     = label
        self._mov_frac = movement_frac
        self._exit_frac= exit_frac
        self._roi      = roi
        self._phase    = self.Phase.IDLE
        self._open_t   = 0.0
        self._park_t   = 0.0
        self._ref: Optional[np.ndarray] = None
        self._consec   = 0
        self._lock     = _register_lock(TrackedLock(f"ParkMon-{label}"))

    @property
    def phase(self):
        with self._lock: return self._phase

    def record_gate_open(self, frame: np.ndarray):
        with self._lock:
            self._phase  = self.Phase.WAITING
            self._open_t = time.time()
            self._park_t = 0.0
            self._ref    = frame.copy()
            self._consec = 0
        log_event("park_phase", camera=self.label, phase="WAITING")

    def update(self, frame: np.ndarray) -> bool:
        """Returns True when gate should close."""
        with self._lock:
            ph  = self._phase; ot = self._open_t; pt = self._park_t
            ref = self._ref;   mf = self._mov_frac; ef = self._exit_frac
            roi = self._roi

        now  = time.time()
        diff = _frame_diff_frac(ref, frame, roi) if ref is not None else 0.0

        if ph == self.Phase.IDLE:
            return False

        if ph == self.Phase.WAITING:
            if ref is None:
                with self._lock: self._ref = frame.copy()
                return False
            if diff >= mf:
                with self._lock: self._consec += 1
                if self._consec >= MOTION_CONFIRM_FRAMES:
                    with self._lock:
                        self._phase  = self.Phase.PARKED
                        self._park_t = now
                        self._ref    = frame.copy()
                        self._consec = 0
                    print(f"  [{self.label} ParkMon] PARKED  diff={diff:.4f}")
                    log_event("park_phase", camera=self.label,
                              phase="PARKED", diff=round(diff, 4))
            else:
                with self._lock: self._consec = 0
            if now - ot > PARK_CONFIRM_WAIT:
                print(f"  [{self.label} ParkMon] TIMEOUT")
                log_event("park_phase", camera=self.label, phase="TIMEOUT")
                with self._lock:
                    self._phase  = self.Phase.IDLE
                    self._consec = 0
                return True
            return False

        if ph == self.Phase.PARKED:
            if ref is None: return False
            if diff >= ef and (now - pt) >= MIN_PARK_SECONDS:
                with self._lock: self._consec += 1
                if self._consec >= MOTION_CONFIRM_FRAMES:
                    with self._lock:
                        self._phase  = self.Phase.EXITING
                        self._consec = 0
                    print(f"  [{self.label} ParkMon] EXITING  "
                          f"diff={diff:.4f}  parked={now-pt:.1f}s")
                    log_event("park_phase", camera=self.label, phase="EXITING",
                              diff=round(diff, 4))
            else:
                with self._lock: self._consec = 0
            return False

        if ph == self.Phase.EXITING:
            with self._lock:
                self._phase  = self.Phase.IDLE
                self._consec = 0
            log_event("park_phase", camera=self.label, phase="EXIT_CONFIRMED")
            return True

        return False

    def reset(self):
        with self._lock:
            self._phase  = self.Phase.IDLE
            self._ref    = None
            self._consec = 0


# ══════════════════════════════════════════════════════════════════════════
# 10. STABLE-DETECTION COUNTER
# ══════════════════════════════════════════════════════════════════════════

class StableDetectionCounter:
    def __init__(self):
        self._c: Dict[int, Counter] = defaultdict(Counter)

    def record(self, tid: int, plate: str): self._c[tid][plate] += 1
    def count(self, tid: int, plate: str):  return self._c[tid][plate]
    def ready(self, tid: int, plate: str):
        return self._c[tid][plate] >= REQUIRED_STABLE_DETECTIONS
    def clear(self, tid: int): self._c.pop(tid, None)


# ══════════════════════════════════════════════════════════════════════════
# 11. GATE CONTROLLER  [FIX-3]
# ══════════════════════════════════════════════════════════════════════════

class GateState(Enum):
    IDLE = "IDLE"; OPEN = "OPEN"; CLOSING = "CLOSING"


class GateController:
    """
    [FIX-3] ONE TrackedLock for own state only.
    Never calls registry while locked.
    acquire_and_open() releases lock before calling park_mon or _fire().
    """

    def __init__(self, gate_ip: str, label: str,
                 park_mon: ParkingMonitor, registry: GlobalPlateRegistry,
                 open_dur: float = GATE_OPEN_DURATION):
        self.gate_ip  = gate_ip
        self.label    = label
        self._pm      = park_mon
        self._reg     = registry
        self.open_dur = open_dur
        self._mu      = _register_lock(TrackedLock(f"Gate-{label}"))
        self._state   = GateState.IDLE
        self._locked  = False
        self._lu      = 0.0
        self._plate   = ""
        self._timer: Optional[threading.Timer] = None
        self._last_det = 0.0
        self._car      = False
        self._dark     = False
        self._car_ts   = 0.0
        self._light_ts = 0.0

    @property
    def state(self): return self._state

    def is_idle(self) -> bool:
        with self._mu: return not self._locked

    def acquire_and_open(self, plate: str,
                         frame: Optional[np.ndarray] = None) -> bool:
        """
        Open gate for plate.
        [FIX-3] Releases own lock before calling park_mon or firing HTTP.
        """
        with self._mu:
            now = time.monotonic()
            if self._locked:
                if now > self._lu + (MAX_LOCK_TIME - self.open_dur):
                    print(f"  [{self.label}] Stale lock cleared")
                    self._locked = False; self._state = GateState.IDLE
                    if self._timer: self._timer.cancel(); self._timer = None
            if self._locked:
                print(f"  [{self.label}] Gate busy "
                      f"{max(0, self._lu-now):.1f}s left")
                return False
            self._locked = True
            self._plate  = plate
            self._state  = GateState.OPEN
            self._lu     = now + self.open_dur
            self._timer  = threading.Timer(self.open_dur, self._auto_release)
            self._timer.daemon = True
            self._timer.start()
            log_event("gate_acquired", camera=self.label, plate=plate)

        # ── Own lock released — safe to call external code ────────────────
        if self._pm and frame is not None:
            self._pm.record_gate_open(frame)
        self._fire("open")
        return True

    def record_detection(self):
        self._last_det = time.time()

    def _auto_release(self):
        while time.time() - self._last_det < DETECTION_LINGER:
            time.sleep(DETECTION_LINGER)
        plate = ""
        with self._mu:
            self._state = GateState.CLOSING
            plate = self._plate
        self._do("close"); time.sleep(1.0); self._do("clear")
        with self._mu:
            self._state  = GateState.IDLE
            self._locked = False
            self._plate  = ""
            self._timer  = None
        # Release AFTER own lock is free — no inversion risk [FIX-3]
        self._reg.release(plate, self.label)
        log_event("gate_auto_released", camera=self.label, plate=plate)
        print(f"  [{self.label}] Auto-released → IDLE")

    def close_now(self, reason: str = "exit"):
        plate = ""
        with self._mu:
            if self._timer: self._timer.cancel(); self._timer = None
            self._state = GateState.CLOSING
            plate = self._plate
        threading.Thread(target=self._close_seq,
                         args=(plate, reason), daemon=True).start()

    def _close_seq(self, plate: str, reason: str):
        self._do("close"); time.sleep(1.0); self._do("clear")
        with self._mu:
            self._state  = GateState.IDLE
            self._locked = False
            self._plate  = ""
        self._reg.release(plate, self.label)
        log_event("gate_closed", camera=self.label, plate=plate, reason=reason)
        print(f"  [{self.label}] Closed ({reason}) → IDLE")

    def force_release(self):
        plate = ""
        with self._mu:
            if self._timer: self._timer.cancel(); self._timer = None
            self._state  = GateState.IDLE
            self._locked = False
            plate = self._plate; self._plate = ""
        if plate: self._reg.release(plate, self.label)

    def _do(self, ep: str) -> bool:
        url = f"{self.gate_ip}/{ep}"
        for i in range(1, GATE_RETRY_COUNT + 1):
            try:
                if requests.post(url,
                                 timeout=GATE_HTTP_TIMEOUT).status_code == 200:
                    return True
            except Exception as e:
                print(f"  [{self.label}] /{ep} err #{i}: {e}")
            if i < GATE_RETRY_COUNT: time.sleep(GATE_RETRY_DELAY)
        return False

    def _fire(self, ep: str):
        threading.Thread(target=self._do, args=(ep,), daemon=True).start()

    def update_car(self, present: bool):
        if time.time() - self._car_ts < CAR_CHECK_INTERVAL: return
        if present != self._car:
            self._car = present; self._fire("car" if present else "clear")
        self._car_ts = time.time()

    def update_light(self, dark: bool):
        if time.time() - self._light_ts < LOW_LIGHT_CHECK_INTERVAL: return
        if dark != self._dark:
            self._dark = dark; self._fire("dark" if dark else "light")
        self._light_ts = time.time()

    def overlay_label(self) -> Tuple[str, tuple]:
        with self._mu:
            st = self._state; lk = self._locked
            lu = self._lu;    pl = self._plate
        rem = max(0.0, lu - time.monotonic()) if lk else 0
        dk  = " [DARK]" if self._dark else ""
        if st == GateState.OPEN:    return (f"OPEN {rem:.0f}s [{pl}]{dk}", (0,255,0))
        if st == GateState.CLOSING: return (f"CLOSING…{dk}", (0,165,255))
        return (f"IDLE{dk}", (180,180,180))


# ══════════════════════════════════════════════════════════════════════════
# 12. HELPERS
# ══════════════════════════════════════════════════════════════════════════

def adaptive_ocr_gate(yolo_conf: float) -> float:
    return BASE_OCR_CONF + ADAPTIVE_SCALE * (1.0 - max(0.35, min(1.0, yolo_conf)))


def detect_low_light(frame: np.ndarray, is_dark: bool) -> bool:
    mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    return mean < (LOW_LIGHT_HYSTERESIS if is_dark else LOW_LIGHT_THRESHOLD)


def composite_score(yolo_conf: float, ocr_conf: float,
                    correction_score: float) -> float:
    """[FIX-1] Higher = better candidate for Best-Camera-Wins."""
    quality = 1.0 - correction_score
    return yolo_conf * 0.4 + ocr_conf * 0.4 + quality * 0.2


# ── RabbitMQ ─────────────────────────────────────────────────────────────
rmq_conn = rmq_channel = None
_rmq_lock = threading.Lock()
if USE_RABBITMQ:
    try:
        import pika
        _conn = pika.BlockingConnection(pika.ConnectionParameters(
            RMQ_HOST, RMQ_PORT, "/",
            pika.PlainCredentials(RMQ_USER, RMQ_PASS)))
        rmq_conn = _conn; rmq_channel = _conn.channel()
        rmq_channel.queue_declare(queue=RMQ_QUEUE, durable=True)
        print("✅ RabbitMQ connected!")
    except Exception as e:
        print(f"⚠️  RabbitMQ disabled: {e}"); USE_RABBITMQ = False


def _publish(payload: dict):
    if not USE_RABBITMQ or not rmq_channel: return
    body = json.dumps(payload, ensure_ascii=False).encode()
    with _rmq_lock:
        try:
            import pika
            rmq_channel.basic_publish(
                exchange="", routing_key=RMQ_QUEUE, body=body,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE))
        except Exception as e: print(f"RabbitMQ: {e}")


_dbg_n = 0; _dbg_lock = threading.Lock()


def save_debug(tag: str, raw: np.ndarray, prep: np.ndarray):
    global _dbg_n
    if not DEBUG_SAVE_CROPS: return
    with _dbg_lock:
        if _dbg_n >= DEBUG_MAX_SAVES: return
        ts = int(time.time() * 1000)
        cv2.imwrite(f"debug_{ts}_{tag}_raw.jpg", raw)
        cv2.imwrite(f"debug_{ts}_{tag}_prep.jpg", prep)
        _dbg_n += 1


# ── ROI helper ────────────────────────────────────────────────────────────
_roi_clicks: Dict[str, list] = {}


def _roi_mouse_cb(event, x, y, flags, param):
    label = param
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = _roi_clicks.setdefault(label, [])
        pts.append((x, y))
        print(f"  [ROI {label}] click #{len(pts)}: ({x}, {y})")
        if len(pts) == 2:
            x1, y1 = pts[0]; x2, y2 = pts[1]
            roi = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            print(f"\n  ✅  CAMERA_ROI[\"{label}\"] = {roi}"
                  f"\n  Copy into config and restart with SHOW_ROI_HELPER=False\n")


# ══════════════════════════════════════════════════════════════════════════
# 13. CAMERA WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    """
    Independent ALPR pipeline per camera.

    [FIX-1] Submits PlateCandidate (with real plate string) to registry.
            Registry resolves winner and calls _open_gate_cb.
    [BUG-1] winner.plate is used everywhere — cam_id is never confused
            with the plate string.
    [BUG-2] _on_registry_win stub removed entirely. The only callback
            path is _open_gate_cb, registered via
            registry.register_open_callback() in __init__.
    [FIX-3] No lock is held when calling registry or gate methods.
    """

    def __init__(self, cam_url, gate_ip, flip, label,
                 model, reader, llm, registry, frame_slot, ocr_sem):
        super().__init__(name=f"worker-{label}", daemon=True)
        self.cam_url  = cam_url
        self.flip     = flip
        self.label    = label
        self.model    = model
        self.reader   = reader
        self.llm      = llm
        self.registry = registry
        self._slot    = frame_slot
        self._ocr_sem = ocr_sem
        self._stop    = threading.Event()

        thr = CAMERA_MOTION.get(label, {})
        roi = CAMERA_ROI.get(label)
        self.park_mon = ParkingMonitor(label,
                                       thr.get("movement_frac", 0.04),
                                       thr.get("exit_frac", 0.06),
                                       roi)
        self.gate     = GateController(gate_ip, label, self.park_mon, registry)
        self.stab_ctr = StableDetectionCounter()

        self._tracker   = PlateTracker()
        self._voters:    Dict[int, PlateMajorityVoter] = {}
        self._stability: Dict[int, PlateStabilityGate] = {}
        self._hist_buf:  Dict[int, deque] = {}
        self._last_det  = deque(maxlen=CAR_ABSENT_FRAMES)
        self._fc        = 0

        # [BUG-2] Single callback path — no stub _on_registry_win
        registry.register_open_callback(label, self._open_gate_cb)

    # ── Registry win callback  ────────────────────────────────────────────

    def _open_gate_cb(self, winner: PlateCandidate):
        """
        Called by GlobalPlateRegistry after the challenge window closes
        and THIS camera's candidate won.

        [BUG-1] winner.plate is the real plate string (e.g. "กข1234").
        [BUG-2] This is the ONLY path — no dead stubs.
        [FIX-3] Registry lock is already released before this is called.
        """
        plate   = winner.plate          # ← correct: actual plate, not cam_id
        frame   = winner.frame

        self.gate.record_detection()
        opened  = self.gate.acquire_and_open(plate=plate, frame=frame)

        prov_en = winner.prov_info["en"] if winner.prov_info else None
        status  = "🟢 GATE OPEN" if opened else "🔴 gate busy"
        print(f"✅ [{self.label}] {plate}  "
              f"YOLO={winner.yolo_conf:.2f}  OCR={winner.ocr_conf:.2f}  "
              f"score={winner.score:.3f}  {status}")

        log_event("plate_detected",
                  camera=self.label, plate=plate, province=prov_en,
                  yolo=round(winner.yolo_conf, 4),
                  ocr=round(winner.ocr_conf, 4),
                  score=round(winner.score, 4),
                  source=winner.source, gate_triggered=opened)

        _publish({"ts": int(time.time()), "camera": self.label,
                  "plate": plate, "province_en": prov_en,
                  "score": round(winner.score, 4),
                  "gate_triggered": opened})

        # Draw on winner's frame snapshot
        colour = (0, 255, 0) if opened else (0, 165, 255)
        lbl    = f"[{self.label}] {plate}"
        if prov_en:  lbl += f" {prov_en}"
        if opened:   lbl += " [OPEN]"
        else:        lbl += " [GATE BUSY]"
        cv2.rectangle(frame, (winner.x1, winner.y1),
                      (winner.x2, winner.y2), colour, 2)
        cv2.putText(frame, lbl, (winner.x1, max(winner.y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    # ── Track helpers ─────────────────────────────────────────────────────

    def _voter(self, tid):
        if tid not in self._voters:
            self._voters[tid] = PlateMajorityVoter(
                window=VOTE_WINDOW, min_votes=VOTE_MIN_VOTES,
                llm_corrector=self.llm, track_id=tid)
        return self._voters[tid]

    def _stab(self, tid):
        if tid not in self._stability:
            self._stability[tid] = PlateStabilityGate(
                min_stable_frames=STABILITY_MIN_FRAMES,
                max_pixel_drift=STABILITY_MAX_DRIFT)
        return self._stability[tid]

    def _hist(self, tid):
        if tid not in self._hist_buf:
            self._hist_buf[tid] = deque(maxlen=LLM_CANDIDATE_HISTORY)
        return self._hist_buf[tid]

    def _add_hist(self, tid, results):
        h = self._hist(tid); seen = set(h)
        for r in results:
            if len(r) >= 2:
                t = str(r[1]).strip()
                if t and t not in seen: h.append(t); seen.add(t)

    def _cleanup(self, tid):
        for d in (self._voters, self._stability, self._hist_buf):
            d.pop(tid, None)
        self.stab_ctr.clear(tid)
        self.llm.invalidate_track(tid)

    # ── Submit plate to registry ───────────────────────────────────────────

    def _submit(self, frame: np.ndarray,
                plate: str, ocr_conf: float, yolo_conf: float,
                correction_score: float, prov_info: Optional[dict],
                source: str,
                x1: int, y1: int, x2: int, y2: int):
        """
        Build PlateCandidate and submit to registry.
        [BUG-1] plate is stored in candidate.plate — never confused with cam_id.
        [FIX-3] No lock held here.
        """
        if not is_plate_allowed(plate):
            log_event("access_denied", camera=self.label, plate=plate)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"[{self.label}] {plate} [DENIED]",
                        (x1, max(y1-8,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2,
                        cv2.LINE_AA)
            return

        score = composite_score(yolo_conf, ocr_conf, correction_score)
        cand  = PlateCandidate(
            cam_id=self.label,
            plate=plate,                 # ← real plate string [BUG-1]
            score=score,
            yolo_conf=yolo_conf,
            ocr_conf=ocr_conf,
            correction_score=correction_score,
            x1=x1, y1=y1, x2=x2, y2=y2,
            prov_info=prov_info,
            source=source,
            frame=frame.copy())

        accepted = self.registry.submit_claim(cand)
        colour   = (0, 200, 255) if accepted else (100, 100, 100)
        lbl      = f"[{self.label}] {plate} {'⏳' if accepted else '(dup)'}"
        cv2.rectangle(frame, (x1,y1),(x2,y2), colour, 2)
        cv2.putText(frame, lbl, (x1, max(y1-8,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    # ── Auto-restart wrapper ──────────────────────────────────────────────

    def run(self):
        print(f"[{self.label}] Worker started  {self.cam_url}")
        while not self._stop.is_set():
            try:
                self._run_inner()
            except Exception:
                tb = traceback.format_exc()
                print(f"\n[{self.label}] ⚠️  CRASH:\n{tb}")
                log_event("worker_crash", camera=self.label, traceback=tb[:500])
                time.sleep(2.0)
        print(f"[{self.label}] Worker stopped.")

    def _run_inner(self):
        cap = None
        win = f"ROI Helper — {self.label}" if SHOW_ROI_HELPER else None
        if SHOW_ROI_HELPER and win:
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, _roi_mouse_cb, self.label)

        while not self._stop.is_set():
            # ── Stream ────────────────────────────────────────────────────
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.cam_url)
                if not cap.isOpened():
                    cap = None; time.sleep(STREAM_RETRY_DELAY); continue
                print(f"[{self.label}] Stream connected.")

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release(); cap = None; continue

            if self.flip is not None:
                frame = cv2.flip(frame, self.flip)

            self._fc += 1

            # Parking monitor
            if self.park_mon.update(frame) and not self.gate.is_idle():
                self.gate.close_now("park_monitor")

            # Low-light
            dark = detect_low_light(frame, self.gate._dark)
            self.gate.update_light(dark)

            # ROI overlay
            roi = CAMERA_ROI.get(self.label)
            if roi:
                cv2.rectangle(frame, (roi[0],roi[1]),(roi[2],roi[3]),
                              (0,255,255), 1)
                cv2.putText(frame, "ROI", (roi[0]+4, roi[1]+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            # Status overlays
            g_lbl, g_col = self.gate.overlay_label()
            cv2.putText(frame, f"{self.label} | {g_lbl}", (8,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, g_col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Park:{self.park_mon.phase.value}", (8,42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1,
                        cv2.LINE_AA)
            if dark:
                cv2.putText(frame, "LOW LIGHT", (8,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,255), 1,
                            cv2.LINE_AA)

            # Frame skip
            if self._fc % FRAME_SKIP != 0:
                self._slot[0] = frame.copy()
                if SHOW_ROI_HELPER and win: cv2.imshow(win, frame)
                continue

            # ── YOLO ──────────────────────────────────────────────────────
            try:
                yres = self.model(frame, verbose=False)[0]
            except Exception as e:
                print(f"[{self.label}] YOLO: {e}")
                self._slot[0] = frame.copy(); continue

            fh, fw = frame.shape[:2]
            boxes: List[Tuple] = []; confs: List[float] = []
            for b in yres.boxes:
                x1, y1b, x2, y2b = map(int, b.xyxy[0])
                yc = float(b.conf[0])
                if yc < YOLO_CONF_THRESHOLD: continue
                x1,y1b = max(0,x1),max(0,y1b)
                x2,y2b = min(fw,x2),min(fh,y2b)
                if x2<=x1 or y2b<=y1b: continue
                boxes.append((x1,y1b,x2,y2b)); confs.append(yc)

            self._last_det.append(len(boxes) > 0)
            self.gate.update_car(any(self._last_det))

            assigns = self._tracker.update(boxes)

            def _yc(box):
                try: return confs[boxes.index(box)]
                except: return YOLO_CONF_THRESHOLD

            for tid, (x1, y1b, x2, y2b) in assigns:
                yolo_conf = _yc((x1,y1b,x2,y2b))
                stab  = self._stab(tid)
                voter = self._voter(tid)

                # Stability gate
                if not stab.is_stable(x1, y1b, x2, y2b):
                    cv2.rectangle(frame,(x1,y1b),(x2,y2b),(0,210,210),1)
                    cv2.putText(frame, f"#{tid} stab…",
                                (x1,max(y1b-5,0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,210,210),1)
                    continue

                # Skip OCR if barely moved
                lb = self._tracker.get_last_ocr_box(tid)
                cx, cy = (x1+x2)/2.0, (y1b+y2b)/2.0
                if lb:
                    drift = math.hypot(cx-(lb[0]+lb[2])/2,
                                       cy-(lb[1]+lb[3])/2)
                    if drift < OCR_SKIP_DRIFT:
                        cv2.rectangle(frame,(x1,y1b),(x2,y2b),(0,165,255),1)
                        lr = self.llm.query(tid, list(self._hist(tid)),
                                            voter._consonant_candidates)
                        if lr and is_valid_plate(lr[0]):
                            pt, lc = lr
                            self.stab_ctr.record(tid, pt)
                            if self.stab_ctr.ready(tid, pt):
                                pv = self._tracker.get_province_cache(tid)
                                if pv is False: pv = None
                                voter.reset(); stab.reset()
                                self._tracker.reset_track(tid)
                                self._cleanup(tid)
                                self._submit(frame, pt, lc, yolo_conf, 0.0,
                                             pv, "llm_fallback",
                                             x1, y1b, x2, y2b)
                        continue

                self._tracker.set_last_ocr_box(tid, (x1,y1b,x2,y2b))
                crop = frame[y1b:y2b, x1:x2]
                if crop.size == 0: continue

                crop = deskew_plate(crop)
                try:
                    prep = preprocess_plate_image(
                        crop, scale=PREPROCESS_SCALE, number_zone_only=True)
                except Exception as e:
                    print(f"  [{self.label} #{tid}] Prep: {e}"); continue

                # [FIX-3] OCR semaphore — only shared resource between threads
                with self._ocr_sem:
                    ep = run_easyocr(self.reader, prep)
                    er = run_easyocr_raw(self.reader, crop)
                    ts = run_tesseract(prep)

                self._add_hist(tid, ep+er+ts)

                ocr_min  = adaptive_ocr_gate(yolo_conf)
                fuse_min = adaptive_min_confidence(yolo_conf)
                fused    = fuse_ocr_results(ep+er, ts, min_confidence=fuse_min)
                fused    = [r for r in fused if float(r[2]) >= ocr_min]

                print(f"  [{self.label} #{tid}] "
                      f"easy={[(r[1],round(float(r[2]),2)) for r in ep]}"
                      f"  gate={ocr_min:.2f} pass={len(fused)}")

                # Province
                pc = self._tracker.get_province_cache(tid)
                if pc is None:
                    pi = extract_province_from_ocr_tokens(ep+er+ts)
                    if not pi:
                        pi = extract_province_from_crop(
                            crop, easyocr_reader=self.reader)
                    self._tracker.set_province_cache(tid, pi or False)
                    prov = pi
                elif pc is False: prov = None
                else:             prov = pc

                cand = extract_best_plate_read(
                    fused, min_confidence=ocr_min,
                    max_correction_score=MAX_CORRECTION_SCORE)
                if cand: save_debug(f"{self.label}_t{tid}", crop, prep)

                winner_vote = voter.update(cand, yolo_conf=yolo_conf)

                if winner_vote is not None:
                    plate_text, ocr_conf = winner_vote
                    if not is_valid_plate(plate_text): continue

                    self.stab_ctr.record(tid, plate_text)
                    cnt = self.stab_ctr.count(tid, plate_text)
                    print(f"  [{self.label} #{tid}] "
                          f"Stable {cnt}/{REQUIRED_STABLE_DETECTIONS}")

                    if not self.stab_ctr.ready(tid, plate_text):
                        cv2.rectangle(frame,(x1,y1b),(x2,y2b),(0,165,255),2)
                        cv2.putText(
                            frame,
                            f"#{tid} {plate_text} "
                            f"[{cnt}/{REQUIRED_STABLE_DETECTIONS}]",
                            (x1, max(y1b-8,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                        continue

                    corr_score = max(0.0, 1.0 - ocr_conf)
                    pv = self._tracker.get_province_cache(tid)
                    if pv is False: pv = None
                    voter.reset(); stab.reset()
                    self._tracker.reset_track(tid); self._cleanup(tid)

                    # [FIX-1] Submit to registry (not direct gate open)
                    self._submit(frame, plate_text, ocr_conf,
                                 yolo_conf, corr_score,
                                 pv, "voter",
                                 x1, y1b, x2, y2b)

                else:
                    lr = self.llm.query(tid, list(self._hist(tid)),
                                        voter._consonant_candidates)
                    if lr and is_valid_plate(lr[0]):
                        pt, lc = lr
                        self.stab_ctr.record(tid, pt)
                        if self.stab_ctr.ready(tid, pt):
                            pv = self._tracker.get_province_cache(tid)
                            if pv is False: pv = None
                            voter.reset(); stab.reset()
                            self._tracker.reset_track(tid); self._cleanup(tid)
                            self._submit(frame, pt, lc, yolo_conf, 0.0,
                                         pv, "llm_fallback",
                                         x1, y1b, x2, y2b)
                    else:
                        cv2.rectangle(frame,(x1,y1b),(x2,y2b),(0,165,255),2)

            self._slot[0] = frame.copy()
            if SHOW_ROI_HELPER and win: cv2.imshow(win, frame)

        if cap: cap.release()

    def stop(self):
        self._stop.set()
        self.gate.force_release()
        self.park_mon.reset()


# ══════════════════════════════════════════════════════════════════════════
# 14. INITIALISATION & DISPLAY LOOP
# ══════════════════════════════════════════════════════════════════════════

for f in glob.glob("debug_plate_*.jpg"):
    try: os.remove(f)
    except OSError: pass

reload_allowed_plates()

threading.Thread(target=_deadlock_watchdog,
                 daemon=True, name="deadlock-watchdog").start()

print("Loading EasyOCR…")
reader = easyocr.Reader(["th", "en"], gpu=False)

print("Loading YOLO model…")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
model      = YOLO(model_path)

print("Initialising LLM fallback…")
llm = LLMCorrector()

registry = GlobalPlateRegistry()
ocr_sem  = threading.Semaphore(2)

frame_slots = [[np.zeros((360, 640, 3), dtype=np.uint8)] for _ in CAMERAS]

workers: List[CameraWorker] = []
for i, (cam_url, gate_ip, flip, label) in enumerate(CAMERAS):
    w = CameraWorker(
        cam_url=cam_url, gate_ip=gate_ip, flip=flip, label=label,
        model=model, reader=reader, llm=llm, registry=registry,
        frame_slot=frame_slots[i], ocr_sem=ocr_sem)
    workers.append(w)

for w in workers:
    w.start()

print(f"\n🅿️  Smart Parking ALPR v15.1  ({len(CAMERAS)} cameras)")
for cam_url, gate_ip, _, label in CAMERAS:
    thr = CAMERA_MOTION.get(label, {})
    roi = CAMERA_ROI.get(label)
    print(f"   {label}: {cam_url}  →  gate {gate_ip}"
          f"  ROI={roi}"
          f"  arrive={thr.get('movement_frac')}"
          f"  exit={thr.get('exit_frac')}")
print(f"   Best-Camera-Wins window : {CLAIM_WINDOW_SEC}s")
print(f"   Stable gate             : {REQUIRED_STABLE_DETECTIONS} confirmations")
print(f"   Motion confirm          : {MOTION_CONFIRM_FRAMES} consecutive frames")
print(f"   Blob filter             : {MIN_BLOB_AREA_FRAC*100:.1f}% of ROI")
if SHOW_ROI_HELPER:
    print("\n   ⚠️  ROI HELPER MODE — click two corners per camera window")
print("   Press Q to quit.\n")

log_event("system_start", cameras=[c[3] for c in CAMERAS],
          claim_window=CLAIM_WINDOW_SEC,
          stable_req=REQUIRED_STABLE_DETECTIONS,
          motion_confirm=MOTION_CONFIRM_FRAMES)

while True:
    panels = []
    for slot in frame_slots:
        f = slot[0]
        if f is None or f.size == 0:
            f = np.zeros((360, PANEL_W, 3), dtype=np.uint8)
        h, wf = f.shape[:2]
        f = cv2.resize(f, (PANEL_W, int(h * PANEL_W / max(wf, 1))))
        panels.append(f)

    max_h  = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, max_h-p.shape[0]),(0,0),(0,0)))
              for p in panels]
    combined = np.concatenate(padded, axis=1)

    active = registry.active_plates()
    if active:
        info = "  |  ".join(f"{p}→{v['cam_id']}" for p,v in active.items())
        cv2.rectangle(combined,
                      (0, combined.shape[0]-22),
                      (combined.shape[1], combined.shape[0]),
                      (20,20,20), -1)
        cv2.putText(combined, f"Active: {info}",
                    (8, combined.shape[0]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1,
                    cv2.LINE_AA)

    cv2.imshow("Smart Parking ALPR v15.1", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("\nShutting down…")
log_event("system_stop")
for w in workers: w.stop()
for w in workers: w.join(timeout=3.0)
cv2.destroyAllWindows()

if USE_RABBITMQ:
    try:
        with _rmq_lock:
            if rmq_conn and rmq_conn.is_open: rmq_conn.close()
    except Exception: pass

print("Done.")