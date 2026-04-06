"""
main.py — Thai licence-plate ALPR  (ESP32-CAM → YOLO → OCR → RabbitMQ)

v9 changes vs v8  (see plate_utils.py and plate_corrector.py for detail):
  [CV-6]  Multi-strategy binarization (Otsu + adaptive-Gaussian-31px +
          adaptive-Mean-25px). Best strategy selected per frame by
          connected-component plausibility score. Fixes the root cause:
          11px adaptive kernel was destroying Thai consonant strokes.
  [CV-7]  3×3 morphological closing (was 2×2) to reconnect Thai strokes.
  [CV-8]  Tighter number-zone crop — cuts province strip more aggressively.
  [CV-9]  Smart province-line detection via row-mean brightness scanning.
  [OCR-5] Raw-color parallel EasyOCR path — runs on BGR crop without
          binarization, normalised to 80px height.
  [OCR-6] Beamsearch decoder (beamWidth=10) on raw path.
  [PL-4]  Partial-evidence voter — accumulates digit suffix + consonant
          candidates across frames for reconstruction when full OCR fails.
  [PC-1]  extract_digit_suffix() helper in plate_corrector.
  [PC-2]  merge_digit_evidence() helper in plate_corrector.
  All v8 improvements retained.
"""

import cv2
import json
import os
import sys
import argparse
import glob
import time
import math
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr

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
from rabbitmq import connect, publish_entry_event, publish_exit_event
from camera_registry import get_camera, load_cameras
from slot_presence import SlotState


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Thai licence-plate ALPR")
parser.add_argument("--camera", "-c", default=None,
                    help="Camera ID from cameras.json (e.g. cam-01)")
args = parser.parse_args()

# Load camera config from registry
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

COOLDOWN_TIME        = 10         # seconds before same plate re-fires
FRAME_SKIP           = 3          # process every Nth frame
YOLO_CONF_THRESHOLD  = 0.35
STREAM_RETRY_DELAY   = 2

PREPROCESS_SCALE     = 3
VOTE_WINDOW          = 8
VOTE_MIN_VOTES       = 3

OCR_SKIP_DRIFT       = 3.0        # px: skip OCR if box barely moved [PF-1]

STABILITY_MIN_FRAMES = 4
STABILITY_MAX_DRIFT  = 6.0

MAX_CORRECTION_SCORE = 0.65

DEBUG_SAVE_CROPS     = True
DEBUG_MAX_SAVES      = 10
_debug_save_count    = 0

HF_REPO_ID           = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME    = "HurricaneOD_beta.pt"


# ══════════════════════════════════════════════════════════════════════════
# 2. INITIALISATION
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

print("Connecting to RabbitMQ...")
try:
    rmq_conn, rmq_channel = connect()
    print(f"RabbitMQ connected!  lot={LOT_ID}  cam={CAM_ID}  slot={SLOT_ID}")
except Exception as exc:
    print(f"RabbitMQ failed: {exc}")
    raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════
# 3. STREAM HELPERS
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
# 4. PER-TRACK STATE  [PL-1]
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


def _cleanup_track(tid: int):
    _track_voters.pop(tid, None)
    _track_stability.pop(tid, None)


# ══════════════════════════════════════════════════════════════════════════
# 5. YOLO CONFIDENCE INDEX HELPER
# ══════════════════════════════════════════════════════════════════════════

def _get_yolo_conf(raw_boxes: list, yolo_confs: list, box: tuple) -> float:
    """
    Safely retrieve YOLO confidence for a given box.
    Falls back to YOLO_CONF_THRESHOLD if box not found
    (can happen for newly-created tracks that inherit a previously unseen box).
    """
    try:
        idx = raw_boxes.index(box)
        return yolo_confs[idx]
    except (ValueError, IndexError):
        return YOLO_CONF_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

cap              = None
frame_count      = 0
last_seen_plates: dict[str, float] = {}
tracker          = PlateTracker()
slot             = SlotState()

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

    # ── Gather boxes → tracker  [PL-1] ────────────────────────────────────
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

    for tid, (x1, y1, x2, y2) in track_assignments:
        yolo_conf = _get_yolo_conf(raw_boxes, yolo_confs, (x1, y1, x2, y2))
        stability = _get_stability(tid)
        voter     = _get_voter(tid)

        # ── Stability gate  [PL-3] ─────────────────────────────────────────
        if not stability.is_stable(x1, y1, x2, y2):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 210), 1)
            cv2.putText(frame, f"#{tid} stabilising...", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 210), 1,
                        cv2.LINE_AA)
            continue

        # ── [PF-1] Skip OCR if box barely moved ───────────────────────────
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

        # ── Step 1: deskew  [CV-5] ─────────────────────────────────────────
        plate_crop = deskew_plate(plate_crop)

        # ── Step 2: province detection  [PF-3] cached per track ───────────
        province_info = tracker.get_province_cache(tid)
        if province_info is None:
            province_info = extract_province_from_crop(
                plate_crop, easyocr_reader=reader,
            )
            tracker.set_province_cache(tid, province_info or False)
        elif province_info is False:
            province_info = None

        if province_info:
            print(f"  [#{tid}] Province: {province_info['thai']} "
                  f"({province_info['en']}) score={province_info['score']:.2f}")
        else:
            print(f"  [#{tid}] Province: unknown")

        # ── Step 3: preprocess  [CV-1, CV-4, CV-6, CV-7, CV-8, CV-9] ──────
        try:
            preprocessed = preprocess_plate_image(
                plate_crop,
                scale=PREPROCESS_SCALE,
                number_zone_only=True,
            )
        except Exception as exc:
            print(f"  [#{tid}] Preprocess error: {exc}")
            continue

        # ── Step 4: three OCR paths  [OCR-1, OCR-5, OCR-6] ────────────────
        # Path A: EasyOCR on binarized/preprocessed image (existing)
        easy_results_prep = run_easyocr(reader, preprocessed)

        # Path B: EasyOCR on raw BGR crop — NEW [OCR-5]
        # This is the critical fix: raw image preserves Thai consonant detail
        # that binarization destroys. Both paths are fused before correction.
        easy_results_raw = run_easyocr_raw(reader, plate_crop)

        # Path C: Tesseract on preprocessed image [PF-2]
        tess_results = run_tesseract(preprocessed)

        min_conf = adaptive_min_confidence(yolo_conf)   # [OCR-2]

        # Fuse all three paths — agreement between any two boosts confidence
        fused = fuse_ocr_results(
            easy_results_prep + easy_results_raw,   # merged EasyOCR paths
            tess_results,
            min_confidence=min_conf,
        )

        print(f"  [#{tid}] EasyOCR-prep: "
              f"{[(r[1], round(float(r[2]), 2)) for r in easy_results_prep]}")
        print(f"  [#{tid}] EasyOCR-raw:  "
              f"{[(r[1], round(float(r[2]), 2)) for r in easy_results_raw]}")
        print(f"  [#{tid}] Tesseract:    "
              f"{[(r[1], round(float(r[2]), 2)) for r in tess_results]}")

        # ── Step 5: domain-enforced post-processing  [PP-1, PP-2] ─────────
        candidate = extract_best_plate_read(
            fused,
            min_confidence=min_conf,
            max_correction_score=MAX_CORRECTION_SCORE,
        )
        print(f"  [#{tid}] Candidate (corrected): {candidate}")

        if candidate is not None:
            save_debug(f"tid{tid}", plate_crop, preprocessed)

        # ── Step 6: weighted majority vote + partial evidence  [PL-2, PL-4] ─
        winner = voter.update(candidate, yolo_conf=yolo_conf)
        if winner is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            continue

        plate_text, ocr_conf = winner
        current_time = time.time()

        # ── [PP-3] Final valid-plate gate ──────────────────────────────────
        if not is_valid_plate(plate_text):
            print(f"  [#{tid}] Voter winner failed validation: {plate_text!r}")
            continue

        if not should_send_plate(plate_text, last_seen_plates,
                                 current_time, COOLDOWN_TIME):
            print(f"  [#{tid}] [{plate_text}] in cooldown.")
            voter.reset()
            stability.reset()
            tracker.reset_track(tid)
            continue

        # ── Step 7: build & publish payload ───────────────────────────────
        province_thai   = province_info["thai"]   if province_info else None
        province_en     = province_info["en"]     if province_info else None
        province_region = province_info["region"] if province_info else None

        print(f"✅  [#{tid}] {plate_text}  "
              f"province={province_thai} ({province_en})  "
              f"YOLO={yolo_conf:.2f}  OCR≈{ocr_conf:.2f}")

        last_seen_plates[plate_text] = current_time
        voter.reset()
        stability.reset()
        tracker.reset_track(tid)

        registration = plate_text
        province_str = province_thai if province_thai else ""

        # ── Slot presence logic ────────────────────────────────────────
        if not slot.is_occupied:
            # New car arrived → ENTRY
            slot.park(plate_text, province_str, current_time)
            try:
                publish_entry_event(rmq_channel, registration,
                                    province_str, LOT_ID, CAM_ID)
                print(f"  Published ENTRY: {registration} "
                      f"province={province_str}")
            except Exception:
                print("RabbitMQ lost — reconnecting...")
                try:
                    if rmq_conn and rmq_conn.is_open:
                        rmq_conn.close()
                except Exception:
                    pass
                rmq_conn, rmq_channel = connect()
                publish_entry_event(rmq_channel, registration,
                                    province_str, LOT_ID, CAM_ID)
        elif slot.plate == plate_text:
            # Same car still here — refresh last-seen time
            slot.see(current_time)
        else:
            # Different plate on a slot that's occupied — ignore
            # (could be a misread or a car passing by)
            print(f"  [#{tid}] Ignoring {plate_text}, "
                  f"slot occupied by {slot.plate}")

        label = (f"#{tid} {plate_text}  {province_en}"
                 if province_en else f"#{tid} {plate_text}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # ── Check for car departure (no plate seen for ABSENCE_TIMEOUT) ────
    if slot.check_departure(time.time()):
        departed = slot.plate
        prov = slot.province or ""
        print(f"🚗  Car departed: {departed}")
        try:
            publish_exit_event(rmq_channel, departed, prov, LOT_ID, CAM_ID)
            print(f"  Published EXIT: {departed}")
        except Exception:
            print("RabbitMQ lost — reconnecting...")
            try:
                if rmq_conn and rmq_conn.is_open:
                    rmq_conn.close()
            except Exception:
                pass
            rmq_conn, rmq_channel = connect()
            publish_exit_event(rmq_channel, departed, prov, LOT_ID, CAM_ID)
        slot.clear()

    cv2.imshow("ESP32-CAM ALPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ══════════════════════════════════════════════════════════════════════════
# 7. CLEANUP
# ══════════════════════════════════════════════════════════════════════════

if cap:
    cap.release()
cv2.destroyAllWindows()
try:
    if rmq_conn and rmq_conn.is_open:
        rmq_conn.close()
        print("RabbitMQ connection closed.")
except Exception:
    pass