"""
main.py — Thai licence-plate ALPR  (ESP32-CAM → YOLO → OCR → RabbitMQ)

v10.1 changes vs v10  (see plate_utils.py for PROV-1…7 detail):
  [PROV-7] PROVINCE FROM OCR TOKENS — after the three OCR engines run on
           the number zone, every non-digit token is immediately tested
           against normalize_province().  EasyOCR on the raw color crop
           already reads province text as a second bounding box (e.g.
           'กรทพมหวบกร' alongside '3กบ7744').  Mining these free tokens
           avoids the separate province-strip binarization pass for the
           majority of frames where the colour crop is clear enough.
           The strip-based fallback [PROV-2…5] still runs when the token
           mining produces no result.

  [FIX-1,2] Leading-digit consonant bug fixed in plate_utils.py /
            plate_corrector.py (see those files for detail).
"""

import cv2
import json
import os
import glob
import time
import math
from collections import deque
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
    extract_province_from_ocr_tokens,   # [PROV-7]
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

ESP32_URL            = "http://172.20.10.3:81/stream"
ESP32_FLIP_CODE      = 1          # 1=horiz, 0=vert, -1=both, None=off

RMQ_HOST             = "localhost"
RMQ_PORT             = 5672
RMQ_USER             = "guest"
RMQ_PASS             = "guest"
RMQ_QUEUE            = "alpr_queue"

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

LLM_CANDIDATE_HISTORY = 20


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

print("Initialising LLM OCR fallback...")
llm_fallback = LLMOCRFallback()
print("LLM fallback ready." if llm_fallback._api_key else
      "LLM fallback DISABLED (no API key).")


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

_track_voters:    dict[int, PlateMajorityVoter]   = {}
_track_stability: dict[int, PlateStabilityGate]   = {}
_track_ocr_history: dict[int, deque] = {}


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


def _get_ocr_history(tid: int) -> deque:
    if tid not in _track_ocr_history:
        _track_ocr_history[tid] = deque(maxlen=LLM_CANDIDATE_HISTORY)
    return _track_ocr_history[tid]


def _add_to_ocr_history(tid: int, ocr_results: list):
    history  = _get_ocr_history(tid)
    existing = set(history)
    for item in ocr_results:
        if len(item) >= 2:
            text = str(item[1]).strip()
            if text and text not in existing:
                history.append(text)
                existing.add(text)


def _cleanup_track(tid: int):
    _track_voters.pop(tid, None)
    _track_stability.pop(tid, None)
    _track_ocr_history.pop(tid, None)
    llm_fallback.invalidate_track(tid)


# ══════════════════════════════════════════════════════════════════════════
# 5. YOLO CONFIDENCE INDEX HELPER
# ══════════════════════════════════════════════════════════════════════════

def _get_yolo_conf(raw_boxes: list, yolo_confs: list, box: tuple) -> float:
    try:
        idx = raw_boxes.index(box)
        return yolo_confs[idx]
    except (ValueError, IndexError):
        return YOLO_CONF_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════
# 6. PLATE PUBLISHING HELPER
# ══════════════════════════════════════════════════════════════════════════

def _publish_and_display(
    frame,
    tid: int,
    x1: int, y1: int, x2: int, y2: int,
    plate_text: str,
    ocr_conf: float,
    yolo_conf: float,
    province_info,
    source: str = "voter",
):
    global rmq_conn, rmq_channel

    province_thai   = province_info["thai"]   if province_info else None
    province_en     = province_info["en"]     if province_info else None
    province_region = province_info["region"] if province_info else None

    print(f"✅  [#{tid}] {plate_text}  source={source}  "
          f"province={province_thai} ({province_en})  "
          f"YOLO={yolo_conf:.2f}  OCR≈{ocr_conf:.2f}")

    payload = {
        "timestamp"       : int(time.time()),
        "plate_text"      : plate_text,
        "province_thai"   : province_thai,
        "province_en"     : province_en,
        "province_region" : province_region,
        "yolo_confidence" : round(yolo_conf, 4),
        "ocr_confidence"  : round(float(ocr_conf), 4),
        "camera_id"       : "esp32_cam_gate_1",
        "track_id"        : tid,
        "ocr_source"      : source,
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

    colour = (0, 255, 0) if source == "voter" else (255, 180, 0)
    label  = (f"#{tid} {plate_text}  {province_en}"
              if province_en else f"#{tid} {plate_text}")
    if source == "llm_fallback":
        label += " [LLM]"
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN LOOP
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
                llm_result = llm_fallback.query(
                    track_id         = tid,
                    ocr_candidates   = list(_get_ocr_history(tid)),
                    prior_consonants = voter._consonant_candidates,
                )
                if llm_result is not None:
                    plate_text, llm_conf = llm_result
                    current_time = time.time()
                    if (is_valid_plate(plate_text)
                            and should_send_plate(plate_text, last_seen_plates,
                                                  current_time, COOLDOWN_TIME)):
                        last_seen_plates[plate_text] = current_time
                        province_info = tracker.get_province_cache(tid)
                        if province_info is False:
                            province_info = None
                        voter.reset()
                        stability.reset()
                        tracker.reset_track(tid)
                        _cleanup_track(tid)
                        _publish_and_display(
                            frame, tid, x1, y1, x2, y2,
                            plate_text, llm_conf, yolo_conf,
                            province_info, source="llm_fallback",
                        )
                continue

        tracker.set_last_ocr_box(tid, (x1, y1, x2, y2))

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        # ── Step 1: deskew  [CV-5] ─────────────────────────────────────────
        plate_crop = deskew_plate(plate_crop)

        # ── Step 2: preprocess  [CV-1, CV-4, CV-6, CV-7, CV-8, CV-9] ──────
        try:
            preprocessed = preprocess_plate_image(
                plate_crop,
                scale=PREPROCESS_SCALE,
                number_zone_only=True,
            )
        except Exception as exc:
            print(f"  [#{tid}] Preprocess error: {exc}")
            continue

        # ── Step 3: three OCR paths  [OCR-1, OCR-5, OCR-6] ────────────────
        easy_results_prep = run_easyocr(reader, preprocessed)
        easy_results_raw  = run_easyocr_raw(reader, plate_crop)
        tess_results      = run_tesseract(preprocessed)

        _add_to_ocr_history(tid, easy_results_prep + easy_results_raw + tess_results)

        min_conf = adaptive_min_confidence(yolo_conf)

        fused = fuse_ocr_results(
            easy_results_prep + easy_results_raw,
            tess_results,
            min_confidence=min_conf,
        )

        print(f"  [#{tid}] EasyOCR-prep: "
              f"{[(r[1], round(float(r[2]), 2)) for r in easy_results_prep]}")
        print(f"  [#{tid}] EasyOCR-raw:  "
              f"{[(r[1], round(float(r[2]), 2)) for r in easy_results_raw]}")
        print(f"  [#{tid}] Tesseract:    "
              f"{[(r[1], round(float(r[2]), 2)) for r in tess_results]}")

        # ── Step 4: province detection — OCR token mining first  [PROV-7] ──
        # Mine province text from the tokens already collected in Step 3.
        # EasyOCR on the raw color crop frequently reads the province line
        # as a second bounding box (e.g. 'กรทพมหวบกร' at conf 0.42).
        # This is free; no extra OCR call is needed.
        province_info = tracker.get_province_cache(tid)
        if province_info is None:
            # [PROV-7] Try token mining first (fast, free)
            all_tokens = easy_results_prep + easy_results_raw + tess_results
            province_info = extract_province_from_ocr_tokens(all_tokens)

            if province_info is None:
                # [PROV-2…5] Fallback: dedicated multi-strategy strip scan
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

        if winner is not None:
            plate_text, ocr_conf = winner
            current_time = time.time()

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

            last_seen_plates[plate_text] = current_time
            province_info_pub = tracker.get_province_cache(tid)
            if province_info_pub is False:
                province_info_pub = None
            voter.reset()
            stability.reset()
            tracker.reset_track(tid)
            _cleanup_track(tid)

            _publish_and_display(
                frame, tid, x1, y1, x2, y2,
                plate_text, ocr_conf, yolo_conf,
                province_info_pub, source="voter",
            )

        else:
            # ── LLM fallback [LLM-1] ──────────────────────────────────────
            llm_result = llm_fallback.query(
                track_id         = tid,
                ocr_candidates   = list(_get_ocr_history(tid)),
                prior_consonants = voter._consonant_candidates,
            )

            if llm_result is not None:
                plate_text, llm_conf = llm_result
                current_time = time.time()

                if not is_valid_plate(plate_text):
                    print(f"  [#{tid}] LLM result failed validation: "
                          f"{plate_text!r} — discarded")
                elif not should_send_plate(plate_text, last_seen_plates,
                                           current_time, COOLDOWN_TIME):
                    print(f"  [#{tid}] [{plate_text}] in cooldown (LLM).")
                    voter.reset()
                    stability.reset()
                    tracker.reset_track(tid)
                    _cleanup_track(tid)
                else:
                    last_seen_plates[plate_text] = current_time
                    province_info_pub = tracker.get_province_cache(tid)
                    if province_info_pub is False:
                        province_info_pub = None
                    voter.reset()
                    stability.reset()
                    tracker.reset_track(tid)
                    _cleanup_track(tid)

                    _publish_and_display(
                        frame, tid, x1, y1, x2, y2,
                        plate_text, llm_conf, yolo_conf,
                        province_info_pub, source="llm_fallback",
                    )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    cv2.imshow("ESP32-CAM ALPR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ══════════════════════════════════════════════════════════════════════════
# 8. CLEANUP
# ══════════════════════════════════════════════════════════════════════════

if cap:
    cap.release()
cv2.destroyAllWindows()
try:
    if rmq_conn and rmq_conn.is_open:
        rmq_conn.close()
except Exception:
    pass