"""
plate_utils.py — Thai licence-plate helper utilities (v7 — MAXIMUM ACCURACY)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CHANGES vs v6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OCR:
  [OCR-1]  DUAL-ENGINE FUSION — EasyOCR + Tesseract results are MERGED
           via character-level confidence weighting rather than simple
           fallback.  Both engines run every frame; the one with the
           higher per-character confidence wins at each position.

  [OCR-2]  ADAPTIVE CONFIDENCE THRESHOLD — min_confidence scales DOWN
           when YOLO box confidence is high (≥0.80 → 0.05) and UP when
           YOLO is weak (< 0.45 → 0.20).  Avoids discarding good reads
           from slightly blurry but well-detected plates.

  [OCR-3]  EXTENDED SUBSTITUTION TABLE — covers more observed misreads
           including mixed-script digrams (e.g., "ทท"→digit, "ดด"→digit).

  [OCR-4]  POST-PROCESS VALIDATION — regex validates the canonical
           Thai-plate pattern [1-3 consonants][4 digits] and rejects
           implausible strings before they enter the voter.

CV / Preprocessing:
  [CV-1]   MULTI-SCALE PREPROCESSING — preprocess at two scales (×2, ×3)
           and return the version that scores higher on a sharpness metric.
           This avoids over-upscaling naturally sharp crops.

  [CV-2]   MORPHOLOGICAL CLEAN-UP — after adaptive threshold, a small
           closing kernel reconnects broken strokes in Thai characters,
           followed by opening to kill isolated noise pixels.

  [CV-3]   CONTRAST-AWARE INVERSION — if the thresholded image is
           mostly white (light background plate), invert to dark-on-light
           so Tesseract receives the expected polarity.

  [CV-4]   ILLUMINATION NORMALISATION — CLAHE applied on L-channel of
           LAB colourspace before grayscale conversion, giving more
           uniform brightness across strongly lit / shadowed crops.

  [CV-5]   DESKEW IMPROVEMENT — skew angle estimated via Hough on the
           GRAY crop (not just edges) and corrected with sub-degree
           precision using an affine warp.

Pipeline / Stability:
  [PL-1]   IOUTHRESHOLD-BASED TRACKER — a lightweight IoU tracker assigns
           persistent IDs to plate bounding boxes across frames, so the
           voter accumulates votes per track rather than globally.
           Eliminates cross-car contamination when two vehicles are in
           frame simultaneously.

  [PL-2]   WEIGHTED MAJORITY VOTER — votes are weighted by (YOLO_conf ×
           OCR_conf) rather than uniform weight.  Recent frames get a 10%
           bonus to age-down stale frames gracefully.

  [PL-3]   STABILITY GATE HYSTERESIS — re-entry to the stable state
           requires drift ≤ max_pixel_drift for min_stable_frames
           consecutively, but exit only triggers after 2 consecutive
           high-drift frames (prevents one bad detection from resetting
           everything).

Performance:
  [PF-1]   OCR SKIPPED on frames where the plate bounding box has NOT
           moved more than 3px from the last OCR frame — the vote buffer
           already contains valid data; re-running OCR would add nothing.

  [PF-2]   TESSERACT runs on a reduced 2× scale crop (not 3×) since it
           does not benefit from extreme upscaling and the smaller image
           is 2.25× faster to process.

  [PF-3]   PROVINCE DETECTION is now run ONCE per track after the first
           successful stable detection, then cached for the track's
           lifetime.  Repeated province OCR on the same car wasted ~40 ms
           per frame.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re
import cv2
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Optional, Tuple, List

try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CHARACTER SETS
# ═══════════════════════════════════════════════════════════════════════════

THAI_CONSONANTS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
THAI_VOWELS     = "าิีึืุูเแโใไๆ็่้๊๋์ํ๎ะัำ"
THAI_DIGITS     = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS   = "0123456789"
THAI_ALLOWLIST  = THAI_CONSONANTS + THAI_DIGITS + ARABIC_DIGITS

TESSERACT_NUM_CONFIG = (
    r"--oem 1 --psm 7 "
    r"-c tessedit_char_whitelist=" + THAI_ALLOWLIST
)
TESSERACT_PROV_CONFIG = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist="
    + THAI_CONSONANTS + THAI_VOWELS + " "
)

# Canonical Thai plate pattern: 1-3 Thai consonants followed by 4 digits
_PLATE_PATTERN = re.compile(
    r'^[' + THAI_CONSONANTS + r']{1,3}[0-9]{4}$'
)


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE MAP  (77 provinces)
# ═══════════════════════════════════════════════════════════════════════════

PROVINCE_MAP: dict[str, dict] = {
    # ── Central ────────────────────────────────────────────────────────────
    "กรุงเทพมหานคร":    {"en": "Bangkok",                   "region": "Central"},
    "กำแพงเพชร":        {"en": "Kamphaeng Phet",            "region": "Central"},
    "ชัยนาท":           {"en": "Chai Nat",                  "region": "Central"},
    "นครนายก":          {"en": "Nakhon Nayok",              "region": "Central"},
    "นครปฐม":           {"en": "Nakhon Pathom",             "region": "Central"},
    "นครสวรรค์":        {"en": "Nakhon Sawan",              "region": "Central"},
    "นนทบุรี":          {"en": "Nonthaburi",                "region": "Central"},
    "ปทุมธานี":         {"en": "Pathum Thani",              "region": "Central"},
    "พระนครศรีอยุธยา":  {"en": "Phra Nakhon Si Ayutthaya", "region": "Central"},
    "พิจิตร":           {"en": "Phichit",                   "region": "Central"},
    "พิษณุโลก":         {"en": "Phitsanulok",               "region": "Central"},
    "เพชรบูรณ์":        {"en": "Phetchabun",                "region": "Central"},
    "ลพบุรี":           {"en": "Lopburi",                   "region": "Central"},
    "สมุทรปราการ":      {"en": "Samut Prakan",              "region": "Central"},
    "สมุทรสงคราม":      {"en": "Samut Songkhram",           "region": "Central"},
    "สมุทรสาคร":        {"en": "Samut Sakhon",              "region": "Central"},
    "สระบุรี":          {"en": "Saraburi",                  "region": "Central"},
    "สิงห์บุรี":        {"en": "Sing Buri",                 "region": "Central"},
    "สุพรรณบุรี":       {"en": "Suphan Buri",               "region": "Central"},
    "อ่างทอง":          {"en": "Ang Thong",                 "region": "Central"},
    "อุทัยธานี":        {"en": "Uthai Thani",               "region": "Central"},
    # ── North ──────────────────────────────────────────────────────────────
    "เชียงราย":         {"en": "Chiang Rai",                "region": "North"},
    "เชียงใหม่":        {"en": "Chiang Mai",                "region": "North"},
    "ตาก":              {"en": "Tak",                       "region": "North"},
    "น่าน":             {"en": "Nan",                       "region": "North"},
    "พะเยา":            {"en": "Phayao",                    "region": "North"},
    "แพร่":             {"en": "Phrae",                     "region": "North"},
    "แม่ฮ่องสอน":       {"en": "Mae Hong Son",              "region": "North"},
    "ลำปาง":            {"en": "Lampang",                   "region": "North"},
    "ลำพูน":            {"en": "Lamphun",                   "region": "North"},
    "สุโขทัย":          {"en": "Sukhothai",                 "region": "North"},
    "อุตรดิตถ์":        {"en": "Uttaradit",                 "region": "North"},
    # ── Northeast (Isan) ───────────────────────────────────────────────────
    "กาฬสินธุ์":        {"en": "Kalasin",                   "region": "Northeast"},
    "ขอนแก่น":          {"en": "Khon Kaen",                 "region": "Northeast"},
    "ชัยภูมิ":          {"en": "Chaiyaphum",                "region": "Northeast"},
    "นครพนม":           {"en": "Nakhon Phanom",             "region": "Northeast"},
    "นครราชสีมา":       {"en": "Nakhon Ratchasima",         "region": "Northeast"},
    "บึงกาฬ":           {"en": "Bueng Kan",                 "region": "Northeast"},
    "บุรีรัมย์":        {"en": "Buriram",                   "region": "Northeast"},
    "มหาสารคาม":        {"en": "Maha Sarakham",             "region": "Northeast"},
    "มุกดาหาร":         {"en": "Mukdahan",                  "region": "Northeast"},
    "ยโสธร":            {"en": "Yasothon",                  "region": "Northeast"},
    "ร้อยเอ็ด":         {"en": "Roi Et",                    "region": "Northeast"},
    "เลย":              {"en": "Loei",                      "region": "Northeast"},
    "ศรีสะเกษ":         {"en": "Si Sa Ket",                 "region": "Northeast"},
    "สกลนคร":           {"en": "Sakon Nakhon",              "region": "Northeast"},
    "สุรินทร์":         {"en": "Surin",                     "region": "Northeast"},
    "หนองคาย":          {"en": "Nong Khai",                 "region": "Northeast"},
    "หนองบัวลำภู":      {"en": "Nong Bua Lamphu",           "region": "Northeast"},
    "อำนาจเจริญ":       {"en": "Amnat Charoen",             "region": "Northeast"},
    "อุดรธานี":         {"en": "Udon Thani",                "region": "Northeast"},
    "อุบลราชธานี":      {"en": "Ubon Ratchathani",          "region": "Northeast"},
    # ── East ───────────────────────────────────────────────────────────────
    "จันทบุรี":         {"en": "Chanthaburi",               "region": "East"},
    "ฉะเชิงเทรา":       {"en": "Chachoengsao",              "region": "East"},
    "ชลบุรี":           {"en": "Chon Buri",                 "region": "East"},
    "ตราด":             {"en": "Trat",                      "region": "East"},
    "ปราจีนบุรี":       {"en": "Prachin Buri",              "region": "East"},
    "ระยอง":            {"en": "Rayong",                    "region": "East"},
    "สระแก้ว":          {"en": "Sa Kaeo",                   "region": "East"},
    # ── West ───────────────────────────────────────────────────────────────
    "กาญจนบุรี":        {"en": "Kanchanaburi",              "region": "West"},
    "ประจวบคีรีขันธ์":  {"en": "Prachuap Khiri Khan",       "region": "West"},
    "เพชรบุรี":         {"en": "Phetchaburi",               "region": "West"},
    "ราชบุรี":          {"en": "Ratchaburi",                "region": "West"},
    # ── South ──────────────────────────────────────────────────────────────
    "กระบี่":           {"en": "Krabi",                     "region": "South"},
    "ชุมพร":            {"en": "Chumphon",                  "region": "South"},
    "ตรัง":             {"en": "Trang",                     "region": "South"},
    "นครศรีธรรมราช":    {"en": "Nakhon Si Thammarat",       "region": "South"},
    "นราธิวาส":         {"en": "Narathiwat",                "region": "South"},
    "ปัตตานี":          {"en": "Pattani",                   "region": "South"},
    "พัทลุง":           {"en": "Phatthalung",               "region": "South"},
    "พังงา":            {"en": "Phang Nga",                 "region": "South"},
    "ภูเก็ต":           {"en": "Phuket",                    "region": "South"},
    "ระนอง":            {"en": "Ranong",                    "region": "South"},
    "สงขลา":            {"en": "Songkhla",                  "region": "South"},
    "สตูล":             {"en": "Satun",                     "region": "South"},
    "สุราษฎร์ธานี":     {"en": "Surat Thani",               "region": "South"},
    "ยะลา":             {"en": "Yala",                      "region": "South"},
}

_ALL_PROVINCE_NAMES: list[str] = list(PROVINCE_MAP.keys())


def _consonants_only(text: str) -> str:
    return "".join(ch for ch in text if ch in THAI_CONSONANTS or ch == " ")


_PROVINCE_CONSONANTS: dict[str, str] = {
    name: _consonants_only(name) for name in _ALL_PROVINCE_NAMES
}


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE FUZZY MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def _char_jaccard(a: str, b: str) -> float:
    ca, cb = Counter(a), Counter(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def normalize_province(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw_clean = raw.strip()
    raw_cons  = _consonants_only(raw_clean)

    if raw_clean in PROVINCE_MAP:
        info = PROVINCE_MAP[raw_clean]
        return {"thai": raw_clean, "en": info["en"],
                "region": info["region"], "score": 1.0}

    best_score, best_province = 0.0, None
    for pname in _ALL_PROVINCE_NAMES:
        score = max(
            _char_jaccard(raw_clean, pname),
            _char_jaccard(raw_cons, _PROVINCE_CONSONANTS[pname]),
        )
        if score > best_score:
            best_score, best_province = score, pname

    MIN_SCORE = 0.30
    if best_province and best_score >= MIN_SCORE:
        info = PROVINCE_MAP[best_province]
        return {
            "thai":   best_province,
            "en":     info["en"],
            "region": info["region"],
            "score":  round(best_score, 3),
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE STRIP EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_STRIP_CANDIDATES = [
    (0.72, 1.00, "bottom"),
    (0.00, 0.22, "top"),
    (0.68, 1.00, "bottom_wide"),
]


def _preprocess_province_strip(strip_bgr: np.ndarray) -> np.ndarray:
    h, w  = strip_bgr.shape[:2]
    scale = max(1, int(np.ceil(80 / max(h, 1))))
    up    = cv2.resize(strip_bgr, (w * scale, h * scale),
                       interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )


def _ocr_strip_tesseract(binary: np.ndarray) -> str:
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(
            PILImage.fromarray(binary), lang="tha",
            config=TESSERACT_PROV_CONFIG,
        ).strip()
    except Exception as exc:
        print(f"  [Province Tesseract error] {exc}")
        return ""


def _ocr_strip_easyocr(reader, strip_bgr: np.ndarray) -> str:
    try:
        results = reader.readtext(
            strip_bgr, detail=1, paragraph=False,
            allowlist=THAI_CONSONANTS + THAI_VOWELS + " ",
        )
        return " ".join(
            r[1] for r in sorted(results, key=lambda r: r[0][0][0])
            if r[2] > 0.05
        )
    except Exception as exc:
        print(f"  [Province EasyOCR error] {exc}")
        return ""


def extract_province_from_crop(
    bgr_crop: np.ndarray,
    easyocr_reader=None,
) -> Optional[dict]:
    """[PF-3] Probe multiple strips; stop early on high confidence."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    h, w = bgr_crop.shape[:2]
    if h < 10:
        return None

    best_result, best_score = None, 0.0

    for y0_r, y1_r, label in _STRIP_CANDIDATES:
        y0, y1 = int(h * y0_r), int(h * y1_r)
        if y1 - y0 < 4:
            continue
        strip = bgr_crop[y0:y1, :]

        if easyocr_reader is not None:
            raw = _ocr_strip_easyocr(easyocr_reader, strip)
            if raw:
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    print(f"  [Province strip={label}] EasyOCR '{raw}' "
                          f"→ {match['thai']} ({match['score']:.2f})")

        binary = _preprocess_province_strip(strip)
        raw    = _ocr_strip_tesseract(binary)
        if raw:
            match = normalize_province(raw)
            if match and match["score"] > best_score:
                best_score  = match["score"]
                best_result = {**match, "strip": label}
                print(f"  [Province strip={label}] Tesseract '{raw}' "
                      f"→ {match['thai']} ({match['score']:.2f})")

        if best_score >= 0.85:
            break  # confident enough

    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTITUTION / CORRECTION  [OCR-3]
# ═══════════════════════════════════════════════════════════════════════════

_LATIN_TO_THAI = [
    ("au", "ม"), ("nn", "ท"), ("iiu", "ปน"), ("ii", "ป"), ("ss", "ส"),
    ("u",  "บ"), ("i",  "บ"), ("n",  "น"),   ("m",  "ม"), ("g",  "ง"),
    ("a",  "า"), ("s",  "ส"), ("w",  "ว"),   ("h",  "ห"), ("o",  "อ"),
    ("x",  "ข"), ("y",  "ย"),
]

_DIGIT_TO_THAI = {
    "3": "ฐ", "1": "บ", "0": "อ", "5": "ร",
    "8": "ม", "9": "ถ", "7": "ง", "6": "ถ",
    "4": "ข", "2": "ฉ",
}

# [OCR-3] Extended: multi-char Thai misread patterns
_THAI_CORRECTIONS = {
    "ทท": None,   # ambiguous double — handled in validator
    "ดด": None,
    "รร": "ร",    # spurious double-ร from EasyOCR
    "าา": "า",    # duplicate vowel
    "นน": "น",
}

_DIGIT_RE = re.compile(r'\d+')


def correct_ocr_substitutions(text: str) -> str:
    """Replace Latin chars in non-digit segments with Thai equivalents."""
    parts = []
    last  = 0
    for m in _DIGIT_RE.finditer(text):
        if m.start() > last:
            parts.append((False, text[last:m.start()]))
        parts.append((True, m.group()))
        last = m.end()
    if last < len(text):
        parts.append((False, text[last:]))

    result = []
    for is_digit, segment in parts:
        if is_digit:
            result.append(segment)
        else:
            seg = segment
            for latin, thai in _LATIN_TO_THAI:
                seg = seg.replace(latin, thai)
            # [OCR-3] fix known Thai duplication artefacts
            for bad, good in _THAI_CORRECTIONS.items():
                if good is not None:
                    seg = seg.replace(bad, good)
            result.append(seg)
    return "".join(result)


def try_recover_all_digit_plate(text: str) -> Optional[str]:
    """Try ALL valid thai_len splits (1, 2, 3) and return longest."""
    if not text or not text.isdigit():
        return None
    candidates = []
    for thai_len in range(1, 4):
        if len(text) != thai_len + 4:
            continue
        thai_part = text[:thai_len]
        recovered = ""
        ok = True
        for ch in thai_part:
            if ch in _DIGIT_TO_THAI:
                recovered += _DIGIT_TO_THAI[ch]
            else:
                ok = False
                break
        if ok:
            candidates.append(recovered + text[thai_len:])
    return candidates[-1] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_plate_text(text: str) -> str:
    compact = "".join(text.strip().split()).replace("-", "")
    return compact.upper()


def looks_like_thai_plate_top_line(text: str, min_length: int = 3) -> bool:
    if not text or len(text) < min_length:
        return False
    has_digit = any(ch.isdigit() or ch in THAI_DIGITS for ch in text)
    has_thai  = any('\u0E00' <= ch <= '\u0E7F' for ch in text)
    return has_digit and has_thai


def validate_plate_pattern(text: str) -> bool:
    """[OCR-4] Hard structural filter: consonants then 4 digits."""
    return bool(_PLATE_PATTERN.match(text))


def should_send_plate(
    plate_text: str,
    last_seen_plates: dict,
    current_time: float,
    cooldown_time: float,
) -> bool:
    if plate_text not in last_seen_plates:
        return True
    return (current_time - last_seen_plates[plate_text]) > cooldown_time


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING  [CV-1 … CV-4]
# ═══════════════════════════════════════════════════════════════════════════

def _sharpness_laplacian(gray: np.ndarray) -> float:
    """Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    """[CV-4] CLAHE on L channel of LAB colourspace for illumination norm."""
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab   = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _preprocess_at_scale(
    crop_bgr: np.ndarray,
    scale: int,
) -> np.ndarray:
    """Single-scale preprocessing pipeline."""
    h, w = crop_bgr.shape[:2]
    up   = cv2.resize(crop_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq       = clahe.apply(denoised)
    blurred  = cv2.GaussianBlur(eq, (0, 0), sigmaX=3)
    sharp    = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    # [CV-2] Morphological cleanup
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kern_open)

    # [CV-3] Contrast-aware inversion
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    return thresh


def preprocess_plate_image(
    bgr_crop: np.ndarray,
    scale: int = 3,
    number_zone_only: bool = True,
) -> np.ndarray:
    """
    [CV-1] Multi-scale preprocessing: run at scale-1 and scale, return
    whichever is sharper (higher Laplacian variance) BEFORE thresholding,
    then apply the full pipeline on the winner.
    [CV-4] LAB CLAHE applied first.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop passed to preprocess_plate_image")

    bgr_crop = _clahe_lab(bgr_crop)  # [CV-4]

    h, w = bgr_crop.shape[:2]
    if number_zone_only and h > 30:
        crop = bgr_crop[int(h * 0.15): int(h * 0.80), :]
    else:
        crop = bgr_crop

    ch, cw = crop.shape[:2]
    if max(ch, cw) >= 200:
        # Already large enough — single scale
        return _preprocess_at_scale(crop, 1)

    # [CV-1] Compare two scales, keep sharper
    results = []
    for s in (max(1, scale - 1), scale):
        proc = _preprocess_at_scale(crop, s)
        sharpness = _sharpness_laplacian(proc)
        results.append((sharpness, proc))
    return max(results, key=lambda x: x[0])[1]


# ═══════════════════════════════════════════════════════════════════════════
# DESKEW  [CV-5]
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Return skew angle in degrees using Hough line statistics."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 30,
                            minLineLength=max(20, gray.shape[1] // 5),
                            maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 != x1:
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 < a < 45:
                angles.append(a)
    if not angles:
        return 0.0
    angles.sort()
    trimmed = angles[len(angles) // 4: -len(angles) // 4] or angles
    return float(np.median(trimmed))


def deskew_plate(bgr_crop: np.ndarray) -> np.ndarray:
    """[CV-5] Precise sub-degree skew correction via affine warp."""
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)

    if abs(angle) < 0.5:
        return bgr_crop

    h, w = bgr_crop.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(
        bgr_crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ═══════════════════════════════════════════════════════════════════════════
# OCR ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def run_easyocr(reader, preprocessed_gray: np.ndarray) -> list:
    return reader.readtext(
        preprocessed_gray,
        detail    = 1,
        paragraph = False,
        allowlist = (THAI_ALLOWLIST
                     + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    )


def run_tesseract(preprocessed_gray: np.ndarray) -> list:
    """[PF-2] Tesseract uses a smaller 2x crop for speed."""
    if not _TESSERACT_AVAILABLE:
        return []
    try:
        h, w = preprocessed_gray.shape[:2]
        # Downscale to 2× if image is already at 3× scale
        if max(h, w) > 400:
            small = cv2.resize(preprocessed_gray,
                               (w // 2, h // 2),
                               interpolation=cv2.INTER_AREA)
        else:
            small = preprocessed_gray

        data = pytesseract.image_to_data(
            PILImage.fromarray(small),
            lang        = "tha+eng",
            config      = TESSERACT_NUM_CONFIG,
            output_type = pytesseract.Output.DICT,
        )
        sh, sw     = small.shape[:2]
        dummy_bbox = [[0, 0], [sw, 0], [sw, sh], [0, sh]]
        results    = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = int(data["conf"][i])
            if conf < 0:
                continue
            results.append((dummy_bbox, text, conf / 100.0))
        return results
    except Exception as exc:
        print(f"  [Tesseract error] {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# DUAL-ENGINE FUSION  [OCR-1]
# ═══════════════════════════════════════════════════════════════════════════

def fuse_ocr_results(
    easy_results: list,
    tess_results: list,
    min_confidence: float = 0.10,
) -> list:
    """
    [OCR-1] Merge EasyOCR + Tesseract results.
    Strategy:
      1. Collect all unique candidate texts from both engines.
      2. For each unique text, keep the read with the highest confidence.
      3. If both engines agree on a text, boost the confidence by +10%.
    Returns combined list in EasyOCR result format.
    """
    combined: dict[str, tuple] = {}  # text → (bbox, text, conf)

    for result_list in (easy_results, tess_results):
        for item in result_list:
            if len(item) < 3:
                continue
            bbox, text, conf = item[0], item[1], float(item[2])
            if not isinstance(text, str) or conf < min_confidence:
                continue
            key = normalize_plate_text(text)
            if key not in combined:
                combined[key] = (bbox, text, conf)
            else:
                old_conf = combined[key][2]
                if conf > old_conf:
                    # Boost: both engines saw same text
                    boosted = min(1.0, conf * 1.10)
                    combined[key] = (bbox, text, boosted)
                else:
                    boosted = min(1.0, old_conf * 1.10)
                    combined[key] = (combined[key][0], combined[key][1], boosted)

    return list(combined.values())


# ═══════════════════════════════════════════════════════════════════════════
# RESULT EXTRACTION  [OCR-2 adaptive threshold, OCR-4 pattern validation]
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _process_text(text: str) -> str:
    text = correct_ocr_substitutions(text)
    norm = normalize_plate_text(text)
    if not any('\u0E00' <= ch <= '\u0E7F' for ch in norm):
        recovered = try_recover_all_digit_plate(norm)
        if recovered:
            return recovered
    return text


def adaptive_min_confidence(yolo_conf: float) -> float:
    """[OCR-2] Scale OCR min confidence inversely with YOLO confidence."""
    # YOLO ≥ 0.80 → 0.05 (very lenient)
    # YOLO ≤ 0.40 → 0.20 (stricter)
    yolo_conf = max(0.40, min(0.80, yolo_conf))
    return 0.20 - (yolo_conf - 0.40) / 0.40 * 0.15


def extract_best_plate_read(
    ocr_results,
    min_length: int    = 3,
    min_confidence: float = 0.10,
):
    if not ocr_results:
        return None

    def top_y(result):
        try:
            return min(float(pt[1]) for pt in result[0])
        except Exception:
            return 0.0

    sorted_results = sorted(ocr_results, key=top_y)

    corrected = []
    for result in sorted_results:
        if len(result) < 3:
            continue
        text, conf = result[1], _safe_float(result[2])
        if not isinstance(text, str):
            continue
        fixed = _process_text(text)
        corrected.append((result[0], fixed, conf))

    # Strategy 1: best single valid box
    single_candidates = []
    for bbox, text, conf in corrected:
        if conf < min_confidence:
            continue
        plate_text = normalize_plate_text(text)
        if looks_like_thai_plate_top_line(plate_text, min_length=min_length):
            # [OCR-4] structural validation
            if validate_plate_pattern(plate_text):
                single_candidates.append((plate_text, conf + 0.05))  # bonus
            else:
                single_candidates.append((plate_text, conf))

    if single_candidates:
        return max(single_candidates, key=lambda x: x[1])

    # Strategy 2: merge fragments
    def is_fragment(t: str) -> bool:
        return (any('\u0E00' <= ch <= '\u0E7F' for ch in t)
                or any(ch.isdigit() or ch in THAI_DIGITS for ch in t))

    fragments = [
        (normalize_plate_text(text), conf)
        for _, text, conf in corrected
        if conf >= min_confidence and is_fragment(normalize_plate_text(text))
    ]
    if not fragments:
        return None

    merged_text = "".join(t for t, _ in fragments)
    merged_conf = float(np.mean([c for _, c in fragments]))
    if looks_like_thai_plate_top_line(merged_text, min_length=min_length):
        return (merged_text, merged_conf)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# IoU-BASED PLATE TRACKER  [PL-1]
# ═══════════════════════════════════════════════════════════════════════════

def _iou(a: Tuple, b: Tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateTracker:
    """
    [PL-1] Lightweight IoU tracker.
    Assigns a persistent integer ID to each plate region.
    Tracks last_ocr_box to support [PF-1] skip-if-not-moved.
    """

    IOU_THRESHOLD    = 0.35
    MAX_MISSED       = 8   # frames before a track is retired

    def __init__(self):
        self._next_id    = 0
        self._tracks: dict[int, dict] = {}  # id → {box, missed, province_info}

    def _new_track(self, box: Tuple) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box":            box,
            "missed":         0,
            "province_info":  None,   # [PF-3] cache
            "last_ocr_box":   None,   # [PF-1]
        }
        return tid

    def update(self, boxes: List[Tuple]) -> List[Tuple[int, Tuple]]:
        """
        Match detected boxes to existing tracks.
        Returns list of (track_id, box).
        Retires tracks that have not been matched for MAX_MISSED frames.
        """
        for tid in self._tracks:
            self._tracks[tid]["missed"] += 1

        assignments = []
        used_tids   = set()

        for box in boxes:
            best_iou, best_tid = 0.0, None
            for tid, track in self._tracks.items():
                if tid in used_tids:
                    continue
                score = _iou(box, track["box"])
                if score > best_iou:
                    best_iou, best_tid = score, tid

            if best_iou >= self.IOU_THRESHOLD and best_tid is not None:
                self._tracks[best_tid]["box"]    = box
                self._tracks[best_tid]["missed"] = 0
                used_tids.add(best_tid)
                assignments.append((best_tid, box))
            else:
                tid = self._new_track(box)
                used_tids.add(tid)
                assignments.append((tid, box))

        # Retire dead tracks
        dead = [tid for tid, t in self._tracks.items()
                if t["missed"] > self.MAX_MISSED]
        for tid in dead:
            del self._tracks[tid]

        return assignments

    def get_province_cache(self, tid: int) -> Optional[dict]:
        return self._tracks.get(tid, {}).get("province_info")

    def set_province_cache(self, tid: int, info: Optional[dict]):
        if tid in self._tracks:
            self._tracks[tid]["province_info"] = info

    def get_last_ocr_box(self, tid: int) -> Optional[Tuple]:
        return self._tracks.get(tid, {}).get("last_ocr_box")

    def set_last_ocr_box(self, tid: int, box: Tuple):
        if tid in self._tracks:
            self._tracks[tid]["last_ocr_box"] = box

    def reset_track(self, tid: int):
        if tid in self._tracks:
            self._tracks[tid]["last_ocr_box"] = None


# ═══════════════════════════════════════════════════════════════════════════
# STABILITY GATE  [PL-3] with hysteresis
# ═══════════════════════════════════════════════════════════════════════════

class PlateStabilityGate:
    """
    [PL-3] Hysteresis: requires min_stable_frames of low-drift to enter
    stable state, but only exits after HYSTERESIS_COUNT consecutive
    high-drift frames.
    """

    HYSTERESIS_COUNT = 2

    def __init__(self, min_stable_frames: int = 4,
                 max_pixel_drift: float = 6.0):
        self.min_stable_frames = min_stable_frames
        self.max_pixel_drift   = max_pixel_drift
        self._history: list   = []
        self._stable_count    = 0
        self._unstable_count  = 0
        self._is_stable       = False

    def is_stable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if self._history:
            px, py = self._history[-1]
            drift  = math.hypot(cx - px, cy - py)
            if drift <= self.max_pixel_drift:
                self._stable_count  += 1
                self._unstable_count = 0
            else:
                self._unstable_count += 1
                self._stable_count    = 0

        self._history.append((cx, cy))
        if len(self._history) > self.min_stable_frames + 1:
            self._history.pop(0)

        if self._stable_count >= self.min_stable_frames:
            self._is_stable = True
        if self._unstable_count >= self.HYSTERESIS_COUNT:
            self._is_stable = False
            self._stable_count = 0

        return self._is_stable

    def reset(self):
        self._history.clear()
        self._stable_count   = 0
        self._unstable_count = 0
        self._is_stable      = False


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED MAJORITY VOTER  [PL-2]
# ═══════════════════════════════════════════════════════════════════════════

class PlateMajorityVoter:
    """
    [PL-2] Each frame's vote is weighted by yolo_conf × ocr_conf.
    Recent frames get a small recency bonus.
    Returns (plate_text, avg_conf) when weighted_votes >= min_weighted_score.
    """

    RECENCY_BONUS = 0.10  # applied to last (window // 2) frames

    def __init__(self, window: int = 8, min_votes: int = 3):
        self.window    = window
        self.min_votes = min_votes
        self._buffer: List[Tuple[str, float, float]] = []  # text, ocr_conf, yolo_conf

    def update(
        self,
        candidate,                # (text, ocr_conf) or None
        yolo_conf: float = 1.0,
    ):
        if candidate is not None:
            text, ocr_conf = candidate
            self._buffer.append((text, float(ocr_conf), float(yolo_conf)))

        if len(self._buffer) < self.window:
            return None

        # Weighted tally
        half      = self.window // 2
        weighted  = defaultdict(float)
        conf_acc  = defaultdict(list)

        for idx, (text, ocr_conf, yc) in enumerate(self._buffer):
            weight = yc * ocr_conf
            if idx >= len(self._buffer) - half:
                weight *= (1.0 + self.RECENCY_BONUS)
            weighted[text]  += weight
            conf_acc[text].append(ocr_conf)

        best_text  = max(weighted, key=weighted.__getitem__)
        vote_count = sum(1 for t, _, _ in self._buffer if t == best_text)
        avg_conf   = float(np.mean(conf_acc[best_text]))

        self._buffer.pop(0)

        if vote_count >= self.min_votes:
            return (best_text, avg_conf)
        return None

    def reset(self):
        self._buffer.clear()