"""
plate_utils.py — Thai licence-plate helper utilities  (v11 — MAX ACCURACY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY CHANGES vs v9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Province detection (biggest UX improvement):
  [PROV-1] Province strip crop now takes bottom 28% of plate height
           (was ~28% but with a fixed 72% top-cut). The strip candidates
           are re-ordered so the widest bottom strip is tried first.
           This gives much more vertical space for province text like
           "กรุงเทพมหานคร" which is a long word.

  [PROV-2] Province fuzzy match MIN_SCORE lowered from 0.30 → 0.25.
           Thai province names share many consonants; a looser threshold
           recovers partial OCR reads of long names like กรุงเทพมหานคร
           that EasyOCR often truncates.

  [PROV-3] Province strip preprocessing now tries both upright and
           inverted polarity before Tesseract, since some plates have
           dark-background province strips (newer Thai plates).

  [PROV-4] extract_province_from_crop() now passes the FULL original
           crop to the province reader rather than a pre-cropped slice,
           giving EasyOCR the natural context it needs for Thai text.

Voter / partial evidence:
  [PL-5]  PlateMajorityVoter now tracks leading_digit_counter separately
          from consonant_candidates. When reconstructing via partial
          evidence, the most-voted leading digit is passed to
          merge_digit_evidence() so "5กข2662" is reconstructed correctly
          instead of dropping the "5" and emitting "กข2662".

  [PL-6]  extract_leading_digit() called on every candidate frame to
          accumulate the zone prefix evidence across frames.

CV (binarization):
  [CV-10] _preprocess_at_scale() now also tries a fourth strategy:
          sauvola-style local threshold approximated via Gaussian diff
          (mu - k*sigma). Helps with non-uniform illumination plates.

All v9 improvements retained:
  [CV-6]  Multi-strategy binarization (Otsu + adaptive-Gaussian-31px +
          adaptive-Mean-25px) + best-by-CC-score selection.
  [CV-7]  3×3 morphological closing.
  [CV-8]  Tighter number-zone crop (10%–72%).
  [CV-9]  Smart province-line detection.
  [OCR-5] Raw-color parallel EasyOCR path.
  [OCR-6] Beamsearch decoder (beamWidth=10) on raw path.
  [PL-4]  Partial-evidence voter.
  [PF-1]  Skip OCR if box barely moved.
  [PF-2]  Tesseract at 2× scale.
  [PF-3]  Province detection cached per track.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

from plate_corrector import (
    correct_plate,
    correct_candidates,
    is_valid_plate,
    extract_digit_suffix,
    extract_leading_digit,
    merge_digit_evidence,
    THAI_CONSONANTS as _CORRECTOR_CONSONANTS,
)

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

# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE MAP  (77 provinces)
# ═══════════════════════════════════════════════════════════════════════════

PROVINCE_MAP: dict[str, dict] = {
    # ── Central ─────────────────────────────────────────────────────────────
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
    # ── North ───────────────────────────────────────────────────────────────
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
    # ── Northeast (Isan) ────────────────────────────────────────────────────
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
    # ── East ────────────────────────────────────────────────────────────────
    "จันทบุรี":         {"en": "Chanthaburi",               "region": "East"},
    "ฉะเชิงเทรา":       {"en": "Chachoengsao",              "region": "East"},
    "ชลบุรี":           {"en": "Chon Buri",                 "region": "East"},
    "ตราด":             {"en": "Trat",                      "region": "East"},
    "ปราจีนบุรี":       {"en": "Prachin Buri",              "region": "East"},
    "ระยอง":            {"en": "Rayong",                    "region": "East"},
    "สระแก้ว":          {"en": "Sa Kaeo",                   "region": "East"},
    # ── West ────────────────────────────────────────────────────────────────
    "กาญจนบุรี":        {"en": "Kanchanaburi",              "region": "West"},
    "ประจวบคีรีขันธ์":  {"en": "Prachuap Khiri Khan",       "region": "West"},
    "เพชรบุรี":         {"en": "Phetchaburi",               "region": "West"},
    "ราชบุรี":          {"en": "Ratchaburi",                "region": "West"},
    # ── South ───────────────────────────────────────────────────────────────
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
# PROVINCE FUZZY MATCHING  [PROV-2]
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

    # [PROV-2] Lowered threshold: 0.25 (was 0.30) to catch truncated long names
    if best_province and best_score >= 0.25:
        info = PROVINCE_MAP[best_province]
        return {
            "thai":   best_province,
            "en":     info["en"],
            "region": info["region"],
            "score":  round(best_score, 3),
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE STRIP EXTRACTION  [PROV-1, PROV-3, PROV-4]
# ═══════════════════════════════════════════════════════════════════════════

# [PROV-1] Wider strips — province text needs more vertical pixels.
# Ordered: widest/most-reliable first to maximize early-exit hits.
_STRIP_CANDIDATES = [
    (0.65, 1.00, "bottom_wide"),    # 35% of height — generous
    (0.70, 1.00, "bottom"),         # 30% of height
    (0.75, 1.00, "bottom_tight"),   # 25% of height
    (0.00, 0.25, "top"),            # top strip fallback
]


def _preprocess_province_strip(strip_bgr: np.ndarray) -> list:
    """
    [PROV-3] Returns TWO binarized versions: normal + inverted polarity.
    Some plates (newer Thai plates) have a dark province strip background.
    """
    h, w  = strip_bgr.shape[:2]
    scale = max(1, int(np.ceil(80 / max(h, 1))))
    up    = cv2.resize(strip_bgr, (w * scale, h * scale),
                       interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    normal = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    inverted = cv2.bitwise_not(normal)
    return [normal, inverted]


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
    """
    [PROV-1] Probe multiple strips with wider bottom cuts.
    [PROV-3] Try both polarities for Tesseract.
    [PROV-4] Also run EasyOCR on the full crop (not just the strip)
             to give it maximum context for long province names.
    Stop early when confidence is high enough.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    h, w = bgr_crop.shape[:2]
    if h < 10:
        return None

    best_result, best_score = None, 0.0

    # [PROV-4] Try full crop with EasyOCR first — gives maximum context
    if easyocr_reader is not None:
        raw = _ocr_strip_easyocr(easyocr_reader, bgr_crop)
        if raw:
            # Filter out plate number tokens (digits/consonants only segments)
            # Keep only tokens that look like province text (vowels present)
            province_tokens = []
            for token in raw.split():
                if any(ch in THAI_VOWELS for ch in token):
                    province_tokens.append(token)
            province_raw = " ".join(province_tokens) if province_tokens else raw
            match = normalize_province(province_raw)
            if match and match["score"] > best_score:
                best_score  = match["score"]
                best_result = {**match, "strip": "full_crop"}
                print(f"  [Province full_crop] EasyOCR '{province_raw}' "
                      f"→ {match['thai']} ({match['score']:.2f})")
            if best_score >= 0.90:
                return best_result

    for y0_r, y1_r, label in _STRIP_CANDIDATES:
        y0, y1 = int(h * y0_r), int(h * y1_r)
        if y1 - y0 < 6:
            continue
        strip = bgr_crop[y0:y1, :]

        # EasyOCR on BGR strip
        if easyocr_reader is not None:
            raw = _ocr_strip_easyocr(easyocr_reader, strip)
            if raw:
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    print(f"  [Province strip={label}] EasyOCR '{raw}' "
                          f"→ {match['thai']} ({match['score']:.2f})")

        # [PROV-3] Tesseract on both polarities
        for binary in _preprocess_province_strip(strip):
            raw = _ocr_strip_tesseract(binary)
            if raw:
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    print(f"  [Province strip={label}] Tesseract '{raw}' "
                          f"→ {match['thai']} ({match['score']:.2f})")

        if best_score >= 0.85:
            break

    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY SUBSTITUTION HELPERS  (kept for province strip OCR only)
# ═══════════════════════════════════════════════════════════════════════════

_LATIN_TO_THAI = [
    ("au", "ม"), ("nn", "ท"), ("iiu", "ปน"), ("ii", "ป"), ("ss", "ส"),
    ("u",  "บ"), ("i",  "บ"), ("n",  "น"),   ("m",  "ม"), ("g",  "ง"),
    ("a",  "า"), ("s",  "ส"), ("w",  "ว"),   ("h",  "ห"), ("o",  "อ"),
    ("x",  "ข"), ("y",  "ย"),
]
_THAI_CORRECTIONS = {
    "รร": "ร",
    "าา": "า",
    "นน": "น",
}
_DIGIT_RE = re.compile(r'\d+')


def correct_ocr_substitutions(text: str) -> str:
    parts, last = [], 0
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
            for bad, good in _THAI_CORRECTIONS.items():
                seg = seg.replace(bad, good)
            result.append(seg)
    return "".join(result)


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


def adaptive_min_confidence(yolo_conf: float) -> float:
    """[OCR-2] Scale OCR min confidence inversely with YOLO confidence."""
    yolo_conf = max(0.40, min(0.80, yolo_conf))
    return 0.20 - (yolo_conf - 0.40) / 0.40 * 0.15


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
# IMAGE PRE-PROCESSING  [CV-1, CV-4, CV-6, CV-7, CV-8, CV-9, CV-10]
# ═══════════════════════════════════════════════════════════════════════════

def _sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    """[CV-4] CLAHE on L channel of LAB colourspace."""
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab     = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _score_binarized(img: np.ndarray) -> float:
    """
    [CV-6] Score a binarized image by connected-component plausibility.
    Target: 6–8 large blobs (2 consonants + 4 digits + possible diacritics).
    Returns a score in [0, 1] — higher is better.
    """
    work = cv2.bitwise_not(img) if np.mean(img) > 127 else img.copy()
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    if num_labels <= 1:
        return 0.0
    h_img, w_img = img.shape
    total_area   = h_img * w_img
    area_min     = total_area * 0.003
    area_max     = total_area * 0.25
    valid = sum(
        1 for i in range(1, num_labels)
        if area_min < stats[i, cv2.CC_STAT_AREA] < area_max
    )
    score = 1.0 - abs(valid - 7) / 10.0
    return max(0.0, score)


def _preprocess_at_scale(
    crop_bgr: np.ndarray,
    scale: int,
) -> np.ndarray:
    """
    [CV-6]  Multi-strategy binarization — four parallel strategies,
            winner selected by connected-component plausibility score.
    [CV-7]  Larger 3×3 morphological closing kernel for Thai strokes.
    [CV-10] Fourth strategy: local contrast enhancement via Gaussian diff.
    """
    h, w = crop_bgr.shape[:2]
    up   = cv2.resize(crop_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq       = clahe.apply(denoised)
    blurred  = cv2.GaussianBlur(eq, (0, 0), sigmaX=3)
    sharp    = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)

    # [CV-6] Strategy A: large adaptive Gaussian — key fix for Thai consonants
    thresh_a = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8,
    )

    # [CV-6] Strategy B: Otsu on CLAHE-equalised gray
    _, thresh_b = cv2.threshold(
        sharp, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # [CV-6] Strategy C: adaptive mean — gentler on thin strokes
    thresh_c = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 25, 6,
    )

    # [CV-10] Strategy D: local contrast (Sauvola-like approximation)
    # mu - k*sigma via Gaussian blur difference
    blur_d   = cv2.GaussianBlur(eq, (21, 21), 0)
    diff_d   = cv2.subtract(eq, blur_d)
    _, thresh_d = cv2.threshold(
        diff_d, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Select strategy with highest CC plausibility score
    scored = [(
        _score_binarized(t), t
    ) for t in (thresh_a, thresh_b, thresh_c, thresh_d)]
    best_score_val, thresh = max(scored, key=lambda x: x[0])

    # Log which strategy won (useful for debugging)
    strategy_names = ['AdaptGauss31', 'Otsu', 'AdaptMean25', 'LocalContrast']
    best_idx = scored.index((best_score_val, thresh))
    _ = strategy_names[best_idx]   # available for debug print if needed

    # [CV-3] Contrast-aware polarity inversion
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # [CV-7] Larger closing kernel: 3×3 reconnects broken Thai strokes
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kern_open)

    return thresh


def crop_number_zone(bgr_crop: np.ndarray) -> np.ndarray:
    """
    [CV-9] Smart province-line detection with padding.

    YOLO already crops the plate tightly, so we must be conservative:
    - Top cut: only 5% (was 10%) — avoid clipping tall Thai consonants
    - Bottom: detect province strip boundary via row brightness drop
    - Padding: add 4px white padding on all sides after cropping so
      characters never touch the binarized image edge (fixes right-side
      digit clipping seen in debug images)

    Thai plate anatomy (approximate):
      Top border:      ~5–8% of height
      Number line:     ~8–70% of height
      Province strip:  ~70–90% of height
      Bottom border:   ~90–100% of height
    """
    h, w = bgr_crop.shape[:2]
    if h < 20:
        return bgr_crop

    gray     = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    row_mean = np.mean(gray, axis=1)

    scan_start   = int(h * 0.55)
    scan_zone    = row_mean[scan_start:]
    overall_mean = float(np.mean(row_mean))

    # Province strip is noticeably darker than the white number zone
    threshold = overall_mean * 0.80
    drop_rows = np.where(scan_zone < threshold)[0]

    if len(drop_rows) > 0:
        province_row = scan_start + int(drop_rows[0])
        # Cut just above the province strip; clamp to [60%, 85%]
        cut_bottom = max(int(h * 0.60), min(province_row - 2, int(h * 0.85)))
    else:
        cut_bottom = int(h * 0.74)

    # [CV-11] Conservative top cut: 5% only (was 10%)
    cut_top = int(h * 0.05)

    cropped = bgr_crop[cut_top: cut_bottom, :]

    # [CV-11] Add padding — more on left/right (chars clip horizontally)
    # than top/bottom. 8px sides, 4px top/bottom.
    PAD_H = 8   # horizontal — fixes right-digit clipping
    PAD_V = 4   # vertical
    padded = cv2.copyMakeBorder(
        cropped, PAD_V, PAD_V, PAD_H, PAD_H,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    return padded


def preprocess_plate_image(
    bgr_crop: np.ndarray,
    scale: int = 3,
    number_zone_only: bool = True,
) -> np.ndarray:
    """
    [CV-1] Multi-scale preprocessing: run at two scales, return sharper.
    [CV-4] LAB-CLAHE illumination normalisation applied first.
    [CV-8] Tighter crop: uses crop_number_zone() for smart province cutoff.
    [CV-6, CV-10] Multi-strategy binarization inside _preprocess_at_scale().
    [CV-11] crop_number_zone() now adds 4px padding so chars never clip.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop passed to preprocess_plate_image")

    bgr_crop = _clahe_lab(bgr_crop)  # [CV-4]

    h, w = bgr_crop.shape[:2]
    if number_zone_only and h > 30:
        crop = crop_number_zone(bgr_crop)   # [CV-8, CV-9, CV-11]
    else:
        # Still add padding even when not cropping number zone only
        PAD  = 4
        crop = cv2.copyMakeBorder(
            bgr_crop, PAD, PAD, PAD, PAD,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )

    ch, cw = crop.shape[:2]
    if max(ch, cw) >= 200:
        return _preprocess_at_scale(crop, 1)

    # [CV-1] Compare two scales; keep the sharper result
    results = []
    for s in (max(1, scale - 1), scale):
        proc      = _preprocess_at_scale(crop, s)
        sharpness = _sharpness_laplacian(proc)
        results.append((sharpness, proc))
    return max(results, key=lambda x: x[0])[1]


# ═══════════════════════════════════════════════════════════════════════════
# DESKEW  [CV-5]
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_skew_angle(gray: np.ndarray) -> float:
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
    """[CV-5] Sub-degree affine deskew."""
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop
    gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.5:
        return bgr_crop
    h, w  = bgr_crop.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(bgr_crop, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════════════
# OCR ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def run_easyocr(reader, preprocessed_gray: np.ndarray) -> list:
    """Standard EasyOCR on preprocessed/binarized image."""
    return reader.readtext(
        preprocessed_gray,
        detail    = 1,
        paragraph = False,
        allowlist = (THAI_ALLOWLIST
                     + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    )


def run_easyocr_raw(reader, raw_bgr: np.ndarray) -> list:
    """
    [OCR-5] EasyOCR on raw BGR crop — no binarization.
    [OCR-6] beamWidth=10 for wider hypothesis search.
    [OCR-7] MIN_CONFIDENCE floor lowered to 0.05.
    """
    if raw_bgr is None or raw_bgr.size == 0:
        return []

    h, w = raw_bgr.shape[:2]
    if h < 30:
        return []
    if h < 60 or h > 110:
        target_h  = 80
        new_scale = target_h / max(h, 1)
        raw_bgr   = cv2.resize(raw_bgr,
                               (int(w * new_scale), target_h),
                               interpolation=cv2.INTER_CUBIC)

    bgr = _clahe_lab(raw_bgr)

    try:
        results = reader.readtext(
            bgr,
            detail     = 1,
            paragraph  = False,
            allowlist  = (THAI_ALLOWLIST
                          + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
            decoder    = "beamsearch",
            beamWidth  = 10,
        )
        return [(bbox, text, conf) for bbox, text, conf in results
                if conf >= 0.05]
    except Exception as exc:
        print(f"  [EasyOCR raw error] {exc}")
        try:
            return reader.readtext(
                bgr,
                detail    = 1,
                paragraph = False,
                allowlist = (THAI_ALLOWLIST
                             + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
            )
        except Exception as exc2:
            print(f"  [EasyOCR raw fallback error] {exc2}")
            return []


def run_tesseract(preprocessed_gray: np.ndarray) -> list:
    """[PF-2] Internally downscales to 2× for speed."""
    if not _TESSERACT_AVAILABLE:
        return []
    try:
        h, w  = preprocessed_gray.shape[:2]
        small = (cv2.resize(preprocessed_gray, (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
                 if max(h, w) > 400 else preprocessed_gray)
        data  = pytesseract.image_to_data(
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
    min_confidence: float = 0.05,
) -> list:
    """
    [OCR-1] Merge EasyOCR (preprocessed + raw paths) + Tesseract.
    Agreement between any two sources boosts confidence by +10%.
    """
    combined: dict[str, tuple] = {}

    for result_list in (easy_results, tess_results):
        for item in result_list:
            if len(item) < 3:
                continue
            bbox, text, conf = item[0], item[1], float(item[2])
            if not isinstance(text, str) or conf < min_confidence:
                continue
            key = normalize_plate_text(text)
            if not key:
                continue
            if key not in combined:
                combined[key] = (bbox, text, conf)
            else:
                old_conf = combined[key][2]
                better   = max(conf, old_conf)
                boosted  = min(1.0, better * 1.10)
                combined[key] = (
                    bbox if conf > old_conf else combined[key][0],
                    text if conf > old_conf else combined[key][1],
                    boosted,
                )

    return list(combined.values())


# ═══════════════════════════════════════════════════════════════════════════
# POST-PROCESSING EXTRACTION  [PP-1, PP-2]
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_best_plate_read(
    ocr_results,
    min_length: int             = 3,
    min_confidence: float       = 0.05,
    max_correction_score: float = 0.65,
) -> Optional[Tuple[str, float]]:
    """
    [PP-1] Domain-enforced corrector on every OCR candidate.
    [PP-2] Correction-score confidence penalty.
    [PP-4] Completeness preference — a standard plate (consonants + digits)
           always beats a digit-only partial read, regardless of confidence.
           Within the same tier, highest confidence wins.
           Tiers (highest → lowest):
             3 = standard plate with leading digit  e.g. 5กข1234
             2 = standard plate no leading digit    e.g. กข1234
             1 = digit-only 3–4 chars               e.g. 6912
             0 = digit-only 1–2 chars               e.g. 69
    """
    if not ocr_results:
        return None

    def top_y(result):
        try:
            return min(float(pt[1]) for pt in result[0])
        except Exception:
            return 0.0

    sorted_results = sorted(ocr_results, key=top_y)

    raw_candidates = []
    for result in sorted_results:
        if len(result) < 3:
            continue
        text, conf = result[1], _safe_float(result[2])
        if not isinstance(text, str) or conf < min_confidence:
            continue
        raw_candidates.append((text, conf))

    if not raw_candidates:
        return None

    # Also try left-to-right merge of all fragments
    merged      = "".join(t for t, _ in raw_candidates)
    merged_conf = float(np.mean([c for _, c in raw_candidates]))
    raw_candidates.append((merged, merged_conf))

    corrected_candidates = correct_candidates(
        raw_candidates,
        max_correction_score=max_correction_score,
    )

    if not corrected_candidates:
        return None

    def _completeness_tier(plate: str) -> int:
        """Higher = more complete / preferred."""
        from plate_corrector import _PLATE_RE, _SPECIAL_RE
        norm = plate.upper().strip()
        m = _PLATE_RE.fullmatch(norm)
        if m:
            return 3 if m.group(1) else 2   # with / without leading digit
        if _SPECIAL_RE.fullmatch(norm):
            return 1 if len(norm) >= 3 else 0
        return 0

    # Sort: completeness tier DESC, then confidence DESC
    corrected_candidates.sort(key=lambda x: (_completeness_tier(x[0]), x[1]),
                              reverse=True)
    return corrected_candidates[0]


# ═══════════════════════════════════════════════════════════════════════════
# IoU-BASED PLATE TRACKER  [PL-1]
# ═══════════════════════════════════════════════════════════════════════════

def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateTracker:
    """[PL-1] Lightweight IoU tracker with province cache and OCR-skip state."""

    IOU_THRESHOLD = 0.35
    MAX_MISSED    = 8

    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}

    def _new_track(self, box: Tuple) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box":           box,
            "missed":        0,
            "province_info": None,
            "last_ocr_box":  None,
        }
        return tid

    def update(self, boxes: List[Tuple]) -> List[Tuple[int, Tuple]]:
        for tid in self._tracks:
            self._tracks[tid]["missed"] += 1

        assignments, used_tids = [], set()

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

        dead = [t for t, d in self._tracks.items()
                if d["missed"] > self.MAX_MISSED]
        for tid in dead:
            del self._tracks[tid]

        return assignments

    def get_province_cache(self, tid: int) -> Optional[dict]:
        return self._tracks.get(tid, {}).get("province_info")

    def set_province_cache(self, tid: int, info):
        if tid in self._tracks:
            self._tracks[tid]["province_info"] = info

    def get_last_ocr_box(self, tid: int) -> Optional[Tuple]:
        return self._tracks.get(tid, {}).get("last_ocr_box")

    def set_last_ocr_box(self, tid: int, box: Tuple):
        if tid in self._tracks:
            self._tracks[tid]["last_ocr_box"] = box

    def reset_track(self, tid: int):
        if tid in self._tracks:
            self._tracks[tid]["last_ocr_box"]  = None
            self._tracks[tid]["province_info"] = None


# ═══════════════════════════════════════════════════════════════════════════
# STABILITY GATE  [PL-3]
# ═══════════════════════════════════════════════════════════════════════════

class PlateStabilityGate:
    """[PL-3] Hysteresis: exit stable only after HYSTERESIS_COUNT bad frames."""

    HYSTERESIS_COUNT = 2

    def __init__(self, min_stable_frames: int = 4,
                 max_pixel_drift: float = 6.0):
        self.min_stable_frames = min_stable_frames
        self.max_pixel_drift   = max_pixel_drift
        self._history: list    = []
        self._stable_count     = 0
        self._unstable_count   = 0
        self._is_stable        = False

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
            self._is_stable    = False
            self._stable_count = 0

        return self._is_stable

    def reset(self):
        self._history.clear()
        self._stable_count   = 0
        self._unstable_count = 0
        self._is_stable      = False


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED MAJORITY VOTER  [PL-2, PL-4, PL-5, PL-6]
# ═══════════════════════════════════════════════════════════════════════════

class PlateMajorityVoter:
    """
    [PL-2] Weighted vote: yolo_conf × adjusted_ocr_conf, recency bonus.
    [PP-3] Winner is re-validated via is_valid_plate() before return.
    [PL-4] Partial-evidence accumulation for consonants + digit suffix.
    [PL-5] Leading zone digit tracked separately and passed to
           merge_digit_evidence() so the reconstructed plate includes
           the prefix (e.g., 5กข2662 not กข2662).
    [PL-6] extract_leading_digit() called per frame to accumulate evidence.
    """

    RECENCY_BONUS      = 0.10
    MIN_DIGIT_VOTES    = 3     # frames with same digit string before firing
    EARLY_EXIT_VOTES   = 3     # consecutive identical valid reads → fire immediately
    EARLY_EXIT_MIN_OCR = 0.40

    def __init__(self, window: int = 8, min_votes: int = 3):
        self.window    = window
        self.min_votes = min_votes
        self._buffer: List[Tuple[str, float, float]] = []
        self._digit_suffix_counter: Counter = Counter()
        self._consonant_candidates: list    = []
        self._leading_digit_counter: Counter = Counter()
        self._streak_text: Optional[str]    = None
        self._streak_confs: list            = []

    def update(self, candidate, yolo_conf: float = 1.0):
        if candidate is not None:
            text, ocr_conf = candidate

            # [PL-7] Early exit — same valid plate N frames in a row.
            # BLOCK digit-only early exit if consonants have been seen —
            # means we have a standard plate and "69" is just a fragment.
            if is_valid_plate(text):
                norm = normalize_plate_text(text)
                is_digit_only = norm.isdigit()
                has_seen_consonants = bool(self._consonant_candidates)

                if is_digit_only and has_seen_consonants:
                    # Don't fire — wait for the full plate reconstruction
                    pass
                else:
                    if text == self._streak_text:
                        self._streak_confs.append(float(ocr_conf))
                        if len(self._streak_confs) >= self.EARLY_EXIT_VOTES:
                            avg = float(np.mean(self._streak_confs))
                            if avg >= self.EARLY_EXIT_MIN_OCR:
                                print(f"  [Voter PL-7] Early exit: {text} "
                                      f"x{len(self._streak_confs)} avg_conf={avg:.2f}")
                                return (text, avg)
                    else:
                        self._streak_text  = text
                        self._streak_confs = [float(ocr_conf)]

            # [PL-6] Leading digit evidence
            leading = extract_leading_digit(text)
            if leading:
                self._leading_digit_counter[leading] += 1

            # [PL-4] Digit suffix evidence — also track raw digit-only reads
            # so "69" seen repeatedly contributes even without 4 full digits
            digit_suffix = extract_digit_suffix(text)
            if digit_suffix:
                self._digit_suffix_counter[digit_suffix] += 1
            elif normalize_plate_text(text).isdigit():
                # Raw digit fragment (e.g. "69") — count it directly
                self._digit_suffix_counter[normalize_plate_text(text)] += 1

            # [PL-4] Consonant evidence
            norm           = normalize_plate_text(text)
            consonant_part = "".join(c for c in norm if c in _CORRECTOR_CONSONANTS)
            if len(consonant_part) in (2, 3) and float(ocr_conf) > 0.15:
                if consonant_part not in self._consonant_candidates:
                    self._consonant_candidates.append(consonant_part)

            # [PL-5] Leading digit + consonant group (e.g. "5กข")
            if len(norm) >= 3 and norm[0].isdigit() and len(norm) > 4:
                cons_with_prefix = "".join(
                    c for c in norm if c in _CORRECTOR_CONSONANTS or c.isdigit()
                )
                if (len(cons_with_prefix) >= 3
                        and cons_with_prefix[0].isdigit()
                        and all(c in _CORRECTOR_CONSONANTS for c in cons_with_prefix[1:3])):
                    cand = cons_with_prefix[:3]
                    if cand not in self._consonant_candidates:
                        self._consonant_candidates.append(cand)

            if is_valid_plate(text):
                self._buffer.append((text, float(ocr_conf), float(yolo_conf)))

        # ── Full-plate majority vote ───────────────────────────────────────
        if len(self._buffer) >= self.window:
            half     = self.window // 2
            weighted = defaultdict(float)
            conf_acc = defaultdict(list)

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

            if vote_count >= self.min_votes and is_valid_plate(best_text):
                return (best_text, avg_conf)

        # ── [PL-4, PL-5] Partial-evidence reconstruction ──────────────────
        if self._digit_suffix_counter:
            best_suffix, suffix_count = self._digit_suffix_counter.most_common(1)[0]
            if suffix_count >= self.MIN_DIGIT_VOTES and self._consonant_candidates:
                best_leading = None
                if self._leading_digit_counter:
                    best_leading, _ = self._leading_digit_counter.most_common(1)[0]
                reconstructed = merge_digit_evidence(
                    best_suffix, self._consonant_candidates,
                    leading_digit=best_leading,
                )
                if reconstructed:
                    print(f"  [Voter PL-4/5] Reconstructed: {reconstructed[0]} "
                          f"(suffix={best_suffix} x{suffix_count}, "
                          f"consonants={self._consonant_candidates}, "
                          f"leading={best_leading})")
                    return (reconstructed[0], 0.45)

        return None

    def reset(self):
        self._buffer.clear()
        self._digit_suffix_counter.clear()
        self._consonant_candidates.clear()
        self._leading_digit_counter.clear()
        self._streak_text  = None
        self._streak_confs = []