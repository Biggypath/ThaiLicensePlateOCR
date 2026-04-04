"""
plate_utils.py — Thai licence-plate helper utilities  (v10.1 — Single Camera)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single-camera version — no dual-camera cross-contamination logic.

Key improvements retained from multi-camera version:
  [OCR-1]  Dual-engine fusion (EasyOCR + Tesseract)
  [OCR-2]  Adaptive confidence threshold
  [CV-1]   Multi-scale preprocessing
  [CV-2]   Morphological cleanup
  [CV-3]   Contrast-aware inversion
  [CV-4]   CLAHE LAB illumination normalisation
  [CV-5]   Deskew via Hough
  [PL-1]   IoU tracker
  [PL-2]   Weighted majority voter
  [PL-3]   Stability gate with hysteresis
  [PL-4]   Partial evidence reconstruction
  [FIX-1]  Leading-digit consonant preserved ("3กบ" not "กบ")
  [FIX-2]  Valid plate stored in consonant candidates immediately
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
    merge_digit_evidence,
    THAI_CONSONANTS as _CORRECTOR_CONSONANTS,
    normalise_raw as _normalise_raw,
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
    "-c tessedit_char_whitelist=" + THAI_CONSONANTS + THAI_VOWELS + " "
)

# [FIX-1] Capture optional leading digit + exactly 2 consonants
_CONS_PREFIX_RE = re.compile(
    r'^([0-9]?)([' + _CORRECTOR_CONSONANTS + r']{2})'
)

# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE MAP (77 provinces)
# ═══════════════════════════════════════════════════════════════════════════

PROVINCE_MAP: dict[str, dict] = {
    "กรุงเทพมหานคร":   {"en": "Bangkok",                   "region": "Central"},
    "กำแพงเพชร":       {"en": "Kamphaeng Phet",            "region": "Central"},
    "ชัยนาท":          {"en": "Chai Nat",                  "region": "Central"},
    "นครนายก":         {"en": "Nakhon Nayok",              "region": "Central"},
    "นครปฐม":          {"en": "Nakhon Pathom",             "region": "Central"},
    "นครสวรรค์":       {"en": "Nakhon Sawan",              "region": "Central"},
    "นนทบุรี":         {"en": "Nonthaburi",                "region": "Central"},
    "ปทุมธานี":        {"en": "Pathum Thani",              "region": "Central"},
    "พระนครศรีอยุธยา": {"en": "Phra Nakhon Si Ayutthaya", "region": "Central"},
    "พิจิตร":          {"en": "Phichit",                   "region": "Central"},
    "พิษณุโลก":        {"en": "Phitsanulok",               "region": "Central"},
    "เพชรบูรณ์":       {"en": "Phetchabun",                "region": "Central"},
    "ลพบุรี":          {"en": "Lopburi",                   "region": "Central"},
    "สมุทรปราการ":     {"en": "Samut Prakan",              "region": "Central"},
    "สมุทรสงคราม":     {"en": "Samut Songkhram",           "region": "Central"},
    "สมุทรสาคร":       {"en": "Samut Sakhon",              "region": "Central"},
    "สระบุรี":         {"en": "Saraburi",                  "region": "Central"},
    "สิงห์บุรี":       {"en": "Sing Buri",                 "region": "Central"},
    "สุพรรณบุรี":      {"en": "Suphan Buri",               "region": "Central"},
    "อ่างทอง":         {"en": "Ang Thong",                 "region": "Central"},
    "อุทัยธานี":       {"en": "Uthai Thani",               "region": "Central"},
    "เชียงราย":        {"en": "Chiang Rai",                "region": "North"},
    "เชียงใหม่":       {"en": "Chiang Mai",                "region": "North"},
    "ตาก":             {"en": "Tak",                       "region": "North"},
    "น่าน":            {"en": "Nan",                       "region": "North"},
    "พะเยา":           {"en": "Phayao",                    "region": "North"},
    "แพร่":            {"en": "Phrae",                     "region": "North"},
    "แม่ฮ่องสอน":      {"en": "Mae Hong Son",              "region": "North"},
    "ลำปาง":           {"en": "Lampang",                   "region": "North"},
    "ลำพูน":           {"en": "Lamphun",                   "region": "North"},
    "สุโขทัย":         {"en": "Sukhothai",                 "region": "North"},
    "อุตรดิตถ์":       {"en": "Uttaradit",                 "region": "North"},
    "กาฬสินธุ์":       {"en": "Kalasin",                   "region": "Northeast"},
    "ขอนแก่น":         {"en": "Khon Kaen",                 "region": "Northeast"},
    "ชัยภูมิ":         {"en": "Chaiyaphum",                "region": "Northeast"},
    "นครพนม":          {"en": "Nakhon Phanom",             "region": "Northeast"},
    "นครราชสีมา":      {"en": "Nakhon Ratchasima",         "region": "Northeast"},
    "บึงกาฬ":          {"en": "Bueng Kan",                 "region": "Northeast"},
    "บุรีรัมย์":       {"en": "Buriram",                   "region": "Northeast"},
    "มหาสารคาม":       {"en": "Maha Sarakham",             "region": "Northeast"},
    "มุกดาหาร":        {"en": "Mukdahan",                  "region": "Northeast"},
    "ยโสธร":           {"en": "Yasothon",                  "region": "Northeast"},
    "ร้อยเอ็ด":        {"en": "Roi Et",                    "region": "Northeast"},
    "เลย":             {"en": "Loei",                      "region": "Northeast"},
    "ศรีสะเกษ":        {"en": "Si Sa Ket",                 "region": "Northeast"},
    "สกลนคร":          {"en": "Sakon Nakhon",              "region": "Northeast"},
    "สุรินทร์":        {"en": "Surin",                     "region": "Northeast"},
    "หนองคาย":         {"en": "Nong Khai",                 "region": "Northeast"},
    "หนองบัวลำภู":     {"en": "Nong Bua Lamphu",           "region": "Northeast"},
    "อำนาจเจริญ":      {"en": "Amnat Charoen",             "region": "Northeast"},
    "อุดรธานี":        {"en": "Udon Thani",                "region": "Northeast"},
    "อุบลราชธานี":     {"en": "Ubon Ratchathani",          "region": "Northeast"},
    "จันทบุรี":        {"en": "Chanthaburi",               "region": "East"},
    "ฉะเชิงเทรา":      {"en": "Chachoengsao",              "region": "East"},
    "ชลบุรี":          {"en": "Chon Buri",                 "region": "East"},
    "ตราด":            {"en": "Trat",                      "region": "East"},
    "ปราจีนบุรี":      {"en": "Prachin Buri",              "region": "East"},
    "ระยอง":           {"en": "Rayong",                    "region": "East"},
    "สระแก้ว":         {"en": "Sa Kaeo",                   "region": "East"},
    "กาญจนบุรี":       {"en": "Kanchanaburi",              "region": "West"},
    "ประจวบคีรีขันธ์": {"en": "Prachuap Khiri Khan",       "region": "West"},
    "เพชรบุรี":        {"en": "Phetchaburi",               "region": "West"},
    "ราชบุรี":         {"en": "Ratchaburi",                "region": "West"},
    "กระบี่":          {"en": "Krabi",                     "region": "South"},
    "ชุมพร":           {"en": "Chumphon",                  "region": "South"},
    "ตรัง":            {"en": "Trang",                     "region": "South"},
    "นครศรีธรรมราช":   {"en": "Nakhon Si Thammarat",       "region": "South"},
    "นราธิวาส":        {"en": "Narathiwat",                "region": "South"},
    "ปัตตานี":         {"en": "Pattani",                   "region": "South"},
    "พัทลุง":          {"en": "Phatthalung",               "region": "South"},
    "พังงา":           {"en": "Phang Nga",                 "region": "South"},
    "ภูเก็ต":          {"en": "Phuket",                    "region": "South"},
    "ระนอง":           {"en": "Ranong",                    "region": "South"},
    "สงขลา":           {"en": "Songkhla",                  "region": "South"},
    "สตูล":            {"en": "Satun",                     "region": "South"},
    "สุราษฎร์ธานี":    {"en": "Surat Thani",               "region": "South"},
    "ยะลา":            {"en": "Yala",                      "region": "South"},
}

_ALL_PROVINCE_NAMES: list[str] = list(PROVINCE_MAP.keys())


def _consonants_only(text: str) -> str:
    return "".join(ch for ch in text if ch in THAI_CONSONANTS or ch == " ")


_PROVINCE_CONSONANTS = {name: _consonants_only(name) for name in _ALL_PROVINCE_NAMES}


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE FUZZY MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def _char_jaccard(a: str, b: str) -> float:
    ca, cb = Counter(a), Counter(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def _ngram_score(a: str, b: str, n: int = 2) -> float:
    def ngrams(s): return Counter(s[i:i+n] for i in range(len(s) - n + 1))
    ca, cb = ngrams(a), ngrams(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def _prefix_score(raw: str, province: str) -> float:
    rc, pc = _consonants_only(raw), _consonants_only(province)
    if not rc or not pc: return 0.0
    min_len = min(len(rc), len(pc))
    matches = sum(1 for i in range(min_len) if rc[i] == pc[i])
    if matches < 2: return 0.0
    return min(0.85, matches / len(rc))


def normalize_province(raw: str) -> Optional[dict]:
    if not raw or len(raw.strip()) < 2: return None
    raw_clean = raw.strip()
    raw_cons  = _consonants_only(raw_clean)
    if raw_clean in PROVINCE_MAP:
        info = PROVINCE_MAP[raw_clean]
        return {"thai": raw_clean, "en": info["en"], "region": info["region"], "score": 1.0}
    best_score, best_province = 0.0, None
    for pname in _ALL_PROVINCE_NAMES:
        pcons = _PROVINCE_CONSONANTS[pname]
        score = max(
            _char_jaccard(raw_clean, pname),
            _char_jaccard(raw_cons, pcons),
            _ngram_score(raw_cons, pcons, n=2) if len(raw_cons) >= 2 else 0.0,
            _prefix_score(raw_clean, pname),
        )
        if score > best_score: best_score, best_province = score, pname
    if best_province and best_score >= 0.20:
        info = PROVINCE_MAP[best_province]
        return {"thai": best_province, "en": info["en"], "region": info["region"], "score": round(best_score, 3)}
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE STRIP EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_STRIP_CANDIDATES = [
    (0.72, 1.00, "bottom"),
    (0.68, 1.00, "bottom_wide"),
    (0.75, 1.00, "bottom_tight"),
    (0.00, 0.22, "top"),
]
_PROV_PSM_MODES = [7, 6, 8]


def _preprocess_province_strip_multi(strip_bgr: np.ndarray) -> list:
    h, w   = strip_bgr.shape[:2]
    scale  = max(1.0, 60 / max(h, 1))
    up     = cv2.resize(strip_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq     = clahe.apply(gray)
    results = []
    for block, c in [(31, 10), (15, 6)]:
        t = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
        img = t if np.mean(t) < 127 else cv2.bitwise_not(t)
        results.append(img)
    _, thr_c = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = thr_c if np.mean(thr_c) < 127 else cv2.bitwise_not(thr_c)
    results.append(img)
    return results


def _ocr_strip_tesseract_multi(strip_bgr: np.ndarray) -> str:
    if not _TESSERACT_AVAILABLE: return ""
    candidates = []
    binaries = _preprocess_province_strip_multi(strip_bgr)
    for psm in _PROV_PSM_MODES:
        config = (f"--oem 1 --psm {psm} "
                  f"-c tessedit_char_whitelist={THAI_CONSONANTS + THAI_VOWELS + ' '}")
        for binary in binaries:
            try:
                text = pytesseract.image_to_string(PILImage.fromarray(binary), lang="tha", config=config).strip()
                if text: candidates.append(text)
            except Exception: pass
    return max(candidates, key=len) if candidates else ""


def _ocr_strip_easyocr(reader, strip_bgr: np.ndarray) -> list:
    results_out = []
    for scale in (1, 2, 3):
        if scale == 1: img = strip_bgr
        else:
            h, w = strip_bgr.shape[:2]
            img  = cv2.resize(strip_bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        try:
            hits = reader.readtext(img, detail=1, paragraph=False,
                                   allowlist=THAI_CONSONANTS + THAI_VOWELS + " ")
            text = " ".join(r[1] for r in sorted(hits, key=lambda r: r[0][0][0]) if r[2] > 0.03)
            if text: results_out.append(text)
        except Exception as exc:
            print(f"  [Province EasyOCR scale={scale}] {exc}")
    return results_out


def extract_province_from_crop(bgr_crop: np.ndarray, easyocr_reader=None) -> Optional[dict]:
    if bgr_crop is None or bgr_crop.size == 0: return None
    h, w = bgr_crop.shape[:2]
    if h < 10: return None
    best_result, best_score = None, 0.0
    for y0_r, y1_r, label in _STRIP_CANDIDATES:
        y0, y1 = int(h * y0_r), int(h * y1_r)
        if y1 - y0 < 4: continue
        strip = bgr_crop[y0:y1, :]
        if easyocr_reader is not None:
            for raw in _ocr_strip_easyocr(easyocr_reader, strip):
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    if best_score >= 0.75: return best_result
        raw = _ocr_strip_tesseract_multi(strip)
        if raw:
            match = normalize_province(raw)
            if match and match["score"] > best_score:
                best_score  = match["score"]
                best_result = {**match, "strip": label}
                if best_score >= 0.75: return best_result
    return best_result


def extract_province_from_ocr_tokens(ocr_results: list, min_len: int = 4) -> Optional[dict]:
    best_result, best_score = None, 0.0
    for item in ocr_results:
        if len(item) < 3: continue
        text, conf = str(item[1]).strip(), float(item[2])
        if any(ch.isdigit() for ch in text): continue
        if len(text) < min_len or conf < 0.03: continue
        thai_chars = sum(1 for ch in text if '\u0E00' <= ch <= '\u0E7F')
        if thai_chars < min_len: continue
        match = normalize_province(text)
        if match and match["score"] > best_score:
            best_score  = match["score"]
            best_result = {**match, "strip": "ocr_token", "ocr_conf": conf}
    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_plate_text(text: str) -> str:
    return "".join(text.strip().split()).replace("-", "").upper()


def looks_like_thai_plate_top_line(text: str, min_length: int = 3) -> bool:
    if not text or len(text) < min_length: return False
    has_digit = any(ch.isdigit() or ch in THAI_DIGITS for ch in text)
    has_thai  = any('\u0E00' <= ch <= '\u0E7F' for ch in text)
    return has_digit and has_thai


def adaptive_min_confidence(yolo_conf: float) -> float:
    yolo_conf = max(0.40, min(0.80, yolo_conf))
    return 0.20 - (yolo_conf - 0.40) / 0.40 * 0.15


def should_send_plate(plate_text: str, last_seen_plates: dict,
                      current_time: float, cooldown_time: float) -> bool:
    if plate_text not in last_seen_plates: return True
    return (current_time - last_seen_plates[plate_text]) > cooldown_time


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def _sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab     = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _score_binarized(img: np.ndarray) -> float:
    work = cv2.bitwise_not(img) if np.mean(img) > 127 else img.copy()
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    if num_labels <= 1: return 0.0
    h_img, w_img = img.shape
    total_area   = h_img * w_img
    area_min, area_max = total_area * 0.003, total_area * 0.25
    valid = sum(1 for i in range(1, num_labels)
                if area_min < stats[i, cv2.CC_STAT_AREA] < area_max)
    return max(0.0, 1.0 - abs(valid - 7) / 10.0)


def _preprocess_at_scale(crop_bgr: np.ndarray, scale: int) -> np.ndarray:
    h, w     = crop_bgr.shape[:2]
    up       = cv2.resize(crop_bgr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq       = clahe.apply(denoised)
    blurred  = cv2.GaussianBlur(eq, (0, 0), sigmaX=3)
    sharp    = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
    thresh_a = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    _, thresh_b = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_c = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 6)
    scored   = [(_score_binarized(t), t) for t in (thresh_a, thresh_b, thresh_c)]
    _, thresh = max(scored, key=lambda x: x[0])
    if np.mean(thresh) > 127: thresh = cv2.bitwise_not(thresh)
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kern_open)
    return thresh


def crop_number_zone(bgr_crop: np.ndarray) -> np.ndarray:
    h, w = bgr_crop.shape[:2]
    if h < 20: return bgr_crop
    gray         = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    row_mean     = np.mean(gray, axis=1)
    scan_start   = int(h * 0.55)
    scan_zone    = row_mean[scan_start:]
    overall_mean = float(np.mean(row_mean))
    threshold    = overall_mean * 0.78
    drop_rows    = np.where(scan_zone < threshold)[0]
    if len(drop_rows) > 0:
        province_row = scan_start + int(drop_rows[0])
        cut_bottom   = max(int(h * 0.60), min(province_row - 4, int(h * 0.85)))
    else:
        cut_bottom = int(h * 0.72)
    return bgr_crop[int(h * 0.10): cut_bottom, :]


def preprocess_plate_image(bgr_crop: np.ndarray, scale: int = 3,
                           number_zone_only: bool = True) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop")
    bgr_crop = _clahe_lab(bgr_crop)
    h, w     = bgr_crop.shape[:2]
    crop     = crop_number_zone(bgr_crop) if (number_zone_only and h > 30) else bgr_crop
    ch, cw   = crop.shape[:2]
    if max(ch, cw) >= 200: return _preprocess_at_scale(crop, 1)
    results = []
    for s in (max(1, scale - 1), scale):
        proc      = _preprocess_at_scale(crop, s)
        sharpness = _sharpness_laplacian(proc)
        results.append((sharpness, proc))
    return max(results, key=lambda x: x[0])[1]


# ═══════════════════════════════════════════════════════════════════════════
# DESKEW
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, 30,
                            minLineLength=max(20, gray.shape[1]//5), maxLineGap=10)
    if lines is None: return 0.0
    angles = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 != x1:
            a = math.degrees(math.atan2(y2-y1, x2-x1))
            if -45 < a < 45: angles.append(a)
    if not angles: return 0.0
    angles.sort()
    trimmed = angles[len(angles)//4: -len(angles)//4] or angles
    return float(np.median(trimmed))


def deskew_plate(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0: return bgr_crop
    gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.5: return bgr_crop
    h, w = bgr_crop.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    return cv2.warpAffine(bgr_crop, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════════════
# OCR ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def run_easyocr(reader, preprocessed_gray: np.ndarray) -> list:
    return reader.readtext(
        preprocessed_gray, detail=1, paragraph=False,
        allowlist=(THAI_ALLOWLIST + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    )


def run_easyocr_raw(reader, raw_bgr: np.ndarray) -> list:
    if raw_bgr is None or raw_bgr.size == 0: return []
    h, w = raw_bgr.shape[:2]
    if h < 30: return []
    if h < 60 or h > 110:
        new_scale = 80 / max(h, 1)
        raw_bgr   = cv2.resize(raw_bgr, (int(w*new_scale), 80), interpolation=cv2.INTER_CUBIC)
    bgr = _clahe_lab(raw_bgr)
    try:
        results = reader.readtext(
            bgr, detail=1, paragraph=False,
            allowlist=(THAI_ALLOWLIST + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
            decoder="beamsearch", beamWidth=10,
        )
        return [(bbox, text, conf) for bbox, text, conf in results if conf >= 0.05]
    except Exception as exc:
        print(f"  [EasyOCR raw] {exc}")
        try:
            return reader.readtext(bgr, detail=1, paragraph=False,
                                   allowlist=(THAI_ALLOWLIST + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        except Exception:
            return []


def run_tesseract(preprocessed_gray: np.ndarray) -> list:
    if not _TESSERACT_AVAILABLE: return []
    try:
        h, w  = preprocessed_gray.shape[:2]
        small = (cv2.resize(preprocessed_gray, (w//2, h//2), interpolation=cv2.INTER_AREA)
                 if max(h, w) > 400 else preprocessed_gray)
        data  = pytesseract.image_to_data(
            PILImage.fromarray(small), lang="tha+eng",
            config=TESSERACT_NUM_CONFIG, output_type=pytesseract.Output.DICT)
        sh, sw     = small.shape[:2]
        dummy_bbox = [[0, 0], [sw, 0], [sw, sh], [0, sh]]
        results    = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text: continue
            conf = int(data["conf"][i])
            if conf < 0: continue
            results.append((dummy_bbox, text, conf / 100.0))
        return results
    except Exception as exc:
        print(f"  [Tesseract] {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# DUAL-ENGINE FUSION
# ═══════════════════════════════════════════════════════════════════════════

def fuse_ocr_results(easy_results: list, tess_results: list,
                     min_confidence: float = 0.05) -> list:
    combined: dict[str, tuple] = {}
    for result_list in (easy_results, tess_results):
        for item in result_list:
            if len(item) < 3: continue
            bbox, text, conf = item[0], item[1], float(item[2])
            if not isinstance(text, str) or conf < min_confidence: continue
            key = normalize_plate_text(text)
            if not key: continue
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
# POST-PROCESSING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    try:    return float(value)
    except: return default


def extract_best_plate_read(ocr_results, min_length: int = 3,
                            min_confidence: float = 0.05,
                            max_correction_score: float = 0.65) -> Optional[Tuple[str, float]]:
    if not ocr_results: return None

    def top_y(r):
        try:    return min(float(pt[1]) for pt in r[0])
        except: return 0.0

    sorted_results = sorted(ocr_results, key=top_y)
    raw_candidates = []
    for result in sorted_results:
        if len(result) < 3: continue
        text, conf = result[1], _safe_float(result[2])
        if not isinstance(text, str) or conf < min_confidence: continue
        raw_candidates.append((text, conf))

    if not raw_candidates: return None
    merged      = "".join(t for t, _ in raw_candidates)
    merged_conf = float(np.mean([c for _, c in raw_candidates]))
    raw_candidates.append((merged, merged_conf))

    corrected = correct_candidates(raw_candidates, max_correction_score=max_correction_score)
    if not corrected: return None
    return max(corrected, key=lambda x: x[1])


# ═══════════════════════════════════════════════════════════════════════════
# IoU-BASED PLATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════

def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter  = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (ax2-ax1) * (ay2-ay1); area_b = (bx2-bx1) * (by2-by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateTracker:
    IOU_THRESHOLD = 0.35
    MAX_MISSED    = 8

    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}

    def _new_track(self, box: Tuple) -> int:
        tid = self._next_id; self._next_id += 1
        self._tracks[tid] = {"box": box, "missed": 0, "province_info": None, "last_ocr_box": None}
        return tid

    def update(self, boxes: List[Tuple]) -> List[Tuple[int, Tuple]]:
        for tid in self._tracks: self._tracks[tid]["missed"] += 1
        assignments, used_tids = [], set()
        for box in boxes:
            best_iou, best_tid = 0.0, None
            for tid, track in self._tracks.items():
                if tid in used_tids: continue
                score = _iou(box, track["box"])
                if score > best_iou: best_iou, best_tid = score, tid
            if best_iou >= self.IOU_THRESHOLD and best_tid is not None:
                self._tracks[best_tid]["box"] = box; self._tracks[best_tid]["missed"] = 0
                used_tids.add(best_tid); assignments.append((best_tid, box))
            else:
                tid = self._new_track(box); used_tids.add(tid); assignments.append((tid, box))
        dead = [t for t, d in self._tracks.items() if d["missed"] > self.MAX_MISSED]
        for tid in dead: del self._tracks[tid]
        return assignments

    def get_province_cache(self, tid): return self._tracks.get(tid, {}).get("province_info")
    def set_province_cache(self, tid, info):
        if tid in self._tracks: self._tracks[tid]["province_info"] = info
    def get_last_ocr_box(self, tid): return self._tracks.get(tid, {}).get("last_ocr_box")
    def set_last_ocr_box(self, tid, box):
        if tid in self._tracks: self._tracks[tid]["last_ocr_box"] = box
    def reset_track(self, tid):
        if tid in self._tracks:
            self._tracks[tid]["last_ocr_box"] = None
            self._tracks[tid]["province_info"] = None


# ═══════════════════════════════════════════════════════════════════════════
# STABILITY GATE
# ═══════════════════════════════════════════════════════════════════════════

class PlateStabilityGate:
    HYSTERESIS_COUNT = 2

    def __init__(self, min_stable_frames: int = 4, max_pixel_drift: float = 6.0):
        self.min_stable_frames = min_stable_frames
        self.max_pixel_drift   = max_pixel_drift
        self._history: list = []
        self._stable_count  = 0; self._unstable_count = 0; self._is_stable = False

    def is_stable(self, x1, y1, x2, y2) -> bool:
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        if self._history:
            px, py = self._history[-1]
            drift  = math.hypot(cx-px, cy-py)
            if drift <= self.max_pixel_drift:
                self._stable_count += 1; self._unstable_count = 0
            else:
                self._unstable_count += 1; self._stable_count = 0
        self._history.append((cx, cy))
        if len(self._history) > self.min_stable_frames + 1: self._history.pop(0)
        if self._stable_count >= self.min_stable_frames: self._is_stable = True
        if self._unstable_count >= self.HYSTERESIS_COUNT:
            self._is_stable = False; self._stable_count = 0
        return self._is_stable

    def reset(self):
        self._history.clear()
        self._stable_count = 0; self._unstable_count = 0; self._is_stable = False


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED MAJORITY VOTER (with Tier-2 partial evidence + Tier-3 LLM)
# ═══════════════════════════════════════════════════════════════════════════

class PlateMajorityVoter:
    RECENCY_BONUS         = 0.10
    MIN_DIGIT_VOTES       = 5
    MIN_LLM_TRIGGER_VOTES = 8

    def __init__(self, window: int = 8, min_votes: int = 3,
                 llm_corrector=None, track_id: int = -1):
        self.window      = window
        self.min_votes   = min_votes
        self._llm        = llm_corrector
        self._tid        = track_id
        self._buffer:               List[Tuple[str, float, float]] = []
        self._digit_suffix_counter: Counter = Counter()
        self._consonant_candidates: list    = []
        self._raw_ocr_seen:         list    = []
        self._frame_count:          int     = 0

    def update(self, candidate, yolo_conf: float = 1.0):
        self._frame_count += 1
        if candidate is not None:
            text, ocr_conf = candidate
            norm = normalize_plate_text(text)
            if norm and norm not in self._raw_ocr_seen:
                self._raw_ocr_seen.append(norm)
            if len(self._raw_ocr_seen) > 20: self._raw_ocr_seen = self._raw_ocr_seen[-20:]

            digit_suffix = extract_digit_suffix(text)
            if digit_suffix: self._digit_suffix_counter[digit_suffix] += 1

            # [FIX-1] Preserve leading digit in consonant candidate
            norm_for_cons = _normalise_raw(text)
            m_cons = _CONS_PREFIX_RE.match(norm_for_cons)
            if m_cons and float(ocr_conf) > 0.15:
                consonant_part = m_cons.group(0)
                if consonant_part not in self._consonant_candidates:
                    self._consonant_candidates.append(consonant_part)

            # [FIX-2] Store full valid plate prefix immediately
            if is_valid_plate(text) and float(ocr_conf) > 0.15:
                norm_valid = _normalise_raw(text)
                m_full = _CONS_PREFIX_RE.match(norm_valid)
                if m_full:
                    full_prefix = m_full.group(0)
                    if full_prefix not in self._consonant_candidates:
                        self._consonant_candidates.append(full_prefix)
                self._buffer.append((text, float(ocr_conf), float(yolo_conf)))
            elif is_valid_plate(text):
                self._buffer.append((text, float(ocr_conf), float(yolo_conf)))

        # Tier 1: full-plate majority vote
        if len(self._buffer) >= self.window:
            half     = self.window // 2
            weighted = defaultdict(float); conf_acc = defaultdict(list)
            for idx, (text, ocr_conf, yc) in enumerate(self._buffer):
                weight = yc * ocr_conf
                if idx >= len(self._buffer) - half: weight *= (1.0 + self.RECENCY_BONUS)
                weighted[text] += weight; conf_acc[text].append(ocr_conf)
            best_text  = max(weighted, key=weighted.__getitem__)
            vote_count = sum(1 for t, _, _ in self._buffer if t == best_text)
            avg_conf   = float(np.mean(conf_acc[best_text]))
            self._buffer.pop(0)
            if vote_count >= self.min_votes and is_valid_plate(best_text):
                return (best_text, avg_conf)

        # Tier 2: partial evidence reconstruction
        if self._digit_suffix_counter:
            best_suffix, suffix_count = self._digit_suffix_counter.most_common(1)[0]
            if suffix_count >= self.MIN_DIGIT_VOTES and self._consonant_candidates:
                sorted_cons   = sorted(self._consonant_candidates, key=len, reverse=True)
                reconstructed = merge_digit_evidence(best_suffix, sorted_cons)
                if reconstructed:
                    print(f"  [Voter Tier2 #{self._tid}] {reconstructed[0]} "
                          f"suffix={best_suffix}×{suffix_count} cons={sorted_cons}")
                    return (reconstructed[0], 0.45)

            # Tier 3: LLM fallback
            if (self._llm is not None and self._llm.is_enabled()
                    and suffix_count >= self.MIN_LLM_TRIGGER_VOTES):
                llm_result = self._llm.get_cached(self._tid)
                if llm_result is not None:
                    plate, conf = llm_result
                    if is_valid_plate(plate):
                        return (plate, conf)
                if (not self._llm.is_pending(self._tid)
                        and self._llm.get_cached(self._tid) is None):
                    self._llm.request_async(
                        tid=self._tid,
                        ocr_candidates=self._raw_ocr_seen[-10:],
                        prior_consonants=self._consonant_candidates,
                        digit_suffix=best_suffix,
                    )
        return None

    def reset(self):
        self._buffer.clear(); self._digit_suffix_counter.clear()
        self._consonant_candidates.clear(); self._raw_ocr_seen.clear()
        self._frame_count = 0
        if self._llm is not None: self._llm.clear_track(self._tid)