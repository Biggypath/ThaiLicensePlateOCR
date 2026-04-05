"""
plate_utils.py — Thai licence-plate helper utilities  v8.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGES vs v7.1 (bug-fix release)
══════════════════════════════════
Bug fixes carried forward
  [FIX-1]  _PLATE_PATTERN requires exactly 2 consonants (was {1,3})
  [FIX-2]  PlateStabilityGate off-by-one: first frame now counts as stable
  [FIX-3]  PlateMajorityVoter accepts llm_corrector= / track_id= kwargs;
           _consonant_candidates attribute present for LLM access
  [FIX-4]  run_easyocr_raw() added
  [FIX-5]  extract_province_from_ocr_tokens() added
  [FIX-6]  extract_best_plate_read() accepts max_correction_score kwarg
           and calls plate_corrector's correct_candidates() properly
  [FIX-7]  PlateMajorityVoter.update() only buffers plate-like reads;
           pure noise no longer consumes vote slots

Enhancements
  [ENH-1]  _preprocess_at_scale() now scores three binarisation methods
           (adaptive-Gaussian, Otsu, adaptive-Mean) and picks the best
           based on connected-component plausibility → cleaner binary
  [ENH-2]  preprocess_plate_image() uses smarter number-zone crop that
           detects the province-strip boundary via row-brightness dip
  [ENH-3]  run_easyocr_raw() tries 3 scales (×1, ×2, ×3) and merges all
           results before returning, giving more coverage on small crops
  [ENH-4]  fuse_ocr_results() tracks per-engine agreement and gives a
           larger boost (+15%) when both engines agree on the same key
  [ENH-5]  PlateMajorityVoter: recency bonus raised to 15%; window slides
           only when at least one valid plate-like read was buffered in
           the current window, preventing stale windows from firing
  [ENH-6]  PlateTracker: IOU threshold tightened to 0.40; on ID-switch
           the old track state (province cache, consonant candidates) is
           migrated to the new ID so a brief occlusion doesn't reset work
  [ENH-7]  Province matching adds bigram overlap score for better fuzzy
           matching of partially-read province strips
  [ENH-8]  _THAI_CORRECTIONS table extended with more observed EasyOCR
           duplication artefacts
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

from plate_corrector import correct_candidates


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

# [FIX-1] Exactly 2 consonants — matches plate_corrector.py and LLM prompt.
_PLATE_PATTERN = re.compile(
    r'^([0-9])?'
    r'([' + THAI_CONSONANTS + r']{2})'
    r'([0-9]{4})$'
)

_CONSONANT_SET = set(THAI_CONSONANTS)


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


_PROVINCE_CONSONANTS: dict[str, str] = {
    name: _consonants_only(name) for name in _ALL_PROVINCE_NAMES
}


# ═══════════════════════════════════════════════════════════════════════════
# PROVINCE FUZZY MATCHING  [ENH-7]
# ═══════════════════════════════════════════════════════════════════════════

def _char_jaccard(a: str, b: str) -> float:
    ca, cb = Counter(a), Counter(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def _bigram_jaccard(a: str, b: str) -> float:
    """[ENH-7] Bigram overlap — handles partially read province strips better."""
    def bigrams(s: str) -> Counter:
        return Counter(s[i:i+2] for i in range(len(s) - 1))
    ca, cb = bigrams(a), bigrams(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def _prefix_score(raw: str, province: str) -> float:
    rc, pc = _consonants_only(raw), _consonants_only(province)
    if not rc or not pc:
        return 0.0
    min_len = min(len(rc), len(pc))
    matches = sum(1 for i in range(min_len) if rc[i] == pc[i])
    if matches < 2:
        return 0.0
    return min(0.85, matches / max(len(rc), 1))


def normalize_province(raw: str) -> Optional[dict]:
    if not raw or len(raw.strip()) < 2:
        return None
    raw_clean = raw.strip()
    raw_cons  = _consonants_only(raw_clean)

    if raw_clean in PROVINCE_MAP:
        info = PROVINCE_MAP[raw_clean]
        return {"thai": raw_clean, "en": info["en"],
                "region": info["region"], "score": 1.0}

    best_score, best_province = 0.0, None
    for pname in _ALL_PROVINCE_NAMES:
        pcons = _PROVINCE_CONSONANTS[pname]
        score = max(
            _char_jaccard(raw_clean, pname),
            _char_jaccard(raw_cons,  pcons),
            _bigram_jaccard(raw_cons, pcons) if len(raw_cons) >= 2 else 0.0,
            _prefix_score(raw_clean, pname),
        )
        if score > best_score:
            best_score, best_province = score, pname

    if best_province and best_score >= 0.20:
        info = PROVINCE_MAP[best_province]
        return {"thai": best_province, "en": info["en"],
                "region": info["region"], "score": round(best_score, 3)}
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


def _preprocess_province_strip(strip_bgr: np.ndarray) -> np.ndarray:
    h, w   = strip_bgr.shape[:2]
    scale  = max(1.0, 60 / max(h, 1))
    up     = cv2.resize(strip_bgr, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq     = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(eq, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 6)
    return thresh if np.mean(thresh) < 127 else cv2.bitwise_not(thresh)


def _ocr_strip_tesseract(strip_bgr: np.ndarray) -> str:
    if not _TESSERACT_AVAILABLE:
        return ""
    binary = _preprocess_province_strip(strip_bgr)
    best   = ""
    for psm in _PROV_PSM_MODES:
        cfg = (f"--oem 1 --psm {psm} "
               f"-c tessedit_char_whitelist={THAI_CONSONANTS + THAI_VOWELS + ' '}")
        try:
            text = pytesseract.image_to_string(
                PILImage.fromarray(binary), lang="tha", config=cfg).strip()
            if len(text) > len(best):
                best = text
        except Exception:
            pass
    return best


def _ocr_strip_easyocr(reader, strip_bgr: np.ndarray) -> list[str]:
    results_out = []
    for scale in (1, 2, 3):
        img = strip_bgr if scale == 1 else cv2.resize(
            strip_bgr,
            (strip_bgr.shape[1] * scale, strip_bgr.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC)
        try:
            hits = reader.readtext(img, detail=1, paragraph=False,
                                   allowlist=THAI_CONSONANTS + THAI_VOWELS + " ")
            text = " ".join(
                r[1] for r in sorted(hits, key=lambda r: r[0][0][0])
                if r[2] > 0.03)
            if text:
                results_out.append(text)
        except Exception as exc:
            print(f"  [Province EasyOCR scale={scale}] {exc}")
    return results_out


def extract_province_from_crop(bgr_crop: np.ndarray,
                                easyocr_reader=None) -> Optional[dict]:
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    h, _ = bgr_crop.shape[:2]
    if h < 10:
        return None

    best_result, best_score = None, 0.0

    for y0_r, y1_r, label in _STRIP_CANDIDATES:
        y0, y1 = int(h * y0_r), int(h * y1_r)
        if y1 - y0 < 4:
            continue
        strip = bgr_crop[y0:y1, :]

        if easyocr_reader is not None:
            for raw in _ocr_strip_easyocr(easyocr_reader, strip):
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    if best_score >= 0.80:
                        return best_result

        raw = _ocr_strip_tesseract(strip)
        if raw:
            match = normalize_province(raw)
            if match and match["score"] > best_score:
                best_score  = match["score"]
                best_result = {**match, "strip": label}
                if best_score >= 0.80:
                    return best_result

    return best_result


def extract_province_from_ocr_tokens(ocr_results: list,
                                      min_len: int = 4) -> Optional[dict]:
    """[FIX-5] Scan OCR result tuples for province-like tokens."""
    best_result, best_score = None, 0.0
    for item in ocr_results:
        if len(item) < 3:
            continue
        text, conf = str(item[1]).strip(), float(item[2])
        if any(ch.isdigit() for ch in text):
            continue
        if len(text) < min_len or conf < 0.03:
            continue
        thai_chars = sum(1 for ch in text if '\u0E00' <= ch <= '\u0E7F')
        if thai_chars < min_len:
            continue
        match = normalize_province(text)
        if match and match["score"] > best_score:
            best_score  = match["score"]
            best_result = {**match, "strip": "ocr_token", "ocr_conf": conf}
    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTITUTION / CORRECTION  [ENH-8]
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

# [ENH-8] Extended duplication artefact table
_THAI_CORRECTIONS = {
    "รร": "ร",
    "าา": "า",
    "นน": "น",
    "งง": "ง",
    "กก": "ก",
    "ขข": "ข",
    "ดด": "ด",
    "ทท": "ท",
    "ปป": "ป",
    "บบ": "บ",
    "มม": "ม",
    "สส": "ส",
    "หห": "ห",
    "อออ": "อ",
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


def try_recover_all_digit_plate(text: str) -> Optional[str]:
    if not text or not text.isdigit():
        return None
    candidates = []
    for thai_len in range(1, 4):
        if len(text) != thai_len + 4:
            continue
        recovered, ok = "", True
        for ch in text[:thai_len]:
            if ch in _DIGIT_TO_THAI:
                recovered += _DIGIT_TO_THAI[ch]
            else:
                ok = False; break
        if ok:
            candidates.append(recovered + text[thai_len:])
    return candidates[-1] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_plate_text(text: str) -> str:
    return "".join(text.strip().split()).replace("-", "").upper()


def looks_like_thai_plate_top_line(text: str, min_length: int = 3) -> bool:
    if not text or len(text) < min_length:
        return False
    has_digit = any(ch.isdigit() or ch in THAI_DIGITS for ch in text)
    has_thai  = any('\u0E00' <= ch <= '\u0E7F' for ch in text)
    return has_digit and has_thai


def validate_plate_pattern(text: str) -> bool:
    """[FIX-1] Exactly 2 consonants then 4 digits."""
    return bool(_PLATE_PATTERN.match(text))


def adaptive_min_confidence(yolo_conf: float) -> float:
    yolo_conf = max(0.40, min(0.80, yolo_conf))
    return 0.20 - (yolo_conf - 0.40) / 0.40 * 0.15


def should_send_plate(plate_text: str, last_seen_plates: dict,
                      current_time: float, cooldown_time: float) -> bool:
    if plate_text not in last_seen_plates:
        return True
    return (current_time - last_seen_plates[plate_text]) > cooldown_time


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING  [ENH-1] [ENH-2]
# ═══════════════════════════════════════════════════════════════════════════

def _sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _score_binarized(img: np.ndarray) -> float:
    """[ENH-1] Score how plausible a binarised image is as a plate.

    Counts connected components whose area is in the range expected for a
    Thai plate character (~0.3%–25% of image area).  Ideal plate has 6–8
    such components.  Score peaks at 7 components and degrades linearly.
    """
    work = cv2.bitwise_not(img) if np.mean(img) > 127 else img
    n, _, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    if n <= 1:
        return 0.0
    total = img.shape[0] * img.shape[1]
    lo, hi = total * 0.003, total * 0.25
    valid  = sum(1 for i in range(1, n) if lo < stats[i, cv2.CC_STAT_AREA] < hi)
    return max(0.0, 1.0 - abs(valid - 7) / 10.0)


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab     = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _preprocess_at_scale(crop_bgr: np.ndarray, scale: int) -> np.ndarray:
    """[ENH-1] Picks the best of three binarisation methods per scale."""
    h, w = crop_bgr.shape[:2]
    up   = cv2.resize(crop_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq       = clahe.apply(denoised)
    blurred  = cv2.GaussianBlur(eq, (0, 0), sigmaX=3)
    sharp    = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)

    # Three binarisation candidates
    t_adapt_g = cv2.adaptiveThreshold(sharp, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 8)
    _, t_otsu  = cv2.threshold(sharp, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_adapt_m  = cv2.adaptiveThreshold(sharp, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 25, 6)

    # Pick best by component plausibility
    scored = [(_score_binarized(t), t)
              for t in (t_adapt_g, t_otsu, t_adapt_m)]
    thresh = max(scored, key=lambda x: x[0])[1]

    # Ensure dark-on-light (text is dark)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kern_open)
    return thresh


def _crop_number_zone(bgr_crop: np.ndarray) -> np.ndarray:
    """[ENH-2] Detect province-strip boundary via row-brightness dip."""
    h, w = bgr_crop.shape[:2]
    if h < 20:
        return bgr_crop
    gray     = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    row_mean = np.mean(gray, axis=1)
    # Scan bottom half for a darkening row (province strip top edge)
    scan_start  = int(h * 0.55)
    scan_zone   = row_mean[scan_start:]
    overall_mean = float(np.mean(row_mean))
    threshold    = overall_mean * 0.78
    drop_rows    = np.where(scan_zone < threshold)[0]
    if len(drop_rows) > 0:
        province_row = scan_start + int(drop_rows[0])
        cut_bottom   = max(int(h * 0.60),
                           min(province_row - 4, int(h * 0.85)))
    else:
        cut_bottom = int(h * 0.72)
    return bgr_crop[int(h * 0.10): cut_bottom, :]


def preprocess_plate_image(bgr_crop: np.ndarray, scale: int = 3,
                            number_zone_only: bool = True) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop passed to preprocess_plate_image")

    bgr_crop = _clahe_lab(bgr_crop)
    crop     = (_crop_number_zone(bgr_crop)
                if number_zone_only and bgr_crop.shape[0] > 30
                else bgr_crop)
    ch, cw   = crop.shape[:2]

    if max(ch, cw) >= 200:
        return _preprocess_at_scale(crop, 1)

    # Multi-scale: pick sharpest result
    results = []
    for s in (max(1, scale - 1), scale):
        proc = _preprocess_at_scale(crop, s)
        results.append((_sharpness_laplacian(proc), proc))
    return max(results, key=lambda x: x[0])[1]


# ═══════════════════════════════════════════════════════════════════════════
# DESKEW
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 30,
                            minLineLength=max(20, gray.shape[1] // 5),
                            maxLineGap=10)
    if lines is None:
        return 0.0
    angles = [math.degrees(math.atan2(y2 - y1, x2 - x1))
              for ln in lines
              for x1, y1, x2, y2 in [ln[0]]
              if x2 != x1 and -45 < math.degrees(math.atan2(y2 - y1, x2 - x1)) < 45]
    if not angles:
        return 0.0
    angles.sort()
    trimmed = angles[len(angles) // 4: -len(angles) // 4] or angles
    return float(np.median(trimmed))


def deskew_plate(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop
    gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.5:
        return bgr_crop
    h, w = bgr_crop.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(bgr_crop, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════════════
# OCR ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def run_easyocr(reader, preprocessed_gray: np.ndarray) -> list:
    return reader.readtext(
        preprocessed_gray, detail=1, paragraph=False,
        allowlist=(THAI_ALLOWLIST
                   + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    )


def run_easyocr_raw(reader, raw_bgr: np.ndarray) -> list:
    """[FIX-4][ENH-3] Run EasyOCR at 3 scales on the raw BGR crop.

    Results from all scales are merged; duplicates (same normalised key)
    keep the highest confidence.  Returns list of (bbox, text, conf).
    """
    if raw_bgr is None or raw_bgr.size == 0:
        return []
    h, w = raw_bgr.shape[:2]
    if h < 10:
        return []

    allowlist = (THAI_ALLOWLIST
                 + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    merged: dict[str, tuple] = {}

    for scale in (1, 2, 3):
        if scale == 1:
            img = raw_bgr
        else:
            new_h = max(30, min(120, int(h * scale)))
            new_w = max(1, int(w * new_h / h))
            img   = cv2.resize(raw_bgr, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
        img = _clahe_lab(img)
        try:
            hits = reader.readtext(img, detail=1, paragraph=False,
                                   allowlist=allowlist,
                                   decoder="beamsearch", beamWidth=10)
        except Exception:
            try:
                hits = reader.readtext(img, detail=1, paragraph=False,
                                       allowlist=allowlist)
            except Exception:
                continue

        for bbox, text, conf in hits:
            if conf < 0.05:
                continue
            key = normalize_plate_text(text)
            if not key:
                continue
            if key not in merged or conf > merged[key][2]:
                merged[key] = (bbox, text, float(conf))

    return list(merged.values())


def run_tesseract(preprocessed_gray: np.ndarray) -> list:
    if not _TESSERACT_AVAILABLE:
        return []
    try:
        h, w  = preprocessed_gray.shape[:2]
        small = (cv2.resize(preprocessed_gray, (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA)
                 if max(h, w) > 400 else preprocessed_gray)
        data  = pytesseract.image_to_data(
            PILImage.fromarray(small), lang="tha+eng",
            config=TESSERACT_NUM_CONFIG,
            output_type=pytesseract.Output.DICT)
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
# DUAL-ENGINE FUSION  [ENH-4]
# ═══════════════════════════════════════════════════════════════════════════

def fuse_ocr_results(easy_results: list, tess_results: list,
                     min_confidence: float = 0.05) -> list:
    """[ENH-4] Merge EasyOCR + Tesseract.

    When both engines agree on the same normalised key the confidence is
    boosted by +15% (was +10%), encouraging the voter to converge faster
    on genuine reads.
    """
    combined: dict[str, tuple] = {}   # norm_key → (bbox, text, conf, seen_count)

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
                combined[key] = (bbox, text, conf, 1)
            else:
                ob, ot, oc, cnt = combined[key]
                new_cnt = cnt + 1
                # Agreement boost: +15% per additional engine that agrees
                boosted = min(1.0, max(conf, oc) * (1.0 + 0.15 * (new_cnt - 1)))
                if conf >= oc:
                    combined[key] = (bbox, text, boosted, new_cnt)
                else:
                    combined[key] = (ob, ot, boosted, new_cnt)

    # Strip the internal count field before returning
    return [(b, t, c) for b, t, c, _ in combined.values()]


# ═══════════════════════════════════════════════════════════════════════════
# RESULT EXTRACTION  [FIX-6]
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    try:    return float(value)
    except: return default


def _process_text(text: str) -> str:
    text = correct_ocr_substitutions(text)
    norm = normalize_plate_text(text)
    if not any('\u0E00' <= ch <= '\u0E7F' for ch in norm):
        recovered = try_recover_all_digit_plate(norm)
        if recovered:
            return recovered
    return text


def extract_best_plate_read(
    ocr_results,
    min_length: int             = 3,
    min_confidence: float       = 0.05,
    max_correction_score: float = 0.65,
) -> Optional[Tuple[str, float]]:
    """[FIX-6] Full corrector pipeline.

    1. Collect and clean raw reads.
    2. Append a merged-fragment candidate.
    3. Run plate_corrector.correct_candidates() on all.
    4. Return the highest-confidence corrected result.
    5. Fallback: return best plate-like raw read for the voter to handle.
    """
    if not ocr_results:
        return None

    def top_y(r: tuple) -> float:
        try:    return min(float(pt[1]) for pt in r[0])
        except: return 0.0

    raw_pairs: list[Tuple[str, float]] = []
    for result in sorted(ocr_results, key=top_y):
        if len(result) < 3:
            continue
        text, conf = result[1], _safe_float(result[2])
        if not isinstance(text, str) or conf < min_confidence:
            continue
        raw_pairs.append((_process_text(text), conf))

    if not raw_pairs:
        return None

    # Merged fragment
    merged = "".join(t for t, _ in raw_pairs)
    if looks_like_thai_plate_top_line(merged, min_length):
        raw_pairs.append((merged, float(np.mean([c for _, c in raw_pairs]))))

    # Corrector pass
    corrected = correct_candidates(raw_pairs,
                                   max_correction_score=max_correction_score)
    if corrected:
        return max(corrected, key=lambda x: x[1])

    # Fallback: return best plate-like read for the voter
    plate_like = [
        (normalize_plate_text(t), c)
        for t, c in raw_pairs
        if looks_like_thai_plate_top_line(normalize_plate_text(t), min_length)
    ]
    return max(plate_like, key=lambda x: x[1]) if plate_like else None


# ═══════════════════════════════════════════════════════════════════════════
# IoU-BASED PLATE TRACKER  [ENH-6]
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
    """[ENH-6] IoU tracker with tighter threshold and state migration.

    When a new track is opened for a box that just missed the IoU threshold
    of an existing track (occlusion recovery), the old track's province
    cache is migrated to the new ID so no work is lost.
    """

    IOU_THRESHOLD        = 0.40   # [ENH-6] tightened from 0.35
    IOU_MIGRATE_THRESHOLD = 0.20  # below this we migrate province cache
    MAX_MISSED           = 8

    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}

    def _new_track(self, box: Tuple,
                   province_info=None, last_ocr_box=None) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box":           box,
            "missed":        0,
            "province_info": province_info,
            "last_ocr_box":  last_ocr_box,
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
                s = _iou(box, track["box"])
                if s > best_iou:
                    best_iou, best_tid = s, tid

            if best_iou >= self.IOU_THRESHOLD and best_tid is not None:
                self._tracks[best_tid]["box"]    = box
                self._tracks[best_tid]["missed"] = 0
                used_tids.add(best_tid)
                assignments.append((best_tid, box))
            else:
                # [ENH-6] Migrate province cache from nearest retiring track
                migrate_prov = None
                migrate_ocr  = None
                if best_tid is not None and best_iou >= self.IOU_MIGRATE_THRESHOLD:
                    migrate_prov = self._tracks[best_tid].get("province_info")
                    migrate_ocr  = self._tracks[best_tid].get("last_ocr_box")

                tid = self._new_track(box,
                                      province_info=migrate_prov,
                                      last_ocr_box=migrate_ocr)
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
# STABILITY GATE  [FIX-2]
# ═══════════════════════════════════════════════════════════════════════════

class PlateStabilityGate:
    """Hysteresis stability gate.

    [FIX-2] First frame seeds history AND counts as stable_count=1,
    so stable state is reached in exactly min_stable_frames low-drift frames.
    Exit requires HYSTERESIS_COUNT consecutive high-drift frames.
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

        if not self._history:
            self._stable_count = 1          # [FIX-2] first frame counts
        else:
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
# WEIGHTED MAJORITY VOTER  [FIX-3] [FIX-7] [ENH-5]
# ═══════════════════════════════════════════════════════════════════════════

class PlateMajorityVoter:
    """Weighted majority voter with recency bias.

    [FIX-3]  Accepts llm_corrector= / track_id= kwargs; exposes
             _consonant_candidates for the LLM corrector.
    [FIX-7]  Only buffers reads that look like plate fragments — noise
             no longer consumes vote slots.
    [ENH-5]  Recency bonus raised to 15%.  Window slides only when the
             buffer actually contains at least min_votes entries so stale
             half-filled windows cannot fire spurious results.
    """

    RECENCY_BONUS = 0.15    # [ENH-5] was 0.10

    def __init__(self, window: int = 8, min_votes: int = 3,
                 llm_corrector=None, track_id: int = -1):
        self.window    = window
        self.min_votes = min_votes
        self._llm      = llm_corrector
        self._tid      = track_id
        self._buffer:               List[Tuple[str, float, float]] = []
        self._consonant_candidates: list = []   # [FIX-3]

    def _update_consonant_candidates(self, text: str, conf: float):
        if conf < 0.15:
            return
        norm = normalize_plate_text(text)
        # Extract leading consonant run (1-3 chars)
        cons = ""
        for ch in norm:
            if ch in _CONSONANT_SET:
                cons += ch
            else:
                break
        if 1 <= len(cons) <= 3 and cons not in self._consonant_candidates:
            self._consonant_candidates.append(cons)

    def update(self, candidate, yolo_conf: float = 1.0):
        if candidate is not None:
            text, ocr_conf = candidate
            self._update_consonant_candidates(text, float(ocr_conf))

            # [FIX-7] Only buffer plate-like reads
            norm = normalize_plate_text(text)
            if looks_like_thai_plate_top_line(norm, min_length=3):
                self._buffer.append((text, float(ocr_conf), float(yolo_conf)))

        # [ENH-5] Only slide window if buffer has enough entries
        if len(self._buffer) < self.window:
            return None

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
        self._consonant_candidates.clear()