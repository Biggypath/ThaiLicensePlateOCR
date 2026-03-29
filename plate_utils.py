"""
plate_utils.py — Thai licence-plate helper utilities (v6)

Key fixes vs v5:
  ─────────────────────────────────────────────────────────────────────────
  PROBLEM 1 — Province strip was cropped from the TOP of the YOLO box.
              Thai plates actually have:
                  top  ~15 %  → province / city  (small text)
                  mid  ~60 %  → plate number     (big text)
                  bot  ~25 %  → city again / blank
              BUT the YOLO model crops tightly around the NUMBER area,
              so the province strip is at the BOTTOM of the crop, not top.
              Fix: try BOTH top and bottom strips; pick whichever gives a
              valid province match.

  PROBLEM 2 — Province OCR used only Tesseract.  Tesseract struggles with
              blurry low-res Thai text.  Fix: use EasyOCR as primary for
              the province strip (it handles Thai small text much better),
              Tesseract as fallback.

  PROBLEM 3 — Number-line OCR got confused by the province text row being
              included in the main preprocessed image.  Fix: preprocess
              only the middle 65% of the crop for number OCR so the small
              province text is excluded.

  PROBLEM 4 — try_recover_all_digit_plate only tried 2 or 3 leading chars.
              Fix: try ALL valid split points (1, 2, 3 leading chars) and
              return the longest valid match.
  ─────────────────────────────────────────────────────────────────────────

Observed EasyOCR misreads for plate ฐบ 3699 (from live logs):
  '313699'  → ฐ=3, บ=1  (already handled)
  '669833'  → garbage from upside-down number zone
  '3บ3699'  → ฐ=3 still digit, บ kept as Thai (acceptable)
  'รบ3699'  → ร instead of ฐ from 513699 (acceptable — voter corrects)
  'ต669835' → ต leaking from province strip → fixed by number_zone_only crop

Province text in images: กรงเทพมหานคร (noisy for กรุงเทพมหานคร)
  Fuzzy match now also runs consonants-only comparison to handle lost vowels.
"""

import re
import cv2
import numpy as np
from collections import Counter
from typing import Optional


try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thai character sets
# ---------------------------------------------------------------------------
THAI_CONSONANTS  = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
THAI_VOWELS      = "าิีึืุูเแโใไๆ็่้๊๋์ํ๎ะัำ"
THAI_DIGITS      = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS    = "0123456789"
THAI_ALLOWLIST   = THAI_CONSONANTS + THAI_DIGITS + ARABIC_DIGITS

TESSERACT_NUM_CONFIG = (
    r"--oem 1 --psm 7 "
    r"-c tessedit_char_whitelist=" + THAI_ALLOWLIST
)

TESSERACT_PROV_CONFIG = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist="
    + THAI_CONSONANTS + THAI_VOWELS + " "
)


# ---------------------------------------------------------------------------
# 77 Thai Provinces
# ---------------------------------------------------------------------------
PROVINCE_MAP: dict[str, dict] = {
    # ── Central ────────────────────────────────────────────────────────────
    "กรุงเทพมหานคร":    {"en": "Bangkok",                    "region": "Central"},
    "กำแพงเพชร":        {"en": "Kamphaeng Phet",             "region": "Central"},
    "ชัยนาท":           {"en": "Chai Nat",                   "region": "Central"},
    "นครนายก":          {"en": "Nakhon Nayok",               "region": "Central"},
    "นครปฐม":           {"en": "Nakhon Pathom",              "region": "Central"},
    "นครสวรรค์":        {"en": "Nakhon Sawan",               "region": "Central"},
    "นนทบุรี":          {"en": "Nonthaburi",                 "region": "Central"},
    "ปทุมธานี":         {"en": "Pathum Thani",               "region": "Central"},
    "พระนครศรีอยุธยา":  {"en": "Phra Nakhon Si Ayutthaya",  "region": "Central"},
    "พิจิตร":           {"en": "Phichit",                    "region": "Central"},
    "พิษณุโลก":         {"en": "Phitsanulok",                "region": "Central"},
    "เพชรบูรณ์":        {"en": "Phetchabun",                 "region": "Central"},
    "ลพบุรี":           {"en": "Lopburi",                    "region": "Central"},
    "สมุทรปราการ":      {"en": "Samut Prakan",               "region": "Central"},
    "สมุทรสงคราม":      {"en": "Samut Songkhram",            "region": "Central"},
    "สมุทรสาคร":        {"en": "Samut Sakhon",               "region": "Central"},
    "สระบุรี":          {"en": "Saraburi",                   "region": "Central"},
    "สิงห์บุรี":        {"en": "Sing Buri",                  "region": "Central"},
    "สุพรรณบุรี":       {"en": "Suphan Buri",                "region": "Central"},
    "อ่างทอง":          {"en": "Ang Thong",                  "region": "Central"},
    "อุทัยธานี":        {"en": "Uthai Thani",                "region": "Central"},
    # ── North ──────────────────────────────────────────────────────────────
    "เชียงราย":         {"en": "Chiang Rai",                 "region": "North"},
    "เชียงใหม่":        {"en": "Chiang Mai",                 "region": "North"},
    "ตาก":              {"en": "Tak",                        "region": "North"},
    "น่าน":             {"en": "Nan",                        "region": "North"},
    "พะเยา":            {"en": "Phayao",                     "region": "North"},
    "แพร่":             {"en": "Phrae",                      "region": "North"},
    "แม่ฮ่องสอน":       {"en": "Mae Hong Son",               "region": "North"},
    "ลำปาง":            {"en": "Lampang",                    "region": "North"},
    "ลำพูน":            {"en": "Lamphun",                    "region": "North"},
    "สุโขทัย":          {"en": "Sukhothai",                  "region": "North"},
    "อุตรดิตถ์":        {"en": "Uttaradit",                  "region": "North"},
    # ── Northeast (Isan) ───────────────────────────────────────────────────
    "กาฬสินธุ์":        {"en": "Kalasin",                    "region": "Northeast"},
    "ขอนแก่น":          {"en": "Khon Kaen",                  "region": "Northeast"},
    "ชัยภูมิ":          {"en": "Chaiyaphum",                 "region": "Northeast"},
    "นครพนม":           {"en": "Nakhon Phanom",              "region": "Northeast"},
    "นครราชสีมา":       {"en": "Nakhon Ratchasima",          "region": "Northeast"},
    "บึงกาฬ":           {"en": "Bueng Kan",                  "region": "Northeast"},
    "บุรีรัมย์":        {"en": "Buriram",                    "region": "Northeast"},
    "มหาสารคาม":        {"en": "Maha Sarakham",              "region": "Northeast"},
    "มุกดาหาร":         {"en": "Mukdahan",                   "region": "Northeast"},
    "ยโสธร":            {"en": "Yasothon",                   "region": "Northeast"},
    "ร้อยเอ็ด":         {"en": "Roi Et",                     "region": "Northeast"},
    "เลย":              {"en": "Loei",                       "region": "Northeast"},
    "ศรีสะเกษ":         {"en": "Si Sa Ket",                  "region": "Northeast"},
    "สกลนคร":           {"en": "Sakon Nakhon",               "region": "Northeast"},
    "สุรินทร์":         {"en": "Surin",                      "region": "Northeast"},
    "หนองคาย":          {"en": "Nong Khai",                  "region": "Northeast"},
    "หนองบัวลำภู":      {"en": "Nong Bua Lamphu",            "region": "Northeast"},
    "อำนาจเจริญ":       {"en": "Amnat Charoen",              "region": "Northeast"},
    "อุดรธานี":         {"en": "Udon Thani",                 "region": "Northeast"},
    "อุบลราชธานี":      {"en": "Ubon Ratchathani",           "region": "Northeast"},
    # ── East ───────────────────────────────────────────────────────────────
    "จันทบุรี":         {"en": "Chanthaburi",                "region": "East"},
    "ฉะเชิงเทรา":       {"en": "Chachoengsao",               "region": "East"},
    "ชลบุรี":           {"en": "Chon Buri",                  "region": "East"},
    "ตราด":             {"en": "Trat",                       "region": "East"},
    "ปราจีนบุรี":       {"en": "Prachin Buri",               "region": "East"},
    "ระยอง":            {"en": "Rayong",                     "region": "East"},
    "สระแก้ว":          {"en": "Sa Kaeo",                    "region": "East"},
    # ── West ───────────────────────────────────────────────────────────────
    "กาญจนบุรี":        {"en": "Kanchanaburi",               "region": "West"},
    "ประจวบคีรีขันธ์":  {"en": "Prachuap Khiri Khan",        "region": "West"},
    "เพชรบุรี":         {"en": "Phetchaburi",                "region": "West"},
    "ราชบุรี":          {"en": "Ratchaburi",                 "region": "West"},
    # ── South ──────────────────────────────────────────────────────────────
    "กระบี่":           {"en": "Krabi",                      "region": "South"},
    "ชุมพร":            {"en": "Chumphon",                   "region": "South"},
    "ตรัง":             {"en": "Trang",                      "region": "South"},
    "นครศรีธรรมราช":    {"en": "Nakhon Si Thammarat",        "region": "South"},
    "นราธิวาส":         {"en": "Narathiwat",                 "region": "South"},
    "ปัตตานี":          {"en": "Pattani",                    "region": "South"},
    "พัทลุง":           {"en": "Phatthalung",                "region": "South"},
    "พังงา":            {"en": "Phang Nga",                  "region": "South"},
    "ภูเก็ต":           {"en": "Phuket",                     "region": "South"},
    "ระนอง":            {"en": "Ranong",                     "region": "South"},
    "สงขลา":            {"en": "Songkhla",                   "region": "South"},
    "สตูล":             {"en": "Satun",                      "region": "South"},
    "สุราษฎร์ธานี":     {"en": "Surat Thani",                "region": "South"},
    "ยะลา":             {"en": "Yala",                       "region": "South"},
}

_ALL_PROVINCE_NAMES: list[str] = list(PROVINCE_MAP.keys())


def _consonants_only(text: str) -> str:
    """Keep only Thai consonants + spaces — drops vowel/tone marks."""
    return "".join(ch for ch in text if ch in THAI_CONSONANTS or ch == " ")


_PROVINCE_CONSONANTS: dict[str, str] = {
    name: _consonants_only(name) for name in _ALL_PROVINCE_NAMES
}


# ---------------------------------------------------------------------------
# Province fuzzy matching
# ---------------------------------------------------------------------------

def _char_jaccard(a: str, b: str) -> float:
    ca, cb = Counter(a), Counter(b)
    inter  = sum((ca & cb).values())
    union  = sum((ca | cb).values())
    return inter / union if union else 0.0


def normalize_province(raw: str) -> Optional[dict]:
    """
    Match raw OCR text to a canonical province via three passes:
      1. Exact string match.
      2. Full-character Jaccard (handles extra/swapped chars).
      3. Consonants-only Jaccard (handles lost vowel marks like ุ ิ ั).
    Returns dict with thai/en/region/score, or None if below threshold.
    """
    if not raw:
        return None

    raw_clean = raw.strip()
    raw_cons  = _consonants_only(raw_clean)

    # Pass 1 — exact
    if raw_clean in PROVINCE_MAP:
        info = PROVINCE_MAP[raw_clean]
        return {"thai": raw_clean, "en": info["en"],
                "region": info["region"], "score": 1.0}

    best_score    = 0.0
    best_province = None

    for pname in _ALL_PROVINCE_NAMES:
        s_full = _char_jaccard(raw_clean, pname)
        s_cons = _char_jaccard(raw_cons, _PROVINCE_CONSONANTS[pname])
        score  = max(s_full, s_cons)
        if score > best_score:
            best_score    = score
            best_province = pname

    MIN_SCORE = 0.30   # lower threshold: blurry crops lose many vowel marks
    if best_province and best_score >= MIN_SCORE:
        info = PROVINCE_MAP[best_province]
        return {
            "thai":   best_province,
            "en":     info["en"],
            "region": info["region"],
            "score":  round(best_score, 3),
        }
    return None


# ---------------------------------------------------------------------------
# Province strip extraction
# ---------------------------------------------------------------------------

# Vertical slices to try when hunting for province text.
# (y_start_ratio, y_end_ratio, label)
# Province text is usually at the BOTTOM of a tight YOLO crop,
# but try top too for cameras that capture the full plate.
_STRIP_CANDIDATES = [
    (0.72, 1.00, "bottom"),
    (0.00, 0.22, "top"),
    (0.68, 1.00, "bottom_wide"),
]


def _preprocess_province_strip(strip_bgr: np.ndarray) -> np.ndarray:
    """Upscale + adaptive threshold optimised for small Thai province text."""
    h, w   = strip_bgr.shape[:2]
    scale  = max(1, int(np.ceil(80 / max(h, 1))))
    up     = cv2.resize(strip_bgr, (w * scale, h * scale),
                        interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray   = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    return binary


def _ocr_strip_tesseract(binary: np.ndarray) -> str:
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        pil_img = PILImage.fromarray(binary)
        return pytesseract.image_to_string(
            pil_img, lang="tha", config=TESSERACT_PROV_CONFIG,
        ).strip()
    except Exception as exc:
        print(f"  [Province Tesseract error] {exc}")
        return ""


def _ocr_strip_easyocr(reader, strip_bgr: np.ndarray) -> str:
    try:
        results = reader.readtext(
            strip_bgr,
            detail    = 1,
            paragraph = False,
            allowlist = THAI_CONSONANTS + THAI_VOWELS + " ",
        )
        results_sorted = sorted(results, key=lambda r: r[0][0][0])
        return " ".join(r[1] for r in results_sorted if r[2] > 0.05)
    except Exception as exc:
        print(f"  [Province EasyOCR error] {exc}")
        return ""


def extract_province_from_crop(
    bgr_crop: np.ndarray,
    easyocr_reader=None,
) -> Optional[dict]:
    """
    Try multiple horizontal strips of the plate crop to find province text.
    Uses EasyOCR (primary) + Tesseract (fallback) on each strip.

    Parameters
    ----------
    bgr_crop       : full plate crop from YOLO (BGR)
    easyocr_reader : initialised easyocr.Reader (strongly recommended)

    Returns dict {"thai", "en", "region", "score", "strip"} or None.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = bgr_crop.shape[:2]
    if h < 10:
        return None

    best_result = None
    best_score  = 0.0

    for y0_r, y1_r, label in _STRIP_CANDIDATES:
        y0 = int(h * y0_r)
        y1 = int(h * y1_r)
        if y1 - y0 < 4:
            continue
        strip = bgr_crop[y0:y1, :]

        # EasyOCR — primary (works on colour strip directly)
        if easyocr_reader is not None:
            raw = _ocr_strip_easyocr(easyocr_reader, strip)
            if raw:
                match = normalize_province(raw)
                if match and match["score"] > best_score:
                    best_score  = match["score"]
                    best_result = {**match, "strip": label}
                    print(f"  [Province strip={label}] EasyOCR '{raw}' "
                          f"→ {match['thai']} ({match['score']:.2f})")

        # Tesseract — fallback (needs pre-processed binary)
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
            break   # confident enough — stop early

    return best_result


# ---------------------------------------------------------------------------
# Substitution maps
# ---------------------------------------------------------------------------
_LATIN_TO_THAI = [
    ("au", "ม"), ("nn", "ท"), ("iiu", "ปน"), ("ii", "ป"), ("ss", "ส"),
    ("u",  "บ"), ("i",  "บ"), ("n",  "น"),   ("m",  "ม"), ("g",  "ง"),
    ("a",  "า"), ("s",  "ส"), ("w",  "ว"),   ("h",  "ห"), ("o",  "อ"),
    ("x",  "ข"), ("y",  "ย"),
]

# Expanded digit→Thai map based on observed live misreads
_DIGIT_TO_THAI = {
    "3": "ฐ",   # ฐ looks like 3
    "1": "บ",   # บ looks like 1
    "0": "อ",   # อ looks like 0
    "5": "ร",   # ร looks like 5
    "8": "ม",   # ม looks like 8
    "9": "ถ",   # ถ looks like 9 (rare)
    "7": "ง",   # ง looks like 7 (rare)
    "6": "ถ",   # ถ can look like 6
    "4": "ข",   # ข looks like 4 (rare)
    "2": "ฉ",   # ฉ looks like 2 (very rare)
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
            result.append(seg)
    return "".join(result)


def try_recover_all_digit_plate(text: str) -> Optional[str]:
    """
    If text is all-digits, try all valid Thai-consonant + 4-digit splits.
    Thai plate top-line: [1-3 Thai consonants][4 Arabic digits]
    Returns the longest valid recovery, or None.
    """
    if not text or not text.isdigit():
        return None

    candidates = []
    for thai_len in range(1, 4):
        if len(text) != thai_len + 4:
            continue
        thai_part  = text[:thai_len]
        digit_part = text[thai_len:]
        recovered  = ""
        ok = True
        for ch in thai_part:
            if ch in _DIGIT_TO_THAI:
                recovered += _DIGIT_TO_THAI[ch]
            else:
                ok = False
                break
        if ok:
            candidates.append(recovered + digit_part)

    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_plate_text(text: str) -> str:
    compact = "".join(text.strip().split())
    compact = compact.replace("-", "")
    return compact.upper()


def looks_like_thai_plate_top_line(text: str, min_length: int = 3) -> bool:
    if not text or len(text) < min_length:
        return False
    has_digit = any(ch.isdigit() or ch in THAI_DIGITS for ch in text)
    has_thai  = any('\u0E00' <= ch <= '\u0E7F' for ch in text)
    return has_digit and has_thai


def should_send_plate(plate_text, last_seen_plates, current_time, cooldown_time):
    if plate_text not in last_seen_plates:
        return True
    return (current_time - last_seen_plates[plate_text]) > cooldown_time


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def preprocess_plate_image(
    bgr_crop: np.ndarray,
    scale: int = 3,
    number_zone_only: bool = True,
) -> np.ndarray:
    """
    Pre-process a plate crop for number OCR.
    number_zone_only=True: slice middle 15-80% vertically to exclude the
    province text strip at the bottom (prevents 'ต', 'ว' bleed-in).
    """
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop passed to preprocess_plate_image")

    h, w = bgr_crop.shape[:2]
    if number_zone_only and h > 30:
        crop = bgr_crop[int(h * 0.15): int(h * 0.80), :]
    else:
        crop = bgr_crop

    ch, cw = crop.shape[:2]
    if max(ch, cw) < 200:
        upscaled = cv2.resize(crop, (cw * scale, ch * scale),
                              interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = crop.copy()

    gray      = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised  = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)
    blurred   = cv2.GaussianBlur(equalized, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(equalized, 1.5, blurred, -0.5, 0)
    return sharpened


def deskew_plate(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    def horizontal_score(img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=20, maxLineGap=10)
        if lines is None:
            return 0
        angles = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if x2 != x1:
                angles.append(abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if not angles:
            return 0
        return sum(1 for a in angles if a < 15 or a > 165) / len(angles)

    rotations = [
        bgr_crop,
        cv2.rotate(bgr_crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(bgr_crop, cv2.ROTATE_180),
        cv2.rotate(bgr_crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    scores = [horizontal_score(r) for r in rotations]
    return rotations[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def run_easyocr(reader, preprocessed_gray: np.ndarray) -> list:
    return reader.readtext(
        preprocessed_gray,
        detail    = 1,
        paragraph = False,
        allowlist = (THAI_ALLOWLIST
                     + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    )


def run_tesseract(preprocessed_gray: np.ndarray) -> list:
    if not _TESSERACT_AVAILABLE:
        return []
    try:
        pil_img = PILImage.fromarray(preprocessed_gray)
        data = pytesseract.image_to_data(
            pil_img,
            lang        = "tha+eng",
            config      = TESSERACT_NUM_CONFIG,
            output_type = pytesseract.Output.DICT,
        )
        h, w       = preprocessed_gray.shape[:2]
        dummy_bbox = [[0, 0], [w, 0], [w, h], [0, h]]
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


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _safe_float(value, default=0.0):
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
            text = recovered
    return text


def extract_best_plate_read(
    ocr_results,
    min_length: int = 3,
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

    # Strategy 1: best single box
    single_candidates = []
    for bbox, text, conf in corrected:
        if conf < min_confidence:
            continue
        plate_text = normalize_plate_text(text)
        if looks_like_thai_plate_top_line(plate_text, min_length=min_length):
            single_candidates.append((plate_text, conf))

    if single_candidates:
        return max(single_candidates, key=lambda x: x[1])

    # Strategy 2: merge fragments
    def is_fragment(t):
        return (any('\u0E00' <= ch <= '\u0E7F' for ch in t)
                or any(ch.isdigit() or ch in THAI_DIGITS for ch in t))

    fragments = []
    for bbox, text, conf in corrected:
        if conf < min_confidence:
            continue
        norm = normalize_plate_text(text)
        if is_fragment(norm):
            fragments.append((norm, conf))

    if not fragments:
        return None

    merged_text = "".join(t for t, _ in fragments)
    merged_conf = float(np.mean([c for _, c in fragments]))
    if looks_like_thai_plate_top_line(merged_text, min_length=min_length):
        return (merged_text, merged_conf)

    return None


# ---------------------------------------------------------------------------
# Stability gate
# ---------------------------------------------------------------------------

class PlateStabilityGate:
    """
    Holds back OCR until the bounding-box centre has been steady for
    `min_stable_frames` consecutive frames (drift ≤ `max_pixel_drift` px).
    Call reset() after each successful plate publish.
    """

    def __init__(self, min_stable_frames: int = 4, max_pixel_drift: float = 6.0):
        self.min_stable_frames = min_stable_frames
        self.max_pixel_drift   = max_pixel_drift
        self._history: list[tuple[float, float]] = []
        self._stable_count: int = 0

    def is_stable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if self._history:
            px, py = self._history[-1]
            drift  = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            self._stable_count = (
                self._stable_count + 1 if drift <= self.max_pixel_drift else 0
            )
        self._history.append((cx, cy))
        if len(self._history) > self.min_stable_frames + 1:
            self._history.pop(0)
        return self._stable_count >= self.min_stable_frames

    def reset(self):
        self._history.clear()
        self._stable_count = 0


# ---------------------------------------------------------------------------
# Multi-frame majority voter
# ---------------------------------------------------------------------------

class PlateMajorityVoter:
    def __init__(self, window: int = 8, min_votes: int = 3):
        self.window    = window
        self.min_votes = min_votes
        self._buffer   = []

    def update(self, candidate):
        if candidate is not None:
            self._buffer.append(candidate)
        if len(self._buffer) < self.window:
            return None
        texts            = [t for t, _ in self._buffer]
        best_text, count = Counter(texts).most_common(1)[0]
        confs            = [c for t, c in self._buffer if t == best_text]
        avg_conf         = float(np.mean(confs))
        self._buffer.pop(0)
        return (best_text, avg_conf) if count >= self.min_votes else None

    def reset(self):
        self._buffer.clear()