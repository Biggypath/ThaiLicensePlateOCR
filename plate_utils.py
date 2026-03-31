"""
plate_utils.py — Thai licence-plate helper utilities  (v9 — MAX ACCURACY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY CHANGES vs v8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREPROCESSING (highest impact):
  [CV-6]  MULTI-STRATEGY BINARIZATION — three strategies run in parallel:
            A) Adaptive Gaussian with LARGER 31px kernel (was 11px)
               Critical fix: 11px kernel destroys Thai consonant strokes
            B) Otsu on CLAHE-equalised gray — works well with good contrast
            C) Adaptive Mean with 25px kernel — gentler on thin strokes
          Best strategy selected by connected-component plausibility score
          targeting 6–8 large blobs (2 consonants + 4 digits + diacritics).

  [CV-7]  LARGER MORPHOLOGICAL CLOSING — 3×3 kernel (was 2×2).
          Thai consonants like ฐ and บ have many thin joining strokes;
          a 2×2 closing kernel cannot bridge gaps caused by binarization.
          3×3 reliably reconnects broken strokes.

  [CV-8]  TIGHTER NUMBER-ZONE CROP — cuts province strip more aggressively.
          Top 10%–72% of height (was 15%–80%). Province text at bottom
          bleeds into the number zone and produces garbage OCR tokens.

  [CV-9]  SMART PROVINCE-LINE DETECTION — scans row means in the bottom
          40% of the crop to locate the province strip boundary and cuts
          precisely above it, adapting to variable plate aspect ratios.

OCR (new parallel path):
  [OCR-5] RAW-COLOR PARALLEL OCR — EasyOCR is run on the raw BGR crop
          (no binarization) at a normalised 80px height in addition to
          the preprocessed path. Thai plates have natural high contrast;
          the raw image often preserves consonant detail that binarization
          destroys. Both paths are fused before correction.

  [OCR-6] BEAMSEARCH DECODER for raw path — beamWidth=10 captures more
          Thai consonant hypotheses at low confidence.

  [OCR-7] MIN_CONFIDENCE floor lowered to 0.05 for raw path results;
          the corrector and voter gates downstream handle quality.

Voter:
  [PL-4]  PARTIAL-EVIDENCE ACCUMULATION in PlateMajorityVoter.
          Even when correct_plate() returns None (consonants destroyed),
          the voter now accumulates:
            - digit_suffix_counter: 4-digit tails seen per frame
            - consonant_candidates: any consonant pairs seen at conf > 0.15
          After MIN_DIGIT_VOTES frames with the same digit suffix, the
          voter attempts reconstruction via merge_digit_evidence().
          This recovers plates like ฐบ3699 from runs of "3699" + "ฐบ..." .

Corrections carried forward from v8:
  [PP-1]  Domain-enforced format correction (plate_corrector.py 5-pass).
  [PP-2]  Correction-score confidence penalty.
  [PP-3]  Valid-plate gating on voter output.
  [OCR-1] Dual-engine fusion with agreement boost.
  [OCR-2] Adaptive confidence threshold (YOLO conf → OCR threshold).
  [CV-1]  Multi-scale preprocessing (sharpest scale wins).
  [CV-2]  Morphological cleanup (now upgraded to [CV-7]).
  [CV-3]  Contrast-aware polarity inversion.
  [CV-4]  LAB-CLAHE illumination normalisation.
  [CV-5]  Sub-degree affine deskew via Hough median angle.
  [PL-1]  IoU-based per-track voter.
  [PL-2]  Weighted majority vote (YOLO × OCR × correction penalty).
  [PL-3]  Stability gate with exit-hysteresis.
  [PF-1]  Skip OCR when box barely moved.
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

    if best_province and best_score >= 0.30:
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
# IMAGE PRE-PROCESSING  [CV-1, CV-4, CV-6, CV-7, CV-8, CV-9]
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
    # Ensure dark-on-light polarity for CC analysis
    work = cv2.bitwise_not(img) if np.mean(img) > 127 else img.copy()
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    if num_labels <= 1:
        return 0.0
    h_img, w_img = img.shape
    total_area   = h_img * w_img
    area_min     = total_area * 0.003   # at least 0.3% of image
    area_max     = total_area * 0.25    # at most 25%
    valid = sum(
        1 for i in range(1, num_labels)
        if area_min < stats[i, cv2.CC_STAT_AREA] < area_max
    )
    # Ideal target: 7 components (ฐ=1, บ=1 or 2, 3=1, 6=1, 9=1, 9=1)
    score = 1.0 - abs(valid - 7) / 10.0
    return max(0.0, score)


def _preprocess_at_scale(
    crop_bgr: np.ndarray,
    scale: int,
) -> np.ndarray:
    """
    [CV-6] Multi-strategy binarization — three parallel strategies,
    winner selected by connected-component plausibility score.
    [CV-7] Larger 3×3 morphological closing kernel for Thai strokes.
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
    # 31px kernel preserves complex stroke topology destroyed by 11px kernel
    thresh_a = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8,
    )

    # [CV-6] Strategy B: Otsu — works well when plate has clean contrast
    _, thresh_b = cv2.threshold(
        sharp, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # [CV-6] Strategy C: adaptive mean — gentler on thin strokes than Gaussian
    thresh_c = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 25, 6,
    )

    # Select strategy with highest CC plausibility score
    scored = [(
        _score_binarized(t), t
    ) for t in (thresh_a, thresh_b, thresh_c)]
    _, thresh = max(scored, key=lambda x: x[0])

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
    [CV-9] Smart province-line detection.
    Scans row means in the bottom 40% to find the province strip boundary
    and cuts precisely above it. Falls back to fixed 10%–72% if detection
    is ambiguous.

    Thai plate anatomy (approximate):
      Top border:      ~8–10% of height
      Number line:     ~10–72% of height
      Province strip:  ~72–90% of height
      Bottom border:   ~90–100% of height
    """
    h, w = bgr_crop.shape[:2]
    if h < 20:
        return bgr_crop

    gray     = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    row_mean = np.mean(gray, axis=1)   # mean brightness per row

    # Province strip is darker than the white number zone
    # Scan the bottom 40% for a significant brightness drop
    scan_start = int(h * 0.55)
    scan_zone  = row_mean[scan_start:]
    overall_mean = float(np.mean(row_mean))

    # Find first row (from scan_start down) where mean drops below threshold
    threshold = overall_mean * 0.78
    drop_rows = np.where(scan_zone < threshold)[0]

    if len(drop_rows) > 0:
        province_row = scan_start + int(drop_rows[0])
        # Add 4px buffer above the province strip
        cut_bottom = max(int(h * 0.60), min(province_row - 4, int(h * 0.85)))
    else:
        # No clear drop found — use fixed ratio
        cut_bottom = int(h * 0.72)

    cut_top = int(h * 0.10)
    return bgr_crop[cut_top: cut_bottom, :]


def preprocess_plate_image(
    bgr_crop: np.ndarray,
    scale: int = 3,
    number_zone_only: bool = True,
) -> np.ndarray:
    """
    [CV-1] Multi-scale preprocessing: run at two scales, return sharper.
    [CV-4] LAB-CLAHE illumination normalisation applied first.
    [CV-8] Tighter crop: uses crop_number_zone() for smart province cutoff.
    [CV-6] Multi-strategy binarization inside _preprocess_at_scale().
    """
    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("Empty crop passed to preprocess_plate_image")

    bgr_crop = _clahe_lab(bgr_crop)  # [CV-4]

    h, w = bgr_crop.shape[:2]
    if number_zone_only and h > 30:
        crop = crop_number_zone(bgr_crop)   # [CV-8, CV-9]
    else:
        crop = bgr_crop

    ch, cw = crop.shape[:2]
    if max(ch, cw) >= 200:
        # Already large — single scale, no upscaling
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
    Thai plates have natural high contrast; the raw image often preserves
    consonant stroke detail that aggressive binarization destroys.
    Uses beamsearch decoder for better Thai consonant recall.
    [OCR-6] beamWidth=10 for wider hypothesis search.
    """
    if raw_bgr is None or raw_bgr.size == 0:
        return []

    # Normalise height to 80px — EasyOCR's sweet spot for Thai characters
    h, w = raw_bgr.shape[:2]
    if h < 30:
        return []
    if h < 60 or h > 110:
        target_h  = 80
        new_scale = target_h / max(h, 1)
        raw_bgr   = cv2.resize(raw_bgr,
                               (int(w * new_scale), target_h),
                               interpolation=cv2.INTER_CUBIC)

    # Apply light CLAHE to boost contrast before raw OCR
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
        # [OCR-7] Lower the confidence floor for raw results — downstream
        # corrector and voter gate quality; we want to capture weak consonants
        return [(bbox, text, conf) for bbox, text, conf in results
                if conf >= 0.05]
    except Exception as exc:
        print(f"  [EasyOCR raw error] {exc}")
        # Fallback: try without beamsearch (some EasyOCR versions differ)
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
    min_confidence lowered to 0.05 to capture weak consonant reads
    from the raw path; downstream corrector handles quality.
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
                # Agreement boost: same text from multiple sources
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

    min_confidence lowered to 0.05 to capture weak raw-path reads.
    Returns (plate_text, adjusted_conf) or None.
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

    # Also try merging all fragments into one candidate
    merged      = "".join(t for t, _ in raw_candidates)
    merged_conf = float(np.mean([c for _, c in raw_candidates]))
    raw_candidates.append((merged, merged_conf))

    # [PP-1, PP-2] Apply domain corrector to every candidate
    corrected_candidates = correct_candidates(
        raw_candidates,
        max_correction_score=max_correction_score,
    )

    if not corrected_candidates:
        return None

    return max(corrected_candidates, key=lambda x: x[1])


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
# WEIGHTED MAJORITY VOTER  [PL-2, PL-4]
# ═══════════════════════════════════════════════════════════════════════════

class PlateMajorityVoter:
    """
    [PL-2] Weighted vote: yolo_conf × adjusted_ocr_conf, recency bonus.
    [PP-3] Winner is re-validated via is_valid_plate() before return.
    [PL-4] Partial-evidence accumulation:
           Accumulates digit suffixes and consonant candidates from frames
           where correct_plate() returned None. After MIN_DIGIT_VOTES
           frames with the same digit suffix, attempts reconstruction via
           merge_digit_evidence(). This is the key recovery path for
           cases where binarization destroys consonants but digits survive.
    """

    RECENCY_BONUS   = 0.10
    MIN_DIGIT_VOTES = 5     # frames with same digit suffix before reconstruction

    def __init__(self, window: int = 8, min_votes: int = 3):
        self.window    = window
        self.min_votes = min_votes
        self._buffer: List[Tuple[str, float, float]] = []
        # [PL-4] Partial evidence stores
        self._digit_suffix_counter: Counter = Counter()
        self._consonant_candidates: list    = []

    def update(
        self,
        candidate,
        yolo_conf: float = 1.0,
    ):
        if candidate is not None:
            text, ocr_conf = candidate

            # [PL-4] Always try to extract digit evidence
            digit_suffix = extract_digit_suffix(text)
            if digit_suffix:
                self._digit_suffix_counter[digit_suffix] += 1

            # [PL-4] Extract consonant portion (2-3 chars) if present
            norm           = normalize_plate_text(text)
            consonant_part = "".join(
                c for c in norm if c in _CORRECTOR_CONSONANTS
            )
            if len(consonant_part) in (2, 3) and float(ocr_conf) > 0.15:
                if consonant_part not in self._consonant_candidates:
                    self._consonant_candidates.append(consonant_part)

            # Full valid plate enters the voting buffer [PP-3]
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

        # ── [PL-4] Partial-evidence reconstruction ────────────────────────
        # Triggered when consonants are consistently destroyed by binarization
        # but digits reliably survive (the most common failure mode observed)
        if self._digit_suffix_counter:
            best_suffix, suffix_count = self._digit_suffix_counter.most_common(1)[0]
            if (suffix_count >= self.MIN_DIGIT_VOTES
                    and self._consonant_candidates):
                reconstructed = merge_digit_evidence(
                    best_suffix, self._consonant_candidates
                )
                if reconstructed:
                    print(f"  [Voter PL-4] Reconstructed from partial evidence: "
                          f"{reconstructed[0]} "
                          f"(digit_suffix={best_suffix} x{suffix_count}, "
                          f"consonants={self._consonant_candidates})")
                    # Penalised confidence — this is a reconstruction
                    return (reconstructed[0], 0.45)

        return None

    def reset(self):
        self._buffer.clear()
        self._digit_suffix_counter.clear()
        self._consonant_candidates.clear()