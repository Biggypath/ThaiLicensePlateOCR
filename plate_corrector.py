"""
plate_corrector.py — Strict Thai Licence-Plate Format Corrector  (v9.1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THAI PLATE DOMAIN RULES
────────────────────────
Standard passenger plate: [optional 1 digit] [exactly 2 consonants] [exactly 4 digits]
  กข1234   ✓     1กข1234  ✓     3กบ7744  ✓
  กข12     ✗ (3 digits)      3บ3099  ✗ (1 consonant)

Changes vs v9:
  [PC-6]  merge_digit_evidence() sorted by SPECIFICITY — "3กบ" before "กบ"
  [PC-1]  extract_digit_suffix()
  [PC-2]  merge_digit_evidence() base logic
  [PC-3]  Pass C extended (strip leading junk)
  [PC-4]  Pass D extended (length-8 all-digit)
  [PC-5]  normalise_raw() diacritic stripping
"""

import re
from typing import Optional, Tuple
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════
# CHARACTER SETS
# ═══════════════════════════════════════════════════════════════════════════

THAI_CONSONANTS: str = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
THAI_VOWELS: str     = "าิีึืุูเแโใไๆ็่้๊๋์ํ๎ะัำ"
THAI_DIGITS: str     = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS: str   = "0123456789"

_CONSONANT_SET = set(THAI_CONSONANTS)
_VOWEL_SET     = set(THAI_VOWELS)
_DIGIT_SET     = set(ARABIC_DIGITS + THAI_DIGITS)
_DIACRITIC_SET = set("็่้๊๋์ํ๎ัิีึืุู")

_PLATE_RE = re.compile(
    r'^([0-9])?'
    r'([' + THAI_CONSONANTS + r']{2})'
    r'([0-9]{4})$'
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFUSION MAPS
# ═══════════════════════════════════════════════════════════════════════════

_DIGIT_TO_CONSONANT: dict[str, str] = {
    "0": "อ", "1": "บ", "2": "ฉ", "3": "ฐ",
    "4": "ข", "5": "ร", "6": "ถ", "7": "ง",
    "8": "ม", "9": "ถ",
}

_THAI_TO_DIGIT: dict[str, str] = {
    "อ": "0", "บ": "1", "ป": "1", "ต": "1", "า": "1",
    "ฉ": "2", "ฐ": "3", "ข": "4", "ร": "5", "ถ": "6",
    "ง": "7", "ม": "8", "ก": "8", "ว": "0",
    "ิ": "", "ี": "", "ุ": "", "ู": "", "็": "", "่": "",
    "้": "", "๊": "", "๋": "", "์": "", "ํ": "", "ะ": "",
    "ั": "", "ำ": "",
}

_LATIN_TO_CONSONANT: dict[str, str] = {
    "A": "า", "B": "บ", "C": "ซ", "D": "ด", "F": "ฝ",
    "G": "ง", "H": "ห", "I": "บ", "J": "จ", "K": "ก",
    "L": "ล", "M": "ม", "N": "น", "O": "อ", "P": "ป",
    "Q": "ถ", "R": "ร", "S": "ส", "T": "ท", "U": "บ",
    "V": "ว", "W": "ว", "X": "ข", "Y": "ย", "Z": "ส",
}

# ═══════════════════════════════════════════════════════════════════════════
# CHARACTER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _char_type(ch: str) -> str:
    if ch in _CONSONANT_SET: return "consonant"
    if ch in _VOWEL_SET or ch in THAI_DIGITS: return "vowel"
    if ch in _DIGIT_SET: return "digit"
    if ch.isalpha() and ch.isascii(): return "latin"
    return "other"


def _to_consonant(ch: str) -> Optional[str]:
    if ch in _CONSONANT_SET: return ch
    if ch in _DIGIT_TO_CONSONANT: return _DIGIT_TO_CONSONANT[ch]
    if ch.upper() in _LATIN_TO_CONSONANT:
        result = _LATIN_TO_CONSONANT[ch.upper()]
        return result if result in _CONSONANT_SET else None
    return None


def _to_digit(ch: str) -> Optional[str]:
    if ch in ARABIC_DIGITS: return ch
    if ch in _THAI_TO_DIGIT:
        result = _THAI_TO_DIGIT[ch]
        return result if result.isdigit() else None
    return None

# ═══════════════════════════════════════════════════════════════════════════
# NORMALISATION  [PC-5]
# ═══════════════════════════════════════════════════════════════════════════

def normalise_raw(text: str) -> str:
    text = "".join(text.strip().split())
    text = text.replace("-", "").replace("|", "").replace(".", "")
    text = "".join(ch for ch in text if ch not in _DIACRITIC_SET)
    thai_digit_map = dict(zip("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    text = "".join(thai_digit_map.get(ch, ch) for ch in text)
    return text.upper()

# ═══════════════════════════════════════════════════════════════════════════
# PASSES A–E
# ═══════════════════════════════════════════════════════════════════════════

def _pass_a(text: str) -> Optional[str]:
    if _PLATE_RE.fullmatch(text): return text
    return None


def _pass_b(text: str) -> Optional[str]:
    consonant_runs, digit_runs = [], []
    i = 0
    while i < len(text):
        if _char_type(text[i]) == "consonant":
            j = i
            while j < len(text) and _char_type(text[j]) == "consonant": j += 1
            consonant_runs.append((i, j)); i = j
        elif _char_type(text[i]) == "digit":
            j = i
            while j < len(text) and _char_type(text[j]) == "digit": j += 1
            digit_runs.append((i, j)); i = j
        else:
            i += 1
    long_c = [(s, e) for s, e in consonant_runs if e - s >= 2]
    long_d = [(s, e) for s, e in digit_runs     if e - s >= 4]
    if len(long_c) == 1 and len(long_d) == 1:
        cs, ce = long_c[0]; ds, de = long_d[0]
        consonants = text[cs: cs + 2]; digits = text[ds: ds + 4]
        prefix  = text[:cs].strip()
        leading = prefix[-1] if (prefix and prefix[-1].isdigit()) else ""
        candidate = leading + consonants + digits
        if _PLATE_RE.fullmatch(candidate): return candidate
    return None


def _pass_c(text: str) -> Optional[str]:
    slot_patterns = [
        (6, ['c', 'c', 'd', 'd', 'd', 'd']),
        (7, ['L', 'c', 'c', 'd', 'd', 'd', 'd']),
    ]
    def try_align(src, slots):
        if len(src) != len(slots): return None
        out = []
        for ch, slot in zip(src, slots):
            if slot == 'd':
                v = _to_digit(ch);     (lambda: None)() if v is None else None
                if v is None: return None
                out.append(v)
            elif slot == 'c':
                v = _to_consonant(ch);
                if v is None: return None
                out.append(v)
            elif slot == 'L':
                d = _to_digit(ch)
                if d is None: return None
                out.append(d)
        return "".join(out)

    results = []
    for length, slots in slot_patterns:
        if len(text) == length:
            result = try_align(text, slots)
            if result and _PLATE_RE.fullmatch(result): results.append(result)
    for strip_n in (1, 2):
        stripped = text[strip_n:]
        for length, slots in slot_patterns:
            if len(stripped) == length:
                result = try_align(stripped, slots)
                if result and _PLATE_RE.fullmatch(result): results.append(result)
    return results[0] if results else None


def _pass_d(text: str) -> Optional[str]:
    if not text.isdigit(): return None
    candidates = []
    if len(text) == 6:
        c1 = _DIGIT_TO_CONSONANT.get(text[0]); c2 = _DIGIT_TO_CONSONANT.get(text[1])
        if c1 and c2:
            cand = c1 + c2 + text[2:6]
            if _PLATE_RE.fullmatch(cand): candidates.append(cand)
    if len(text) == 7:
        c1 = _DIGIT_TO_CONSONANT.get(text[1]); c2 = _DIGIT_TO_CONSONANT.get(text[2])
        if c1 and c2:
            cand = text[0] + c1 + c2 + text[3:7]
            if _PLATE_RE.fullmatch(cand): candidates.append(cand)
    if len(text) == 8:
        sub = text[1:]
        c1 = _DIGIT_TO_CONSONANT.get(sub[1]); c2 = _DIGIT_TO_CONSONANT.get(sub[2])
        if c1 and c2:
            cand = sub[0] + c1 + c2 + sub[3:7]
            if _PLATE_RE.fullmatch(cand): candidates.append(cand)
        tail = text[-6:]
        c1 = _DIGIT_TO_CONSONANT.get(tail[0]); c2 = _DIGIT_TO_CONSONANT.get(tail[1])
        if c1 and c2:
            cand = c1 + c2 + tail[2:6]
            if _PLATE_RE.fullmatch(cand): candidates.append(cand)
    return candidates[0] if candidates else None


def _pass_e(text: str) -> Optional[str]:
    if len(text) <= 7: return None
    for length in (7, 6):
        for start in range(len(text) - length + 1):
            window = text[start: start + length]
            for fn in (_pass_a, _pass_c, _pass_d):
                result = fn(window)
                if result: return result
    return None

# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL EVIDENCE HELPERS  [PC-1, PC-2, PC-6]
# ═══════════════════════════════════════════════════════════════════════════

def extract_digit_suffix(text: str) -> Optional[str]:
    """[PC-1] Extract reliable 4-digit suffix from partial/noisy OCR."""
    if not text: return None
    norm = normalise_raw(text)
    m = _PLATE_RE.fullmatch(norm)
    if m: return m.group(3)
    digit_runs = re.findall(r'\d+', norm)
    if not digit_runs: return None
    longest = max(digit_runs, key=len)
    if len(longest) == 4: return longest
    if len(longest) >= 5: return longest[-4:]
    return None


def merge_digit_evidence(digit_suffix: str, consonant_candidates: list) -> list:
    """[PC-2, PC-6] Combine digit suffix with consonant candidates. Specific results first."""
    if not digit_suffix or len(digit_suffix) != 4 or not digit_suffix.isdigit():
        return []
    with_leading, without_leading, seen = [], [], set()
    for cons in consonant_candidates:
        if len(cons) == 2 and all(c in _CONSONANT_SET for c in cons):
            cand = cons + digit_suffix
            if _PLATE_RE.fullmatch(cand) and cand not in seen:
                without_leading.append(cand); seen.add(cand)
        elif (len(cons) == 3 and cons[0].isdigit()
              and all(c in _CONSONANT_SET for c in cons[1:])):
            cand = cons + digit_suffix
            if _PLATE_RE.fullmatch(cand) and cand not in seen:
                with_leading.append(cand); seen.add(cand)
    return with_leading + without_leading

# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def correct_plate(raw: str) -> Tuple[Optional[str], float]:
    if not raw: return None, 0.0
    text = normalise_raw(raw)
    if not text: return None, 0.0
    for fn, score_hint in [
        (_pass_a, 0.0),
        (_pass_b, None),
        (_pass_c, None),
        (_pass_d, None),
        (_pass_e, None),
    ]:
        result = fn(text)
        if result:
            sc = 0.0 if score_hint == 0.0 else _correction_score(raw, result)
            return result, sc
    return None, 1.0


def _correction_score(original: str, corrected: str) -> float:
    norm_orig = normalise_raw(original)
    if not norm_orig or not corrected: return 1.0
    length = max(len(norm_orig), len(corrected))
    if length == 0: return 0.0
    diffs = sum(1 for a, b in zip(norm_orig.ljust(length), corrected.ljust(length)) if a != b)
    return round(diffs / length, 3)


def is_valid_plate(text: str) -> bool:
    return bool(_PLATE_RE.fullmatch(normalise_raw(text)))


def correct_candidates(candidates: list, max_correction_score: float = 0.6) -> list:
    out = []
    for item in candidates:
        text, ocr_conf = item[0], float(item[1])
        corrected, cscore = correct_plate(text)
        if corrected is None or cscore > max_correction_score: continue
        adjusted_conf = ocr_conf * (1.0 - cscore * 0.5)
        out.append((corrected, round(adjusted_conf, 4)))
    return out