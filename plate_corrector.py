"""
plate_corrector.py — Strict Thai Licence-Plate Format Corrector  (v9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGES vs v8
─────────────
  [PC-1]  extract_digit_suffix() — pull reliable 4-digit suffix from
          partial/noisy OCR strings even when consonants are gone.
          Used by PlateMajorityVoter for partial-evidence accumulation.

  [PC-2]  merge_digit_evidence() — combine a confirmed digit suffix with
          consonant candidates seen across frames to reconstruct plates.

  [PC-3]  Pass C extended — improved positional aligner handles 8-char
          noisy strings (province bleed + plate) by first stripping
          any leading Thai vowel/non-consonant chars.

  [PC-4]  Pass D extended — handles length-8 all-digit strings that
          include a spurious leading digit from YOLO crop bleed.

  [PC-5]  normalise_raw() now strips Thai vowel diacritics that bleed
          from the province strip into the number zone read.

THAI PLATE DOMAIN RULES
────────────────────────
Standard passenger plate: [optional 1 digit] [exactly 2 consonants] [exactly 4 digits]
  กข1234          ✓
  1กข1234         ✓
  กข12            ✗  too few digits
  3บ3099          ✗  only 1 consonant — needs correction
  ฐบ3699          ✓  correct

CHARACTER CONFUSION MAPS
────────────────────────
Digit  → Thai consonant : used when a position MUST hold a consonant
Thai   → Digit          : used when a position MUST hold a digit
Latin  → Thai consonant : EasyOCR sometimes returns Latin for Thai
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

# Thai vowel diacritics that bleed from province strip — strip in normalise
_DIACRITIC_SET = set("็่้๊๋์ํ๎ัิีึืุู")

# Canonical plate regex
# Group 1: optional leading digit (city/zone code)
# Group 2: exactly 2 Thai consonants
# Group 3: exactly 4 Arabic digits
_PLATE_RE = re.compile(
    r'^([0-9])?'
    r'([' + THAI_CONSONANTS + r']{2})'
    r'([0-9]{4})$'
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFUSION MAPS
# ═══════════════════════════════════════════════════════════════════════════

_DIGIT_TO_CONSONANT: dict[str, str] = {
    "0": "อ",
    "1": "บ",
    "2": "ฉ",
    "3": "ฐ",
    "4": "ข",
    "5": "ร",
    "6": "ถ",
    "7": "ง",
    "8": "ม",
    "9": "ถ",
}

_THAI_TO_DIGIT: dict[str, str] = {
    "อ": "0",
    "บ": "1",
    "ป": "1",
    "ต": "1",
    "า": "1",
    "ฉ": "2",
    "ฐ": "3",
    "ข": "4",
    "ร": "5",
    "ถ": "6",
    "ง": "7",
    "ม": "8",
    "ก": "8",
    "ว": "0",
    # vowel noise — drop
    "ิ": "",
    "ี": "",
    "ุ": "",
    "ู": "",
    "็": "",
    "่": "",
    "้": "",
    "๊": "",
    "๋": "",
    "์": "",
    "ํ": "",
    "ะ": "",
    "ั": "",
    "ำ": "",
}

_LATIN_TO_CONSONANT: dict[str, str] = {
    "A": "า",
    "B": "บ",
    "C": "ซ",
    "D": "ด",
    "F": "ฝ",
    "G": "ง",
    "H": "ห",
    "I": "บ",
    "J": "จ",
    "K": "ก",
    "L": "ล",
    "M": "ม",
    "N": "น",
    "O": "อ",
    "P": "ป",
    "Q": "ถ",
    "R": "ร",
    "S": "ส",
    "T": "ท",
    "U": "บ",
    "V": "ว",
    "W": "ว",
    "X": "ข",
    "Y": "ย",
    "Z": "ส",
}

# ═══════════════════════════════════════════════════════════════════════════
# CHARACTER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _char_type(ch: str) -> str:
    if ch in _CONSONANT_SET:
        return "consonant"
    if ch in _VOWEL_SET or ch in THAI_DIGITS:
        return "vowel"
    if ch in _DIGIT_SET:
        return "digit"
    if ch.isalpha() and ch.isascii():
        return "latin"
    return "other"


def _to_consonant(ch: str) -> Optional[str]:
    if ch in _CONSONANT_SET:
        return ch
    if ch in _DIGIT_TO_CONSONANT:
        return _DIGIT_TO_CONSONANT[ch]
    if ch.upper() in _LATIN_TO_CONSONANT:
        result = _LATIN_TO_CONSONANT[ch.upper()]
        return result if result in _CONSONANT_SET else None
    return None


def _to_digit(ch: str) -> Optional[str]:
    if ch in ARABIC_DIGITS:
        return ch
    if ch in _THAI_TO_DIGIT:
        result = _THAI_TO_DIGIT[ch]
        return result if result.isdigit() else None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# NORMALISATION  [PC-5]
# ═══════════════════════════════════════════════════════════════════════════

def normalise_raw(text: str) -> str:
    """
    Strip spaces, dashes, pipes, dots, Thai diacritics that bleed from
    province strip. Convert Thai digit glyphs to Arabic. Upper-case Latin.
    [PC-5] Also strips diacritic marks that are common province-bleed noise.
    """
    text = "".join(text.strip().split())
    text = text.replace("-", "").replace("|", "").replace(".", "")
    # [PC-5] Strip diacritics that leak from province OCR
    text = "".join(ch for ch in text if ch not in _DIACRITIC_SET)
    # Thai digit glyphs → Arabic
    thai_digit_map = dict(zip("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    text = "".join(thai_digit_map.get(ch, ch) for ch in text)
    return text.upper()


# ═══════════════════════════════════════════════════════════════════════════
# PASS A — Exact pattern match
# ═══════════════════════════════════════════════════════════════════════════

def _pass_a(text: str) -> Optional[str]:
    if _PLATE_RE.fullmatch(text):
        return text
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PASS B — Structural split
# ═══════════════════════════════════════════════════════════════════════════

def _pass_b(text: str) -> Optional[str]:
    consonant_runs: list[tuple[int, int]] = []
    digit_runs:     list[tuple[int, int]] = []

    i = 0
    while i < len(text):
        if _char_type(text[i]) == "consonant":
            j = i
            while j < len(text) and _char_type(text[j]) == "consonant":
                j += 1
            consonant_runs.append((i, j))
            i = j
        elif _char_type(text[i]) == "digit":
            j = i
            while j < len(text) and _char_type(text[j]) == "digit":
                j += 1
            digit_runs.append((i, j))
            i = j
        else:
            i += 1

    long_c = [(s, e) for s, e in consonant_runs if e - s >= 2]
    long_d = [(s, e) for s, e in digit_runs     if e - s >= 4]

    if len(long_c) == 1 and len(long_d) == 1:
        cs, ce = long_c[0]
        ds, de = long_d[0]
        consonants = text[cs: cs + 2]
        digits     = text[ds: ds + 4]
        prefix = text[:cs].strip()
        leading = prefix[-1] if (prefix and prefix[-1].isdigit()) else ""
        candidate = leading + consonants + digits
        if _PLATE_RE.fullmatch(candidate):
            return candidate

    return None


# ═══════════════════════════════════════════════════════════════════════════
# PASS C — Positional forced substitution  [PC-3]
# ═══════════════════════════════════════════════════════════════════════════

def _pass_c(text: str) -> Optional[str]:
    """
    [PC-3] Extended: also tries to strip a leading non-consonant prefix
    (vowel or garbage char) before aligning the template, handling cases
    where province bleed prepends extra characters.
    """
    slot_patterns = [
        (6, ['c', 'c', 'd', 'd', 'd', 'd']),
        (7, ['L', 'c', 'c', 'd', 'd', 'd', 'd']),
    ]

    def try_align(src: str, slots: list) -> Optional[str]:
        if len(src) != len(slots):
            return None
        out = []
        for ch, slot in zip(src, slots):
            if slot == 'd':
                converted = _to_digit(ch)
                if converted is None:
                    return None
                out.append(converted)
            elif slot == 'c':
                converted = _to_consonant(ch)
                if converted is None:
                    return None
                out.append(converted)
            elif slot == 'L':
                d = _to_digit(ch)
                if d is None:
                    return None
                out.append(d)
        return "".join(out)

    results = []

    # Try direct alignment
    for length, slots in slot_patterns:
        if len(text) == length:
            result = try_align(text, slots)
            if result and _PLATE_RE.fullmatch(result):
                results.append(result)

    # [PC-3] Try stripping 1-2 leading junk chars then aligning
    for strip_n in (1, 2):
        stripped = text[strip_n:]
        for length, slots in slot_patterns:
            if len(stripped) == length:
                result = try_align(stripped, slots)
                if result and _PLATE_RE.fullmatch(result):
                    results.append(result)

    return results[0] if results else None


# ═══════════════════════════════════════════════════════════════════════════
# PASS D — All-digit recovery  [PC-4]
# ═══════════════════════════════════════════════════════════════════════════

def _pass_d(text: str) -> Optional[str]:
    """
    [PC-4] Extended: handles length-8 all-digit strings in addition to
    lengths 6 and 7, where the extra digit is province-bleed noise.
    """
    if not text.isdigit():
        return None

    candidates = []

    if len(text) == 6:
        c1 = _DIGIT_TO_CONSONANT.get(text[0])
        c2 = _DIGIT_TO_CONSONANT.get(text[1])
        if c1 and c2:
            candidate = c1 + c2 + text[2:6]
            if _PLATE_RE.fullmatch(candidate):
                candidates.append(candidate)

    if len(text) == 7:
        c1 = _DIGIT_TO_CONSONANT.get(text[1])
        c2 = _DIGIT_TO_CONSONANT.get(text[2])
        if c1 and c2:
            candidate = text[0] + c1 + c2 + text[3:7]
            if _PLATE_RE.fullmatch(candidate):
                candidates.append(candidate)

    # [PC-4] Length 8: strip first char as noise, try length-7 template
    if len(text) == 8:
        sub = text[1:]  # strip leading noise digit
        c1  = _DIGIT_TO_CONSONANT.get(sub[1])
        c2  = _DIGIT_TO_CONSONANT.get(sub[2])
        if c1 and c2:
            candidate = sub[0] + c1 + c2 + sub[3:7]
            if _PLATE_RE.fullmatch(candidate):
                candidates.append(candidate)
        # Also try length-6 template on last 6 chars
        tail = text[-6:]
        c1   = _DIGIT_TO_CONSONANT.get(tail[0])
        c2   = _DIGIT_TO_CONSONANT.get(tail[1])
        if c1 and c2:
            candidate = c1 + c2 + tail[2:6]
            if _PLATE_RE.fullmatch(candidate):
                candidates.append(candidate)

    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# PASS E — Sliding window
# ═══════════════════════════════════════════════════════════════════════════

def _pass_e(text: str) -> Optional[str]:
    if len(text) <= 7:
        return None

    for length in (7, 6):
        for start in range(len(text) - length + 1):
            window = text[start: start + length]
            for fn in (_pass_a, _pass_c, _pass_d):
                result = fn(window)
                if result:
                    return result

    return None


# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL EVIDENCE HELPERS  [PC-1, PC-2]
# ═══════════════════════════════════════════════════════════════════════════

def extract_digit_suffix(text: str) -> Optional[str]:
    """
    [PC-1] Extract a reliable 4-digit suffix from a partial / noisy OCR string.
    Called by PlateMajorityVoter to accumulate digit evidence even when
    Thai consonants have been destroyed by preprocessing.

    Examples:
      "3099"       → "3099"
      "3699"       → "3699"
      "13699"      → "3699"  (5 digits — last 4 are number zone)
      "3023099"    → "3099"  (last 4 of long all-digit string)
      "ฐบ3699"     → "3699"  (valid plate — extract digit group)
      "52d9"       → None    (< 4 clean digits)
      "3023099d"   → "3099"  (trailing Latin stripped)
    """
    if not text:
        return None

    norm = normalise_raw(text)

    # Already a valid plate — return its digit part directly
    m = _PLATE_RE.fullmatch(norm)
    if m:
        return m.group(3)

    # Extract all digit runs
    digit_runs = re.findall(r'\d+', norm)
    if not digit_runs:
        return None

    longest = max(digit_runs, key=len)

    if len(longest) == 4:
        return longest
    if len(longest) >= 5:
        # Last 4 digits are the number zone; any prefix is noise
        return longest[-4:]

    return None


def merge_digit_evidence(
    digit_suffix: str,
    consonant_candidates: list,
) -> list:
    """
    [PC-2] Combine a confirmed digit suffix with consonant candidates
    observed across frames in the same track to produce full plate strings.

    Parameters
    ----------
    digit_suffix        : 4-digit string (from extract_digit_suffix)
    consonant_candidates: list of 2- or 3-char strings containing
                          Thai consonants (possibly with leading digit)

    Returns list of valid plate strings, most likely first.
    """
    if not digit_suffix or len(digit_suffix) != 4:
        return []
    if not digit_suffix.isdigit():
        return []

    results = []
    seen    = set()

    for cons in consonant_candidates:
        # 2-consonant case: กข + 1234
        if len(cons) == 2 and all(c in _CONSONANT_SET for c in cons):
            candidate = cons + digit_suffix
            if _PLATE_RE.fullmatch(candidate) and candidate not in seen:
                results.append(candidate)
                seen.add(candidate)

        # 3-char case: leading digit + 2 consonants (e.g. "1กข")
        elif (len(cons) == 3
              and cons[0].isdigit()
              and all(c in _CONSONANT_SET for c in cons[1:])):
            candidate = cons + digit_suffix
            if _PLATE_RE.fullmatch(candidate) and candidate not in seen:
                results.append(candidate)
                seen.add(candidate)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def correct_plate(raw: str) -> Tuple[Optional[str], float]:
    """
    Main entry point. Apply all correction passes in order.

    Returns
    -------
    (corrected_text, correction_score)
        corrected_text  : canonical plate string or None if uncorrectable.
        correction_score: 0.0 = perfect match, 1.0 = fully reconstructed.
    """
    if not raw:
        return None, 0.0

    text = normalise_raw(raw)
    if not text:
        return None, 0.0

    # Pass A — already valid
    result = _pass_a(text)
    if result:
        return result, 0.0

    # Pass B — structural split
    result = _pass_b(text)
    if result:
        return result, _correction_score(raw, result)

    # Pass C — positional forced substitution
    result = _pass_c(text)
    if result:
        return result, _correction_score(raw, result)

    # Pass D — all-digit recovery
    result = _pass_d(text)
    if result:
        return result, _correction_score(raw, result)

    # Pass E — sliding window
    result = _pass_e(text)
    if result:
        return result, _correction_score(raw, result)

    return None, 1.0


def _correction_score(original: str, corrected: str) -> float:
    norm_orig = normalise_raw(original)
    if not norm_orig or not corrected:
        return 1.0
    length = max(len(norm_orig), len(corrected))
    if length == 0:
        return 0.0
    diffs = sum(
        1 for a, b in zip(norm_orig.ljust(length), corrected.ljust(length))
        if a != b
    )
    return round(diffs / length, 3)


def is_valid_plate(text: str) -> bool:
    """Quick structural check without correction."""
    return bool(_PLATE_RE.fullmatch(normalise_raw(text)))


def correct_candidates(
    candidates: list,
    max_correction_score: float = 0.6,
) -> list:
    """
    Apply correct_plate() to each candidate and filter out uncorrectable
    or heavily-corrected strings.

    Accepts 2-tuples (text, ocr_conf) or 3-tuples (text, ocr_conf, yolo_conf).
    Returns list of (corrected_text, adjusted_conf).
    adjusted_conf = ocr_conf * (1 - correction_score * 0.5)
    """
    out = []
    for item in candidates:
        text, ocr_conf = item[0], float(item[1])
        corrected, cscore = correct_plate(text)
        if corrected is None:
            continue
        if cscore > max_correction_score:
            continue
        adjusted_conf = ocr_conf * (1.0 - cscore * 0.5)
        out.append((corrected, round(adjusted_conf, 4)))
    return out