"""
llm_plate_corrector.py — LLM-based Thai Licence-Plate OCR Fallback  (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGES vs v1
─────────────
  [P-1]  Completely redesigned _SYSTEM_PROMPT:
           • Structured JSON *input* schema so the model knows exactly
             what each field means (candidates, confidences, prior
             consonants, digit suffix, frame_count).
           • Full positional correction tables (digit→consonant,
             consonant/vowel→digit, Latin→consonant).
           • Explicit province-noise stripping rules.
           • 9-step numbered reasoning chain the model must follow.
           • Worked examples covering every real failure mode seen in
             the v6/v7/v10 live logs (leading-digit plates, all-digit
             strings, mixed noise, partial evidence).
           • Strict output JSON schema with null contract.

  [P-2]  _build_user_message() now emits structured JSON so the model
         always sees a consistent, parseable input (not free-form text).
         Includes per-candidate confidence values.

  [P-3]  Response parser now tolerates the model wrapping the JSON in
         a top-level "result" key (occasionally emitted by Haiku).

  [P-4]  Model updated to claude-haiku-4-5-20251001 (unchanged from v1
         — fastest / cheapest; the rich prompt compensates).

PURPOSE
───────
Fallback layer called AFTER plate_corrector.py (5-pass deterministic
corrector) and PlateMajorityVoter (PL-4 partial evidence).  Only fires
when both fail after MIN_LLM_TRIGGER_VOTES frames.
"""

import os
import json
import time
import threading
import urllib.request
import urllib.error
from typing import Optional

from plate_corrector import is_valid_plate, normalise_raw

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

_API_URL            = "https://api.anthropic.com/v1/messages"
_MODEL              = "claude-haiku-4-5-20251001"
_MAX_TOKENS         = 200
_MAX_CALLS_PER_MIN  = 10
_LLM_CONF_CAP       = 0.60
_REQUEST_TIMEOUT    = 6.0

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  [P-1]
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert Thai Automatic Licence-Plate Recognition (ALPR) OCR
post-processor running on a real-time camera system in Thailand.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THAI LICENCE-PLATE FORMAT (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regex (Python):  ^([0-9])?([ก-ฮ]{2})([0-9]{4})$

Structure:
  • OPTIONAL leading Arabic digit  (zone/city code, e.g. "3" in "3กบ7744")
  • EXACTLY 2 Thai consonants      (ก ข ฃ ค ฅ ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ
                                    ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ
                                    ส ห ฬ อ ฮ — 44 consonants total)
  • EXACTLY 4 Arabic digits        (0-9)

Valid examples:  กข1234   1กข1234   ฐบ3699   3กบ7744   รบ5500
Invalid:         กข123  (3 digits)   กขค1234  (3 consonants)   3บ3699  (1 consonant)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OCR CONFUSION MAPS  (apply positionally)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) Digit → Thai consonant  (when slot MUST be a consonant)
   0→อ  1→บ  2→ฉ  3→ฐ  4→ข  5→ร  6→ถ  7→ง  8→ม  9→ถ

B) Thai/vowel → Arabic digit  (when slot MUST be a digit)
   ว→0  บ→1  ป→1  ต→1  า→1  ฉ→2  ฐ→3  ข→4  ร→5  ถ→6  ง→7  ม→8  ก→8  อ→0

C) Latin ASCII → Thai consonant  (EasyOCR sometimes emits Latin)
   B→บ  I→บ  U→บ  O→อ  G→ง  M→ม  N→น  R→ร  S→ส  T→ท
   X→ข  K→ก  D→ด  H→ห  W→ว  Y→ย  Z→ส  P→ป  Q→ถ  F→ฝ

D) Province text to STRIP (appears below plate number — never part of plate)
   Common: กรุงเทพมหานคร เชียงใหม่ ชลบุรี ภูเก็ต นนทบุรี สมุทรปราการ
   Rule: any Thai sequence > 4 chars that is NOT a consonant-pair is province.

E) Diacritics to STRIP (bleed from province OCR):
   ็ ่ ้ ๊ ๋ ์ ํ ะ ั ำ ิ ี ึ ื ุ ู

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT JSON SCHEMA (what you receive)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "ocr_candidates": [
    {"text": "<raw OCR string>", "conf": 0.0-1.0, "engine": "easyocr|tesseract"}
  ],
  "prior_consonants": ["กบ", "3กบ"],  // consonant patterns accumulated across frames
  "digit_suffix": "7744",              // most consistent 4-digit suffix seen (or null)
  "frame_count": 5                     // frames accumulated before LLM called
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9-STEP CORRECTION PROCEDURE (follow in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — NORMALISE each candidate
  • Strip spaces, dashes (-), pipes (|), dots (.)
  • Remove diacritics from set E above
  • Convert Thai digit glyphs ๐๑๒๓๔๕๖๗๘๙ → 0123456789
  • Upper-case any Latin characters

STEP 2 — VALIDATE normalised candidates
  • If any normalised candidate already matches ^([0-9])?[ก-ฮ]{2}[0-9]{4}$
    → return it immediately (score = candidate confidence, capped at 0.60)
  • Prefer the candidate with the highest confidence.

STEP 3 — SELECT BEST WINDOW (if candidate length > 7)
  • Province text often bleeds in — slide a 6- or 7-character window
    over the normalised string to find the best plate-like sub-string.
  • Evaluate each window with Steps 4–6 and pick the one that produces
    a valid plate.

STEP 4 — POSITIONAL SLOT ALIGNMENT
  For a 6-char string apply template: [c][c][d][d][d][d]
  For a 7-char string apply template: [L][c][c][d][d][d][d]
    where L = optional leading Arabic digit (0-9)
          c = Thai consonant slot → apply map A then C
          d = Arabic digit slot   → apply map B
  Any slot that cannot be resolved → plate is unrecoverable (go to Step 8).

STEP 5 — ALL-DIGIT RECOVERY
  If the entire normalised string is digits:
    • Length 6: treat chars 0-1 as consonant-coded (map A), chars 2-5 as digits
    • Length 7: treat char 0 as leading digit, chars 1-2 as consonant-coded, 3-6 as digits
    • Length 8: strip first char as noise, treat as length-7
  Example: "313699" → 3→ฐ, 1→บ, digits=3699 → "ฐบ3699"

STEP 6 — USE PRIOR CONSONANTS (when digit_suffix is available)
  If prior_consonants is non-empty and digit_suffix is 4 digits:
    • Try each element of prior_consonants as the consonant prefix
    • If 2 chars: build <cons><digit_suffix>           e.g. "กบ"+"7744" = "กบ7744"
    • If 3 chars starting with digit: build as-is + digit_suffix  e.g. "3กบ"+"7744" = "3กบ7744"
    • Validate and return highest-confidence reconstruction

STEP 7 — CROSS-CANDIDATE FUSION
  If multiple candidates partially agree (e.g. one has correct consonants,
  another has correct digits), merge them:
    • Consonants from the candidate where they appear in consonant slots
    • Digits from the candidate with highest digit-zone confidence
    • Prefer digit_suffix if provided

STEP 8 — UNRECOVERABLE CHECK
  If after Steps 2-7 no valid plate was produced:
    • Set plate to null
    • Set confidence to 0.0

STEP 9 — CONFIDENCE SCORING
  1.0  = candidate was already valid (Step 2)
  0.85 = positional alignment succeeded with no ambiguous chars
  0.70 = 1-2 chars were substituted with high-confidence maps
  0.55 = 3+ substitutions, or sliding-window was needed
  0.40 = prior_consonants were the sole consonant source
  0.0  = null result

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 — leading-digit plate already valid
  Input:  ocr_candidates=[{"text":"3กบ7744","conf":0.82}], prior_consonants=["กบ","3กบ"], digit_suffix="7744"
  Step 2: normalise → "3กบ7744", matches regex → valid immediately
  Output: {"plate":"3กบ7744","confidence":0.82,"reason":"Already valid; leading digit 3 preserved"}

Example 2 — all-digit string
  Input:  ocr_candidates=[{"text":"313699","conf":0.67}], prior_consonants=["ฐบ"], digit_suffix="3699"
  Step 5: length 6 all-digit → 3→ฐ, 1→บ, 3699 → "ฐบ3699"
  Output: {"plate":"ฐบ3699","confidence":0.70,"reason":"Digits 3→ฐ 1→บ; suffix 3699 confirmed"}

Example 3 — mixed consonant/digit with one digit in consonant slot
  Input:  ocr_candidates=[{"text":"ฐ13699","conf":0.60}], prior_consonants=["ฐบ"], digit_suffix="3699"
  Step 4: template [c][c][d][d][d][d] → ฐ=ฐ, 1→บ, 3699 → "ฐบ3699"
  Output: {"plate":"ฐบ3699","confidence":0.75,"reason":"ฐ valid; 1→บ in consonant slot 2"}

Example 4 — prior consonants + digit suffix
  Input:  ocr_candidates=[{"text":"สm","conf":0.01},{"text":"ศกร","conf":0.11}], prior_consonants=["กบ"], digit_suffix="7744"
  Step 6: OCR too noisy; use prior "กบ" + suffix "7744" → "กบ7744"
  Output: {"plate":"กบ7744","confidence":0.40,"reason":"OCR unusable; reconstructed from prior consonants and suffix"}

Example 5 — province noise bleed
  Input:  ocr_candidates=[{"text":"กบ7744กรงเทพ","conf":0.55}], prior_consonants=["กบ"], digit_suffix="7744"
  Step 3: slide window, "กบ7744" (first 6) matches → valid
  Output: {"plate":"กบ7744","confidence":0.70,"reason":"Province noise stripped from suffix"}

Example 6 — truly unrecoverable
  Input:  ocr_candidates=[{"text":"ภญณฏ","conf":0.0},{"text":"วกฬทกยก","conf":0.01}], prior_consonants=[], digit_suffix=null
  Steps 2-8 all fail — no digit run, no valid window, no prior context
  Output: {"plate":null,"confidence":0.0,"reason":"Insufficient evidence; all candidates below threshold"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (return ONLY this JSON, no markdown fences)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "plate": "<valid plate string matching ^([0-9])?[ก-ฮ]{2}[0-9]{4}$ or null>",
  "confidence": <float 0.0–1.0>,
  "reason": "<≤15 words explaining the key correction step used>"
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════

class _TokenBucket:
    def __init__(self, calls_per_minute: int):
        self._capacity   = calls_per_minute
        self._tokens     = calls_per_minute
        self._refill_sec = 60.0 / calls_per_minute
        self._last_time  = time.monotonic()
        self._lock       = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now   = time.monotonic()
            added = (now - self._last_time) / self._refill_sec
            self._tokens    = min(self._capacity, self._tokens + added)
            self._last_time = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class LLMCorrector:
    """
    Thread-safe LLM fallback corrector.

    Usage in main.py:
        llm = LLMCorrector()                          # once at startup

        # Inside per-track OCR loop, after voter returns None:
        llm_result = llm.query(
            track_id         = tid,
            ocr_candidates   = list(_get_ocr_history(tid)),
            prior_consonants = voter._consonant_candidates,
        )
        if llm_result:
            plate_text, conf = llm_result
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._enabled  = bool(self._api_key)
        self._limiter  = _TokenBucket(_MAX_CALLS_PER_MIN)
        self._lock     = threading.Lock()
        self._cache:   dict[int, object] = {}
        self._pending: set[int]          = set()

        if not self._enabled:
            print("[LLM] ANTHROPIC_API_KEY not set — LLM fallback disabled.")

    # ── Public API ──────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled

    def is_pending(self, tid: int) -> bool:
        with self._lock:
            return tid in self._pending

    def get_cached(self, tid: int) -> Optional[tuple]:
        with self._lock:
            val = self._cache.get(tid)
        if val and val is not False:
            return val
        return None

    def clear_track(self, tid: int):
        with self._lock:
            self._cache.pop(tid, None)
            self._pending.discard(tid)

    def invalidate_track(self, tid: int):
        self.clear_track(tid)

    def query(
        self,
        track_id: int,
        ocr_candidates: list,
        prior_consonants: list,
        digit_suffix: Optional[str] = None,
    ) -> Optional[tuple]:
        """
        Compatibility wrapper for main.py.
        Returns a cached LLM result if available, otherwise starts
        an async request and returns None.
        """
        result = self.get_cached(track_id)
        if result is not None:
            return result
        if not self.is_pending(track_id):
            self.request_async(track_id, ocr_candidates,
                               prior_consonants, digit_suffix)
        return None

    def request_async(
        self,
        tid: int,
        ocr_candidates: list,
        prior_consonants: list,
        digit_suffix: Optional[str] = None,
    ):
        """[LLM-1] Fire-and-forget background thread."""
        if not self._enabled:
            return
        with self._lock:
            if tid in self._pending or tid in self._cache:
                return
            if not self._limiter.try_acquire():
                print(f"  [LLM #{tid}] Rate limited — skipping")
                return
            self._pending.add(tid)

        t = threading.Thread(
            target = self._worker,
            args   = (tid, list(ocr_candidates),
                      list(prior_consonants), digit_suffix),
            daemon = True,
            name   = f"llm-corrector-{tid}",
        )
        t.start()

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        ocr_candidates: list,
        prior_consonants: list,
        digit_suffix: Optional[str],
    ) -> str:
        """
        [P-2] Emit structured JSON so the model always receives a
        consistent, parseable input with per-candidate confidence values.

        ocr_candidates can be:
          - list of str  (raw OCR strings from history)
          - list of tuple (text, conf) or (text, conf, yolo_conf)
        """
        # Normalise candidates into {"text", "conf", "engine"} dicts
        structured: list[dict] = []
        for item in ocr_candidates:
            if isinstance(item, str):
                structured.append({"text": item, "conf": None, "engine": "unknown"})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[0])
                try:
                    conf = round(float(item[1]), 3)
                except (TypeError, ValueError):
                    conf = None
                engine = str(item[2]) if len(item) > 2 else "unknown"
                structured.append({"text": text, "conf": conf, "engine": engine})
            else:
                continue

        # Deduplicate by text, keep highest confidence
        seen: dict[str, dict] = {}
        for c in structured:
            key = c["text"].strip()
            if not key:
                continue
            if key not in seen:
                seen[key] = c
            else:
                if (c["conf"] or 0) > (seen[key]["conf"] or 0):
                    seen[key] = c

        payload = {
            "ocr_candidates": list(seen.values()),
            "prior_consonants": prior_consonants,
            "digit_suffix": digit_suffix,
            "frame_count": len(structured),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _call_api(self, user_message: str) -> Optional[dict]:
        """Synchronous API call. Returns parsed JSON dict or None."""
        payload = json.dumps({
            "model"      : _MODEL,
            "max_tokens" : _MAX_TOKENS,
            "system"     : _SYSTEM_PROMPT,
            "messages"   : [{"role": "user", "content": user_message}],
        }).encode("utf-8")

        req = urllib.request.Request(
            _API_URL,
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type"      : "application/json",
                "x-api-key"         : self._api_key,
                "anthropic-version" : "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            print(f"  [LLM] HTTP {exc.code}: {exc.read().decode()[:200]}")
            return None
        except Exception as exc:
            print(f"  [LLM] Request error: {exc}")
            return None

        try:
            data = json.loads(raw)
            text = "".join(
                b.get("text", "") for b in data.get("content", [])
                if b.get("type") == "text"
            ).strip()
            # Strip markdown fences
            text = text.replace("```json", "").replace("```", "").strip()

            # [P-3] Tolerate top-level "result" wrapper occasionally emitted by Haiku
            parsed = json.loads(text)
            if "result" in parsed and isinstance(parsed["result"], dict):
                parsed = parsed["result"]
            return parsed

        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  [LLM] Parse error: {exc} | raw: {raw[:300]}")
            return None

    def _worker(
        self,
        tid: int,
        ocr_candidates: list,
        prior_consonants: list,
        digit_suffix: Optional[str],
    ):
        """Background thread body."""
        try:
            user_msg = self._build_user_message(
                ocr_candidates, prior_consonants, digit_suffix
            )
            print(f"  [LLM #{tid}] Querying — candidates: "
                  f"{[c if isinstance(c, str) else c[0] for c in ocr_candidates[:5]]}")
            result = self._call_api(user_msg)

            if result is None:
                with self._lock:
                    self._cache[tid] = False
                    self._pending.discard(tid)
                return

            plate  = result.get("plate")
            conf   = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")

            if plate and is_valid_plate(plate):
                capped = min(conf, _LLM_CONF_CAP)
                print(f"  [LLM #{tid}] ✓ {plate} conf={capped:.2f} | {reason}")
                with self._lock:
                    self._cache[tid] = (plate, capped)
            else:
                print(f"  [LLM #{tid}] ✗ null or invalid: {plate!r} | {reason}")
                with self._lock:
                    self._cache[tid] = False

        except Exception as exc:
            print(f"  [LLM #{tid}] Worker exception: {exc}")
            with self._lock:
                self._cache[tid] = False
        finally:
            with self._lock:
                self._pending.discard(tid)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: synchronous version for testing / offline use
# ═══════════════════════════════════════════════════════════════════════════

def llm_correct_plate_sync(
    ocr_candidates: list,
    prior_consonants: Optional[list] = None,
    digit_suffix: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[tuple]:
    """
    Synchronous wrapper — blocks until LLM responds.
    Returns (plate_text, confidence) or None.
    Useful for testing and batch processing outside the real-time loop.
    """
    corrector = LLMCorrector(api_key=api_key)
    if not corrector.is_enabled():
        return None
    msg    = corrector._build_user_message(
        ocr_candidates,
        prior_consonants or [],
        digit_suffix,
    )
    result = corrector._call_api(msg)
    if result is None:
        return None
    plate = result.get("plate")
    conf  = float(result.get("confidence", 0.0))
    if plate and is_valid_plate(plate):
        return (plate, min(conf, _LLM_CONF_CAP))
    return None