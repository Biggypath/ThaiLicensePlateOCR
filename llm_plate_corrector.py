"""
llm_plate_corrector.py — LLM-based Thai Licence-Plate OCR Fallback  (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fallback layer called AFTER plate_corrector.py (5-pass deterministic corrector)
and PlateMajorityVoter Tier-2 (partial evidence). Only fires when both fail
after MIN_LLM_TRIGGER_VOTES frames.

Set ANTHROPIC_API_KEY environment variable to enable.
"""

import os, json, time, threading, urllib.request, urllib.error
from typing import Optional
from plate_corrector import is_valid_plate, normalise_raw

_API_URL           = "https://api.anthropic.com/v1/messages"
_MODEL             = "claude-haiku-4-5-20251001"
_MAX_TOKENS        = 200
_MAX_CALLS_PER_MIN = 10
_LLM_CONF_CAP      = 0.60
_REQUEST_TIMEOUT   = 6.0

_SYSTEM_PROMPT = """\
You are an expert Thai ALPR OCR post-processor.

Thai plate format: ^([0-9])?([ก-ฮ]{2})([0-9]{4})$
Examples: กข1234  1กข1234  3กบ7744  ฐบ3699

OCR confusion maps:
  Digit→consonant: 0→อ 1→บ 2→ฉ 3→ฐ 4→ข 5→ร 6→ถ 7→ง 8→ม 9→ถ
  Thai→digit: อ→0 บ→1 ป→1 ฉ→2 ฐ→3 ข→4 ร→5 ถ→6 ง→7 ม→8 ว→0

Input JSON: {"ocr_candidates":[{"text":"...","conf":0.0-1.0}],"prior_consonants":[],"digit_suffix":"","frame_count":0}

Return ONLY JSON (no markdown):
{"plate":"<valid plate or null>","confidence":0.0-1.0,"reason":"<≤15 words>"}
"""


class _TokenBucket:
    def __init__(self, calls_per_minute: int):
        self._capacity = calls_per_minute; self._tokens = calls_per_minute
        self._refill_sec = 60.0 / calls_per_minute
        self._last_time  = time.monotonic(); self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now   = time.monotonic()
            added = (now - self._last_time) / self._refill_sec
            self._tokens    = min(self._capacity, self._tokens + added)
            self._last_time = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0; return True
            return False


class LLMCorrector:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._enabled  = bool(self._api_key)
        self._limiter  = _TokenBucket(_MAX_CALLS_PER_MIN)
        self._lock     = threading.Lock()
        self._cache:   dict[int, object] = {}
        self._pending: set[int]          = set()
        if not self._enabled:
            print("[LLM] ANTHROPIC_API_KEY not set — LLM fallback disabled.")

    def is_enabled(self) -> bool: return self._enabled
    def is_pending(self, tid: int) -> bool:
        with self._lock: return tid in self._pending

    def get_cached(self, tid: int) -> Optional[tuple]:
        with self._lock: val = self._cache.get(tid)
        if val and val is not False: return val
        return None

    def clear_track(self, tid: int):
        with self._lock:
            self._cache.pop(tid, None); self._pending.discard(tid)

    def invalidate_track(self, tid: int): self.clear_track(tid)

    def query(self, track_id: int, ocr_candidates: list,
              prior_consonants: list, digit_suffix: Optional[str] = None) -> Optional[tuple]:
        result = self.get_cached(track_id)
        if result is not None: return result
        if not self.is_pending(track_id):
            self.request_async(track_id, ocr_candidates, prior_consonants, digit_suffix)
        return None

    def request_async(self, tid: int, ocr_candidates: list,
                      prior_consonants: list, digit_suffix: Optional[str] = None):
        if not self._enabled: return
        with self._lock:
            if tid in self._pending or tid in self._cache: return
            if not self._limiter.try_acquire():
                print(f"  [LLM #{tid}] Rate limited"); return
            self._pending.add(tid)
        t = threading.Thread(
            target=self._worker,
            args=(tid, list(ocr_candidates), list(prior_consonants), digit_suffix),
            daemon=True, name=f"llm-{tid}")
        t.start()

    def _build_user_message(self, ocr_candidates, prior_consonants, digit_suffix) -> str:
        structured = []
        for item in ocr_candidates:
            if isinstance(item, str):
                structured.append({"text": item, "conf": None})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                structured.append({"text": str(item[0]), "conf": round(float(item[1]), 3)
                                   if item[1] is not None else None})
        seen = {}
        for c in structured:
            key = c["text"].strip()
            if not key: continue
            if key not in seen or (c["conf"] or 0) > (seen[key]["conf"] or 0):
                seen[key] = c
        payload = {"ocr_candidates": list(seen.values()), "prior_consonants": prior_consonants,
                   "digit_suffix": digit_suffix, "frame_count": len(structured)}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _call_api(self, user_message: str) -> Optional[dict]:
        payload = json.dumps({"model": _MODEL, "max_tokens": _MAX_TOKENS,
                              "system": _SYSTEM_PROMPT,
                              "messages": [{"role": "user", "content": user_message}]
                              }).encode("utf-8")
        req = urllib.request.Request(
            _API_URL, data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "x-api-key": self._api_key,
                     "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            print(f"  [LLM] HTTP {exc.code}"); return None
        except Exception as exc:
            print(f"  [LLM] Error: {exc}"); return None
        try:
            data   = json.loads(raw)
            text   = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text").strip()
            text   = text.replace("```json","").replace("```","").strip()
            parsed = json.loads(text)
            if "result" in parsed and isinstance(parsed["result"], dict): parsed = parsed["result"]
            return parsed
        except Exception as exc:
            print(f"  [LLM] Parse error: {exc}"); return None

    def _worker(self, tid, ocr_candidates, prior_consonants, digit_suffix):
        try:
            msg    = self._build_user_message(ocr_candidates, prior_consonants, digit_suffix)
            result = self._call_api(msg)
            if result is None:
                with self._lock: self._cache[tid] = False; self._pending.discard(tid)
                return
            plate  = result.get("plate"); conf = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")
            if plate and is_valid_plate(plate):
                capped = min(conf, _LLM_CONF_CAP)
                print(f"  [LLM #{tid}] ✓ {plate} conf={capped:.2f} | {reason}")
                with self._lock: self._cache[tid] = (plate, capped)
            else:
                print(f"  [LLM #{tid}] ✗ {plate!r} | {reason}")
                with self._lock: self._cache[tid] = False
        except Exception as exc:
            print(f"  [LLM #{tid}] Exception: {exc}")
            with self._lock: self._cache[tid] = False
        finally:
            with self._lock: self._pending.discard(tid)