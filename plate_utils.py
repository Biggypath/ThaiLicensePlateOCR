"""Utilities for plate text extraction and deduplication logic."""


def normalize_plate_text(text: str) -> str:
    """Normalize OCR text for stable plate matching."""
    compact = "".join(text.strip().split())
    compact = compact.replace("-", "")
    return compact.upper()


def extract_best_plate_read(ocr_results, min_length: int = 3):
    """Return the highest-confidence normalized plate read as (text, confidence)."""
    if not ocr_results:
        return None

    valid_results = []
    for result in ocr_results:
        if len(result) < 3:
            continue
        text = result[1]
        confidence = result[2]
        if isinstance(text, str):
            valid_results.append((text, confidence))

    if not valid_results:
        return None

    best_text, best_conf = max(valid_results, key=lambda item: float(item[1]))
    plate_text = normalize_plate_text(best_text)

    if len(plate_text) < min_length:
        return None

    return plate_text, float(best_conf)


def should_send_plate(plate_text: str, last_seen_plates: dict, current_time: float, cooldown_time: int) -> bool:
    """Check if enough time has passed to send this plate again."""
    last_seen = last_seen_plates.get(plate_text, 0)
    return (current_time - last_seen) > cooldown_time
