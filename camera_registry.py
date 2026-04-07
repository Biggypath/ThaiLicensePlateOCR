"""
camera_registry.py — Load camera configs from cameras.json.

Each camera entry:
  {
    "camId":        "cam-01",
    "lotId":        "uuid-of-parking-lot",
    "slotId":       "A1",
    "streamUrl":    "http://<ESP32_CAM_IP>:81/stream",
    "camUrl":       "http://<ESP32_CAM_IP>",         ← cam board HTTP base URL
    "gateUrl":      "http://<GATE_BOARD_IP>",        ← motor+ultrasonic board
    "currentPlate": null,                            ← persisted plate value (null = empty)
    "flipCode":     1                                ← 1=horiz, 0=vert, -1=both, null=off
  }

camUrl       : used by gate_controller to poll /status for notify_clear_pending
gateUrl      : used by gate_controller to POST /open /close etc.
currentPlate : written by main.py whenever a plate is detected or cleared;
               any external service can read cameras.json to know which plate
               is currently occupying each slot without querying the live process.

Usage:
  python main.py --camera cam-01
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "cameras.json")

# File-write lock — prevents corruption if two threads ever call
# update_camera_plate() simultaneously (unlikely but safe).
_write_lock = threading.Lock()


@dataclass
class CameraConfig:
    cam_id:        str
    lot_id:        str
    slot_id:       str
    stream_url:    str
    cam_url:       str             # cam board HTTP base (departure polling)
    gate_url:      str             # gate board HTTP base
    flip_code:     Optional[int]   # 1, 0, -1, or None
    current_plate: Optional[str] = field(default=None)   # NEW: persisted plate


def load_cameras(path: str = _CONFIG_PATH) -> list[CameraConfig]:
    """Parse cameras.json and return a list of CameraConfig objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cameras = []
    for entry in raw:
        cameras.append(CameraConfig(
            cam_id        = entry["camId"],
            lot_id        = entry["lotId"],
            slot_id       = entry.get("slotId", ""),
            stream_url    = entry["streamUrl"],
            cam_url       = entry.get("camUrl", ""),
            gate_url      = entry.get("gateUrl", ""),
            flip_code     = entry.get("flipCode"),
            current_plate = entry.get("currentPlate"),   # NEW
        ))
    return cameras


def get_camera(cam_id: str, path: str = _CONFIG_PATH) -> CameraConfig:
    """Return the CameraConfig for cam_id, or raise ValueError."""
    for cam in load_cameras(path):
        if cam.cam_id == cam_id:
            return cam
    available = [c.cam_id for c in load_cameras(path)]
    raise ValueError(
        f"Camera '{cam_id}' not found in {path}. "
        f"Available: {available}"
    )


def update_camera_plate(
    cam_id: str,
    plate: Optional[str],
    path: str = _CONFIG_PATH,
) -> None:
    """
    Persist the current plate value for cam_id back into cameras.json.

    plate = None  → slot is now empty  (currentPlate written as null)
    plate = "กข1234" → slot is occupied by this registration

    Thread-safe via a module-level lock.  Designed for a single writer
    (the main detection loop); concurrent reads from external processes
    are safe because JSON writes are atomic on most filesystems.
    """
    with _write_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            updated = False
            for entry in raw:
                if entry["camId"] == cam_id:
                    entry["currentPlate"] = plate
                    updated = True
                    break

            if not updated:
                print(f"[CameraRegistry] WARNING: cam_id '{cam_id}' not found "
                      f"in {path} — plate not persisted")
                return

            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)

            action = f"set → {plate!r}" if plate else "cleared"
            print(f"[CameraRegistry] {cam_id} currentPlate {action}")

        except Exception as exc:
            print(f"[CameraRegistry] Failed to update {path}: {exc}")