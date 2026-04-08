"""
camera_registry.py — Load camera configs from cameras.json.

Each camera entry:
  {
    "camId":        "cam-01",
    "lotId":        "uuid-of-parking-lot",
    "slotId":       "A1",
    "streamUrl":    "http://<ESP32_CAM_IP>:81/stream",
    "camUrl":       "http://<ESP32_CAM_IP>",
    "gateUrl":      "http://<GATE_BOARD_IP>",
    "currentPlate": null,
    "flipCode":     1
  }

New in this version:
  reset_all_plates() — sets currentPlate=null for ALL cameras in the JSON.
  Called once at process startup so stale plate values never survive a restart.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "cameras.json")
_write_lock  = threading.Lock()


@dataclass
class CameraConfig:
    cam_id:        str
    lot_id:        str
    slot_id:       str
    stream_url:    str
    cam_url:       str
    gate_url:      str
    flip_code:     Optional[int]
    current_plate: Optional[str] = field(default=None)


def load_cameras(path: str = _CONFIG_PATH) -> list[CameraConfig]:
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
            current_plate = entry.get("currentPlate"),
        ))
    return cameras


def get_camera(cam_id: str, path: str = _CONFIG_PATH) -> CameraConfig:
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
    """Persist currentPlate for one camera. plate=None clears it."""
    with _write_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                if entry["camId"] == cam_id:
                    entry["currentPlate"] = plate
                    break
            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
            action = f"set → {plate!r}" if plate else "cleared"
            print(f"[CameraRegistry] {cam_id} currentPlate {action}")
        except Exception as exc:
            print(f"[CameraRegistry] Failed to update {path}: {exc}")


def reset_all_plates(path: str = _CONFIG_PATH) -> None:
    """
    Set currentPlate=null for every camera in cameras.json.
    Called once at process startup so no stale plate survives a restart.
    """
    with _write_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                entry["currentPlate"] = None
            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
            print(f"[CameraRegistry] All currentPlate values reset to null at startup.")
        except Exception as exc:
            print(f"[CameraRegistry] Failed to reset plates: {exc}")