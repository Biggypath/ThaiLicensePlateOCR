"""
camera_registry.py — Load camera configs from cameras.json.

Each camera entry:
  {
    "camId":     "cam-01",
    "lotId":     "uuid-of-parking-lot",
    "slotId":    "A1",
    "streamUrl": "http://<ESP32_IP>:81/stream",
    "flipCode":  1          // 1=horiz, 0=vert, -1=both, null=off
  }

Each camera watches one parking slot and detects both entry and exit.

Usage:
  python main.py --camera cam-01
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "cameras.json")


@dataclass
class CameraConfig:
    cam_id: str
    lot_id: str
    slot_id: str
    stream_url: str
    flip_code: Optional[int]  # 1, 0, -1, or None


def load_cameras(path: str = _CONFIG_PATH) -> list[CameraConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cameras = []
    for entry in raw:
        cameras.append(CameraConfig(
            cam_id=entry["camId"],
            lot_id=entry["lotId"],
            slot_id=entry.get("slotId", ""),
            stream_url=entry["streamUrl"],
            flip_code=entry.get("flipCode"),
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
