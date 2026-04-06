"""
slot_presence.py — Detect entry/exit from a single slot camera.

Logic:
  - When a validated plate is read and no car is currently parked → ENTRY
  - When no plate is detected for ABSENCE_TIMEOUT seconds after a car
    was parked → EXIT (the car drove away)

State machine per camera:
  EMPTY  ──plate detected──►  OCCUPIED  ──plate absent for N sec──►  EMPTY
                                  │                                     │
                                  └──────── same plate seen ◄───────────┘
                                            (stays occupied)
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# How many seconds of no plate detection before we declare the car has left
ABSENCE_TIMEOUT = 15.0


@dataclass
class SlotState:
    """Tracks whether a parking slot is occupied."""
    plate: Optional[str] = None          # registration of parked car
    province: Optional[str] = None
    entry_time: float = 0.0
    last_seen_time: float = 0.0          # last frame that still saw the plate
    _announced_exit: bool = field(default=False, repr=False)

    @property
    def is_occupied(self) -> bool:
        return self.plate is not None

    def park(self, plate: str, province: str, now: float) -> None:
        self.plate = plate
        self.province = province
        self.entry_time = now
        self.last_seen_time = now
        self._announced_exit = False

    def see(self, now: float) -> None:
        """Call every frame where the plate is still visible."""
        self.last_seen_time = now

    def check_departure(self, now: float) -> bool:
        """Returns True once when the car has been absent long enough."""
        if not self.is_occupied:
            return False
        if self._announced_exit:
            return False
        if now - self.last_seen_time >= ABSENCE_TIMEOUT:
            self._announced_exit = True
            return True
        return False

    def clear(self) -> None:
        """Reset to empty after exit has been published."""
        departed_plate = self.plate
        self.plate = None
        self.province = None
        self.entry_time = 0.0
        self.last_seen_time = 0.0
        self._announced_exit = False
        return departed_plate
