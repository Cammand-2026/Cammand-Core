from __future__ import annotations

from enum import Enum


class Phase(Enum):
    IDLE = "IDLE"
    SELECTING = "SELECTING"
    SELECTED = "SELECTED"
    ONOFF_CONTROL = "ONOFF_CONTROL"
    CONTROLLING = "CONTROLLING"
