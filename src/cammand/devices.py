from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    id: str
    name: str
    gesture_key: str  # "ONE" ~ "FIVE"
    knob_min: float
    knob_max: float
    knob_step: float
    knob_off: float
    unit: str


DEVICES: dict[str, DeviceConfig] = {
    "ONE":   DeviceConfig("room_light",  "방조명",      "ONE",   0,  100, 1,   0,  "%"),
    "TWO":   DeviceConfig("stand_light", "스탠드 조명", "TWO",   0,  100, 1,   0,  "%"),
    "THREE": DeviceConfig("fan",         "선풍기",      "THREE", 1,  4,   1,   0,  "단"),
    "FOUR":  DeviceConfig("aircon",      "에어컨",      "FOUR",  18, 30,  0.5, 18, "°C"),
    "FIVE":  DeviceConfig("humidifier",  "가습기",      "FIVE",  0,  100, 5,   0,  "%"),
}
