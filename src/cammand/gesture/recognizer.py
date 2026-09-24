from __future__ import annotations

import math

from ..engine.base import HandLandmarks, Landmark


def _palm_size(lm: HandLandmarks) -> float:
    return math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)


def _finger_open(lm: HandLandmarks, tip: int, pip: int, ref: float) -> bool:
    if lm[tip].z < lm[pip].z - ref * 0.3:
        return False
    y_open = lm[tip].y < lm[pip].y
    d_tip_wrist = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
    d_pip_wrist = math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)
    d_open = d_tip_wrist > d_pip_wrist
    return y_open or d_open


def _four_fingers_open(lm: HandLandmarks, ref: float) -> tuple[bool, bool, bool, bool]:
    return (
        _finger_open(lm, 8,  6,  ref),
        _finger_open(lm, 12, 10, ref),
        _finger_open(lm, 16, 14, ref),
        _finger_open(lm, 20, 18, ref),
    )


def _thumb_open(lm: HandLandmarks) -> bool:
    d_tip = math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y)
    d_ip = math.hypot(lm[3].x - lm[0].x, lm[3].y - lm[0].y)
    return d_tip > d_ip * 1.15


def _dist3(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _thumb_open_3d(lm: HandLandmarks) -> bool:
    d_tip = _dist3(lm[4], lm[2])
    d_ip = _dist3(lm[3], lm[2])
    return d_tip > d_ip * 1.15


def _is_horizontal(lm: HandLandmarks) -> bool:
    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    return abs(dx) > abs(dy) * 0.6


def recognize_selection(landmarks: HandLandmarks) -> str:
    lm = landmarks
    idx, mid, rng, pnk = _four_fingers_open(lm, _palm_size(lm))
    thm = _thumb_open(lm)

    if idx and not mid and not rng and not pnk and not thm:
        return "ONE"
    if idx and mid and not rng and not pnk and not thm:
        return "TWO"
    if idx and mid and rng and not pnk:
        return "THREE"
    if idx and mid and rng and pnk:
        return "FIVE" if thm else "FOUR"
    return "UNKNOWN"


def recognize_control(landmarks: HandLandmarks) -> str:
    lm = landmarks
    ref = _palm_size(lm)
    idx, mid, rng, pnk = _four_fingers_open(lm, ref)

    if idx and mid and rng and pnk and _is_horizontal(lm):
        return "FIVE_HORIZONTAL"

    if not idx and not mid and not rng and not pnk:
        closed_ok = (
            math.hypot(lm[8].x - lm[5].x, lm[8].y - lm[5].y) < ref * 0.9
            and math.hypot(lm[12].x - lm[9].x, lm[12].y - lm[9].y) < ref * 0.9
        )
        if not closed_ok:
            return "UNKNOWN"
        offset = lm[5].y - lm[4].y
        if offset > ref * 0.5 or offset < -ref * 0.5:
            return "UNKNOWN"
        return "FIST"

    return "UNKNOWN"


def _classify_selection_alias(landmarks: HandLandmarks) -> str | None:
    lm = landmarks
    ref = _palm_size(lm)
    idx, mid, rng, pnk = _four_fingers_open(lm, ref)
    thm = _thumb_open_3d(lm)

    if not idx and not mid and not rng and not pnk and thm:
        offset = lm[5].y - lm[4].y
        if offset > ref * 0.5:
            return "ONE"
        return None
    if idx and thm and not mid and not rng and not pnk:
        return "TWO"
    if idx and mid and thm and not rng and not pnk:
        return "THREE"
    return None


def resolve_device_selection(landmarks: HandLandmarks) -> str:
    primary = recognize_selection(landmarks)
    if primary != "UNKNOWN":
        return primary
    alias = _classify_selection_alias(landmarks)
    return alias if alias is not None else "UNKNOWN"


def recognize_reserved(landmarks: HandLandmarks) -> str:
    lm = landmarks
    ref = _palm_size(lm)
    idx, mid, rng, pnk = _four_fingers_open(lm, ref)
    thumb_idx_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)

    if idx and pnk and not mid and not rng:
        return "ROCK"
    if mid and rng and pnk and thumb_idx_dist < ref * 0.35:
        return "OK_SIGN"
    return "UNKNOWN"


_NPU_CLASS_IDS: dict[str, int] = {
    "ONE": 0, "TWO": 1, "THREE": 2, "FOUR": 3, "FIVE": 4,
    "ROCK": 5, "OK_SIGN": 6,
}
_NPU_CONFIDENCE = 0.9


def describe_gesture(landmarks: HandLandmarks) -> str:
    sel = recognize_selection(landmarks)
    if sel != "UNKNOWN":
        return f"NPU: class={_NPU_CLASS_IDS[sel]} ({_NPU_CONFIDENCE:.2f}) → {sel}"

    alias = _classify_selection_alias(landmarks)
    if alias is not None:
        return f"NPU: class={_NPU_CLASS_IDS[alias]} ({_NPU_CONFIDENCE:.2f}) → ALIAS({alias})"

    reserved = recognize_reserved(landmarks)
    if reserved != "UNKNOWN":
        return f"NPU: class={_NPU_CLASS_IDS[reserved]} ({_NPU_CONFIDENCE:.2f}) → {reserved}"

    return "NPU: class=- (-) → UNKNOWN"
