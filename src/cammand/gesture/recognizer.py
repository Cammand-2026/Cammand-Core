from __future__ import annotations

import math

from ..engine.base import HandLandmarks


def _finger_open(lm: HandLandmarks, tip: int, pip: int, ref: float) -> bool:
    """
    손가락 펼침 여부. 3단계 판별:
    [1] z 체크: 손바닥 방향에서 tip이 카메라 쪽으로 말리면 즉시 닫힘
    [2] y 체크: tip이 pip보다 위에 있으면 펼침 (기본 케이스)
    [3] 손목 거리 체크: tip이 손목 기준 pip보다 멀면 펼침
        - distance(tip, wrist) > distance(pip, wrist): 방향 인식 → 손목으로
          내려간 tip을 정확히 닫힘으로 판정
    """
    # [1] 손바닥 방향 닫힘: tip.z << pip.z
    if lm[tip].z < lm[pip].z - ref * 0.3:
        return False
    # [2] y 비교
    y_open = lm[tip].y < lm[pip].y
    # [3] 손목 기준 거리 비교
    d_tip_wrist = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
    d_pip_wrist = math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)
    d_open = d_tip_wrist > d_pip_wrist
    return y_open or d_open


def _thumb_open(lm: HandLandmarks) -> bool:
    """엄지 끝(4)이 IP 관절(3)보다 손목(0)에서 더 멀리 있으면 펼침."""
    d_tip = math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y)
    d_ip = math.hypot(lm[3].x - lm[0].x, lm[3].y - lm[0].y)
    return d_tip > d_ip * 1.15


def _is_horizontal(lm: HandLandmarks) -> bool:
    """손목(0)→중지MCP(9) 벡터 기준 가로 방향 여부. ±59° 이내면 가로로 판단."""
    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    return abs(dx) > abs(dy) * 0.6


def recognize_selection(landmarks: HandLandmarks) -> str:
    """
    손가락 1~5개를 명확히 구분.
    반환값: ONE | TWO | THREE | FOUR | FIVE | UNKNOWN

    - FIVE_HORIZONTAL·THUMB 계열은 절대 반환하지 않음
      (SELECTING 중 가습기(FIVE) 선택이 노브 제스처로 오인되는 문제 방지)
    """
    lm = landmarks
    ref = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)

    idx = _finger_open(lm, 8,  6,  ref)
    mid = _finger_open(lm, 12, 10, ref)
    rng = _finger_open(lm, 16, 14, ref)
    pnk = _finger_open(lm, 20, 18, ref)
    thm = _thumb_open(lm)

    if idx and not mid and not rng and not pnk:
        return "ONE"
    if idx and mid and not rng and not pnk:
        return "TWO"
    if idx and mid and rng and not pnk:
        return "THREE"
    if idx and mid and rng and pnk:
        return "FIVE" if thm else "FOUR"
    return "UNKNOWN"


def recognize_control(landmarks: HandLandmarks) -> str:
    """
    선택된 기기를 조작하는 제스처를 구분.
    반환값: THUMB_UP | THUMB_DOWN | FIVE_HORIZONTAL | FIST | UNKNOWN

    - 선택 제스처(ONE~FIVE)는 절대 반환하지 않음
    - FIVE_HORIZONTAL: 4개 손가락 모두 펼침 + 가로 방향
    - FIST / THUMB_UP / THUMB_DOWN: 4개 손가락 모두 닫힘 + 엄지 방향
    """
    lm = landmarks
    ref = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)

    idx = _finger_open(lm, 8,  6,  ref)
    mid = _finger_open(lm, 12, 10, ref)
    rng = _finger_open(lm, 16, 14, ref)
    pnk = _finger_open(lm, 20, 18, ref)

    # 노브 모드: 4개 손가락 모두 펼침 + 가로 방향
    if idx and mid and rng and pnk and _is_horizontal(lm):
        return "FIVE_HORIZONTAL"

    # 주먹 계열: 4개 손가락 모두 닫힘
    if not idx and not mid and not rng and not pnk:
        closed_ok = (
            math.hypot(lm[8].x - lm[5].x, lm[8].y - lm[5].y) < ref * 0.9
            and math.hypot(lm[12].x - lm[9].x, lm[12].y - lm[9].y) < ref * 0.9
        )
        if not closed_ok:
            return "UNKNOWN"
        # y축은 아래가 양수이므로, offset > 0 이면 엄지가 위로 향함 → THUMB_UP
        offset = lm[5].y - lm[4].y
        if offset > ref * 0.5:
            return "THUMB_UP"
        if offset < -ref * 0.5:
            return "THUMB_DOWN"
        return "FIST"

    return "UNKNOWN"
