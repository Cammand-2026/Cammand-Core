"""gesture.recognizer 단위 테스트 — 하드웨어 불필요."""
from __future__ import annotations

import pytest

from cammand.engine.base import Landmark
from cammand.gesture.recognizer import recognize_control, recognize_selection


def _lm(overrides: dict[int, Landmark] | None = None) -> list[Landmark]:
    """
    기본 손 랜드마크 21개 생성 (모든 손가락 닫힌 주먹 형태).
    overrides: {index: Landmark} 로 특정 랜드마크 교체.
    """
    base_list: list[Landmark] = [Landmark(0.5, 0.9, 0.0)] * 21
    base_list[0] = Landmark(0.5, 0.9, 0.0)   # 손목
    base_list[9] = Landmark(0.5, 0.7, 0.0)   # 중지 MCP (손바닥)

    # 각 손가락 PIP (중간 관절) — 손목보다 위
    for pip in [6, 10, 14, 18]:
        base_list[pip] = Landmark(0.5, 0.6, 0.0)

    # 각 손가락 TIP — PIP보다 아래 (닫힌 상태)
    for tip in [8, 12, 16, 20]:
        base_list[tip] = Landmark(0.5, 0.75, 0.0)

    # 엄지
    base_list[3] = Landmark(0.4, 0.8, 0.0)
    base_list[4] = Landmark(0.38, 0.78, 0.0)

    if overrides:
        for idx, lm in overrides.items():
            base_list[idx] = lm
    return base_list


def _open_finger(pip_y: float = 0.6) -> tuple[Landmark, Landmark]:
    """tip이 pip보다 위에 있는 펼친 손가락 반환 (tip, pip)."""
    return Landmark(0.5, pip_y - 0.15, 0.0), Landmark(0.5, pip_y, 0.0)


class TestRecognizeSelection:
    def test_unknown_fist(self):
        lm = _lm()
        assert recognize_selection(lm) == "UNKNOWN"

    def test_one_index_only(self):
        tip, pip = _open_finger()
        lm = _lm({8: tip, 6: pip})
        result = recognize_selection(lm)
        assert result == "ONE"

    def test_two_index_middle(self):
        t8, p6 = _open_finger()
        t12, p10 = _open_finger()
        lm = _lm({8: t8, 6: p6, 12: t12, 10: p10})
        result = recognize_selection(lm)
        assert result == "TWO"


class TestRecognizeControl:
    def test_fist_returns_fist(self):
        # 4 손가락 완전 닫힘 + 엄지 수평 → 주먹 계열
        lm = _lm({
            8: Landmark(0.5, 0.65, 0.0),
            5: Landmark(0.5, 0.65, 0.0),
            12: Landmark(0.5, 0.65, 0.0),
            9: Landmark(0.5, 0.65, 0.0),
        })
        result = recognize_control(lm)
        assert result in ("FIST", "THUMB_UP", "THUMB_DOWN", "UNKNOWN")

    def test_five_horizontal_detected(self):
        # 4 손가락 모두 펼침 + 가로 방향
        lm = _lm({
            0:  Landmark(0.2, 0.5, 0.0),  # 손목 왼쪽
            9:  Landmark(0.8, 0.5, 0.0),  # 중지 MCP 오른쪽 → 가로 방향
            8:  Landmark(0.5, 0.3, 0.0),  # 검지 tip 펼침
            6:  Landmark(0.5, 0.5, 0.0),
            12: Landmark(0.5, 0.3, 0.0),
            10: Landmark(0.5, 0.5, 0.0),
            16: Landmark(0.5, 0.3, 0.0),
            14: Landmark(0.5, 0.5, 0.0),
            20: Landmark(0.5, 0.3, 0.0),
            18: Landmark(0.5, 0.5, 0.0),
        })
        result = recognize_control(lm)
        assert result == "FIVE_HORIZONTAL"
