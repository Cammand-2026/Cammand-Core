"""gesture.recognizer 단위 테스트 — 하드웨어 불필요."""
from __future__ import annotations

import pytest

from cammand.engine.base import Landmark
from cammand.gesture.recognizer import (
    recognize_control,
    recognize_reserved,
    recognize_selection,
    resolve_device_selection,
)


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

    # 엄지 (닫힘: tip이 IP보다 손목에 더 가까움)
    base_list[3] = Landmark(0.42, 0.82, 0.0)
    base_list[4] = Landmark(0.46, 0.87, 0.0)

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


_THUMB_OPEN_TIP = Landmark(0.38, 0.78, 0.0)
_THUMB_OPEN_IP = Landmark(0.4, 0.8, 0.0)


class TestSelectionAliases:
    """별칭 제스처는 resolve_device_selection()에서만 인식된다."""

    def test_gun_aliases_to_two(self):
        # 총모양: 검지+엄지 펼침, 나머지 접힘 → TWO 별칭
        tip, pip = _open_finger()
        lm = _lm({8: tip, 6: pip, 3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP})
        assert resolve_device_selection(lm) == "TWO"

    def test_thumbs_up_aliases_to_one(self):
        # 엄지척(따봉): 엄지만 펼침(위 방향), 나머지 접힘 → ONE 별칭
        lm = _lm({3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP})
        assert resolve_device_selection(lm) == "ONE"

    def test_thumb_index_middle_aliases_to_three(self):
        # 엄지+검지+중지 펼침 → THREE 별칭
        t8, p6 = _open_finger()
        t12, p10 = _open_finger()
        lm = _lm({
            8: t8, 6: p6, 12: t12, 10: p10,
            3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP,
        })
        assert resolve_device_selection(lm) == "THREE"


class TestOnoffEntryExclusion:
    """ON/OFF 진입(recognize_selection() == "ONE")은 검지 단독 포즈만 허용."""

    def test_index_only_is_one(self):
        tip, pip = _open_finger()
        lm = _lm({8: tip, 6: pip})
        assert recognize_selection(lm) == "ONE"

    def test_thumbs_up_alias_is_not_one_in_primary(self):
        lm = _lm({3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP})
        assert recognize_selection(lm) != "ONE"

    def test_gun_is_not_one_in_primary(self):
        tip, pip = _open_finger()
        lm = _lm({8: tip, 6: pip, 3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP})
        assert recognize_selection(lm) != "ONE"


class TestRecognizeReserved:
    def test_fist_is_not_reserved(self):
        lm = _lm()
        assert recognize_reserved(lm) == "UNKNOWN"

    def test_one_is_not_reserved(self):
        tip, pip = _open_finger()
        lm = _lm({8: tip, 6: pip})
        assert recognize_reserved(lm) == "UNKNOWN"

    def test_rock(self):
        t8, p6 = _open_finger()
        t20, p18 = _open_finger()
        lm = _lm({8: t8, 6: p6, 20: t20, 18: p18})
        assert recognize_reserved(lm) == "ROCK"

    def test_rock_with_thumb_extended(self):
        # 엄지가 펴져 있어도(자연스러운 락앤롤 포즈) 인식되어야 함
        t8, p6 = _open_finger()
        t20, p18 = _open_finger()
        lm = _lm({
            8: t8, 6: p6, 20: t20, 18: p18,
            3: _THUMB_OPEN_IP, 4: _THUMB_OPEN_TIP,
        })
        assert recognize_reserved(lm) == "ROCK"

    def test_ok_sign(self):
        t12, p10 = _open_finger()
        t16, p14 = _open_finger()
        t20, p18 = _open_finger()
        lm = _lm({
            12: t12, 10: p10, 16: t16, 14: p14, 20: t20, 18: p18,
            4: Landmark(0.5, 0.68, 0.0),
            8: Landmark(0.5, 0.68, 0.0),
        })
        assert recognize_reserved(lm) == "OK_SIGN"


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
        assert result in ("FIST", "UNKNOWN")

    def test_five_horizontal_detected(self):
        # 4 손가락 모두 펼침(각 손가락 mcp-pip-tip 일직선) + 가로 방향
        lm = _lm({
            0:  Landmark(0.1, 0.5, 0.0),  # 손목 왼쪽
            5:  Landmark(0.3, 0.4, 0.0), 6:  Landmark(0.5, 0.4, 0.0), 8:  Landmark(0.7, 0.4, 0.0),
            9:  Landmark(0.3, 0.5, 0.0), 10: Landmark(0.5, 0.5, 0.0), 12: Landmark(0.7, 0.5, 0.0),
            13: Landmark(0.3, 0.6, 0.0), 14: Landmark(0.5, 0.6, 0.0), 16: Landmark(0.7, 0.6, 0.0),
            17: Landmark(0.3, 0.7, 0.0), 18: Landmark(0.5, 0.7, 0.0), 20: Landmark(0.7, 0.7, 0.0),
        })
        result = recognize_control(lm)
        assert result == "FIVE_HORIZONTAL"
