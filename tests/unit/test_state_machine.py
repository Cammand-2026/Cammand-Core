"""StateMachine 단위 테스트 — 하드웨어 불필요."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from cammand.state.machine import StateMachine
from cammand.state.phase import Phase


def _make_machine():
    on_feedback = MagicMock()
    on_power = MagicMock()
    on_knob = MagicMock()
    machine = StateMachine(on_feedback, on_power, on_knob)
    return machine, on_feedback, on_power, on_knob


def _lm_fist():
    from cammand.engine.base import Landmark
    base = [Landmark(0.5, 0.9, 0.0)] * 21
    base[9] = Landmark(0.5, 0.7, 0.0)
    for pip in (6, 10, 14, 18):
        base[pip] = Landmark(0.5, 0.6, 0.0)
    for tip in (8, 12, 16, 20):
        base[tip] = Landmark(0.5, 0.75, 0.0)
    base[3] = Landmark(0.42, 0.82, 0.0)
    base[4] = Landmark(0.46, 0.87, 0.0)
    return base


class TestStateMachineTimeout:
    def test_timeout_returns_to_idle(self):
        machine, on_feedback, _, _ = _make_machine()
        # 강제로 SELECTING 상태 진입
        machine._s.phase = Phase.SELECTING
        machine._s.last_hand_time = time.monotonic() - 10.0  # 10초 전

        asyncio.run(machine.update(None, 640, time.monotonic()))

        assert machine._s.phase == Phase.IDLE

    def test_idle_stays_idle_on_no_hand(self):
        machine, _, _, _ = _make_machine()
        asyncio.run(machine.update(None, 640, time.monotonic()))
        assert machine._s.phase == Phase.IDLE


class TestStateMachineFeedback:
    def test_idle_feedback_deduplicated(self):
        machine, on_feedback, _, _ = _make_machine()
        # 같은 피드백 두 번 → 한 번만 발행
        asyncio.run(machine.update(None, 640, time.monotonic()))
        asyncio.run(machine.update(None, 640, time.monotonic()))
        assert on_feedback.call_count == 1


class TestStateMachineNpuDebug:
    def test_npu_debug_deduplicated(self):
        on_feedback = MagicMock()
        on_power = MagicMock()
        on_knob = MagicMock()
        on_npu_debug = MagicMock()
        machine = StateMachine(on_feedback, on_power, on_knob, on_npu_debug)

        lm = _lm_fist()
        asyncio.run(machine.update(lm, 640, time.monotonic()))
        asyncio.run(machine.update(lm, 640, time.monotonic()))
        assert on_npu_debug.call_count == 1


def _lm_rock():
    from cammand.engine.base import Landmark
    lm = _lm_fist()
    lm[8] = Landmark(0.5, 0.45, 0.0)   # 검지 tip 펼침
    lm[6] = Landmark(0.5, 0.6, 0.0)
    lm[20] = Landmark(0.5, 0.45, 0.0)  # 새끼 tip 펼침
    lm[18] = Landmark(0.5, 0.6, 0.0)
    return lm


class TestStateMachineReservedFeedback:
    def test_idle_rock_publishes_recognized_feedback(self):
        machine, on_feedback, _, _ = _make_machine()
        asyncio.run(machine.update(_lm_rock(), 640, time.monotonic()))
        on_feedback.assert_called_with("락앤롤 인식됨")

    def test_idle_reverts_to_default_after_rock(self):
        machine, on_feedback, _, _ = _make_machine()
        asyncio.run(machine.update(_lm_rock(), 640, time.monotonic()))
        asyncio.run(machine.update(_lm_fist(), 640, time.monotonic()))
        on_feedback.assert_called_with("손을 카메라에 보여주세요")
