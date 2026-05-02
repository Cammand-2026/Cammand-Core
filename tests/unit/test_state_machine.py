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
