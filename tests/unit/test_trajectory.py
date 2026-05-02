"""TrajectoryTracker 단위 테스트 — 하드웨어 불필요."""
from __future__ import annotations

import math

import pytest

from cammand.gesture.trajectory import TrajectoryTracker


def _make_circle(n: int = 25, r: float = 0.15) -> list[tuple[float, float]]:
    """반시계 방향 원형 궤적 생성 (n+2점 → 360도 초과 보장)."""
    return [
        (0.5 + r * math.cos(2 * math.pi * i / n),
         0.5 + r * math.sin(2 * math.pi * i / n))
        for i in range(n + 2)  # 360도를 넘어 부동소수점 오차 여유 확보
    ]


def _make_shake(n_reversals: int = 3, amp: float = 0.12) -> list[tuple[float, float]]:
    """좌우 도리도리 궤적 생성."""
    points = []
    x = 0.5
    direction = 1
    step = amp / 5
    for _ in range(n_reversals):
        for _ in range(5):
            x += direction * step
            points.append((x, 0.5))
        direction *= -1
    return points


class TestTrajectoryTracker:
    def test_circle_detected(self):
        # n=25로 25+1=26점 → BUFFER_LEN(35) 안에서 1.04바퀴 완주
        t = TrajectoryTracker()
        for x, y in _make_circle(25):
            t.add(x, y)
        assert t.check_circle()

    def test_shake_detected(self):
        t = TrajectoryTracker()
        for x, y in _make_shake(3):
            t.add(x, y)
        assert t.check_shake()

    def test_circle_not_shake(self):
        t = TrajectoryTracker()
        for x, y in _make_circle(30):
            t.add(x, y)
        assert not t.check_shake()

    def test_shake_not_circle(self):
        t = TrajectoryTracker()
        for x, y in _make_shake(3):
            t.add(x, y)
        assert not t.check_circle()

    def test_empty_returns_false(self):
        t = TrajectoryTracker()
        assert not t.check_circle()
        assert not t.check_shake()

    def test_reset_clears(self):
        t = TrajectoryTracker()
        for x, y in _make_circle(30):
            t.add(x, y)
        t.reset()
        assert not t.check_circle()

    def test_buffer_limit(self):
        t = TrajectoryTracker()
        for i in range(100):
            t.add(float(i) / 100, 0.5)
        assert len(t.points) == TrajectoryTracker.BUFFER_LEN
