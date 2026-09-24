from __future__ import annotations

import math


class TrajectoryTracker:
    BUFFER_LEN: int = 35
    MIN_RADIUS: float = 0.08
    MIN_CIRCLE_PTS: int = 20
    MIN_SHAKE_PTS: int = 10
    SHAKE_AMP: float = 0.08
    YX_THRESHOLD: float = 0.4

    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []

    def add(self, x: float, y: float) -> None:
        self.points.append((x, y))
        if len(self.points) > self.BUFFER_LEN:
            self.points.pop(0)

    def reset(self) -> None:
        self.points.clear()

    def _yx_ratio(self) -> float:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return y_range / x_range if x_range > 1e-6 else 0.0

    def check_circle(self) -> bool:
        if len(self.points) < self.MIN_CIRCLE_PTS:
            return False
        if self._yx_ratio() < self.YX_THRESHOLD:
            return False
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radii = [math.hypot(x - cx, y - cy) for x, y in self.points]
        if sum(radii) / len(radii) < self.MIN_RADIUS:
            return False
        total_angle = 0.0
        for i in range(1, len(self.points)):
            a0 = math.atan2(self.points[i - 1][1] - cy, self.points[i - 1][0] - cx)
            a1 = math.atan2(self.points[i][1] - cy, self.points[i][0] - cx)
            da = a1 - a0
            if da > math.pi:
                da -= 2 * math.pi
            if da < -math.pi:
                da += 2 * math.pi
            total_angle += da
        return abs(total_angle) >= 2 * math.pi

    def check_shake(self) -> bool:
        if len(self.points) < self.MIN_SHAKE_PTS:
            return False
        if self._yx_ratio() >= self.YX_THRESHOLD:
            return False
        xs = [p[0] for p in self.points]
        reversals = 0
        last_extreme_x = xs[0]
        direction = 0
        for x in xs[1:]:
            delta = x - last_extreme_x
            if direction == 0:
                if abs(delta) >= self.SHAKE_AMP:
                    direction = 1 if delta > 0 else -1
                    last_extreme_x = x
            elif direction == 1 and delta < -self.SHAKE_AMP:
                reversals += 1
                direction = -1
                last_extreme_x = x
            elif direction == -1 and delta > self.SHAKE_AMP:
                reversals += 1
                direction = 1
                last_extreme_x = x
        return reversals >= 2
