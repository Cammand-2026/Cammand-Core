from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import settings
from ..devices import DEVICES, DeviceConfig
from ..engine.base import HandLandmarks
from ..feedback import (
    feedback_fist_countdown,
    feedback_idle,
    feedback_knob,
    feedback_onoff_entering,
    feedback_onoff_mode,
    feedback_power,
    feedback_selected,
    feedback_selecting,
)
from ..gesture.recognizer import recognize_control, recognize_selection
from ..gesture.trajectory import TrajectoryTracker
from .phase import Phase


@dataclass
class _State:
    """상태 머신의 모든 내부 상태를 한 객체에 집약."""

    phase: Phase = Phase.IDLE
    selected_device: DeviceConfig | None = None
    selected_gesture: str = ""

    # SELECTING 단계 타이머
    sel_candidate: str = ""
    sel_start: float = 0.0

    # SELECTED 단계 — 기기 전환 타이머
    switch_candidate: str = ""
    switch_start: float = 0.0

    # SELECTED 단계 — 조작 제스처 타이머
    five_h_start: float = 0.0
    fist_start: float = 0.0
    onoff_entry_start: float = 0.0

    # ONOFF_CONTROL
    trajectory: TrajectoryTracker = field(default_factory=TrajectoryTracker)

    # CONTROLLING
    knob_start_y: float = 0.0
    knob_init_val: float = 0.0
    knob_last_pub: float = 0.0
    knob_values: dict[str, float] = field(default_factory=dict)

    # power=on 후 노브 복원 지연 (HA 충돌 방지)
    pending_restore: dict[str, tuple[float, float]] = field(default_factory=dict)

    last_hand_time: float = field(default_factory=time.monotonic)


def _knob_fmt(value: float, step: float) -> str:
    return f"{value:.0f}" if step == int(step) else f"{value:.1f}"


class StateMachine:
    """
    제스처 상태 머신.

    MQTT를 직접 import하지 않고 콜백으로 I/O를 주입받아
    단위 테스트에서 mock 콜백으로 동작 검증이 가능합니다.
    """

    def __init__(
        self,
        on_feedback: Callable[[str], None],
        on_power: Callable[[str, str], None],
        on_knob: Callable[[str, str], None],
    ) -> None:
        self._s = _State()
        self._cfg = settings
        self._on_feedback = on_feedback
        self._on_power = on_power
        self._on_knob = on_knob
        self._prev_feedback: str = ""

    def _publish_feedback(self, text: str) -> None:
        if text != self._prev_feedback:
            self._on_feedback(text)
            self._prev_feedback = text

    def _go_idle(self) -> None:
        s = self._s
        s.phase = Phase.IDLE
        s.selected_device = None
        s.selected_gesture = ""
        s.sel_candidate = ""
        s.sel_start = 0.0
        s.switch_candidate = ""
        s.switch_start = 0.0
        s.five_h_start = 0.0
        s.fist_start = 0.0
        s.onoff_entry_start = 0.0
        s.trajectory.reset()
        self._publish_feedback(feedback_idle())

    async def update(
        self,
        landmarks: HandLandmarks | None,
        frame_height: int,
        now: float,
    ) -> None:
        """한 프레임 처리. landmarks=None이면 손 미감지로 처리."""
        s = self._s

        # 노브 복원 대기 처리
        for did, (val, after) in list(s.pending_restore.items()):
            if now >= after:
                dev = next(d for d in DEVICES.values() if d.id == did)
                self._on_knob(did, _knob_fmt(val, dev.knob_step))
                del s.pending_restore[did]

        # 타임아웃
        hand_detected = landmarks is not None
        if hand_detected:
            s.last_hand_time = now
        elif now - s.last_hand_time > self._cfg.timeout_sec and s.phase != Phase.IDLE:
            self._go_idle()

        if not hand_detected:
            if s.phase == Phase.IDLE:
                self._publish_feedback(feedback_idle())
            return

        lm = landmarks
        wrist_y = lm[0].y * frame_height  # 픽셀 단위

        if s.phase == Phase.IDLE:
            self._handle_idle(lm, now)
        elif s.phase == Phase.SELECTING:
            self._handle_selecting(lm, now)
        elif s.phase == Phase.SELECTED:
            self._handle_selected(lm, now, wrist_y)
        elif s.phase == Phase.ONOFF_CONTROL:
            self._handle_onoff_control(lm, now, wrist_y)
        elif s.phase == Phase.CONTROLLING:
            self._handle_controlling(lm, now, wrist_y, frame_height)

    # ── 각 Phase 핸들러 ──────────────────────────────────────────────────────

    def _handle_idle(self, lm: HandLandmarks, now: float) -> None:
        s = self._s
        gesture = recognize_selection(lm)
        if gesture in DEVICES:
            s.phase = Phase.SELECTING
            s.sel_candidate = gesture
            s.sel_start = now
            self._publish_feedback(feedback_selecting(DEVICES[gesture].name, 0.0))

    def _handle_selecting(self, lm: HandLandmarks, now: float) -> None:
        s = self._s
        gesture = recognize_selection(lm)
        if gesture != s.sel_candidate:
            if gesture in DEVICES:
                s.sel_candidate = gesture
                s.sel_start = now
                self._publish_feedback(feedback_selecting(DEVICES[gesture].name, 0.0))
            else:
                self._go_idle()
        else:
            elapsed = now - s.sel_start
            self._publish_feedback(feedback_selecting(DEVICES[gesture].name, elapsed))
            if elapsed >= self._cfg.selection_hold_sec:
                s.phase = Phase.SELECTED
                s.selected_device = DEVICES[gesture]
                s.selected_gesture = gesture
                s.sel_candidate = ""
                s.sel_start = 0.0
                self._publish_feedback(feedback_selected(s.selected_device.name))

    def _handle_selected(self, lm: HandLandmarks, now: float, wrist_y: float) -> None:
        s = self._s
        ctrl = recognize_control(lm)
        sel = recognize_selection(lm)

        if sel == "ONE":
            # ONE(검지) 유지 → ONOFF_CONTROL 진입
            s.fist_start = 0.0
            s.five_h_start = 0.0
            s.switch_candidate = ""
            s.switch_start = 0.0
            if s.onoff_entry_start == 0.0:
                s.onoff_entry_start = now
            elapsed = now - s.onoff_entry_start
            if elapsed >= self._cfg.onoff_entry_sec:
                s.phase = Phase.ONOFF_CONTROL
                s.trajectory.reset()
                s.onoff_entry_start = 0.0
                self._publish_feedback(feedback_onoff_mode(s.selected_device.name))
            else:
                self._publish_feedback(feedback_onoff_entering(elapsed))

        elif ctrl == "FIVE_HORIZONTAL":
            s.onoff_entry_start = 0.0
            s.fist_start = 0.0
            s.switch_candidate = ""
            s.switch_start = 0.0
            if s.five_h_start == 0.0:
                s.five_h_start = now
            elif now - s.five_h_start >= self._cfg.five_h_confirm_sec:
                dev = s.selected_device
                s.phase = Phase.CONTROLLING
                s.knob_start_y = wrist_y
                s.knob_init_val = s.knob_values.get(
                    dev.id, (dev.knob_min + dev.knob_max) / 2.0
                )
                s.knob_last_pub = 0.0
                s.five_h_start = 0.0

        elif ctrl == "FIST":
            s.onoff_entry_start = 0.0
            s.five_h_start = 0.0
            s.switch_candidate = ""
            s.switch_start = 0.0
            if s.fist_start == 0.0:
                s.fist_start = now
            elapsed = now - s.fist_start
            if elapsed >= self._cfg.fist_hold_sec:
                self._go_idle()
            else:
                self._publish_feedback(feedback_fist_countdown(elapsed))

        elif sel in DEVICES and sel != s.selected_gesture and sel != "ONE":
            # 기기 전환 후보 — SWITCH_BUFFER_SEC 이후 카운트다운 시작
            s.onoff_entry_start = 0.0
            s.five_h_start = 0.0
            s.fist_start = 0.0
            if sel != s.switch_candidate:
                s.switch_candidate = sel
                s.switch_start = now
            else:
                elapsed = now - s.switch_start
                buffered = elapsed - self._cfg.switch_buffer_sec
                if buffered >= self._cfg.switch_hold_sec:
                    s.selected_device = DEVICES[sel]
                    s.selected_gesture = sel
                    s.switch_candidate = ""
                    s.switch_start = 0.0
                    self._publish_feedback(feedback_selected(s.selected_device.name))
                elif buffered > 0:
                    self._publish_feedback(
                        feedback_selecting(DEVICES[sel].name, buffered, self._cfg.switch_hold_sec)
                    )

        else:
            # UNKNOWN 또는 동일 기기 → 모든 타이머 리셋
            s.onoff_entry_start = 0.0
            s.five_h_start = 0.0
            s.fist_start = 0.0
            s.switch_candidate = ""
            s.switch_start = 0.0

    def _handle_onoff_control(
        self, lm: HandLandmarks, now: float, wrist_y: float
    ) -> None:
        s = self._s
        ctrl = recognize_control(lm)

        if ctrl == "FIVE_HORIZONTAL":
            s.trajectory.reset()
            s.fist_start = 0.0
            if s.five_h_start == 0.0:
                s.five_h_start = now
            elif now - s.five_h_start >= self._cfg.five_h_confirm_sec:
                dev = s.selected_device
                s.phase = Phase.CONTROLLING
                s.knob_start_y = wrist_y
                s.knob_init_val = s.knob_values.get(
                    dev.id, (dev.knob_min + dev.knob_max) / 2.0
                )
                s.knob_last_pub = 0.0
                s.five_h_start = 0.0

        elif ctrl == "FIST":
            s.trajectory.reset()
            s.five_h_start = 0.0
            if s.fist_start == 0.0:
                s.fist_start = now
            elapsed = now - s.fist_start
            if elapsed >= self._cfg.fist_hold_sec:
                self._go_idle()
            else:
                self._publish_feedback(feedback_fist_countdown(elapsed))

        else:
            s.fist_start = 0.0
            s.five_h_start = 0.0
            s.trajectory.add(lm[8].x, lm[8].y)

            did = s.selected_device.id
            if s.trajectory.check_circle():
                self._on_power(did, "on")
                if did in s.knob_values:
                    s.pending_restore[did] = (s.knob_values[did], now + 0.3)
                self._publish_feedback(feedback_power(s.selected_device.name, "on"))
                s.trajectory.reset()

            elif s.trajectory.check_shake():
                dev = s.selected_device
                self._on_power(did, "off")
                self._on_knob(did, _knob_fmt(dev.knob_off, dev.knob_step))
                self._publish_feedback(feedback_power(dev.name, "off"))
                s.trajectory.reset()

            else:
                self._publish_feedback(feedback_onoff_mode(s.selected_device.name))

    def _handle_controlling(
        self, lm: HandLandmarks, now: float, wrist_y: float, frame_height: int
    ) -> None:
        s = self._s
        ctrl = recognize_control(lm)

        if ctrl == "FIVE_HORIZONTAL":
            dev = s.selected_device
            delta_y = s.knob_start_y - wrist_y  # 위로 올리면 양수
            ratio = max(-1.0, min(1.0, delta_y / (frame_height / 2.0)))
            knob_range = dev.knob_max - dev.knob_min
            raw_val = s.knob_init_val + ratio * knob_range
            new_val = round(raw_val / dev.knob_step) * dev.knob_step
            new_val = max(dev.knob_min, min(dev.knob_max, new_val))

            if now - s.knob_last_pub >= 1.0 / self._cfg.knob_publish_hz:
                self._on_knob(dev.id, _knob_fmt(new_val, dev.knob_step))
                s.knob_values[dev.id] = new_val
                self._publish_feedback(feedback_knob(dev.name, new_val, dev.unit))
                s.knob_last_pub = now
        else:
            # FIVE_HORIZONTAL 해제 → SELECTED 복귀
            s.phase = Phase.SELECTED
            s.five_h_start = 0.0
            self._publish_feedback(feedback_selected(s.selected_device.name))
