from __future__ import annotations

from .config import settings


def _progress_dots(elapsed: float, total: float) -> str:
    filled = min(3, int(elapsed / total * 3))
    return "●" * filled + "○" * (3 - filled)


def feedback_idle() -> str:
    return "손을 카메라에 보여주세요"


def feedback_selecting(
    device_name: str,
    elapsed: float,
    total: float | None = None,
) -> str:
    hold = total if total is not None else settings.selection_hold_sec
    return f"{device_name} 선택 중  {_progress_dots(elapsed, hold)}"


def feedback_selected(device_name: str) -> str:
    return f"{device_name}\n조작 제스처를 취해주세요"


def feedback_fist_countdown(elapsed: float) -> str:
    return f"기기선택 복귀 중  {_progress_dots(elapsed, settings.fist_hold_sec)}"


def feedback_power(device_name: str, state: str) -> str:
    label = "켜짐" if state == "on" else "꺼짐"
    return f"{device_name}  {label}"


def feedback_knob(device_name: str, value: float, unit: str) -> str:
    return f"{device_name} 조절 중  {value:.1f}{unit}"


def feedback_onoff_mode(device_name: str) -> str:
    return f"{device_name}\n원형=ON  도리도리=OFF"


def feedback_onoff_entering(elapsed: float) -> str:  # noqa: ARG001
    return "ON/OFF 모드 진입 중"
