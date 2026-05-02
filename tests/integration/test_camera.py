"""카메라 통합 테스트 — 실제 Picamera2 필요 (라즈베리파이에서 실행)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_camera_read():
    """카메라에서 프레임을 한 장 캡처하고 크기를 확인한다."""
    from cammand.io.camera import CameraReader

    cam = CameraReader()
    cam.start()
    try:
        frame = cam.read()
        assert frame is not None
        assert frame.ndim == 3
    finally:
        cam.stop()
