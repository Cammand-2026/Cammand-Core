from __future__ import annotations

import cv2
import numpy as np
from picamera2 import Picamera2

from ..config import settings


class CameraReader:
    """Picamera2 래퍼. 설정에 따라 프레임을 캡처하고 회전한다."""

    def __init__(self) -> None:
        self._cam = Picamera2()
        self._cam.configure(
            self._cam.create_video_configuration(
                main={
                    "size": (settings.camera_width, settings.camera_height),
                    "format": "RGB888",
                }
            )
        )

    def start(self) -> None:
        self._cam.start()

    def read(self) -> np.ndarray:
        frame = self._cam.capture_array()
        if settings.camera_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def stop(self) -> None:
        self._cam.stop()
