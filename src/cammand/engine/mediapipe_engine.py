from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .base import EngineResult, GestureEngine, Landmark


class MediaPipeEngine(GestureEngine):
    """MediaPipe Hands 기반 CPU 제스처 엔진."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> EngineResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            self._drawing.draw_landmarks(annotated, hl, self._mp_hands.HAND_CONNECTIONS)
            landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in hl.landmark]
            return EngineResult(True, landmarks, frame, annotated)

        return EngineResult(False, None, frame, annotated)

    def close(self) -> None:
        self._hands.close()
