from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    """정규화된 3D 랜드마크 좌표 (0.0~1.0)."""
    x: float
    y: float
    z: float


# MediaPipe 21개 랜드마크 인덱스와 동일한 순서 보장
HandLandmarks = list[Landmark]


@dataclass
class EngineResult:
    """엔진 한 프레임 처리 결과."""
    hand_detected: bool
    landmarks: HandLandmarks | None  # None이면 hand_detected=False
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    npu_debug: str | None = None  # HailoEngine만 채움: "[NPU] starfish (82.1%) → FIVE"


class GestureEngine(ABC):
    """
    제스처 엔진 인터페이스 (MediaPipeEngine, HailoEngine).

    계약:
      - process()는 항상 EngineResult를 반환 (예외 전파하지 않음)
      - landmarks가 존재할 경우 항상 len == 21
      - 좌표는 항상 0.0~1.0으로 정규화됨
    """

    @abstractmethod
    def process(self, frame: np.ndarray) -> EngineResult:
        """단일 프레임에서 랜드마크 추출."""
        ...

    @abstractmethod
    def close(self) -> None:
        """리소스 해제."""
        ...

    def __enter__(self) -> GestureEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
