from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


HandLandmarks = list[Landmark]


@dataclass
class EngineResult:
    hand_detected: bool
    landmarks: HandLandmarks | None
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    npu_debug: str | None = None


class GestureEngine(ABC):

    @abstractmethod
    def process(self, frame: np.ndarray) -> EngineResult:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> GestureEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
