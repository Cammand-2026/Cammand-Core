"""
Hailo-8L NPU 기반 제스처 엔진 — placeholder.

실제 구현은 Model 레포의 .hef 파일이 models/ 에 배치된 후 작성합니다.

설치 방법:
    # models/ 폴더에 .hef 파일 배치 후:
    pip install hailo-platform  # Hailo SDK 설치 필요
"""
from __future__ import annotations

import numpy as np

from .base import EngineResult, GestureEngine


class HailoEngine(GestureEngine):
    """
    Hailo-8L NPU 랜드마크 추출 엔진 (미구현).
    .env에서 GESTURE_ENGINE=mediapipe 설정을 사용하세요.
    """

    def __init__(self, hef_path: str) -> None:
        raise NotImplementedError(
            "HailoEngine은 아직 구현되지 않았습니다.\n"
            f"  hef_path={hef_path}\n"
            "  models/ 폴더에 .hef 파일을 배치하고 hailo-platform SDK를 설치하세요.\n"
            "  현재는 GESTURE_ENGINE=mediapipe 설정을 사용하세요."
        )

    def process(self, frame: np.ndarray) -> EngineResult: ...  # type: ignore[empty-body]

    def close(self) -> None: ...
