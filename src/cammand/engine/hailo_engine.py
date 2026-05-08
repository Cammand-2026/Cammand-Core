"""
Hailo-8L NPU 기반 제스처 엔진.

현재: mobilenetv2-12.hef stand-in으로 파이프라인 인프라 검증용.
실제 MLP .hef 교체 시 전처리 로직(_preprocess)과 레이블 매핑만 수정.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .base import EngineResult, GestureEngine, Landmark
from .imagenet_classes import IMAGENET_CLASSES

# ImageNet 정규화 상수
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_STATIC_LABELS = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]

# ImageNet 클래스 중 숫자/손과 의미론적으로 연결되는 클래스 오버라이드
# (MobileNetV2 stand-in 전용 — 실제 MLP 전환 시 삭제)
_SEMANTIC_OVERRIDES: dict[int, str] = {
    328: "FIVE",  # starfish (팔 5개 — 펼친 손 연상)
    523: "ONE",   # crutch (단일 막대형 — 검지 1개 연상)
    733: "ONE",   # punching bag (주먹)
    794: "TWO",   # scissors (날 2개 — V자 제스처)
}


def _class_to_gesture(class_id: int) -> str:
    """ImageNet class_id → ONE~FIVE 매핑 (시맨틱 오버라이드 + 균등 range fallback)."""
    return _SEMANTIC_OVERRIDES.get(class_id, _STATIC_LABELS[min(class_id // 200, 4)])


class HailoEngine(GestureEngine):
    """
    CPU(MediaPipe 랜드마크 추출) + NPU(Hailo-8L 이미지 분류) 혼합 엔진.

    - MediaPipe: 기존 룰베이스 상태 머신용 21개 랜드마크 추출
    - Hailo NPU: 카메라 프레임 → HEF 추론 → npu_debug 문자열 생성
    """

    def __init__(self, hef_path: str) -> None:
        # ── MediaPipe 초기화 (MediaPipeEngine과 동일) ─────────────────────────
        _mp = mp.solutions.hands
        self._mp_hands = _mp
        self._drawing = mp.solutions.drawing_utils
        self._hands = _mp.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # ── Hailo VDevice + HEF 로드 ──────────────────────────────────────────
        try:
            from hailo_platform import (  # type: ignore[import]
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform 패키지를 찾을 수 없습니다.\n"
                "  sudo apt install hailo-all  후 재시도하세요."
            ) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)

        infer_model = self._vdevice.create_infer_model(hef_path)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        self._input_shape  = tuple(infer_model.input().shape)   # e.g. (224, 224, 3)
        self._output_shape = tuple(infer_model.output().shape)  # e.g. (1000,)
        self._input_name   = infer_model.input().name
        self._output_name  = infer_model.output().name

        # configure()는 컨텍스트 매니저 — __enter__ 직접 호출해 수명 관리
        self._configured_ctx = infer_model.configure()
        self._configured = self._configured_ctx.__enter__()

        # 추론 버퍼 사전 할당 (매 프레임 재사용)
        self._in_buf  = np.zeros(self._input_shape,  dtype=np.float32)
        self._out_buf = np.zeros(self._output_shape, dtype=np.float32)
        self._bindings = self._configured.create_bindings(
            input_buffers  = {self._input_name:  self._in_buf},
            output_buffers = {self._output_name: self._out_buf},
        )

        print(f">> HailoEngine: {hef_path} 로드 완료")
        print(f">> 입력: {self._input_shape}  출력: {self._output_shape}")

    def _preprocess(self, frame: np.ndarray) -> None:
        """카메라 프레임 → self._in_buf (ImageNet 전처리, in-place)."""
        h, w = self._input_shape[0], self._input_shape[1]
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _MEAN) / _STD
        np.copyto(self._in_buf, normalized)

    def process(self, frame: np.ndarray) -> EngineResult:
        # ── MediaPipe 랜드마크 추출 ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        landmarks = None
        if mp_result.multi_hand_landmarks:
            hl = mp_result.multi_hand_landmarks[0]
            self._drawing.draw_landmarks(
                annotated, hl, self._mp_hands.HAND_CONNECTIONS
            )
            landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in hl.landmark]

        # ── Hailo NPU 추론 ────────────────────────────────────────────────────
        self._preprocess(frame)
        self._configured.run([self._bindings], 1000)  # timeout 1000 ms

        out = self._bindings.output().get_buffer()
        class_id = int(np.argmax(out))

        # 로짓 → softmax 확률 (overflow 방지)
        probs = np.exp(out - out.max())
        probs /= probs.sum()
        conf = float(probs[class_id]) * 100

        class_name = IMAGENET_CLASSES[class_id] if class_id < len(IMAGENET_CLASSES) else f"class_{class_id}"
        gesture    = _class_to_gesture(class_id)

        npu_debug = f"[NPU] {class_name} ({conf:.1f}%) → {gesture}"

        return EngineResult(
            hand_detected  = landmarks is not None,
            landmarks      = landmarks,
            raw_frame      = frame,
            annotated_frame= annotated,
            npu_debug      = npu_debug,
        )

    def close(self) -> None:
        self._hands.close()
        self._configured_ctx.__exit__(None, None, None)
        self._vdevice.release()
