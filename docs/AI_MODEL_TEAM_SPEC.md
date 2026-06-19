# Cammand AI 모델팀 인수인계 문서

> **작성일**: 2026-05-08  
> **최종 수정**: 2026-06-19  
> **수신**: AI 모델팀 (제스처 분류 MLP 개발 담당)  
> **발신**: HW/SW 담당 (백승찬)

---

## 1. 프로젝트 개요

Cammand는 라즈베리파이5 위에서 동작하는 제스처 인식 스마트홈 허브입니다.  
카메라 앞에서 손 제스처만으로 조명, 선풍기, 에어컨, 가습기를 제어합니다.

**현재 개발 단계**: CPU 룰베이스 MVP 완성 + Hailo NPU 파이프라인 인프라 검증 완료  
**다음 단계**: AI팀이 개발한 MLP `.pt` 모델 수령 → HW/SW팀이 `.hef`로 변환 → HailoEngine에 연결

---

## 2. 시스템 아키텍처

```
카메라 (Picamera2, 640×360)
        │
        ▼
[CPU] MediaPipe Hands
  21개 관절 좌표 추출 (x, y, z, 각 0.0~1.0 정규화)
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
[CPU] 룰베이스 상태 머신                    [NPU] Hailo-8L MLP 모델
  기기 선택 / 전원 / 노브 제어               제스처 분류
  (현재 MVP 동작 중)                        (AI팀 모델 연결 예정)
        │                                              │
        ▼                                              ▼
[MQTT] Home Assistant                     [MQTT] 분류 결과 전송
  기기 제어 명령                             cammand/npu_debug 토픽
```

---

## 3. AI팀이 개발해야 할 모델 스펙

### 3-1. 정적 제스처 모델 (손가락 수 분류)

| 항목 | 규격 |
|------|------|
| 입력 | `float32 (1, 63)` — 21개 관절 × xyz |
| 출력 | `float32 (1, 5)` — logit (Softmax 미포함) |
| 클래스 수 | 5개 |
| 클래스 정의 | 0:ONE, 1:TWO, 2:THREE, 3:FOUR, 4:FIVE |
| **납품 파일 형식** | **`.pt` (PyTorch)** |

**클래스 의미**:

| 클래스 | 제스처 | 기기 |
|--------|--------|------|
| ONE (0) | 검지 1개 | 방 조명 |
| TWO (1) | 검지+중지 2개 | 스탠드 조명 |
| THREE (2) | 3개 | 선풍기 |
| FOUR (3) | 4개 | 에어컨 |
| FIVE (4) | 5개 | 가습기 |

### 3-2. 동적 제스처 모델 (궤적 분류)

| 항목 | 규격 |
|------|------|
| 입력 | `float32 (1, 1890)` — 30프레임 × 21개 × xyz |
| 출력 | `float32 (1, 4)` — logit (Softmax 미포함) |
| 클래스 수 | 4개 |
| 클래스 정의 | 0:CIRCLE, 1:CROSS, 2:SWIPE_UP, 3:SWIPE_DOWN |
| **납품 파일 형식** | **`.pt` (PyTorch)** |

**클래스 의미**:

| 클래스 | 제스처 | 동작 |
|--------|--------|------|
| CIRCLE (0) | O 궤적 | 전원 ON |
| CROSS (1) | X 궤적 | 전원 OFF |
| SWIPE_UP (2) | 위 스와이프 | 노브 UP |
| SWIPE_DOWN (3) | 아래 스와이프 | 노브 DOWN |

> **주의**: 두 모델 모두 Softmax를 모델 내부에 포함하지 않음.  
> 추론 코드에서 argmax 또는 softmax를 직접 적용함.

---

## 3-3. Hailo-8L 지원 연산자 제약 (모델 설계 필독)

> **출처**: Hailo Dataflow Compiler User Guide v3.27~3.30 (공식 문서)  https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf
> AI팀은 아래 제약을 반드시 확인하고 모델을 설계해야 합니다.  
> 지원되지 않는 연산자를 사용하면 `.pt → .onnx → .hef` 변환 파이프라인에서 **파싱 에러**가 발생합니다.

---

### ✅ 사용 가능한 레이어

| 레이어 | PyTorch | 비고 |
|--------|---------|------|
| Dense (FC) | `nn.Linear` | 첫 번째 레이어 또는 Dense/Conv/Pool 뒤에만 사용 가능 |
| Batch Normalization | `nn.BatchNorm1d` | Dense/Conv에 자동 fuse됨 |
| Dropout | `nn.Dropout` | 학습 시에만 동작, 추론 시 자동 제거됨 |
| Reshape | — | Conv↔Dense 전환 시에만 허용. **마지막 레이어에 사용 금지** |
| Softmax | `nn.Softmax` | Dense 뒤 `(batch, features)` 형태에서만 지원 |
| Add / Subtract | — | bias addition 등 기본 연산 지원 |

---

### ✅ 사용 가능한 활성화 함수

| 활성화 함수 | PyTorch | 안정성 |
|-------------|---------|--------|
| ReLU | `nn.ReLU` | ✅ 안정 (권장) |
| ReLU6 | `nn.ReLU6` | ✅ 안정 |
| Sigmoid | `nn.Sigmoid` | ✅ 안정 |
| Tanh | `nn.Tanh` | ✅ 안정 |
| Leaky ReLU | `nn.LeakyReLU` | ✅ 안정 |
| Hard-Sigmoid | `nn.Hardsigmoid` | ✅ 안정 |
| ELU | `nn.ELU` | ✅ 안정 |
| GeLU | `nn.GELU` | ⚠️ preview (불안정, **사용 금지**) |
| Hard-Swish | `nn.Hardswish` | ⚠️ preview (불안정, **사용 금지**) |

---

### ❌ 절대 사용 금지

```python
# ❌ forward() 안에 Python 제어 흐름 사용 금지
# ONNX export 시 변환 불가
def forward(self, x):
    if x.shape[0] > 1:   # ❌
        ...
    for i in range(n):   # ❌
        ...

# ❌ 커스텀 레이어 / 커스텀 연산자 사용 금지
# Hailo parser가 인식 불가

# ❌ 동적 shape 사용 금지
# HailoRT는 고정 입력 shape만 지원

# ❌ GeLU, Hardswish 사용 금지
# preview 단계로 변환 실패 가능성 높음
```

---

## 4. 현재 구현된 추론 파이프라인

### 4-1. 파이프라인 검증 현황

MobileNetV2 `.hef`를 stand-in으로 사용해 전체 파이프라인을 검증 완료:

```
✅ Hailo VDevice 초기화
✅ HEF 로드 및 입출력 shape 확인
✅ 매 프레임 추론 실행 (asyncio.to_thread 비동기 처리)
✅ 추론 결과 MQTT → Home Assistant 전송
✅ MediaPipe + Hailo 동시 실행 (CPU/NPU 병렬)
```

### 4-2. HailoEngine 코드 구조

**파일 위치**: `src/cammand/engine/hailo_engine.py`

```python
class HailoEngine(GestureEngine):

    def __init__(self, hef_path: str):
        # 1. MediaPipe 초기화 (랜드마크 추출용)
        self._hands = mp.solutions.hands.Hands(...)

        # 2. Hailo VDevice 생성
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)

        # 3. HEF 로드 및 설정
        infer_model = self._vdevice.create_infer_model(hef_path)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        # 4. 추론 컨텍스트 진입 (앱 수명 동안 유지)
        self._configured_ctx = infer_model.configure()
        self._configured = self._configured_ctx.__enter__()

        # 5. 입출력 버퍼 사전 할당 (매 프레임 재사용, 메모리 할당 없음)
        self._in_buf  = np.zeros(input_shape,  dtype=np.float32)
        self._out_buf = np.zeros(output_shape, dtype=np.float32)
        self._bindings = self._configured.create_bindings(
            input_buffers  = {input_name:  self._in_buf},
            output_buffers = {output_name: self._out_buf},
        )

    def process(self, frame: np.ndarray) -> EngineResult:
        # Step 1: CPU — MediaPipe 랜드마크 추출
        landmarks = mediapipe_extract(frame)  # list[Landmark], len=21

        # Step 2: CPU — 전처리 (현재: 이미지 리사이즈 / 교체 후: 랜드마크 flatten)
        self._preprocess(frame)               # → self._in_buf 채움

        # Step 3: NPU — 추론
        self._configured.run([self._bindings], timeout_ms=1000)

        # Step 4: CPU — 후처리
        out      = self._bindings.output().get_buffer()
        class_id = int(np.argmax(out))
        probs    = softmax(out)
        conf     = probs[class_id] * 100      # %

        return EngineResult(landmarks=landmarks, npu_debug=f"[NPU] ... → gesture")

    def close(self):
        self._hands.close()
        self._configured_ctx.__exit__(None, None, None)
        self._vdevice.release()
```

### 4-3. 엔진 인터페이스 (GestureEngine ABC)

**파일 위치**: `src/cammand/engine/base.py`

```python
@dataclass(frozen=True)
class Landmark:
    x: float  # 0.0~1.0 정규화
    y: float
    z: float

@dataclass
class EngineResult:
    hand_detected:   bool
    landmarks:       list[Landmark] | None  # 항상 len=21 또는 None
    raw_frame:       np.ndarray
    annotated_frame: np.ndarray
    npu_debug:       str | None             # "[NPU] ... → gesture"

class GestureEngine(ABC):
    def process(self, frame: np.ndarray) -> EngineResult: ...
    def close(self) -> None: ...
```

---

## 5. AI팀 모델 수령 후 교체 절차 (Hot Swap)

### Step 1 — `.pt` → `.hef` 변환 (HW/SW팀 담당)

AI팀으로부터 `.pt` 파일을 수령한 후, HW/SW팀이 Hailo SDK를 사용해 `.hef`로 변환합니다.

```bash
# 1) PyTorch → ONNX 변환 (AI팀 환경 또는 로컬 PC에서)
python -c "
import torch
model = torch.load('gesture_static.pt')
model.eval()
dummy = torch.zeros(1, 63)
torch.onnx.export(model, dummy, 'gesture_static.onnx', opset_version=11)
"

# 2) ONNX → HEF 변환 (Hailo SDK 설치된 환경에서)
hailomz compile \
  --onnx gesture_static.onnx \
  --hw-arch hailo8l \
  --output-name gesture_static
# 결과: gesture_static.hef 생성
```

변환된 `.hef` 파일은 `models/` 폴더에 배치:

```
Cammand/
└── models/
    ├── gesture_static.hef    ← 정적 제스처 모델 (손가락 수)
    └── gesture_dynamic.hef   ← 동적 제스처 모델 (궤적)
```

### Step 2 — `hailo_engine.py` 수정 (전처리 교체)

현재 `_preprocess()` (MobileNetV2용 이미지 전처리):
```python
def _preprocess(self, frame: np.ndarray) -> None:
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    np.copyto(self._in_buf, normalized)
```

교체 후 `_preprocess()` (MLP용 랜드마크 전처리):
```python
def _preprocess(self, landmarks: list[Landmark]) -> None:
    # 21개 × xyz = 63 float32 값으로 flatten
    flat = np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32).flatten()  # shape: (63,)
    np.copyto(self._in_buf, flat.reshape(self._input_shape))
```

동적 모델의 경우 30프레임 슬라이딩 윈도우 버퍼 추가:
```python
# __init__ 에 추가
from collections import deque
self._frame_buffer: deque = deque(maxlen=30)

# _preprocess 에서
self._frame_buffer.append(landmarks_flat_63)
if len(self._frame_buffer) == 30:
    sequence = np.concatenate(list(self._frame_buffer))  # shape: (1890,)
    np.copyto(self._in_buf, sequence.reshape(self._input_shape))
```

### Step 3 — `config.py` 경로 설정 추가

```python
hailo_static_hef:  str = "models/gesture_static.hef"
hailo_dynamic_hef: str = "models/gesture_dynamic.hef"
```

> 이 3단계 외 나머지 파이프라인 코드(main.py, state machine, MQTT 등)는 **변경 없음**.

---

## 6. 하드웨어 및 런타임 환경

| 항목 | 스펙 |
|------|------|
| SBC | Raspberry Pi 5 (aarch64) |
| NPU | Hailo-8L (PCIe 연결, `/dev/hailo0`) |
| HailoRT | v4.20.0 (`hailo_platform` Python 패키지) |
| Python | 3.11+ (가상환경: `Cammand/bin/activate`) |
| 카메라 | Raspberry Pi Camera Module 3 (imx708) |
| 해상도 | 640×360 (처리), 9:16 비율 |

> **[HW/SW팀 참고]** HEF 변환 시 반드시 `hailo8l` 타겟으로 컴파일 (`--hw-arch hailo8l`).  
> Hailo-8(h8)용 HEF와 호환되지 않음.

---

## 7. 데이터 수집 가이드라인

### 정적 제스처 (손가락 수)

- 각 제스처당 최소 수집 권장 샘플 수: 500개 이상
- 다양한 조명, 피부톤, 카메라 각도 포함 권장
- 입력 포맷: MediaPipe 21개 랜드마크 → `(63,)` float32 flatten
- 좌표 정규화: MediaPipe 출력 기준 이미 0.0~1.0 정규화됨

### 동적 제스처 (궤적)

- 시퀀스 길이: **고정 30프레임** (가변 길이 불가)
- 카메라 fps 기준: 30fps → 1초 동작
- 입력 포맷: 30프레임 × 63 = `(1890,)` float32
- delta 계산 여부: 현재 미정 — gesture_infer_requirements.docx 참고

---

## 8. 검증 방법

모델 수령 및 변환 후 아래 순서로 검증:

```bash
# 1. .pt → .hef 변환 완료 후 models/ 폴더에 배치 (Step 1 참고)

# 2. .env 수정
GESTURE_ENGINE=hailo
HAILO_HEF_PATH=models/gesture_static.hef  # 또는 dynamic

# 3. 실행
cd /home/rpi5/Cammand
source Cammand/bin/activate
cammand

# 4. 확인 포인트
# - 터미널: "HailoEngine: ... 로드 완료 / 입력: (1, 63) 출력: (1, 5)"
# - HA 대시보드: "Cammand NPU Debug" 엔티티에 분류 결과 실시간 갱신
# - 손가락 수 바꿀 때 클래스 변화 확인
```

---

## 9. 레포지토리

| 레포 | 주소 | 내용 |
|------|------|------|
| Core | `github.com/Cammand-2026/Cammand-Core` | SW 전체 (이 문서) |
| Models | `github.com/Cammand-2026/cammand-models` | `.pt` 파일 및 변환 산출물 관리 |

**AI팀 산출물 전달 위치**: `cammand-models` 레포에 `.pt` 파일 업로드 후 Core팀에 통보.

---

## 10. 문의

| 역할 | 이름 | 담당 |
|------|------|------|
| HW/SW (인퍼런스 파이프라인) | 백승찬 | 이 문서 관련 문의 |
| AI (모델 아키텍처) | 김성한 | 모델 설계 관련 문의 |
| AI (모델 개발 총괄) | 채선우 | 데이터/학습 관련 문의 |
