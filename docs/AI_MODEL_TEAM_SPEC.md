# Cammand AI 모델 스펙

> **작성일**: 2026-05-08  
> **최종 수정**: 2026-09-29

---

## 1. 프로젝트 개요

Cammand는 라즈베리파이5 위에서 동작하는 제스처 인식 스마트홈 허브입니다.  
카메라 앞에서 손 제스처만으로 조명, 선풍기, 에어컨, 가습기를 제어합니다.

---

## 2. 모델 스펙

### 2-1. 정적 제스처 모델 (손가락 수 분류)

| 항목 | 규격 |
|------|------|
| 입력 | `float32 (1, 63)` — 21개 관절 × xyz |
| 출력 | `float32 (1, 7)` — logit (Softmax 미포함) |
| 클래스 수 | 7개 |
| 클래스 정의 | 0:ONE, 1:TWO, 2:THREE, 3:FOUR, 4:FIVE, 5:ROCK, 6:OK_SIGN |
| 파일 형식 | `.pt` (PyTorch) |

**클래스 의미** (보조 제스처는 같은 클래스로 라벨링):

| 클래스 | 기본 제스처 | 보조 제스처 | 용도 |
|--------|-------------|-------------|------|
| ONE (0) | 검지 (엄지 접음) | 엄지척 (엄지만 펴고 위로) | 방 조명 |
| TWO (1) | 검지+중지 (엄지 접음) | 총모양 (엄지+검지) | 스탠드 조명 |
| THREE (2) | 검지+중지+약지 | 엄지+검지+중지 | 선풍기 |
| FOUR (3) | 검지+중지+약지+새끼 (엄지 접음) | — | 에어컨 |
| FIVE (4) | 다섯 손가락 모두 폄 | — | 가습기 |
| ROCK (5) | 검지+새끼 | — | 예약 (피드백만) |
| OK_SIGN (6) | 엄지·검지 끝 맞닿음 + 중지·약지·새끼 폄 | — | 예약 (피드백만) |

### 2-2. 동적 제스처 모델 (궤적 분류)

| 항목 | 규격 |
|------|------|
| 입력 | `float32 (1, 630)` — 10프레임 × 21개 × xyz |
| 출력 | `float32 (1, 4)` — logit (Softmax 미포함) |
| 클래스 수 | 4개 |
| 클래스 정의 | 0:CIRCLE, 1:SHAKE, 2:SWIPE_UP, 3:SWIPE_DOWN |
| 파일 형식 | `.pt` (PyTorch) |

**클래스 의미**:

| 클래스 | 제스처 | 동작 |
|--------|--------|------|
| CIRCLE (0) | 검지로 원 그리기 | 전원 ON |
| SHAKE (1) | 검지 좌우 흔들기 | 전원 OFF |
| SWIPE_UP (2) | 위 스와이프 | 노브 UP |
| SWIPE_DOWN (3) | 아래 스와이프 | 노브 DOWN |

---

## 3. Hailo-8L 연산자 제약

> 출처: Hailo Dataflow Compiler User Guide v3.27~3.30  
> https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf

**사용 가능한 레이어**

| 레이어 | PyTorch | 비고 |
|--------|---------|------|
| Dense (FC) | `nn.Linear` | |
| Batch Normalization | `nn.BatchNorm1d` | Dense에 fuse됨 |
| Dropout | `nn.Dropout` | 추론 시 제거됨 |

**사용 가능한 활성화 함수**: `nn.ReLU`(권장), `nn.ReLU6`, `nn.Sigmoid`, `nn.Tanh`, `nn.LeakyReLU`, `nn.Hardsigmoid`, `nn.ELU`

**사용 금지**

- `nn.GELU`, `nn.Hardswish`
- `forward()` 안의 Python 제어 흐름 (`if`, `for`)
- 커스텀 레이어 / 커스텀 연산자
- 동적 shape (입력 shape 고정)
- 마지막 레이어의 Reshape

---

## 4. 모델 파일

| 모델 | 경로 | 설정 키 (`config.py` / `.env`) |
|------|------|--------------------------------|
| 정적 | `models/gesture_static.hef` | `hailo_static_hef` / `HAILO_STATIC_HEF` |
| 동적 | `models/gesture_dynamic.hef` | `hailo_dynamic_hef` / `HAILO_DYNAMIC_HEF` |

---

## 5. 하드웨어 및 런타임 환경

| 항목 | 스펙 |
|------|------|
| SBC | Raspberry Pi 5 (aarch64) |
| NPU | Hailo-8L (PCIe 연결, `/dev/hailo0`) |
| HEF 타겟 | `hailo8l` (Hailo-8용 HEF와 호환되지 않음) |
| HailoRT | v4.20.0 (`hailo_platform` Python 패키지) |
| Python | 3.11+ (가상환경: `Cammand/bin/activate`) |
| 카메라 | Raspberry Pi Camera Module 3 (imx708) |
| 해상도 | 640×360 (처리), 9:16 비율 |
| 처리 주기 | 10Hz (`PROCESS_INTERVAL_SEC=0.1`) |

### 5-1. 실측 처리 시간 (2026-09-29, 라이브 카메라 프레임, 엔진별 20초)

| 엔진 | 프레임 수 | 평균 | p50 | p95 |
|------|-----------|------|-----|-----|
| MediaPipeEngine (CPU) | 461 | 41.8 ms | 39.4 ms | 57.1 ms |
| HailoEngine (MediaPipe + NPU) | 391 | 49.6 ms | 46.9 ms | 69.9 ms |

- 카메라 캡처: 25.7 fps (640×360, 회전 포함)
- `engine.process(frame)` 1회 소요 시간 기준

---

## 6. 데이터 수집 가이드라인

### 정적 제스처 (손가락 수)

- 클래스당 500개 이상 (보조 제스처도 같은 클래스로 포함)
- 다양한 조명, 피부톤, 카메라 각도 포함
- 입력 포맷: MediaPipe 21개 랜드마크 → `(63,)` float32 flatten
- 좌표: MediaPipe 출력 기준 0.0~1.0 정규화

### 동적 제스처 (궤적)

- 시퀀스 길이: 고정 10프레임 (가변 길이 불가)
- 10Hz 기준 10프레임 = 1초 동작 (0.1초 간격 샘플링)
- 입력 포맷: 10프레임 × 63 = `(630,)` float32

---

## 7. 레포지토리

| 레포 | 주소 | 내용 |
|------|------|------|
| Core | `github.com/Cammand-2026/Cammand-Core` | SW 전체 |
| Models | `github.com/Cammand-2026/Command-Model` | `.pt` 파일 및 변환 산출물 |
