# Cammand

> **Camera + Command** — 카메라 기반 제스처로 스마트홈을 제어하는 임베디드 AI 시스템

---

## Overview

Cammand는 라즈베리파이5 위에서 동작하는 제스처 인식 스마트홈 허브입니다.  
사용자는 카메라 앞에서 손 제스처만으로 조명·선풍기·에어컨·가습기를 켜고 끄거나 세기를 조절할 수 있습니다.

- **MediaPipe**로 손 스켈레톤을 실시간 추출하고
- **Hailo-8L NPU**에서 제스처 분류 모델을 추론하며
- **Home Assistant** REST/WebSocket API로 기기를 제어합니다

---

## Demo

| 기기 선택 | 전원 제어 | 노브 조절 |
|:---------:|:---------:|:---------:|
| 손가락 수로 기기 선택 | 따봉 ON / 붕따 OFF | 손 수평 후 상하 이동 |

---

## Gesture Reference

### Step 1 — 기기 선택 (손가락 수)

| 제스처 | 기기 | HA Entity |
|--------|------|-----------|
| ☝️ 검지 1개 | 방 조명 | `light.room_ceiling` |
| ✌️ 검지+중지 2개 | 스탠드 조명 | `light.standing_lamp` |
| 🤟 3개 | 선풍기 | `fan.room_fan` |
| 🖖 4개 | 에어컨 | `climate.aircon` |
| 🖐 5개 | 가습기 | `humidifier.room` |

### Step 2 — 기기 조작

| 제스처 | 동작 |
|--------|------|
| 👍 따봉 (주먹+엄지 위) | 전원 **ON** |
| 👎 붕따 (주먹+엄지 아래) | 전원 **OFF** |
| 🤚 손 수평 유지 후 상하 이동 | 노브 **UP / DOWN** |
| ✊ 주먹 3초 유지 | 기기 선택 **초기화** |

---

## System Architecture

```
Camera (Picamera2)
    │
    ▼
MediaPipe Hands ── Skeleton (21 landmarks)
    │
    ├─ recognize_selection()   IDLE / SELECTING 단계
    │       ONE ~ FIVE
    │
    └─ recognize_control()     SELECTED / CONTROLLING 단계
            THUMB_UP / THUMB_DOWN / FIVE_HORIZONTAL / FIST
                │
                ▼
        State Machine (Phase)
        IDLE → SELECTING → SELECTED → CONTROLLING
                │
                ▼
        MQTT Broker (127.0.0.1:1883)
                │
                ▼
        Home Assistant
```

---

## Hardware

| 구성 요소 | 사양 |
|-----------|------|
| SBC | Raspberry Pi 5 (aarch64) |
| NPU | Hailo-8L |
| 카메라 | Raspberry Pi Camera (9:16) |
| 디스플레이 | 터치스크린 |

---

## Software Stack

| 분류 | 항목 |
|------|------|
| Language | Python 3.11+ |
| 손 추적 | MediaPipe Hands |
| 영상 스트리밍 | Picamera2 + Flask MJPEG |
| 통신 | MQTT (paho-mqtt) |
| 스마트홈 연동 | Home Assistant REST / WebSocket |

---

## Project Structure

```
src/cammand/
├── config.py          # 설정 (.env 기반, pydantic-settings)
├── devices.py         # 기기 정의 (DeviceConfig dataclass)
├── feedback.py        # 피드백 텍스트 생성 함수
├── main.py            # asyncio 진입점
├── engine/            # 추론 엔진 추상화 레이어
│   ├── base.py        # GestureEngine ABC
│   ├── mediapipe_engine.py  # CPU 구현체 (현재)
│   └── hailo_engine.py      # Hailo-8L NPU 구현체 (예정)
├── gesture/           # 순수 제스처 인식 로직
│   ├── recognizer.py
│   └── trajectory.py
├── state/             # 상태 머신
│   ├── phase.py
│   └── machine.py
└── io/                # 외부 I/O 어댑터
    ├── camera.py
    ├── mqtt_client.py
    └── stream_server.py
```

> AI 모델 파일(.hef)은 별도 [cammand-models](https://github.com/Cammand-2026/cammand-models) 레포에서 관리됩니다.  
> 현재는 `GESTURE_ENGINE=mediapipe` (CPU) 설정을 사용합니다.

---

## Installation

```bash
# 1. 가상환경 생성 (system-site-packages 필수: Picamera2 의존)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 2. 패키지 설치
pip install -e ".[dev]"

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에서 MQTT_BROKER 등 필요한 값 수정

# 4. 실행
cammand
```

스트리밍 주소: `http://<raspberry-pi-ip>:8080/stream`

---

## Testing

```bash
# 단위 테스트 (하드웨어 불필요 — CI 환경에서 실행 가능)
pytest tests/unit/ -v

# 통합 테스트 (라즈베리파이 + MQTT 브로커 필요)
pytest tests/integration/ -v -m integration
```

---

## State Machine

```
        손 감지
IDLE ──────────────► SELECTING
  ◄── UNKNOWN / 손 없음        │ 2초 유지
                               ▼
        주먹 3초         SELECTED
        ◄──────────────────────┤
                               │ 손 수평 0.4초
                               ▼
                         CONTROLLING
                         (노브 조절)
```

---

## Team

| 역할 | 이름 | 담당 |
|------|------|------|
| **PM** | 백승찬 | 프로젝트 총괄 · Embedded SW · IoT 백엔드 · HW 회로 · AI 모델 포팅 |
| **AI** | 김성한 | 제스처 모델 아키텍처 설계 · 데이터 증강 (Diffusion) · 전처리 필터 개발 |
| **AI** | 채선우 | 제스처 AI 모델 개발 총괄 · 데이터 수집 · 제스처 정의 · 통합 테스트 |

---

## License

MIT License
