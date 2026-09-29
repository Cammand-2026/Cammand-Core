# Cammand

> **Camera + Command** — 카메라 제스처로 스마트홈을 제어하는 라즈베리파이5 + Hailo-8L 기반 시스템

---

## Gesture

### 기기 선택 (2초 유지)

| 제스처 | 별칭 | 기기 |
|--------|------|------|
| ☝️ 검지 (엄지 접음) | 👍 엄지척 | 방 조명 |
| ✌️ 검지+중지 (엄지 접음) | 총모양 (엄지+검지) | 스탠드 조명 |
| 검지+중지+약지 | 엄지+검지+중지 | 선풍기 |
| 검지~새끼 4개 (엄지 접음) | — | 에어컨 |
| 🖐 5개 | — | 가습기 |

### 기기 조작

| 제스처 | 동작 |
|--------|------|
| ☝️ 검지 0.5초 유지 후 원 그리기 / 좌우 흔들기 | 전원 ON / OFF (엄지척 별칭은 진입 불가) |
| 🤚 손바닥 수평 0.4초 유지 후 상하 이동 | 노브 UP / DOWN |
| 다른 기기 제스처 유지 (1초 대기 + 1.5초) | 기기 전환 (방 조명은 전환 불가 — 주먹으로 해제 후 선택) |
| ✊ 주먹 1.5초 유지 | 선택 해제 |

### 예약 제스처

| 제스처 | 동작 |
|--------|------|
| 🤘 검지+새끼 (ROCK) | 대기 상태에서 "락앤롤 인식됨" 피드백만 |
| 👌 OK사인 (OK_SIGN) | 대기 상태에서 "OK사인 인식됨" 피드백만 |

손이 1.5초 이상 보이지 않으면 대기 상태로 돌아갑니다.

---

## Project Structure

```
src/cammand/
├── engine/    # MediaPipe / Hailo 엔진
├── gesture/   # 제스처 인식
├── state/     # 상태 머신
└── io/        # 카메라, MQTT, MJPEG 스트리밍
docs/          # AI 모델팀 문서
models/        # gesture_static.hef, gesture_dynamic.hef (Cammand-2026/Command-Model)
```

---

## Installation

```bash
python3 -m venv Cammand --system-site-packages
source Cammand/bin/activate
pip install -e ".[dev]"
cp .env.example .env
cammand
```

스트리밍: `http://<raspberry-pi-ip>:8080/stream`

---

## Testing

```bash
pytest tests/unit/ -v                        # 하드웨어 불필요
pytest tests/integration/ -v -m integration  # 라즈베리파이 + MQTT 브로커 필요
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
