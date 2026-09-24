# Models

`.hef` 모델 파일 위치. 모델은 [Command-Model 레포](https://github.com/Cammand-2026/Command-Model)에서 관리합니다.

| 파일 | 용도 |
|------|------|
| `mobilenetv2-12.hef` | NPU 파이프라인 검증용 stand-in |
| `gesture.hef` | MLP 모델 (수령 후 배치) |

Hailo 엔진 사용: `sudo apt install hailo-all` 후 `.env`에 `GESTURE_ENGINE=hailo`, `HAILO_HEF_PATH=models/<파일>.hef` 설정.
