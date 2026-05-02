# Models

이 폴더에는 AI 모델 파일이 배치됩니다.
모델 파일은 용량/보안 이유로 별도 레포에서 관리됩니다.

## 현재 엔진: MediaPipe (CPU)

추가 파일 불필요. `.env`에서 `GESTURE_ENGINE=mediapipe` 설정을 사용하세요.

## Hailo-8L 엔진 사용 시

1. [cammand-models 레포](https://github.com/<org>/cammand-models)에서 `.hef` 파일 다운로드
2. 이 폴더에 `gesture.hef` 파일 배치
3. `hailo-platform` SDK 설치
4. `.env`에서 `GESTURE_ENGINE=hailo` 설정
