from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CammandSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # MQTT
    mqtt_broker: str = "127.0.0.1"
    mqtt_port: int = 1883

    # 스트리밍
    stream_port: int = 8080
    stream_fps: int = 12
    stream_jpeg_quality: int = 70

    # 엔진 선택
    gesture_engine: str = Field(default="mediapipe", pattern="^(mediapipe|hailo)$")
    hailo_hef_path: str = "models/gesture.hef"

    # 카메라
    camera_width: int = 640
    camera_height: int = 360
    camera_rotate: bool = True

    # 타이밍
    selection_hold_sec: float = 2.0
    onoff_entry_sec: float = 0.5
    switch_hold_sec: float = 1.5
    switch_buffer_sec: float = 1.0
    five_h_confirm_sec: float = 0.4
    fist_hold_sec: float = 1.5
    timeout_sec: float = 1.5
    knob_publish_hz: int = 10
    process_interval_sec: float = 0.1


settings = CammandSettings()
