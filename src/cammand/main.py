"""
Cammand 진입점 — 의존성 조립 및 asyncio 메인 루프.
비즈니스 로직 없음: 모듈을 연결하고 루프를 돌립니다.
"""
from __future__ import annotations

import asyncio
import threading
import time

from .config import settings
from .engine.base import GestureEngine
from .feedback import feedback_idle
from .io.camera import CameraReader
from .io.mqtt_client import MqttPublisher
from .io.stream_server import MjpegServer
from .state.machine import StateMachine


def _build_engine() -> GestureEngine:
    """설정에 따라 엔진 구현체 선택 (팩토리)."""
    if settings.gesture_engine == "mediapipe":
        from .engine.mediapipe_engine import MediaPipeEngine
        return MediaPipeEngine()
    if settings.gesture_engine == "hailo":
        from .engine.hailo_engine import HailoEngine
        return HailoEngine(settings.hailo_hef_path)
    raise ValueError(f"알 수 없는 엔진: {settings.gesture_engine}")


async def _main_loop(
    camera: CameraReader,
    engine: GestureEngine,
    machine: StateMachine,
    streamer: MjpegServer,
) -> None:
    last_process = 0.0
    while True:
        # blocking I/O → thread pool 위임 (asyncio 이벤트 루프 블록 방지)
        frame = await asyncio.to_thread(camera.read)
        streamer.push(frame)

        now = time.monotonic()
        if now - last_process < settings.process_interval_sec:
            await asyncio.sleep(0)
            continue
        last_process = now

        result = await asyncio.to_thread(engine.process, frame)
        await machine.update(result.landmarks, result.raw_frame.shape[0], now)


def main() -> None:
    mqtt = MqttPublisher()
    mqtt.connect()

    camera = CameraReader()
    camera.start()

    streamer = MjpegServer()
    threading.Thread(target=streamer.run, daemon=True, name="mjpeg").start()
    print(f">> 스트리밍 서버: http://0.0.0.0:{settings.stream_port}/stream")

    engine = _build_engine()

    machine = StateMachine(
        on_feedback=mqtt.publish_feedback,
        on_power=mqtt.publish_power,
        on_knob=mqtt.publish_knob,
    )

    mqtt.publish_feedback(feedback_idle())

    try:
        asyncio.run(_main_loop(camera, engine, machine, streamer))
    except KeyboardInterrupt:
        print("\n>> 종료 중...")
    finally:
        engine.close()
        camera.stop()
        mqtt.disconnect()


if __name__ == "__main__":
    main()
