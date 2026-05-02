from __future__ import annotations

import json

import paho.mqtt.client as mqtt

from ..config import settings

_SENSOR_CONFIG_TOPIC = "homeassistant/sensor/cammand_gesture/config"
_FEEDBACK_CONFIG_TOPIC = "homeassistant/sensor/cammand_feedback/config"
_SENSOR_STATE_TOPIC = "cammand/sensor/gesture/state"

_TOPIC_FEEDBACK = "cammand/feedback"
_TOPIC_POWER_PREFIX = "cammand/power"
_TOPIC_KNOB_PREFIX = "cammand/knob"

_SENSOR_CONFIG = {
    "name": "Cammand Gesture",
    "unique_id": "cammand_gesture_sensor",
    "stat_t": _SENSOR_STATE_TOPIC,
    "icon": "mdi:hand-wave",
}
_FEEDBACK_CONFIG = {
    "name": "Cammand Feedback",
    "unique_id": "cammand_feedback_sensor",
    "stat_t": _TOPIC_FEEDBACK,
    "icon": "mdi:gesture-tap",
}


class MqttPublisher:
    """paho MQTT 래퍼. Home Assistant Auto-Discovery 등록 및 토픽 발행."""

    def __init__(self) -> None:
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect

    def connect(self) -> None:
        self._client.connect(settings.mqtt_broker, settings.mqtt_port, 60)
        self._client.loop_start()

    def _on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        print(f">> MQTT 연결 성공: rc={rc}")
        client.publish(_SENSOR_CONFIG_TOPIC, json.dumps(_SENSOR_CONFIG), retain=True)
        client.publish(_FEEDBACK_CONFIG_TOPIC, json.dumps(_FEEDBACK_CONFIG), retain=True)
        print(">> HA Auto-Discovery 등록 완료")

    def publish_feedback(self, text: str) -> None:
        self._client.publish(_TOPIC_FEEDBACK, text)

    def publish_power(self, device_id: str, state: str) -> None:
        self._client.publish(f"{_TOPIC_POWER_PREFIX}/{device_id}", state)

    def publish_knob(self, device_id: str, value_str: str) -> None:
        self._client.publish(f"{_TOPIC_KNOB_PREFIX}/{device_id}", value_str)

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
