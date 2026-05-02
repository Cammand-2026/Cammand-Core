"""MQTT 통합 테스트 — 실제 MQTT 브로커 필요 (127.0.0.1:1883)."""
from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.integration


def test_mqtt_connect_and_publish():
    """브로커에 연결하고 피드백 토픽에 메시지를 발행한다."""
    from cammand.io.mqtt_client import MqttPublisher

    pub = MqttPublisher()
    pub.connect()
    time.sleep(1)  # 연결 대기
    pub.publish_feedback("통합 테스트")
    pub.disconnect()
