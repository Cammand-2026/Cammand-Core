import cv2
import math
import mediapipe as mp
import paho.mqtt.client as mqtt
import json
import time
import threading
from enum import Enum
from flask import Flask, Response
from picamera2 import Picamera2

# ── MQTT 설정 ──────────────────────────────────────────────────────────────
BROKER = "127.0.0.1"
PORT   = 1883

# ── 스트리밍 설정 ───────────────────────────────────────────────────────────
STREAM_PORT         = 8080
STREAM_FPS          = 12
STREAM_JPEG_QUALITY = 70
_frame_lock         = threading.Lock()
_latest_jpeg: bytes = b""

# ── MQTT 토픽 ───────────────────────────────────────────────────────────────
TOPIC_FEEDBACK      = "cammand/feedback"
TOPIC_POWER_PREFIX  = "cammand/power"   # /{device_id} → "on" / "off"
TOPIC_KNOB_PREFIX   = "cammand/knob"    # /{device_id} → numeric value

SENSOR_CONFIG_TOPIC   = "homeassistant/sensor/cammand_gesture/config"
SENSOR_STATE_TOPIC    = "cammand/sensor/gesture/state"
FEEDBACK_CONFIG_TOPIC = "homeassistant/sensor/cammand_feedback/config"

sensor_config = {
    "name": "Cammand Gesture",
    "unique_id": "cammand_gesture_sensor",
    "stat_t": SENSOR_STATE_TOPIC,
    "icon": "mdi:hand-wave",
}
feedback_sensor_config = {
    "name": "Cammand Feedback",
    "unique_id": "cammand_feedback_sensor",
    "stat_t": TOPIC_FEEDBACK,
    "icon": "mdi:gesture-tap",
}

# ── 기기 정의 ────────────────────────────────────────────────────────────────
DEVICES: dict[str, dict] = {
    "ONE":   {"id": "room_light",  "name": "방조명",      "knob_min": 0,  "knob_max": 100, "knob_step": 1,   "knob_off": 0 },
    "TWO":   {"id": "stand_light", "name": "스탠드 조명",  "knob_min": 0,  "knob_max": 100, "knob_step": 1,   "knob_off": 0 },
    "THREE": {"id": "fan",         "name": "선풍기",       "knob_min": 1,  "knob_max": 4,   "knob_step": 1,   "knob_off": 0 },
    "FOUR":  {"id": "aircon",      "name": "에어컨",       "knob_min": 18, "knob_max": 30,  "knob_step": 0.5, "knob_off": 18},
    "FIVE":  {"id": "humidifier",  "name": "가습기",       "knob_min": 0,  "knob_max": 100, "knob_step": 5,   "knob_off": 0 },
}

DEVICE_UNITS: dict[str, str] = {
    "room_light":  "%",
    "stand_light": "%",
    "fan":         "단",
    "aircon":      "°C",
    "humidifier":  "%",
}

# ── 상태 머신 ────────────────────────────────────────────────────────────────
class Phase(Enum):
    IDLE        = "IDLE"
    SELECTING   = "SELECTING"
    SELECTED    = "SELECTED"
    CONTROLLING = "CONTROLLING"

SELECTION_HOLD_SEC = 2.0   # 기기 선택 유지 시간
SWITCH_HOLD_SEC    = 3.0   # SELECTED 상태 기기 전환 유지 시간
SWITCH_BUFFER_SEC  = 1.0   # 기기 전환 카운트다운 시작 전 대기 (오인식 방지)
FIVE_H_CONFIRM_SEC = 0.4   # FIVE_HORIZONTAL 확인 시간 (오인식 방지)
FIST_HOLD_SEC      = 3.0   # 주먹 유지 → IDLE 복귀 시간
TIMEOUT_SEC        = 3.0   # 손 없으면 IDLE 복귀
KNOB_PUBLISH_HZ    = 10    # 노브 발행 최대 빈도 (초당)
PROCESS_INTERVAL   = 0.1   # 제스처 처리 주기 (초)


# ── 제스처 인식 헬퍼 ──────────────────────────────────────────────────────────
def _finger_open(lm, tip: int, pip: int, mcp: int, ref: float) -> bool:
    """
    손가락 펼침 여부.
    y비교(세로 방향) OR 거리비교(기울었을 때) — 둘 중 하나라도 통과하면 펼침으로 판단.
    """
    y_open = lm[tip].y < lm[pip].y
    d_open = math.hypot(lm[tip].x - lm[mcp].x, lm[tip].y - lm[mcp].y) > ref * 0.55
    return y_open or d_open


def _thumb_open(lm) -> bool:
    """엄지 끝(4)이 IP 관절(3)보다 손목(0)에서 더 멀리 있으면 펼침."""
    d_tip = math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y)
    d_ip  = math.hypot(lm[3].x - lm[0].x, lm[3].y - lm[0].y)
    return d_tip > d_ip * 1.15


def _is_horizontal(lm) -> bool:
    """손목(0)→중지MCP(9) 벡터 기준 가로 방향 여부. ±59° 이내면 가로로 판단."""
    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    return abs(dx) > abs(dy) * 0.6


# ── Step 1: 기기 선택 제스처 (IDLE / SELECTING 전용) ─────────────────────────
def recognize_selection(hand_landmarks) -> str:
    """
    손가락 1~5개를 명확히 구분.
    반환값: ONE | TWO | THREE | FOUR | FIVE | UNKNOWN

    - FIVE_HORIZONTAL·THUMB 계열은 절대 반환하지 않음
      (SELECTING 중 가습기(FIVE) 선택이 노브 제스처로 오인되는 문제 방지)
    - 각 제스처는 "어떤 손가락이 열려야 하고, 어떤 손가락이 닫혀야 하는지"를
      명시적으로 검사 (단순 카운트 대비 오인식 감소)
    """
    lm  = hand_landmarks.landmark
    ref = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)

    idx = _finger_open(lm, 8,  6,  5,  ref)
    mid = _finger_open(lm, 12, 10, 9,  ref)
    rng = _finger_open(lm, 16, 14, 13, ref)
    pnk = _finger_open(lm, 20, 18, 17, ref)
    thm = _thumb_open(lm)

    if     idx and not mid and not rng and not pnk: return "ONE"
    if     idx and     mid and not rng and not pnk: return "TWO"
    if     idx and     mid and     rng and not pnk: return "THREE"
    if     idx and     mid and     rng and     pnk: return "FIVE" if thm else "FOUR"
    return "UNKNOWN"


# ── Step 2: 기기 조작 제스처 (SELECTED / CONTROLLING 전용) ───────────────────
def recognize_control(hand_landmarks) -> str:
    """
    선택된 기기를 조작하는 제스처를 구분.
    반환값: THUMB_UP | THUMB_DOWN | FIVE_HORIZONTAL | FIST | UNKNOWN

    - 선택 제스처(ONE~FIVE)는 절대 반환하지 않음
    - FIVE_HORIZONTAL: 4개 손가락 모두 펼침 + 가로 방향
    - FIST / THUMB_UP / THUMB_DOWN: 4개 손가락 모두 닫힘 + 엄지 방향
    """
    lm  = hand_landmarks.landmark
    ref = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)

    idx = _finger_open(lm, 8,  6,  5,  ref)
    mid = _finger_open(lm, 12, 10, 9,  ref)
    rng = _finger_open(lm, 16, 14, 13, ref)
    pnk = _finger_open(lm, 20, 18, 17, ref)

    # 노브 모드: 4개 손가락 모두 펼침 + 가로 방향
    if idx and mid and rng and pnk and _is_horizontal(lm):
        return "FIVE_HORIZONTAL"

    # 주먹 계열: 4개 손가락 모두 닫힘
    if not idx and not mid and not rng and not pnk:
        # 거리로 재확인 (손이 기울었을 때 y비교 오판 방지)
        closed_ok = (
            math.hypot(lm[8].x - lm[5].x, lm[8].y - lm[5].y) < ref * 0.9 and
            math.hypot(lm[12].x - lm[9].x, lm[12].y - lm[9].y) < ref * 0.9
        )
        if not closed_ok:
            return "UNKNOWN"
        # 엄지 방향: index MCP(5) y - thumb tip(4) y
        # y축은 아래가 양수이므로, offset > 0 이면 엄지가 위로 향함 → THUMB_UP
        offset = lm[5].y - lm[4].y
        if offset > ref * 0.5:
            return "THUMB_UP"
        if offset < -ref * 0.5:
            return "THUMB_DOWN"
        return "FIST"

    return "UNKNOWN"


# ── 피드백 텍스트 생성 ────────────────────────────────────────────────────────
def _progress_dots(elapsed: float, total: float = SELECTION_HOLD_SEC) -> str:
    filled = int(elapsed / total * 3)
    return "●" * filled + "○" * (3 - filled)

def feedback_idle() -> str:
    return "손을 카메라에 보여주세요"

def feedback_selecting(device_name: str, elapsed: float, total: float = SELECTION_HOLD_SEC) -> str:
    return f"{device_name} 선택 중  {_progress_dots(elapsed, total)}"

def feedback_selected(device_name: str) -> str:
    return f"{device_name}\n조작 제스처를 취해주세요"

def feedback_fist_countdown(elapsed: float) -> str:
    return f"기기선택 복귀 중  {_progress_dots(elapsed, FIST_HOLD_SEC)}"

def feedback_power(device_name: str, state: str) -> str:
    label = "켜짐" if state == "on" else "꺼짐"
    return f"{device_name}  {label}"

def feedback_knob(device_name: str, value: float, unit: str) -> str:
    return f"{device_name} 조절 중  {value:.1f}{unit}"


# ── 노브 값 포맷 헬퍼 ─────────────────────────────────────────────────────────
def _knob_fmt(value: float, step: float) -> str:
    """step이 정수면 정수 포맷, 소수면 소수점 1자리 포맷."""
    return f"{value:.0f}" if step == int(step) else f"{value:.1f}"


# ── MJPEG 스트리밍 서버 ───────────────────────────────────────────────────────
_flask_app = Flask(__name__)

@_flask_app.route("/stream")
def stream():
    def generate():
        interval = 1.0 / STREAM_FPS
        while True:
            with _frame_lock:
                jpeg = _latest_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            time.sleep(interval)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@_flask_app.route("/snapshot")
def snapshot():
    with _frame_lock:
        jpeg = _latest_jpeg
    if not jpeg:
        return Response(status=503)
    return Response(jpeg, mimetype="image/jpeg")

def _start_stream_server() -> None:
    _flask_app.run(
        host="0.0.0.0",
        port=STREAM_PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


# ── MQTT 콜백 ────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    print(f">> MQTT 연결 성공: {rc}")
    client.publish(SENSOR_CONFIG_TOPIC,   json.dumps(sensor_config),         retain=True)
    client.publish(FEEDBACK_CONFIG_TOPIC, json.dumps(feedback_sensor_config), retain=True)
    print(">> 자동 발견 기기 등록 완료")


# ── 메인 루프 ────────────────────────────────────────────────────────────────
def main():
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands      = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (640, 360), "format": "RGB888"}
        )
    )
    picam2.start()

    threading.Thread(target=_start_stream_server, daemon=True, name="mjpeg-server").start()
    print(f">> 스트리밍 서버: http://0.0.0.0:{STREAM_PORT}/stream")

    global _latest_jpeg

    # ── 상태 변수 ──────────────────────────────────────────────────────────────
    phase: Phase          = Phase.IDLE
    selected_device: dict = {}
    selected_gesture: str = ""

    # SELECTING 단계 타이머
    sel_candidate: str = ""
    sel_start: float   = 0.0

    # SELECTED 단계 — 기기 전환 타이머
    switch_candidate: str = ""
    switch_start: float   = 0.0

    # SELECTED 단계 — 조작 제스처 타이머
    five_h_start: float = 0.0   # FIVE_HORIZONTAL 확인 타이머 (0.0 = 비활성)
    fist_start: float   = 0.0   # FIST 복귀 타이머 (0.0 = 비활성)
    last_ctrl: str      = ""    # 마지막 조작 제스처 (THUMB 중복 발행 방지용)

    # 노브 변수
    knob_start_y: float  = 0.0
    knob_init_val: float = 0.0
    knob_last_pub: float = 0.0
    knob_values: dict[str, float]                   = {}
    pending_restore: dict[str, tuple[float, float]] = {}  # {device_id: (value, publish_after)}

    last_hand_time: float    = time.time()
    last_process_time: float = 0.0
    cached_results           = None
    prev_feedback: str       = ""

    def publish_feedback(text: str) -> None:
        nonlocal prev_feedback
        if text != prev_feedback:
            mqtt_client.publish(TOPIC_FEEDBACK, text)
            prev_feedback = text

    def go_idle() -> None:
        nonlocal phase, selected_device, selected_gesture
        nonlocal sel_candidate, sel_start
        nonlocal switch_candidate, switch_start
        nonlocal five_h_start, fist_start, last_ctrl
        phase            = Phase.IDLE
        selected_device  = {}
        selected_gesture = ""
        sel_candidate    = ""
        sel_start        = 0.0
        switch_candidate = ""
        switch_start     = 0.0
        five_h_start     = 0.0
        fist_start       = 0.0
        last_ctrl        = ""
        publish_feedback(feedback_idle())
        print(">> [IDLE]")

    publish_feedback(feedback_idle())

    try:
        while True:
            frame   = picam2.capture_array()
            frame   = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_h = frame.shape[0]   # 회전 후 세로 픽셀 (~640)

            # 스트리밍 버퍼 갱신
            ret, jpeg_buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )
            if ret:
                with _frame_lock:
                    _latest_jpeg = jpeg_buf.tobytes()

            now = time.time()

            # 노브 복원 대기 처리 (power=on 후 0.3초 지연 발행 — HA 충돌 방지)
            for _did, (_val, _t) in list(pending_restore.items()):
                if now >= _t:
                    _dev = next(d for d in DEVICES.values() if d["id"] == _did)
                    mqtt_client.publish(
                        f"{TOPIC_KNOB_PREFIX}/{_did}",
                        _knob_fmt(_val, _dev["knob_step"])
                    )
                    del pending_restore[_did]
                    print(f">> 노브 복원: {_did} = {_val}")

            # 제스처 처리 주기 제한
            if now - last_process_time < PROCESS_INTERVAL:
                if cached_results and cached_results.multi_hand_landmarks:
                    for hl in cached_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Cammand MVP", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            last_process_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            cached_results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            hand_detected = bool(cached_results and cached_results.multi_hand_landmarks)

            # ── 타임아웃 ────────────────────────────────────────────────────
            if hand_detected:
                last_hand_time = now
            elif now - last_hand_time > TIMEOUT_SEC and phase != Phase.IDLE:
                go_idle()

            if not hand_detected:
                if phase == Phase.IDLE:
                    publish_feedback(feedback_idle())
                cv2.imshow("Cammand MVP", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            hand_landmarks = cached_results.multi_hand_landmarks[0]
            lm             = hand_landmarks.landmark
            wrist_y        = lm[0].y * frame_h   # 픽셀 단위

            # ── IDLE ─────────────────────────────────────────────────────────
            if phase == Phase.IDLE:
                gesture = recognize_selection(hand_landmarks)
                if gesture in DEVICES:
                    phase         = Phase.SELECTING
                    sel_candidate = gesture
                    sel_start     = now
                    publish_feedback(feedback_selecting(DEVICES[gesture]["name"], 0.0))
                    print(f">> [SELECTING] {DEVICES[gesture]['name']}")

            # ── SELECTING ────────────────────────────────────────────────────
            elif phase == Phase.SELECTING:
                gesture = recognize_selection(hand_landmarks)
                if gesture != sel_candidate:
                    if gesture in DEVICES:
                        sel_candidate = gesture
                        sel_start     = now
                        publish_feedback(feedback_selecting(DEVICES[gesture]["name"], 0.0))
                    else:
                        go_idle()
                else:
                    elapsed = now - sel_start
                    publish_feedback(feedback_selecting(DEVICES[gesture]["name"], elapsed))
                    if elapsed >= SELECTION_HOLD_SEC:
                        phase            = Phase.SELECTED
                        selected_device  = DEVICES[gesture]
                        selected_gesture = gesture
                        sel_candidate    = ""
                        sel_start        = 0.0
                        publish_feedback(feedback_selected(selected_device["name"]))
                        print(f">> [SELECTED] {selected_device['name']}")

            # ── SELECTED ─────────────────────────────────────────────────────
            elif phase == Phase.SELECTED:
                ctrl = recognize_control(hand_landmarks)
                sel  = recognize_selection(hand_landmarks)

                if ctrl == "FIVE_HORIZONTAL":
                    # 오인식 방지: FIVE_H_CONFIRM_SEC 이상 연속 유지 시에만 노브 모드 진입
                    fist_start       = 0.0
                    switch_candidate = ""
                    switch_start     = 0.0
                    if five_h_start == 0.0:
                        five_h_start = now
                        last_ctrl    = ctrl
                    elif now - five_h_start >= FIVE_H_CONFIRM_SEC:
                        phase         = Phase.CONTROLLING
                        knob_start_y  = wrist_y
                        knob_init_val = knob_values.get(
                            selected_device["id"],
                            (selected_device["knob_min"] + selected_device["knob_max"]) / 2.0,
                        )
                        knob_last_pub = 0.0
                        last_ctrl     = ""   # CONTROLLING 진입 시 초기화 (중복 발행 방지 변수 리셋)
                        print(f">> [CONTROLLING] {selected_device['name']} (초기값: {knob_init_val:.1f})")
                    else:
                        last_ctrl = ctrl

                elif ctrl == "FIST":
                    # FIST_HOLD_SEC 동안 유지하면 IDLE 복귀
                    five_h_start     = 0.0
                    switch_candidate = ""
                    switch_start     = 0.0
                    last_ctrl        = ctrl
                    if fist_start == 0.0:
                        fist_start = now
                    elapsed = now - fist_start
                    if elapsed >= FIST_HOLD_SEC:
                        go_idle()   # 내부에서 fist_start·last_ctrl 포함 전체 리셋
                    else:
                        publish_feedback(feedback_fist_countdown(elapsed))

                elif ctrl == "THUMB_UP":
                    # 최초 감지 시에만 발행 (홀드 중 중복 발행 방지)
                    five_h_start     = 0.0
                    fist_start       = 0.0
                    switch_candidate = ""
                    switch_start     = 0.0
                    if last_ctrl != "THUMB_UP":
                        did = selected_device["id"]
                        mqtt_client.publish(f"{TOPIC_POWER_PREFIX}/{did}", "on")
                        if did in knob_values:
                            pending_restore[did] = (knob_values[did], now + 0.3)
                        publish_feedback(feedback_power(selected_device["name"], "on"))
                        print(f">> 전원 ON: {did}")
                    last_ctrl = ctrl

                elif ctrl == "THUMB_DOWN":
                    # 최초 감지 시에만 발행 (홀드 중 중복 발행 방지)
                    five_h_start     = 0.0
                    fist_start       = 0.0
                    switch_candidate = ""
                    switch_start     = 0.0
                    if last_ctrl != "THUMB_DOWN":
                        dev     = selected_device
                        did     = dev["id"]
                        off_val = dev["knob_off"]
                        mqtt_client.publish(f"{TOPIC_POWER_PREFIX}/{did}", "off")
                        mqtt_client.publish(f"{TOPIC_KNOB_PREFIX}/{did}", _knob_fmt(off_val, dev["knob_step"]))
                        publish_feedback(feedback_power(dev["name"], "off"))
                        print(f">> 전원 OFF: {did}")
                    last_ctrl = ctrl

                elif sel in DEVICES and sel != selected_gesture:
                    # 기기 전환 후보 — SWITCH_BUFFER_SEC 이후 카운트다운 시작
                    five_h_start = 0.0
                    fist_start   = 0.0
                    last_ctrl    = ""
                    if sel != switch_candidate:
                        switch_candidate = sel
                        switch_start     = now
                    else:
                        elapsed  = now - switch_start
                        buffered = elapsed - SWITCH_BUFFER_SEC
                        if buffered >= SWITCH_HOLD_SEC:
                            selected_device  = DEVICES[sel]
                            selected_gesture = sel
                            switch_candidate = ""
                            switch_start     = 0.0
                            publish_feedback(feedback_selected(selected_device["name"]))
                            print(f">> [SELECTED] 기기 전환: {selected_device['name']}")
                        elif buffered > 0:
                            publish_feedback(
                                feedback_selecting(DEVICES[sel]["name"], buffered, SWITCH_HOLD_SEC)
                            )

                else:
                    # 현재 선택 기기와 동일하거나 UNKNOWN → 모든 타이머 리셋
                    five_h_start     = 0.0
                    fist_start       = 0.0
                    last_ctrl        = ""
                    switch_candidate = ""
                    switch_start     = 0.0

            # ── CONTROLLING ──────────────────────────────────────────────────
            elif phase == Phase.CONTROLLING:
                ctrl = recognize_control(hand_landmarks)
                if ctrl == "FIVE_HORIZONTAL":
                    dev        = selected_device
                    delta_y    = knob_start_y - wrist_y        # 위로 올리면 양수
                    ratio      = max(-1.0, min(1.0, delta_y / (frame_h / 2.0)))
                    knob_range = dev["knob_max"] - dev["knob_min"]
                    raw_val    = knob_init_val + ratio * knob_range
                    new_val    = round(raw_val / dev["knob_step"]) * dev["knob_step"]
                    new_val    = max(dev["knob_min"], min(dev["knob_max"], new_val))

                    if now - knob_last_pub >= 1.0 / KNOB_PUBLISH_HZ:
                        did = dev["id"]
                        mqtt_client.publish(f"{TOPIC_KNOB_PREFIX}/{did}", _knob_fmt(new_val, dev["knob_step"]))
                        knob_values[did] = new_val
                        publish_feedback(feedback_knob(dev["name"], new_val, DEVICE_UNITS.get(did, "")))
                        knob_last_pub = now
                else:
                    # FIVE_HORIZONTAL 해제 → SELECTED 복귀
                    phase        = Phase.SELECTED
                    five_h_start = 0.0
                    last_ctrl    = ""
                    publish_feedback(feedback_selected(selected_device["name"]))
                    print(">> [SELECTED] 노브 종료")

            # ── 랜드마크 렌더링 ───────────────────────────────────────────────
            for hl in cached_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Cammand MVP", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"오류 발생: {e}")
        raise

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        hands.close()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()
