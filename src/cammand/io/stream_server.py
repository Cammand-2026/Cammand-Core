from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

from ..config import settings

_app = Flask(__name__)
_frame_lock = threading.Lock()
_latest_jpeg: bytes = b""


@_app.route("/stream")
def _stream() -> Response:
    def generate():
        interval = 1.0 / settings.stream_fps
        while True:
            with _frame_lock:
                jpeg = _latest_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(interval)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@_app.route("/snapshot")
def _snapshot() -> Response:
    with _frame_lock:
        jpeg = _latest_jpeg
    if not jpeg:
        return Response(status=503)
    return Response(jpeg, mimetype="image/jpeg")


class MjpegServer:
    """Flask MJPEG 스트리밍 서버. daemon thread에서 실행."""

    def push(self, frame: np.ndarray) -> None:
        """메인 루프에서 호출: 최신 프레임을 JPEG로 인코딩해 버퍼에 저장."""
        global _latest_jpeg
        ret, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.stream_jpeg_quality]
        )
        if ret:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()

    def run(self) -> None:
        """daemon thread에서 실행. Flask 서버를 blocking으로 기동."""
        _app.run(
            host="0.0.0.0",
            port=settings.stream_port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
