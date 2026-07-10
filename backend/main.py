#!/usr/bin/env python3
# coding=utf-8
"""APP backend entrypoint.

Ports:
- 6500: static frontend and MJPEG video stream
- 6501: control/sensor REST API
"""

import os
import signal
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, make_response
from gevent import pywsgi

from .hardware import RosmasterHardware, VideoConfig, VideoStreamer
from .voice import VoicePipeline
from . import tts
from .person_tracker import PersonTracker
from .llm_agent import LLMAgent

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

WEB_PORT = int(os.environ.get("APP_WEB_PORT", "6500"))
API_PORT = int(os.environ.get("APP_API_PORT", "6501"))
DEBUG = os.environ.get("APP_DEBUG", "0") not in ("0", "false", "False")

VIDEO_CONFIG = VideoConfig(
    device=os.environ.get("ROSMASTER_CAMERA_DEVICE", "/dev/video0"),
    width=int(os.environ.get("ROSMASTER_VIDEO_WIDTH", "640")),
    height=int(os.environ.get("ROSMASTER_VIDEO_HEIGHT", "480")),
    fps=max(1, int(os.environ.get("ROSMASTER_VIDEO_FPS", "20"))),
    quality=max(30, min(95, int(os.environ.get("ROSMASTER_VIDEO_QUALITY", "80")))),
    log_interval=max(0.5, float(os.environ.get("ROSMASTER_VIDEO_LOG_INTERVAL", "3"))),
)

hardware = RosmasterHardware(debug=os.environ.get("HARDWARE_DEBUG", "0") not in ("0", "false", "False"))
video = VideoStreamer(VIDEO_CONFIG, debug=os.environ.get("VIDEO_DEBUG", "0") not in ("0", "false", "False"))
voice = VoicePipeline(debug=os.environ.get("VOICE_DEBUG", "0") not in ("0", "false", "False"))
tracker = PersonTracker(video, hardware, debug=os.environ.get("TRACK_DEBUG", "0") not in ("0", "false", "False"))
llm_agent = LLMAgent(debug=os.environ.get("LLM_DEBUG", "0") not in ("0", "false", "False"))
web_app = Flask("rosmaster_web")
api_app = Flask("rosmaster_api")


def add_common_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Cache-Control"] = "no-store"
    return response


def get_car_state() -> dict:
    return {
        "status": hardware.get_status(),
        "sensors": hardware.get_sensors(),
        "camera_ok": video.camera_ok,
        "voice_available": voice.available,
        "tracking": tracker.active,
        "detection": tracker.get_detection(),
    }


def execute_llm_action(result: dict) -> dict:
    action = result.get("action")

    if action == "none":
        return {"executed": False, "action": "none"}

    if action == "move":
        cmd = result.get("cmd")
        speed = result.get("speed") or int(os.environ.get("LLM_DEFAULT_SPEED", "40"))
        duration = result.get("duration") or float(os.environ.get("LLM_DEFAULT_DURATION", "0.8"))

        if cmd is None:
            return {"executed": False, "error": "缺少移动指令"}

        hardware.move(cmd, speed)

        def delayed_stop():
            time.sleep(duration)
            hardware.stop()

        threading.Thread(target=delayed_stop, name="llm_delayed_stop", daemon=True).start()
        return {
            "executed": True,
            "action": "move",
            "cmd": cmd,
            "speed": speed,
            "duration": duration,
        }

    if action == "stop":
        hardware.stop()
        return {"executed": True, "action": "stop"}

    if action == "light":
        on = bool(result.get("light_on"))
        hardware.set_light(on)
        return {"executed": True, "action": "light", "on": on}

    if action == "beep":
        hardware.beep(100)
        return {"executed": True, "action": "beep"}

    if action == "track_start":
        if not video.camera_ok:
            return {"executed": False, "error": "摄像头不可用"}
        tracker.start()
        return {"executed": True, "action": "track_start"}

    if action == "track_stop":
        tracker.stop()
        return {"executed": True, "action": "track_stop"}

    return {"executed": False, "error": f"未知动作: {action}"}


web_app.after_request(add_common_headers)
api_app.after_request(add_common_headers)


@api_app.route("/api/<path:path>", methods=["OPTIONS"])
def api_cors_preflight(path):
    """统一处理所有 /api/* 的 OPTIONS 预检请求。"""
    resp = make_response("", 204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@web_app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@web_app.route("/<path:path>")
def static_files(path):
    target = FRONTEND_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


@web_app.route("/video_feed")
def video_feed():
    return Response(video.mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@api_app.route("/api/ping", methods=["GET"])
def api_ping():
    return jsonify({"ok": True, "service": "rosmaster-api", "port": API_PORT})


@api_app.route("/api/status", methods=["GET"])
def api_status():
    data = hardware.get_status()
    data["camera_ok"] = video.camera_ok
    data["voice_available"] = voice.available
    data["llm_available"] = llm_agent.available
    data["tracking"] = tracker.active
    data["video"] = {
        "device": VIDEO_CONFIG.device,
        "width": VIDEO_CONFIG.width,
        "height": VIDEO_CONFIG.height,
        "fps": VIDEO_CONFIG.fps,
        "quality": VIDEO_CONFIG.quality,
        "frame_age": None if video.latest_age() is None else round(video.latest_age(), 3),
    }
    return jsonify({"ok": True, "data": data})


@api_app.route("/api/sensors", methods=["GET"])
def api_sensors():
    return jsonify({"ok": True, "data": hardware.get_sensors()})


@api_app.route("/api/move", methods=["POST"])
def api_move():
    body = request.get_json(silent=True)
    print(f"[api] move raw json: {body!r}, data={request.data!r}, content_type={request.content_type!r}")
    body = body or {}
    cmd = int(body.get("cmd", 0))
    speed = body.get("speed")
    hardware.move(cmd, speed)
    return jsonify({"ok": True, "cmd": cmd, "speed": speed})


@api_app.route("/api/joystick", methods=["POST"])
def api_joystick():
    body = request.get_json(silent=True) or {}
    hardware.move_joystick(body.get("x", 0), body.get("y", 0))
    return jsonify({"ok": True})


@api_app.route("/api/stop", methods=["POST"])
def api_stop():
    hardware.stop()
    return jsonify({"ok": True})


@api_app.route("/api/light", methods=["POST"])
def api_light():
    body = request.get_json(silent=True) or {}
    on = bool(body.get("on", False))
    hardware.set_light(on)
    return jsonify({"ok": True, "light_on": on})


@api_app.route("/api/beep", methods=["POST"])
def api_beep():
    body = request.get_json(silent=True) or {}
    hardware.beep(body.get("duration", 80))
    return jsonify({"ok": True})


@api_app.route("/api/voice", methods=["POST"])
def api_voice():
    """语音交互：录音 → ASR → 大模型理解 → 执行/对话 → TTS 回复。"""
    try:
        parsed = voice.transcribe()
        if not parsed.get("text"):
            return jsonify(parsed)

        user_text = parsed["text"]
        llm_result = llm_agent.ask(user_text, car_state=get_car_state())
        exec_result = execute_llm_action(llm_result)
        reply = llm_result.get("reply") or "好的。"
        tts.speak(reply, blocking=False)

        return jsonify({
            "ok": True,
            "text": user_text,
            "llm": llm_result,
            "exec": exec_result,
            "reply": reply,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        tts.speak("处理失败，请稍后再试。", blocking=False)
        return jsonify({"ok": False, "error": str(exc), "text": None, "reply": None}), 500


@api_app.route("/api/chat", methods=["POST"])
def api_chat():
    """文本交互入口，便于先调试大模型和小车指令。"""
    try:
        body = request.get_json(silent=True) or {}
        text = str(body.get("text", "")).strip()
        if not text:
            return jsonify({"ok": False, "error": "text 不能为空"}), 400

        llm_result = llm_agent.ask(text, car_state=get_car_state())
        exec_result = execute_llm_action(llm_result)
        reply = llm_result.get("reply") or "好的。"

        if body.get("speak", True):
            tts.speak(reply, blocking=False)

        return jsonify({
            "ok": True,
            "text": text,
            "llm": llm_result,
            "exec": exec_result,
            "reply": reply,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        tts.speak("处理失败，请稍后再试。", blocking=False)
        return jsonify({"ok": False, "error": str(exc), "text": None, "reply": None}), 500


@api_app.route("/api/track/start", methods=["POST"])
def api_track_start():
    """启动人物追踪。"""
    if not video.camera_ok:
        return jsonify({"ok": False, "error": "摄像头不可用"})
    tracker.start()
    return jsonify({"ok": True, "tracking": True})


@api_app.route("/api/track/stop", methods=["POST"])
def api_track_stop():
    """停止人物追踪。"""
    tracker.stop()
    return jsonify({"ok": True, "tracking": False})


@api_app.route("/api/track/status", methods=["GET"])
def api_track_status():
    """获取追踪状态和当前检测结果。"""
    return jsonify({
        "ok": True,
        "tracking": tracker.active,
        "detection": tracker.get_detection(),
    })


def shutdown(*_args):
    print("Shutting down...")
    tracker.stop()
    video.stop()
    hardware.shutdown()
    sys.exit(0)


def _run_api():
    """在独立线程中启动 Flask API，带异常捕获。"""
    try:
        api_app.run(host="0.0.0.0", port=API_PORT, threaded=True, use_reloader=False)
    except Exception as exc:
        import traceback
        print(f"[api] Flask API 崩溃: {exc}")
        traceback.print_exc()


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    video.start()

    print("Rosmaster APP")
    print(f"Web   : http://0.0.0.0:{WEB_PORT}/")
    print(f"Video : http://0.0.0.0:{WEB_PORT}/video_feed")
    print(f"API   : http://0.0.0.0:{API_PORT}/api/")
    print(
        "Video settings: "
        f"device={VIDEO_CONFIG.device}, {VIDEO_CONFIG.width}x{VIDEO_CONFIG.height}@{VIDEO_CONFIG.fps}fps, "
        f"quality={VIDEO_CONFIG.quality}"
    )
    print(f"Voice : {'可用' if voice.available else '不可用'} (backend={voice.backend})")
    print(f"LLM   : {'可用' if llm_agent.available else '未配置/不可用'}")

    api_thread = threading.Thread(
        target=_run_api, name="api_http", daemon=True,
    )
    api_thread.start()
    time.sleep(0.5)
    # 验证 API 是否真正启动
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/api/ping", timeout=3)
        print(f"API   : http://0.0.0.0:{API_PORT}/api/ [OK]")
    except Exception:
        print(f"API   : http://0.0.0.0:{API_PORT}/api/ [启动失败!]")

    server = pywsgi.WSGIServer(("0.0.0.0", WEB_PORT), web_app)
    server.serve_forever()


if __name__ == "__main__":
    main()
