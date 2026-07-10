#!/usr/bin/env python3
# coding=utf-8
"""TTS 语音合成：播报识别结果。

默认使用 edge-tts（微软免费 TTS，中文质量好），
自动回退到 pyttsx3（离线 espeak）。

环境变量：
  TTS_BACKEND = edge | pyttsx3
  TTS_VOICE   = zh-CN-XiaoxiaoNeural | ...  (edge-tts 语音名)
"""

import os
import subprocess
import tempfile
import threading

TTS_BACKEND = os.environ.get("TTS_BACKEND", "edge")
TTS_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")


def _speak_edge(text: str):
    """使用 edge-tts 合成并播放（需要网络）。"""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts 未安装，请执行: pip install edge-tts")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        communicate.save_sync(tmp_path)
        # 播放：优先 ffplay，其次 aplay
        for player in ("ffplay", "aplay", "paplay", "mpg123", "play"):
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run(
                    [player, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
        # 都没找到，直接尝试 ffplay
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _speak_pyttsx3(text: str):
    """使用 pyttsx3 离线合成。"""
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError("pyttsx3 未安装，请执行: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.say(text)
    engine.runAndWait()


def speak(text: str, blocking: bool = True):
    """播报文本。blocking=False 时异步后台播放。"""
    if not text or not text.strip():
        return

    def _run():
        try:
            if TTS_BACKEND == "pyttsx3":
                _speak_pyttsx3(text)
            else:
                _speak_edge(text)
        except Exception as exc:
            print(f"[tts] 播报失败: {exc}")

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, name="tts_worker", daemon=True)
        t.start()
