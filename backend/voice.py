#!/usr/bin/env python3
# coding=utf-8
"""Voice pipeline: 麦克风采集 → 能量VAD → ASR 语音识别 → 指令解析。

支持两种 ASR 后端：
- local  (默认): openai-whisper，Python 3.7+ 兼容
- cloud : OpenAI Whisper API

环境变量：
  VOICE_ASR_BACKEND = local | cloud
  VOICE_MODEL_SIZE  = tiny | small | medium  (whisper 模型大小)
  VOICE_SILENCE_DURATION = 0.8  (静音检测阈值，秒)
  VOICE_SILENCE_THRESHOLD = 0.015  (能量阈值，0~1)
  VOICE_MAX_DURATION = 10  (最长录音时间，秒)
  VOICE_SAMPLE_RATE = 16000
  OPENAI_API_KEY  = sk-...  (cloud 模式必需)
"""

import io
import os
import struct
import threading
import time
import wave
from typing import List, Optional
import numpy as np

_sounddevice_available = False
try:
    import sounddevice as sd

    _sounddevice_available = True
except OSError:
    pass

# ---------------------------------------------------------------------------
# 繁简转换表（语音指令中可能出现的字）
# ---------------------------------------------------------------------------

_T2S_TABLE = str.maketrans({
    "進": "进", "後": "后", "開": "开", "關": "关",
    "轉": "转", "車": "车", "燈": "灯", "聽": "听",
    "聲": "声", "鳴": "鸣", "馬": "马", "門": "门",
    "為": "为", "會": "会", "個": "个", "來": "来",
    "對": "对", "時": "时", "說": "说", "話": "话",
    "讓": "让", "從": "从", "動": "动", "過": "过",
    "頭": "头", "體": "体", "點": "点", "機": "机",
    "氣": "气", "電": "电", "線": "线", "號": "号",
})


def _t2s(text: str) -> str:
    """轻量繁→简转换，无需额外依赖。"""
    return text.translate(_T2S_TABLE)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "16000"))
SILENCE_DURATION = float(os.environ.get("VOICE_SILENCE_DURATION", "0.4"))
SILENCE_THRESHOLD = float(os.environ.get("VOICE_SILENCE_THRESHOLD", "0.015"))
MAX_DURATION = float(os.environ.get("VOICE_MAX_DURATION", "10"))
ASR_BACKEND = os.environ.get("VOICE_ASR_BACKEND", "local")
MODEL_SIZE = os.environ.get("VOICE_MODEL_SIZE", "small")
VOICE_DEBUG = os.environ.get("VOICE_DEBUG", "0") not in ("0", "false", "False")
VOICE_INPUT_DEVICE = os.environ.get("VOICE_INPUT_DEVICE", "")  # 设备名或索引号

# ---------------------------------------------------------------------------
# 语音指令 → 小车动作映射
# ---------------------------------------------------------------------------

CMD_FORWARD = 1
CMD_BACKWARD = 2
CMD_LEFT = 3
CMD_RIGHT = 4
CMD_SPIN_LEFT = 5
CMD_SPIN_RIGHT = 6
CMD_STOP = 0

VOICE_COMMANDS = {
    # 前进
    "前进": ("move", CMD_FORWARD),
    "向前": ("move", CMD_FORWARD),
    "往前走": ("move", CMD_FORWARD),
    "直走": ("move", CMD_FORWARD),
    "forward": ("move", CMD_FORWARD),
    "go": ("move", CMD_FORWARD),
    "go forward": ("move", CMD_FORWARD),
    # 后退
    "后退": ("move", CMD_BACKWARD),
    "向后": ("move", CMD_BACKWARD),
    "倒车": ("move", CMD_BACKWARD),
    "往后退": ("move", CMD_BACKWARD),
    "backward": ("move", CMD_BACKWARD),
    "back": ("move", CMD_BACKWARD),
    "go back": ("move", CMD_BACKWARD),
    # 左转
    "左转": ("move", CMD_LEFT),
    "向左": ("move", CMD_LEFT),
    "向左转": ("move", CMD_LEFT),
    "左边": ("move", CMD_LEFT),
    "left": ("move", CMD_LEFT),
    "turn left": ("move", CMD_LEFT),
    # 右转
    "右转": ("move", CMD_RIGHT),
    "向右": ("move", CMD_RIGHT),
    "向右转": ("move", CMD_RIGHT),
    "右边": ("move", CMD_RIGHT),
    "right": ("move", CMD_RIGHT),
    "turn right": ("move", CMD_RIGHT),
    # 左旋
    "左旋": ("move", CMD_SPIN_LEFT),
    "向左旋": ("move", CMD_SPIN_LEFT),
    "spin left": ("move", CMD_SPIN_LEFT),
    # 右旋
    "右旋": ("move", CMD_SPIN_RIGHT),
    "向右旋": ("move", CMD_SPIN_RIGHT),
    "spin right": ("move", CMD_SPIN_RIGHT),
    # 停止
    "停止": ("stop", None),
    "停下": ("stop", None),
    "停车": ("stop", None),
    "停": ("stop", None),
    "stop": ("stop", None),
    "halt": ("stop", None),
    # 车灯
    "开灯": ("light", True),
    "打开车灯": ("light", True),
    "关灯": ("light", False),
    "关闭车灯": ("light", False),
    "车灯": ("light_toggle", None),
    "light on": ("light", True),
    "light off": ("light", False),
    # 蜂鸣
    "蜂鸣": ("beep", None),
    "喇叭": ("beep", None),
    "beep": ("beep", None),
}


def parse_voice_command(text: str):
    """从识别文本中匹配语音指令，返回 (action, param) 或 None。"""
    # 繁简转换
    text = _t2s(text)
    text_lower = text.strip().lower().rstrip("。，！？,.!?")
    text_lower = text_lower.replace(" ", "")

    # 精确匹配（中文去空格）
    for keyword, result in VOICE_COMMANDS.items():
        kw = keyword.replace(" ", "").lower()
        if kw in text_lower:
            return result

    # 英文模糊匹配
    for keyword, result in VOICE_COMMANDS.items():
        if " " in keyword and keyword.lower() in text.strip().lower():
            return result

    return None


# ---------------------------------------------------------------------------
# 能量 VAD 录音
# ---------------------------------------------------------------------------

def _compute_rms(audio: np.ndarray) -> float:
    """计算音频块的能量（RMS 归一化到 0~1）。"""
    if audio.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    return float(rms)


def record_with_vad(
    sample_rate: int = SAMPLE_RATE,
    silence_duration: float = SILENCE_DURATION,
    silence_threshold: float = SILENCE_THRESHOLD,
    max_duration: float = MAX_DURATION,
    device=None,
) -> Optional[bytes]:
    """使用麦克风录音，能量 VAD 检测到静音后自动停止。

    自动适配多声道设备（如 6+2 麦阵列），取第 0 路作为单声道输入。

    Returns:
        WAV 格式的音频字节，或 None（无有效语音/设备不可用）。
    """
    if not _sounddevice_available:
        raise RuntimeError(
            "sounddevice 不可用（可能缺少 PortAudio 库）。"
            "在 Jetson 上安装：sudo apt-get install libportaudio2 portaudio19-dev"
        )

    # 查询设备声道数，多声道设备只取第 0 路
    device_channels = 1
    try:
        dev_info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
        device_channels = dev_info["max_input_channels"]
    except Exception:
        pass

    use_multichannel = device_channels > 1
    if use_multichannel and VOICE_DEBUG:
        print(f"[voice] 检测到 {device_channels} 声道设备，自动取第 0 路麦克风")

    chunk_samples = int(sample_rate * 0.1)  # 100ms 窗口
    silence_samples = int(silence_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)

    frames: List[np.ndarray] = []
    total_samples = 0
    silent_samples = 0
    has_voice = False
    last_debug = 0.0

    def _callback(indata, _frames_count, _time_info, _status):
        nonlocal total_samples, silent_samples, has_voice, last_debug
        # 多声道设备：只取第 0 路麦克风
        if use_multichannel and indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0:1].copy()  # shape: (N, 1)
        else:
            mono = indata.copy()
        frames.append(mono)
        total_samples += mono.shape[0]
        rms = _compute_rms(mono)
        if VOICE_DEBUG:
            now = time.time()
            if now - last_debug >= 0.5:
                print(f"[voice] RMS={rms:.6f}  threshold={silence_threshold:.6f}  has_voice={has_voice}")
                last_debug = now
        if rms > silence_threshold:
            has_voice = True
            silent_samples = 0
        elif has_voice:
            silent_samples += mono.shape[0]

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=device_channels,
            callback=_callback,
            dtype="int16",
            blocksize=chunk_samples,
            device=device,
        ):
            while True:
                if has_voice and silent_samples >= silence_samples:
                    break
                if total_samples >= max_samples:
                    break
                time.sleep(0.05)
    except sd.PortAudioError as exc:
        raise RuntimeError(f"音频设备错误: {exc}") from exc

    if not has_voice or len(frames) == 0:
        if VOICE_DEBUG:
            print(f"[voice] 未检测到语音 — 请尝试降低 VOICE_SILENCE_THRESHOLD（当前={silence_threshold}）")
        return None

    audio = np.concatenate(frames)
    # 截掉尾部静音
    if has_voice and silent_samples > 0:
        audio = audio[: audio.shape[0] - silent_samples]

    # 写入 WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# openai-whisper 本地 ASR（Python 3.7+）
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        import whisper

        print(f"[voice] loading openai-whisper model '{MODEL_SIZE}' ...")
        _whisper_model = whisper.load_model(MODEL_SIZE)
        print("[voice] openai-whisper model loaded")
        return _whisper_model


def transcribe_local(wav_bytes: bytes) -> Optional[str]:
    """使用 openai-whisper 本地推理。"""
    import tempfile

    model = _get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="zh", fp16=False)
        text = result["text"].strip()
        return text if text else None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OpenAI Whisper API 云端 ASR
# ---------------------------------------------------------------------------

def transcribe_cloud(wav_bytes: bytes) -> Optional[str]:
    """使用 OpenAI Whisper API。"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    if not api_key:
        raise RuntimeError("cloud 模式需要设置 OPENAI_API_KEY 环境变量")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    wav_file = io.BytesIO(wav_bytes)
    wav_file.name = "audio.wav"

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_file,
        language="zh",
        response_format="text",
    )
    text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
    return text if text else None


# ---------------------------------------------------------------------------
# VoicePipeline 统一入口
# ---------------------------------------------------------------------------

class VoicePipeline:
    """语音全链路：录音 + VAD + ASR + 指令解析。"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.backend = ASR_BACKEND
        self._busy = threading.Lock()
        self._device = self._resolve_device()
        if self.debug:
            print(f"[voice] backend={self.backend} model_size={MODEL_SIZE} "
                  f"sr={SAMPLE_RATE} silence={SILENCE_DURATION}s threshold={SILENCE_THRESHOLD}")
            if self._device is not None:
                print(f"[voice] 录音设备: {self._device}")
            self._print_devices()

    def _resolve_device(self):
        """解析录音设备。返回 device id/name 或 None（默认）。"""
        if not _sounddevice_available:
            return None
        raw = VOICE_INPUT_DEVICE.strip()
        if not raw:
            return None
        # 尝试按数字索引解析
        try:
            return int(raw)
        except ValueError:
            pass
        # 按名称模糊匹配
        raw_lower = raw.lower()
        try:
            devices = sd.query_devices()
            for d in devices:
                name = d["name"].lower()
                if raw_lower in name and d["max_input_channels"] > 0:
                    print(f"[voice] 自动匹配录音设备: '{d['name']}' (index={d['index']})")
                    return int(d["index"])
        except Exception:
            pass
        # 直接传名称
        return raw

    def _print_devices(self):
        """打印所有音频设备列表，帮助诊断。"""
        try:
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            print("[voice] 音频设备列表:")
            for d in devices:
                marker = " ← 默认" if d["index"] == default_in else ""
                tag = "IN" if d["max_input_channels"] > 0 else "OUT"
                print(f"  [{d['index']}] {tag} {d['name']} "
                      f"sr={int(d['default_samplerate'])} ch_in={d['max_input_channels']}{marker}")
        except Exception as exc:
            print(f"[voice] 无法列出音频设备: {exc}")

    @property
    def available(self) -> bool:
        """麦克风是否可用。"""
        if not _sounddevice_available:
            return False
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            return len(input_devices) > 0
        except Exception:
            return False

    def transcribe(self) -> dict:
        """录制并识别语音，返回 {"ok": bool, "text": str, "action": str, "param": any}。"""
        if not self.available:
            return {"ok": False, "error": "麦克风不可用", "text": None, "action": None, "param": None}

        acquired = self._busy.acquire(blocking=False)
        if not acquired:
            return {"ok": False, "error": "语音识别正在进行中", "text": None, "action": None, "param": None}

        try:
            # 1. 录音 + VAD
            if self.debug:
                print("[voice] 开始录音...")
            wav_bytes = record_with_vad(device=self._device)
            if wav_bytes is None:
                return {"ok": False, "error": "未检测到有效语音", "text": None, "action": None, "param": None}
            if self.debug:
                print(f"[voice] 录音完成, 大小={len(wav_bytes)} bytes")

            # 2. ASR
            if self.backend == "cloud":
                text = transcribe_cloud(wav_bytes)
            else:
                text = transcribe_local(wav_bytes)

            if not text:
                return {"ok": False, "error": "语音识别结果为空", "text": None, "action": None, "param": None}
            if self.debug:
                print(f"[voice] 识别结果: {text}")

            # 3. 指令解析
            result = parse_voice_command(text)
            if result is None:
                return {"ok": True, "text": text, "action": None, "param": None, "error": "未匹配到指令"}

            action, param = result
            return {"ok": True, "text": text, "action": action, "param": param, "error": None}

        finally:
            self._busy.release()

    def execute_command(self, parsed: dict, hardware) -> dict:
        """将解析结果下发到小车硬件层。

        hardware: RosmasterHardware 实例，需提供 move/stop/set_light/beep 方法。
        """
        action = parsed.get("action")
        param = parsed.get("param")

        if action == "move":
            hardware.move(param)
            return {"executed": True, "action": "move", "cmd": param}
        elif action == "stop":
            hardware.stop()
            return {"executed": True, "action": "stop"}
        elif action == "light":
            hardware.set_light(bool(param))
            return {"executed": True, "action": "light", "on": bool(param)}
        elif action == "light_toggle":
            current = hardware._light_on
            hardware.set_light(not current)
            return {"executed": True, "action": "light", "on": not current}
        elif action == "beep":
            hardware.beep(100)
            return {"executed": True, "action": "beep"}
        return {"executed": False, "error": f"未知动作: {action}"}
