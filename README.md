# Rosmaster APP

独立前后端分离版控制台。复制整个 `APP` 文件夹到小车后运行即可。

## 启动

```bash
cd APP
pip3 install -r requirements.txt
python3 run.py
```

或者：

```bash
bash scripts/start.sh
```

## 端口

- Web 页面和视频流：`6500`
- 控制/传感器 API：`6501`

打开：

```text
http://小车IP:6500/
```

## 常用环境变量

```bash
export ROSMASTER_PORT=/dev/ttyUSB0
export ROSMASTER_CAR_TYPE=1
export ROSMASTER_CAMERA_DEVICE=/dev/video0
export ROSMASTER_VIDEO_WIDTH=640
export ROSMASTER_VIDEO_HEIGHT=480
export ROSMASTER_VIDEO_FPS=20
export ROSMASTER_VIDEO_QUALITY=80
export ROSMASTER_WATCHDOG_TIMEOUT=3.0
python3 run.py
```

车型：`X3=1`，`X3PLUS=2`，`X1=4`，`R2=5`。

## API

- `GET  /api/ping`
- `GET  /api/status`
- `GET  /api/sensors`
- `POST /api/move`，JSON: `{ "cmd": 1, "speed": 50 }`
- `POST /api/stop`
- `POST /api/light`，JSON: `{ "on": true }`
- `POST /api/beep`，JSON: `{ "duration": 100 }`

### 人物追踪

- `POST /api/track/start` — 启动 YOLOv8 人物检测与居中追踪
- `POST /api/track/stop`  — 停止追踪
- `GET  /api/track/status` — 追踪状态 + 当前检测框

方向命令：`0=停止`，`1=前进`，`2=后退`，`3=左`，`4=右`，`5=左旋`，`6=右旋`。

## 语音控制

`POST /api/voice` — 按下即录音，VAD 自动检测语音结束，ASR 识别为指令后执行并 TTS 播报。

### 环境变量

```bash
# 语音输入设备（六麦阵列等）
VOICE_INPUT_DEVICE=XFM      # 设备名或索引（如 plughw:2,0）
VOICE_MODEL_SIZE=tiny       # whisper 模型: tiny/small/medium
VOICE_SILENCE_THRESHOLD=0.005  # VAD 能量阈值，越小越灵敏

# TTS 播报
TTS_BACKEND=edge            # edge（微软云端，中文好）| pyttsx3（离线）
TTS_VOICE=zh-CN-XiaoxiaoNeural  # edge-tts 语音名

# 调试（按需开启）
VOICE_DEBUG=1               # 语音 RMS/VAD 日志
HARDWARE_DEBUG=1            # 底盘串口日志

# 人物追踪
TRACK_MODEL_PATH=yolov8n.pt     # YOLO 模型路径
TRACK_CONF_THRES=0.4            # 检测置信度阈值
TRACK_DEADZONE=40               # 居中死区(像素)
TRACK_PID_P=0.003               # 偏移→转速比例系数
```
