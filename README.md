# Rosmaster APP

独立前后端分离版控制台。复制整个 `APP` 文件夹到小车后运行即可。

## 功能概览

- Web 控制台和 MJPEG 实时视频流
- 小车方向控制、急停、车灯、蜂鸣器
- 温湿度、电池电压读取
- Whisper 语音识别
- TTS 语音播报
- YOLOv8 人物检测与居中追踪
- 火山方舟大模型接入：自然语言对话 + 自然语言控制小车

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

## 首次配置步骤

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

如果你在 Windows 本地临时调试，也可以使用：

```bash
pip install -r requirements.txt
```

### 2. 创建本地 `.env` 配置文件

项目使用 `.env` 保存本地隐私配置，例如火山方舟 API Key。真实 `.env` 文件已经被 `.gitignore` 忽略，不要提交到 Git。

复制模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

### 3. 修改 `.env`

打开项目根目录下的 `.env`，至少填写以下内容：

```bash
OPENAI_API_KEY=你的火山方舟API_KEY
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=你的火山方舟推理接入点ID
LLM_ENABLED=1
LLM_DEBUG=1
```

注意：`LLM_MODEL` 通常填写火山方舟控制台里的“推理接入点 ID”，一般形如：

```bash
LLM_MODEL=ep-xxxxxxxxxxxxxxxx
```

不要把真实 API Key 写进代码，也不要写进 `.env.example`。

### 4. 配置小车硬件参数

在 `.env` 中根据实际设备修改：

```bash
ROSMASTER_PORT=/dev/ttyUSB0
ROSMASTER_CAR_TYPE=1
ROSMASTER_CAMERA_DEVICE=/dev/video0
ROSMASTER_VIDEO_WIDTH=640
ROSMASTER_VIDEO_HEIGHT=480
ROSMASTER_VIDEO_FPS=20
ROSMASTER_VIDEO_QUALITY=80
ROSMASTER_WATCHDOG_TIMEOUT=3.0
```

车型：`X3=1`，`X3PLUS=2`，`X1=4`，`R2=5`。

如果在 Windows 本地临时调试，摄像头和串口可能类似：

```bash
ROSMASTER_CAMERA_DEVICE=0
ROSMASTER_PORT=COM3
```

### 5. 启动程序

```bash
python3 run.py
```

Windows：

```bash
python run.py
```

启动日志中如果看到：

```text
LLM   : 可用
```

说明大模型配置已生效。如果看到：

```text
LLM   : 未配置/不可用
```

请检查 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL`。

## 火山方舟大模型接入

本项目通过 `openai` Python SDK 使用火山方舟 OpenAI-compatible API。

### 必填配置

```bash
OPENAI_API_KEY=你的火山方舟API_KEY
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=你的火山方舟推理接入点ID
```

### 可选配置

```bash
LLM_ENABLED=1              # 1=启用大模型，0=关闭大模型
LLM_DEBUG=0                # 1=打印大模型原始返回，调试时可开启
LLM_TEMPERATURE=0.2        # 模型温度，越低越稳定
LLM_DEFAULT_SPEED=40       # 大模型控制小车时的默认速度
LLM_DEFAULT_DURATION=0.8   # 大模型控制小车时的默认移动时间，单位秒
```

### 大模型安全限制

后端不会直接相信大模型输出，会进行二次校验：

- 只允许动作：`none`、`move`、`stop`、`light`、`beep`、`track_start`、`track_stop`
- 方向命令只允许：`0` 到 `6`
- 速度最大限制为 `60`
- 单次移动最长限制为 `2.0` 秒
- 未识别到明确方向时不会移动
- 未识别到开灯/关灯时不会执行灯光动作

因此即使大模型返回异常内容，小车也不会长时间高速运动。

## API

- `GET  /api/ping`
- `GET  /api/status`
- `GET  /api/sensors`
- `POST /api/move`，JSON: `{ "cmd": 1, "speed": 50 }`
- `POST /api/stop`
- `POST /api/light`，JSON: `{ "on": true }`
- `POST /api/beep`，JSON: `{ "duration": 100 }`

方向命令：`0=停止`，`1=前进`，`2=后退`，`3=左`，`4=右`，`5=左旋`，`6=右旋`。

### 文本大模型交互

`POST /api/chat` — 用文本和小车对话，也可以通过自然语言控制小车。

请求示例：

```json
{
  "text": "你是谁？",
  "speak": false
}
```

控制示例：

```json
{
  "text": "小车往前走一点",
  "speak": true
}
```

字段说明：

- `text`：用户输入文本
- `speak`：是否用 TTS 播报回复，默认 `true`

返回中主要字段：

- `reply`：小车回复的话
- `llm`：大模型解析后的结构化结果
- `exec`：小车动作执行结果

建议先用 `/api/chat` 测试大模型配置，再测试语音。

### 语音大模型交互

`POST /api/voice` — 服务端录音，VAD 自动检测语音结束，Whisper 识别文字后交给大模型理解，再执行动作或对话，最后 TTS 播报回复。

现在语音链路是：

```text
麦克风录音 → VAD → Whisper ASR → 火山方舟大模型 → 安全执行 → TTS 播报
```

可尝试说：

```text
你是谁
往前走一点
停下
打开车灯
关闭车灯
蜂鸣一下
开始人物追踪
停止人物追踪
```

### 人物追踪

- `POST /api/track/start` — 启动 YOLOv8 人物检测与居中追踪
- `POST /api/track/stop`  — 停止追踪
- `GET  /api/track/status` — 追踪状态 + 当前检测框

## 语音控制配置

```bash
# 语音输入设备（六麦阵列等）
VOICE_INPUT_DEVICE=XFM         # 设备名或索引（如 plughw:2,0）
VOICE_ASR_BACKEND=local        # local | cloud
VOICE_MODEL_SIZE=tiny          # whisper 模型: tiny/small/medium
VOICE_SILENCE_THRESHOLD=0.005  # VAD 能量阈值，越小越灵敏
VOICE_SILENCE_DURATION=0.4     # 静音持续多久后结束录音，单位秒
VOICE_MAX_DURATION=10          # 单次最长录音时间，单位秒

# TTS 播报
TTS_BACKEND=edge               # edge（微软云端，中文好）| pyttsx3（离线）
TTS_VOICE=zh-CN-XiaoxiaoNeural # edge-tts 语音名

# 调试（按需开启）
VOICE_DEBUG=1                  # 语音 RMS/VAD 日志
HARDWARE_DEBUG=1               # 底盘串口日志
LLM_DEBUG=1                    # 大模型调试日志
```

## 人物追踪配置

```bash
TRACK_MODEL_PATH=yolov8n.pt  # YOLO 模型路径
TRACK_CONF_THRES=0.4         # 检测置信度阈值
TRACK_DEADZONE=40            # 居中死区(像素)
TRACK_PID_P=0.003            # 偏移→转速比例系数
```

## 测试建议

### 1. 先测试大模型文本接口

Linux/macOS：

```bash
curl -X POST http://127.0.0.1:6501/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"你是谁？","speak":false}'
```

Windows PowerShell：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:6501/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"你是谁？","speak":false}'
```

### 2. 再测试小车控制

测试控制前，请确保小车处于安全区域，必要时先架空车轮。

```bash
curl -X POST http://127.0.0.1:6501/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"往前走一点","speak":false}'
```

### 3. 最后测试语音控制

打开 Web 页面：

```text
http://小车IP:6500/
```

点击语音按钮后说话，例如：

```text
往前走一点
你能做什么
停下
```

## 隐私与安全注意事项

- 真实 API Key 只写入 `.env`
- 不要提交 `.env`
- `.env.example` 只放模板，不放真实密钥
- 如果连接公共网络，建议后续给控制 API 增加鉴权
- 测试大模型控制小车时，先使用低速、短时长，并保持小车周围安全
