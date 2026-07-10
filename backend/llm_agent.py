#!/usr/bin/env python3
# coding=utf-8
"""LLM agent for OpenAI-compatible chat models.

The default configuration targets Volcengine Ark's OpenAI-compatible API.
Secrets are read from environment variables, typically loaded from a local .env file.
"""

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI


class LLMAgent:
    """Turn natural language into safe car actions or conversational replies."""

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    ALLOWED_ACTIONS = {
        "none",
        "move",
        "stop",
        "light",
        "beep",
        "track_start",
        "track_stop",
    }
    ALLOWED_CMDS = {0, 1, 2, 3, 4, 5, 6}

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.base_url = os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL).strip()
        self.model = os.environ.get("LLM_MODEL", "").strip()
        self.enabled = os.environ.get("LLM_ENABLED", "1") not in ("0", "false", "False")
        self.history = []
        self.client: Optional[OpenAI] = None

        if not self.enabled:
            return

        if not self.api_key or not self.model:
            print("[llm] 未配置 OPENAI_API_KEY 或 LLM_MODEL，大模型功能将不可用")
            self.enabled = False
            return

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def available(self) -> bool:
        return self.enabled and self.client is not None

    def ask(self, user_text: str, car_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask the LLM to produce a sanitized action JSON."""
        if not self.available or self.client is None:
            return {
                "type": "chat",
                "action": "none",
                "cmd": None,
                "speed": None,
                "duration": None,
                "light_on": None,
                "reply": "大模型还没有配置，请先设置 OPENAI_API_KEY 和 LLM_MODEL。",
            }

        system_prompt = """
你是 Rosmaster 小车的智能语音助手。

你可以做两类事情：
1. 和用户自然对话。
2. 把用户的话转换成安全的小车控制指令。

你必须只输出 JSON，不要输出 Markdown，不要解释。

小车方向 cmd 定义：
0 = 停止
1 = 前进
2 = 后退
3 = 左转
4 = 右转
5 = 左旋
6 = 右旋

允许的 action：
- "none": 不执行动作，只聊天
- "move": 移动小车
- "stop": 停止小车
- "light": 控制车灯
- "beep": 蜂鸣
- "track_start": 启动人物追踪
- "track_stop": 停止人物追踪

你必须输出如下 JSON：
{
  "type": "chat" 或 "control",
  "action": "none" | "move" | "stop" | "light" | "beep" | "track_start" | "track_stop",
  "cmd": 0-6 或 null,
  "speed": 0-100 或 null,
  "duration": 秒数或 null,
  "light_on": true 或 false 或 null,
  "reply": "回复用户的话"
}

安全规则：
- 如果用户没有明确要求移动，不要让小车移动。
- 如果用户说停止、停下、别动、停车，必须输出 stop。
- 如果用户要求前进、后退、转向，默认 duration 不超过 1 秒。
- 不允许高速、长时间、撞击、冲刺等危险动作。
- 如果用户要求危险动作，应拒绝，并说明为了安全不能执行。
- reply 必须简短、自然、中文。
""".strip()

        state_text = ""
        if car_state:
            state_text = "当前小车状态：\n" + json.dumps(car_state, ensure_ascii=False)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": f"{state_text}\n\n用户说：{user_text}"})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
            )
            content = response.choices[0].message.content.strip()
        except Exception as exc:
            if self.debug:
                print(f"[llm] request failed: {exc}")
            return {
                "type": "chat",
                "action": "none",
                "cmd": None,
                "speed": None,
                "duration": None,
                "light_on": None,
                "reply": "大模型请求失败，请检查网络和火山方舟配置。",
                "error": str(exc),
            }

        if self.debug:
            print(f"[llm] raw: {content}")

        data = self._loads_json(content)
        data = self._sanitize(data)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": data["reply"]})
        self.history = self.history[-12:]

        return data

    def _loads_json(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {
            "type": "chat",
            "action": "none",
            "cmd": None,
            "speed": None,
            "duration": None,
            "light_on": None,
            "reply": "我刚才没有理解清楚，请再说一遍。",
        }

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = data.get("action", "none")
        if action not in self.ALLOWED_ACTIONS:
            action = "none"

        cmd = self._to_int_or_none(data.get("cmd"))
        if cmd not in self.ALLOWED_CMDS:
            cmd = None

        speed = self._to_int_or_none(data.get("speed"))
        if speed is not None:
            speed = max(0, min(60, speed))

        duration = self._to_float_or_none(data.get("duration"))
        if duration is not None:
            duration = max(0.1, min(2.0, duration))

        light_on = data.get("light_on")
        if not isinstance(light_on, bool):
            light_on = None

        reply = str(data.get("reply") or "好的。").strip()

        if action == "move" and cmd is None:
            action = "none"
            reply = "我没有识别到明确的移动方向。"

        if action == "light" and light_on is None:
            action = "none"
            reply = "我没有识别到要开灯还是关灯。"

        return {
            "type": data.get("type", "chat") if data.get("type") in ("chat", "control") else "chat",
            "action": action,
            "cmd": cmd,
            "speed": speed,
            "duration": duration,
            "light_on": light_on,
            "reply": reply,
        }

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
