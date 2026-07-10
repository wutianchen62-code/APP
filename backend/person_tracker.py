#!/usr/bin/env python3
# coding=utf-8
"""YOLOv8 人体检测 + 居中追踪。

判定逻辑：
  "最近的人" = 摄像头画面中检测框面积最大的人（像素面积越大 → 越近）。
  预留 depth_fallback() 方法，后续接入深度相机后只需在此方法中替换距离计算。

控制逻辑：
  人的水平中心偏离画面中心的像素差 → 换算为旋转速度 → 下发串口指令。

环境变量:
  TRACK_PID_P = 0.003       # 比例系数: 像素偏差 → 转速
  TRACK_DEADZONE = 40       # 死区像素: 偏差在此范围内不转
  TRACK_MIN_SPEED = 10      # 最小转速
  TRACK_MAX_SPEED = 60      # 最大转速
  TRACK_MODEL_PATH = backend/yolov8n.pt  # 模型路径
  TRACK_CONF_THRES = 0.4    # 检测置信度阈值
  TRACK_INTERVAL = 0.15     # 控制循环间隔(秒)
"""

import os
import threading
import time
from typing import Optional, Tuple, List

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "TRACK_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt"),
)
TRACK_PID_P = float(os.environ.get("TRACK_PID_P", "0.003"))
TRACK_DEADZONE = int(os.environ.get("TRACK_DEADZONE", "40"))
TRACK_MIN_SPEED = int(os.environ.get("TRACK_MIN_SPEED", "10"))
TRACK_MAX_SPEED = int(os.environ.get("TRACK_MAX_SPEED", "60"))
TRACK_CONF_THRES = float(os.environ.get("TRACK_CONF_THRES", "0.4"))
TRACK_INTERVAL = float(os.environ.get("TRACK_INTERVAL", "0.15"))

# YOLO class 0 → person
PERSON_CLASS = 0


# ---------------------------------------------------------------------------
# 检测结果
# ---------------------------------------------------------------------------
class Detection:
    __slots__ = ("bbox", "conf", "area", "cx", "cy")
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float
    area: float
    cx: float  # 水平中心
    cy: float  # 垂直中心

    def __init__(self, bbox, conf):
        self.bbox = bbox
        self.conf = conf
        x1, y1, x2, y2 = bbox
        self.area = (x2 - x1) * (y2 - y1)
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0


# ---------------------------------------------------------------------------
# PersonTracker
# ---------------------------------------------------------------------------
class PersonTracker:
    """基于 YOLOv8 的人体检测与居中追踪。

    设计上预留了深度相机扩展点 depth_fallback()，
    未来接入深度相机后只需修改此处即可切换到真实距离判定。
    """

    def __init__(
        self,
        video_streamer,    # VideoStreamer instance (提供 raw frames)
        hardware,          # RosmasterHardware instance (提供 spin_left/spin_right/stop)
        debug: bool = False,
    ):
        self._video = video_streamer
        self._hardware = hardware
        self.debug = debug

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self._latest_detection: Optional[Detection] = None
        self._lock = threading.Lock()

    # ------ 模型加载 ------

    def _load_model(self):
        from ultralytics import YOLO
        path = os.path.abspath(MODEL_PATH)
        if self.debug:
            print(f"[tracker] loading YOLO model: {path}")
        self._model = YOLO(path)
        if self.debug:
            print("[tracker] YOLO model loaded")

    # ------ 启动 / 停止 ------

    def start(self):
        if self._running:
            return
        if self._model is None:
            self._load_model()
        self._running = True
        self._thread = threading.Thread(target=self._track_loop, name="person_tracker", daemon=True)
        self._thread.start()
        if self.debug:
            print("[tracker] started")

    def stop(self):
        self._running = False
        self._hardware.stop()
        if self.debug:
            print("[tracker] stopped")

    # ------ 状态查询 ------

    @property
    def active(self) -> bool:
        return self._running

    def get_detection(self) -> Optional[dict]:
        """返回当前最大人体检测结果，供前端渲染叠加层。"""
        with self._lock:
            d = self._latest_detection
        if d is None:
            return None
        return {
            "bbox": list(d.bbox),
            "conf": round(float(d.conf), 3),
            "area": int(d.area),
            "cx": round(d.cx, 1),
        }

    # ------ 深度判定占位（未来替换） ------

    def _find_nearest_person(self, detections: List[Detection]) -> Optional[Detection]:
        """选出"最近"的人。

        当前实现: 像素面积最大（面积 ≈ 距离²，物理上合理）。
        未来接入深度相机后，替换为: 遍历检测框，取 bbox 中心点对应的深度值最小者。
        """
        return self._depth_fallback(detections)

    def _depth_fallback(self, detections: List[Detection]) -> Optional[Detection]:
        """无深度相机时的回退策略：取面积最大的人。

        未来替换为:
            depth_map = get_depth_frame()
            for d in detections:
                d.depth = depth_map[int(d.cy), int(d.cx)]
            return min(detections, key=lambda d: d.depth)
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    # ------ 控制计算 ------

    def _compute_control(self, nearest: Detection, frame_width: int) -> int:
        """根据人的水平偏移计算旋转命令和速度。

        Returns:
            int: 正=右旋, 负=左旋, 0=停止
        """
        center_x = frame_width / 2.0
        offset = nearest.cx - center_x  # 正=人在右边, 负=人在左边

        if abs(offset) <= TRACK_DEADZONE:
            return 0

        # PID 比例控制: offset * gain, 限制在 min/max 之间
        raw = abs(offset) * TRACK_PID_P
        speed = max(TRACK_MIN_SPEED, min(TRACK_MAX_SPEED, int(raw)))

        if offset > 0:
            return speed   # 人在右边 → 右旋
        else:
            return -speed  # 人在左边 → 左旋

    # ------ 主循环 ------

    def _track_loop(self):
        while self._running:
            # 取当前帧（raw BGR）
            frame = self._video.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # YOLO 推理
            try:
                results = self._model(frame, conf=TRACK_CONF_THRES, verbose=False)
            except Exception as exc:
                if self.debug:
                    print(f"[tracker] inference error: {exc}")
                time.sleep(TRACK_INTERVAL)
                continue

            # 提取人体检测
            detections: List[Detection] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        if cls_id != PERSON_CLASS:
                            continue
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        detections.append(Detection(bbox, conf))

            # 找最近的人
            nearest = self._find_nearest_person(detections)

            with self._lock:
                self._latest_detection = nearest

            # 下发控制
            if nearest is None:
                self._hardware.stop()
            else:
                cmd = self._compute_control(nearest, w)
                self._apply_command(cmd)

            if self.debug and nearest is not None:
                print(f"[tracker] person at cx={nearest.cx:.0f} area={nearest.area:.0f} → cmd={cmd}")

            time.sleep(TRACK_INTERVAL)

    def _apply_command(self, cmd: int):
        """将计算结果转为底盘指令。"""
        if cmd == 0:
            self._hardware.stop()
            return
        # 正=右旋(6), 负=左旋(5)
        direction = 6 if cmd > 0 else 5
        speed = abs(cmd)
        self._hardware.move(direction, speed)
