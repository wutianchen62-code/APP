#!/usr/bin/env python3
# coding=utf-8
"""Rosmaster hardware layer: serial control, sensors, and lightweight camera capture."""

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import serial
from Rosmaster_Lib import Rosmaster
from serial.tools import list_ports


@dataclass(frozen=True)
class VideoConfig:
    device: str
    width: int
    height: int
    fps: int
    quality: int
    log_interval: float


class VideoStreamer:
    def __init__(self, config: VideoConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self._capture: Optional[cv.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera_ok = False

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="video_capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._camera_ok = False

    def latest_age(self) -> Optional[float]:
        with self._lock:
            if self._latest_jpeg is None:
                return None
            return time.time() - self._latest_time

    def get_latest_frame(self) -> Optional["np.ndarray"]:
        """获取最新的 raw BGR 帧（供 YOLO 等 CV 模块使用）。"""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _open_capture(self) -> cv.VideoCapture:
        source = self.config.device
        try:
            source_value = int(source)
        except ValueError:
            source_value = source

        cap = cv.VideoCapture(source_value)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc("M", "J", "P", "G"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv.CAP_PROP_FPS, self.config.fps)

        if self.debug:
            print(f"video open source={source!r}")
            print("video requested width:", self.config.width)
            print("video requested height:", self.config.height)
            print("video requested fps:", self.config.fps)
            print("video actual width:", cap.get(cv.CAP_PROP_FRAME_WIDTH))
            print("video actual height:", cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            print("video actual fps:", cap.get(cv.CAP_PROP_FPS))
        return cap

    def _ensure_capture(self) -> cv.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            if self._capture is not None:
                self._capture.release()
            self._capture = self._open_capture()
        return self._capture

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(1, self.config.fps)
        quality = [int(cv.IMWRITE_JPEG_QUALITY), self.config.quality]
        last_log = time.time()
        count = 0
        read_total = resize_total = encode_total = loop_total = 0.0

        while self._running:
            loop_start = time.time()
            cap = self._ensure_capture()

            read_start = time.time()
            success, frame = cap.read()
            read_end = time.time()
            if not success:
                self._camera_ok = False
                if self.debug:
                    print("video capture failed, reconnecting...")
                cap.release()
                self._capture = None
                time.sleep(0.5)
                continue

            self._camera_ok = True
            resize_start = time.time()
            if self.config.width > 0 and self.config.height > 0:
                frame = cv.resize(frame, (self.config.width, self.config.height))
            resize_end = time.time()

            encode_start = time.time()
            ok, encoded = cv.imencode(".jpg", frame, quality)
            encode_end = time.time()
            if ok:
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()
                    self._latest_frame = frame
                    self._latest_time = time.time()

            elapsed = time.time() - loop_start
            count += 1
            read_total += read_end - read_start
            resize_total += resize_end - resize_start
            encode_total += encode_end - encode_start
            loop_total += elapsed

            now = time.time()
            if self.debug and now - last_log >= self.config.log_interval and count > 0:
                actual_fps = count / (now - last_log)
                print(
                    "video timing "
                    f"read={(read_total / count) * 1000:.1f}ms "
                    f"resize={(resize_total / count) * 1000:.1f}ms "
                    f"encode={(encode_total / count) * 1000:.1f}ms "
                    f"loop={(loop_total / count) * 1000:.1f}ms "
                    f"actual_fps={actual_fps:.1f} target_fps={self.config.fps} quality={self.config.quality}"
                )
                last_log = now
                count = 0
                read_total = resize_total = encode_total = loop_total = 0.0

            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def mjpeg_stream(self):
        frame_interval = 1.0 / max(1, self.config.fps)
        while self._running:
            with self._lock:
                frame = self._latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(frame_interval)


class RosmasterHardware:
    
    CMD_STOP = 0
    CMD_FORWARD = 1
    CMD_BACKWARD = 2
    CMD_LEFT = 3
    CMD_RIGHT = 4
    CMD_SPIN_LEFT = 5
    CMD_SPIN_RIGHT = 6

    WATCHDOG_INTERVAL = 0.5

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._lock = threading.RLock()
        self._queue: queue.Queue[Tuple[str, tuple]] = queue.Queue(maxsize=50)
        self._running = True
        self._moving = False
        self._speed = int(os.environ.get("ROSMASTER_DEFAULT_SPEED", "50"))
        self._light_on = False
        self._last_cmd_time = time.time()
        self.watchdog_timeout = float(os.environ.get("ROSMASTER_WATCHDOG_TIMEOUT", "3.0"))
        self.port = os.environ.get("ROSMASTER_PORT", "/dev/myserial")
        self.car_type = int(os.environ.get("ROSMASTER_CAR_TYPE", "1"))
        self.car_type_name = {1: "X3", 2: "X3PLUS", 4: "X1", 5: "R2"}.get(self.car_type, "UNKNOWN")

        self.bot = self._create_bot()
        self.bot.set_car_type(self.car_type)
        self.bot.set_auto_report_state(True)
        self.bot.create_receive_threading()
        if self.debug:
            print(f"hardware car_type={self.car_type} ({self.car_type_name})")

        self._worker = threading.Thread(target=self._control_loop, name="control_worker", daemon=True)
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="watchdog", daemon=True)
        self._worker.start()
        self._watchdog.start()

    def _detect_port(self) -> str:
        if self.port:
            return self.port
        ports = list(list_ports.comports())
        preferred = []
        for port in ports:
            text = f"{port.device} {port.description} {port.hwid}".lower()
            if any(token in text for token in ("usb", "uart", "ch340", "cp210", "serial", "ttyacm", "ttyusb", "com")):
                preferred.append(port.device)
        if preferred:
            if self.debug:
                print("serial candidates:", preferred)
            return preferred[0]
        if ports:
            return ports[0].device
        raise RuntimeError("未检测到 Rosmaster 串口设备，请设置 ROSMASTER_PORT，例如 /dev/ttyUSB0")

    def _create_bot(self):
        first_port = self._detect_port()
        candidates = [first_port] + [p.device for p in list_ports.comports() if p.device != first_port]
        last_error = None
        for port in candidates:
            try:
                if self.debug:
                    print(f"opening Rosmaster serial {port}")
                return Rosmaster(car_type=self.car_type, com=port, debug=self.debug)
            except serial.SerialException as exc:
                last_error = exc
                if self.port:
                    break
        raise RuntimeError(f"无法打开 Rosmaster 串口: {last_error}")

    def _enqueue(self, name: str, *args) -> None:
        try:
            self._queue.put_nowait((name, args))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((name, args))

    def _control_loop(self) -> None:
        while self._running:
            try:
                name, args = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if name == "move":
                    self._move_now(*args)
                elif name == "stop":
                    self._stop_now()
                elif name == "light":
                    self.bot.set_light(1 if args[0] else 0)
                elif name == "beep":
                    self.bot.set_beep(args[0])
            except Exception as exc:
                if self.debug:
                    print(f"control {name} failed: {exc}")

    def _watchdog_loop(self) -> None:
        while self._running:
            time.sleep(self.WATCHDOG_INTERVAL)
            with self._lock:
                should_stop = self._moving and (time.time() - self._last_cmd_time > self.watchdog_timeout)
            if should_stop:
                if self.debug:
                    print("watchdog auto stop")
                self.stop()

    def _touch(self) -> None:
        self._last_cmd_time = time.time()

    def move(self, cmd: int, speed: Optional[int] = None) -> None:
        cmd = int(cmd)
        if cmd not in (self.CMD_STOP, self.CMD_FORWARD, self.CMD_BACKWARD, self.CMD_LEFT, self.CMD_RIGHT, self.CMD_SPIN_LEFT, self.CMD_SPIN_RIGHT):
            raise ValueError(f"未知控制指令: {cmd}")
        speed = self._speed if speed is None else max(0, min(100, int(speed)))
        with self._lock:
            self._speed = speed
            self._moving = cmd != self.CMD_STOP
            self._touch()
        self._enqueue("move", cmd, speed)
        if self.debug:
            print(f"move queued cmd={cmd} speed={speed}")

    def _move_now(self, cmd: int, speed: int) -> None:
        if cmd == self.CMD_STOP:
            self._stop_now()
        elif cmd in (self.CMD_SPIN_LEFT, self.CMD_SPIN_RIGHT):
            self.bot.set_car_run(cmd, speed)
        else:
            self.bot.set_car_run(cmd, speed, 0)
        if self.debug:
            print(f"move sent cmd={cmd} speed={speed}")

    def move_joystick(self, x: int, y: int) -> None:
        x = max(-100, min(100, int(x)))
        y = max(-100, min(100, int(y)))
        if y > 0:
            cmd = self.CMD_FORWARD
        elif y < 0:
            cmd = self.CMD_BACKWARD
        elif x > 0:
            cmd = self.CMD_RIGHT
        elif x < 0:
            cmd = self.CMD_LEFT
        else:
            cmd = self.CMD_STOP
        self.move(cmd, max(abs(x), abs(y), self._speed))

    def stop(self) -> None:
        with self._lock:
            self._moving = False
            self._touch()
        self._enqueue("stop")

    def _stop_now(self) -> None:
        self.bot.set_car_run(0, 0)
        if self.debug:
            print("stop sent")

    def set_light(self, on: bool) -> None:
        with self._lock:
            self._light_on = bool(on)
            self._touch()
        self._enqueue("light", bool(on))

    def beep(self, duration_ms: int = 50) -> None:
        duration_ms = max(0, min(2000, int(duration_ms)))
        self._enqueue("beep", duration_ms)

    def get_sensors(self) -> dict:
        return {
            "temperature": round(self.bot.get_temperature_data(), 1),
            "humidity": round(self.bot.get_humidity_data(), 1),
            "battery": round(self.bot.get_battery_voltage(), 2),
        }

    def get_status(self) -> dict:
        with self._lock:
            return {
                "version": self.bot.get_version(),
                "moving": self._moving,
                "speed": self._speed,
                "light_on": self._light_on,
                "car_type": self.car_type,
                "car_type_name": self.car_type_name,
            }

    def shutdown(self) -> None:
        self._running = False
        try:
            self.bot.set_car_run(0, 0)
            self.bot.set_beep(0)
            self.bot.set_light(0)
        except Exception as exc:
            if self.debug:
                print(f"shutdown cleanup failed: {exc}")
