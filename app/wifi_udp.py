"""Cyton WiFi raw UDP receiver with the data access methods used by the GUI."""
from __future__ import annotations

from collections import deque
import json
import re
import socket
import threading
import time
import urllib.request

import numpy as np


class WiFiUDPBoard:
    SAMPLE_RATE = 1000
    GAIN_VALUES = (1, 2, 4, 6, 8, 12, 24)

    def __init__(self, host: str, local_ip: str, port: int):
        self.host = host
        self.local_ip = local_ip
        self.port = port
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._socket = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples = deque(maxlen=60 * self.SAMPLE_RATE)
        self._gains = [24] * 8
        self._configured = False
        self._streaming = False
        self._last_timestamp = 0.0
        self._error = ""
        self._stats = {}
        self._reset_stats()

    def _reset_stats(self):
        self._stats = {
            "datagrams": 0, "frames": 0, "malformed_bytes": 0,
            "foreign_datagrams": 0, "buffer_overruns": 0,
        }

    def _request(self, path: str, method: str = "GET", body=None) -> str:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"http://{self.host}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(request, timeout=3.0) as response:
            return response.read().decode("utf-8", errors="replace").strip()

    def prepare_session(self):
        board = json.loads(self._request("/board"))
        if board.get("board_type") != "cyton" or not board.get("board_connected"):
            raise RuntimeError("UDP 版本需要已连接的 8 通道 Cyton")
        tcp = json.loads(self._request("/tcp"))
        if tcp.get("connected"):
            raise RuntimeError("Shield 已有 TCP 客户端，请先断开其他上位机")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self._socket.bind((self.local_ip, self.port))
            self.port = self._socket.getsockname()[1]
            self._socket.settimeout(0.2)
            # Mark ownership before POST so a timeout can still clean up our setup.
            self._configured = True
            self._request("/udp", "POST", {
                "ip": self.local_ip, "port": self.port, "output": "raw",
                "delimiter": False, "latency": 10000, "redundancy": False,
            })
            # Cyton replies "Success: Sample rate is 1000Hz" to ~4.
            # ~? is not a supported query and returns "invalid sample value".
            response = self.config_board("~4")
            match = re.search(r"(\d+)\s*Hz", response, re.IGNORECASE)
            if not match or int(match.group(1)) != self.SAMPLE_RATE:
                raise RuntimeError(f"无法确认板端 1000 Hz 采样率: {response}")
        except Exception:
            self.release_session()
            raise

    def observe_command(self, command: str):
        """Keep conversion gains aligned with successful channel commands."""
        for match in re.finditer(r"x([1-8])[01]([0-6])[0-7][01][01][01]X", command):
            self._gains[int(match.group(1)) - 1] = self.GAIN_VALUES[int(match.group(2))]

    def config_board(self, command: str) -> str:
        response = self._request("/command", "POST", {"command": command})
        if re.search(r"failure|timeout|error", response, re.IGNORECASE):
            raise RuntimeError(response)
        self.observe_command(command)
        return response

    def start_stream(self):
        if self._streaming:
            raise RuntimeError("UDP 采集已启动")
        if self._socket is None:
            raise RuntimeError("UDP 接收端尚未配置")
        # Stop has joined the old reader; drain late datagrams before a new trial.
        self._socket.setblocking(False)
        try:
            while True:
                self._socket.recvfrom(65535)
        except BlockingIOError:
            pass
        finally:
            self._socket.settimeout(0.2)
        with self._lock:
            self._samples.clear()
            self._reset_stats()
        self._last_timestamp = 0.0
        self._error = ""
        self._stop.clear()
        self._streaming = True
        self._thread = threading.Thread(target=self._receive, name="CytonUDP", daemon=True)
        self._thread.start()
        try:
            self._request("/stream/start")
        except Exception:
            try:
                self.stop_stream()
            except Exception:
                pass
            raise

    def _decode_datagram(self, payload: bytes, received: float) -> list[np.ndarray]:
        frames = []
        offset = 0
        while offset + 33 <= len(payload):
            if payload[offset] != 0xA0 or not 0xC0 <= payload[offset + 32] <= 0xC6:
                offset += 1
                self._stats["malformed_bytes"] += 1
                continue
            raw = payload[offset:offset + 33]
            row = np.zeros(24, dtype=np.float64)
            row[0] = raw[1]
            for channel in range(8):
                start = 2 + channel * 3
                value = int.from_bytes(raw[start:start + 3], "big", signed=True)
                row[channel + 1] = value * 4.5e6 / ((2**23 - 1) * self._gains[channel])
            row[12] = raw[32]  # packet type and unmodified auxiliary bytes
            row[13:19] = list(raw[26:32])
            if raw[32] == 0xC0:
                for axis in range(3):
                    start = 26 + axis * 2
                    row[9 + axis] = int.from_bytes(raw[start:start + 2], "big", signed=True) * 0.002 / 16
            frames.append(row)
            offset += 33
        # Datagram boundaries are independent: never join a truncated packet to
        # the next datagram, which might follow lost or reordered UDP traffic.
        self._stats["malformed_bytes"] += len(payload) - offset
        for index, row in enumerate(frames):
            estimated = received - (len(frames) - 1 - index) / self.SAMPLE_RATE
            row[22] = max(estimated, self._last_timestamp + 0.000001)
            self._last_timestamp = row[22]
        return frames

    def _receive(self):
        while not self._stop.is_set():
            try:
                payload, peer = self._socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._error = str(exc)
                break
            with self._lock:
                if peer[0] != self.host:
                    self._stats["foreign_datagrams"] += 1
                    continue
                self._stats["datagrams"] += 1
                frames = self._decode_datagram(payload, time.time())
                self._stats["frames"] += len(frames)
                self._stats["buffer_overruns"] += max(
                    0, len(self._samples) + len(frames) - self._samples.maxlen
                )
                self._samples.extend(frames)

    def get_board_data(self):
        with self._lock:
            samples = list(self._samples)
            self._samples.clear()
        if self._error:
            raise RuntimeError(f"UDP 接收失败: {self._error}")
        return np.stack(samples, axis=1) if samples else np.empty((24, 0))

    def get_current_board_data(self, n_samples: int):
        if n_samples <= 0:
            return np.empty((24, 0))
        with self._lock:
            samples = list(self._samples)[-n_samples:]
        if self._error:
            raise RuntimeError(f"UDP 接收失败: {self._error}")
        return np.stack(samples, axis=1) if samples else np.empty((24, 0))

    def stop_stream(self):
        try:
            if self._streaming:
                self._request("/stream/stop")
        finally:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=4.0)
                self._thread = None
            self._streaming = False

    def release_session(self):
        try:
            self.stop_stream()
        finally:
            try:
                if self._configured:
                    self._request("/udp", "DELETE")
            finally:
                self._configured = False
                if self._socket:
                    self._socket.close()
                    self._socket = None

    @property
    def receiver_status(self) -> dict:
        with self._lock:
            return {**self._stats, "error": self._error}
