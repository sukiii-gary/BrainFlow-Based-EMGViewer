"""Cyton 板通信管理 —— 基于 BrainFlow 库。

BrainFlow 是 OpenBCI 官方唯一推荐的 Python 库。pyOpenBCI / openbci-python 均已弃用。

Cyton 8ch 数据布局 (BrainFlow 返回的 2D array):
  row[0]  = 包计数器 (0-255 循环)
  row[1-8] = EMG/EEG 通道 (已是微伏, BrainFlow 自动转换)
  row[9-11] = 加速度计 X/Y/Z
  row[22] = 时间戳 (UNIX)
  row[23] = 标记
"""
from __future__ import annotations
import ipaddress
import json
import threading
import time
import re
import socket
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
import numpy as np

from impedance import estimate_impedance_ohms
from wifi_udp import WiFiUDPBoard

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    BoardShim = None  # type: ignore


class CytonManager:
    """OpenBCI Cyton 板管理器。

    使用方式:
        manager = CytonManager(serial_port="/dev/cu.usbserial-xxx")
        if manager.connect():
            manager.start_stream()
            while running:
                data = manager.get_new_data(n_samples=250)  # 取最新 250 个样本
                # data shape: (8, n_samples), 单位 μV
            manager.stop()
    """

    WIFI_SAMPLE_RATE = 1000.0
    IMPEDANCE_TEST_SETTINGS = {
        "gain": 0, "bias_on": False, "srb2_on": True,
    }

    def __init__(self, serial_port: str = "", board_id: int = 5,
                 wifi_ip: str = "192.168.4.1", wifi_port: int = 6677):
        """
        Args:
            serial_port: 串口路径。macOS: /dev/cu.usbserial-*, Windows: COM3
            board_id: BrainFlow board_id.
                      5 = Cyton WiFi Shield (默认)
                      0 = Cyton (Serial Dongle)
                      -1 = Synthetic (模拟正弦波，无需硬件)
            wifi_ip: WiFi Shield IP 地址
            wifi_port: 电脑本机接收 Shield 数据的 UDP 端口
        """
        self.serial_port = serial_port
        self.board_id = board_id
        self.wifi_ip = wifi_ip
        self.wifi_port = wifi_port
        self._board = None
        self._eeg_channels = None
        self._timestamp_channel = None
        self._sample_rate = None
        self._is_streaming = False
        self._is_connected = False
        self._received_samples = 0
        self._counter_advanced = False
        self._last_counter_value: int | None = None
        self._last_timestamps = np.array([], dtype=np.float64)
        self._last_connection: dict = {
            "stage": "idle", "detail": "", "local_ip": "",
        }

        # 硬件设置 (仅存内存，采集前统一下发)
        self.bias_on = True       # BIAS (DRL 主动地)
        self.srb2_on = True       # SRB2 (参考电极)
        self.gain = 6             # PGA 增益: 0=1x 1=2x 2=4x 3=6x 4=8x 5=12x 6=24x
        self.channel_settings = {
            ch: {"gain": 6, "bias_on": True, "srb2_on": True}
            for ch in range(1, 9)
        }
        self._last_readback: dict = {
            "state": "pending", "channels": {}, "raw": ""
        }
        self._impedance_restore_verified = True
        self._impedance_stream_stop_failed = False

    # ========== 连接管理 ==========

    def configure_wifi_address(self, wifi_ip: str) -> None:
        """在断开状态设置要连接的 WiFi Shield 静态 IPv4 地址。"""
        if self._is_connected or self._board:
            raise RuntimeError("请先断开板卡再修改 Shield IP")
        address = ipaddress.IPv4Address(wifi_ip.strip())
        if address.is_unspecified or address.is_multicast or address.is_loopback or address.is_reserved:
            raise ValueError("请输入 Shield 可使用的单播 IPv4 地址")
        self.wifi_ip = str(address)

    def _get_local_ip_for_wifi(self) -> str:
        """返回系统访问 Shield 时选择的本机 IPv4 地址。"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((self.wifi_ip, 80))
            return str(sock.getsockname()[0])

    def _probe_wifi_board(self) -> str:
        """通过官方 `/board` 接口确认 Shield 的 HTTP 控制链路可达。"""
        request = urllib.request.Request(
            f"http://{self.wifi_ip}/board",
            headers={"Accept": "application/json, text/plain"},
            method="GET",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3.0) as response:
            return response.read().decode("utf-8", errors="replace").strip()

    def connect(self) -> bool:
        """打开串口/WiFi 连接，准备采集。返回成功与否。"""
        self._last_connection = {"stage": "starting", "detail": "", "local_ip": ""}
        if not BRAINFLOW_AVAILABLE:
            detail = "BrainFlow 未安装，请运行: pip install brainflow"
            self._last_connection = {"stage": "dependency", "detail": detail, "local_ip": ""}
            print(f"[CytonManager] {detail}")
            return False

        params = BrainFlowInputParams()

        if self.board_id == BoardIds.CYTON_BOARD.value:  # 0, Serial Dongle
            if not self.serial_port:
                print("[CytonManager] 请设置 serial_port (如 /dev/cu.usbserial-xxx)")
                return False
            params.serial_port = self.serial_port
        elif self.board_id == BoardIds.CYTON_WIFI_BOARD.value:  # 5, WiFi Shield
            try:
                self.configure_wifi_address(self.wifi_ip)
                local_ip = self._get_local_ip_for_wifi()
                self._last_connection = {
                    "stage": "http_probe", "detail": "", "local_ip": local_ip,
                }
                self._probe_wifi_board()
            except Exception as e:
                detail = f"Shield HTTP 不可达 ({self.wifi_ip}/board): {e}"
                self._last_connection = {
                    "stage": "http_probe", "detail": detail,
                    "local_ip": self._last_connection.get("local_ip", ""),
                }
                print(f"[CytonManager] 连接失败: {detail}")
                return False
            params.ip_address = self.wifi_ip
            params.ip_port = self.wifi_port
        # board_id = -1 (SYNTHETIC) 无需额外参数

        try:
            self._board = (
                WiFiUDPBoard(self.wifi_ip, local_ip, self.wifi_port)
                if self.board_id == 5 else BoardShim(self.board_id, params)
            )
            self._board.prepare_session()
            if self.board_id == 5:
                self.wifi_port = self._board.port
            self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            if self.board_id == BoardIds.SYNTHETIC_BOARD.value:
                self._eeg_channels = self._eeg_channels[:8]
            self._timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            self._sample_rate = BoardShim.get_sampling_rate(self.board_id)
            if (
                self.board_id == BoardIds.CYTON_WIFI_BOARD.value
                and float(self._sample_rate) != self.WIFI_SAMPLE_RATE
            ):
                raise RuntimeError(
                    f"Cyton WiFi 实际采样率为 {self._sample_rate}Hz，"
                    f"期望 {self.WIFI_SAMPLE_RATE:.0f}Hz"
                )
            self._is_connected = True
            self._impedance_restore_verified = True
            self._last_connection.update({
                "stage": "connected",
                "detail": (
                    "HTTP 控制已连接，UDP 接收端已配置；开始采集后验证数据"
                    if self.board_id == 5 else "板卡已连接"
                ),
            })

            if self.board_id == BoardIds.SYNTHETIC_BOARD.value:  # -1
                desc = "SYNTHETIC (模拟)"
            elif self.board_id == BoardIds.CYTON_WIFI_BOARD.value:  # 5
                desc = (
                    f"WiFi Shield {self.wifi_ip}, "
                    f"本机接收端口 {self.wifi_port}"
                )
            else:  # 0, Serial
                desc = f"串口 {self.serial_port}"
            print(f"[CytonManager] 已连接: {desc}, {len(self._eeg_channels)} 通道, "
                  f"{self._sample_rate}Hz")
            return True
        except Exception as e:
            detail = (
                f"UDP 连接配置失败: {e}"
                if self.board_id == 5 else f"连接失败: {e}"
            )
            self._last_connection.update({"stage": "brainflow", "detail": detail})
            print(f"[CytonManager] {detail}")
            if self._board:
                try:
                    self._board.release_session()
                except Exception:
                    pass
            self._board = None
            self._is_connected = False
            self._is_streaming = False
            return False

    def disconnect(self):
        """安全断开连接。"""
        if self._is_streaming:
            self.stop_stream()
        if self._board:
            try:
                self._board.release_session()
            except Exception as e:
                print(f"[CytonManager] 断开时出错: {e}")
            self._board = None
        self._is_connected = False
        self._impedance_restore_verified = True
        print("[CytonManager] 已断开")

    # ========== 硬件设置 (Gain / BIAS / SRB2) ==========

    def set_hw_settings(self, bias_on: bool = True, srb2_on: bool = True, gain: int = 6):
        """兼容旧调用：把相同设置应用到全部 Cyton 通道。

        新界面使用 set_channel_settings() 保存逐通道设置。
        """
        self.bias_on = bias_on
        self.srb2_on = srb2_on
        self.gain = gain
        n_ch = len(self._eeg_channels) if self._eeg_channels else 8
        self.channel_settings = {
            ch: {"gain": gain, "bias_on": bias_on, "srb2_on": srb2_on}
            for ch in range(1, n_ch + 1)
        }
        print(f"[CytonManager] 硬件设置已记录(待下发): "
              f"Gain={self.gain}({self._gain_to_x(self.gain)}x), "
              f"BIAS={'ON' if bias_on else 'OFF'}, "
              f"SRB2={'ON' if srb2_on else 'OFF'}")

    def set_channel_settings(self, settings: list[dict]):
        """保存逐通道硬件设置，不立即下发。"""
        normalized = {}
        for item in settings:
            channel = int(item["channel"])
            if not 1 <= channel <= 8:
                raise ValueError(f"通道号超出范围: {channel}")
            gain = int(item.get("gain", 6))
            if gain not in range(7):
                raise ValueError(f"通道{channel}增益编码无效: {gain}")
            normalized[channel] = {
                "gain": gain,
                "bias_on": bool(item.get("bias_on", True)),
                "srb2_on": bool(item.get("srb2_on", True)),
            }
        self.channel_settings = normalized
        self._last_readback = {"state": "pending", "channels": {}, "raw": ""}
        print(f"[CytonManager] 已记录 {len(normalized)} 个通道设置(待下发)")

    @staticmethod
    def _gain_to_x(gain: int) -> int:
        """gain 编码 → 实际放大倍数。"""
        return {0: 1, 1: 2, 2: 4, 3: 6, 4: 8, 5: 12, 6: 24}.get(gain, 24)

    @staticmethod
    def build_channel_command(channel: int, settings: dict) -> str:
        """生成 Cyton x(CH)(PWR)(GAIN)(INPUT)(BIAS)(SRB2)(SRB1)X 命令。"""
        gain = int(settings["gain"])
        bias = '1' if settings["bias_on"] else '0'
        srb2 = '1' if settings["srb2_on"] else '0'
        return f"x{channel}0{gain}0{bias}{srb2}0X"

    @staticmethod
    def _response_has_error(response: str) -> bool:
        return bool(re.search(r'failure|timeout|error', response or "", re.IGNORECASE))

    def _send_channel_config(self) -> dict:
        """下发配置并保留每个通道的命令与板端响应。

        config_board() 没抛异常只代表传输层接受了调用；响应为空时不能
        宣称板端已经执行，后续必须依赖寄存器回读。
        """
        report = {"channels": {}, "sent": True}
        if not self._board or self.board_id < 0:
            return report

        for ch, settings in sorted(self.channel_settings.items()):
            cmd = self.build_channel_command(ch, settings)
            item = {"command": cmd, "response": "", "sent": False}
            try:
                response = (
                    self._send_wifi_command_http(cmd)
                    if self.board_id == 5
                    else self._board.config_board(cmd)
                ) or ""
                item["response"] = response
                item["sent"] = not self._response_has_error(response)
                time.sleep(0.05)  # 通道间留小延迟
            except Exception as e:
                item["response"] = f"发送异常: {e}"
                print(f"[CytonManager] 通道{ch}配置失败: {e}")
            report["channels"][ch] = item
            report["sent"] = report["sent"] and item["sent"]
            if not item["sent"]:
                break

        ok_count = sum(item["sent"] for item in report["channels"].values())
        if report["sent"]:
            print(f"[CytonManager] 已发送通道配置: {ok_count}/{len(report['channels'])} 通道")
        else:
            print(f"[CytonManager] 通道配置发送失败: {ok_count}/{len(report['channels'])} 通道")
        return report

    @classmethod
    def parse_register_report(cls, response: str) -> dict:
        """解析 `?` 返回的 ADS1299 寄存器报告。

        串口固件用 ``$$$`` 结束；WiFi 固件把 EOT 转成 HTTP 响应结束，
        文本本身以 LIS3DH 最后一个寄存器 ``0x3D`` 结束。
        """
        raw = response or ""
        header = re.search(r'Board ADS Registers', raw, re.IGNORECASE)
        terminator_pos = raw.find("$$$", header.start() if header else 0)
        lis3dh_header = re.search(r'^\s*LIS3DH Registers\s*$', raw, re.IGNORECASE | re.MULTILINE)
        lis3dh_tail = re.search(
            r'^\s*0x3D\s+[0-9A-Fa-f]{2}[\s\x00]*\Z',
            raw,
            re.IGNORECASE | re.MULTILINE,
        )
        wifi_structure_complete = bool(
            header and lis3dh_header and lis3dh_tail
            and lis3dh_header.start() > header.start()
        )
        completion = (
            "dollar_eot" if terminator_pos >= 0
            else "lis3dh_tail" if wifi_structure_complete
            else "incomplete"
        )
        parse_text = raw
        if header:
            end = terminator_pos + 3 if terminator_pos >= 0 else len(raw)
            parse_text = raw[header.start():end]
        result = {
            "channels": {}, "bias_sensp": None, "bias_sensn": None,
            "complete": bool(header and completion != "incomplete"),
            "completion": completion,
            "raw": raw,
        }

        # 固件格式示例: CH1SET, 05, 68, 0, 1, 1, 0, 1, 0, 0, 0
        pattern = re.compile(
            r'^\s*CH(\d+)SET\s*,\s*[0-9A-Fa-f]{2}\s*,\s*([0-9A-Fa-f]{2})\b',
            re.IGNORECASE | re.MULTILINE,
        )
        for match in pattern.finditer(parse_text):
            channel = int(match.group(1))
            value = int(match.group(2), 16)
            gain = (value >> 4) & 0x07       # CHnSET[6:4]
            result["channels"][channel] = {
                "register": f"0x{value:02X}",
                "powered_on": not bool(value & 0x80),
                "gain": gain,
                "gain_x": cls._gain_to_x(gain),
                "srb2": bool(value & 0x08),  # CHnSET[3]
                "input_type": value & 0x07,
            }

        def register_value(name: str):
            match = re.search(
                rf'^\s*{name}\s*,\s*[0-9A-Fa-f]{{2}}\s*,\s*([0-9A-Fa-f]{{2}})\b',
                parse_text, re.IGNORECASE | re.MULTILINE,
            )
            return int(match.group(1), 16) if match else None

        result["bias_sensp"] = register_value("BIAS_SENSP")
        result["bias_sensn"] = register_value("BIAS_SENSN")
        for channel, values in result["channels"].items():
            mask = 1 << (channel - 1)
            bias_p = None if result["bias_sensp"] is None else bool(result["bias_sensp"] & mask)
            bias_n = None if result["bias_sensn"] is None else bool(result["bias_sensn"] & mask)
            values["bias_p"] = bias_p
            values["bias_n"] = bias_n
            values["bias"] = bias_p if bias_p is not None and bias_p == bias_n else None
        return result

    @staticmethod
    def _decode_wifi_http_response(payload: bytes) -> str:
        """把 WiFi Shield /command 的纯文本或 JSON 响应转成寄存器文本。"""
        text = payload.decode("utf-8", errors="replace").strip()
        if not text:
            return ""

        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

        strings = []

        def collect(item):
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)
            elif isinstance(item, list):
                for child in item:
                    collect(child)

        collect(value)
        return "\n".join(strings).strip()

    def _send_wifi_command_http(self, command: str) -> str:
        """Send one command through the Shield HTTP API and return its body."""
        url = f"http://{self.wifi_ip}/command"
        body = json.dumps({"command": command}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/plain, application/json"},
            method="POST",
        )
        # Shield 是局域网设备，不能让系统 HTTP 代理接管此请求。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=3.0) as response:
                payload = response.read()
        except urllib.error.HTTPError as e:
            detail = self._decode_wifi_http_response(e.read())
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"HTTP {e.code}{suffix}") from e
        result = self._decode_wifi_http_response(payload)
        if isinstance(self._board, WiFiUDPBoard) and not self._response_has_error(result):
            self._board.observe_command(command)
        return result

    def _read_wifi_registers_http(self) -> str:
        """直接读取 Shield /command 响应，绕过 BrainFlow 丢失的响应体。"""
        return self._send_wifi_command_http("?")

    def read_back_settings(self) -> dict:
        """读回板端实际生效的 Gain/BIAS/SRB2。

        发送 `?` 查询，解析 ADS1299 寄存器：
          CHnSET bit[6:4]=Gain, bit[3]=SRB2；BIAS_SENSP/N bit0=通道1 bias。
        仅在采集前调用，避免干扰数据流。

        Returns:
            包含逐通道寄存器、BIAS、完整性和原始响应的字典。
        """
        result = {
            "channels": {}, "complete": False, "raw": "",
            "source": "none", "diagnostic": "",
        }
        if not self._board or self.board_id < 0:
            result["raw"] = "(合成板无此功能)"
            return result

        if self.board_id == 5:
            try:
                resp = self._read_wifi_registers_http()
            except Exception as e:
                detail = f"WiFi HTTP /command 回读失败: {e}"
                result["raw"] = detail
                result["diagnostic"] = detail
                result["source"] = "wifi_http"
                print(f"[CytonManager] {detail}")
                return result

            if resp:
                print(f"[CytonManager] WiFi HTTP 读回原始响应 ({len(resp)} 字符):")
                print(resp[:500])
                print(f"[CytonManager] 读回尾部: {resp[-160:]!r}")
                parsed = self.parse_register_report(resp)
                parsed["source"] = "wifi_http"
                parsed["diagnostic"] = (
                    "" if parsed["complete"]
                    else "WiFi HTTP 已响应，但报告既没有 $$$，也没有完整的 LIS3DH 0x3D 结尾"
                )
                print(
                    f"[CytonManager] 解析结果: complete={parsed['complete']}, "
                    f"completion={parsed['completion']}, "
                    f"channels={sorted(parsed['channels'])}"
                )
                return parsed

            detail = "WiFi HTTP /command 返回空响应"
            result["raw"] = detail
            result["diagnostic"] = detail
            result["source"] = "wifi_http"
            print(f"[CytonManager] {detail}")
            return result

        try:
            resp = self._board.config_board('?')
        except Exception as e:
            detail = f"BrainFlow 读回失败: {e}"
            result["raw"] = detail
            result["diagnostic"] = detail
            result["source"] = "brainflow"
            print(f"[CytonManager] {detail}")
            return result

        if not resp:
            detail = "BrainFlow 读回为空"
            result["raw"] = detail
            result["diagnostic"] = detail
            result["source"] = "brainflow"
            print(f"[CytonManager] {detail}")
            return result

        print(f"[CytonManager] BrainFlow 读回原始响应 ({len(resp)} 字符):")
        print(resp[:500])
        print(f"[CytonManager] 读回尾部: {resp[-160:]!r}")
        parsed = self.parse_register_report(resp)
        parsed["source"] = "brainflow"
        parsed["diagnostic"] = (
            "" if parsed["complete"]
            else "BrainFlow 返回的报告既没有 $$$，也没有完整的 LIS3DH 0x3D 结尾"
        )
        print(
            f"[CytonManager] 解析结果: complete={parsed['complete']}, "
            f"completion={parsed['completion']}, "
            f"channels={sorted(parsed['channels'])}"
        )
        return parsed

    def apply_and_verify_settings(self) -> dict:
        """发送逐通道配置，随后用 ADS1299 寄存器回读验证。"""
        if self.board_id < 0:
            self._last_readback = {
                "state": "not_applicable", "channels": {},
                "raw": "(合成板无此功能)", "transport": "synthetic",
            }
            return self._last_readback

        send_report = self._send_channel_config()
        if not send_report["sent"]:
            state = "failed"
            failed_items = [
                f"Ch{channel}: {item['response']}"
                for channel, item in send_report["channels"].items()
                if not item["sent"]
            ]
            detail = "通道配置发送失败" + (
                f" ({'; '.join(failed_items)})" if failed_items else ""
            )
            readback = {
                "channels": {}, "complete": False, "raw": detail,
                "source": "wifi_http" if self.board_id == 5 else "brainflow",
                "diagnostic": detail,
            }
        else:
            time.sleep(0.1)
            readback = self.read_back_settings()
            state = "sent_unverified"
        if send_report["sent"] and readback.get("complete"):
            mismatches = []
            for channel, target in sorted(self.channel_settings.items()):
                actual = readback.get("channels", {}).get(channel)
                if not actual:
                    mismatches.append(channel)
                    continue
                actual["matches"] = (
                    actual["gain"] == target["gain"]
                    and actual["srb2"] == target["srb2_on"]
                    and actual["bias"] == target["bias_on"]
                )
                if not actual["matches"]:
                    mismatches.append(channel)
            state = "verified" if not mismatches else "failed"

        self._last_readback = {
            **readback,
            "state": state,
            "transport": "wifi" if self.board_id == 5 else "serial",
            "sent": send_report,
            "target": self.get_channel_settings(),
        }
        if state == "verified":
            self._impedance_restore_verified = True
        print(f"[CytonManager] 硬件设置状态: {state}")
        return self._last_readback

    def get_last_readback(self) -> dict:
        """返回最近一次读回结果。"""
        return self._last_readback

    def get_channel_settings(self) -> list[dict]:
        return [
            {"channel": channel, **settings}
            for channel, settings in sorted(self.channel_settings.items())
        ]

    def get_hardware_metadata(self) -> dict:
        """返回可写入采集文件头或导出报告的硬件审计信息。"""
        try:
            brainflow_version = version("brainflow")
        except PackageNotFoundError:
            brainflow_version = "not-installed"
        return {
            "board_id": self.board_id,
            "transport": "wifi" if self.board_id == 5 else "serial" if self.board_id >= 0 else "synthetic",
            "brainflow_version": brainflow_version,
            "wifi_ip": self.wifi_ip if self.board_id == 5 else "",
            "local_data_port": self.wifi_port if self.board_id == 5 else None,
            "data_protocol": "udp" if self.board_id == 5 else None,
            "timestamp_source": "host_receive_estimate" if self.board_id == 5 else "brainflow",
            "udp_receiver": self._board.receiver_status if isinstance(self._board, WiFiUDPBoard) else None,
            "sampling_rate": self.sample_rate,
            "connection": self.last_connection_status,
            "channel_settings": self.get_channel_settings(),
            "verification": self._last_readback,
        }

    # ========== 电极阻抗测试 ==========

    @classmethod
    def build_impedance_command(cls, channel: int, active: bool = True) -> str:
        """Build the official Cyton N-input lead-off command sequence."""
        if not 1 <= int(channel) <= 8:
            raise ValueError(f"通道号超出范围: {channel}")
        lead_off = f"z{int(channel)}{'01' if active else '00'}Z"
        if not active:
            return lead_off
        return cls.build_channel_command(
            int(channel), cls.IMPEDANCE_TEST_SETTINGS
        ) + lead_off

    def _send_tracked_command(self, command: str) -> str:
        """Send through BrainFlow so its per-channel gain scaler stays current."""
        response = self._board.config_board(command) or ""
        if self._response_has_error(response):
            raise RuntimeError(response)
        return response

    def _collect_impedance_samples(
        self,
        channel: int,
        duration_sec: float,
        settle_sec: float,
        cancel_event: threading.Event,
    ) -> np.ndarray:
        """Run the dedicated test stream and return post-settling channel data."""
        try:
            self._board.get_board_data()  # discard stale samples
        except Exception:
            pass

        self._board.start_stream()
        self._is_streaming = True
        started = time.monotonic()
        settle_at = started + settle_sec
        finish_at = started + duration_sec
        chunks = []
        try:
            while time.monotonic() < finish_at:
                if cancel_event.is_set():
                    break
                time.sleep(min(0.05, max(0.0, finish_at - time.monotonic())))
                data = self._board.get_board_data()
                if (
                    time.monotonic() >= settle_at
                    and getattr(data, "size", 0)
                ):
                    row = self._eeg_channels[channel - 1]
                    chunks.append(np.asarray(data[row, :], dtype=np.float64))
        finally:
            try:
                self._board.stop_stream()
            except Exception:
                self._impedance_stream_stop_failed = True
                raise
            finally:
                self._is_streaming = False

        if cancel_event.is_set():
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)

    def _restore_after_impedance(self, original_settings: list[dict]) -> dict:
        errors = []
        if self._impedance_stream_stop_failed:
            errors.append("阻抗测试数据流未能确认停止")
        for channel in range(1, self.n_channels + 1):
            try:
                self._send_tracked_command(
                    self.build_impedance_command(channel, active=False)
                )
            except Exception as exc:
                errors.append(f"CH{channel} 关闭激励失败: {exc}")
        for settings in original_settings:
            channel = int(settings["channel"])
            command = self.build_channel_command(channel, settings)
            try:
                self._send_tracked_command(command)
                time.sleep(0.05)
            except Exception as exc:
                errors.append(f"CH{channel}: {exc}")

        readback = self.read_back_settings()
        mismatches = []
        if readback.get("complete"):
            for settings in original_settings:
                channel = int(settings["channel"])
                actual = readback.get("channels", {}).get(channel)
                matches = bool(
                    actual
                    and actual["gain"] == int(settings["gain"])
                    and actual["bias"] == bool(settings["bias_on"])
                    and actual["srb2"] == bool(settings["srb2_on"])
                )
                if actual is not None:
                    actual["matches"] = matches
                if not matches:
                    mismatches.append(channel)

        if errors or mismatches:
            state = "failed"
        elif readback.get("complete"):
            state = "verified"
        else:
            state = "sent_unverified"
        diagnostic = "; ".join(errors)
        if mismatches:
            diagnostic = "; ".join(filter(None, [
                diagnostic,
                "恢复设置不匹配: " + ", ".join(f"CH{ch}" for ch in mismatches),
            ]))
        if not diagnostic:
            diagnostic = readback.get("diagnostic", "")
        self._last_readback = {
            **readback,
            "state": state,
            "diagnostic": diagnostic,
            "transport": "wifi" if self.board_id == 5 else "serial",
            "target": [dict(item) for item in original_settings],
        }
        self._impedance_restore_verified = state == "verified"
        self._impedance_stream_stop_failed = False
        return self._last_readback

    def measure_impedances(
        self,
        channels,
        duration_sec: float = 5.0,
        settle_sec: float = 1.0,
        progress_callback=None,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        """Measure selected N inputs serially and always restore acquisition settings."""
        if not self._board or not self._is_connected:
            raise RuntimeError("请先连接 Cyton 板卡")
        if self.board_id not in (0, 5):
            raise RuntimeError("阻抗测试仅支持 Cyton 串口或 Cyton WiFi")
        if self._is_streaming:
            raise RuntimeError("请先停止正式采集")
        if duration_sec <= settle_sec or settle_sec < 0:
            raise ValueError("测试时长必须大于稳定等待时间")

        selected = []
        for value in channels:
            channel = int(value)
            if not 1 <= channel <= self.n_channels:
                raise ValueError(f"通道号超出范围: {channel}")
            if channel not in selected:
                selected.append(channel)
        if not selected:
            raise ValueError("请至少选择一个通道")

        cancel = cancel_event or threading.Event()
        original_settings = self.get_channel_settings()
        results = []
        restore_report = None
        self._impedance_restore_verified = False
        self._impedance_stream_stop_failed = False
        try:
            for channel in range(1, self.n_channels + 1):
                self._send_tracked_command(
                    self.build_impedance_command(channel, active=False)
                )
            for channel in selected:
                if cancel.is_set():
                    break
                if progress_callback:
                    progress_callback(channel, "configuring", None)
                self._send_tracked_command(self.build_impedance_command(channel))
                time.sleep(0.1)
                if progress_callback:
                    progress_callback(channel, "measuring", None)
                samples = self._collect_impedance_samples(
                    channel, duration_sec, settle_sec, cancel
                )
                if cancel.is_set():
                    break
                minimum_samples = max(
                    16,
                    int((duration_sec - settle_sec) * self.sample_rate * 0.25),
                )
                if samples.size < minimum_samples:
                    raise RuntimeError(
                        f"CH{channel} 有效样本不足: {samples.size}/{minimum_samples}"
                    )
                result = {"channel": channel, **estimate_impedance_ohms(samples)}
                results.append(result)
                if progress_callback:
                    progress_callback(channel, "complete", dict(result))
                original = next(
                    item for item in original_settings
                    if int(item["channel"]) == channel
                )
                self._send_tracked_command(
                    self.build_channel_command(channel, original)
                    + self.build_impedance_command(channel, active=False)
                )
        finally:
            if self._is_streaming:
                try:
                    self._board.stop_stream()
                finally:
                    self._is_streaming = False
            restore_report = self._restore_after_impedance(original_settings)

        return {
            "results": results,
            "cancelled": cancel.is_set(),
            "restore": restore_report,
        }

    # ========== 数据流控制 ==========

    def start_stream(self, *, settings_prepared: bool = False) -> bool:
        """开始数据采集。采集前统一下发硬件配置并读回校验。"""
        if not self._board:
            print("[CytonManager] 请先 connect()")
            return False
        if not self._impedance_restore_verified:
            print("[CytonManager] 阻抗测试后的硬件恢复尚未通过寄存器验证")
            return False
        try:
            # 1. 下发硬件配置并读回校验（采集前，不干扰数据流）
            if not settings_prepared:
                report = self.apply_and_verify_settings()
                if report.get("state") == "failed":
                    print("[CytonManager] 硬件设置验证失败，拒绝开始采集")
                    return False
            # 3. 开始采集
            self._board.start_stream()
            self._is_streaming = True
            self._received_samples = 0
            self._counter_advanced = False
            self._last_counter_value = None
            print(f"[CytonManager] 开始采集 @ {self._sample_rate}Hz")
            return True
        except Exception as e:
            print(f"[CytonManager] 启动采集失败: {e}")
            return False

    def stop_stream(self):
        """停止数据采集。"""
        if self._board and self._is_streaming:
            try:
                self._board.stop_stream()
            except Exception as e:
                print(f"[CytonManager] 停止采集时出错: {e}")
        self._is_streaming = False
        print("[CytonManager] 采集已停止")

    # ========== 数据获取 ==========

    def get_new_data(self, n_samples: int | None = None, *, allow_stopped: bool = False) -> np.ndarray | None:
        """取最新的 n_samples 个样本，同时缓存包计数器供丢包检测。

        Args:
            n_samples: 要取的样本数。None = 取全部缓冲中的新数据。

        Returns:
            np.ndarray shape (n_channels, n) in μV，失败返回 None
        """
        if not self._board or (not self._is_streaming and not allow_stopped):
            return None

        try:
            if n_samples is None:
                data = self._board.get_board_data()  # 取全部并清空缓冲
            else:
                data = self._board.get_current_board_data(n_samples)

            if data.size == 0:
                self._last_counter = np.array([], dtype=np.int32)
                self._last_timestamps = np.array([], dtype=np.float64)
                return np.zeros((len(self._eeg_channels), 0))

            # 缓存包计数器 (row[0]) 供丢包检测
            self._last_counter = np.asarray(data[0, :], dtype=np.int32)
            if (
                self._timestamp_channel is not None
                and 0 <= self._timestamp_channel < data.shape[0]
            ):
                self._last_timestamps = np.asarray(
                    data[self._timestamp_channel, :], dtype=np.float64
                )
            else:
                self._last_timestamps = np.array([], dtype=np.float64)
            if self._last_counter.size:
                first_counter = int(self._last_counter[0])
                if (
                    self._last_counter.size > 1
                    and np.any(self._last_counter[1:] != self._last_counter[:-1])
                ) or (
                    self._last_counter_value is not None
                    and first_counter != self._last_counter_value
                ):
                    self._counter_advanced = True
                self._last_counter_value = int(self._last_counter[-1])
                self._received_samples += int(self._last_counter.size)
            # 只取 EMG 通道 (rows 1-8)
            eeg_data = data[self._eeg_channels, :]
            return np.asarray(eeg_data, dtype=np.float32)

        except Exception as e:
            print(f"[CytonManager] 取数据时出错: {e}")
            return None

    def get_counter_data(self, n_samples: int = 250) -> np.ndarray | None:
        """取包计数器行 (row[0])，用于丢包检测。"""
        if not self._board or not self._is_streaming:
            return None

        try:
            data = self._board.get_current_board_data(n_samples)
            if data.size == 0:
                return np.array([])
            return np.asarray(data[0, :], dtype=np.int32)
        except Exception as e:
            print(f"[CytonManager] 取包计数器时出错: {e}")
            return None

    # ========== 属性 ==========

    @property
    def sample_rate(self) -> float:
        if self._sample_rate:
            return self._sample_rate
        return self.WIFI_SAMPLE_RATE if self.board_id == 5 else 250.0

    @property
    def last_connection_status(self) -> dict:
        return dict(self._last_connection)

    @property
    def data_stream_verified(self) -> bool:
        return self._received_samples > 0 and self._counter_advanced

    @property
    def n_channels(self) -> int:
        return len(self._eeg_channels) if self._eeg_channels else 8

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def impedance_restore_verified(self) -> bool:
        return self._impedance_restore_verified

    # ========== 便捷方法 ==========

    def stop(self):
        """一键停止: 停采集 + 断连接。"""
        self.stop_stream()
        self.disconnect()

    @staticmethod
    def list_ports() -> list[str]:
        """扫描系统可用串口。需要 pyserial。"""
        try:
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            print("[CytonManager] pyserial 未安装，无法扫描串口")
            return []

    @staticmethod
    def get_sample_rate_for(board_id: int) -> float:
        """查询指定 board_id 的采样率(不需要连接硬件)。"""
        if not BRAINFLOW_AVAILABLE:
            return 250.0
        return BoardShim.get_sampling_rate(board_id)


# ========== 快速测试 ==========

if __name__ == "__main__":
    """用合成板测试数据流。不需要硬件。"""
    print("=== CytonManager 合成板测试 ===\n")

    manager = CytonManager(board_id=-1)  # SYNTHETIC: 生成模拟正弦波

    if not manager.connect():
        print("连接失败，请确认 brainflow 已安装: pip install brainflow")
        exit(1)

    manager.start_stream()
    print("采集 2 秒...")
    time.sleep(2.0)

    # 取最新 250 个样本 (1秒)
    data = manager.get_new_data(n_samples=250)
    manager.stop()

    if data is not None and data.size > 0:
        print(f"\n数据 shape: {data.shape}")
        print(f"通道数: {data.shape[0]}, 样本数: {data.shape[1]}")
        print(f"Ch0 范围: [{data[0].min():.1f}, {data[0].max():.1f}] μV")
        print(f"Ch0 均值: {data[0].mean():.1f} μV")
        print(f"Ch0 标准差: {data[0].std():.1f} μV")
        print("\n✅ 数据流验证通过！")
    else:
        print("\n❌ 未收到数据")
