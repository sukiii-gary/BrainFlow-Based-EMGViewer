"""EMG Viewer 配置 —— dataclass 风格，与现有 emg_python/config.py 保持一致。

所有可调参数集中于此，方便 AI 修改。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys


_PROJECT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)


@dataclass
class ViewerConfig:
    """实时采集上位机的全部配置。"""

    # ====== 硬件连接 ======
    serial_port: str = ""               # 串口路径，如 "/dev/cu.usbserial-*" 或 "COM3"
                                         # macOS 注意: 必须用 /dev/cu.* 而非 /dev/tty.*
    board_id: int = 5                    # BrainFlow board_id: 5=CYTON_WIFI(默认), 0=CYTON_SERIAL, -1=SYNTHETIC
    wifi_ip: str = "192.168.4.1"        # WiFi Shield 默认 IP
    wifi_port: int = 6677               # 电脑本机接收 Shield 数据的 UDP 端口
    wifi_ip_explicit: bool = False       # True 时命令行地址优先于 GUI 记忆值

    # ====== 采集参数 ======
    sampling_rate: float = 1000.0        # Cyton WiFi 固定 1000Hz
    n_channels: int = 8                  # Cyton MCU 的 8 个 EMG 通道
    data_ring_buffer_sec: float = 60.0   # 内存中保留的原始数据长度 (秒)
    poll_interval_ms: int = 40           # 从 BrainFlow 拉数据的间隔 (ms), 25Hz 拉取

    # ====== 显示参数 ======
    display_window_sec: float = 15.0     # 时域图滚动窗口 (秒)
    ui_refresh_ms: int = 100             # 波形刷新间隔；8通道运行时最低200ms(5Hz)
    channel_colors: tuple = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    )

    # ====== EMG 滤波参数 (1000Hz 采样 → Nyquist=500Hz) ======
    bandpass_low: float = 20.0           # 高通截止 (Hz), 去掉运动伪迹
    bandpass_high: float = 450.0         # 低通截止 (Hz), 抗混叠 (需采样率≥1000Hz)
    bandpass_order: int = 5              # Butterworth 阶数
    notch_hz: float = 50.0               # 工频 (中国 50Hz, 美国 60Hz)
    notch_bandwidth_hz: float = 8.0      # 每谐波保持的绝对带宽 (Q = f/bw)
    notch_harmonics: int = 3             # 生成 f0*(1..nh+1), 超 Nyquist 的自动裁掉
    envelope_window_ms: float = 100.0    # RMS 包络滑窗 (ms)

    # ====== 丢包统计 ======
    packet_loss_windows_sec: tuple = (1, 60, 600)  # 三个时间窗口

    # ====== 数据记录 ======
    record_group: str = ""               # 组别，如 "对照组"、"实验组"
    record_subject: str = ""             # 被试 ID，如 "S01"
    record_task: str = ""                # 任务名称，如 "握拳"、"步态"
    record_flush_sec: float = 5.0        # 缓冲刷盘间隔
    record_dir: str = field(
        default_factory=lambda: str(_PROJECT_DIR / "recordings")
    )                                    # 固定为上位机项目目录下的 recordings

    # ====== FFT 显示 ======
    fft_window_sec: float = 2.0          # 用于 PSD 计算的数据长度
    fft_refresh_hz: int = 10             # 频谱刷新率 (Hz)


# 便捷单例
default_config = ViewerConfig()
