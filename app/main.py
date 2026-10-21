"""EMG Viewer — YMS (based on BrainFlow) —— OpenBCI Cyton 定制上位机入口。

用法:
    python main.py                  # 启动 GUI (WiFi Shield, 默认 192.168.4.1:6677)
    python main.py --serial PORT    # 用串口 Dongle 连接
    python main.py --wifi IP PORT   # WiFi Shield 自定义 IP/端口
    python main.py --synthetic      # 合成板模拟 (无需硬件，用于调试)
    python main.py --cli            # 命令行测试模式

依赖:
    pip install brainflow pyqt6 pyqtgraph
"""
from __future__ import annotations
import sys
from pathlib import Path

# 添加 emg_python 到 Python 路径
_EMG_PYTHON = Path(__file__).resolve().parents[1] / "emg_python"
if _EMG_PYTHON.exists():
    sys.path.insert(0, str(_EMG_PYTHON))

from config_viewer import ViewerConfig


def run_package_self_test():
    """Exercise frozen native dependencies without requiring attached hardware."""
    import json
    import tempfile
    import time
    import numpy as np
    import cv2
    import serial
    from brainflow.board_shim import BoardIds
    from board_manager import CytonManager
    from PyQt6.QtCore import QT_VERSION_STR

    report = {
        "python": sys.version,
        "opencv": cv2.__version__,
        "pyserial": serial.VERSION,
        "qt": QT_VERSION_STR,
        "checks": {},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "opencv_writer_test.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 writer could not be opened")
        for index in range(10):
            writer.write(np.full((240, 320, 3), index * 20, dtype=np.uint8))
        writer.release()
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None or video_path.stat().st_size <= 0:
            raise RuntimeError("OpenCV could not read its packaged MP4 output")
        report["checks"]["opencv_mp4_roundtrip"] = True

    board = CytonManager(board_id=BoardIds.SYNTHETIC_BOARD.value)
    try:
        if not board.connect() or not board.start_stream():
            raise RuntimeError("BrainFlow synthetic board failed to start")
        time.sleep(0.25)
        data = board.get_new_data(n_samples=25)
        if data is None or data.size == 0:
            raise RuntimeError("BrainFlow synthetic board returned no samples")
        report["checks"]["brainflow_synthetic"] = True
        report["synthetic_shape"] = list(data.shape)
    finally:
        board.stop()

    # Exercise the packaged UDP receiver with actual loopback datagrams. HTTP
    # start/stop is stubbed here so this self-test never touches physical boards.
    import socket
    from wifi_udp import WiFiUDPBoard
    udp = WiFiUDPBoard("127.0.0.1", "127.0.0.1", 0)
    udp._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp._socket.bind(("127.0.0.1", 0))
    udp._socket.settimeout(0.2)
    udp._request = lambda *args, **kwargs: "{}"
    try:
        udp.start_stream()
        raw = b"\xa0\x01" + (100).to_bytes(3, "big", signed=True) * 8 + bytes(6) + b"\xc0"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(raw, udp._socket.getsockname())
        deadline = time.monotonic() + 2.0
        while udp.receiver_status["frames"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        data = udp.get_board_data()
        expected = 100 * 4.5e6 / ((2**23 - 1) * 24)
        if data.shape != (24, 1) or not np.isclose(data[1, 0], expected):
            raise RuntimeError("Packaged UDP receive/scaling test failed")
        report["checks"]["udp_loopback_and_microvolts"] = True
    finally:
        udp.release_session()

    output_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    output = output_dir / "package_self_test.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def run_cli_test(config: ViewerConfig):
    """命令行测试模式: 采集 5 秒，打印统计。"""
    from board_manager import CytonManager
    import time, numpy as np

    print("=" * 60)
    print("  EMG Viewer — CLI 测试模式")
    print("=" * 60)

    board_type = {5: "WiFi Shield", 0: "串口 Dongle", -1: "SYNTHETIC (模拟)"}
    mode = board_type.get(config.board_id, f"Board {config.board_id}")
    if config.board_id == 5:
        mode += f" {config.wifi_ip}:{config.wifi_port}"
    elif config.board_id == 0:
        mode += f" {config.serial_port}"
    print(f"模式: {mode}")

    manager = CytonManager(
        serial_port=config.serial_port,
        board_id=config.board_id,
        wifi_ip=config.wifi_ip,
        wifi_port=config.wifi_port,
    )

    if not manager.connect():
        print("\n❌ 连接失败")
        print("   pip install brainflow")
        return 1

    manager.start_stream()
    print(f"采样率: {manager.sample_rate} Hz, 通道数: {manager.n_channels}")
    print("采集 5 秒...")
    time.sleep(5.0)

    data = manager.get_new_data(n_samples=250)
    counter = manager.get_counter_data(n_samples=250)
    manager.stop()

    if data is not None and data.size > 0:
        print(f"\n数据 shape: {data.shape}")
        for ch in range(data.shape[0]):
            ch_data = data[ch]
            print(f"  Ch{ch+1}: "
                  f"min={ch_data.min():+8.1f}  max={ch_data.max():+8.1f}  "
                  f"mean={ch_data.mean():+8.1f}  std={ch_data.std():.1f} μV")
        print("\n✅ 数据流验证通过")
        return 0
    else:
        print("\n❌ 未收到数据")
        return 1


def run_gui(config: ViewerConfig):
    """启动 GUI 模式。"""
    from PyQt6.QtWidgets import QApplication
    from board_manager import CytonManager
    from signal_chain import SignalChain
    from metrics import PacketLossTracker
    from recorder import Recorder
    from ui.main_window import MainWindow

    if config.board_id == 5:
        config.sampling_rate = CytonManager.WIFI_SAMPLE_RATE
    elif config.board_id == -1:
        config.record_subject = config.record_subject or "Synthetic"
        config.record_task = config.record_task or "synthetic"

    app = QApplication(sys.argv)
    app.setApplicationName("EMG Viewer Sync Impedance UDP")

    # 创建各模块
    board = CytonManager(
        serial_port=config.serial_port,
        board_id=config.board_id,
        wifi_ip=config.wifi_ip,
        wifi_port=config.wifi_port,
    )

    chain = SignalChain(
        fs=config.sampling_rate,
        n_channels=config.n_channels,
        bandpass=(config.bandpass_low, config.bandpass_high),
        notch_hz=config.notch_hz,
        notch_bw=config.notch_bandwidth_hz,
        notch_harmonics=config.notch_harmonics,
        envelope_ms=config.envelope_window_ms,
    )

    tracker = PacketLossTracker(
        sample_rate=config.sampling_rate,
        windows_sec=config.packet_loss_windows_sec,
    )

    rec = Recorder(
        data_dir=config.record_dir,
        flush_sec=config.record_flush_sec,
        n_channels=config.n_channels,
    )

    window = MainWindow(
        board_manager=board,
        signal_chain=chain,
        metrics_engine=tracker,
        recorder=rec,
        config=config,
    )

    if config.board_id == -1:
        # 合成板: 自动连接 + 开始采集
        window._on_connect()
        if window._connect_thread is not None:
            window._connect_thread.finished.connect(window._on_start)

    window.show()
    return app.exec()


def main():
    if "--package-self-test" in sys.argv[1:]:
        return run_package_self_test()
    config = ViewerConfig()
    # 默认 WiFi Shield (board_id=5 已在 ViewerConfig 中设置)

    cli_mode = False

    for arg in sys.argv[1:]:
        if arg == "--cli":
            cli_mode = True
        elif arg == "--serial":
            config.board_id = 0
            config.serial_port = sys.argv[sys.argv.index(arg) + 1] if sys.argv.index(arg) + 1 < len(sys.argv) else ""
        elif arg == "--wifi":
            config.board_id = 5
            # 可选: --wifi IP PORT
            idx = sys.argv.index(arg)
            config.wifi_ip = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--") else "192.168.4.1"
            config.wifi_port = int(sys.argv[idx + 2]) if idx + 2 < len(sys.argv) and not sys.argv[idx + 2].startswith("--") else 6677
            config.wifi_ip_explicit = True
        elif arg == "--synthetic":
            config.board_id = -1
        elif arg == "--gui":
            pass  # 使用当前配置

    if cli_mode:
        return run_cli_test(config)
    return run_gui(config)


if __name__ == "__main__":
    sys.exit(main())
