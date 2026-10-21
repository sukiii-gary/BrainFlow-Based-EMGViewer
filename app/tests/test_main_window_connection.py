import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from config_viewer import ViewerConfig
from ui.main_window import MainWindow


def wait_for_connection(window, timeout=2.0):
    deadline = time.monotonic() + timeout
    while window._connect_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    if window._connect_thread is not None:
        raise AssertionError("connection worker did not finish")


def wait_for_hardware_apply(window, timeout=2.0):
    deadline = time.monotonic() + timeout
    while window._hw_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    if window._hw_thread is not None:
        raise AssertionError("hardware apply worker did not finish")


class FakeSettings:
    values = {}

    def __init__(self, *_args):
        pass

    def value(self, key, default="", type=None):
        value = self.values.get(key, default)
        return type(value) if type is not None else value

    def setValue(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)

    def sync(self):
        pass


class FakeWifiBoard:
    board_id = 5
    wifi_port = 6677
    sample_rate = 1000.0
    n_channels = 8
    is_streaming = False
    data_stream_verified = False

    def __init__(self, connect_result=True):
        self.wifi_ip = "192.168.4.1"
        self.is_connected = False
        self.connect_result = connect_result
        self.connect_calls = 0
        self.last_connection_status = {
            "stage": "idle", "detail": "", "local_ip": "192.168.10.5",
        }

    def configure_wifi_address(self, wifi_ip):
        self.wifi_ip = wifi_ip

    def set_channel_settings(self, _settings):
        pass

    def connect(self):
        self.connect_calls += 1
        self.is_connected = self.connect_result
        if not self.connect_result:
            self.last_connection_status = {
                "stage": "http_probe",
                "detail": "Shield HTTP 不可达",
                "local_ip": "",
            }
        return self.connect_result

    def stop_stream(self):
        self.is_streaming = False

    def disconnect(self):
        self.is_connected = False


class FakeRecorder:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.session_dir = None
        self.current_trial_info = None
        self.is_recording = False

    def list_trials(self):
        return []

    def stop_recording(self):
        self.is_recording = False
        return None

    def end_session(self):
        pass


class MainWindowConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings = FakeSettings()
        self.settings.remove("connection/wifi_ip")
        self.settings_patch = patch("ui.main_window.QSettings", FakeSettings)
        self.settings_patch.start()

    def tearDown(self):
        self.settings_patch.stop()
        self.settings.remove("connection/wifi_ip")

    def test_rejects_invalid_static_ip_before_connecting(self):
        board = FakeWifiBoard()
        window = MainWindow(board_manager=board, config=ViewerConfig())
        window._input_wifi_ip.setText("openbci.local")

        window._on_connect()

        self.assertEqual(board.connect_calls, 0)
        self.assertIn("Shield IP 无效", window.statusBar().currentMessage())
        window.close()

    def test_applies_and_remembers_static_ip_after_successful_connect(self):
        board = FakeWifiBoard()
        window = MainWindow(board_manager=board, config=ViewerConfig())
        window._input_wifi_ip.setText("192.168.10.42")

        window._on_connect()
        wait_for_connection(window)

        self.assertEqual(board.wifi_ip, "192.168.10.42")
        self.assertEqual(board.connect_calls, 1)
        self.assertFalse(window._input_wifi_ip.isEnabled())
        self.settings.sync()
        self.assertEqual(
            self.settings.value("connection/wifi_ip", "", type=str),
            "192.168.10.42",
        )
        window.close()

    def test_shows_connection_stage_failure_detail(self):
        board = FakeWifiBoard(connect_result=False)
        window = MainWindow(board_manager=board, config=ViewerConfig())

        window._on_connect()
        wait_for_connection(window)

        self.assertEqual(window.statusBar().currentMessage(), "Shield HTTP 不可达")
        self.assertTrue(window._input_wifi_ip.isEnabled())
        window.close()

    def test_connection_exception_is_reported_without_escaping_ui_handler(self):
        board = FakeWifiBoard()

        def raise_connect_error():
            raise RuntimeError("unexpected transport failure")

        board.connect = raise_connect_error
        window = MainWindow(board_manager=board, config=ViewerConfig())

        window._on_connect()
        wait_for_connection(window)

        self.assertTrue(window._input_wifi_ip.isEnabled())
        self.assertIn("连接异常", window.statusBar().currentMessage())
        window.close()

    def test_slow_connection_does_not_block_gui_thread(self):
        board = FakeWifiBoard()
        entered = threading.Event()
        release = threading.Event()

        def slow_connect():
            entered.set()
            release.wait(1.0)
            board.is_connected = True
            return True

        board.connect = slow_connect
        window = MainWindow(board_manager=board, config=ViewerConfig())

        started = time.monotonic()
        window._on_connect()

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(entered.wait(0.5))
        self.assertIsNotNone(window._connect_thread)
        self.assertFalse(window._btn_connect.isEnabled())
        release.set()
        wait_for_connection(window)
        self.assertTrue(board.is_connected)
        window.close()

    def test_slow_hardware_verification_does_not_block_gui_thread(self):
        board = FakeWifiBoard()
        board.is_connected = True
        entered = threading.Event()
        release = threading.Event()

        def slow_apply():
            entered.set()
            release.wait(1.0)
            return {"state": "verified", "channels": {}}

        board.apply_and_verify_settings = slow_apply
        window = MainWindow(board_manager=board, config=ViewerConfig())

        started = time.monotonic()
        window._on_apply_hw_settings()

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(entered.wait(0.5))
        self.assertIsNotNone(window._hw_thread)
        self.assertFalse(window._btn_connect.isEnabled())
        release.set()
        wait_for_hardware_apply(window)
        self.assertIn("寄存器验证通过", window.statusBar().currentMessage())
        window.close()

    def test_explicit_command_line_ip_overrides_remembered_ip(self):
        self.settings.setValue("connection/wifi_ip", "192.168.10.42")
        config = ViewerConfig(wifi_ip="192.168.4.1", wifi_ip_explicit=True)

        window = MainWindow(board_manager=FakeWifiBoard(), config=config)

        self.assertEqual(window._input_wifi_ip.text(), "192.168.4.1")
        window.close()

    def test_history_comparison_receives_only_record_root_and_viewer_config(self):
        config = ViewerConfig()
        recorder = FakeRecorder("D:/test-recordings")
        window = MainWindow(recorder=recorder, config=config)

        with patch("ui.main_window.HistoryComparisonDialog") as dialog_class:
            window._on_compare_history()

        dialog_class.assert_called_once_with(
            recorder.data_dir,
            viewer_config=config,
            parent=window,
        )
        dialog_class.return_value.exec.assert_called_once_with()
        window.close()

    def test_history_comparison_is_disabled_and_rejected_during_acquisition(self):
        board = FakeWifiBoard()
        board.is_streaming = True
        recorder = FakeRecorder("D:/test-recordings")
        window = MainWindow(board_manager=board, recorder=recorder, config=ViewerConfig())
        window._refresh_session_trials()

        self.assertFalse(window._btn_compare_history.isEnabled())
        with patch("ui.main_window.HistoryComparisonDialog") as dialog_class:
            window._on_compare_history()

        dialog_class.assert_not_called()
        self.assertIn("请先停止采集", window.statusBar().currentMessage())
        window.close()

    def test_history_comparison_is_disabled_and_rejected_during_recording(self):
        recorder = FakeRecorder("D:/test-recordings")
        window = MainWindow(recorder=recorder, config=ViewerConfig())
        window._recording = True
        window._refresh_session_trials()

        self.assertFalse(window._btn_compare_history.isEnabled())
        with patch("ui.main_window.HistoryComparisonDialog") as dialog_class:
            window._on_compare_history()

        dialog_class.assert_not_called()
        window.close()

    def test_history_comparison_is_disabled_during_delsys_acquisition(self):
        recorder = FakeRecorder("D:/test-recordings")
        window = MainWindow(recorder=recorder, config=ViewerConfig())
        window._delsys.active = True
        window._refresh_session_trials()

        self.assertFalse(window._btn_compare_history.isEnabled())
        with patch("ui.main_window.HistoryComparisonDialog") as dialog_class:
            window._on_compare_history()

        dialog_class.assert_not_called()
        window.close()


if __name__ == "__main__":
    unittest.main()
