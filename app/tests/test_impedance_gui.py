import os
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from config_viewer import ViewerConfig
from tests.test_main_window_connection import FakeSettings, FakeWifiBoard
from ui.impedance_dialog import ImpedanceDialog
from ui.main_window import MainWindow


class DialogBoard:
    n_channels = 2

    def measure_impedances(
        self, channels, duration_sec, settle_sec, progress_callback, cancel_event
    ):
        self.channels = channels
        for channel in channels:
            result = {
                "channel": channel,
                "impedance_ohms": 123_400.0,
                "std_uv": 532.1,
                "sample_count": 4000,
            }
            progress_callback(channel, "configuring", None)
            progress_callback(channel, "measuring", None)
            progress_callback(channel, "complete", result)
        return {
            "results": [],
            "cancelled": False,
            "restore": {"state": "verified"},
        }

    def get_last_readback(self):
        return {"state": "verified", "channels": {}}


class ImpedanceGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_runs_in_worker_and_displays_result(self):
        board = DialogBoard()
        dialog = ImpedanceDialog(board)
        dialog._checks[1].setChecked(False)

        dialog._start_test()
        deadline = time.monotonic() + 2.0
        while dialog._thread is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.001)

        self.assertIsNone(dialog._thread)
        self.assertEqual(board.channels, [1])
        self.assertEqual(dialog._table.item(0, 2).text(), "123.4")
        self.assertEqual(dialog._table.item(0, 5).text(), "高阻，建议处理")
        self.assertIn("官方 GUI", dialog._table.item(0, 6).text())
        self.assertIn("恢复并验证", dialog._restore_label.text())
        dialog.close()

    def test_long_restore_diagnostic_is_kept_in_tooltip(self):
        dialog = ImpedanceDialog(DialogBoard())
        detail = "; ".join(f"CH{channel} 恢复失败" for channel in range(1, 9))

        dialog._on_finished({
            "results": [],
            "cancelled": False,
            "restore": {"state": "failed", "diagnostic": detail},
        })

        self.assertEqual(dialog._restore_label.toolTip(), detail)
        self.assertNotIn(detail, dialog._restore_label.text())
        self.assertIn("正式采集已锁定", dialog._restore_label.text())
        dialog.close()

    @patch("ui.main_window.QSettings", FakeSettings)
    def test_main_window_only_enables_test_for_idle_real_cyton(self):
        board = FakeWifiBoard()
        window = MainWindow(board_manager=board, config=ViewerConfig())
        self.assertFalse(window._btn_impedance.isEnabled())

        board.is_connected = True
        window._set_sync_busy(False)
        self.assertTrue(window._btn_impedance.isEnabled())

        board.is_streaming = True
        window._set_sync_busy(False)
        self.assertFalse(window._btn_impedance.isEnabled())
        window.close()

    @patch("ui.main_window.QSettings", FakeSettings)
    @patch("ui.main_window.ImpedanceDialog")
    def test_handler_does_not_touch_delsys_or_recorder(self, dialog_class):
        board = FakeWifiBoard()
        board.is_connected = True
        board.impedance_restore_verified = True
        dialog = MagicMock()
        dialog.last_report = {
            "restore": {"state": "verified"}, "cancelled": False,
        }
        dialog_class.return_value = dialog
        recorder = MagicMock()
        recorder.is_recording = False
        window = MainWindow(
            board_manager=board, recorder=recorder, config=ViewerConfig()
        )
        nano = MagicMock()
        window._delsys.client = nano

        window._on_impedance_test()

        dialog.exec.assert_called_once()
        self.assertEqual(nano.method_calls, [])
        recorder.start_recording.assert_not_called()
        recorder.stop_recording.assert_not_called()
        window.close()


if __name__ == "__main__":
    unittest.main()
