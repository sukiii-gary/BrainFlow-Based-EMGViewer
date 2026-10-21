"""Offscreen GUI coverage for the optional camera synchronization path."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from config_viewer import ViewerConfig
from recorder import Recorder
from tests.test_gui_delsys_sync import FakeNano, SyncBoard, SyncChain
from tests.test_session_trial_workflow import FakeSettings
from ui.main_window import MainWindow


class FakeCamera:
    def __init__(self, ready=True):
        self.ready = ready
        self.active = False
        self.stops = 0
        self.latest_frame = None

    def status(self):
        return {"connected": self.ready, "ready": self.ready, "recording": self.active}

    def prepare_recording(self, trial_dir, metadata=None):
        self.trial_dir = Path(trial_dir)
        return {"video_path": self.trial_dir / "camera.mp4"}

    def start_recording(self):
        self.active = True
        return {"running": True, "monotonic_ns": time.perf_counter_ns()}

    def stop_recording(self):
        self.active = False
        self.stops += 1
        return {"frame_count": 3}

    def disconnect(self):
        self.ready = False


class GuiCameraSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = patch("ui.main_window.QSettings", FakeSettings)
        self.ports = patch("serial.tools.list_ports.comports", return_value=[])
        self.settings.start()
        self.ports.start()
        self.startup_drain = patch(
            "delsys_controller._drain_openbci_startup",
            return_value={
                "sample_count": 1,
                "first_batch_sample_count": 1,
                "poll_count": 1,
                "wait_duration_ms": 1.0,
                "completed_stable_windows": 2,
            },
        )
        self.startup_drain.start()
        self.board = SyncBoard()
        self.camera = FakeCamera()
        self.recorder = Recorder(self.temp.name, n_channels=2)
        self.window = MainWindow(
            board_manager=self.board,
            signal_chain=SyncChain(),
            recorder=self.recorder,
            config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            camera_controller=self.camera,
        )
        self.window._sidebar.set_recording_metadata(subject="CAM", task="triple")
        self.window._delsys.client = FakeNano()
        self.window._sync_enabled.setChecked(True)
        self.window._camera_enabled.setChecked(True)

    def tearDown(self):
        if self.window._delsys.active:
            self.window._on_stop()
            self.wait_job()
        self.window.close()
        self.window._delsys.finish("test_cleanup")
        self.startup_drain.stop()
        self.ports.stop()
        self.settings.stop()
        self.temp.cleanup()

    def wait_job(self):
        deadline = time.monotonic() + 4
        while self.window._sync_thread is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.001)
        self.assertIsNone(self.window._sync_thread)
        for timer in (self.window._poll_timer, self.window._display_timer,
                      self.window._metrics_timer, self.window._data_verify_timer):
            timer.stop()

    def test_enabled_camera_must_be_connected(self):
        self.camera.ready = False
        self.window._on_start()
        self.assertFalse(self.board.is_streaming)
        self.assertIn("连接摄像机", self.window._camera_label.text())

    def test_three_device_start_and_stop_uses_one_trigger_each(self):
        self.window._on_start()
        self.wait_job()
        self.assertTrue(self.camera.active)
        self.assertTrue(self.board.is_streaming)
        self.assertEqual(self.window._delsys.client.commands, ["START"])

        self.window._on_stop()
        self.wait_job()
        self.assertFalse(self.camera.active)
        self.assertEqual(self.camera.stops, 1)
        self.assertEqual(self.window._delsys.client.commands, ["START", "STOP"])

    def test_camera_gate_closes_before_blocking_board_stop(self):
        self.window._on_start()
        self.wait_job()
        camera_states_at_board_stop = []
        original_stop = self.board.stop_stream

        def observe_stop():
            camera_states_at_board_stop.append(self.camera.active)
            return original_stop()

        self.board.stop_stream = observe_stop
        self.window._on_stop()
        self.wait_job()

        self.assertEqual(camera_states_at_board_stop, [False])
        self.assertEqual(self.camera.stops, 1)


if __name__ == "__main__":
    unittest.main()
