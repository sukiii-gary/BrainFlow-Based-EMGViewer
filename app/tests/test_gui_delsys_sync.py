"""Offscreen workflow tests: simulated devices, real recording files, no serial IO."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from config_viewer import ViewerConfig
from recorder import Recorder
from metrics import PacketLossTracker
from tests.test_session_trial_workflow import FakeBoard, FakeChain, FakeSettings
from ui.main_window import MainWindow


class SyncBoard(FakeBoard):
    def __init__(self):
        super().__init__()
        self.is_connected = True
        self.starts = []
        self.apply_gate = None
        self.apply_entered = threading.Event()
        self.pending = None
        self.drains = []
        self.startup_pending = False

    def apply_and_verify_settings(self):
        self.apply_entered.set()
        if self.apply_gate is not None and not self.apply_gate.wait(3):
            raise RuntimeError("test apply gate timed out")
        return {"state": "not_applicable"}

    def start_stream(self, settings_prepared=False):
        self.starts.append(settings_prepared)
        started = super().start_stream()
        self.startup_pending = bool(started)
        return started

    def get_new_data(self, n_samples=None, allow_stopped=False):
        self.drains.append((self.is_streaming, allow_stopped))
        if self.startup_pending:
            self.startup_pending = False
            self._last_timestamps = np.array([99.999], dtype=np.float64)
            return np.ones((2, 1), dtype=np.float32)
        if self.pending is None:
            return np.zeros((2, 0), dtype=np.float32)
        data, self.pending = self.pending, None
        self._last_timestamps = 100.0 + np.arange(data.shape[1]) / self.sample_rate
        return data


class SyncChain(FakeChain):
    def push_samples(self, data):
        pass


class FakeNano:
    def __init__(self, fail_start=False):
        self.ready = True
        self.commands = []
        self.fail_start = fail_start
        self.closed = False

    def status(self):
        return {"response": "STATUS IDLE 0"}

    def trigger(self, action):
        self.commands.append(action)
        if action == "START" and self.fail_start:
            raise TimeoutError("synthetic ACK loss")
        return {"action": action, "status": "acknowledged", "id": len(self.commands)}

    def close(self):
        self.ready = False
        self.closed = True


class GuiDelsysSyncTests(unittest.TestCase):
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
            side_effect=lambda board, sample_rate: {
                "sample_count": int(board.get_new_data(n_samples=None).shape[-1]),
                "first_batch_sample_count": 1,
                "poll_count": 1,
                "wait_duration_ms": 1.0,
                "completed_stable_windows": 2,
            },
        )
        self.startup_drain.start()
        self.board = SyncBoard()
        self.recorder = Recorder(data_dir=self.temp.name, n_channels=2)
        self.window = MainWindow(
            board_manager=self.board, signal_chain=SyncChain(), recorder=self.recorder,
            config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
        )
        self.window._sidebar.set_recording_metadata(subject="TEST", task="sync")
        self.nano = FakeNano()

    def tearDown(self):
        if self.board.apply_gate is not None:
            self.board.apply_gate.set()
        self.wait_job()
        if self.window._delsys.active:
            self.window._on_stop()
            self.wait_job()
        self.window.close()
        # Close any sidecar even if an assertion exposed incomplete GUI cleanup.
        self.window._delsys.finish("test_cleanup")
        self.ports.stop()
        self.startup_drain.stop()
        self.settings.stop()
        self.temp.cleanup()

    def wait_job(self):
        deadline = time.monotonic() + 4
        while self.window._sync_thread is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.001)
        self.assertIsNone(self.window._sync_thread, "sync worker did not finish")
        for timer in (self.window._poll_timer, self.window._display_timer,
                      self.window._metrics_timer, self.window._data_verify_timer):
            timer.stop()

    def enable(self):
        self.window._delsys.client = self.nano
        self.window._sync_enabled.setChecked(True)

    def start(self):
        self.enable()
        self.window._on_start()
        self.wait_job()
        self.assertTrue(self.window._recording)

    def events(self):
        logs = list(Path(self.temp.name).rglob("sync_events.jsonl"))
        self.assertEqual(len(logs), 1)
        return [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]

    def test_default_disabled_preserves_recording_without_nano(self):
        self.assertFalse(self.window._sync_enabled.isChecked())
        self.window._on_start()
        self.assertTrue(self.window._recording)
        self.assertFalse(self.window._sync_enabled.isEnabled())
        self.recorder.append(np.ones((2, 4)), np.arange(4) / 1000)
        self.window._on_stop()
        trials = self.recorder.list_trials()
        self.assertEqual(len(trials), 1)
        self.assertTrue(Path(trials[0]["txt_path"]).is_file())
        self.assertEqual(self.board.starts, [False])
        self.assertFalse(list(Path(self.temp.name).rglob("sync_events.jsonl")))

    def test_default_mode_poll_path_does_not_create_packet_sidecar(self):
        self.window.metrics = PacketLossTracker(sample_rate=1000)
        self.window._on_start()
        self.board.startup_pending = False
        self.board.pending = np.ones((2, 3), dtype=np.float32)
        self.board._last_counter = np.array([7, 8, 9])

        self.window._on_poll()
        self.window._on_stop()

        trial = self.recorder.list_trials()[0]
        self.assertEqual(trial["samples"], 3)
        self.assertIsNone(trial["packet_path"])
        self.assertFalse((Path(trial["path"]) / "openbci_packets.csv").exists())
        self.assertEqual(self.window.metrics.get_diagnostics()["received"], 3)
        self.assertEqual(self.board.starts, [False])

    def test_enabled_without_nano_refuses_to_start(self):
        self.window._sync_enabled.setChecked(True)
        self.window._on_start()
        self.assertFalse(self.board.is_streaming)
        self.assertFalse(self.recorder.is_recording)
        self.assertIn("Nano", self.window._sync_label.text())
        self.assertEqual(self.board.starts, [])

    def test_controller_failure_is_not_treated_as_running_board(self):
        self.board.is_streaming = True

        self.window._finish_sync_start({
            "result": {"running": False, "warning": "synthetic failure"}
        })

        self.assertFalse(self.board.is_streaming)
        self.assertFalse(self.window._streaming)
        self.assertFalse(self.window._poll_timer.isActive())
        self.assertIn("synthetic failure", self.window._sync_label.text())

    def test_start_stop_once_logs_clicks_and_saves_samples(self):
        self.enable()
        self.window._on_start()
        self.window._on_start()
        self.wait_job()
        self.window._on_start()
        self.assertEqual(self.board.starts, [True])
        self.assertEqual(self.nano.commands, ["START"])
        self.board.pending = np.ones((2, 4), dtype=np.float32)
        self.window._on_poll()
        self.board.pending = np.full((2, 3), 2., dtype=np.float32)
        self.window._on_stop()
        self.window._on_stop()
        self.wait_job()
        self.window._on_stop()
        self.assertEqual(self.nano.commands, ["START", "STOP"])
        self.assertFalse(self.recorder.is_recording)
        trials = self.recorder.list_trials()
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["samples"], 7)
        self.assertIn((False, True), self.board.drains)
        self.assertTrue(Path(trials[0]["txt_path"]).is_file())
        events = self.events()
        names = [event["event"] for event in events]
        for expected in ("start_clicked", "stop_clicked", "start_trigger", "stop_trigger",
                         "first_openbci_data_received", "trial_finished"):
            self.assertEqual(names.count(expected), 1)
        for event in events:
            if event["event"].endswith("clicked"):
                self.assertGreater(event["click"]["unix_ns"], 0)
                self.assertGreater(event["click"]["monotonic_ns"], 0)
        self.assertIsNone(self.window._delsys._log)

    def test_pretrigger_samples_are_trimmed_by_timestamp(self):
        self.start()
        self.window._delsys.sync_start_unix_ns = 100_002_000_000
        data = np.arange(8, dtype=np.float32).reshape(2, 4)
        timestamps = np.array([100.000, 100.001, 100.002, 100.003])

        trimmed, trimmed_timestamps = self.window._post_sync_samples(data, timestamps)

        np.testing.assert_array_equal(trimmed, data[:, 2:])
        np.testing.assert_array_equal(trimmed_timestamps, timestamps[2:])

    def test_final_samples_are_trimmed_at_confirmed_stop_boundary(self):
        self.start()
        self.window._delsys.sync_start_unix_ns = 100_000_000_000
        data = np.arange(8, dtype=np.float32).reshape(2, 4)
        timestamps = np.array([100.000, 100.001, 100.002, 100.003])

        trimmed, trimmed_timestamps = self.window._post_sync_samples(
            data, timestamps, stop_boundary_unix_ns=100_002_000_000
        )

        np.testing.assert_array_equal(trimmed, data[:, :3])
        np.testing.assert_array_equal(trimmed_timestamps, timestamps[:3])

    def test_counter_uses_exact_same_start_boundary_mask_as_recorded_data(self):
        self.start()
        self.window.metrics = PacketLossTracker(sample_rate=1000)
        self.window._delsys.sync_start_unix_ns = 100_002_000_000
        self.board.pending = np.arange(8, dtype=np.float32).reshape(2, 4)
        self.board._last_counter = np.array([10, 11, 12, 13])

        self.window._on_poll()

        diagnostics = self.window.metrics.get_diagnostics()
        self.assertEqual(diagnostics["received"], 2)
        self.assertEqual(diagnostics["first_counter"], 12)
        self.assertEqual(diagnostics["last_counter"], 13)

    def test_final_drain_uses_same_stop_mask_for_data_and_counters(self):
        self.start()
        self.window.metrics = PacketLossTracker(sample_rate=1000)
        self.window._delsys.sync_start_unix_ns = 100_000_000_000
        self.board.pending = np.arange(8, dtype=np.float32).reshape(2, 4)
        self.board._last_counter = np.array([20, 21, 22, 23])

        trial = self.window._stop_all(
            reason="test_stop", stop_boundary_unix_ns=100_002_000_000
        )

        self.assertEqual(trial["samples"], 3)
        diagnostics = trial["packet_diagnostics"]
        self.assertEqual(diagnostics["received"], 3)
        self.assertEqual(diagnostics["first_counter"], 20)
        self.assertEqual(diagnostics["last_counter"], 22)
        packet_lines = Path(trial["packet_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual([line.split(",")[1] for line in packet_lines[1:]],
                         ["20", "21", "22"])

    def test_each_synchronized_trial_resets_packet_diagnostics(self):
        self.window.metrics = PacketLossTracker(sample_rate=1000)
        self.start()
        self.board.pending = np.ones((2, 2), dtype=np.float32)
        self.board._last_counter = np.array([10, 12])
        self.window._on_poll()
        first = self.window._stop_all(reason="first")
        self.assertEqual(first["packet_diagnostics"]["lost"], 1)

        self.start()
        self.board.pending = np.ones((2, 2), dtype=np.float32)
        self.board._last_counter = np.array([40, 41])
        self.window._on_poll()
        second = self.window._stop_all(reason="second")

        self.assertEqual(second["packet_diagnostics"]["received"], 2)
        self.assertEqual(second["packet_diagnostics"]["lost"], 0)
        self.assertEqual(second["packet_diagnostics"]["first_counter"], 40)

    def test_missing_counter_marks_packet_diagnostics_incomplete(self):
        self.window.metrics = PacketLossTracker(sample_rate=1000)
        self.start()
        self.board.pending = np.ones((2, 3), dtype=np.float32)
        self.board._last_counter = None

        self.window._on_poll()
        trial = self.window._stop_all(reason="missing_counter")

        diagnostics = trial["packet_diagnostics"]
        self.assertFalse(diagnostics["complete"])
        self.assertEqual(diagnostics["missing_counter_batch_count"], 1)
        self.assertEqual(diagnostics["missing_counter_sample_count"], 3)

    def test_stop_cleanup_continues_when_stop_alignment_write_fails(self):
        self.start()
        with patch.object(
            self.recorder, "mark_acquisition_stop", side_effect=RuntimeError("disk state")
        ):
            self.window._finish_sync_stop({
                "result": {
                    "running": True,
                    "sync_stop_monotonic_ns": 10_000_000_000,
                    "sync_stop_unix_ns": 20_000_000_000,
                }
            })

        self.assertFalse(self.board.is_streaming)
        self.assertFalse(self.recorder.is_recording)
        self.assertFalse(self.window._delsys.active)
        self.assertIn("STOP 时间基准写入失败", self.window._sync_label.text())

    def test_missing_timestamps_records_boundary_filter_warning(self):
        self.start()
        self.window._delsys.sync_start_unix_ns = 100_000_000_000
        data = np.ones((2, 3), dtype=np.float32)

        returned, timestamps = self.window._post_sync_samples(data, None)

        np.testing.assert_array_equal(returned, data)
        self.assertIsNone(timestamps)
        self.assertIn("无法可靠执行", self.window._delsys.warning)
        self.assertTrue(any(
            event["event"] == "openbci_boundary_filter_skipped"
            for event in self.events()
        ))

    def test_final_drain_records_first_data_and_returns_report_path(self):
        self.start()
        self.board.pending = np.ones((2, 3), dtype=np.float32)
        report_path = Path(self.temp.name) / "sync_report.json"
        with patch("delsys_controller.generate_sync_report",
                   return_value=(report_path, {"status": "estimated"})):
            self.window._on_stop()
            self.wait_job()

        events = self.events()
        self.assertEqual(
            sum(event["event"] == "first_openbci_data_received" for event in events),
            1,
        )
        self.assertEqual(self.window._delsys.last_report_path, report_path)
        self.assertIn("同步报告 JSON 已生成", self.window._sync_label.text())
        self.assertIn(str(report_path), self.window.statusBar().currentMessage())

    def test_stop_all_adds_sync_report_path_to_trial_result(self):
        self.start()
        report_path = Path(self.temp.name) / "synchronization_report.json"
        with patch("delsys_controller.generate_sync_report",
                   return_value=(report_path, {"status": "incomplete"})):
            trial = self.window._stop_all(reason="synthetic_error")

        self.assertEqual(trial["sync_report_path"], str(report_path))

    def test_final_drain_failure_still_closes_trial_and_resets_ui(self):
        self.start()

        with patch.object(self.board, "get_new_data", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(OSError, "read failed"):
                self.window._stop_all(reason="synthetic_error")

        self.assertFalse(self.board.is_streaming)
        self.assertFalse(self.recorder.is_recording)
        self.assertFalse(self.window._recording)
        self.assertFalse(self.window._streaming)
        self.assertTrue(self.window._btn_start.isEnabled())

    def test_report_failure_is_visible_after_successful_stop(self):
        self.start()
        with patch("delsys_controller.generate_sync_report",
                   side_effect=OSError("disk full")):
            self.window._on_stop()
            self.wait_job()

        self.assertIn("STOP 脉冲已执行", self.window._sync_label.text())
        self.assertIn("同步报告 JSON 生成失败", self.window._sync_label.text())

    def test_lost_start_ack_keeps_recording_warns_and_never_blindly_stops(self):
        self.nano.fail_start = True
        self.start()
        self.assertTrue(self.board.is_streaming)
        self.assertIn("未确认", self.window._sync_label.text())
        self.assertFalse(self.nano.ready)
        self.recorder.append(np.ones((2, 2)), np.array([1., 1.001]))
        self.window._on_stop()
        self.wait_job()
        self.assertEqual(self.nano.commands, ["START"])
        self.assertIn("未确认", self.window._sync_label.text())
        self.assertEqual(self.recorder.list_trials()[0]["samples"], 2)
        self.assertIn("stop_unconfirmed", [event["event"] for event in self.events()])

    def test_disconnect_does_not_emit_stop_and_warns(self):
        self.start()
        self.window._on_connect()
        self.assertEqual(self.nano.commands, ["START"])
        self.assertTrue(self.board.is_connected)
        self.assertTrue(self.window._recording)
        self.assertIn("停止", self.window._sync_label.text() + self.window.statusBar().currentMessage())

    def test_close_is_blocked_without_emitting_stop(self):
        self.start()
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertEqual(self.nano.commands, ["START"])
        self.assertFalse(self.nano.closed)
        self.assertTrue(self.window._recording)

    def test_exception_cleanup_does_not_emit_stop_and_logs_warning(self):
        self.start()
        self.window._stop_all(reason="synthetic_error")
        self.assertEqual(self.nano.commands, ["START"])
        finished = [item for item in self.events() if item["event"] == "trial_finished"]
        self.assertIn("STOP 未确认", finished[0]["warning"])

    def test_worker_blocks_close_but_not_gui_events(self):
        self.enable()
        self.board.apply_gate = threading.Event()
        self.window._on_start()
        self.assertTrue(self.board.apply_entered.wait(1))
        self.assertFalse(self.window._btn_start.isEnabled())
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        processed = []
        QTimer.singleShot(0, lambda: processed.append(True))
        QApplication.processEvents()
        self.assertEqual(processed, [True])
        self.assertTrue(self.board.is_connected)
        self.assertEqual(self.nano.commands, [])
        self.board.apply_gate.set()
        self.wait_job()
        self.window._on_stop()
        self.wait_job()


if __name__ == "__main__":
    unittest.main()
