import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from config_viewer import ViewerConfig
from action_review import save_action_segments
from recorder import Recorder
from time_review import save_time_review
from ui.main_window import MainWindow


def wait_for_connection(window, timeout=2.0):
    deadline = time.monotonic() + timeout
    while window._connect_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    if window._connect_thread is not None:
        raise AssertionError("connection worker did not finish")


class FakeSettings:
    values = {}

    def __init__(self, *_args):
        pass

    def value(self, key, default="", type=None):
        value = self.values.get(key, default)
        return type(value) if type is not None else value

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


class FakeBoard:
    board_id = -1
    sample_rate = 1000.0
    n_channels = 2
    data_stream_verified = False

    def __init__(self):
        self.is_connected = False
        self.is_streaming = False

    def set_channel_settings(self, _settings):
        pass

    def connect(self):
        self.is_connected = True
        return True

    def disconnect(self):
        self.is_connected = False

    def start_stream(self):
        self.is_streaming = True
        return True

    def stop_stream(self):
        self.is_streaming = False

    def get_new_data(self, n_samples=None):
        return np.zeros((2, 0), dtype=np.float32)


class FakeChain:
    bandpass = (20.0, 450.0)
    notch_hz = 50.0
    notch_bw = 8.0
    notch_harmonics = 3
    envelope_ms = 100.0

    def update_fs(self, _sample_rate):
        pass

    def clear(self):
        pass


class SessionTrialWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_start_and_stop_automatically_record_one_trial(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            config = ViewerConfig(n_channels=2, sampling_rate=1000.0)
            window = MainWindow(
                board_manager=board,
                signal_chain=FakeChain(),
                recorder=recorder,
                config=config,
            )
            window._sidebar.set_recording_metadata(subject="S01", task="sit_base")

            window._on_connect()
            wait_for_connection(window)
            session = recorder.session_dir
            window._on_start()
            recorder.append(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([100.0, 100.001]),
            )
            window._on_stop()

            trials = recorder.list_trials()
            self.assertEqual(len(trials), 1)
            self.assertEqual(window._session_table.rowCount(), 1)
            self.assertEqual(window._session_table.item(0, 5).text(), "已保存")
            self.assertEqual(
                window.dockWidgetArea(window._session_dock),
                Qt.DockWidgetArea.RightDockWidgetArea,
            )
            self.assertIn(
                str(Path(temp_dir).resolve()),
                window._recordings_path_label.text(),
            )
            self.assertEqual(trials[0]["path"].parent, session)
            self.assertRegex(session.name, r"^\d{8}_S01_Session\d+$")
            self.assertEqual(trials[0]["path"].name, "sit_base")
            self.assertTrue(Path(trials[0]["txt_path"]).is_file())
            self.assertFalse(list(session.rglob("*.h5")))
            self.assertFalse(list(session.rglob("*.json")))

            trial_dir = trials[0]["path"]
            save_time_review(trial_dir, 0, 2, 2, 1000.0)
            save_action_segments(
                trial_dir,
                reference_channel=0,
                segments=[{
                    "start_sample": 0,
                    "peak_sample": 0,
                    "end_sample_exclusive": 2,
                }],
                n_samples=2,
                sample_rate=1000.0,
            )
            window._refresh_session_trials()
            self.assertEqual(window._session_table.item(0, 6).text(), "已审查")
            self.assertEqual(window._session_table.item(0, 7).text(), "1 段已审查")
            self.assertTrue(window._btn_review_action.isEnabled())
            window._session_table.selectRow(0)
            with patch(
                "ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window._on_delete_selected_trial()
            self.assertFalse(trial_dir.exists())
            self.assertEqual(window._session_table.rowCount(), 0)
            window.close()

    def test_connection_allows_empty_subject_but_recording_requires_it(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            window = MainWindow(
                board_manager=board,
                signal_chain=FakeChain(),
                recorder=recorder,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )

            window._on_connect()
            wait_for_connection(window)

            self.assertTrue(board.is_connected)
            self.assertIsNone(recorder.session_dir)
            window._on_start()
            self.assertFalse(board.is_streaming)
            self.assertIn("请先填写被试 ID", window.statusBar().currentMessage())

            window._sidebar.set_recording_metadata(subject="S02", task="sit_base")
            window._on_start()
            self.assertTrue(board.is_streaming)
            self.assertTrue(recorder.is_session_active)
            self.assertRegex(recorder.session_dir.name, r"^\d{8}_S02_Session\d+$")
            window.close()

    def test_missing_session_folder_self_heals_on_next_start(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            window = MainWindow(
                board_manager=board, signal_chain=FakeChain(), recorder=recorder,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._sidebar.set_recording_metadata(subject="S01", task="after_delete")
            window._on_connect()
            wait_for_connection(window)
            deleted_session = recorder.session_dir
            deleted_session.rmdir()

            window._refresh_session_trials()
            self.assertFalse(window._btn_open_session.isEnabled())
            window._on_start()

            self.assertTrue(board.is_streaming)
            self.assertTrue(recorder.session_dir.is_dir())
            window._on_stop()
            window.close()

    def test_new_and_delete_session_do_not_require_reconnecting_board(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            window = MainWindow(
                board_manager=board, signal_chain=FakeChain(), recorder=recorder,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._sidebar.set_recording_metadata(subject="S01", task="task")
            window._on_connect()
            wait_for_connection(window)
            first = recorder.session_dir

            window._on_new_session()
            second = recorder.session_dir

            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertNotEqual(first, second)
            old_marker = second / "old_session_marker.txt"
            old_marker.write_text("old", encoding="utf-8")
            with patch(
                "ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window._on_delete_session()
            self.assertFalse(second.exists())
            self.assertIsNone(recorder.session_dir)
            self.assertFalse(old_marker.exists())
            self.assertTrue(board.is_connected)

            window._on_start()
            replacement = recorder.session_dir
            self.assertTrue(replacement.is_dir())
            self.assertTrue(board.is_streaming)
            window._on_stop()
            window.close()

    def test_delete_session_does_not_immediately_recreate_the_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            window = MainWindow(
                board_manager=board, signal_chain=FakeChain(), recorder=recorder,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._sidebar.set_recording_metadata(subject="S01", task="task")
            window._on_connect()
            wait_for_connection(window)
            deleted = recorder.session_dir

            with patch(
                "ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ), patch.object(
                recorder, "start_session", wraps=recorder.start_session
            ) as start_session:
                window._on_delete_session()

            start_session.assert_not_called()
            self.assertFalse(deleted.exists())
            self.assertIsNone(recorder.session_dir)
            self.assertEqual(window._session_table.rowCount(), 0)
            self.assertIn("已删除 Session", window.statusBar().currentMessage())
            window.close()

    def test_trial_table_multiselects_and_batch_deletes(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ui.main_window.QSettings", FakeSettings
        ):
            board = FakeBoard()
            recorder = Recorder(data_dir=temp_dir, n_channels=2)
            window = MainWindow(
                board_manager=board, signal_chain=FakeChain(), recorder=recorder,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._sidebar.set_recording_metadata(subject="S01", task="task")
            window._on_connect()
            wait_for_connection(window)
            trials = []
            for task in ("one", "two", "three"):
                recorder.start_recording({
                    "subject_id": "S01", "task": task, "sampling_rate": 1000,
                })
                recorder.append(np.ones((2, 2)), np.array([1.0, 1.001]))
                trials.append(recorder.stop_recording())
            window._refresh_session_trials()
            selection = window._session_table.selectionModel()
            selection.clearSelection()
            flags = (
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows
            )
            selection.select(window._session_table.model().index(0, 0), flags)
            selection.select(window._session_table.model().index(2, 0), flags)

            self.assertEqual(
                window._selected_trial_paths(),
                [trials[0]["path"], trials[2]["path"]],
            )
            self.assertFalse(window._btn_open_trial.isEnabled())
            self.assertFalse(window._btn_review_time.isEnabled())
            self.assertTrue(window._btn_export_features.isEnabled())
            self.assertTrue(window._btn_delete_trial.isEnabled())
            with patch(
                "ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window._on_delete_selected_trial()

            self.assertFalse(trials[0]["path"].exists())
            self.assertTrue(trials[1]["path"].exists())
            self.assertFalse(trials[2]["path"].exists())
            self.assertEqual(window._session_table.rowCount(), 1)
            window.close()

    def test_eight_channel_display_uses_five_hz_refresh(self):
        with patch("ui.main_window.QSettings", FakeSettings):
            window = MainWindow(
                board_manager=FakeBoard(),
                signal_chain=FakeChain(),
                config=ViewerConfig(n_channels=8, ui_refresh_ms=100),
            )

            self.assertEqual(window._display_timer.interval(), 200)
            window.close()

    def test_hotspot_uses_recent_filtered_rms_for_hidden_channels(self):
        class DisplayChain(FakeChain):
            def __init__(self):
                self.filtered_calls = []

            def get_filtered(self, channel, n_samples=None):
                self.filtered_calls.append((channel, n_samples))
                return np.full(n_samples or 0, channel + 1.0)

            def get_envelope(self, _channel, n_samples=None):
                return np.zeros(n_samples or 0)

        with patch("ui.main_window.QSettings", FakeSettings):
            chain = DisplayChain()
            window = MainWindow(
                board_manager=FakeBoard(),
                signal_chain=chain,
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._sidebar._ch_checkboxes[0].setChecked(False)

            window._on_update_display()

            self.assertIn((0, 250), chain.filtered_calls)
            self.assertIn((1, 250), chain.filtered_calls)
            self.assertTrue(window._activation_map._has_data)
            window.close()

    def test_poll_exception_stops_trial_without_escaping_qt_callback(self):
        class ExplodingChain(FakeChain):
            def push_samples(self, _data):
                raise RuntimeError("synthetic poll failure")

        class DataBoard(FakeBoard):
            def get_new_data(self, n_samples=None):
                return np.ones((2, 4), dtype=np.float32)

        with patch("ui.main_window.QSettings", FakeSettings):
            board = DataBoard()
            window = MainWindow(
                board_manager=board,
                signal_chain=ExplodingChain(),
                config=ViewerConfig(n_channels=2, sampling_rate=1000.0),
            )
            window._on_connect()
            wait_for_connection(window)
            window._on_start()
            self.assertTrue(board.is_streaming)

            window._on_poll_timer()

            self.assertFalse(board.is_streaming)
            self.assertIn("数据轮询异常", window.statusBar().currentMessage())
            window.close()


if __name__ == "__main__":
    unittest.main()
