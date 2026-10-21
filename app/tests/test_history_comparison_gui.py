import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QItemSelectionModel
from PyQt6.QtWidgets import QApplication, QMessageBox

from action_review import save_action_segments
from config_viewer import ViewerConfig
from recorder import Recorder
from time_review import save_time_review
from ui.history_comparison_dialog import HistoryComparisonDialog


class HistoryComparisonDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.recorder = Recorder(self.root, n_channels=2)
        self.trials = [
            self._record_trial("S01", "sit_base", 80.0),
            self._record_trial("S02", "walk", 110.0),
        ]
        for trial in self.trials:
            save_time_review(trial, 100, 900, 1000, 1000.0)
            save_action_segments(
                trial,
                reference_channel=0,
                segments=[{
                    "start_sample": 200,
                    "peak_sample": 500,
                    "end_sample_exclusive": 800,
                }],
                n_samples=1000,
                sample_rate=1000.0,
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record_trial(self, subject, task, frequency):
        self.recorder.end_session()
        self.recorder.start_session(subject)
        self.recorder.start_recording({
            "subject_id": subject,
            "task": task,
            "sampling_rate": 1000.0,
        })
        t = np.arange(1000, dtype=float) / 1000.0
        self.recorder.append(
            np.vstack([
                20.0 * np.sin(2.0 * np.pi * frequency * t),
                12.0 * np.sin(2.0 * np.pi * (frequency + 20.0) * t),
            ]),
            100.0 + t,
        )
        record = self.recorder.stop_recording()
        return Path(record["path"])

    @staticmethod
    def _select_rows(dialog, rows):
        selection = dialog._table.selectionModel()
        selection.clearSelection()
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        for row in rows:
            selection.select(dialog._table.model().index(row, 0), flags)
        QApplication.processEvents()

    @staticmethod
    def _wait_for_discovery(dialog, timeout=5.0):
        deadline = time.monotonic() + timeout
        while dialog._discovery_thread is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.002)
        if dialog._discovery_thread is not None:
            raise AssertionError("history discovery did not finish")

    def test_discovers_all_sessions_filters_and_limits_selection(self):
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)

        self.assertEqual(dialog._table.rowCount(), 6)
        self.assertEqual(
            {dialog._table.item(row, 0).text() for row in range(6)},
            {self.trials[0].parent.name, self.trials[1].parent.name},
        )
        dialog._filter.setText("walk")
        QApplication.processEvents()
        visible = [
            row for row in range(6) if not dialog._table.isRowHidden(row)
        ]
        self.assertEqual(len(visible), 3)

        dialog._filter.clear()
        self._select_rows(dialog, [0])
        self.assertFalse(dialog._compare_button.isEnabled())
        self._select_rows(dialog, [0, 1])
        self.assertTrue(dialog._compare_button.isEnabled())
        self._select_rows(dialog, list(range(6)))
        self.assertTrue(dialog._compare_button.isEnabled())
        dialog.close()

    def test_peak_alignment_requires_action_segments(self):
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)
        full_rows = [
            row for row, ref in enumerate(dialog._refs) if ref.source == "full"
        ]
        action_rows = [
            row for row, ref in enumerate(dialog._refs) if ref.source == "action"
        ]

        self._select_rows(dialog, full_rows)
        peak_item = dialog._alignment.model().item(dialog._alignment.findData("peak"))
        self.assertFalse(peak_item.isEnabled())
        self._select_rows(dialog, action_rows)
        self.assertTrue(peak_item.isEnabled())
        dialog.close()

    def test_worker_prepares_and_plots_two_segments(self):
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)
        full_rows = [
            row for row, ref in enumerate(dialog._refs) if ref.source == "full"
        ]
        self._select_rows(dialog, full_rows)
        dialog._mode.setCurrentIndex(dialog._mode.findData("filtered"))
        dialog._start_comparison()

        deadline = time.monotonic() + 5.0
        while dialog._thread is not None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.002)

        self.assertIsNone(dialog._thread)
        self.assertIsNotNone(dialog.last_result)
        self.assertEqual(len(dialog.last_result.series), 2)
        self.assertEqual(dialog._channel.count(), 2)
        self.assertEqual(dialog._channel.itemText(0), "EXG Channel 0")
        self.assertEqual(len(dialog._plot.listDataItems()), 2)
        self.assertIn("共同频带", dialog._result_label.text())
        dialog.close()

    def test_processing_failure_is_reported_without_closing(self):
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)
        self._select_rows(dialog, [0, 3])
        with patch(
            "ui.history_comparison_dialog.prepare_comparison",
            side_effect=ValueError("synthetic history failure"),
        ), patch("ui.history_comparison_dialog.traceback.print_exc"):
            dialog._start_comparison()
            deadline = time.monotonic() + 5.0
            while dialog._thread is not None and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.002)

        self.assertIsNone(dialog.last_result)
        self.assertIn("synthetic history failure", dialog._state_label.text())
        dialog.close()

    def test_rejects_out_of_range_selection(self):
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)
        self._select_rows(dialog, [0])
        with patch(
            "ui.history_comparison_dialog.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Ok,
        ) as warning:
            dialog._start_comparison()
        warning.assert_called_once()
        self.assertIsNone(dialog._thread)
        dialog.close()

    def test_more_than_six_segments_disables_comparison(self):
        self._record_trial("S03", "extra", 60.0)
        dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
        self._wait_for_discovery(dialog)
        self.assertGreater(dialog._table.rowCount(), 6)

        self._select_rows(dialog, list(range(7)))

        self.assertFalse(dialog._compare_button.isEnabled())
        self.assertIn("已选择 7 段", dialog._selection_label.text())
        dialog.close()

    def test_discovery_runs_without_blocking_dialog_construction(self):
        started = threading.Event()
        release = threading.Event()

        def slow_discovery(_root):
            started.set()
            release.wait(2.0)
            return []

        with patch(
            "ui.history_comparison_dialog.discover_history",
            side_effect=slow_discovery,
        ):
            dialog = HistoryComparisonDialog(self.root, ViewerConfig(n_channels=2))
            deadline = time.monotonic() + 1.0
            while not started.is_set() and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.002)
            self.assertTrue(started.is_set())
            self.assertIsNotNone(dialog._discovery_thread)
            self.assertFalse(dialog._table.isEnabled())
            release.set()
            self._wait_for_discovery(dialog)

        self.assertTrue(dialog._table.isEnabled())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
