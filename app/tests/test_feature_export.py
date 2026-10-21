import csv
import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from action_review import remove_action_segments, save_action_segments
from config_viewer import ViewerConfig
from feature_export import (
    _diagnostic_psd,
    _quality_values,
    export_trial_features,
    normalize_task_code,
    results_dir,
    task_display_name,
)
from recorder import Recorder
from time_review import load_time_review, save_time_review
from ui.feature_export_dialog import BatchExportResult, FeatureExportDialog, _ExportWorker


class FeatureExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        recorder = Recorder(self.temp_dir.name, n_channels=2)
        recorder.start_recording({"sampling_rate": 1000, "task": "sit_base"})
        t = np.arange(10000) / 1000.0
        amplitude = 2.0 + sum(
            80 * np.exp(-0.5 * ((t - peak) / 0.18) ** 2)
            for peak in (2.0, 4.0, 6.0, 8.0)
        )
        recorder.append(
            np.vstack([
                amplitude * np.sin(2 * np.pi * 80 * t),
                amplitude * np.sin(2 * np.pi * 90 * t),
            ]),
            100.0 + t,
        )
        self.trial = recorder.stop_recording()
        self.trial_dir = Path(self.trial["path"])
        self.txt_path = Path(self.trial["txt_path"])
        save_time_review(self.trial_dir, 1000, 9000, 10000, 1000.0)
        save_action_segments(
            self.trial_dir,
            0,
            [
                {"start_sample": 1000, "peak_sample": 2000, "end_sample_exclusive": 3000},
                {"start_sample": 3000, "peak_sample": 4000, "end_sample_exclusive": 5000},
                {"start_sample": 5000, "peak_sample": 6000, "end_sample_exclusive": 7000},
                {"start_sample": 7000, "peak_sample": 8000, "end_sample_exclusive": 9000},
            ],
            10000,
            1000.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _read_rows(self):
        with (results_dir(self.trial_dir) / "features.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as source:
            return list(csv.DictReader(source))

    def test_exports_only_selected_metric_for_each_action_and_channel(self):
        raw_hash = hashlib.sha256(self.txt_path.read_bytes()).digest()

        result = export_trial_features(
            self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2)
        )

        rows = self._read_rows()
        self.assertEqual(result.scope, "action")
        self.assertEqual(result.row_count, 8)
        self.assertEqual(len(rows), 8)
        self.assertIn("MNF_Hz", rows[0])
        self.assertNotIn("MDF_Hz", rows[0])
        self.assertTrue((result.output_dir / "mnf.png").is_file())
        self.assertFalse((result.output_dir / "mdf.png").exists())
        self.assertTrue(70.0 < float(rows[0]["MNF_Hz"]) < 90.0)
        self.assertEqual(hashlib.sha256(self.txt_path.read_bytes()).digest(), raw_hash)

    def test_reexport_replaces_stale_outputs_with_current_selection(self):
        export_trial_features(self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2))

        result = export_trial_features(
            self.trial_dir, {"envelope", "mdf"}, ViewerConfig(n_channels=2)
        )

        rows = self._read_rows()
        self.assertIn("MDF_Hz", rows[0])
        self.assertNotIn("MNF_Hz", rows[0])
        self.assertTrue((result.output_dir / "envelope.png").is_file())
        self.assertTrue((result.output_dir / "mdf.png").is_file())
        self.assertFalse((result.output_dir / "mnf.png").exists())

    def test_time_review_change_invalidates_previous_exports(self):
        export_trial_features(self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2))
        self.assertTrue(results_dir(self.trial_dir).is_dir())

        save_time_review(self.trial_dir, 1500, 8500, 10000, 1000.0)

        self.assertFalse(results_dir(self.trial_dir).exists())

    def test_action_change_invalidates_previous_exports(self):
        export_trial_features(self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2))

        save_action_segments(
            self.trial_dir,
            0,
            [{"start_sample": 1500, "peak_sample": 2000, "end_sample_exclusive": 2500}],
            10000,
            1000.0,
        )

        self.assertFalse(results_dir(self.trial_dir).exists())

    def test_without_action_segments_exports_reviewed_time_window(self):
        remove_action_segments(self.trial_dir)

        result = export_trial_features(
            self.trial_dir, {"mdf"}, ViewerConfig(n_channels=2)
        )

        rows = self._read_rows()
        self.assertEqual(result.scope, "time_window")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["scope"] == "time_window" for row in rows))
        self.assertTrue(all(row["start_sample"] == "1000" for row in rows))
        self.assertTrue(all(row["end_sample_exclusive"] == "9000" for row in rows))

    def test_without_reviews_exports_full_trial_without_creating_review(self):
        other = self.trial_dir.parent / "trial_without_review"
        other.mkdir()
        other_txt = other / "emg_data.txt"
        other_txt.write_bytes(self.txt_path.read_bytes())
        raw_hash = hashlib.sha256(other_txt.read_bytes()).digest()

        result = export_trial_features(
            other, {"mnf"}, ViewerConfig(n_channels=2)
        )

        with (results_dir(other) / "features.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(result.scope, "full_trial")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["scope"] == "full_trial" for row in rows))
        self.assertTrue(all(row["start_sample"] == "0" for row in rows))
        self.assertTrue(all(row["end_sample_exclusive"] == "10000" for row in rows))
        self.assertIsNone(load_time_review(other))
        self.assertEqual(hashlib.sha256(other_txt.read_bytes()).digest(), raw_hash)

    def test_dialog_reports_selected_metrics(self):
        dialog = FeatureExportDialog(self.trial_dir, ViewerConfig(n_channels=2))
        dialog._mnf.setChecked(False)

        self.assertEqual(dialog.selected_metrics(), {"envelope", "mdf"})
        dialog.close()

    def test_dialog_exposes_new_visualization_choices(self):
        dialog = FeatureExportDialog(self.trial_dir, ViewerConfig(n_channels=2))
        dialog._raw.setChecked(True)
        dialog._spectrum.setChecked(True)
        dialog._pli.setChecked(True)
        dialog._mar.setChecked(True)

        self.assertEqual(
            dialog.selected_metrics(),
            {"raw", "envelope", "spectrum", "pli", "mar", "mnf", "mdf"},
        )
        dialog.close()

    def test_export_worker_returns_result_without_raising(self):
        worker = _ExportWorker(self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2))
        results = []
        failures = []
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)
        expected = object()

        with patch(
            "ui.feature_export_dialog.export_trial_features",
            return_value=expected,
        ):
            worker.run()

        self.assertEqual(results, [expected])
        self.assertEqual(failures, [])

    def test_export_worker_reports_unexpected_error_without_raising(self):
        worker = _ExportWorker(self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2))
        failures = []
        worker.failed.connect(failures.append)

        with patch(
            "ui.feature_export_dialog.export_trial_features",
            side_effect=RuntimeError("synthetic export failure"),
        ), patch("ui.feature_export_dialog.traceback.print_exc"):
            worker.run()

        self.assertEqual(failures, ["synthetic export failure"])

    def test_batch_worker_continues_after_one_trial_fails(self):
        second = self.trial_dir.parent / "second"
        third = self.trial_dir.parent / "third"
        second.mkdir()
        third.mkdir()
        worker = _ExportWorker(
            [self.trial_dir, second, third], {"mnf"}, ViewerConfig(n_channels=2)
        )
        emitted = []
        worker.finished.connect(emitted.append)

        with patch(
            "ui.feature_export_dialog.export_trial_features",
            side_effect=["first", ValueError("broken"), "third"],
        ), patch("ui.feature_export_dialog.traceback.print_exc"):
            worker.run()

        self.assertEqual(len(emitted), 1)
        self.assertIsInstance(emitted[0], BatchExportResult)
        self.assertEqual(emitted[0].results, ("first", "third"))
        self.assertEqual(len(emitted[0].failures), 1)
        self.assertEqual(emitted[0].failures[0]["path"], str(second.resolve()))

    def test_dialog_exports_on_worker_thread_and_cleans_it_up(self):
        dialog = FeatureExportDialog(self.trial_dir, ViewerConfig(n_channels=2))
        expected = object()
        worker_threads = []

        def fake_export(*_args, **_kwargs):
            worker_threads.append(QThread.currentThread())
            return expected

        with patch(
            "ui.feature_export_dialog.export_trial_features",
            side_effect=fake_export,
        ):
            dialog._export()
            deadline = time.monotonic() + 2.0
            while dialog._export_thread is not None and time.monotonic() < deadline:
                self.app.processEvents()

        self.assertIsNone(dialog._export_thread)
        self.assertIs(dialog.result, expected)
        self.assertEqual(len(worker_threads), 1)
        self.assertIsNot(worker_threads[0], self.app.thread())
        dialog.close()

    def test_exports_new_diagnostic_figures_without_changing_raw_txt(self):
        raw_hash = hashlib.sha256(self.txt_path.read_bytes()).digest()

        result = export_trial_features(
            self.trial_dir,
            {"raw", "filtered", "spectrum", "pli", "mar"},
            ViewerConfig(n_channels=2),
        )

        self.assertEqual(result.row_count, 0)
        self.assertIsNone(result.csv_path)
        self.assertEqual(
            set(result.image_paths),
            {"raw", "filtered", "spectrum", "pli", "mar"},
        )
        self.assertTrue(all(path.is_file() for path in result.image_paths.values()))
        self.assertEqual(hashlib.sha256(self.txt_path.read_bytes()).digest(), raw_hash)

    def test_raw_figure_selection_does_not_run_filters(self):
        with patch("feature_export.apply_filters") as filter_mock:
            result = export_trial_features(
                self.trial_dir, {"raw"}, ViewerConfig(n_channels=2)
            )

        filter_mock.assert_not_called()
        self.assertEqual(set(result.image_paths), {"raw"})

    def test_quality_bars_distinguish_line_noise_and_low_frequency_artifact(self):
        sample_rate = 1000.0
        t = np.arange(0.0, 4.0, 1.0 / sample_rate)
        signals = [
            100.0 * np.sin(2 * np.pi * 50.0 * t)
            + 2.0 * np.sin(2 * np.pi * 80.0 * t),
            100.0 * np.sin(2 * np.pi * 10.0 * t)
            + 2.0 * np.sin(2 * np.pi * 80.0 * t),
        ]

        values = _quality_values(
            signals, sample_rate, 50.0, 0, len(t), {"pli", "mar"}
        )

        self.assertGreater(values["pli"][0], values["pli"][1])
        self.assertGreater(values["mar"][1], values["mar"][0])

    def test_diagnostic_psd_keeps_each_block_at_or_below_ten_seconds(self):
        signal = np.arange(25500, dtype=float)
        calls = []

        def fake_psd(block, sample_rate, normalize=False):
            calls.append(np.asarray(block).copy())
            return np.array([0.0, 1.0]), np.array([1.0, 2.0])

        with patch("feature_export.compute_psd", side_effect=fake_psd):
            _diagnostic_psd(signal, 1000.0)

        self.assertGreater(len(calls), 1)
        self.assertTrue(all(len(block) <= 10000 for block in calls))

    def test_diagnostic_psd_averages_block_power(self):
        powers = iter((
            np.array([1.0, 3.0]),
            np.array([3.0, 5.0]),
            np.array([5.0, 7.0]),
        ))

        with patch(
            "feature_export.compute_psd",
            side_effect=lambda *_args, **_kwargs: (
                np.array([0.0, 1.0]), next(powers)
            ),
        ):
            frequencies, power = _diagnostic_psd(
                np.arange(25000, dtype=float), 1000.0
            )

        np.testing.assert_array_equal(frequencies, [0.0, 1.0])
        np.testing.assert_array_equal(power, [3.0, 5.0])

    def test_diagnostic_psd_short_signal_matches_regular_psd(self):
        sample_rate = 1000.0
        t = np.arange(4000) / sample_rate
        signal = np.sin(2.0 * np.pi * 80.0 * t)

        expected = _diagnostic_psd(signal, sample_rate, chunk_sec=100.0)
        actual = _diagnostic_psd(signal, sample_rate)

        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])

    def test_known_task_codes_have_normalized_chinese_display_names(self):
        expected = {
            "sit_base": "坐姿静息 (sit_base)",
            "std_df": "站姿背屈 (std_df)",
            "sit_pf": "坐姿跖屈 (sit_pf)",
            "std_pf": "站姿跖屈 (std_pf)",
            "walk1.5": "跑步机 1.5 km/h 行走 (walk1.5)",
            "sit_tf": "坐姿趾屈 (sit_tf)",
            "std_te": "站姿趾伸 (std_te)",
        }
        for code, label in expected.items():
            self.assertEqual(task_display_name(code), label)

        self.assertEqual(normalize_task_code("SIT_BASE_2"), "sit_base")
        self.assertEqual(task_display_name("custom_task"), "custom_task")

    def test_export_passes_task_from_txt_header_to_plot(self):
        with patch("feature_export._plot_frequency_metric") as plot_mock:
            export_trial_features(
                self.trial_dir, {"mnf"}, ViewerConfig(n_channels=2)
            )

        self.assertEqual(plot_mock.call_args.args[3], "sit_base")

    def test_requires_at_least_one_metric(self):
        with self.assertRaisesRegex(ValueError, "至少选择"):
            export_trial_features(self.trial_dir, set(), ViewerConfig(n_channels=2))


if __name__ == "__main__":
    unittest.main()
