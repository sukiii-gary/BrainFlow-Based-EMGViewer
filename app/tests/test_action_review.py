import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from action_review import (
    detect_rhythmic_segments,
    load_action_segments,
    save_action_segments,
)
from config_viewer import ViewerConfig
from recorder import Recorder
from time_review import save_time_review
from ui.action_review_dialog import ActionReviewDialog


class ActionReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        recorder = Recorder(self.temp_dir.name, n_channels=2)
        recorder.start_recording({"sampling_rate": 1000, "task": "背屈"})
        t = np.arange(10000) / 1000.0
        amplitude = 2.0 + sum(
            80 * np.exp(-0.5 * ((t - peak) / 0.18) ** 2)
            for peak in (2.0, 4.0, 6.0, 8.0)
        )
        recorder.append(
            np.vstack([
                amplitude * np.sin(2 * np.pi * 80 * t),
                0.7 * amplitude * np.sin(2 * np.pi * 90 * t),
            ]),
            100.0 + t,
        )
        self.trial = recorder.stop_recording()
        self.trial_dir = Path(self.trial["path"])
        save_time_review(self.trial_dir, 1000, 9000, 10000, 1000.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detects_four_synthetic_repetitions(self):
        t = np.arange(10000) / 1000.0
        envelope = np.full(10000, 2.0)
        for peak in (2.0, 4.0, 6.0, 8.0):
            envelope += 80 * np.exp(-0.5 * ((t - peak) / 0.12) ** 2)

        segments = detect_rhythmic_segments(envelope, 1000, 1000, 9000)

        self.assertEqual(len(segments), 4)
        self.assertEqual([item["peak_sample"] for item in segments], [2000, 4000, 6000, 8000])

    def test_saving_replaces_trial_rows_and_rejects_overlap(self):
        initial = [
            {"start_sample": 1000, "peak_sample": 2000, "end_sample_exclusive": 3000},
            {"start_sample": 3000, "peak_sample": 4000, "end_sample_exclusive": 5000},
        ]
        save_action_segments(self.trial_dir, 0, initial, 10000, 1000.0)
        updated = [
            {"start_sample": 1200, "peak_sample": 2000, "end_sample_exclusive": 2800}
        ]
        save_action_segments(self.trial_dir, 0, updated, 10000, 1000.0)

        stored = load_action_segments(self.trial_dir)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["start_sample"], 1200)

        overlapping = [
            {"start_sample": 1000, "peak_sample": 2000, "end_sample_exclusive": 3500},
            {"start_sample": 3000, "peak_sample": 4000, "end_sample_exclusive": 5000},
        ]
        with self.assertRaises(ValueError):
            save_action_segments(self.trial_dir, 0, overlapping, 10000, 1000.0)

    def test_saving_rejects_segments_outside_reviewed_time_range(self):
        outside = [
            {"start_sample": 500, "peak_sample": 1000, "end_sample_exclusive": 1500}
        ]

        with self.assertRaisesRegex(ValueError, "已审核时间范围"):
            save_action_segments(self.trial_dir, 0, outside, 10000, 1000.0)

    def test_changing_time_review_invalidates_action_segments(self):
        segments = [
            {"start_sample": 1000, "peak_sample": 2000, "end_sample_exclusive": 3000}
        ]
        save_action_segments(self.trial_dir, 0, segments, 10000, 1000.0)

        save_time_review(self.trial_dir, 1500, 8500, 10000, 1000.0)

        self.assertEqual(load_action_segments(self.trial_dir), [])

    def test_dialog_detects_moves_and_saves_segments(self):
        dialog = ActionReviewDialog(
            self.trial_dir,
            viewer_config=ViewerConfig(n_channels=2),
        )
        self.assertEqual(len(dialog.segments), 4)

        dialog._regions[0].setRegion((1.2, 2.8))
        dialog._region_changed(0, dialog._regions[0])
        dialog._save()

        stored = load_action_segments(self.trial_dir)
        self.assertEqual(len(stored), 4)
        self.assertEqual(stored[0]["start_sample"], 1200)
        self.assertEqual(stored[0]["end_sample_exclusive"], 2800)
        dialog.close()

    def test_dialog_deletes_and_adds_segment(self):
        dialog = ActionReviewDialog(
            self.trial_dir,
            viewer_config=ViewerConfig(n_channels=2),
        )
        dialog._table.selectRow(1)
        dialog._delete_segment()
        self.assertEqual(len(dialog.segments), 3)

        dialog._add_segment()

        self.assertEqual(len(dialog.segments), 4)
        self.assertTrue(any(
            segment["start_sample"] <= 4000 < segment["end_sample_exclusive"]
            for segment in dialog.segments
        ))
        dialog.close()

    def test_deleting_trial_removes_action_and_time_review_rows(self):
        segments = [
            {"start_sample": 1000, "peak_sample": 2000, "end_sample_exclusive": 3000}
        ]
        save_action_segments(self.trial_dir, 0, segments, 10000, 1000.0)
        recorder = Recorder(self.temp_dir.name, n_channels=2)
        recorder._session_dir = self.trial_dir.parent
        recorder._session_active = True

        self.assertTrue(recorder.delete_trial(self.trial_dir))
        self.assertEqual(load_action_segments(self.trial_dir), [])
        self.assertFalse(self.trial_dir.exists())


if __name__ == "__main__":
    unittest.main()
