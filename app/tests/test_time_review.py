import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from config_viewer import ViewerConfig
from recorder import Recorder
from time_review import (
    default_middle_window,
    load_time_review,
    load_trial_txt,
    save_time_review,
)
from ui.time_review_dialog import TimeReviewDialog


class TimeReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        recorder = Recorder(self.temp_dir.name, n_channels=2)
        recorder.start_recording({"sampling_rate": 1000, "task": "test"})
        t = np.arange(1000) / 1000.0
        recorder.append(
            np.vstack([
                20 * np.sin(2 * np.pi * 40 * t),
                10 * np.sin(2 * np.pi * 80 * t),
            ]),
            100.0 + t,
        )
        self.trial = recorder.stop_recording()
        self.trial_dir = Path(self.trial["path"])
        self.txt_path = Path(self.trial["txt_path"])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_recorder_txt_and_defaults_to_middle_seventy_percent(self):
        data = load_trial_txt(self.txt_path)

        self.assertEqual(data.emg.shape, (1000, 2))
        self.assertEqual(data.sample_rate, 1000.0)
        self.assertEqual(default_middle_window(data.n_samples), (150, 850))

    def test_saving_review_updates_one_row_without_modifying_raw_txt(self):
        raw_hash = hashlib.sha256(self.txt_path.read_bytes()).digest()
        save_time_review(self.trial_dir, 150, 850, 1000, 1000.0)
        save_time_review(self.trial_dir, 200, 900, 1000, 1000.0)

        stored = load_time_review(self.trial_dir)
        csv_lines = (self.trial_dir.parent / "review_segments.csv").read_text(
            encoding="utf-8-sig"
        ).splitlines()
        self.assertEqual(stored["start_sample"], 200)
        self.assertEqual(stored["end_sample_exclusive"], 900)
        self.assertEqual(len(csv_lines), 2)
        self.assertEqual(hashlib.sha256(self.txt_path.read_bytes()).digest(), raw_hash)

    def test_dialog_saves_dragged_range(self):
        dialog = TimeReviewDialog(
            self.trial_dir,
            viewer_config=ViewerConfig(n_channels=2),
        )
        dialog._set_range(0.2, 0.9)
        dialog._save()

        stored = load_time_review(self.trial_dir)
        self.assertEqual(stored["start_sample"], 200)
        self.assertEqual(stored["end_sample_exclusive"], 900)
        self.assertEqual(len(dialog._regions), 2)
        dialog.close()

    def test_deleting_trial_removes_its_review_row(self):
        save_time_review(self.trial_dir, 150, 850, 1000, 1000.0)
        recorder = Recorder(self.temp_dir.name, n_channels=2)
        recorder._session_dir = self.trial_dir.parent
        recorder._session_active = True

        self.assertTrue(recorder.delete_trial(self.trial_dir))
        self.assertIsNone(load_time_review(self.trial_dir))


if __name__ == "__main__":
    unittest.main()
