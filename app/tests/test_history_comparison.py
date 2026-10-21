import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from action_review import save_action_segments
from history_comparison import (
    SegmentRef,
    discover_history,
    list_segment_candidates,
    load_segment,
    prepare_comparison,
)
from recorder import Recorder
from time_review import save_time_review


class HistoryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record(self, recorder, task, sample_rate=1000, seconds=2.0):
        recorder.start_recording({
            "group": "测试组",
            "subject_id": "S01",
            "task": task,
            "sampling_rate": sample_rate,
        })
        count = int(sample_rate * seconds)
        time = np.arange(count) / sample_rate
        recorder.append(
            np.vstack([
                30 * np.sin(2 * np.pi * 40 * time),
                20 * np.sin(2 * np.pi * 70 * time),
            ]),
            100.0 + time,
        )
        return recorder.stop_recording()

    def test_discovers_multiple_sessions_without_loading_waveforms(self):
        recorder = Recorder(self.root, n_channels=2)
        first = self._record(recorder, "背屈")
        recorder.end_session()
        second = self._record(recorder, "跖屈")

        with patch("history_comparison.load_trial_txt") as loader:
            trials = discover_history(self.root)

        loader.assert_not_called()
        self.assertEqual(len(trials), 2)
        self.assertEqual({item.task for item in trials}, {"背屈", "跖屈"})
        self.assertEqual({item.n_samples for item in trials}, {2000})
        self.assertTrue(all(item.selectable for item in trials))
        self.assertEqual(
            discover_history(self.root, exclude_trial=first["path"])[0].trial_dir,
            second["path"].resolve(),
        )

    def test_packet_sidecar_does_not_change_history_or_time_review_schema(self):
        recorder = Recorder(self.root, n_channels=2)
        recorder.start_recording({
            "subject_id": "S01", "task": "sync", "sampling_rate": 1000,
            "delsys_sync": {"verification": {}},
        })
        recorder.append(
            np.ones((2, 3)), np.array([100.0, 100.001, 100.002]),
            np.array([254, 255, 0]),
        )
        trial = recorder.stop_recording()

        history_trial = discover_history(self.root)[0]
        loaded = load_segment(list_segment_candidates(history_trial)[0])
        save_time_review(trial["path"], 0, 3, 3, 1000.0)

        self.assertEqual(loaded.emg.shape, (3, 2))
        self.assertEqual(loaded.channel_names,
                         ("EXG Channel 0", "EXG Channel 1"))
        self.assertTrue(trial["packet_path"].is_file())

    def test_corrupt_trial_is_isolated_from_valid_history(self):
        recorder = Recorder(self.root, n_channels=2)
        valid = self._record(recorder, "valid")
        broken = self.root / "broken_session" / "broken_trial"
        broken.mkdir(parents=True)
        (broken / "emg_data.txt").write_text("not a Recorder file\n", encoding="utf-8")

        trials = discover_history(self.root)

        self.assertEqual(len(trials), 2)
        by_path = {item.trial_dir: item for item in trials}
        self.assertTrue(by_path[valid["path"].resolve()].selectable)
        self.assertFalse(by_path[broken.resolve()].selectable)
        self.assertTrue(by_path[broken.resolve()].error)

    def test_discovery_rejects_a_resolved_path_outside_root(self):
        outside_root = self.root.parent / f"{self.root.name}_outside"
        outside_txt = outside_root / "session" / "trial" / "emg_data.txt"
        outside_txt.parent.mkdir(parents=True)
        outside_txt.write_text("external", encoding="utf-8")
        try:
            with patch.object(Path, "glob", return_value=[outside_txt]), \
                    patch("history_comparison._read_trial_summary") as reader:
                trials = discover_history(self.root)
            self.assertEqual(trials, [])
            reader.assert_not_called()
        finally:
            outside_txt.unlink()
            outside_txt.parent.rmdir()
            outside_txt.parent.parent.rmdir()

    def test_candidates_include_full_review_and_each_action(self):
        recorder = Recorder(self.root, n_channels=2)
        trial = self._record(recorder, "动作", seconds=3.0)
        save_time_review(trial["path"], 200, 2800, 3000, 1000.0)
        save_action_segments(
            trial["path"],
            0,
            [
                {"start_sample": 300, "peak_sample": 500,
                 "end_sample_exclusive": 800},
                {"start_sample": 1000, "peak_sample": 1300,
                 "end_sample_exclusive": 1700},
            ],
            3000,
            1000.0,
        )
        history_trial = discover_history(self.root)[0]

        candidates = list_segment_candidates(history_trial)

        self.assertEqual([item.source for item in candidates], [
            "full", "review", "action", "action",
        ])
        self.assertEqual(candidates[2].peak_sample, 500)
        self.assertEqual(candidates[3].start_sample, 1000)

    def test_load_segment_is_lazy_and_rejects_out_of_bounds(self):
        recorder = Recorder(self.root, n_channels=2)
        trial = self._record(recorder, "slice")
        history_trial = discover_history(self.root)[0]
        full = list_segment_candidates(history_trial)[0]
        selected = SegmentRef(history_trial, "review", "selected", 100, 300)

        loaded = load_segment(selected)

        self.assertEqual(loaded.emg.shape, (200, 2))
        self.assertEqual(loaded.sample_indices[[0, -1]].tolist(), [100, 299])
        invalid = SegmentRef(
            history_trial, "review", "invalid", 0, history_trial.n_samples + 1
        )
        with self.assertRaisesRegex(ValueError, "越界"):
            load_segment(invalid)
        self.assertTrue(trial["txt_path"].is_file())

    def test_prepare_raw_and_peak_aligned_series_without_modifying_sources(self):
        recorder = Recorder(self.root, n_channels=2)
        trial = self._record(recorder, "peak")
        save_time_review(trial["path"], 0, 2000, 2000, 1000.0)
        save_action_segments(
            trial["path"], 0,
            [{"start_sample": 300, "peak_sample": 500,
              "end_sample_exclusive": 900}],
            2000, 1000.0,
        )
        raw_hash = hashlib.sha256(trial["txt_path"].read_bytes()).digest()
        action = list_segment_candidates(discover_history(self.root)[0])[-1]

        result = prepare_comparison([action], mode="raw", alignment="peak")

        self.assertEqual(result.mode, "raw")
        self.assertEqual(result.channel_names, ("EXG Channel 0", "EXG Channel 1"))
        self.assertAlmostEqual(result.series[0].time_sec[200], 0.0)
        self.assertEqual(result.series[0].values.shape, (600, 2))
        self.assertEqual(hashlib.sha256(trial["txt_path"].read_bytes()).digest(), raw_hash)

    def test_filtered_and_rms_use_shared_band_across_sample_rates(self):
        recorder_a = Recorder(self.root, n_channels=2)
        first = self._record(recorder_a, "fast", sample_rate=1000)
        recorder_a.end_session()
        recorder_b = Recorder(self.root, n_channels=2)
        second = self._record(recorder_b, "slow", sample_rate=250)
        trials = {item.task: item for item in discover_history(self.root)}
        refs = [
            list_segment_candidates(trials["fast"])[0],
            list_segment_candidates(trials["slow"])[0],
        ]

        filtered = prepare_comparison(refs, mode="filtered")
        rms = prepare_comparison(refs, mode="rms", rms_window_ms=80)

        self.assertEqual(filtered.common_band_hz, (20.0, 124.0))
        self.assertEqual([item.sample_rate for item in filtered.series], [1000.0, 250.0])
        self.assertEqual(filtered.series[0].values.shape, (2000, 2))
        self.assertEqual(filtered.series[1].values.shape, (500, 2))
        self.assertTrue(np.all(rms.series[0].values >= 0))
        self.assertTrue(np.all(rms.series[1].values >= 0))
        self.assertTrue(first["txt_path"].is_file())
        self.assertTrue(second["txt_path"].is_file())

    def test_same_trial_is_loaded_once_and_processed_before_slicing(self):
        recorder = Recorder(self.root, n_channels=2)
        self._record(recorder, "two_segments")
        trial = discover_history(self.root)[0]
        full = list_segment_candidates(trial)[0]
        short = SegmentRef(trial, "review", "short", 100, 350)

        from history_comparison import load_trial_txt as real_loader
        with patch("history_comparison.load_trial_txt", wraps=real_loader) as loader:
            filtered = prepare_comparison([full, short], mode="filtered")

        self.assertEqual(loader.call_count, 1)
        np.testing.assert_allclose(
            filtered.series[1].values,
            filtered.series[0].values[100:350],
            rtol=0,
            atol=0,
        )

        with patch("history_comparison.load_trial_txt", wraps=real_loader) as loader:
            rms = prepare_comparison([full, short], mode="rms")
        self.assertEqual(loader.call_count, 1)
        np.testing.assert_allclose(
            rms.series[1].values,
            rms.series[0].values[100:350],
            rtol=0,
            atol=0,
        )

    def test_peak_alignment_requires_reviewed_peak(self):
        recorder = Recorder(self.root, n_channels=2)
        self._record(recorder, "full")
        full = list_segment_candidates(discover_history(self.root)[0])[0]

        with self.assertRaisesRegex(ValueError, "没有已审核峰值"):
            prepare_comparison([full], alignment="peak")

    def test_raw_mode_does_not_require_a_filterable_sample_rate(self):
        recorder = Recorder(self.root, n_channels=2)
        self._record(recorder, "low_rate", sample_rate=30)
        full = list_segment_candidates(discover_history(self.root)[0])[0]

        result = prepare_comparison([full], mode="raw")

        self.assertIsNone(result.common_band_hz)
        self.assertEqual(result.series[0].sample_rate, 30.0)


if __name__ == "__main__":
    unittest.main()
