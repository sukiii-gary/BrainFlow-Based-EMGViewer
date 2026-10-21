import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from recorder import Recorder
from config_viewer import ViewerConfig


class RecorderSessionTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.recorder = Recorder(
            data_dir=str(self.root), flush_sec=60.0, n_channels=2
        )

    def tearDown(self):
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        self.temp_dir.cleanup()

    def _record_trial(self, task="背屈"):
        self.assertTrue(self.recorder.start_recording({
            "group": "测试组",
            "subject_id": "S01",
            "task": task,
            "sampling_rate": 1000,
            "start_time": "2026-09-04T10:00:00",
        }))
        self.recorder.append(
            np.array([[1.25, 2.5], [-3.75, 4.0]], dtype=np.float32),
            np.array([100.0, 100.001]),
        )
        return self.recorder.stop_recording()

    def test_one_session_contains_multiple_txt_only_trials(self):
        session = self.recorder.start_session()
        first = self._record_trial("背屈")
        second = self._record_trial("跖屈")

        self.assertEqual(first["path"].parent, session)
        self.assertEqual(second["path"].parent, session)
        self.assertEqual(len(self.recorder.list_trials()), 2)
        self.assertEqual(
            sorted(path.name for path in first["path"].iterdir()),
            ["emg_data.txt"],
        )
        self.assertFalse(list(session.rglob("*.h5")))
        self.assertFalse(list(session.rglob("*.json")))

        lines = first["txt_path"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 9)
        self.assertIn("Task = 背屈", lines[2])
        self.assertEqual(
            lines[6],
            "Sample Index,EXG Channel 0,EXG Channel 1,Timestamp",
        )
        self.assertTrue(lines[7].startswith("0,1.25000000,-3.75000000,"))
        self.assertTrue(lines[8].startswith("1,2.50000000,4.00000000,"))

    def test_disconnect_then_reconnect_creates_a_new_session(self):
        first_session = self.recorder.start_session("S01")
        self.recorder.end_session()

        second_session = self.recorder.start_session("S01")

        self.assertNotEqual(first_session, second_session)
        self.assertRegex(first_session.name, r"^\d{8}_S01_Session1$")
        self.assertRegex(second_session.name, r"^\d{8}_S01_Session2$")
        self.assertTrue(first_session.is_dir())
        self.assertTrue(second_session.is_dir())

    def test_externally_deleted_session_is_recreated_without_restart(self):
        first_session = self.recorder.start_session("S01")
        first_session.rmdir()

        self.assertFalse(self.recorder.is_session_active)
        replacement = self.recorder.start_session("S01")

        self.assertTrue(replacement.is_dir())
        self.assertTrue(self.recorder.is_session_active)
        self.assertEqual(self.recorder.list_trials(), [])

    def test_start_recording_self_heals_missing_session_parent(self):
        session = self.recorder.start_session("S01")
        session.rmdir()

        self.assertTrue(self.recorder.start_recording({
            "subject_id": "S01", "task": "after_delete", "sampling_rate": 1000,
        }))
        trial = self.recorder.stop_recording()

        self.assertTrue(trial["path"].is_dir())
        self.assertTrue(trial["txt_path"].is_file())

    def test_delete_current_session_clears_state(self):
        session = self.recorder.start_session("S01")
        trial = self._record_trial("discard")

        deleted = self.recorder.delete_current_session()

        self.assertEqual(deleted, session.resolve())
        self.assertFalse(session.exists())
        self.assertFalse(self.recorder.is_session_active)
        self.assertIsNone(self.recorder.session_dir)
        self.assertEqual(self.recorder.list_trials(), [])
        self.assertFalse(trial["txt_path"].exists())

    def test_session_uses_largest_existing_number_and_sanitized_subject(self):
        (self.root / "20260904_S_SZ_Session1").mkdir()
        (self.root / "20260904_S_SZ_Session3").mkdir()
        with patch("recorder.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime(2026, 9, 4, 9, 30)
            session = self.recorder.start_session(" S:SZ ")

        self.assertEqual(session.name, "20260904_S_SZ_Session4")

    def test_repeated_task_names_never_overwrite_trial(self):
        self.recorder.start_session("S01")
        first = self._record_trial("sit_base")
        second = self._record_trial("sit_base")
        third = self._record_trial("sit:base")

        self.assertEqual(first["path"].name, "sit_base")
        self.assertEqual(second["path"].name, "sit_base_2")
        self.assertEqual(third["path"].name, "sit_base_3")
        self.assertTrue(all(item["txt_path"].is_file() for item in (first, second, third)))

    def test_completed_trial_can_be_deleted_but_external_path_cannot(self):
        self.recorder.start_session()
        trial = self._record_trial()

        self.assertTrue(self.recorder.delete_trial(trial["path"]))
        self.assertFalse(trial["path"].exists())
        self.assertEqual(self.recorder.list_trials(), [])

        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaises(ValueError):
            self.recorder.delete_trial(outside)
        self.assertTrue(outside.exists())

    def test_append_does_not_wait_for_slow_txt_writer(self):
        entered = threading.Event()
        release = threading.Event()
        original_savetxt = np.savetxt

        def slow_savetxt(*args, **kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return original_savetxt(*args, **kwargs)

        with patch("recorder.np.savetxt", side_effect=slow_savetxt):
            self.recorder.start_recording({"sampling_rate": 1000})
            started = time.perf_counter()
            self.recorder.append(np.ones((2, 1000)), np.arange(1000) / 1000)
            append_elapsed = time.perf_counter() - started
            self.assertLess(append_elapsed, 0.05)
            self.assertTrue(entered.wait(timeout=1.0))
            release.set()
            trial = self.recorder.stop_recording()

        self.assertEqual(trial["samples"], 1000)
        self.assertIsNone(trial["write_error"])

    def test_cyton_defaults_record_all_eight_channels(self):
        recorder = Recorder(data_dir=self.root / "eight-channel")
        self.assertEqual(ViewerConfig().n_channels, 8)
        self.assertEqual(recorder.n_channels, 8)
        recorder.start_recording({"sampling_rate": 1000})
        recorder.append(np.arange(16, dtype=float).reshape(8, 2))

        trial = recorder.stop_recording()

        header = trial["txt_path"].read_text(encoding="utf-8").splitlines()[6]
        self.assertIn("EXG Channel 7", header)
        self.assertEqual(len(header.split(",")), 10)

    def test_mark_acquisition_start_rewrites_txt_header(self):
        self.recorder.start_recording({"sampling_rate": 1000})
        unix_ns = 1_788_777_123_456_000_000

        self.recorder.mark_acquisition_start(12_345_000_000, unix_ns)
        trial = self.recorder.stop_recording()

        expected = datetime.fromtimestamp(unix_ns / 1_000_000_000).isoformat()
        header = trial["txt_path"].read_text(encoding="utf-8").splitlines()[2]
        self.assertIn(f"Start = {expected}", header)

    def test_synchronized_duration_keeps_separate_time_bases(self):
        self.recorder.start_recording({"sampling_rate": 1000})
        self.recorder.mark_acquisition_start(10_000_000_000, 20_000_000_000)
        self.recorder.append(
            np.ones((2, 3)), np.array([20.0, 20.001, 20.002])
        )
        self.recorder.mark_acquisition_stop(11_500_000_000, 21_500_000_000)

        trial = self.recorder.stop_recording()

        self.assertEqual(trial["duration_sec"], 1.5)
        self.assertEqual(trial["trigger_duration_sec"], 1.5)
        self.assertEqual(trial["sample_count_duration_sec"], 0.003)
        self.assertAlmostEqual(trial["timestamp_span_sec"], 0.002)
        self.assertEqual(trial["first_sample_timestamp_unix_s"], 20.0)
        self.assertEqual(trial["last_sample_timestamp_unix_s"], 20.002)
        self.assertGreaterEqual(trial["save_completion_latency_sec"], 0)

    def test_synchronized_trial_writes_aligned_packet_sidecar(self):
        self.recorder.start_recording({
            "sampling_rate": 1000,
            "delsys_sync": {"verification": {}},
        })
        self.recorder.append(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([10.0, 10.001, 10.002]),
            np.array([254, 255, 0]),
        )

        trial = self.recorder.stop_recording()

        self.assertTrue(trial["packet_path"].is_file())
        packet_lines = trial["packet_path"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(packet_lines[0], "Sample Index,Package Counter,Timestamp")
        self.assertEqual([line.split(",")[1] for line in packet_lines[1:]],
                         ["254", "255", "0"])
        emg_header = trial["txt_path"].read_text(encoding="utf-8").splitlines()[6]
        self.assertEqual(
            emg_header,
            "Sample Index,EXG Channel 0,EXG Channel 1,Timestamp",
        )

        emg_sample_lines = trial["txt_path"].read_text(
            encoding="utf-8"
        ).splitlines()[7:]
        self.assertEqual(len(packet_lines) - 1, len(emg_sample_lines))
        self.assertEqual(len(emg_sample_lines), trial["samples"])

    def test_synchronized_trial_marks_missing_counters_without_dropping_samples(self):
        self.recorder.start_recording({
            "sampling_rate": 1000,
            "delsys_sync": {"verification": {}},
        })
        self.recorder.append(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([10.0, 10.001]),
        )

        trial = self.recorder.stop_recording()

        packet_lines = trial["packet_path"].read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual([line.split(",")[1] for line in packet_lines[1:]],
                         ["-1", "-1"])
        self.assertEqual(trial["samples"], 2)

    def test_writer_failure_closes_main_and_packet_files(self):
        self.recorder.start_recording({
            "sampling_rate": 1000,
            "delsys_sync": {"verification": {}},
        })
        txt_file = self.recorder._txt_file
        packet_file = self.recorder._packet_file

        with patch("recorder.np.savetxt", side_effect=OSError("disk full")):
            self.recorder.append(
                np.ones((2, 2)), np.array([10.0, 10.001]), np.array([1, 2])
            )
            trial = self.recorder.stop_recording()

        self.assertIn("disk full", trial["write_error"])
        self.assertTrue(txt_file.closed)
        self.assertTrue(packet_file.closed)

    def test_close_failures_are_reported_and_do_not_poison_next_trial(self):
        class CloseFailingProxy:
            def __init__(self, handle, label):
                self.handle = handle
                self.label = label
                self.close_called = False

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def close(self):
                self.close_called = True
                self.handle.close()
                raise OSError(f"{self.label} close")

        self.recorder.start_recording({
            "sampling_rate": 1000,
            "delsys_sync": {"verification": {}},
        })
        txt_proxy = CloseFailingProxy(self.recorder._txt_file, "txt")
        packet_proxy = CloseFailingProxy(self.recorder._packet_file, "packet")
        self.recorder._txt_file = txt_proxy
        self.recorder._packet_file = packet_proxy
        self.recorder.append(
            np.ones((2, 1)), np.array([10.0]), np.array([1])
        )

        trial = self.recorder.stop_recording()

        self.assertTrue(txt_proxy.close_called)
        self.assertTrue(packet_proxy.close_called)
        self.assertIn("emg_data.txt close failed", trial["write_error"])
        self.assertIn("openbci_packets.csv close failed", trial["write_error"])
        self.assertTrue(self.recorder.start_recording({"sampling_rate": 1000}))
        self.recorder.stop_recording()


if __name__ == "__main__":
    unittest.main()
