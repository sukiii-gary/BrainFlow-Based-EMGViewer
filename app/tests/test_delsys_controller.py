import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from delsys_controller import DelsysController, _drain_openbci_startup, clock_event
from recorder import Recorder


class FakeClient:
    def __init__(self, calls):
        self.calls = calls
        self.ready = True
        self.fail_action = None
        self.states = ["IDLE"]

    def status(self):
        self.calls.append("status")
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"response": "STATUS " + state + " 0"}

    def trigger(self, action):
        self.calls.append(action)
        if action == self.fail_action:
            raise TimeoutError("lost ACK")
        now = clock_event()
        return {
            "action": action,
            "status": "firmware_confirmed",
            "host_write_begin_monotonic_ns": now["monotonic_ns"],
            "host_write_end_monotonic_ns": now["monotonic_ns"] + 1000,
            "host_reply_monotonic_ns": now["monotonic_ns"] + 3000,
            "host_write_begin_unix_ns": now["unix_ns"],
        }

    def close(self):
        self.ready = False
        self.calls.append("close")


class FakeBoard:
    def __init__(self, calls):
        self.calls = calls
        self.state = "verified"
        self.starts = True
        self.pretrigger_samples = 10
        self.drain_batches = None

    def apply_and_verify_settings(self):
        self.calls.append("configure")
        return {"state": self.state}

    def start_stream(self, settings_prepared=False):
        assert settings_prepared
        self.calls.append("board_start")
        return self.starts

    def stop_stream(self):
        self.calls.append("board_stop")

    def get_new_data(self, n_samples=None):
        self.calls.append("board_drain")
        if self.drain_batches is not None:
            if self.drain_batches:
                return self.drain_batches.pop(0)
        return np.ones((8, self.pretrigger_samples))


class DelsysControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.controller = DelsysController()
        self.controller.client = FakeClient(self.calls)
        self.board = FakeBoard(self.calls)
        self.recorder = Recorder(Path(self.tmp.name), n_channels=8)
        self.addCleanup(self.controller.finish, "test_cleanup")
        self.addCleanup(self.recorder.stop_recording)

    def start(self):
        return self.controller.start(self.board, self.recorder,
                                     {"sampling_rate": 1000, "task": "sync"}, clock_event())

    def events(self):
        path = next(Path(self.tmp.name).rglob("sync_events.jsonl"))
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_success_order_sidecar_and_original_txt(self):
        self.assertTrue(self.start()["running"])
        self.assertEqual(self.calls[:3], ["status", "configure", "board_start"])
        self.assertEqual(self.calls[-1], "START")
        self.assertTrue(all(call == "board_drain" for call in self.calls[3:-1]))
        self.assertTrue(self.controller.active)
        self.assertIsNotNone(self.controller.sync_start_monotonic_ns)
        self.assertIsNotNone(self.controller.sync_start_unix_ns)
        expected_start = self.recorder.current_trial_info["start_time"]
        header = next(Path(self.tmp.name).rglob("emg_data.txt")).read_text(
            encoding="utf-8"
        ).splitlines()[2]
        self.assertIn(f"Start = {expected_start}", header)
        self.controller.note_first_data(np.ones((8, 2)), np.array([1., 2.]))
        self.controller.note_first_data(np.ones((8, 2)), np.array([3., 4.]))
        stop_result = self.controller.stop_trigger(clock_event())
        self.assertFalse(stop_result["warning"])
        self.assertIsNotNone(stop_result["sync_stop_monotonic_ns"])
        self.assertIsNotNone(stop_result["sync_stop_unix_ns"])
        self.assertTrue(self.controller.active)
        self.assertEqual(self.calls[-2:], ["status", "STOP"])
        self.controller.stop_trigger(clock_event())
        self.assertEqual(self.calls.count("STOP"), 1)
        self.controller.finish("user_stop")
        self.assertFalse(self.controller.active)
        self.recorder.stop_recording()
        events = self.events()
        discarded = next(e for e in events if e["event"] == "openbci_pretrigger_discarded")
        self.assertGreater(discarded["sample_count"], 10)
        self.assertEqual(discarded["first_batch_sample_count"], 10)
        self.assertEqual(sum(e["event"] == "first_openbci_data_received" for e in events), 1)
        self.assertIn("click", events[0])
        txt = next(Path(self.tmp.name).rglob("emg_data.txt")).read_text()
        self.assertIn("EXG Channel 0", txt)
        self.assertNotIn("start_trigger", txt)

    def test_waits_for_first_nonempty_startup_batch_before_trigger(self):
        self.board.drain_batches = [
            np.zeros((8, 0)), np.zeros((8, 0)), np.ones((8, 7))
        ]

        self.assertTrue(self.start()["running"])

        self.assertGreater(self.calls.count("board_drain"), 3)
        self.assertLess(self.calls.index("board_drain"), self.calls.index("START"))
        discarded = next(
            event for event in self.events()
            if event["event"] == "openbci_pretrigger_discarded"
        )
        self.assertGreater(discarded["sample_count"], 7)
        self.assertEqual(discarded["first_batch_sample_count"], 7)
        self.assertGreater(discarded["poll_count"], 3)

    def test_startup_data_timeout_fails_without_camera_or_trigger(self):
        self.board.drain_batches = [np.zeros((8, 0))]
        self.board.pretrigger_samples = 0
        with patch("delsys_controller.OPENBCI_STARTUP_DATA_TIMEOUT_SEC", 0):
            result = self.start()

        self.assertFalse(result["running"])
        self.assertNotIn("START", self.calls)
        self.assertIn("board_stop", self.calls)

    def test_startup_data_read_failure_fails_without_trigger(self):
        self.board.drain_batches = [None]

        result = self.start()

        self.assertFalse(result["running"])
        self.assertNotIn("START", self.calls)
        self.assertIn("board_stop", self.calls)

    def test_recording_open_failure_never_triggers(self):
        with patch.object(self.recorder, "start_recording", return_value=False):
            self.assertFalse(self.start()["running"])
        self.assertEqual(self.calls, ["status", "configure"])

    def test_config_failure_never_opens_file_or_triggers(self):
        self.board.state = "failed"
        self.assertFalse(self.start()["running"])
        self.assertFalse(self.recorder.is_recording)
        self.assertNotIn("START", self.calls)

    def test_board_start_failure_keeps_error_sidecar(self):
        self.board.starts = False
        self.assertFalse(self.start()["running"])
        self.assertFalse(self.recorder.is_recording)
        self.assertNotIn("START", self.calls)
        self.assertIn("start_failed", [e["event"] for e in self.events()])
        self.assertIn("board_stop", self.calls)

    def test_start_timeout_keeps_recording_and_prevents_retry(self):
        self.controller.client.fail_action = "START"
        result = self.start()
        self.assertTrue(result["running"])
        self.assertTrue(result["warning"])
        self.assertTrue(self.recorder.is_recording)
        self.assertTrue(self.controller.active)
        self.assertFalse(self.controller.ready)
        self.assertFalse(self.start()["running"])
        self.assertTrue(self.controller.stop_trigger(clock_event())["warning"])
        self.assertEqual(self.calls.count("START"), 1)
        self.assertNotIn("STOP", self.calls)

    def test_alignment_failure_after_start_warns_without_retriggering(self):
        with patch.object(
            self.recorder, "mark_acquisition_start", side_effect=OSError("disk full")
        ):
            result = self.start()

        self.assertTrue(result["running"])
        self.assertIn("同步时间基准写入失败", result["warning"])
        self.assertTrue(self.recorder.is_recording)
        self.assertTrue(self.controller.active)
        self.assertEqual(self.calls.count("START"), 1)
        self.assertIn(
            "sync_epoch_alignment_failed",
            [event["event"] for event in self.events()],
        )

    def test_stop_timeout_reports_unknown_and_does_not_retry(self):
        self.start()
        self.controller.client.fail_action = "STOP"
        self.assertTrue(self.controller.stop_trigger(clock_event())["warning"])
        self.assertTrue(self.controller.active)
        self.controller.stop_trigger(clock_event())
        self.assertEqual(self.calls.count("STOP"), 1)
        self.assertTrue(self.controller.finish("user_stop"))

    def test_immediate_stop_waits_for_start_pulse(self):
        self.start()
        self.controller.client.states = ["PULSE", "IDLE"]
        self.assertFalse(self.controller.stop_trigger(clock_event())["warning"])
        self.assertEqual(self.calls[-3:], ["status", "status", "STOP"])

    def test_abort_and_disconnect_never_send_stop(self):
        self.start()
        self.assertTrue(self.controller.finish("disconnect"))
        self.controller.disconnect()
        self.assertNotIn("STOP", self.calls)

    def test_log_open_failure_does_not_start_devices(self):
        original = Path.open
        def fail_sidecar(path, *args, **kwargs):
            if path.name == "sync_events.jsonl":
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", fail_sidecar):
            self.assertFalse(self.start()["running"])
        self.assertFalse(self.recorder.is_recording)
        self.assertNotIn("board_start", self.calls)

    def test_log_write_failure_keeps_acquisition_and_reports_warning(self):
        self.start()
        self.controller._log.close()
        result = self.controller.stop_trigger(clock_event())
        self.assertTrue(result["warning"])
        self.assertTrue(self.recorder.is_recording)
        self.assertIn("STOP", self.calls)

    def test_finish_generates_sync_report_after_log_is_closed(self):
        self.start()
        trial_dir = self.recorder.current_trial_dir
        self.recorder.stop_recording()
        report_path = trial_dir / "sync_report.json"

        def generate(path, metadata, reason):
            self.assertIsNone(self.controller._log)
            self.assertEqual(path, trial_dir)
            self.assertEqual(metadata["task"], "sync")
            self.assertEqual(reason, "user_stop")
            return report_path, {"status": "estimated"}

        with patch("delsys_controller.generate_sync_report", side_effect=generate):
            self.controller.finish("user_stop")

        self.assertEqual(self.controller.last_report_path, report_path)
        self.assertEqual(self.controller.last_report, {"status": "estimated"})

    def test_sync_report_failure_is_warning_and_is_not_retried(self):
        self.start()
        self.recorder.stop_recording()

        with patch("delsys_controller.generate_sync_report",
                   side_effect=OSError("disk full")) as generate:
            warning = self.controller.finish("user_stop")
            self.controller.finish("test_cleanup")

        self.assertIn("同步报告 JSON 生成失败", warning)
        self.assertEqual(generate.call_count, 1)
        self.assertIsNone(self.controller.last_report_path)


class StartupDrainTests(unittest.TestCase):
    class Clock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    class ScheduledBoard:
        def __init__(self, clock, batch_size):
            self.clock = clock
            self.batch_size = batch_size
            self.calls = 0

        def get_new_data(self, n_samples=None):
            self.calls += 1
            count = self.batch_size(self.clock.now, self.calls)
            return np.ones((8, count), dtype=np.float32)

    def run_drain(self, board, clock):
        with patch("delsys_controller.time.monotonic", side_effect=clock.monotonic), \
             patch("delsys_controller.time.sleep", side_effect=clock.sleep):
            return _drain_openbci_startup(board, 1000)

    def test_delayed_second_backlog_must_be_cleared_before_stable(self):
        clock = self.Clock()

        def batches(now, call):
            if call == 1:
                return 1000
            if 0.055 <= now <= 0.065:
                return 700
            return 10

        result = self.run_drain(self.ScheduledBoard(clock, batches), clock)

        self.assertGreaterEqual(result["unstable_window_count"], 1)
        self.assertEqual(result["largest_batch_samples"], 1000)
        self.assertGreaterEqual(result["wait_duration_ms"], 600)
        self.assertEqual(result["completed_stable_windows"], 2)

    def test_continuously_oversized_batches_abort(self):
        clock = self.Clock()
        board = self.ScheduledBoard(clock, lambda now, call: 100)

        with self.assertRaisesRegex(Exception, "未稳定"):
            self.run_drain(board, clock)


if __name__ == "__main__":
    unittest.main()
