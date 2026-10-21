import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from delsys_controller import DelsysController, clock_event
from recorder import Recorder
from tests.test_delsys_controller import FakeBoard, FakeClient


class FakeCamera:
    def __init__(self, calls):
        self.calls = calls
        self.ready = True
        self.prepare_result = {"video_path": "camera.mp4"}
        self.start_result = {"running": True, "monotonic_ns": 123}
        self.fail_prepare = False
        self.fail_stop = False
        self.prepared_path = None
        self.prepared_metadata = None
        self.sync_start = None

    def status(self):
        self.calls.append("camera_status")
        return {"connected": True, "prepared": False, "recording": False}

    def prepare_recording(self, trial_dir, metadata=None):
        self.calls.append("camera_prepare")
        self.prepared_path = Path(trial_dir)
        self.prepared_metadata = metadata
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        return self.prepare_result

    def start_recording(self):
        self.calls.append("camera_start")
        return self.start_result

    def mark_acquisition_start(self, monotonic_ns, unix_ns):
        self.sync_start = (monotonic_ns, unix_ns)

    def stop_recording(self):
        self.calls.append("camera_stop")
        if self.fail_stop:
            raise RuntimeError("stop failed")
        return {"frame_count": 10, "video_path": "camera.mp4"}


class CameraSyncControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.camera = FakeCamera(self.calls)
        self.controller = DelsysController(camera=self.camera)
        self.controller.client = FakeClient(self.calls)
        self.board = FakeBoard(self.calls)
        self.recorder = Recorder(Path(self.tmp.name), n_channels=8)
        self.addCleanup(self.controller.finish, "test_cleanup")
        self.addCleanup(self.recorder.stop_recording)
        self.addCleanup(self.controller.stop_camera)

    def start(self):
        return self.controller.start(
            self.board,
            self.recorder,
            {"sampling_rate": 1000, "task": "camera_sync"},
            clock_event(),
        )

    def events(self):
        path = next(Path(self.tmp.name).rglob("sync_events.jsonl"))
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_success_order_and_camera_events(self):
        result = self.start()

        self.assertTrue(result["running"])
        self.assertEqual(
            self.calls[:5],
            ["status", "camera_status", "configure", "camera_prepare", "board_start"],
        )
        self.assertTrue(all(call == "board_drain" for call in self.calls[5:-2]))
        self.assertEqual(self.calls[-2:], ["camera_start", "START"])
        self.assertTrue(self.controller.camera_active)
        self.assertEqual(
            self.camera.sync_start,
            (self.controller.sync_start_monotonic_ns,
             self.controller.sync_start_unix_ns),
        )
        self.assertEqual(self.camera.prepared_path, self.recorder.current_trial_dir)
        self.assertIn("camera_sync", self.camera.prepared_metadata)
        self.assertEqual(
            set(self.camera.prepared_metadata["camera_sync"]["status"]),
            {"source", "size", "fps"},
        )
        names = [event["event"] for event in self.events()]
        for expected in (
            "camera_prepare_begin", "camera_prepare_returned",
            "camera_start_begin", "camera_start_returned", "openbci_start",
            "start_trigger",
        ):
            self.assertEqual(names.count(expected), 1)

    def test_camera_not_ready_refuses_before_trial(self):
        self.camera.ready = False

        result = self.start()

        self.assertFalse(result["running"])
        self.assertIn("摄像机未就绪", result["warning"])
        self.assertEqual(self.calls, [])
        self.assertFalse(self.recorder.is_recording)

    def test_prepare_exception_cleans_camera_and_trial_without_trigger(self):
        self.camera.fail_prepare = True

        result = self.start()

        self.assertFalse(result["running"])
        self.assertEqual(
            self.calls,
            ["status", "camera_status", "configure", "camera_prepare", "camera_stop"],
        )
        self.assertNotIn("board_start", self.calls)
        self.assertNotIn("START", self.calls)
        self.assertFalse(self.recorder.is_recording)
        names = [event["event"] for event in self.events()]
        self.assertIn("start_failed", names)
        self.assertIn("camera_cleanup_returned", names)

    def test_camera_start_failure_cleans_up_before_openbci(self):
        self.camera.start_result = False

        result = self.start()

        self.assertFalse(result["running"])
        self.assertEqual(self.calls[-3:], ["camera_start", "board_stop", "camera_stop"])
        self.assertIn("board_stop", self.calls)
        self.assertNotIn("START", self.calls)
        self.assertFalse(self.recorder.is_recording)

    def test_openbci_start_failure_stops_board_and_camera_without_trigger(self):
        self.board.starts = False

        result = self.start()

        self.assertFalse(result["running"])
        self.assertEqual(
            self.calls[-3:], ["board_start", "board_stop", "camera_stop"],
        )
        self.assertNotIn("START", self.calls)
        self.assertFalse(self.recorder.is_recording)

    def test_camera_stop_is_explicitly_after_nano_stop(self):
        self.assertTrue(self.start()["running"])
        self.controller._last_trigger_monotonic = None

        stop_result = self.controller.stop_trigger(clock_event())
        camera_result = self.controller.stop_camera()
        self.controller.stop_camera()

        self.assertFalse(stop_result["warning"])
        self.assertEqual(camera_result["frame_count"], 10)
        self.assertEqual(self.calls[-3:], ["status", "STOP", "camera_stop"])
        self.assertEqual(self.calls.count("camera_stop"), 1)
        self.assertFalse(self.controller.camera_active)
        names = [event["event"] for event in self.events()]
        self.assertLess(names.index("stop_trigger"), names.index("camera_stop_begin"))

    def test_camera_stop_failure_is_logged_and_not_retried(self):
        self.assertTrue(self.start()["running"])
        self.camera.fail_stop = True

        result = self.controller.stop_camera()
        self.controller.stop_camera()

        self.assertIn("stop failed", result["error"])
        self.assertEqual(self.calls.count("camera_stop"), 1)
        self.assertIn("摄像机停止失败", self.controller.warning)
        self.assertIn("camera_stop_failed", [event["event"] for event in self.events()])

    def test_finish_closes_camera_before_generating_report(self):
        self.assertTrue(self.start()["running"])
        trial_dir = self.recorder.current_trial_dir
        self.recorder.stop_recording()

        def generate(path, metadata, reason):
            self.assertFalse(self.controller.camera_active)
            self.assertEqual(self.calls.count("camera_stop"), 1)
            return path / "sync_report.json", {"reason": reason}

        with patch("delsys_controller.generate_sync_report", side_effect=generate):
            self.controller.finish("user_stop")

        self.assertEqual(self.controller.last_report_path,
                         trial_dir / "sync_report.json")


if __name__ == "__main__":
    unittest.main()
