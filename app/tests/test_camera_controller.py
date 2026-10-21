import csv
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from camera_controller import CameraController


class FakeCapture:
    def __init__(self):
        self.opened = True
        self.released = False
        self.index = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.released:
            return False, None
        time.sleep(0.003)
        self.index += 1
        return True, np.full((2, 3, 3), self.index % 255, dtype=np.uint8)

    def release(self):
        self.released = True
        self.opened = False


class SlowCapture(FakeCapture):
    def read(self):
        if self.released:
            return False, None
        time.sleep(0.05)
        self.index += 1
        return True, np.full((2, 3, 3), self.index % 255, dtype=np.uint8)


class FakeWriter:
    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        self.path.touch()

    def isOpened(self):
        return not self.released

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class CameraControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.captures = []
        self.writers = []

        def capture_factory(source, width, height, fps):
            capture = FakeCapture()
            capture.settings = (source, width, height, fps)
            self.captures.append(capture)
            return capture

        def writer_factory(path, fps, size):
            writer = FakeWriter(path, fps, size)
            self.writers.append(writer)
            return writer

        self.controller = CameraController(capture_factory, writer_factory)

    def tearDown(self):
        self.controller.disconnect()
        self.temp_dir.cleanup()

    @staticmethod
    def wait_until(predicate, timeout=1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return False

    def test_connect_captures_preview_in_background_and_disconnects(self):
        status = self.controller.connect(source=2, width=640, height=480, fps=25)

        self.assertTrue(status["ready"])
        self.assertEqual(self.captures[0].settings, (2, 640, 480, 25))
        self.assertTrue(self.wait_until(lambda: self.controller.latest_frame is not None))
        first = self.controller.latest_frame
        first[:] = 0
        self.assertFalse(np.array_equal(first, self.controller.latest_frame))

        self.controller.disconnect()

        self.assertFalse(self.controller.ready)
        self.assertTrue(self.captures[0].released)
        self.assertIsNone(self.controller.latest_frame)

    def test_recording_gate_writes_only_post_start_frames_and_timestamp_csv(self):
        self.controller.connect()
        self.assertTrue(self.wait_until(lambda: self.controller.latest_frame is not None))
        prepared = self.controller.prepare_recording(
            self.root / "trial", {"task": "walk"}
        )
        time.sleep(0.03)
        self.assertEqual(self.writers[0].frames, [])

        started = self.controller.start_recording()
        sync_unix_ns = 1_788_777_123_456_000_000
        self.controller.mark_acquisition_start(12_345_000_000, sync_unix_ns)
        self.assertTrue(
            self.wait_until(lambda: self.controller.status()["frame_count"] >= 4)
        )
        result = self.controller.stop_recording()

        self.assertEqual(prepared["width"], 3)
        self.assertEqual(prepared["height"], 2)
        self.assertEqual(prepared["fps"], 30.0)
        self.assertEqual(result["start_monotonic_ns"], started["monotonic_ns"])
        self.assertEqual(result["encoded_frame_count"], len(self.writers[0].frames))
        self.assertGreaterEqual(result["frame_count"], 4)
        self.assertTrue(self.writers[0].released)
        session = json.loads(result["session_path"].read_text(encoding="utf-8"))
        self.assertEqual(session["frame_count"], result["frame_count"])
        self.assertEqual(session["metadata"]["task"], "walk")
        self.assertEqual(
            session["metadata"]["start_time"],
            datetime.fromtimestamp(sync_unix_ns / 1_000_000_000).isoformat(),
        )
        self.assertEqual(session["metadata"]["sync_start_monotonic_ns"], 12_345_000_000)
        self.assertEqual(
            session["metadata"]["camera_gate_start_monotonic_ns"],
            started["monotonic_ns"],
        )
        self.assertEqual(
            session["metadata"]["video_time_origin"],
            "camera_gate_start_monotonic_ns",
        )
        self.assertAlmostEqual(
            session["metadata"]["camera_gate_to_sync_start_ms"],
            (12_345_000_000 - started["monotonic_ns"]) / 1_000_000,
        )

        with result["frames_path"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), result["frame_count"])
        self.assertEqual([int(row["frame_index"]) for row in rows], list(range(len(rows))))
        timestamps = [int(row["capture_perf_counter_ns"]) for row in rows]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertTrue(all(timestamp >= started["monotonic_ns"] for timestamp in timestamps))

    def test_stop_closes_gate_but_keeps_preview_connected(self):
        self.controller.connect()
        self.assertTrue(self.wait_until(lambda: self.controller.latest_frame is not None))
        self.controller.prepare_recording(self.root / "trial")
        self.controller.start_recording()
        self.assertTrue(
            self.wait_until(lambda: self.controller.status()["frame_count"] >= 2)
        )

        result = self.controller.stop_recording()
        written = len(self.writers[0].frames)
        time.sleep(0.03)

        self.assertEqual(len(self.writers[0].frames), written)
        self.assertEqual(result["encoded_frame_count"], written)
        self.assertTrue(self.controller.ready)
        self.assertIsNotNone(self.controller.latest_frame)
        self.assertFalse(self.controller.status()["prepared"])

    def test_video_duration_tracks_wall_time_when_camera_is_slower_than_output_fps(self):
        controller = CameraController(
            lambda source, width, height, fps: SlowCapture(),
            lambda path, fps, size: FakeWriter(path, fps, size),
        )
        self.addCleanup(controller.disconnect)
        controller.connect(fps=30)
        self.assertTrue(self.wait_until(lambda: controller.latest_frame is not None))
        controller.prepare_recording(self.root / "slow_trial")
        controller.start_recording()
        time.sleep(0.31)
        result = controller.stop_recording()

        self.assertGreaterEqual(result["frame_count"], 5)
        self.assertGreater(result["duplicated_frame_count"], 0)
        self.assertAlmostEqual(
            result["encoded_duration_ms"], result["wall_duration_ms"], delta=40
        )

    def test_prepare_requires_connection_and_never_uses_serial(self):
        with self.assertRaises(RuntimeError):
            self.controller.prepare_recording(self.root / "trial")
        with self.assertRaises(RuntimeError):
            self.controller.start_recording()

        source = Path(__file__).parents[1] / "camera_controller.py"
        self.assertNotIn("import serial", source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
