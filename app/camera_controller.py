"""Background UVC camera capture and recording.

This module owns no trigger or serial-port logic. Synchronization is performed
by the caller using the monotonic timestamps returned here.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


CaptureFactory = Callable[[int | str, int, int, float], object]
WriterFactory = Callable[[Path, float, tuple[int, int]], object]


def _default_capture_factory(source, width, height, fps):
    import cv2

    capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def _default_writer_factory(path, fps, size):
    import cv2

    return cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )


class CameraController:
    """Capture previews continuously and gate video recording independently."""

    def __init__(
        self,
        capture_factory: CaptureFactory | None = None,
        writer_factory: WriterFactory | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        self._capture_factory = capture_factory or _default_capture_factory
        self._writer_factory = writer_factory or _default_writer_factory
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._first_frame = threading.Event()
        self._stop_capture = threading.Event()
        self._capture = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._latest_frame = None
        self._source: int | str | None = None
        self._requested_size: tuple[int, int] | None = None
        self._fps = 0.0
        self._writer = None
        self._frame_file = None
        self._frame_writer = None
        self._video_path: Path | None = None
        self._frames_path: Path | None = None
        self._session_path: Path | None = None
        self._writer_size: tuple[int, int] | None = None
        self._prepared = False
        self._recording = False
        self._frame_count = 0
        self._encoded_frame_count = 0
        self._duplicated_frame_count = 0
        self._dropped_capture_frame_count = 0
        self._pending_frame = None
        self._start_ns: int | None = None
        self._error = ""
        self._metadata: dict = {}

    @property
    def ready(self) -> bool:
        with self._lock:
            return bool(
                self._connected
                and self._thread is not None
                and self._thread.is_alive()
            )

    @property
    def latest_frame(self):
        """Return a copy of the newest frame, or ``None`` before the first frame."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status(self) -> dict:
        with self._lock:
            size = None
            if self._latest_frame is not None:
                height, width = self._latest_frame.shape[:2]
                size = (width, height)
            return {
                "connected": self._connected,
                "ready": self.ready,
                "prepared": self._prepared,
                "recording": self._recording,
                "source": self._source,
                "size": size or self._requested_size,
                "fps": self._fps,
                "frame_count": self._frame_count,
                "encoded_frame_count": self._encoded_frame_count,
                "error": self._error,
            }

    def connect(
        self,
        source: int | str = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ) -> dict:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")
        self.disconnect()
        capture = self._capture_factory(source, width, height, fps)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise RuntimeError("Cannot open camera")
        with self._lock:
            self._capture = capture
            self._source = source
            self._requested_size = (width, height)
            self._fps = float(fps)
            self._latest_frame = None
            self._error = ""
            self._connected = True
            self._first_frame.clear()
            self._stop_capture.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(capture,),
                name="uvc-camera-capture",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def disconnect(self) -> None:
        if self._prepared:
            self.stop_recording()
        with self._lock:
            capture = self._capture
            thread = self._thread
            self._connected = False
            self._stop_capture.set()
        if capture is not None:
            capture.release()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            if thread is not None and thread.is_alive() and not self._error:
                self._error = "Camera capture thread did not stop"
            self._capture = None
            self._thread = None
            self._latest_frame = None
            self._first_frame.clear()

    def prepare_recording(
        self,
        trial_dir: str | Path,
        metadata: dict | None = None,
    ) -> dict:
        if not self.ready:
            raise RuntimeError("Connect the camera before preparing a recording")
        if self._prepared:
            raise RuntimeError("Camera recording is already prepared")
        if not self._first_frame.wait(timeout=2.0):
            raise RuntimeError("Camera did not provide a preview frame")

        directory = Path(trial_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        video_path = directory / "camera.mp4"
        frames_path = directory / "camera_frames.csv"
        session_path = directory / "camera_session.json"
        if video_path.exists() or frames_path.exists() or session_path.exists():
            raise FileExistsError("Camera recording output already exists")

        frame = self.latest_frame
        if frame is None:
            raise RuntimeError("Camera preview frame is unavailable")
        height, width = frame.shape[:2]
        writer = self._writer_factory(video_path, self._fps, (width, height))
        if writer is None or not writer.isOpened():
            if writer is not None:
                writer.release()
            raise RuntimeError("Cannot create camera video file")
        try:
            frame_file = frames_path.open("x", newline="", encoding="utf-8")
        except Exception:
            writer.release()
            raise
        frame_writer = csv.writer(frame_file, lineterminator="\n")
        frame_writer.writerow(
            ["frame_index", "capture_perf_counter_ns", "from_recording_start_ms"]
        )
        frame_file.flush()

        with self._lock:
            self._writer = writer
            self._frame_file = frame_file
            self._frame_writer = frame_writer
            self._video_path = video_path
            self._frames_path = frames_path
            self._session_path = session_path
            self._writer_size = (width, height)
            self._metadata = dict(metadata or {})
            self._frame_count = 0
            self._encoded_frame_count = 0
            self._duplicated_frame_count = 0
            self._dropped_capture_frame_count = 0
            self._pending_frame = None
            self._start_ns = None
            self._error = ""
            self._prepared = True
            self._recording = False
        return {
            "video_path": video_path,
            "frames_path": frames_path,
            "session_path": session_path,
            "width": width,
            "height": height,
            "fps": self._fps,
        }

    def start_recording(self) -> dict:
        with self._lock:
            if not self._prepared:
                raise RuntimeError("Prepare the camera recording before starting")
            if self._recording:
                return {"monotonic_ns": self._start_ns, "already_recording": True}
            self._start_ns = self._clock_ns()
            self._metadata["camera_gate_start_monotonic_ns"] = self._start_ns
            self._metadata["video_time_origin"] = "camera_gate_start_monotonic_ns"
            self._encoded_frame_count = 0
            self._duplicated_frame_count = 0
            self._dropped_capture_frame_count = 0
            self._pending_frame = None
            self._recording = True
            return {"monotonic_ns": self._start_ns, "already_recording": False}

    def mark_acquisition_start(self, monotonic_ns: int, unix_ns: int) -> None:
        """Store the shared START epoch without changing the camera gate time."""
        with self._lock:
            if not self._prepared:
                raise RuntimeError("Camera recording is not prepared")
            self._metadata["start_time"] = datetime.fromtimestamp(
                unix_ns / 1_000_000_000
            ).isoformat()
            self._metadata["sync_start_monotonic_ns"] = monotonic_ns
            self._metadata["sync_start_unix_ns"] = unix_ns
            self._metadata["camera_gate_to_sync_start_ms"] = (
                (monotonic_ns - self._start_ns) / 1_000_000
                if self._start_ns is not None else None
            )

    def stop_recording(self) -> dict | None:
        with self._lock:
            if not self._prepared:
                return None
            stop_ns = self._clock_ns()
            self._recording = False
            if (
                not self._error and self._pending_frame is not None
                and self._start_ns is not None
            ):
                target_count = self._target_frame_count(stop_ns)
                try:
                    self._write_pending_frame(
                        target_count - self._encoded_frame_count
                    )
                except Exception as exc:
                    self._error = f"Camera recording failed: {exc}"
            writer = self._writer
            frame_file = self._frame_file
            result = {
                "video_path": self._video_path,
                "frames_path": self._frames_path,
                "session_path": self._session_path,
                "frame_count": self._frame_count,
                "encoded_frame_count": self._encoded_frame_count,
                "duplicated_frame_count": self._duplicated_frame_count,
                "dropped_capture_frame_count": self._dropped_capture_frame_count,
                "start_monotonic_ns": self._start_ns,
                "stop_monotonic_ns": stop_ns,
                "error": self._error,
                "source": self._source,
                "width": self._writer_size[0] if self._writer_size else None,
                "height": self._writer_size[1] if self._writer_size else None,
                "fps": self._fps,
                "wall_duration_ms": (
                    (stop_ns - self._start_ns) / 1_000_000
                    if self._start_ns is not None else None
                ),
                "encoded_duration_ms": (
                    self._encoded_frame_count / self._fps * 1000
                    if self._fps > 0 else None
                ),
                "metadata": self._metadata,
            }
            self._writer = None
            self._frame_file = None
            self._frame_writer = None
            self._writer_size = None
            self._pending_frame = None
            self._prepared = False
            self._start_ns = None
        try:
            if writer is not None:
                writer.release()
        finally:
            if frame_file is not None:
                frame_file.flush()
                frame_file.close()
        session_path = result["session_path"]
        if session_path is not None:
            session_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return result

    def _target_frame_count(self, timestamp_ns: int) -> int:
        elapsed_s = max(0, timestamp_ns - self._start_ns) / 1_000_000_000
        return max(1, int(elapsed_s * self._fps + 0.5))

    def _write_pending_frame(self, copies: int) -> None:
        if copies <= 0:
            self._dropped_capture_frame_count += 1
            return
        for _ in range(copies):
            self._writer.write(self._pending_frame)
        self._encoded_frame_count += copies
        self._duplicated_frame_count += max(0, copies - 1)

    def _capture_loop(self, capture) -> None:
        while not self._stop_capture.is_set():
            ok, frame = capture.read()
            captured_ns = self._clock_ns()
            if not ok or frame is None:
                self._stop_capture.wait(0.01)
                continue
            with self._lock:
                self._latest_frame = frame.copy()
                self._first_frame.set()
                if not self._recording:
                    continue
                if self._start_ns is None or captured_ns < self._start_ns:
                    continue
                height, width = frame.shape[:2]
                if (width, height) != self._writer_size:
                    self._error = (
                        f"Camera frame size changed to {width}x{height}; "
                        f"expected {self._writer_size[0]}x{self._writer_size[1]}"
                    )
                    self._recording = False
                    continue
                try:
                    self._frame_writer.writerow(
                        [
                            self._frame_count,
                            captured_ns,
                            f"{(captured_ns - self._start_ns) / 1_000_000:.3f}",
                        ]
                    )
                    self._frame_count += 1
                    if self._pending_frame is not None:
                        target_count = self._target_frame_count(captured_ns)
                        self._write_pending_frame(
                            target_count - self._encoded_frame_count
                        )
                    self._pending_frame = frame.copy()
                    if self._frame_count % 30 == 0:
                        self._frame_file.flush()
                except Exception as exc:
                    self._error = f"Camera recording failed: {exc}"
                    self._recording = False
