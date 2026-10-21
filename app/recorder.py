"""Session/Trial TXT recorder for the EMG acquisition GUI."""
from __future__ import annotations

import csv
import queue
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class Recorder:
    """Record one connection as a Session and each acquisition as a Trial.

    Raw samples are written directly to ``emg_data.txt``. No HDF5 or JSON
    intermediate files are created.
    """

    def __init__(self, data_dir: str | Path | None = None,
                 flush_sec: float = 5.0,
                 n_channels: int = 8):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "recordings"
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.flush_sec = flush_sec
        self.n_channels = n_channels

        self._recording = False
        self._session_active = False
        self._session_dir: Path | None = None
        self._session_started_at: datetime | None = None
        self._trial_dir: Path | None = None
        self._txt_path: Path | None = None
        self._packet_path: Path | None = None
        self._txt_file = None
        self._packet_file = None
        self._writer = None
        self._packet_writer = None
        self._write_queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._write_error: str | None = None
        self._metadata: dict = {}
        self._sample_count = 0
        self._written_count = 0
        self._trial_count = 0
        self._trial_started_monotonic = 0.0
        self._trial_stopped_monotonic: float | None = None
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._trials: list[dict] = []

    @staticmethod
    def _unique_directory(parent: Path, name: str) -> Path:
        candidate = parent / name
        suffix = 2
        while candidate.exists():
            candidate = parent / f"{name}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _header_value(value) -> str:
        return re.sub(r"[\r\n]+", " ", str(value or "")).strip()

    @staticmethod
    def _path_name(value, fallback: str) -> str:
        """Make user-entered metadata safe as one Windows/macOS path component."""
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or ""))
        name = re.sub(r"\s+", "_", name).strip(" ._")
        return name or fallback

    def start_session(self, subject_id: str = "") -> Path:
        """Create the directory representing the current board connection."""
        if (
            self._session_active
            and self._session_dir is not None
            and self._session_dir.is_dir()
        ):
            return self._session_dir
        if self._session_active:
            self._clear_session_state()
        now = datetime.now()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        subject = self._path_name(subject_id, "Unknown")
        prefix = f"{now:%Y%m%d}_{subject}_Session"
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
        existing_numbers = [
            int(match.group(1))
            for path in self.data_dir.iterdir()
            if path.is_dir() and (match := pattern.fullmatch(path.name))
        ]
        session_number = max(existing_numbers, default=0) + 1
        self._session_dir = self.data_dir / f"{prefix}{session_number}"
        while self._session_dir.exists():
            session_number += 1
            self._session_dir = self.data_dir / f"{prefix}{session_number}"
        self._session_dir.mkdir(parents=False)
        self._session_started_at = now
        self._session_active = True
        self._trial_count = 0
        self._trials = []
        print(f"[Recorder] 新 Session -> {self._session_dir}")
        return self._session_dir

    def end_session(self) -> None:
        """Close the current Session after the board disconnects."""
        if self._recording:
            self.stop_recording()
        self._session_active = False

    def _clear_session_state(self) -> None:
        self._session_active = False
        self._session_dir = None
        self._session_started_at = None
        self._trial_count = 0
        self._trials = []

    def delete_current_session(self) -> Path | None:
        """Permanently delete the idle current Session and clear cached state."""
        if self._recording:
            raise RuntimeError("不能删除正在录制的 Session")
        if self._session_dir is None:
            return None
        target = self._session_dir.resolve()
        root = self.data_dir.resolve()
        if target.parent != root:
            raise ValueError("只能删除保存根目录中的当前 Session")
        if target.exists():
            if not target.is_dir():
                raise ValueError("当前 Session 路径不是文件夹")
            shutil.rmtree(target)
        self._clear_session_state()
        return target

    def start_recording(self, metadata: dict | None = None) -> bool:
        """Start a Trial and open its raw TXT file."""
        if self._recording:
            return False
        self._metadata = dict(metadata or {})
        if (
            not self._session_active
            or self._session_dir is None
            or not self._session_dir.is_dir()
        ):
            self.start_session(self._metadata.get("subject_id", ""))

        now = datetime.now()
        self._trial_count += 1
        dirname = self._path_name(self._metadata.get("task", ""), "unnamed_task")
        self._trial_dir = self._unique_directory(self._session_dir, dirname)
        self._txt_path = self._trial_dir / "emg_data.txt"
        self._packet_path = (
            self._trial_dir / "openbci_packets.csv"
            if "delsys_sync" in self._metadata else None
        )

        try:
            self._trial_dir.mkdir(parents=False)
            self._txt_file = self._txt_path.open("w", encoding="utf-8", newline="")
            self._writer = csv.writer(self._txt_file, lineterminator="\n")
            self._write_header(now)
            if self._packet_path is not None:
                self._packet_file = self._packet_path.open(
                    "w", encoding="utf-8", newline=""
                )
                self._packet_writer = csv.writer(
                    self._packet_file, lineterminator="\n"
                )
                self._packet_writer.writerow(
                    ["Sample Index", "Package Counter", "Timestamp"]
                )
                self._packet_file.flush()
        except Exception as exc:
            print(f"[Recorder] 无法创建 Trial: {exc}")
            if self._txt_file:
                self._txt_file.close()
            if self._packet_file:
                self._packet_file.close()
            self._txt_file = None
            self._packet_file = None
            self._writer = None
            self._packet_writer = None
            if self._trial_dir and self._trial_dir.exists():
                shutil.rmtree(self._trial_dir)
            self._trial_dir = None
            self._txt_path = None
            self._packet_path = None
            self._trial_count -= 1
            return False

        self._sample_count = 0
        self._written_count = 0
        self._write_error = None
        self._trial_stopped_monotonic = None
        self._first_timestamp = None
        self._last_timestamp = None
        self._write_queue = queue.Queue()
        self._recording = True
        self._trial_started_monotonic = time.monotonic()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"emg-txt-writer-{self._trial_count:03d}",
            daemon=False,
        )
        self._writer_thread.start()
        print(f"[Recorder] Trial {self._trial_count:03d} 开始 -> {self._txt_path}")
        return True

    def _write_header(self, started_at: datetime) -> None:
        group = self._header_value(self._metadata.get("group", ""))
        subject = self._header_value(self._metadata.get("subject_id", ""))
        task = self._header_value(self._metadata.get("task", ""))
        sample_rate = self._header_value(self._metadata.get("sampling_rate", "?"))
        session_name = self._session_dir.name if self._session_dir else ""
        self._txt_file.write("% EMG Viewer - YMS (based on BrainFlow)\n")
        self._txt_file.write(f"% Session = {session_name}\n")
        self._txt_file.write(
            f"% Trial = {self._trial_dir.name}; Index = {self._trial_count:03d}; "
            f"Start = {started_at.isoformat()}; "
            f"Group = {group}; Subject = {subject}; Task = {task}\n"
        )
        self._txt_file.write(f"% Number of channels = {self.n_channels}\n")
        self._txt_file.write(f"% Sample Rate = {sample_rate} Hz\n")
        self._txt_file.write("% EXG data unit = microvolts; Timestamp unit = Unix seconds\n")
        columns = (
            ["Sample Index"]
            + [f"EXG Channel {index}" for index in range(self.n_channels)]
            + ["Timestamp"]
        )
        self._writer.writerow(columns)
        self._txt_file.flush()

    def append(self, data: np.ndarray, timestamps: np.ndarray | None = None,
               counters: np.ndarray | None = None) -> None:
        """Queue a sample block. ``data`` shape is channels x samples."""
        if not self._recording:
            return
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(
                f"录制数据通道数不匹配: 期望 {self.n_channels}, 实际 {data.shape}"
            )
        n_samples = data.shape[1]
        if timestamps is None:
            timestamps = np.zeros(n_samples, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64).ravel()
        if len(timestamps) != n_samples:
            raise ValueError("时间戳数量与样本数量不一致")
        if counters is not None:
            counters = np.asarray(counters).ravel()
            if len(counters) != n_samples:
                raise ValueError("包计数器数量与样本数量不一致")

        if n_samples == 0:
            return
        start_index = self._sample_count
        chunk = np.array(data.T, dtype=np.float32, copy=True)
        timestamp_copy = np.array(timestamps, dtype=np.float64, copy=True)
        counter_copy = (
            np.array(counters, dtype=np.int16, copy=True)
            if counters is not None else np.full(n_samples, -1, dtype=np.int16)
        )
        finite_timestamps = timestamp_copy[np.isfinite(timestamp_copy)]
        if finite_timestamps.size:
            if self._first_timestamp is None:
                self._first_timestamp = float(finite_timestamps[0])
            self._last_timestamp = float(finite_timestamps[-1])
        self._sample_count += n_samples
        self._write_queue.put((start_index, chunk, timestamp_copy, counter_copy))

    def mark_acquisition_start(self, monotonic_ns: int, unix_ns: int) -> None:
        """Align live duration and Trial metadata to the synchronized START edge."""
        if not self._recording:
            raise RuntimeError("Cannot align a Trial that is not recording")
        self._trial_started_monotonic = monotonic_ns / 1_000_000_000
        started_at = datetime.fromtimestamp(unix_ns / 1_000_000_000)
        self._metadata["start_time"] = started_at.isoformat()
        if self._sample_count == 0 and self._written_count == 0 and self._txt_file:
            self._txt_file.seek(0)
            self._txt_file.truncate()
            self._write_header(started_at)

    def mark_acquisition_stop(self, monotonic_ns: int, unix_ns: int) -> None:
        """Store the confirmed synchronized STOP epoch for duration reporting."""
        if not self._recording:
            raise RuntimeError("Cannot align a Trial that is not recording")
        self._trial_stopped_monotonic = monotonic_ns / 1_000_000_000
        self._metadata["stop_time"] = datetime.fromtimestamp(
            unix_ns / 1_000_000_000
        ).isoformat()

    def _writer_loop(self) -> None:
        """Format and write queued blocks without occupying the Qt GUI thread."""
        next_flush = time.monotonic() + max(0.1, float(self.flush_sec))
        formats = ["%d"] + ["%.8f"] * self.n_channels + ["%.6f"]
        try:
            while True:
                item = self._write_queue.get()
                if item is None:
                    break
                start_index, chunk, timestamps, counters = item
                rows = len(chunk)
                output = np.empty((rows, self.n_channels + 2), dtype=np.float64)
                output[:, 0] = np.arange(start_index, start_index + rows)
                output[:, 1:-1] = chunk
                output[:, -1] = timestamps
                np.savetxt(self._txt_file, output, delimiter=",", fmt=formats)
                if self._packet_file is not None:
                    packet_output = np.empty((rows, 3), dtype=np.float64)
                    packet_output[:, 0] = output[:, 0]
                    packet_output[:, 1] = counters
                    packet_output[:, 2] = timestamps
                    np.savetxt(
                        self._packet_file, packet_output, delimiter=",",
                        fmt=["%d", "%d", "%.6f"],
                    )
                self._written_count += rows
                if time.monotonic() >= next_flush:
                    self._txt_file.flush()
                    if self._packet_file is not None:
                        self._packet_file.flush()
                    next_flush = time.monotonic() + max(0.1, float(self.flush_sec))
            self._txt_file.flush()
            if self._packet_file is not None:
                self._packet_file.flush()
        except Exception as exc:
            self._write_error = str(exc)
            print(f"[Recorder] TXT 后台写入失败: {exc}")

    def stop_recording(self) -> dict | None:
        """Finish the active Trial and return its list record."""
        if not self._recording:
            return None
        self._recording = False
        if self._write_queue is not None:
            self._write_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join()
        close_errors = []
        for label, handle in (
            ("emg_data.txt", self._txt_file),
            ("openbci_packets.csv", self._packet_file),
        ):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception as exc:
                close_errors.append(f"{label} close failed: {exc}")
        if close_errors:
            self._write_error = "; ".join(
                item for item in (self._write_error, *close_errors) if item
            )
        sample_duration = (
            self._written_count / float(self._metadata.get("sampling_rate", 1000.0))
            if self._written_count else 0.0
        )
        timestamp_span = (
            self._last_timestamp - self._first_timestamp
            if self._first_timestamp is not None and self._last_timestamp is not None
            else None
        )
        trigger_duration = (
            self._trial_stopped_monotonic - self._trial_started_monotonic
            if self._trial_stopped_monotonic is not None else None
        )
        wall_duration = max(
            0.0,
            trigger_duration
            if trigger_duration is not None else sample_duration,
        )
        save_completion_latency = (
            max(0.0, time.monotonic() - self._trial_stopped_monotonic)
            if self._trial_stopped_monotonic is not None else None
        )
        record = {
            "number": self._trial_count,
            "task": self._metadata.get("task", "") or "未命名",
            "start_time": self._metadata.get("start_time", ""),
            "duration_sec": wall_duration,
            "wall_duration_sec": wall_duration,
            "trigger_duration_sec": trigger_duration,
            "sample_count_duration_sec": sample_duration,
            "timestamp_span_sec": timestamp_span,
            "first_sample_timestamp_unix_s": self._first_timestamp,
            "last_sample_timestamp_unix_s": self._last_timestamp,
            "save_completion_latency_sec": save_completion_latency,
            "samples": self._written_count,
            "path": self._trial_dir,
            "txt_path": self._txt_path,
            "packet_path": self._packet_path,
            "write_error": self._write_error,
        }
        self._trials.append(record)
        print(
            f"[Recorder] Trial {self._trial_count:03d} 停止: "
            f"{self._written_count} 样本, {wall_duration:.3f} 秒"
        )
        self._txt_file = None
        self._packet_file = None
        self._writer = None
        self._packet_writer = None
        self._write_queue = None
        self._writer_thread = None
        self._trial_dir = None
        self._txt_path = None
        self._packet_path = None
        return dict(record)

    def list_trials(self) -> list[dict]:
        return [dict(record) for record in self._trials]

    @property
    def current_trial_info(self) -> dict | None:
        if not self._recording or self._trial_dir is None:
            return None
        return {
            "number": self._trial_count,
            "task": self._metadata.get("task", "") or "未命名",
            "start_time": self._metadata.get("start_time", ""),
            "duration_sec": self.elapsed_sec,
            "samples": self._sample_count,
            "path": self._trial_dir,
            "txt_path": self._txt_path,
            "recording": True,
        }

    def delete_trial(self, trial_path: str | Path) -> bool:
        """Permanently delete one completed Trial from the current Session."""
        if self._session_dir is None:
            return False
        target = Path(trial_path).resolve()
        session = self._session_dir.resolve()
        if target.parent != session:
            raise ValueError("只能删除当前 Session 内的 Trial")
        if self._recording and self._trial_dir and target == self._trial_dir.resolve():
            raise RuntimeError("不能删除正在录制的 Trial")
        if not target.is_dir() or not (target / "emg_data.txt").is_file():
            return False
        shutil.rmtree(target)
        from action_review import remove_action_segments
        from time_review import remove_time_review
        remove_action_segments(target)
        remove_time_review(target)
        self._trials = [
            record for record in self._trials
            if Path(record["path"]).resolve() != target
        ]
        return True

    def delete_trials(self, trial_paths) -> list[Path]:
        """Validate first, then delete multiple completed Trials."""
        if self._session_dir is None:
            raise ValueError("当前没有 Session")
        session = self._session_dir.resolve()
        targets = []
        for value in trial_paths:
            target = Path(value).resolve()
            if target not in targets:
                targets.append(target)
        if not targets:
            return []
        for target in targets:
            if target.parent != session:
                raise ValueError("只能删除当前 Session 内的 Trial")
            if self._recording and self._trial_dir and target == self._trial_dir.resolve():
                raise RuntimeError("不能删除正在录制的 Trial")
            if not target.is_dir() or not (target / "emg_data.txt").is_file():
                raise FileNotFoundError(f"Trial 数据不存在: {target}")
        deleted = []
        for target in targets:
            if self.delete_trial(target):
                deleted.append(target)
        return deleted

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def current_trial_dir(self) -> Path | None:
        return self._trial_dir

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_session_active(self) -> bool:
        return bool(
            self._session_active
            and self._session_dir is not None
            and self._session_dir.is_dir()
        )

    @property
    def elapsed_sec(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._trial_started_monotonic
