"""Delsys trigger orchestration; serial methods run outside the GUI thread.

Host timestamps are operational evidence, not a shared device sample clock.
No operation reconnects or retries a trigger automatically.
"""
import json
import math
import threading
import time

from nano_client import NanoClient
from sync_report import estimate_nano_start, estimate_nano_trigger, generate_sync_report


OPENBCI_STARTUP_DATA_TIMEOUT_SEC = 2.0
OPENBCI_STARTUP_POLL_SEC = 0.01
OPENBCI_STARTUP_STABILITY_WINDOW_SEC = 0.2
OPENBCI_STARTUP_STABLE_WINDOWS = 2
OPENBCI_STARTUP_STABILIZATION_TIMEOUT_SEC = 3.0
OPENBCI_STARTUP_RATE_MIN_RATIO = 0.5
OPENBCI_STARTUP_RATE_MAX_RATIO = 1.5
OPENBCI_STARTUP_MAX_BATCH_LATENCY_SEC = 0.064


class StartupStabilizationError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def _drain_openbci_startup(board, sample_rate):
    """Discard delayed startup backlog until two live-rate windows are stable."""
    sample_rate = float(sample_rate)
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise StartupStabilizationError("OpenBCI 采样率无效，未发送 Delsys START")

    started = time.monotonic()
    first_deadline = started + OPENBCI_STARTUP_DATA_TIMEOUT_SEC
    poll_count = 0
    total_samples = 0
    largest_batch = 0
    first_batch_samples = None

    while time.monotonic() < first_deadline:
        poll_count += 1
        batch = board.get_new_data(n_samples=None)
        if batch is None:
            raise StartupStabilizationError(
                "OpenBCI 启动缓存读取失败，未发送 Delsys START"
            )
        count = int(batch.shape[-1]) if getattr(batch, "size", 0) else 0
        if count:
            first_batch_samples = count
            total_samples += count
            largest_batch = count
            break
        time.sleep(OPENBCI_STARTUP_POLL_SEC)
    if first_batch_samples is None:
        raise StartupStabilizationError(
            "OpenBCI 启动后 2 秒内未收到数据，未启动摄像机或发送 Delsys START"
        )

    stabilization_deadline = time.monotonic() + OPENBCI_STARTUP_STABILIZATION_TIMEOUT_SEC
    window_started = time.monotonic()
    window_samples = 0
    window_max_batch = 0
    stable_windows = 0
    unstable_windows = 0
    windows = []
    max_batch_limit = max(
        64, int(math.ceil(sample_rate * OPENBCI_STARTUP_MAX_BATCH_LATENCY_SEC))
    )

    while time.monotonic() < stabilization_deadline:
        time.sleep(OPENBCI_STARTUP_POLL_SEC)
        poll_count += 1
        batch = board.get_new_data(n_samples=None)
        if batch is None:
            raise StartupStabilizationError(
                "OpenBCI 稳定化读取失败，未启动摄像机或发送 Delsys START",
                {"sample_count": total_samples, "poll_count": poll_count},
            )
        count = int(batch.shape[-1]) if getattr(batch, "size", 0) else 0
        total_samples += count
        largest_batch = max(largest_batch, count)
        window_samples += count
        window_max_batch = max(window_max_batch, count)
        now = time.monotonic()
        elapsed = now - window_started
        if elapsed < OPENBCI_STARTUP_STABILITY_WINDOW_SEC:
            continue

        expected = sample_rate * elapsed
        stable = (
            expected * OPENBCI_STARTUP_RATE_MIN_RATIO <= window_samples
            <= expected * OPENBCI_STARTUP_RATE_MAX_RATIO
            and window_max_batch <= max_batch_limit
        )
        windows.append({
            "duration_ms": elapsed * 1000,
            "sample_count": window_samples,
            "max_batch_samples": window_max_batch,
            "expected_samples": expected,
            "stable": stable,
        })
        if stable:
            stable_windows += 1
        else:
            stable_windows = 0
            unstable_windows += 1
        if stable_windows >= OPENBCI_STARTUP_STABLE_WINDOWS:
            return {
                "sample_count": total_samples,
                "first_batch_sample_count": first_batch_samples,
                "poll_count": poll_count,
                "wait_duration_ms": (now - started) * 1000,
                "nominal_sample_rate_hz": sample_rate,
                "stability_window_ms": OPENBCI_STARTUP_STABILITY_WINDOW_SEC * 1000,
                "required_stable_windows": OPENBCI_STARTUP_STABLE_WINDOWS,
                "completed_stable_windows": stable_windows,
                "largest_batch_samples": largest_batch,
                "max_stable_batch_samples": max_batch_limit,
                "unstable_window_count": unstable_windows,
                "windows": windows,
            }
        window_started = now
        window_samples = 0
        window_max_batch = 0

    report = {
        "sample_count": total_samples,
        "first_batch_sample_count": first_batch_samples,
        "poll_count": poll_count,
        "wait_duration_ms": (time.monotonic() - started) * 1000,
        "largest_batch_samples": largest_batch,
        "unstable_window_count": unstable_windows,
        "windows": windows,
    }
    raise StartupStabilizationError(
        "OpenBCI 启动数据在 3 秒内未稳定，未启动摄像机或发送 Delsys START",
        report,
    )


def clock_event():
    return {"unix_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns()}


class DelsysController:
    def __init__(self, camera=None):
        self.client = None
        self.camera = camera
        self.active = False
        self.camera_active = False
        self._camera_prepared = False
        self._stop_confirmed = False
        self._log = None
        self._lock = threading.RLock()
        self._first_data = False
        self._log_warning = ""
        self._last_trigger_monotonic = None
        self._trial_dir = None
        self._metadata = None
        self.last_report_path = None
        self.last_report = None
        self.sync_start_monotonic_ns = None
        self.sync_start_unix_ns = None
        self.sync_stop_monotonic_ns = None
        self.sync_stop_unix_ns = None

    @property
    def ready(self):
        return self.client is not None and self.client.ready

    @property
    def warning(self):
        return self._log_warning

    def connect(self, port):
        if self.active:
            raise RuntimeError("采集中不能重新连接 Nano")
        self.disconnect()
        self.client = NanoClient.connect(port)
        return {"port": port, "ready": self.ready, "firmware": "DELSYS_SYNC 2"}

    def disconnect(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None

    def _event(self, name, **fields):
        with self._lock:
            if self._log is None:
                return
            try:
                self._log.write(json.dumps({"event": name, **clock_event(), **fields},
                                           ensure_ascii=False, default=str) + "\n")
                self._log.flush()
            except Exception as exc:
                self._log_warning = f"同步事件日志写入失败: {exc}；EEG 采集状态不受此标记替代"

    def _result(self, running, warning="", report=None, camera_report=None):
        return {"running": running, "warning": "；".join(
            item for item in (warning, self._log_warning) if item),
            "report": report or {}, "camera_report": camera_report or {},
            "sync_start_monotonic_ns": self.sync_start_monotonic_ns,
            "sync_start_unix_ns": self.sync_start_unix_ns,
            "sync_stop_monotonic_ns": self.sync_stop_monotonic_ns,
            "sync_stop_unix_ns": self.sync_stop_unix_ns}

    @staticmethod
    def _camera_call_succeeded(result):
        if result is False:
            return False
        if isinstance(result, dict):
            for key in ("ok", "success", "running"):
                if key in result and result[key] is False:
                    return False
        return True

    def _stop_camera(self, event_prefix="camera_stop"):
        if self.camera is None or not (self.camera_active or self._camera_prepared):
            return {}
        before = clock_event()
        self._event(f"{event_prefix}_begin", call_begin=before)
        try:
            result = self.camera.stop_recording()
            after = clock_event()
            self._event(f"{event_prefix}_returned", call_begin=before,
                        call_end=after, result=result)
            return result or {}
        except Exception as exc:
            self._event(f"{event_prefix}_failed", call_begin=before, error=str(exc))
            self._log_warning = "；".join(item for item in (
                self._log_warning, f"摄像机停止失败: {exc}") if item)
            return {"error": str(exc)}
        finally:
            self.camera_active = False
            self._camera_prepared = False

    def stop_camera(self):
        """Stop camera recording after the Delsys STOP trigger has been attempted."""
        return self._stop_camera()

    def note_event(self, name, **fields):
        self._event(name, **fields)

    def note_warning(self, message, event="sync_warning"):
        """Record one synchronization warning without repeating it every poll."""
        if message in self._log_warning:
            return
        self._log_warning = "；".join(
            item for item in (self._log_warning, message) if item
        )
        self._event(event, warning=message)

    def start(self, board, recorder, metadata, click_event):
        if self.active or recorder.is_recording:
            return self._result(False, "已有采集或触发尚未处理，请先停止")
        if not self.ready:
            return self._result(False, "Nano 未就绪，请重新连接")
        if self.camera is not None and not self.camera.ready:
            return self._result(False, "摄像机未就绪，请重新连接")
        self._first_data = False
        self._stop_confirmed = False
        self._log_warning = ""
        self.camera_active = False
        self._camera_prepared = False
        self._trial_dir = None
        self._metadata = None
        self.last_report_path = None
        self.last_report = None
        self.sync_start_monotonic_ns = None
        self.sync_start_unix_ns = None
        self.sync_stop_monotonic_ns = None
        self.sync_stop_unix_ns = None
        report = {}
        camera_report = {}
        recording_opened = False
        stream_attempted = False
        camera_attempted = False
        camera_start_begin = None
        camera_start_end = None
        try:
            status = self.client.status()
            if status["response"].split()[1] != "IDLE":
                raise RuntimeError("Nano 正在输出脉冲，请等待完成")
            camera_status = self.camera.status() if self.camera is not None else {}
            report = board.apply_and_verify_settings()
            if report.get("state") == "failed":
                raise RuntimeError("OpenBCI 硬件设置验证失败")
            metadata = dict(metadata)
            metadata["delsys_sync"] = {"click": click_event, "verification": report}
            if self.camera is not None:
                metadata["camera_sync"] = {"status": {
                    key: camera_status.get(key)
                    for key in ("source", "size", "fps")
                }}
            if not recorder.start_recording(metadata):
                raise RuntimeError("无法创建录制文件，未启动设备或发送触发")
            recording_opened = True
            self._trial_dir = recorder.current_trial_dir
            self._metadata = metadata
            # Open the independent sidecar before starting either device.
            self._log = (recorder.current_trial_dir / "sync_events.jsonl").open(
                "x", encoding="utf-8")
            self._event("start_clicked", click=click_event)
            self._event("prepared", nano_status=status, verification=report)
            if self._log_warning:
                raise RuntimeError(self._log_warning)
            if self.camera is not None:
                camera_attempted = True
                # A failed prepare call may still have opened a partial writer.
                self._camera_prepared = True
                prepare_begin = clock_event()
                self._event("camera_prepare_begin", call_begin=prepare_begin)
                try:
                    prepared = self.camera.prepare_recording(
                        recorder.current_trial_dir, metadata)
                except Exception as camera_exc:
                    self._event("camera_prepare_failed", call_begin=prepare_begin,
                                error=str(camera_exc))
                    raise
                prepare_end = clock_event()
                if not self._camera_call_succeeded(prepared):
                    self._event("camera_prepare_failed", call_begin=prepare_begin,
                                call_end=prepare_end, result=prepared)
                    raise RuntimeError("摄像机录制准备失败，未启动设备或发送触发")
                self._event("camera_prepare_returned", call_begin=prepare_begin,
                            call_end=prepare_end, result=prepared)
            before = clock_event()
            stream_attempted = True
            started = board.start_stream(settings_prepared=True)
            after = clock_event()
            if not started:
                raise RuntimeError("OpenBCI 启动失败，未发送 Delsys START")
            try:
                startup_sample_rate = metadata.get("sampling_rate")
                if startup_sample_rate is None:
                    startup_sample_rate = board.sample_rate
                startup_drain = _drain_openbci_startup(
                    board, startup_sample_rate
                )
            except StartupStabilizationError as exc:
                self._event(
                    "openbci_pretrigger_stabilization_failed",
                    error=str(exc),
                    **exc.report,
                )
                raise
            self._event("openbci_pretrigger_discarded", **startup_drain)
            if self.camera is not None:
                camera_start_begin = clock_event()
                self._event("camera_start_begin", call_begin=camera_start_begin)
                try:
                    camera_report = self.camera.start_recording()
                except Exception as camera_exc:
                    self._event("camera_start_failed", call_begin=camera_start_begin,
                                error=str(camera_exc))
                    raise
                camera_start_end = clock_event()
                if not self._camera_call_succeeded(camera_report):
                    self._event("camera_start_failed", call_begin=camera_start_begin,
                                call_end=camera_start_end, result=camera_report)
                    raise RuntimeError("摄像机启动失败，未发送 Delsys START")
                camera_report = camera_report or {}
                self.camera_active = True
        except Exception as exc:
            self._event("start_failed", error=str(exc))
            if stream_attempted:
                try:
                    board.stop_stream()
                except Exception as stop_exc:
                    self._event("openbci_cleanup_failed", error=str(stop_exc))
                    self._log_warning = f"OpenBCI 启动失败后停止也失败: {stop_exc}"
            if camera_attempted:
                self._stop_camera("camera_cleanup")
            if recording_opened:
                try:
                    recorder.stop_recording()
                except Exception as save_exc:
                    self._event("recording_cleanup_failed", error=str(save_exc))
                    self._log_warning += f"；录制关闭失败: {save_exc}"
            self.finish("start_failed")
            return self._result(False, str(exc), report, camera_report)

        # Do not write/flush logs between opening the camera gate and this trigger.
        self.active = True  # An exception may mean the edge occurred but ACK was lost.
        try:
            trigger = self.client.trigger("START")
            self._last_trigger_monotonic = time.monotonic()
            warning = ""
        except Exception as exc:
            self.client.ready = False
            trigger = {**getattr(exc, "event", {}), "error": str(exc),
                       "status": "execution_unknown"}
            warning = "Delsys START 未确认，EEG 继续保存；请检查 Delsys，不会自动重发"
        if camera_start_begin is not None:
            self._event("camera_start_returned", call_begin=camera_start_begin,
                        call_end=camera_start_end, result=camera_report)
        self._event("openbci_start", call_begin=before, call_end=after)
        self._event("start_trigger", **trigger)
        try:
            (
                self.sync_start_monotonic_ns,
                self.sync_start_unix_ns,
            ) = estimate_nano_start(trigger)
            if (
                self.sync_start_monotonic_ns is not None
                and self.sync_start_unix_ns is not None
            ):
                recorder.mark_acquisition_start(
                    self.sync_start_monotonic_ns, self.sync_start_unix_ns
                )
                if self.camera is not None:
                    self.camera.mark_acquisition_start(
                        self.sync_start_monotonic_ns, self.sync_start_unix_ns
                    )
                if self._metadata is not None:
                    self._metadata["start_time"] = recorder.current_trial_info["start_time"]
        except Exception as exc:
            self._event("sync_epoch_alignment_failed", error=str(exc))
            warning = "；".join(item for item in (
                warning,
                "START 已执行，但同步时间基准写入失败；请检查本次同步报告",
            ) if item)
        return self._result(True, warning, report, camera_report)

    def stop_trigger(self, click_event):
        self._event("stop_clicked", click=click_event)
        if not self.active or self._stop_confirmed:
            return self._result(False)
        if not self.ready:
            warning = "Nano 未就绪，未发送 STOP；请在 Delsys 软件中确认并手动停止"
            self._event("stop_unconfirmed", error=warning)
            return self._result(True, warning)
        try:
            if self._last_trigger_monotonic is not None:
                remaining = 2.05 - (time.monotonic() - self._last_trigger_monotonic)
                if remaining > 0:
                    time.sleep(remaining)
            deadline = time.monotonic() + 1.0
            while self.client.status()["response"].split()[1] != "IDLE":
                if time.monotonic() >= deadline:
                    raise RuntimeError("等待 START 脉冲结束超时")
                time.sleep(0.01)
            event = self.client.trigger("STOP")
            self._stop_confirmed = True
            self._event("stop_trigger", **event)
            (
                self.sync_stop_monotonic_ns,
                self.sync_stop_unix_ns,
            ) = estimate_nano_trigger(event)
            return self._result(True)
        except Exception as exc:
            self.client.ready = False
            self._event("stop_unconfirmed", error=str(exc),
                        serial_event=getattr(exc, "event", {}))
            return self._result(True, "Delsys STOP 未确认，请检查并手动停止；不会自动重发")

    def note_first_data(self, data, timestamps):
        with self._lock:
            if self._first_data or self._log is None or data is None or data.size == 0:
                return
            self._first_data = True
            self._event("first_openbci_data_received", sample_count=int(data.shape[-1]),
                        first_device_timestamp=float(timestamps[0])
                        if timestamps is not None and len(timestamps) else None,
                        meaning=(
                            "Event clock is first host drain time; first_device_timestamp "
                            "is the first BrainFlow timestamp and is not guaranteed to be "
                            "a board hardware sample clock"
                        ))

    def finish(self, reason):
        if self.camera_active or self._camera_prepared:
            self._stop_camera()
        warning = (
            "Delsys STOP 未确认，请检查 Delsys 并按需手动停止"
            if self.active and not self._stop_confirmed else ""
        )
        self._event("trial_finished", reason=reason, warning=warning)
        self.active = False
        with self._lock:
            if self._log is not None:
                try:
                    self._log.close()
                except Exception as exc:
                    self._log_warning = f"同步日志关闭失败: {exc}"
                self._log = None
        trial_dir, metadata = self._trial_dir, self._metadata
        self._trial_dir = None
        self._metadata = None
        if trial_dir is not None:
            try:
                path, report = generate_sync_report(trial_dir, metadata or {}, reason)
                self.last_report_path = path
                self.last_report = report
            except Exception as exc:
                self._log_warning = "；".join(item for item in (
                    self._log_warning, f"同步报告 JSON 生成失败: {exc}") if item)
        return "；".join(item for item in (warning, self._log_warning) if item)
