"""Build a host-timing synchronization report for one completed Trial.

The report deliberately separates observed host timestamps from physical device
timing.  In particular, a Trigno first sample is only *assumed* to coincide with
the confirmed external START edge; this module does not parse a Delsys export.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.2"


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None else None


def _read_events(path: Path) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    warnings: list[str] = []
    if not path.is_file():
        return events, ["sync_events.jsonl is missing"]
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event is not an object")
                events.append(value)
            except (json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"sync_events.jsonl line {line_number} ignored: {exc}")
    return events, warnings


def _last_event(events: list[dict], name: str) -> dict | None:
    return next((event for event in reversed(events) if event.get("event") == name), None)


def _nano_reference(event: dict | None, name: str = "nano_start_pulse") -> dict:
    result = {
        "name": name,
        "status": "missing",
        "estimated_monotonic_ns": None,
        "estimated_unix_ns": None,
        "estimator": None,
        "conservative_host_bracket": None,
        "mcu_rise_us": None,
    }
    if event is None:
        return result

    result["status"] = (
        "confirmed" if event.get("status") == "firmware_confirmed"
        else "execution_unknown"
    )
    result["mcu_rise_us"] = _integer(event.get("mcu_rise_us"))
    write_begin = _integer(event.get("host_write_begin_monotonic_ns"))
    write_end = _integer(event.get("host_write_end_monotonic_ns"))
    reply = _integer(event.get("host_reply_monotonic_ns"))
    write_begin_unix = _integer(event.get("host_write_begin_unix_ns"))

    if write_begin is not None and reply is not None and reply >= write_begin:
        result["conservative_host_bracket"] = {
            "begin_monotonic_ns": write_begin,
            "end_monotonic_ns": reply,
            "width_ms": (reply - write_begin) / 1_000_000,
            "meaning": "Host serial operation interval; not a measured physical-edge interval",
        }

    if (
        result["status"] == "confirmed"
        and write_end is not None
        and reply is not None
        and reply >= write_end
    ):
        estimate = (write_end + reply) // 2
        result["estimated_monotonic_ns"] = estimate
        result["estimator"] = "midpoint_host_write_end_to_reply"
        if write_begin is not None and write_begin_unix is not None:
            result["estimated_unix_ns"] = write_begin_unix + estimate - write_begin
    return result


def estimate_nano_trigger(event: dict | None) -> tuple[int | None, int | None]:
    """Return the confirmed trigger estimate used by the final report."""
    reference = _nano_reference(event)
    return (
        reference["estimated_monotonic_ns"],
        reference["estimated_unix_ns"],
    )


def estimate_nano_start(event: dict | None) -> tuple[int | None, int | None]:
    """Backward-compatible START-specific wrapper."""
    return estimate_nano_trigger(event)


def _openbci_event(event: dict | None, sample_rate: float | int | None) -> dict:
    result = {
        "start_call_begin_monotonic_ns": None,
        "start_call_end_monotonic_ns": None,
        "first_batch_received_monotonic_ns": None,
        "first_brainflow_timestamp_unix_s": None,
        "first_sample_estimated_monotonic_ns": None,
        "sample_rate_hz": sample_rate,
        "timestamp_semantics": (
            "BrainFlow timestamp-channel time mapped to the host monotonic clock; "
            "board-dependent and not guaranteed to be a board hardware sample time"
        ),
    }
    if event is None:
        return result
    received_mono = _integer(event.get("monotonic_ns"))
    received_unix = _integer(event.get("unix_ns"))
    device_unix_s = _number(event.get("first_device_timestamp"))
    result["first_batch_received_monotonic_ns"] = received_mono
    result["first_brainflow_timestamp_unix_s"] = device_unix_s
    if received_mono is not None and received_unix is not None and device_unix_s is not None:
        result["first_sample_estimated_monotonic_ns"] = (
            round(device_unix_s * 1_000_000_000) + received_mono - received_unix
        )
    return result


def _add_openbci_start(result: dict, event: dict | None) -> None:
    if event is None:
        return
    begin = event.get("call_begin")
    end = event.get("call_end")
    if isinstance(begin, dict):
        result["start_call_begin_monotonic_ns"] = _integer(begin.get("monotonic_ns"))
    if isinstance(end, dict):
        result["start_call_end_monotonic_ns"] = _integer(end.get("monotonic_ns"))


def _camera_event(
    start_event: dict | None,
    frames_path: Path,
    session_path: Path,
    requested_fps: float | int | None,
) -> tuple[dict, list[str], bool]:
    warnings: list[str] = []
    gate = None
    if start_event is not None and isinstance(start_event.get("result"), dict):
        gate = _integer(start_event["result"].get("monotonic_ns"))

    timestamps: list[int] = []
    first_index = None
    if not frames_path.is_file():
        warnings.append("camera_frames.csv is missing")
    else:
        with frames_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 2):
                timestamp = _integer(row.get("capture_perf_counter_ns"))
                if timestamp is None:
                    warnings.append(f"camera_frames.csv row {row_number} ignored: invalid timestamp")
                    continue
                if not timestamps:
                    first_index = _integer(row.get("frame_index"))
                timestamps.append(timestamp)

    positive_deltas = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    measured_period = (
        statistics.median(positive_deltas) / 1_000_000 if positive_deltas else None
    )
    session_error = None
    session_frame_count = None
    if not session_path.is_file():
        warnings.append("camera_session.json is missing")
    else:
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(session, dict):
                raise ValueError("root is not an object")
            session_error = str(session.get("error") or "") or None
            session_frame_count = _integer(session.get("frame_count"))
            if session_error:
                warnings.append(f"camera recording reported an error: {session_error}")
            if session_frame_count is None:
                warnings.append("camera_session.json frame_count is missing or invalid")
            if session_frame_count is not None and session_frame_count != len(timestamps):
                warnings.append(
                    "camera frame count mismatch: "
                    f"session={session_frame_count}, csv={len(timestamps)}"
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"camera_session.json ignored: {exc}")

    healthy = (
        session_path.is_file()
        and session_error is None
        and session_frame_count == len(timestamps)
        and len(timestamps) > 0
    )
    return {
        "recording_gate_monotonic_ns": gate,
        "first_frame_capture_return_monotonic_ns": timestamps[0] if timestamps else None,
        "first_frame_index": first_index,
        "recorded_frame_count_in_csv": len(timestamps),
        "recorded_frame_count_in_session": session_frame_count,
        "recording_error": session_error,
        "requested_fps": requested_fps,
        "measured_frame_period_ms": measured_period,
        "timestamp_semantics": (
            "Host perf_counter immediately after capture.read returned; not sensor exposure "
            "time and not disk-write completion time"
        ),
    }, warnings, healthy


def _difference_ms(left: int | None, right: int | None) -> float | None:
    if left is None or right is None:
        return None
    return (left - right) / 1_000_000


def _iso_utc(unix_ns: int | None) -> str | None:
    if unix_ns is None:
        return None
    return datetime.fromtimestamp(unix_ns / 1_000_000_000, timezone.utc).isoformat()


def _monotonic_to_unix_ns(
    value: int | None, reference_monotonic_ns: int | None,
    reference_unix_ns: int | None,
) -> int | None:
    if value is None or reference_monotonic_ns is None or reference_unix_ns is None:
        return None
    return reference_unix_ns + value - reference_monotonic_ns


def _nested(metadata: dict, *keys: str) -> Any:
    value: Any = metadata
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def generate_sync_report(
    trial_dir: str | Path,
    metadata: dict | None,
    reason: str,
) -> tuple[Path, dict]:
    """Generate ``synchronization_report.json`` and return its path and data."""
    directory = Path(trial_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Trial directory does not exist: {directory}")
    metadata = dict(metadata or {})
    sync_path = directory / "sync_events.jsonl"
    frames_path = directory / "camera_frames.csv"
    camera_session_path = directory / "camera_session.json"
    events, warnings = _read_events(sync_path)

    start_trigger = _last_event(events, "start_trigger")
    reference = _nano_reference(start_trigger)
    stop_trigger = _last_event(events, "stop_trigger")
    stop_reference = _nano_reference(stop_trigger, "nano_stop_pulse")
    sample_rate = _number(metadata.get("sampling_rate"))
    openbci = _openbci_event(
        _last_event(events, "first_openbci_data_received"), sample_rate
    )
    _add_openbci_start(openbci, _last_event(events, "openbci_start"))
    requested_fps = _number(_nested(metadata, "camera_sync", "status", "fps"))
    if requested_fps is None:
        requested_fps = _number(metadata.get("camera_fps"))
    camera, camera_warnings, camera_healthy = _camera_event(
        _last_event(events, "camera_start_returned"), frames_path,
        camera_session_path, requested_fps
    )
    warnings.extend(camera_warnings)

    recording_saved = _last_event(events, "recording_saved")
    trial_record = (
        recording_saved.get("trial")
        if isinstance(recording_saved, dict)
        and isinstance(recording_saved.get("trial"), dict) else {}
    )
    packet_diagnostics = (
        trial_record.get("packet_diagnostics")
        if isinstance(trial_record.get("packet_diagnostics"), dict) else None
    )
    if packet_diagnostics is not None:
        if packet_diagnostics.get("complete") is False:
            warnings.append(
                "OpenBCI package-counter diagnostics are incomplete: "
                f"{_integer(packet_diagnostics.get('missing_counter_batch_count')) or 0} "
                "batch(es) had missing or misaligned counters"
            )
        lost_packets = _integer(packet_diagnostics.get("lost")) or 0
        duplicate_packets = _integer(packet_diagnostics.get("duplicate")) or 0
        out_of_order_packets = _integer(packet_diagnostics.get("out_of_order")) or 0
        if lost_packets:
            warnings.append(
                f"OpenBCI package counter detected {lost_packets} missing packet(s)"
            )
        if duplicate_packets:
            warnings.append(
                f"OpenBCI package counter detected {duplicate_packets} duplicate transition(s)"
            )
        if out_of_order_packets:
            warnings.append(
                "OpenBCI package counter detected "
                f"{out_of_order_packets} out-of-order/reset transition(s)"
            )

    reference_ns = reference["estimated_monotonic_ns"]
    stop_reference_ns = stop_reference["estimated_monotonic_ns"]
    openbci_ns = openbci["first_sample_estimated_monotonic_ns"]
    camera_ns = camera["first_frame_capture_return_monotonic_ns"]
    bracket = reference["conservative_host_bracket"]
    nano_uncertainty = None
    if reference_ns is not None and bracket is not None:
        nano_uncertainty = max(
            reference_ns - bracket["begin_monotonic_ns"],
            bracket["end_monotonic_ns"] - reference_ns,
        ) / 1_000_000

    confirmed = reference["status"] == "confirmed" and reference_ns is not None
    complete = confirmed and openbci_ns is not None and camera_ns is not None
    status = "complete" if complete and camera_healthy else (
        "degraded" if complete else
        "trigger_unknown" if reference["status"] == "execution_unknown" else "incomplete"
    )
    reference_unix_ns = reference["estimated_unix_ns"] if confirmed else None
    openbci_unix_ns = (
        round(openbci["first_brainflow_timestamp_unix_s"] * 1_000_000_000)
        if openbci["first_brainflow_timestamp_unix_s"] is not None else None
    )
    camera_unix_ns = _monotonic_to_unix_ns(
        camera_ns, reference_ns if confirmed else None, reference_unix_ns
    )
    openbci_offset = _difference_ms(openbci_ns, reference_ns) if confirmed else None
    camera_offset = _difference_ms(camera_ns, reference_ns) if confirmed else None
    camera_openbci_offset = _difference_ms(camera_ns, openbci_ns)
    available_starts = [
        value for value in (reference_ns if confirmed else None, openbci_ns, camera_ns)
        if value is not None
    ]
    observed_spread_ms = (
        (max(available_starts) - min(available_starts)) / 1_000_000
        if len(available_starts) == 3 else None
    )
    trigger_duration_sec = (
        (stop_reference_ns - reference_ns) / 1_000_000_000
        if reference_ns is not None and stop_reference_ns is not None else None
    )
    sample_count_duration_sec = _number(
        trial_record.get("sample_count_duration_sec")
    )
    timestamp_span_sec = _number(trial_record.get("timestamp_span_sec"))
    duration_difference_sec = (
        sample_count_duration_sec - trigger_duration_sec
        if sample_count_duration_sec is not None and trigger_duration_sec is not None
        else None
    )
    if (
        duration_difference_sec is not None
        and abs(duration_difference_sec) > max(0.1, trigger_duration_sec * 0.01)
    ):
        warnings.append(
            "OpenBCI sample-count duration differs from Nano START-to-STOP "
            f"duration by {duration_difference_sec:.6f} s"
        )
    if start_trigger is not None and (
        stop_reference["status"] != "confirmed" or stop_reference_ns is None
    ):
        warnings.append("Nano STOP trigger is missing or unconfirmed")

    event_warning_defaults = {
        "openbci_boundary_filter_skipped": "OpenBCI boundary filtering was skipped",
        "openbci_counter_alignment_failed": (
            "OpenBCI package-counter diagnostics are incomplete"
        ),
    }
    for event in events:
        if event.get("event") in event_warning_defaults:
            warning = str(
                event.get("warning")
                or event_warning_defaults[event.get("event")]
            )
            if warning not in warnings:
                warnings.append(warning)

    stop_clicked = _last_event(events, "stop_clicked")
    stop_click_ns = _integer(_nested(stop_clicked or {}, "click", "monotonic_ns"))
    stop_write_begin_ns = _integer(
        stop_trigger.get("host_write_begin_monotonic_ns") if stop_trigger else None
    )
    camera_stop_begin = _last_event(events, "camera_stop_begin")
    camera_stop_ns = _integer(
        _nested(camera_stop_begin or {}, "call_begin", "monotonic_ns")
    )
    openbci_stop_begin = _last_event(events, "openbci_stop_begin")
    openbci_stop_returned = _last_event(events, "openbci_stop_returned")
    openbci_stop_begin_ns = _integer(
        openbci_stop_begin.get("monotonic_ns") if openbci_stop_begin else None
    )
    openbci_stop_end_ns = _integer(
        openbci_stop_returned.get("monotonic_ns") if openbci_stop_returned else None
    )
    recording_saved_ns = _integer(
        recording_saved.get("monotonic_ns") if recording_saved else None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "status_meaning": "Completeness of the host-side estimate, not physical synchronization accuracy",
        "method": "host_timing_estimate",
        "summary": {
            "sync_reference": "Nano START host serial midpoint heuristic",
            "sync_time_estimated_utc": _iso_utc(reference_unix_ns),
            "delsys_acquisition_start_estimated_utc": _iso_utc(reference_unix_ns),
            "openbci_first_sample_estimated_utc": _iso_utc(openbci_unix_ns),
            "camera_first_recorded_frame_estimated_utc": _iso_utc(camera_unix_ns),
            "camera_first_frame_capture_return_estimated_utc": _iso_utc(camera_unix_ns),
            "openbci_minus_delsys_ms": openbci_offset,
            "camera_minus_delsys_ms": camera_offset,
            "camera_minus_openbci_ms": camera_openbci_offset,
            "observed_host_timing_spread_ms": observed_spread_ms,
            "nano_host_mapping_uncertainty_ms": nano_uncertainty,
            "openbci_minus_delsys_host_bracket_ms": (
                {
                    "minimum": _difference_ms(openbci_ns, bracket["end_monotonic_ns"]),
                    "maximum": _difference_ms(openbci_ns, bracket["begin_monotonic_ns"]),
                } if openbci_ns is not None and bracket is not None else None
            ),
            "camera_minus_delsys_host_bracket_ms": (
                {
                    "minimum": _difference_ms(camera_ns, bracket["end_monotonic_ns"]),
                    "maximum": _difference_ms(camera_ns, bracket["begin_monotonic_ns"]),
                } if camera_ns is not None and bracket is not None else None
            ),
            "result_scope": "host-side estimate; Delsys first sample and camera exposure are not directly observed",
        },
        "durations": {
            "nano_start_to_stop_sec": trigger_duration_sec,
            "openbci_sample_count_sec": sample_count_duration_sec,
            "openbci_timestamp_span_sec": timestamp_span_sec,
            "sample_count_minus_trigger_sec": duration_difference_sec,
            "trial_display_duration_sec": _number(trial_record.get("duration_sec")),
            "save_completion_latency_sec": _number(
                trial_record.get("save_completion_latency_sec")
            ),
        },
        "openbci_transport": packet_diagnostics,
        "trial": {
            "name": directory.name,
            "path": str(directory),
            "finish_reason": str(reason),
            "metadata": metadata,
        },
        "reference": reference,
        "stop_reference": stop_reference,
        "events": {
            "delsys_first_sample": {
                "estimated_monotonic_ns": reference_ns if confirmed else None,
                "estimated_unix_ns": reference["estimated_unix_ns"] if confirmed else None,
                "source": "assumed_from_confirmed_external_start",
                "observed_in_delsys_export": False,
            },
            "openbci": openbci,
            "camera": camera,
            "stop": {
                "click_to_serial_write_ms": _difference_ms(
                    stop_write_begin_ns, stop_click_ns
                ),
                "estimated_monotonic_ns": stop_reference_ns,
                "estimated_unix_ns": stop_reference["estimated_unix_ns"],
                "camera_gate_close_minus_stop_ms": _difference_ms(
                    camera_stop_ns, stop_reference_ns
                ),
                "openbci_stop_call_duration_ms": _difference_ms(
                    openbci_stop_end_ns, openbci_stop_begin_ns
                ),
                "recording_saved_minus_stop_ms": _difference_ms(
                    recording_saved_ns, stop_reference_ns
                ),
            },
        },
        "offsets_ms": {
            "openbci_first_sample_minus_delsys_trigger": openbci_offset,
            "camera_first_frame_minus_delsys_trigger": camera_offset,
            "camera_first_frame_minus_openbci_first_sample": camera_openbci_offset,
            "camera_recording_gate_minus_delsys_trigger": (
                _difference_ms(camera["recording_gate_monotonic_ns"], reference_ns)
                if confirmed else None
            ),
            "openbci_start_call_end_minus_delsys_trigger": (
                _difference_ms(openbci["start_call_end_monotonic_ns"], reference_ns)
                if confirmed else None
            ),
            "sign_convention": "positive means the first named event occurred later",
        },
        "uncertainty": {
            "nano_host_mapping_ms": nano_uncertainty,
            "openbci_sample_period_ms": (
                1000 / sample_rate if sample_rate is not None and sample_rate > 0 else None
            ),
            "openbci_transport_latency_ms": None,
            "camera_frame_period_ms": camera["measured_frame_period_ms"] or (
                1000 / requested_fps
                if requested_fps is not None and requested_fps > 0 else None
            ),
            "camera_driver_latency_ms": None,
            "physical_sync_error_bound_ms": None,
            "claimable_physical_accuracy": False,
        },
        "limitations": [
            "All offsets are host-side timing estimates, not a shared hardware-clock measurement.",
            "The Delsys first sample is assumed to coincide with the confirmed Nano START edge; no Trigno export is parsed.",
            "BrainFlow timestamp semantics are board-dependent and may include host or transport timing.",
            "Camera timestamps are captured after frame delivery and do not identify sensor exposure time.",
            "Unknown OpenBCI transport latency and camera driver buffering prevent a physical error bound.",
            "mcu_rise_us is retained as MCU-relative evidence but cannot be mapped to host time from one exchange.",
            "OpenBCI package-counter diagnostics are host-observed 8-bit sequence evidence; gaps of 128 or more cannot be distinguished from resets or reordering.",
        ],
        "warnings": warnings,
        "source_files": {
            "sync_events": "sync_events.jsonl" if sync_path.is_file() else None,
            "camera_frames": "camera_frames.csv" if frames_path.is_file() else None,
            "camera_session": (
                "camera_session.json" if camera_session_path.is_file() else None
            ),
            "openbci_packets": (
                "openbci_packets.csv"
                if (directory / "openbci_packets.csv").is_file() else None
            ),
        },
    }

    target = directory / "synchronization_report.json"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".synchronization_report.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return target, report
