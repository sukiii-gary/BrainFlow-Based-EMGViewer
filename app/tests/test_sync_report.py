import csv
import json
import tempfile
import unittest
from pathlib import Path

from sync_report import generate_sync_report


class SyncReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.trial = Path(self.temporary.name) / "trial"
        self.trial.mkdir()

    def write_events(self, *events):
        with (self.trial / "sync_events.jsonl").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")

    def write_frames(self, rows):
        with (self.trial / "camera_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "capture_perf_counter_ns", "from_recording_start_ms"])
            writer.writerows(rows)

    def write_camera_session(self, frame_count, error=""):
        (self.trial / "camera_session.json").write_text(
            json.dumps({"frame_count": frame_count, "error": error}),
            encoding="utf-8",
        )

    def complete_fixture(self):
        self.write_events(
            {
                "event": "camera_start_returned",
                "result": {"monotonic_ns": 1_000_000_000},
            },
            {
                "event": "openbci_start",
                "call_begin": {"monotonic_ns": 1_010_000_000},
                "call_end": {"monotonic_ns": 1_020_000_000},
            },
            {
                "event": "start_trigger",
                "status": "firmware_confirmed",
                "host_write_begin_monotonic_ns": 1_029_000_000,
                "host_write_begin_unix_ns": 2_029_000_000,
                "host_write_end_monotonic_ns": 1_030_000_000,
                "host_reply_monotonic_ns": 1_034_000_000,
                "mcu_rise_us": 12345,
            },
            {
                "event": "first_openbci_data_received",
                "monotonic_ns": 1_100_000_000,
                "unix_ns": 2_100_000_000,
                "first_device_timestamp": 2.04,
            },
        )
        self.write_frames(
            [
                [0, 1_050_000_000, "50.000"],
                [1, 1_083_000_000, "83.000"],
                [2, 1_117_000_000, "117.000"],
            ]
        )
        self.write_camera_session(3)

    def test_complete_report_calculates_offsets_and_uncertainty(self):
        self.complete_fixture()
        path, report = generate_sync_report(
            self.trial,
            {"sampling_rate": 1000, "camera_sync": {"status": {"fps": 30}}},
            "user_stop",
        )

        self.assertEqual(path.resolve(), (self.trial / "synchronization_report.json").resolve())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["reference"]["estimated_monotonic_ns"], 1_032_000_000)
        self.assertEqual(report["reference"]["estimated_unix_ns"], 2_032_000_000)
        self.assertEqual(
            report["events"]["openbci"]["first_sample_estimated_monotonic_ns"],
            1_040_000_000,
        )
        self.assertEqual(report["offsets_ms"]["openbci_first_sample_minus_delsys_trigger"], 8)
        self.assertEqual(report["offsets_ms"]["camera_first_frame_minus_delsys_trigger"], 18)
        self.assertEqual(report["offsets_ms"]["camera_first_frame_minus_openbci_first_sample"], 10)
        self.assertEqual(report["uncertainty"]["nano_host_mapping_ms"], 3)
        self.assertEqual(report["events"]["camera"]["measured_frame_period_ms"], 33.5)
        self.assertEqual(report["summary"]["openbci_minus_delsys_ms"], 8)
        self.assertEqual(report["summary"]["camera_minus_delsys_ms"], 18)
        self.assertEqual(
            report["summary"]["camera_minus_delsys_host_bracket_ms"],
            {"minimum": 16, "maximum": 21},
        )
        self.assertEqual(report["summary"]["observed_host_timing_spread_ms"], 18)
        self.assertEqual(report["summary"]["sync_time_estimated_utc"],
                         "1970-01-01T00:00:02.032000+00:00")
        self.assertEqual(report["summary"]["openbci_first_sample_estimated_utc"],
                         "1970-01-01T00:00:02.040000+00:00")
        self.assertEqual(report["summary"]["camera_first_recorded_frame_estimated_utc"],
                         "1970-01-01T00:00:02.050000+00:00")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], "1.2")
        self.assertFalse(list(self.trial.glob(".synchronization_report.*.tmp")))

    def test_unknown_trigger_does_not_claim_delsys_or_reference_offsets(self):
        self.write_events(
            {
                "event": "start_trigger",
                "status": "execution_unknown",
                "host_write_begin_monotonic_ns": 10,
                "host_write_end_monotonic_ns": 20,
                "host_error_monotonic_ns": 30,
            },
            {
                "event": "first_openbci_data_received",
                "monotonic_ns": 100,
                "unix_ns": 200,
                "first_device_timestamp": 0.00000019,
            },
        )
        self.write_frames([[0, 110, "0"]])
        self.write_camera_session(1)

        _, report = generate_sync_report(self.trial, {"sampling_rate": 250}, "stop")

        self.assertEqual(report["status"], "trigger_unknown")
        self.assertIsNone(report["reference"]["estimated_monotonic_ns"])
        self.assertIsNone(report["events"]["delsys_first_sample"]["estimated_monotonic_ns"])
        self.assertIsNone(report["offsets_ms"]["openbci_first_sample_minus_delsys_trigger"])
        self.assertFalse(report["uncertainty"]["claimable_physical_accuracy"])

    def test_missing_openbci_and_camera_produces_incomplete_report(self):
        self.write_events(
            {
                "event": "start_trigger",
                "status": "firmware_confirmed",
                "host_write_begin_monotonic_ns": 100,
                "host_write_begin_unix_ns": 200,
                "host_write_end_monotonic_ns": 110,
                "host_reply_monotonic_ns": 130,
            }
        )

        path, report = generate_sync_report(self.trial, {"sampling_rate": 1000}, "short")

        self.assertTrue(path.is_file())
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["events"]["openbci"]["first_sample_estimated_monotonic_ns"])
        self.assertIsNone(report["events"]["camera"]["first_frame_capture_return_monotonic_ns"])
        self.assertIn("camera_frames.csv is missing", report["warnings"])

    def test_camera_period_uses_only_positive_adjacent_intervals(self):
        self.write_events()
        self.write_frames(
            [[0, 100_000_000, "0"], [1, 90_000_000, "1"],
             [2, 120_000_000, "2"], [3, 120_000_000, "3"],
             [4, 160_000_000, "4"]]
        )
        self.write_camera_session(5)

        _, report = generate_sync_report(self.trial, {}, "stop")

        self.assertEqual(report["events"]["camera"]["measured_frame_period_ms"], 35)

    def test_requested_fps_is_fallback_when_only_one_frame_exists(self):
        self.write_events()
        self.write_frames([[0, 100, "0"]])
        self.write_camera_session(1)

        _, report = generate_sync_report(
            self.trial, {"camera_sync": {"status": {"fps": 25}}}, "stop"
        )

        self.assertIsNone(report["events"]["camera"]["measured_frame_period_ms"])
        self.assertEqual(report["uncertainty"]["camera_frame_period_ms"], 40)

    def test_bad_input_lines_are_skipped_and_report_remains_valid(self):
        (self.trial / "sync_events.jsonl").write_text(
            "not json\n" + json.dumps({"event": "prepared"}) + "\n",
            encoding="utf-8",
        )
        with (self.trial / "camera_frames.csv").open("w", encoding="utf-8") as handle:
            handle.write("frame_index,capture_perf_counter_ns\n0,bad\n1,123\n")
        self.write_camera_session(1)

        _, report = generate_sync_report(self.trial, {}, "stop")

        self.assertEqual(report["events"]["camera"]["first_frame_index"], 1)
        self.assertEqual(report["events"]["camera"]["first_frame_capture_return_monotonic_ns"], 123)
        self.assertEqual(len(report["warnings"]), 2)

    def test_camera_recording_error_marks_report_degraded(self):
        self.complete_fixture()
        self.write_camera_session(3, "writer failed")

        _, report = generate_sync_report(self.trial, {"sampling_rate": 1000}, "stop")

        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["events"]["camera"]["recording_error"], "writer failed")
        self.assertTrue(any("writer failed" in item for item in report["warnings"]))

    def test_invalid_camera_frame_count_has_explicit_warning(self):
        self.complete_fixture()
        self.write_camera_session("bad")

        _, report = generate_sync_report(self.trial, {}, "stop")

        self.assertEqual(report["status"], "degraded")
        self.assertIn(
            "camera_session.json frame_count is missing or invalid",
            report["warnings"],
        )

    def test_unconfirmed_stop_event_has_explicit_warning(self):
        self.complete_fixture()
        events = [
            json.loads(line)
            for line in (self.trial / "sync_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        events.append({"event": "stop_trigger", "status": "execution_unknown"})
        self.write_events(*events)

        _, report = generate_sync_report(self.trial, {}, "stop")

        self.assertIn("Nano STOP trigger is missing or unconfirmed", report["warnings"])

    def test_stop_metrics_and_duration_mismatch_warning(self):
        self.complete_fixture()
        events = [
            json.loads(line)
            for line in (self.trial / "sync_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        events.extend([
            {"event": "stop_clicked", "click": {"monotonic_ns": 2_025_000_000}},
            {
                "event": "stop_trigger", "status": "firmware_confirmed",
                "host_write_begin_monotonic_ns": 2_029_000_000,
                "host_write_begin_unix_ns": 3_029_000_000,
                "host_write_end_monotonic_ns": 2_030_000_000,
                "host_reply_monotonic_ns": 2_034_000_000,
            },
            {"event": "camera_stop_begin", "call_begin": {"monotonic_ns": 2_036_000_000}},
            {"event": "openbci_stop_begin", "monotonic_ns": 2_040_000_000},
            {"event": "openbci_stop_returned", "monotonic_ns": 4_040_000_000},
            {
                "event": "recording_saved", "monotonic_ns": 4_050_000_000,
                "trial": {
                    "duration_sec": 1.0,
                    "sample_count_duration_sec": 1.5,
                    "timestamp_span_sec": 1.01,
                    "save_completion_latency_sec": 2.018,
                },
            },
        ])
        self.write_events(*events)

        _, report = generate_sync_report(
            self.trial, {"sampling_rate": 1000}, "user_stop"
        )

        self.assertEqual(report["durations"]["nano_start_to_stop_sec"], 1.0)
        self.assertEqual(report["durations"]["sample_count_minus_trigger_sec"], 0.5)
        self.assertEqual(report["events"]["stop"]["click_to_serial_write_ms"], 4)
        self.assertEqual(report["events"]["stop"]["camera_gate_close_minus_stop_ms"], 4)
        self.assertEqual(report["events"]["stop"]["openbci_stop_call_duration_ms"], 2000)
        self.assertTrue(any("sample-count duration differs" in item for item in report["warnings"]))

    def test_packet_diagnostics_are_preserved_and_warn_on_anomalies(self):
        self.complete_fixture()
        events = [
            json.loads(line)
            for line in (self.trial / "sync_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        diagnostics = {
            "received": 1000, "expected": 1003, "lost": 3,
            "duplicate": 2, "out_of_order": 1,
            "loss_rate_percent": 0.299103,
            "complete": False, "missing_counter_batch_count": 1,
            "missing_counter_sample_count": 4,
        }
        events.append({
            "event": "recording_saved",
            "trial": {"packet_diagnostics": diagnostics},
        })
        self.write_events(*events)
        (self.trial / "openbci_packets.csv").write_text(
            "Sample Index,Package Counter,Timestamp\n", encoding="utf-8"
        )

        _, report = generate_sync_report(self.trial, {}, "stop")

        self.assertEqual(report["openbci_transport"], diagnostics)
        self.assertEqual(report["source_files"]["openbci_packets"],
                         "openbci_packets.csv")
        self.assertTrue(any("missing packet" in item for item in report["warnings"]))
        self.assertTrue(any("duplicate" in item for item in report["warnings"]))
        self.assertTrue(any("out-of-order" in item for item in report["warnings"]))
        self.assertTrue(any("diagnostics are incomplete" in item
                            for item in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
