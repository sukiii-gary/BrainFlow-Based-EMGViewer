import json
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from board_manager import CytonManager


REGISTER_REPORT = """Board ADS Registers
CH1SET, 05, 68, 0, 1, 1, 0, 1, 0, 0, 0
CH2SET, 06, 60, 0, 1, 1, 0, 0, 0, 0, 0
BIAS_SENSP, 0D, 03, 1, 1, 0, 0, 0, 0, 0, 0
BIAS_SENSN, 0E, 03, 1, 1, 0, 0, 0, 0, 0, 0
$$$"""

WIFI_REGISTER_REPORT = """Board ADS Registers
CH1SET, 05, 68, 0, 1, 1, 0, 1, 0, 0, 0
CH2SET, 06, 60, 0, 1, 1, 0, 0, 0, 0, 0
BIAS_SENSP, 0D, 03, 1, 1, 0, 0, 0, 0, 0, 0
BIAS_SENSN, 0E, 03, 1, 1, 0, 0, 0, 0, 0, 0

LIS3DH Registers
0x20 37
0x3D 00"""


class FakeBoard:
    def __init__(self, query_response=REGISTER_REPORT, command_response="Success$$$"):
        self.query_response = query_response
        self.command_response = command_response
        self.commands = []
        self.started = False

    def config_board(self, command):
        self.commands.append(command)
        return self.query_response if command == "?" else self.command_response

    def start_stream(self):
        self.started = True


class BoardManagerTests(unittest.TestCase):
    @patch("board_manager.BoardShim")
    def test_synthetic_board_uses_first_eight_channels(self, board_shim):
        board_shim.get_eeg_channels.return_value = list(range(1, 17))
        board_shim.get_timestamp_channel.return_value = 30
        board_shim.get_sampling_rate.return_value = 250
        manager = CytonManager(board_id=-1)

        self.assertTrue(manager.connect())
        self.assertEqual(manager.n_channels, 8)

    def test_wifi_sample_rate_defaults_to_1000_hz(self):
        manager = CytonManager(board_id=5)

        self.assertEqual(manager.sample_rate, 1000.0)

    def test_configures_static_wifi_ipv4_only_while_disconnected(self):
        manager = CytonManager(board_id=5)

        manager.configure_wifi_address(" 192.168.10.42 ")
        self.assertEqual(manager.wifi_ip, "192.168.10.42")

        with self.assertRaises(ValueError):
            manager.configure_wifi_address("openbci.local")
        with self.assertRaises(ValueError):
            manager.configure_wifi_address("0.0.0.0")

        manager._is_connected = True
        with self.assertRaises(RuntimeError):
            manager.configure_wifi_address("192.168.10.43")

    @patch("board_manager.urllib.request.build_opener")
    def test_wifi_board_probe_uses_static_ip_and_bypasses_proxy(self, build_opener):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"board_connected":true}'
        build_opener.return_value.open.return_value = response
        manager = CytonManager(board_id=5, wifi_ip="192.168.10.42")

        payload = manager._probe_wifi_board()

        request = build_opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://192.168.10.42/board")
        self.assertEqual(payload, '{"board_connected":true}')

    def test_data_stream_verification_requires_advancing_packet_counter(self):
        manager = CytonManager(board_id=5)
        manager._board = MagicMock()
        manager._board.get_board_data.return_value = np.array([
            [10, 11, 12],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        manager._eeg_channels = [1, 2]
        manager._is_streaming = True

        data = manager.get_new_data()

        self.assertEqual(data.shape, (2, 3))
        self.assertTrue(manager.data_stream_verified)

    def test_get_new_data_keeps_brainflow_packet_timestamps(self):
        manager = CytonManager(board_id=5)
        manager._board = MagicMock()
        manager._board.get_board_data.return_value = np.array([
            [10, 11],
            [1.0, 2.0],
            [100.0, 100.001],
        ])
        manager._eeg_channels = [1]
        manager._timestamp_channel = 2
        manager._is_streaming = True

        manager.get_new_data()

        np.testing.assert_array_equal(
            manager._last_timestamps, np.array([100.0, 100.001])
        )

    @patch.object(CytonManager, "_probe_wifi_board", return_value='{"board_connected":true}')
    @patch.object(CytonManager, "_get_local_ip_for_wifi", return_value="192.168.10.5")
    @patch("board_manager.WiFiUDPBoard")
    @patch("board_manager.BoardShim")
    def test_rejects_wifi_connection_when_actual_rate_is_not_1000(
        self, board_shim, udp_board, _local_ip, _probe
    ):
        board_shim.get_eeg_channels.return_value = list(range(1, 9))
        board_shim.get_sampling_rate.return_value = 500
        manager = CytonManager(board_id=5, wifi_ip="192.168.10.42")

        connected = manager.connect()

        self.assertFalse(connected)
        self.assertEqual(manager.last_connection_status["stage"], "brainflow")
        self.assertIn("期望 1000Hz", manager.last_connection_status["detail"])
        udp_board.return_value.release_session.assert_called_once()

    def test_builds_srb2_off_command(self):
        command = CytonManager.build_channel_command(
            1, {"gain": 6, "bias_on": True, "srb2_on": False}
        )
        self.assertEqual(command, "x1060100X")

    def test_parses_real_register_report_format(self):
        parsed = CytonManager.parse_register_report(REGISTER_REPORT)

        self.assertTrue(parsed["complete"])
        self.assertEqual(parsed["channels"][1]["gain"], 6)
        self.assertTrue(parsed["channels"][1]["srb2"])
        self.assertEqual(parsed["channels"][1]["register"], "0x68")
        self.assertFalse(parsed["channels"][2]["srb2"])
        self.assertEqual(parsed["channels"][2]["register"], "0x60")
        self.assertTrue(parsed["channels"][1]["bias"])
        self.assertTrue(parsed["channels"][2]["bias"])

    def test_requires_complete_register_report(self):
        parsed = CytonManager.parse_register_report(REGISTER_REPORT.removesuffix("$$$"))
        self.assertFalse(parsed["complete"])

    def test_accepts_trailing_control_bytes_after_register_terminator(self):
        parsed = CytonManager.parse_register_report(REGISTER_REPORT + "\x00\r\nOK")

        self.assertTrue(parsed["complete"])
        self.assertEqual(parsed["completion"], "dollar_eot")
        self.assertFalse(parsed["channels"][2]["srb2"])

    def test_accepts_wifi_report_ending_at_last_lis3dh_register(self):
        parsed = CytonManager.parse_register_report(WIFI_REGISTER_REPORT)

        self.assertTrue(parsed["complete"])
        self.assertEqual(parsed["completion"], "lis3dh_tail")
        self.assertFalse(parsed["channels"][2]["srb2"])

    def test_rejects_wifi_report_truncated_before_last_lis3dh_register(self):
        parsed = CytonManager.parse_register_report(
            WIFI_REGISTER_REPORT.removesuffix("0x3D 00")
        )

        self.assertFalse(parsed["complete"])

    @patch("board_manager.time.sleep", return_value=None)
    def test_apply_and_verify_reports_verified(self, _sleep):
        manager = CytonManager(board_id=0)
        manager._board = FakeBoard()
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": True},
            {"channel": 2, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "verified")
        self.assertEqual(manager._board.commands, ["x1060110X", "x2060100X", "?"])

    @patch.object(CytonManager, "_send_wifi_command_http", return_value="")
    @patch.object(CytonManager, "_read_wifi_registers_http", return_value="")
    @patch("board_manager.time.sleep", return_value=None)
    def test_wifi_empty_response_is_sent_unverified(
        self, _sleep, _http_readback, _http_send
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard(query_response="", command_response="")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "sent_unverified")

    @patch.object(
        CytonManager,
        "_read_wifi_registers_http",
        return_value=WIFI_REGISTER_REPORT,
    )
    @patch.object(CytonManager, "_send_wifi_command_http", return_value="")
    @patch("board_manager.time.sleep", return_value=None)
    def test_wifi_http_verifies_without_duplicate_brainflow_query(
        self, _sleep, http_send, _http_readback
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard(query_response="", command_response="")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": True},
            {"channel": 2, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "verified")
        self.assertEqual(report["source"], "wifi_http")
        self.assertEqual(manager._board.commands, [])
        self.assertEqual(
            http_send.call_args_list,
            [call("x1060110X"), call("x2060100X")],
        )

    @patch.object(
        CytonManager, "_send_wifi_command_http", side_effect=TimeoutError("timed out")
    )
    @patch.object(CytonManager, "_read_wifi_registers_http")
    @patch("board_manager.time.sleep", return_value=None)
    def test_wifi_command_failure_stops_without_querying_remaining_channels(
        self, _sleep, http_readback, http_send
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard()
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": True},
            {"channel": 2, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "failed")
        http_send.assert_called_once_with("x1060110X")
        http_readback.assert_not_called()

    def test_decodes_json_wrapped_wifi_http_report(self):
        payload = json.dumps({"response": REGISTER_REPORT}).encode("utf-8")

        response = CytonManager._decode_wifi_http_response(payload)

        self.assertEqual(response, REGISTER_REPORT)

    @patch("board_manager.urllib.request.build_opener")
    def test_wifi_http_readback_posts_query_command(self, build_opener):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = REGISTER_REPORT.encode("utf-8")
        opener = build_opener.return_value
        opener.open.return_value = response
        manager = CytonManager(board_id=5, wifi_ip="192.168.4.1")

        report = manager._read_wifi_registers_http()

        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://192.168.4.1/command")
        self.assertEqual(json.loads(request.data), {"command": "?"})
        self.assertEqual(report, REGISTER_REPORT)

    @patch.object(CytonManager, "_send_wifi_command_http", return_value="")
    @patch.object(CytonManager, "_read_wifi_registers_http", side_effect=TimeoutError("timed out"))
    @patch("board_manager.time.sleep", return_value=None)
    def test_wifi_http_timeout_is_sent_unverified(
        self, _sleep, _http_readback, _http_send
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard(query_response="", command_response="")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "sent_unverified")
        self.assertIn("timed out", report["diagnostic"])

    @patch.object(CytonManager, "_send_wifi_command_http", return_value="")
    @patch.object(CytonManager, "_read_wifi_registers_http", return_value=REGISTER_REPORT)
    @patch("board_manager.time.sleep", return_value=None)
    def test_wifi_http_register_mismatch_is_failed(
        self, _sleep, _http_readback, _http_send
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard(query_response="", command_response="")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "failed")
        self.assertFalse(report["channels"][1]["matches"])

    @patch("board_manager.time.sleep", return_value=None)
    def test_explicit_board_failure_blocks_verification(self, _sleep):
        manager = CytonManager(board_id=0)
        manager._board = FakeBoard(command_response="Failure: too few chars$$$")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        report = manager.apply_and_verify_settings()

        self.assertEqual(report["state"], "failed")

    @patch.object(CytonManager, "_send_wifi_command_http", return_value="")
    @patch.object(CytonManager, "_read_wifi_registers_http", return_value="")
    @patch("board_manager.time.sleep", return_value=None)
    def test_start_stream_allows_wifi_unverified_state(
        self, _sleep, _http_readback, _http_send
    ):
        manager = CytonManager(board_id=5)
        manager._board = FakeBoard(query_response="", command_response="")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        self.assertTrue(manager.start_stream())
        self.assertTrue(manager._board.started)

    @patch("board_manager.time.sleep", return_value=None)
    def test_start_stream_rejects_explicit_failure(self, _sleep):
        manager = CytonManager(board_id=0)
        manager._board = FakeBoard(command_response="Failure: too few chars$$$")
        manager.set_channel_settings([
            {"channel": 1, "gain": 6, "bias_on": True, "srb2_on": False},
        ])

        self.assertFalse(manager.start_stream())
        self.assertFalse(manager._board.started)


if __name__ == "__main__":
    unittest.main()
