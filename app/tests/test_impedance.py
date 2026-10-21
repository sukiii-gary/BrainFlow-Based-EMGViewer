import math
import threading
import unittest
from unittest.mock import patch

import numpy as np

from board_manager import CytonManager
from impedance import classify_impedance_ohms, estimate_impedance_ohms


def register_report(gain=6, bias=True, srb2=True):
    gain_bits = {0: 0b000, 1: 0b001, 2: 0b010, 3: 0b011,
                 4: 0b100, 5: 0b101, 6: 0b110}[gain]
    register = (gain_bits << 4) | (int(srb2) << 3)
    bias_byte = 0xFF if bias else 0x00
    lines = ["Board ADS Registers"]
    for channel in range(1, 9):
        lines.append(
            f"CH{channel}SET, {channel + 4:02X}, {register:02X}, 0, 1, 1, 0, "
            f"{int(srb2)}, 0, 0, 0"
        )
    lines.extend([
        f"BIAS_SENSP, 0D, {bias_byte:02X}, 1, 1, 1, 1, 1, 1, 1, 1",
        f"BIAS_SENSN, 0E, {bias_byte:02X}, 1, 1, 1, 1, 1, 1, 1, 1",
        "$$$",
    ])
    return "\n".join(lines)


class FakeBoardShim:
    def __init__(self, query_response=None):
        self.query_response = query_response or register_report()
        self.commands = []
        self.started = 0

    def config_board(self, command):
        self.commands.append(command)
        return self.query_response if command == "?" else "Success$$$"

    def start_stream(self):
        self.started += 1

    def stop_stream(self):
        pass

    def get_board_data(self):
        return np.zeros((10, 0))


def connected_manager(query_response=None):
    manager = CytonManager(board_id=0)
    manager._board = FakeBoardShim(query_response)
    manager._is_connected = True
    manager._eeg_channels = list(range(1, 9))
    manager._sample_rate = 1000.0
    return manager


class ImpedanceCalculationTests(unittest.TestCase):
    def test_official_formula_recovers_known_impedance(self):
        expected = 100_000.0
        std_uv = (
            (expected + 2200.0) * 6.0e-9 / math.sqrt(2.0) * 1.0e6
        )
        samples = np.tile([-std_uv, std_uv], 100)

        result = estimate_impedance_ohms(samples)

        self.assertAlmostEqual(result["impedance_ohms"], expected, places=6)
        self.assertEqual(result["sample_count"], 200)

    def test_nonfinite_values_are_ignored_and_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_impedance_ohms([1.0, np.nan, np.inf])

    def test_emg_goal_and_cyton_gui_reference_are_independent(self):
        target = classify_impedance_ohms(9_999)
        self.assertEqual(target["emg_contact"], "target")
        self.assertEqual(target["cyton_gui_reference"], "green")

        wide_gui_green = classify_impedance_ohms(700_000)
        self.assertEqual(wide_gui_green["cyton_gui_reference"], "green")
        self.assertEqual(wide_gui_green["emg_contact"], "poor")

    def test_cyton_gui_boundaries_are_deterministic(self):
        self.assertEqual(
            classify_impedance_ohms(750_000)["cyton_gui_reference"], "yellow"
        )
        self.assertEqual(
            classify_impedance_ohms(2_500_000)["cyton_gui_reference"], "red"
        )
        self.assertTrue(classify_impedance_ohms(5_000_000)["likely_open_circuit"])


class ImpedanceBoardTests(unittest.TestCase):
    def test_builds_corrected_official_n_input_commands(self):
        self.assertEqual(
            CytonManager.build_impedance_command(1),
            "x1000010Xz101Z",
        )
        self.assertEqual(
            CytonManager.build_impedance_command(8, active=False),
            "z800Z",
        )

    @patch("board_manager.time.sleep", return_value=None)
    def test_selected_channels_are_serial_and_settings_are_restored(self, _sleep):
        manager = connected_manager()
        samples = np.tile([-500.0, 500.0], 1000)
        progress = []
        with patch.object(
            manager, "_collect_impedance_samples", return_value=samples
        ) as collect:
            report = manager.measure_impedances(
                [2, 1], progress_callback=lambda *args: progress.append(args)
            )

        self.assertEqual([item["channel"] for item in report["results"]], [2, 1])
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(report["restore"]["state"], "verified")
        self.assertTrue(manager.impedance_restore_verified)
        commands = manager._board.commands
        first_active = commands.index("x2000010Xz201Z")
        for channel in range(1, 9):
            self.assertLess(commands.index(f"z{channel}00Z"), first_active)
        self.assertIn("x2060110Xz200Z", commands)
        self.assertLess(commands.index("x2060110Xz200Z"), commands.index("x1000010Xz101Z"))
        for channel in range(1, 9):
            self.assertIn(f"z{channel}00Z", commands)
        self.assertEqual(commands[-1], "?")
        self.assertEqual(
            [state for _, state, _ in progress],
            ["configuring", "measuring", "complete"] * 2,
        )

    @patch("board_manager.time.sleep", return_value=None)
    def test_cancel_still_restores_without_returning_partial_measurement(self, _sleep):
        manager = connected_manager()
        cancel = threading.Event()

        def cancelled(*_args):
            cancel.set()
            return np.array([])

        with patch.object(manager, "_collect_impedance_samples", side_effect=cancelled):
            report = manager.measure_impedances([1, 2], cancel_event=cancel)

        self.assertTrue(report["cancelled"])
        self.assertEqual(report["results"], [])
        self.assertEqual(report["restore"]["state"], "verified")
        self.assertNotIn("x2000010Xz201Z", manager._board.commands)

    @patch("board_manager.time.sleep", return_value=None)
    def test_restore_mismatch_locks_normal_acquisition(self, _sleep):
        manager = connected_manager(register_report(gain=0, bias=False, srb2=True))
        samples = np.tile([-500.0, 500.0], 1000)
        with patch.object(
            manager, "_collect_impedance_samples", return_value=samples
        ):
            report = manager.measure_impedances([1])

        self.assertEqual(report["restore"]["state"], "failed")
        self.assertFalse(manager.impedance_restore_verified)
        self.assertFalse(manager.start_stream(settings_prepared=True))
        self.assertEqual(manager._board.started, 0)

    @patch("board_manager.time.sleep", return_value=None)
    def test_measurement_error_still_runs_full_restore(self, _sleep):
        manager = connected_manager()
        with patch.object(
            manager, "_collect_impedance_samples", side_effect=OSError("link lost")
        ):
            with self.assertRaisesRegex(OSError, "link lost"):
                manager.measure_impedances([1])

        self.assertEqual(manager.get_last_readback()["state"], "verified")
        self.assertTrue(manager.impedance_restore_verified)
        self.assertIn("z100Z", manager._board.commands)
        self.assertEqual(manager._board.commands[-1], "?")

    @patch("board_manager.time.sleep", return_value=None)
    def test_stream_stop_failure_restores_all_settings_and_locks_acquisition(self, _sleep):
        manager = connected_manager()
        with patch.object(
            manager._board, "stop_stream", side_effect=OSError("stop failed")
        ):
            with self.assertRaisesRegex(OSError, "stop failed"):
                manager.measure_impedances(
                    [1], duration_sec=0.001, settle_sec=0.0
                )

        commands = manager._board.commands
        for channel in range(1, 9):
            self.assertIn(f"z{channel}00Z", commands)
            self.assertIn(f"x{channel}060110X", commands)
        self.assertEqual(manager.get_last_readback()["state"], "failed")
        self.assertFalse(manager.impedance_restore_verified)

        starts_before = manager._board.started
        self.assertFalse(manager.start_stream(settings_prepared=True))
        self.assertEqual(manager._board.started, starts_before)


if __name__ == "__main__":
    unittest.main()
