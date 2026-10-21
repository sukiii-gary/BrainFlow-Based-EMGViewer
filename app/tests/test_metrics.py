import unittest

import numpy as np

from metrics import PacketLossTracker


class PacketLossTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PacketLossTracker(sample_rate=1000.0)

    def diagnostics(self):
        return self.tracker.get_diagnostics()

    def test_mod256_wrap_is_continuous(self):
        self.tracker.update(np.array([254, 255, 0, 1]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 4)
        self.assertEqual(result["expected"], 4)
        self.assertEqual(result["lost"], 0)
        self.assertEqual(result["transition_count"], 3)

    def test_detects_gap_inside_one_batch(self):
        self.tracker.update(np.array([10, 11, 14, 15]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 4)
        self.assertEqual(result["expected"], 6)
        self.assertEqual(result["lost"], 2)
        self.assertAlmostEqual(result["loss_rate_percent"], 100 * 2 / 6, places=6)

    def test_detects_gap_between_batches(self):
        self.tracker.update(np.array([254, 255]))
        self.tracker.update(np.array([1, 2]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 4)
        self.assertEqual(result["expected"], 5)
        self.assertEqual(result["lost"], 1)

    def test_duplicate_is_not_reported_as_loss(self):
        self.tracker.update(np.array([10, 11, 11, 12]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 4)
        self.assertEqual(result["expected"], 3)
        self.assertEqual(result["duplicate"], 1)
        self.assertEqual(result["lost"], 0)

    def test_backward_counter_is_out_of_order_not_large_loss(self):
        self.tracker.update(np.array([10, 11, 9, 10]))

        result = self.diagnostics()
        self.assertEqual(result["out_of_order"], 1)
        self.assertEqual(result["lost"], 0)
        self.assertEqual(result["expected"], 3)

    def test_half_modulus_delta_is_classified_as_out_of_order(self):
        self.tracker.update(np.array([10, 138]))

        result = self.diagnostics()
        self.assertEqual(result["out_of_order"], 1)
        self.assertEqual(result["lost"], 0)

    def test_timestamp_stall_and_nonmonotonic_count_cross_batches(self):
        self.tracker.update(
            np.array([1, 2]), np.array([100.000, 100.001])
        )
        self.tracker.update(
            np.array([3, 4, 5]), np.array([100.101, 100.100, 100.102])
        )

        result = self.diagnostics()
        self.assertAlmostEqual(result["max_timestamp_stall_ms"], 100.0)
        self.assertEqual(result["nonmonotonic_timestamp_count"], 1)

    def test_reset_clears_totals_and_cross_batch_state(self):
        self.tracker.update(np.array([10, 14]), np.array([1.0, 1.1]))
        self.tracker.reset()
        self.tracker.update(np.array([20]), np.array([2.0]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 1)
        self.assertEqual(result["expected"], 1)
        self.assertEqual(result["lost"], 0)
        self.assertEqual(result["transition_count"], 0)
        self.assertEqual(result["first_counter"], 20)
        self.assertEqual(result["max_timestamp_stall_ms"], 0.0)

    def test_empty_update_does_not_change_diagnostics(self):
        before = self.diagnostics()
        self.tracker.update(np.array([]))
        self.assertEqual(self.diagnostics(), before)

    def test_invalid_counter_does_not_create_false_gap(self):
        self.tracker.update(np.array([10.0, np.nan, 200.0, 201.0]))

        result = self.diagnostics()
        self.assertEqual(result["received"], 4)
        self.assertEqual(result["invalid_counter"], 1)
        self.assertEqual(result["expected"], 3)
        self.assertEqual(result["lost"], 0)

    def test_existing_window_rate_api_remains_compatible(self):
        self.tracker.update(np.array([10, 12]))

        rates = self.tracker.get_rates()
        self.assertEqual(set(rates), {"1s", "60s", "600s"})
        self.assertAlmostEqual(rates["1s"], 100 / 3, places=2)
        self.assertIn("1s:", self.tracker.get_summary())


if __name__ == "__main__":
    unittest.main()
