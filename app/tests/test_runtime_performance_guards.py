import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config_viewer import ViewerConfig
from signal_chain import SignalChain


class RuntimePerformanceGuardTests(unittest.TestCase):
    def test_default_recording_directory_is_inside_project(self):
        expected = Path(__file__).resolve().parents[1] / "recordings"
        self.assertEqual(Path(ViewerConfig().record_dir), expected)

    def test_signal_cache_drops_previous_data_version(self):
        chain = SignalChain(fs=1000, n_channels=2, buffer_sec=60)
        chain.push_samples(np.ones((2, 2000)))
        chain.get_filtered(0, n_samples=1000)
        chain.get_envelope(0, n_samples=1000)
        chain.get_psd(0, n_samples=1000)
        self.assertTrue(chain._cached_filtered)

        chain.push_samples(np.ones((2, 40)))

        self.assertEqual(chain._cached_filtered, {})
        self.assertEqual(chain._cached_envelope, {})
        self.assertEqual(chain._cached_psd, {})

    def test_filtered_window_does_not_expand_to_full_ring_buffer(self):
        chain = SignalChain(fs=1000, n_channels=1, buffer_sec=60)
        chain.push_samples(np.random.default_rng(1).normal(size=(1, 60000)))

        filtered = chain.get_filtered(0, n_samples=15000)

        self.assertEqual(len(filtered), 15000)

    def test_larger_filtered_window_covers_shorter_metric_windows(self):
        chain = SignalChain(fs=1000, n_channels=1, buffer_sec=60)
        chain.push_samples(np.random.default_rng(2).normal(size=(1, 15000)))

        with patch("signal_chain.apply_filters", wraps=__import__(
            "signal_chain"
        ).apply_filters) as apply_mock:
            long_window = chain.get_filtered(0, n_samples=15000)
            for size in (2000, 1000, 500):
                actual = chain.get_filtered(0, n_samples=size)
                np.testing.assert_array_equal(actual, long_window[-size:])

        self.assertEqual(apply_mock.call_count, 1)
        self.assertEqual(len(chain._cached_filtered), 1)

    def test_smaller_filtered_window_does_not_cover_larger_window(self):
        chain = SignalChain(fs=1000, n_channels=1, buffer_sec=60)
        chain.push_samples(np.random.default_rng(3).normal(size=(1, 15000)))

        with patch("signal_chain.apply_filters", wraps=__import__(
            "signal_chain"
        ).apply_filters) as apply_mock:
            chain.get_filtered(0, n_samples=500)
            chain.get_filtered(0, n_samples=2000)

        self.assertEqual(apply_mock.call_count, 2)

    def test_new_samples_invalidate_covering_window(self):
        chain = SignalChain(fs=1000, n_channels=1, buffer_sec=60)
        chain.push_samples(np.random.default_rng(4).normal(size=(1, 15000)))

        with patch("signal_chain.apply_filters", wraps=__import__(
            "signal_chain"
        ).apply_filters) as apply_mock:
            chain.get_filtered(0, n_samples=15000)
            chain.push_samples(np.ones((1, 40)))
            chain.get_filtered(0, n_samples=500)

        self.assertEqual(apply_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
