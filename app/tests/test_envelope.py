import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from signal_chain import compute_envelope


class EnvelopeTests(unittest.TestCase):
    def test_high_dynamic_range_does_not_sqrt_negative_roundoff(self):
        cfg = SimpleNamespace(
            sampling_rate=1000.0,
            segmentation=SimpleNamespace(
                use_preprocess_chain=False,
                smooth_window_ms=100.0,
            ),
        )
        signal = np.zeros(10000)
        signal[0] = 1e12

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            envelope = compute_envelope(signal, cfg)

        self.assertTrue(np.all(np.isfinite(envelope)))
        self.assertTrue(np.all(envelope >= 0.0))
        self.assertFalse(any("sqrt" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
