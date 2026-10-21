import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.psd_view import PsdViewWidget
from ui.time_view import TimeViewWidget


class ChannelPlotLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_time_view_uses_one_plot_per_selected_channel(self):
        view = TimeViewWidget(n_channels=4)
        view.set_active_channels([True, False, True, False])

        self.assertEqual(view._active_channels, (0, 2))
        self.assertIsNotNone(view._plots[0])
        self.assertIsNone(view._plots[1])
        self.assertIsNotNone(view._plots[2])
        self.assertIsNone(view._plots[3])
        self.assertIsNot(view._plots[0], view._plots[2])

    def test_time_view_moves_channel_names_to_compact_side_labels(self):
        view = TimeViewWidget(n_channels=8)

        self.assertTrue(all(plot.titleLabel.text == "" for plot in view._plots))
        self.assertIn("CH1", view._channel_labels[0].text)
        self.assertIn("CH8", view._channel_labels[7].text)
        self.assertLessEqual(view.ci.layout.verticalSpacing(), 1.0)

    def test_time_view_only_shows_bottom_axis_on_last_channel(self):
        view = TimeViewWidget(n_channels=8)

        for channel in range(7):
            self.assertFalse(view._plots[channel].getAxis("bottom").isVisible())
        self.assertTrue(view._plots[7].getAxis("bottom").isVisible())

    def test_time_view_peak_downsamples_dense_traces_but_keeps_full_cache(self):
        view = TimeViewWidget(n_channels=8, display_sec=15.0, fs=1000.0)
        signal = np.zeros(15000)
        signal[7531] = 123.0
        data = [signal.copy() for _ in range(8)]

        view.update_display(data, data)

        rendered = view._curves_raw[0].getData()[1]
        self.assertLessEqual(len(rendered), view.MAX_RENDER_POINTS + 1)
        self.assertIn(123.0, rendered)
        self.assertEqual(len(view._last_raw_data[0]), 15000)

    def test_time_view_clears_all_unselected_channel_panels(self):
        view = TimeViewWidget(n_channels=4)
        view.set_active_channels([False] * 4)

        self.assertEqual(view._active_channels, ())
        self.assertTrue(all(plot is None for plot in view._plots))

    def test_psd_view_removes_unselected_channel_titles(self):
        view = PsdViewWidget(n_channels=4)
        view.set_active_channels([False, True, False, True])
        view.update_display([
            (np.array([]), np.array([])),
            (np.array([10.0]), np.array([1.0])),
            (np.array([]), np.array([])),
            (np.array([20.0]), np.array([2.0])),
        ])

        self.assertEqual(view._active_channels, (1, 3))
        self.assertIsNone(view._plots[0])
        self.assertIsNotNone(view._plots[1])
        self.assertIsNone(view._plots[2])
        self.assertIsNotNone(view._plots[3])

    def test_eight_channel_psd_uses_two_columns_in_narrow_panel(self):
        view = PsdViewWidget(n_channels=8)
        view.resize(420, 320)
        view._setup_ui()

        self.assertEqual(view._active_channels, tuple(range(8)))
        self.assertEqual(view._grid_columns, 2)
        for channel in range(6):
            self.assertFalse(view._plots[channel].getAxis("bottom").isVisible())
        for channel in range(6, 8):
            self.assertTrue(view._plots[channel].getAxis("bottom").isVisible())
        view.close()

    def test_time_view_restores_cached_curves_immediately_after_relayout(self):
        view = TimeViewWidget(n_channels=4, fs=1000.0)
        raw = [np.array([index, index + 1.0]) for index in range(4)]
        envelopes = [values + 10.0 for values in raw]
        view.update_display(raw, envelopes)

        view.set_active_channels([True, False, True, False])

        np.testing.assert_array_equal(view._curves_raw[0].getData()[1], raw[0])
        np.testing.assert_array_equal(view._curves_env[2].getData()[1], envelopes[2])

        view.update_display([raw[0], np.array([]), raw[2], np.array([])], envelopes)
        view.set_active_channels([True, True, True, False])
        np.testing.assert_array_equal(view._curves_raw[1].getData()[1], raw[1])

    def test_psd_view_restores_cached_spectra_immediately_after_relayout(self):
        view = PsdViewWidget(n_channels=4)
        spectra = [
            (np.array([10.0, 20.0]), np.array([index + 1.0, index + 2.0]))
            for index in range(4)
        ]
        view.update_display(spectra)

        view.set_active_channels([False, True, False, True])

        np.testing.assert_array_equal(view._curves[1].getData()[1], spectra[1][1])
        np.testing.assert_array_equal(view._curves[3].getData()[1], spectra[3][1])

    def test_clear_data_removes_cached_frames(self):
        time_view = TimeViewWidget(n_channels=2)
        time_view.update_display(
            [np.array([1.0]), np.array([2.0])],
            [np.array([3.0]), np.array([4.0])],
        )
        psd_view = PsdViewWidget(n_channels=2)
        psd_view.update_display([
            (np.array([10.0]), np.array([1.0])),
            (np.array([20.0]), np.array([2.0])),
        ])

        time_view.clear_data()
        psd_view.clear_data()

        self.assertTrue(all(len(values) == 0 for values in time_view._last_raw_data))
        self.assertTrue(all(len(values[0]) == 0 for values in psd_view._last_psd_data))


if __name__ == "__main__":
    unittest.main()
