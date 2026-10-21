import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from ui.activation_map import (
    AdaptiveActivationNormalizer,
    ActivationMapWidget,
    CHANNEL_POSITIONS,
)


class ActivationMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_channel_positions_follow_supplied_ankle_layout(self):
        self.assertEqual(CHANNEL_POSITIONS[0], (0.50, 0.10))
        self.assertEqual(CHANNEL_POSITIONS[1], (0.22, 0.22))
        self.assertEqual(CHANNEL_POSITIONS[2], (0.10, 0.50))
        self.assertEqual(CHANNEL_POSITIONS[3], (0.22, 0.78))
        self.assertEqual(CHANNEL_POSITIONS[4], (0.50, 0.90))
        self.assertEqual(CHANNEL_POSITIONS[5], (0.78, 0.78))
        self.assertEqual(CHANNEL_POSITIONS[6], (0.90, 0.50))
        self.assertEqual(CHANNEL_POSITIONS[7], (0.78, 0.22))

    def test_normalization_is_independent_for_each_channel(self):
        normalizer = AdaptiveActivationNormalizer(n_channels=2)
        for _ in range(20):
            normalizer.update([10.0, 100.0])

        activation = normalizer.update([40.0, 400.0])

        self.assertGreater(activation[0], 0.9)
        self.assertGreater(activation[1], 0.9)

    def test_invalid_rms_keeps_previous_channel_value(self):
        normalizer = AdaptiveActivationNormalizer(n_channels=2)
        for _ in range(20):
            normalizer.update([10.0, 100.0])
        before = normalizer.update([40.0, 400.0])

        after = normalizer.update([np.nan, np.inf])

        np.testing.assert_array_equal(after, before)

    def test_ch1_hotspot_is_warmer_than_opposite_ch5(self):
        widget = ActivationMapWidget(n_channels=8)
        widget._activation = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        image = widget._render_heatmap()
        size = widget.IMAGE_SIZE - 1
        ch1 = image.pixelColor(int(0.50 * size), int(0.10 * size))
        ch5 = image.pixelColor(int(0.50 * size), int(0.90 * size))

        self.assertGreater(ch1.red(), ch5.red())
        self.assertLess(ch1.blue(), ch5.blue())
        widget.close()

    def test_clear_display_can_reset_normalization_history(self):
        widget = ActivationMapWidget(n_channels=8)
        for _ in range(20):
            widget.update_rms([10.0] * 8)
        widget.update_rms([40.0] * 8)
        self.assertTrue(np.all(widget.activation > 0.9))

        widget.clear_display(reset_history=True)

        np.testing.assert_array_equal(widget.activation, np.zeros(8))
        widget.close()

    def test_colorbar_paints_with_fractional_layout_coordinates(self):
        widget = ActivationMapWidget(n_channels=8)
        canvas = QImage(800, 300, QImage.Format.Format_ARGB32)
        painter = QPainter(canvas)

        widget._paint_colorbar(painter, QRectF(200.5, 25.5, 240.5, 240.5))

        painter.end()
        self.assertFalse(canvas.isNull())
        widget.close()

    def test_narrow_activation_map_paints_without_error(self):
        widget = ActivationMapWidget(n_channels=8)
        widget.resize(400, 300)
        canvas = QImage(widget.size(), QImage.Format.Format_ARGB32)
        canvas.fill(0)

        widget.render(canvas)

        self.assertFalse(canvas.isNull())
        widget.close()


if __name__ == "__main__":
    unittest.main()
