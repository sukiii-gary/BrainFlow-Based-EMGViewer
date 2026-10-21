"""Real-time eight-channel ankle-band activation map."""
from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import QLineF, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor, QFont, QImage, QLinearGradient, QPainter, QPen,
)
from PyQt6.QtWidgets import QWidget


# Normalized screen coordinates copied from the supplied cross-section layout.
# CH1 is anterior/top; CH2..CH8 continue counter-clockwise around the band.
CHANNEL_POSITIONS = (
    (0.50, 0.10),
    (0.22, 0.22),
    (0.10, 0.50),
    (0.22, 0.78),
    (0.50, 0.90),
    (0.78, 0.78),
    (0.90, 0.50),
    (0.78, 0.22),
)

_COLOR_STOPS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
_COLOR_VALUES = np.array([
    (30, 64, 175),
    (40, 180, 220),
    (58, 190, 120),
    (250, 210, 55),
    (210, 45, 45),
], dtype=float)


class AdaptiveActivationNormalizer:
    """Normalize each channel against its own recent robust RMS range."""

    def __init__(self, n_channels: int = 8, history_size: int = 150):
        self.n_channels = n_channels
        self._history = [
            deque(maxlen=history_size) for _ in range(n_channels)
        ]
        self._last = np.zeros(n_channels, dtype=float)

    def update(self, rms_values) -> np.ndarray:
        values = np.asarray(rms_values, dtype=float).ravel()
        result = self._last.copy()
        for channel in range(min(self.n_channels, len(values))):
            value = float(values[channel])
            if not np.isfinite(value) or value < 0.0:
                continue
            history = self._history[channel]
            history.append(value)
            samples = np.asarray(history, dtype=float)
            low = float(np.percentile(samples, 10.0))
            high = float(np.percentile(samples, 95.0))
            minimum_span = max(1.0, 0.25 * max(low, 1.0))
            result[channel] = np.clip(
                (value - low) / max(high - low, minimum_span), 0.0, 1.0
            )
        self._last = result
        return result.copy()

    def reset(self) -> None:
        for history in self._history:
            history.clear()
        self._last.fill(0.0)


class ActivationMapWidget(QWidget):
    """Paint a smooth activation field using the physical channel layout."""

    IMAGE_SIZE = 220

    def __init__(self, parent=None, n_channels: int = 8):
        super().__init__(parent)
        self.n_channels = min(int(n_channels), len(CHANNEL_POSITIONS))
        self._normalizer = AdaptiveActivationNormalizer(self.n_channels)
        self._activation = np.zeros(self.n_channels, dtype=float)
        self._has_data = False
        self._weights, self._mask = self._build_spatial_weights()
        self._image = self._render_heatmap()
        self.setMinimumHeight(210)
        self.setToolTip(
            "每个通道按自身最近 30 秒 RMS 范围归一化；该数值不是 MVC 百分比"
        )

    def sizeHint(self) -> QSize:
        return QSize(420, 300)

    def update_rms(self, rms_values) -> None:
        self._activation = self._normalizer.update(rms_values)
        self._has_data = np.isfinite(np.asarray(rms_values, dtype=float)).any()
        self._image = self._render_heatmap()
        self.update()

    def clear_display(self, reset_history: bool = False) -> None:
        if reset_history:
            self._normalizer.reset()
        self._activation.fill(0.0)
        self._has_data = False
        self._image = self._render_heatmap()
        self.update()

    @property
    def activation(self) -> np.ndarray:
        return self._activation.copy()

    def _build_spatial_weights(self) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.linspace(0.0, 1.0, self.IMAGE_SIZE)
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        sigma = 0.14
        weights = []
        for x, y in CHANNEL_POSITIONS[:self.n_channels]:
            distance_sq = (grid_x - x) ** 2 + (grid_y - y) ** 2
            weights.append(np.exp(-distance_sq / (2.0 * sigma ** 2)))
        radius = np.sqrt((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2)
        return np.asarray(weights), radius <= 0.48

    def _render_heatmap(self) -> QImage:
        denominator = np.sum(self._weights, axis=0) + 1e-12
        heat = np.tensordot(self._activation, self._weights, axes=(0, 0))
        heat = np.clip(heat / denominator, 0.0, 1.0)
        rgb = np.empty((*heat.shape, 3), dtype=np.uint8)
        for component in range(3):
            rgb[..., component] = np.interp(
                heat, _COLOR_STOPS, _COLOR_VALUES[:, component]
            ).astype(np.uint8)
        rgba = np.empty((*heat.shape, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(self._mask, 235, 0).astype(np.uint8)
        return QImage(
            rgba.data,
            self.IMAGE_SIZE,
            self.IMAGE_SIZE,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().window())

        narrow = self.width() < 520
        if narrow:
            content = self.rect().adjusted(72, 30, -72, -88)
        else:
            content = self.rect().adjusted(58, 18, -150, -18)
        diameter = max(80.0, min(float(content.width()), float(content.height())))
        map_rect = QRectF(
            content.center().x() - diameter / 2.0,
            content.center().y() - diameter / 2.0,
            diameter,
            diameter,
        )
        painter.drawImage(map_rect, self._image)
        painter.setPen(QPen(self.palette().mid().color(), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(map_rect)

        center = map_rect.center()
        marker_radius = max(3.5, diameter * 0.022)
        label_radius = diameter * (0.61 if not narrow else 0.64)
        label_font = QFont(self.font())
        label_font.setPointSize(9)
        label_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(label_font)
        for channel, (x, y) in enumerate(CHANNEL_POSITIONS[:self.n_channels]):
            marker_x = map_rect.left() + x * diameter
            marker_y = map_rect.top() + y * diameter
            painter.setPen(QPen(QColor(35, 35, 35), 1.0))
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawEllipse(
                QRectF(
                    marker_x - marker_radius,
                    marker_y - marker_radius,
                    marker_radius * 2.0,
                    marker_radius * 2.0,
                )
            )

            direction_x = (x - 0.5) / 0.4
            direction_y = (y - 0.5) / 0.4
            label_x = center.x() + direction_x * label_radius
            label_y = center.y() + direction_y * label_radius
            value = f"{self._activation[channel] * 100:.0f}%" if self._has_data else "--"
            painter.setPen(self.palette().text().color())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(
                QRectF(label_x - 36.0, label_y - 10.0, 72.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                f"CH{channel + 1}  {value}",
            )

        self._paint_colorbar(painter, map_rect, horizontal=narrow)
        if not self._has_data:
            waiting_font = QFont(self.font())
            waiting_font.setPointSize(10)
            painter.setFont(waiting_font)
            painter.setPen(self.palette().text().color())
            painter.drawText(
                map_rect,
                Qt.AlignmentFlag.AlignCenter,
                "等待数据",
            )
        painter.end()

    def _paint_colorbar(
        self, painter: QPainter, map_rect: QRectF, horizontal: bool = False
    ) -> None:
        if horizontal:
            self._paint_horizontal_colorbar(painter, map_rect)
            return
        bar_height = map_rect.height() * 0.72
        available_right = max(52.0, self.width() - map_rect.right())
        bar_left = min(
            map_rect.right() + max(18.0, available_right - 68.0),
            self.width() - 68.0,
        )
        bar = QRectF(
            bar_left,
            map_rect.center().y() - bar_height / 2.0,
            16.0,
            bar_height,
        )
        gradient = QLinearGradient(bar.topLeft(), bar.bottomLeft())
        colors = [QColor(*map(int, color)) for color in _COLOR_VALUES]
        for position, color in zip(_COLOR_STOPS, colors):
            gradient.setColorAt(1.0 - float(position), color)
        painter.setPen(QPen(self.palette().mid().color(), 1.0))
        painter.setBrush(gradient)
        painter.drawRect(bar)
        painter.setPen(self.palette().text().color())
        small_font = QFont(self.font())
        small_font.setPointSize(8)
        painter.setFont(small_font)
        for ratio, text in ((0.0, "100%"), (0.5, "50%"), (1.0, "0%")):
            y = bar.top() + ratio * bar.height()
            painter.drawLine(QLineF(bar.right(), y, bar.right() + 4.0, y))
            painter.drawText(
                QRectF(bar.right() + 7.0, y - 9.0, 45.0, 18.0),
                Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        painter.drawText(
            QRectF(bar.left() - 20.0, bar.top() - 24.0, 75.0, 18.0),
            Qt.AlignmentFlag.AlignCenter,
            "相对激活",
        )

    def _paint_horizontal_colorbar(self, painter: QPainter, map_rect: QRectF) -> None:
        bar_width = min(map_rect.width(), max(120.0, self.width() - 120.0))
        bar = QRectF(
            (self.width() - bar_width) / 2.0,
            self.height() - 37.0,
            bar_width,
            12.0,
        )
        gradient = QLinearGradient(bar.topLeft(), bar.topRight())
        colors = [QColor(*map(int, color)) for color in _COLOR_VALUES]
        for position, color in zip(_COLOR_STOPS, colors):
            gradient.setColorAt(float(position), color)
        painter.setPen(QPen(self.palette().mid().color(), 1.0))
        painter.setBrush(gradient)
        painter.drawRect(bar)
        painter.setPen(self.palette().text().color())
        small_font = QFont(self.font())
        small_font.setPointSize(8)
        painter.setFont(small_font)
        for ratio, text, alignment in (
            (0.0, "0%", Qt.AlignmentFlag.AlignLeft),
            (0.5, "50%", Qt.AlignmentFlag.AlignCenter),
            (1.0, "100%", Qt.AlignmentFlag.AlignRight),
        ):
            x = bar.left() + ratio * bar.width()
            painter.drawLine(QLineF(x, bar.bottom(), x, bar.bottom() + 3.0))
            painter.drawText(
                QRectF(x - 24.0, bar.bottom() + 3.0, 48.0, 16.0),
                alignment,
                text,
            )
        painter.drawText(
            QRectF(bar.left(), bar.top() - 18.0, bar.width(), 16.0),
            Qt.AlignmentFlag.AlignCenter,
            "相对激活",
        )
