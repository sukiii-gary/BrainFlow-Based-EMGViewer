"""紧凑 8 通道 PSD 视图，实时频谱监控。

用途：定性扫一眼频谱是否正常，不计算量化指标。
- 50Hz 位置标记竖线，工频泄漏一目了然
- 与 time_view 共享通道配色
"""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt


class PsdViewWidget(pg.GraphicsLayoutWidget):
    """根据可用宽度排列通道的紧凑 PSD 网格视图。"""

    COLORS = [
        (31, 119, 180),   # 蓝
        (255, 127, 14),   # 橙
        (44, 160, 44),    # 绿
        (214, 39, 40),    # 红
        (148, 103, 189),  # 紫
        (140, 86, 75),    # 棕
        (227, 119, 194),  # 粉
        (127, 127, 127),  # 灰
    ]

    NOTCH_HZ = 50.0

    def __init__(self, parent=None, n_channels: int = 8):
        super().__init__(parent=parent)
        self.n_channels = n_channels
        self._curves: list[pg.PlotDataItem] = []
        self._notch_lines: list[pg.InfiniteLine] = []
        self._plots: list[pg.PlotItem] = []
        self._freq_labels: list[pg.TextItem] = []
        self._last_psd_data = [
            (np.array([]), np.array([])) for _ in range(n_channels)
        ]
        self._active_channels = tuple(range(n_channels))
        self._grid_columns = 1

        self.setBackground("w")
        self._setup_ui()

    def _column_count(self) -> int:
        active_count = len(self._active_channels)
        if active_count <= 2:
            return max(1, active_count)
        if active_count <= 4:
            return 2
        return 2

    def _setup_ui(self):
        """根据当前活动通道数量创建紧凑 PSD 网格。"""
        self.clear()
        self.ci.layout.setContentsMargins(3, 3, 3, 3)
        self.ci.layout.setHorizontalSpacing(3)
        self.ci.layout.setVerticalSpacing(2)
        self._curves = [None] * self.n_channels
        self._notch_lines = [None] * self.n_channels
        self._plots = [None] * self.n_channels
        n_cols = self._column_count()
        self._grid_columns = n_cols
        row_count = max(1, (len(self._active_channels) + n_cols - 1) // n_cols)
        for position, ch in enumerate(self._active_channels):
            row, col = divmod(position, n_cols)
            plot = self.addPlot(row=row, col=col)
            plot.setTitle(f"CH{ch+1}", color=self.COLORS[ch], size="8pt")
            plot.showGrid(x=True, y=True, alpha=0.2)
            left_axis = plot.getAxis("left")
            left_axis.setStyle(showValues=False)
            left_axis.setWidth(8)
            bottom_axis = plot.getAxis("bottom")
            if row == row_count - 1:
                plot.setLabel("bottom", "Hz", **{"font-size": "8pt"})
            else:
                bottom_axis.hide()

            # X 范围: 0–250 Hz
            plot.setXRange(0, 250, padding=0)
            plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

            r, g, b = self.COLORS[ch]
            curve = plot.plot(pen=pg.mkPen(color=(r, g, b), width=1.5))
            self._curves[ch] = curve

            # 50Hz 工频标记线
            notch = pg.InfiniteLine(
                pos=self.NOTCH_HZ, angle=90,
                pen=pg.mkPen(color=(200, 0, 0, 80), width=1, style=Qt.PenStyle.DashLine),
            )
            plot.addItem(notch)
            self._notch_lines[ch] = notch
            self._plots[ch] = plot
        self._restore_cached_data()

    def set_active_channels(self, active: list[bool]):
        """隐藏未选通道，并让剩余频谱子图重新均分区域。"""
        selected = tuple(
            ch for ch in range(min(len(active), self.n_channels)) if active[ch]
        )
        if selected != self._active_channels:
            self._active_channels = selected
            self._setup_ui()

    def update_display(self, psd_data: list[tuple[np.ndarray, np.ndarray]]):
        """更新所有通道的 PSD 曲线。

        Args:
            psd_data: list of (freqs, power) tuples, 每通道一个。
                      空数组表示该通道数据不足。
        """
        for ch, (freqs, power) in enumerate(psd_data):
            if ch >= len(self._curves):
                break
            if ch not in self._active_channels or self._curves[ch] is None:
                continue
            self._last_psd_data[ch] = (freqs, power)
            if len(freqs) > 0 and len(power) > 0 and len(freqs) == len(power):
                self._curves[ch].setData(freqs, power)
            else:
                self._curves[ch].clear()

    def _restore_cached_data(self):
        for ch in self._active_channels:
            frequencies, power = self._last_psd_data[ch]
            if len(frequencies) > 0 and len(frequencies) == len(power):
                self._curves[ch].setData(frequencies, power)

    def clear_data(self):
        """Clear cached spectra when a new Trial starts."""
        self._last_psd_data = [
            (np.array([]), np.array([])) for _ in range(self.n_channels)
        ]
        for curve in self._curves:
            if curve is not None:
                curve.clear()

    def set_channel_highlight(self, ch: int | None):
        """高亮选中通道，其他变暗。ch=None 恢复全部。"""
        for i, curve in enumerate(self._curves):
            if curve is None:
                continue
            r, g, b = self.COLORS[i]
            if ch is None or i == ch:
                curve.setPen(pg.mkPen(color=(r, g, b), width=1.5))
            else:
                curve.setPen(pg.mkPen(color=(r, g, b, 40), width=1))
