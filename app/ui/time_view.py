"""时域波形显示 —— 每通道独立显示 raw + envelope。

使用 pyqtgraph，针对实时渲染优化。
每个已选择通道占一个 stacked subplot，raw 用半透明细线，
envelope 用实色粗线。
"""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class TimeViewWidget(pg.GraphicsLayoutWidget):
    """每个活动通道使用一个独立子图的实时时域显示器。"""

    MAX_RENDER_POINTS = 600

    # 8 通道配色
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

    def __init__(self, parent=None, n_channels: int = 8,
                 display_sec: float = 5.0, fs: float = 250.0):
        super().__init__(parent=parent)
        self.n_channels = n_channels
        self.display_sec = display_sec
        self.fs = fs
        self.n_display = int(fs * display_sec)  # 显示点数

        self._curves_raw = [None] * n_channels
        self._curves_env = [None] * n_channels
        self._plots = [None] * n_channels
        self._channel_labels = [None] * n_channels
        self._last_raw_data = [np.array([]) for _ in range(n_channels)]
        self._last_envelope_data = [np.array([]) for _ in range(n_channels)]
        self._active_channels = tuple(range(n_channels))
        self._y_range: tuple[float, float] | None = None
        self._selected_ch = 0    # 选中通道（高亮）

        self._setup_ui()

    def _setup_ui(self):
        """按当前活动通道创建等高的纵向子图。"""
        self.setBackground("w")
        self.clear()
        self._curves_raw = [None] * self.n_channels
        self._curves_env = [None] * self.n_channels
        self._plots = [None] * self.n_channels
        self._channel_labels = [None] * self.n_channels
        self.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.ci.layout.setHorizontalSpacing(2)
        self.ci.layout.setVerticalSpacing(1)

        tick_font = QFont(self.font())
        tick_font.setPointSize(8)

        for row, ch in enumerate(self._active_channels):
            r, g, b = self.COLORS[ch]
            label = self.addLabel(
                (
                    f"<span style='font-size:9pt; font-weight:600; "
                    f"color:rgb({r},{g},{b})'>CH{ch + 1}</span><br>"
                    "<span style='font-size:7pt; color:#616161'>μV</span>"
                ),
                row=row,
                col=0,
                justify="right",
            )
            label.setMinimumWidth(44)
            label.setMaximumWidth(44)
            self._channel_labels[ch] = label

            plot = self.addPlot(row=row, col=1)
            plot.layout.setContentsMargins(0, 0, 0, 0)
            plot.showGrid(x=True, y=True, alpha=0.22)
            plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            plot.setXRange(0, self.display_sec, padding=0)
            left_axis = plot.getAxis("left")
            left_axis.setStyle(
                tickFont=tick_font,
                tickTextOffset=2,
                tickLength=-3,
                autoExpandTextSpace=False,
                tickTextWidth=36,
            )
            left_axis.setWidth(42)
            bottom_axis = plot.getAxis("bottom")
            bottom_axis.setStyle(
                tickFont=tick_font,
                tickTextOffset=2,
                tickLength=-3,
            )
            if row == len(self._active_channels) - 1:
                plot.setLabel("bottom", "Time", units="s", **{"font-size": "8pt"})
            else:
                plot.hideAxis("bottom")
            if self._y_range is None:
                plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            else:
                plot.setYRange(*self._y_range, padding=0)

            self._curves_raw[ch] = plot.plot(
                pen=pg.mkPen(color=(r, g, b, 140), width=1)
            )
            self._curves_env[ch] = plot.plot(
                pen=pg.mkPen(color=(r, g, b, 230), width=1.6)
            )
            for curve in (self._curves_raw[ch], self._curves_env[ch]):
                curve.setClipToView(True)
            self._plots[ch] = plot

        self._restore_cached_data()

    def set_active_channels(self, active: list[bool]):
        """仅保留选中通道，并让剩余子图自动均分可用高度。"""
        selected = tuple(
            ch for ch in range(min(len(active), self.n_channels)) if active[ch]
        )
        if selected != self._active_channels:
            self._active_channels = selected
            self._setup_ui()

    # ====== 数据更新 ======

    def _restore_cached_data(self):
        for ch in self._active_channels:
            raw = self._last_raw_data[ch][-self.n_display:]
            if len(raw) > 0:
                x, y = self._display_points(raw)
                self._curves_raw[ch].setData(x, y)
            envelope = self._last_envelope_data[ch][-self.n_display:]
            if len(envelope) > 0:
                x, y = self._display_points(envelope)
                self._curves_env[ch].setData(x, y)

    def _display_points(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reduce dense traces while retaining each time bucket's extrema."""
        values = np.asarray(values)
        count = len(values)
        if count <= self.MAX_RENDER_POINTS:
            return np.arange(count) / self.fs, values

        bucket_size = int(np.ceil(count / (self.MAX_RENDER_POINTS // 2)))
        full_count = (count // bucket_size) * bucket_size
        blocks = values[:full_count].reshape(-1, bucket_size)
        offsets = np.arange(len(blocks)) * bucket_size
        minimum_indices = offsets + np.argmin(blocks, axis=1)
        maximum_indices = offsets + np.argmax(blocks, axis=1)
        indices = np.sort(np.concatenate((minimum_indices, maximum_indices)))
        if indices[-1] != count - 1:
            indices = np.append(indices, count - 1)
        return indices / self.fs, values[indices]

    def clear_data(self):
        """Clear cached frames when a new Trial starts."""
        self._last_raw_data = [np.array([]) for _ in range(self.n_channels)]
        self._last_envelope_data = [np.array([]) for _ in range(self.n_channels)]
        for curve in self._curves_raw + self._curves_env:
            if curve is not None:
                curve.clear()

    def update_display(self, raw_data: list[np.ndarray],
                       envelope_data: list[np.ndarray] | None = None):
        """更新所有通道的显示数据。

        Args:
            raw_data: list of 1D arrays, 每通道的滤波后数据 (用于显示)
            envelope_data: list of 1D arrays, 每通道的包络数据（可选）
        """
        n = min(len(raw_data), self.n_channels)

        for ch in range(n):
            if ch not in self._active_channels:
                continue
            raw = raw_data[ch][-self.n_display:]
            self._last_raw_data[ch] = raw

            # raw 曲线: 空数据用 clear() 确保旧曲线消失
            if ch < len(self._curves_raw) and self._curves_raw[ch] is not None:
                if len(raw) > 0:
                    x, y = self._display_points(raw)
                    self._curves_raw[ch].setData(x, y)
                else:
                    self._curves_raw[ch].clear()

            # envelope 曲线
            if (envelope_data and ch < len(envelope_data)
                    and ch < len(self._curves_env) and self._curves_env[ch] is not None):
                env = envelope_data[ch][-self.n_display:]
                self._last_envelope_data[ch] = env
                if len(env) > 0:
                    x, y = self._display_points(env)
                    self._curves_env[ch].setData(x, y)
                else:
                    self._curves_env[ch].clear()

    def set_display_sec(self, seconds: float):
        """动态调整时间窗口宽度。"""
        self.display_sec = seconds
        self.n_display = int(self.fs * seconds)
        for plot in self._plots:
            if plot is not None:
                plot.setXRange(0, seconds, padding=0)

    def set_y_range(self, y_min: float, y_max: float):
        """统一设置所有子图的 Y 轴范围。"""
        self._y_range = (y_min, y_max)
        for plot in self._plots:
            if plot is not None:
                plot.setYRange(y_min, y_max, padding=0)

    def auto_range(self):
        """启用自动 Y 轴范围。"""
        self._y_range = None
        for plot in self._plots:
            if plot is not None:
                plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def set_selected_channel(self, ch: int):
        """高亮选中通道（其他通道变暗）。"""
        self._selected_ch = ch
        for i, (raw, env) in enumerate(zip(self._curves_raw, self._curves_env)):
            if raw is None or env is None:
                continue
            if i == ch:
                r, g, b = self.COLORS[ch]
                raw.setPen(pg.mkPen(color=(r, g, b, 200), width=2))
                env.setPen(pg.mkPen(color=(r, g, b, 255), width=3))
            else:
                rc, gc, bc = self.COLORS[i]
                raw.setPen(pg.mkPen(color=(rc, gc, bc, 40), width=1))
                env.setPen(pg.mkPen(color=(rc, gc, bc, 80), width=1))


# ====== 快速测试 ======

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    widget = TimeViewWidget(n_channels=8, display_sec=5.0)
    widget.show()

    # 生成测试数据
    fs = 250
    t = np.arange(0, 5.0, 1 / fs)
    raw_test = []
    env_test = []
    for ch in range(8):
        freq = 30 + ch * 15
        raw = 50 * np.sin(2 * np.pi * freq * t) + 10 * np.random.randn(len(t))
        env = np.abs(raw)
        raw_test.append(raw)
        env_test.append(env)

    widget.update_display(raw_test, env_test)
    widget.set_y_range(-300, 300)

    sys.exit(app.exec())
