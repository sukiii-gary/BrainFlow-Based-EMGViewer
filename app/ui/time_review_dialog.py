"""Interactive review of the valid time range for one completed Trial."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

_EMG_PYTHON = Path(__file__).resolve().parents[2] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from preprocessing.envelope import compute_envelope
from preprocessing.filters import apply_filters
from time_review import (
    default_middle_window,
    load_time_review,
    load_trial_txt,
    save_time_review,
)


class TimeReviewDialog(QDialog):
    """Show all EMG channels and persist one manually approved time range."""

    MAX_PLOT_POINTS = 20000
    COLORS = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
    ]

    def __init__(self, trial_dir: str | Path, viewer_config=None, parent=None):
        super().__init__(parent)
        self.trial_dir = Path(trial_dir).resolve()
        self.txt_path = self.trial_dir / "emg_data.txt"
        self.data = load_trial_txt(self.txt_path)
        self.viewer_config = viewer_config
        self.saved_record: dict | None = None
        self._regions: list[pg.LinearRegionItem] = []
        self._syncing = False

        default_start, default_end = default_middle_window(self.data.n_samples)
        stored = load_time_review(self.trial_dir)
        if stored and (
            0 <= stored["start_sample"]
            < stored["end_sample_exclusive"]
            <= self.data.n_samples
        ):
            self._start_sample = stored["start_sample"]
            self._end_sample = stored["end_sample_exclusive"]
            state_text = "已加载上次审核范围"
        else:
            self._start_sample = default_start
            self._end_sample = default_end
            state_text = "默认保留中间 70%，请拖动边界检查"

        self.setWindowTitle(f"时间切分审查 - {self.trial_dir.name}")
        self.resize(1100, 760)
        self.setMinimumSize(760, 520)
        self._setup_ui(state_text)

    def _filter_config(self):
        source = self.viewer_config
        nyquist = self.data.sample_rate / 2.0
        low = float(getattr(source, "bandpass_low", 20.0))
        high = min(float(getattr(source, "bandpass_high", 450.0)), nyquist - 1.0)
        bandpass = (
            low,
            max(high, low + 1.0),
        )
        return SimpleNamespace(
            sampling_rate=self.data.sample_rate,
            preprocess=SimpleNamespace(
                notch_hz=float(getattr(source, "notch_hz", 50.0)),
                notch_bandwidth_hz=float(
                    getattr(source, "notch_bandwidth_hz", 8.0)
                ),
                notch_harmonics=int(getattr(source, "notch_harmonics", 3)),
                bandpass=bandpass,
            ),
            segmentation=SimpleNamespace(
                use_preprocess_chain=False,
                smooth_window_ms=float(
                    getattr(source, "envelope_window_ms", 100.0)
                ),
            ),
        )

    def _setup_ui(self, state_text: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        summary = QLabel(
            f"{self.trial_dir.name}  |  {self.data.n_samples} 样本  |  "
            f"{self.data.sample_rate:g} Hz  |  {self.data.duration_sec:.2f} s"
        )
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(summary)
        self._state_label = QLabel(state_text)
        self._state_label.setStyleSheet("color: #8a6500;")
        root.addWidget(self._state_label)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground("w")
        root.addWidget(self._plot_widget, 1)
        self._build_plots()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("起点 (s):"))
        self._start_spin = self._make_spinbox()
        controls.addWidget(self._start_spin)
        controls.addWidget(QLabel("终点 (s):"))
        self._end_spin = self._make_spinbox()
        controls.addWidget(self._end_spin)
        self._range_label = QLabel()
        controls.addWidget(self._range_label)
        controls.addStretch()
        reset_button = QPushButton("恢复中间 70%")
        reset_button.clicked.connect(self._reset_default)
        controls.addWidget(reset_button)
        root.addLayout(controls)

        self._start_spin.valueChanged.connect(self._on_spin_changed)
        self._end_spin.valueChanged.connect(self._on_spin_changed)
        self._set_range(
            self._start_sample / self.data.sample_rate,
            self._end_sample / self.data.sample_rate,
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存审核范围")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _make_spinbox(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setRange(0.0, self.data.duration_sec)
        spin.setSuffix(" s")
        return spin

    def _build_plots(self) -> None:
        fs = self.data.sample_rate
        config = self._filter_config()
        step = max(1, math.ceil(self.data.n_samples / self.MAX_PLOT_POINTS))
        x = np.arange(0, self.data.n_samples, step, dtype=float) / fs
        start_sec = self._start_sample / fs
        end_sec = self._end_sample / fs

        for channel in range(self.data.emg.shape[1]):
            signal = np.nan_to_num(
                self.data.emg[:, channel],
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            filtered = apply_filters(signal, config, do_notch=True)
            envelope = compute_envelope(
                filtered,
                config,
                window_ms=config.segmentation.smooth_window_ms,
            )
            plot = self._plot_widget.addPlot(row=channel, col=0)
            color = self.COLORS[channel % len(self.COLORS)]
            title = self.data.channel_names[channel]
            plot.setTitle(title, color=color, size="10pt")
            plot.setLabel("left", "uV")
            plot.showGrid(x=True, y=True, alpha=0.25)
            if channel == self.data.emg.shape[1] - 1:
                plot.setLabel("bottom", "Time", units="s")
            plot.plot(x, filtered[::step], pen=pg.mkPen((*color, 150), width=1))
            plot.plot(x, envelope[::step], pen=pg.mkPen(color, width=2))
            plot.setXRange(0.0, self.data.duration_sec, padding=0)

            region = pg.LinearRegionItem(
                values=(start_sec, end_sec),
                orientation=pg.LinearRegionItem.Vertical,
                brush=pg.mkBrush(46, 125, 50, 32),
                pen=pg.mkPen(46, 125, 50, 210, width=2),
                movable=True,
            )
            region.setBounds((0.0, self.data.duration_sec))
            region.setZValue(10)
            region.sigRegionChanged.connect(
                lambda _region=region: self._on_region_changed(_region)
            )
            plot.addItem(region)
            self._regions.append(region)

    def _on_region_changed(self, source: pg.LinearRegionItem) -> None:
        if self._syncing:
            return
        start, end = source.getRegion()
        self._set_range(start, end, source=source)

    def _on_spin_changed(self) -> None:
        if not self._syncing:
            self._set_range(self._start_spin.value(), self._end_spin.value())

    def _set_range(
        self,
        start_sec: float,
        end_sec: float,
        source: pg.LinearRegionItem | None = None,
    ) -> None:
        duration = self.data.duration_sec
        minimum = 1.0 / self.data.sample_rate
        start_sec = min(max(0.0, float(start_sec)), duration - minimum)
        end_sec = min(max(float(end_sec), start_sec + minimum), duration)

        self._syncing = True
        try:
            for region in self._regions:
                if region is not source:
                    region.setRegion((start_sec, end_sec))
            self._start_spin.setValue(start_sec)
            self._end_spin.setValue(end_sec)
            kept = end_sec - start_sec
            percentage = 100.0 * kept / duration
            self._range_label.setText(f"保留 {kept:.3f} s ({percentage:.1f}%)")
        finally:
            self._syncing = False

    def _reset_default(self) -> None:
        start, end = default_middle_window(self.data.n_samples)
        self._set_range(start / self.data.sample_rate, end / self.data.sample_rate)
        self._state_label.setText("已恢复默认中间 70%，保存后生效")

    def _save(self) -> None:
        start_sample = int(round(self._start_spin.value() * self.data.sample_rate))
        end_sample = int(round(self._end_spin.value() * self.data.sample_rate))
        start_sample = min(max(0, start_sample), self.data.n_samples - 1)
        end_sample = min(max(start_sample + 1, end_sample), self.data.n_samples)
        try:
            self.saved_record = save_time_review(
                self.trial_dir,
                start_sample=start_sample,
                end_sample_exclusive=end_sample,
                n_samples=self.data.n_samples,
                sample_rate=self.data.sample_rate,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.accept()
