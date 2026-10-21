"""Interactive review for rhythmic EMG action segments."""
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
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from preprocessing.envelope import compute_envelope
from preprocessing.filters import apply_filters
from action_review import (
    detect_rhythmic_segments,
    load_action_segments,
    save_action_segments,
)
from time_review import load_time_review, load_trial_txt


class ActionReviewDialog(QDialog):
    """Auto-detect repetitions and let the user correct each action window."""

    MAX_PLOT_POINTS = 30000
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
        self.data = load_trial_txt(self.trial_dir / "emg_data.txt")
        self.time_review = load_time_review(self.trial_dir)
        if not self.time_review:
            raise ValueError("请先完成该 Trial 的时间切分审查")
        self.viewer_config = viewer_config
        self.saved_segments: list[dict] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._selected_index = 0

        self._filtered, self._envelopes = self._preprocess_channels()
        stored = load_action_segments(self.trial_dir)
        if stored:
            self.reference_channel = stored[0]["reference_channel"]
            self.segments = [
                {
                    "start_sample": item["start_sample"],
                    "peak_sample": item["peak_sample"],
                    "end_sample_exclusive": item["end_sample_exclusive"],
                }
                for item in stored
            ]
            state = "已加载上次人工审核结果"
        else:
            self.reference_channel = self._best_reference_channel()
            self.segments = self._detect_current_reference()
            state = f"自动检测到 {len(self.segments)} 个节奏动作，请逐段检查"

        self.setWindowTitle(f"节奏动作切分审查 - {self.trial_dir.name}")
        self.resize(1100, 760)
        self.setMinimumSize(780, 560)
        self._setup_ui(state)

    @property
    def valid_start(self) -> int:
        return int(self.time_review["start_sample"])

    @property
    def valid_end(self) -> int:
        return int(self.time_review["end_sample_exclusive"])

    def _filter_config(self):
        source = self.viewer_config
        nyquist = self.data.sample_rate / 2.0
        low = float(getattr(source, "bandpass_low", 20.0))
        high = min(float(getattr(source, "bandpass_high", 450.0)), nyquist - 1.0)
        return SimpleNamespace(
            sampling_rate=self.data.sample_rate,
            preprocess=SimpleNamespace(
                notch_hz=float(getattr(source, "notch_hz", 50.0)),
                notch_bandwidth_hz=float(
                    getattr(source, "notch_bandwidth_hz", 8.0)
                ),
                notch_harmonics=int(getattr(source, "notch_harmonics", 3)),
                bandpass=(low, max(high, low + 1.0)),
            ),
            segmentation=SimpleNamespace(
                use_preprocess_chain=False,
                smooth_window_ms=float(
                    getattr(source, "envelope_window_ms", 100.0)
                ),
            ),
        )

    def _preprocess_channels(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        config = self._filter_config()
        filtered = []
        envelopes = []
        for channel in range(self.data.emg.shape[1]):
            signal = np.nan_to_num(
                self.data.emg[:, channel],
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            band = apply_filters(signal, config, do_notch=True)
            env = compute_envelope(
                band,
                config,
                window_ms=config.segmentation.smooth_window_ms,
            )
            filtered.append(band)
            envelopes.append(env)
        return filtered, envelopes

    def _best_reference_channel(self) -> int:
        scores = []
        selection = slice(self.valid_start, self.valid_end)
        for envelope in self._envelopes:
            values = envelope[selection]
            scores.append(float(np.percentile(values, 95) - np.percentile(values, 20)))
        return int(np.argmax(scores)) if scores else 0

    def _detect_current_reference(self) -> list[dict[str, int]]:
        return detect_rhythmic_segments(
            self._envelopes[self.reference_channel],
            self.data.sample_rate,
            self.valid_start,
            self.valid_end,
        )

    def _setup_ui(self, state: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        summary = QLabel(
            f"{self.trial_dir.name}  |  节奏动作  |  "
            f"有效时间 {self.valid_start / self.data.sample_rate:.3f} - "
            f"{self.valid_end / self.data.sample_rate:.3f} s"
        )
        root.addWidget(summary)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("参考通道:"))
        self._reference_combo = QComboBox()
        self._reference_combo.addItems(list(self.data.channel_names))
        self._reference_combo.setCurrentIndex(self.reference_channel)
        settings.addWidget(self._reference_combo)
        auto_button = QPushButton("重新自动切分")
        auto_button.clicked.connect(self._rerun_detection)
        settings.addWidget(auto_button)
        self._state_label = QLabel(state)
        settings.addWidget(self._state_label, 1)
        root.addLayout(settings)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", "uV")
        self._plot.setLabel("bottom", "Time", units="s")
        root.addWidget(self._plot, 1)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["动作", "起点 (s)", "峰值 (s)", "终点 (s)", "时长 (s)"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(190)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        root.addWidget(self._table)

        actions = QHBoxLayout()
        add_button = QPushButton("添加分段")
        delete_button = QPushButton("删除选中分段")
        add_button.clicked.connect(self._add_segment)
        delete_button.clicked.connect(self._delete_segment)
        actions.addWidget(add_button)
        actions.addWidget(delete_button)
        actions.addStretch()
        root.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存动作切分")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._reference_combo.currentIndexChanged.connect(self._reference_changed)
        self._render_plot_and_table()

    def _render_plot_and_table(self) -> None:
        self._plot.clear()
        self._regions = []
        fs = self.data.sample_rate
        step = max(1, math.ceil(self.data.n_samples / self.MAX_PLOT_POINTS))
        x = np.arange(0, self.data.n_samples, step, dtype=float) / fs
        color = self.COLORS[self.reference_channel % len(self.COLORS)]
        self._plot.plot(
            x,
            self._filtered[self.reference_channel][::step],
            pen=pg.mkPen((*color, 120), width=1),
        )
        self._plot.plot(
            x,
            self._envelopes[self.reference_channel][::step],
            pen=pg.mkPen(color, width=2),
        )
        self._plot.setTitle(self.data.channel_names[self.reference_channel], color=color)
        self._plot.setXRange(self.valid_start / fs, self.valid_end / fs, padding=0)

        for index, segment in enumerate(self.segments):
            region = pg.LinearRegionItem(
                values=(segment["start_sample"] / fs, segment["end_sample_exclusive"] / fs),
                brush=pg.mkBrush(46, 125, 50, 45),
                pen=pg.mkPen(46, 125, 50, 210, width=2),
                movable=True,
            )
            region.setBounds((self.valid_start / fs, self.valid_end / fs))
            region.sigRegionChangeFinished.connect(
                lambda _region=region, _index=index: self._region_changed(
                    _index, _region
                )
            )
            self._plot.addItem(region)
            self._regions.append(region)

            peak = segment["peak_sample"]
            self._plot.plot(
                [peak / fs],
                [self._envelopes[self.reference_channel][peak]],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=pg.mkBrush(198, 40, 40),
            )

        self._refresh_table()

    def _refresh_table(self) -> None:
        fs = self.data.sample_rate
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(len(self.segments))
            for row, segment in enumerate(self.segments):
                start = segment["start_sample"] / fs
                peak = segment["peak_sample"] / fs
                end = segment["end_sample_exclusive"] / fs
                values = [f"#{row + 1}", f"{start:.3f}", f"{peak:.3f}", f"{end:.3f}", f"{end - start:.3f}"]
                for column, value in enumerate(values):
                    self._table.setItem(row, column, QTableWidgetItem(value))
            if self.segments:
                self._selected_index = min(self._selected_index, len(self.segments) - 1)
                self._table.selectRow(self._selected_index)
        finally:
            self._table.blockSignals(False)
        self._highlight_selected()

    def _on_table_selection(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._selected_index = row
            self._highlight_selected()

    def _highlight_selected(self) -> None:
        for index, region in enumerate(self._regions):
            if index == self._selected_index:
                region.setBrush(pg.mkBrush(46, 125, 50, 70))
                region.setZValue(20)
            else:
                region.setBrush(pg.mkBrush(120, 120, 120, 28))
                region.setZValue(10)

    def _recompute_peak(self, segment: dict) -> None:
        start = segment["start_sample"]
        end = segment["end_sample_exclusive"]
        envelope = self._envelopes[self.reference_channel]
        segment["peak_sample"] = start + int(np.argmax(envelope[start:end]))

    def _region_changed(self, index: int, region: pg.LinearRegionItem) -> None:
        if index >= len(self.segments):
            return
        changed = self.segments[index]
        start_sec, end_sec = region.getRegion()
        start = max(self.valid_start, int(round(start_sec * self.data.sample_rate)))
        end = min(self.valid_end, int(round(end_sec * self.data.sample_rate)))
        if end <= start:
            end = min(self.valid_end, start + 1)
        changed["start_sample"] = start
        changed["end_sample_exclusive"] = end
        self._recompute_peak(changed)
        self.segments.sort(key=lambda item: item["start_sample"])
        self._selected_index = next(
            (position for position, item in enumerate(self.segments) if item is changed),
            0,
        )
        self._render_plot_and_table()

    def _reference_changed(self, channel: int) -> None:
        self.reference_channel = channel
        for segment in self.segments:
            self._recompute_peak(segment)
        self._render_plot_and_table()
        self._state_label.setText("参考通道已改变；边界保留，峰值已重新计算")

    def _rerun_detection(self) -> None:
        if self.segments:
            answer = QMessageBox.question(
                self,
                "重新自动切分",
                "重新检测会替换当前尚未保存的人工调整，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.segments = self._detect_current_reference()
        self._selected_index = 0
        self._render_plot_and_table()
        self._state_label.setText(f"重新检测到 {len(self.segments)} 个节奏动作")

    def _add_segment(self) -> None:
        envelope = self._envelopes[self.reference_channel]
        order = np.argsort(envelope[self.valid_start:self.valid_end])[::-1]
        half = max(1, int(round(0.4 * self.data.sample_rate)))
        for local_peak in order:
            peak = self.valid_start + int(local_peak)
            if any(
                segment["start_sample"] <= peak < segment["end_sample_exclusive"]
                for segment in self.segments
            ):
                continue
            left = max(
                [self.valid_start]
                + [segment["end_sample_exclusive"] for segment in self.segments if segment["end_sample_exclusive"] <= peak]
            )
            right = min(
                [self.valid_end]
                + [segment["start_sample"] for segment in self.segments if segment["start_sample"] > peak]
            )
            start = max(left, peak - half)
            end = min(right, peak + half)
            if end - start >= max(3, int(0.15 * self.data.sample_rate)):
                new_segment = {
                    "start_sample": start,
                    "peak_sample": peak,
                    "end_sample_exclusive": end,
                }
                self.segments.append(new_segment)
                self.segments.sort(key=lambda item: item["start_sample"])
                self._selected_index = self.segments.index(new_segment)
                self._render_plot_and_table()
                self._state_label.setText("已添加一个候选分段，请拖动边界确认")
                return
        QMessageBox.information(self, "无法添加", "当前有效范围内没有足够的未切分区域")

    def _delete_segment(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self.segments):
            del self.segments[row]
            self._selected_index = max(0, row - 1)
            self._render_plot_and_table()
            self._state_label.setText("已删除选中分段")

    def _save(self) -> None:
        if not self.segments:
            QMessageBox.warning(self, "无法保存", "至少需要保留一个动作分段")
            return
        ordered = sorted(self.segments, key=lambda item: item["start_sample"])
        if any(
            segment["start_sample"] < self.valid_start
            or segment["end_sample_exclusive"] > self.valid_end
            for segment in ordered
        ):
            QMessageBox.warning(self, "无法保存", "动作分段超出了已审核时间范围")
            return
        try:
            self.saved_segments = save_action_segments(
                self.trial_dir,
                reference_channel=self.reference_channel,
                segments=ordered,
                n_samples=self.data.n_samples,
                sample_rate=self.data.sample_rate,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.accept()
