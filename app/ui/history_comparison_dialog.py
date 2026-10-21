"""Read-only comparison dialog for segments from this version's recordings."""
from __future__ import annotations

import math
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from history_comparison import (
    discover_history,
    list_segment_candidates,
    prepare_comparison,
)


class _ComparisonWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, refs, mode, alignment, viewer_config):
        super().__init__()
        self.refs = list(refs)
        self.mode = mode
        self.alignment = alignment
        self.viewer_config = viewer_config

    @pyqtSlot()
    def run(self):
        try:
            result = prepare_comparison(
                self.refs,
                mode=self.mode,
                alignment=self.alignment,
                viewer_config=self.viewer_config,
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class _DiscoveryWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, recordings_root):
        super().__init__()
        self.recordings_root = recordings_root

    @pyqtSlot()
    def run(self):
        try:
            trials = discover_history(self.recordings_root)
            refs = []
            for trial in trials:
                if getattr(trial, "selectable", True):
                    refs.extend(list_segment_candidates(trial))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.completed.emit((trials, refs))


class HistoryComparisonDialog(QDialog):
    """Select two to six stored segments and overlay one common channel."""

    MAX_SEGMENTS = 6
    MAX_PLOT_POINTS = 30000
    COLORS = [
        (31, 119, 180),
        (230, 126, 34),
        (46, 139, 87),
        (198, 40, 40),
        (133, 92, 162),
        (0, 137, 123),
    ]

    def __init__(self, recordings_root: str | Path, viewer_config=None, parent=None):
        super().__init__(parent)
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.viewer_config = viewer_config
        self._refs = []
        self._discovery_thread: QThread | None = None
        self._discovery_worker: _DiscoveryWorker | None = None
        self._thread: QThread | None = None
        self._worker: _ComparisonWorker | None = None
        self.last_result = None
        self._common_channels: list[str] = []

        self.setWindowTitle("历史数据段对比")
        self.resize(1120, 760)
        self.setMinimumSize(820, 580)
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("筛选:"))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Session、Trial、任务或被试")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        filters.addWidget(self._filter, 1)
        self._refresh_button = QPushButton("刷新")
        self._refresh_button.clicked.connect(self._load_history)
        filters.addWidget(self._refresh_button)
        root.addLayout(filters)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "Session", "Trial", "被试", "任务", "片段", "范围 (s)", "采样率", "通道",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        root.addWidget(self._table, 2)

        controls = QHBoxLayout()
        self._selection_label = QLabel("已选择 0 段（请选择 2–6 段）")
        controls.addWidget(self._selection_label, 1)
        controls.addWidget(QLabel("信号:"))
        self._mode = QComboBox()
        self._mode.addItem("原始", "raw")
        self._mode.addItem("滤波", "filtered")
        self._mode.addItem("RMS", "rms")
        controls.addWidget(self._mode)
        controls.addWidget(QLabel("对齐:"))
        self._alignment = QComboBox()
        self._alignment.addItem("片段起点", "start")
        self._alignment.addItem("动作峰值", "peak")
        controls.addWidget(self._alignment)
        self._compare_button = QPushButton("生成对比")
        self._compare_button.setEnabled(False)
        self._compare_button.clicked.connect(self._start_comparison)
        controls.addWidget(self._compare_button)
        root.addLayout(controls)

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("显示通道:"))
        self._channel = QComboBox()
        self._channel.setEnabled(False)
        self._channel.currentIndexChanged.connect(self._render_plot)
        plot_controls.addWidget(self._channel)
        self._result_label = QLabel("尚未生成对比")
        self._result_label.setWordWrap(True)
        plot_controls.addWidget(self._result_label, 1)
        root.addLayout(plot_controls)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("bottom", "相对时间", units="s")
        self._plot.setLabel("left", "幅值", units="uV")
        self._plot.addLegend(offset=(8, 8))
        root.addWidget(self._plot, 3)

        bottom = QHBoxLayout()
        self._state_label = QLabel("只读对比，不会修改原始数据或审查结果")
        bottom.addWidget(self._state_label, 1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        bottom.addWidget(close_button)
        root.addLayout(bottom)

    def _load_history(self):
        if self._thread is not None or self._discovery_thread is not None:
            return
        self.last_result = None
        self._common_channels = []
        self._channel.clear()
        self._channel.setEnabled(False)
        self._plot.clear()
        self._result_label.setText("尚未生成对比")
        self._set_discovery_busy(True)
        thread = QThread(self)
        worker = _DiscoveryWorker(self.recordings_root)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._history_ready)
        worker.failed.connect(self._history_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._discovery_finished)
        thread.finished.connect(thread.deleteLater)
        self._discovery_thread = thread
        self._discovery_worker = worker
        thread.start()

    def _history_ready(self, payload):
        trials, refs = payload
        self._refs = refs
        self._table.setRowCount(len(refs))
        for row, ref in enumerate(refs):
            trial = ref.trial
            start = float(ref.start_sample) / float(trial.sample_rate)
            end = float(ref.end_sample_exclusive) / float(trial.sample_rate)
            values = [
                trial.session_name,
                trial.trial_name,
                trial.subject_id or "--",
                trial.task or "--",
                ref.label,
                f"{start:.3f} - {end:.3f}",
                f"{float(trial.sample_rate):g} Hz",
                str(int(trial.n_channels)),
            ]
            search_text = " ".join(str(value) for value in values).casefold()
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, row)
                item.setData(int(Qt.ItemDataRole.UserRole) + 1, search_text)
                self._table.setItem(row, column, item)
        self._table.clearSelection()
        self._apply_filter(self._filter.text())
        self._state_label.setText(
            f"发现 {len(trials)} 个历史 Trial、{len(refs)} 个可选片段；数据保持只读"
        )

    def _history_failed(self, message: str):
        self._refs = []
        self._table.setRowCount(0)
        self._state_label.setText(f"无法读取历史记录: {message}")

    def _discovery_finished(self):
        self._discovery_thread = None
        self._discovery_worker = None
        self._set_discovery_busy(False)

    def _set_discovery_busy(self, busy: bool):
        self._table.setEnabled(not busy)
        self._filter.setEnabled(not busy)
        self._refresh_button.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._alignment.setEnabled(not busy)
        self._compare_button.setEnabled(False)
        if busy:
            self._state_label.setText("正在扫描本版本的历史记录…")
        else:
            self._selection_changed()

    def _apply_filter(self, text: str):
        query = text.strip().casefold()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            searchable = item.data(int(Qt.ItemDataRole.UserRole) + 1) if item else ""
            self._table.setRowHidden(row, bool(query and query not in searchable))
        self._selection_changed()

    def selected_refs(self):
        rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        return [self._refs[row] for row in rows if not self._table.isRowHidden(row)]

    def _selection_changed(self):
        refs = self.selected_refs()
        count = len(refs)
        valid = 2 <= count <= self.MAX_SEGMENTS and self._thread is None
        self._selection_label.setText(f"已选择 {count} 段（请选择 2–6 段）")
        self._compare_button.setEnabled(valid)
        all_have_peak = bool(refs) and all(ref.peak_sample is not None for ref in refs)
        peak_index = self._alignment.findData("peak")
        model_item = self._alignment.model().item(peak_index)
        if model_item is not None:
            model_item.setEnabled(all_have_peak)
        if not all_have_peak and self._alignment.currentData() == "peak":
            self._alignment.setCurrentIndex(self._alignment.findData("start"))

    def _set_busy(self, busy: bool):
        self._table.setEnabled(not busy)
        self._filter.setEnabled(not busy)
        self._refresh_button.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._alignment.setEnabled(not busy)
        self._compare_button.setEnabled(False if busy else 2 <= len(self.selected_refs()) <= 6)
        self._state_label.setText("正在处理选中的历史片段…" if busy else self._state_label.text())

    def _start_comparison(self):
        refs = self.selected_refs()
        if not 2 <= len(refs) <= self.MAX_SEGMENTS:
            QMessageBox.warning(self, "无法对比", "请选择 2–6 个历史片段")
            return
        alignment = self._alignment.currentData()
        if alignment == "peak" and any(ref.peak_sample is None for ref in refs):
            QMessageBox.warning(self, "无法按峰值对齐", "只有动作切分片段可以按峰值对齐")
            return

        self._set_busy(True)
        thread = QThread(self)
        worker = _ComparisonWorker(
            refs,
            self._mode.currentData(),
            alignment,
            self.viewer_config,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._comparison_ready)
        worker.failed.connect(self._comparison_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _comparison_ready(self, result):
        self.last_result = result
        common = None
        for series in result.series:
            names = list(series.channel_names)
            common = names if common is None else [name for name in common if name in names]
        self._common_channels = common or []
        self._channel.blockSignals(True)
        self._channel.clear()
        self._channel.addItems(self._common_channels)
        self._channel.blockSignals(False)
        self._channel.setEnabled(bool(self._common_channels))
        if not self._common_channels:
            self._plot.clear()
            self._result_label.setText("所选片段没有共同通道")
            return
        band = getattr(result, "common_band_hz", None)
        band_text = f"；共同频带 {band[0]:g}–{band[1]:g} Hz" if band else ""
        self._result_label.setText(f"已叠加 {len(result.series)} 段{band_text}")
        self._render_plot()

    def _comparison_failed(self, message: str):
        self.last_result = None
        self._plot.clear()
        self._result_label.setText("处理失败")
        self._state_label.setText(f"历史片段处理失败: {message}")

    def _thread_finished(self):
        self._thread = None
        self._worker = None
        self._set_busy(False)
        self._selection_changed()
        if self.last_result is not None:
            self._state_label.setText("对比已完成；数据保持只读")

    def _render_plot(self):
        self._plot.clear()
        if self.last_result is None or not self._common_channels:
            return
        channel_name = self._channel.currentText()
        if not channel_name:
            return
        for index, series in enumerate(self.last_result.series):
            try:
                channel = list(series.channel_names).index(channel_name)
            except ValueError:
                continue
            times = np.asarray(series.time_sec, dtype=float).ravel()
            values = np.asarray(series.values, dtype=float)
            if values.ndim == 1:
                signal = values
            else:
                signal = values[:, channel]
            count = min(times.size, signal.size)
            step = max(1, math.ceil(count / self.MAX_PLOT_POINTS))
            color = self.COLORS[index % len(self.COLORS)]
            self._plot.plot(
                times[:count:step],
                signal[:count:step],
                pen=pg.mkPen(color, width=1.5),
                name=series.label,
            )
        self._plot.setTitle(channel_name, color=(40, 40, 40))

    def reject(self):
        if self._thread is not None or self._discovery_thread is not None:
            self._state_label.setText("正在处理历史片段，请等待处理完成后关闭")
            return
        super().reject()
