"""Modal, non-recording Cyton electrode impedance test."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from impedance import classify_impedance_ohms


class _ImpedanceWorker(QObject):
    progress = pyqtSignal(int, str, object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, board, channels: list[int], duration_sec: float):
        super().__init__()
        self._board = board
        self._channels = channels
        self._duration_sec = duration_sec
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            report = self._board.measure_impedances(
                self._channels,
                duration_sec=self._duration_sec,
                settle_sec=1.0,
                progress_callback=self.progress.emit,
                cancel_event=self._cancel,
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(report)


class ImpedanceDialog(QDialog):
    def __init__(self, board, parent=None):
        super().__init__(parent)
        self.board = board
        self.last_report = None
        self._thread = None
        self._worker = None
        self._pending_error = None
        self._checks = []

        self.setWindowTitle("Cyton 电极阻抗测试")
        self.setMinimumSize(920, 540)
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "使用 Cyton 官方 6 nA、约 31.5 Hz lead-off 激励；仅测试 N 输入，通道将依次测量。"
        ))
        interpretation = QLabel(
            "EMG 接触目标：每个电极 <10 kOhm。Cyton 官方 GUI 的 <750 kOhm 绿色仅为设备显示参考，"
            "不代表 EMG 接触已达标。"
        )
        interpretation.setWordWrap(True)
        interpretation.setStyleSheet("color: #1f4f6e;")
        root.addWidget(interpretation)
        warning = QLabel(
            "测试不会录制 Trial，也不会触发 Delsys。测试完成或取消后必须恢复并验证原硬件设置。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #8a6500;")
        root.addWidget(warning)

        self._table = QTableWidget(self.board.n_channels, 7)
        self._table.setHorizontalHeaderLabels(
            ["选择", "通道", "阻抗 (kOhm)", "标准差 (uV)", "样本数",
             "EMG 接触建议", "Cyton GUI 参考"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for row in range(self.board.n_channels):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            holder = QHBoxLayout()
            holder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            holder.setContentsMargins(0, 0, 0, 0)
            cell = QWidget()
            cell.setLayout(holder)
            holder.addWidget(checkbox)
            self._checks.append(checkbox)
            self._table.setCellWidget(row, 0, cell)
            self._table.setItem(row, 1, QTableWidgetItem(f"CH{row + 1}"))
            for column in range(2, 7):
                self._table.setItem(row, column, QTableWidgetItem("--"))
        self._table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._table, 1)

        options = QHBoxLayout()
        self._select_all = QPushButton("全选")
        self._clear_all = QPushButton("清空")
        options.addWidget(self._select_all)
        options.addWidget(self._clear_all)
        options.addStretch()
        options.addWidget(QLabel("每通道有效采样:"))
        self._duration = QComboBox()
        for effective in (4, 5, 7, 10):
            self._duration.addItem(f"{effective} s (+1 s 稳定)", effective + 1.0)
        options.addWidget(self._duration)
        root.addLayout(options)

        self._restore_label = QLabel("尚未开始")
        self._restore_label.setWordWrap(True)
        root.addWidget(self._restore_label)

        actions = QHBoxLayout()
        self._start = QPushButton("开始测试")
        self._cancel = QPushButton("取消测试")
        self._close = QPushButton("关闭")
        self._cancel.setEnabled(False)
        actions.addStretch()
        actions.addWidget(self._start)
        actions.addWidget(self._cancel)
        actions.addWidget(self._close)
        root.addLayout(actions)

        self._select_all.clicked.connect(
            lambda: [checkbox.setChecked(True) for checkbox in self._checks]
        )
        self._clear_all.clicked.connect(
            lambda: [checkbox.setChecked(False) for checkbox in self._checks]
        )
        self._start.clicked.connect(self._start_test)
        self._cancel.clicked.connect(self._cancel_test)
        self._close.clicked.connect(self.reject)

    def selected_channels(self) -> list[int]:
        return [
            index + 1 for index, checkbox in enumerate(self._checks)
            if checkbox.isChecked()
        ]

    def _start_test(self) -> None:
        if self._thread is not None:
            return
        channels = self.selected_channels()
        if not channels:
            QMessageBox.warning(self, "无法开始", "请至少选择一个通道")
            return
        for channel in channels:
            row = channel - 1
            for column in range(2, 7):
                self._table.item(row, column).setText("--")
            self._table.item(row, 5).setText("等待")
            self._table.item(row, 6).setText("等待")

        thread = QThread(self)
        worker = _ImpedanceWorker(
            self.board, channels, float(self._duration.currentData())
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self._pending_error = None
        self.last_report = None
        self._set_running(True)
        self._restore_label.setText("测试进行中；请勿断开板卡")
        thread.start()

    @pyqtSlot(int, str, object)
    def _on_progress(self, channel: int, state: str, result) -> None:
        row = channel - 1
        labels = {"configuring": "配置测试激励", "measuring": "测量中"}
        if state == "complete" and isinstance(result, dict):
            classification = {
                **classify_impedance_ohms(result.get("impedance_ohms")),
                **result,
            }
            self._table.item(row, 2).setText(
                f"{classification['impedance_ohms'] / 1000.0:.1f}"
            )
            self._table.item(row, 3).setText(f"{classification['std_uv']:.3f}")
            self._table.item(row, 4).setText(str(classification["sample_count"]))
            emg_item = self._table.item(row, 5)
            cyton_item = self._table.item(row, 6)
            emg_item.setText(classification["emg_contact_label"])
            emg_item.setBackground(QColor(classification["emg_contact_color"]))
            cyton_item.setText(classification["cyton_gui_label"])
            cyton_item.setBackground(QColor(classification["cyton_gui_color"]))
            if classification["likely_open_circuit"]:
                emg_item.setToolTip("约 >=5000 kOhm：疑似开路；请检查电极、参考和 BIAS/D_G 连接")
            else:
                emg_item.setToolTip(
                    "这是 EMG 接触建议；仍须检查双极电极匹配、静息噪声、工频、运动伪迹和信噪比"
                )
            cyton_item.setToolTip(
                "Cyton 官方 GUI 显示参考，不是表面 EMG 验收阈值"
            )
        else:
            self._table.item(row, 5).setText(labels.get(state, state))
            self._table.item(row, 6).setText(labels.get(state, state))

    @pyqtSlot(object)
    def _on_finished(self, report) -> None:
        self.last_report = report
        restore = report.get("restore", {}) if isinstance(report, dict) else {}
        state = restore.get("state")
        if state == "verified":
            text = "测试已取消；原硬件设置已恢复并验证" if report.get("cancelled") else "测试完成；原硬件设置已恢复并验证。正式实验请记录本程序版本，并用已知电阻核验。"
            self._restore_label.setStyleSheet("color: #2e7d32;")
            self._restore_label.setToolTip("")
        else:
            detail = restore.get("diagnostic", "无法完成寄存器回读")
            text = "硬件恢复未验证，正式采集已锁定"
            self._restore_label.setStyleSheet("color: #c62828;")
            self._restore_label.setToolTip(detail)
        self._restore_label.setText(text)

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._pending_error = message
        restore = self.board.get_last_readback()
        state = restore.get("state") if isinstance(restore, dict) else None
        suffix = "；原硬件设置已恢复并验证" if state == "verified" else "；硬件恢复未验证，正式采集已锁定"
        self._restore_label.setStyleSheet("color: #c62828;")
        self._restore_label.setText(f"阻抗测试失败：{message}{suffix}")
        if state != "verified" and isinstance(restore, dict):
            self._restore_label.setToolTip(restore.get("diagnostic", ""))

    @pyqtSlot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._start.setEnabled(not running)
        self._cancel.setEnabled(running)
        self._close.setEnabled(not running)
        self._duration.setEnabled(not running)
        self._select_all.setEnabled(not running)
        self._clear_all.setEnabled(not running)
        for checkbox in self._checks:
            checkbox.setEnabled(not running)

    def _cancel_test(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel.setEnabled(False)
            self._restore_label.setText("正在取消并恢复硬件设置...")

    def reject(self) -> None:
        if self._thread is not None:
            self._cancel_test()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            self._cancel_test()
            event.ignore()
            return
        super().closeEvent(event)
