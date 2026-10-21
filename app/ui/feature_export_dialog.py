"""Metric selection dialog for reviewed Trial exports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from action_review import load_action_segments
from feature_export import export_trial_features, results_dir
from time_review import load_time_review


@dataclass(frozen=True)
class BatchExportResult:
    results: tuple
    failures: tuple[dict, ...]


class _ExportWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, trial_dirs, selected: set[str], viewer_config):
        super().__init__()
        if isinstance(trial_dirs, (str, Path)):
            trial_dirs = [trial_dirs]
        self._trial_dirs = tuple(Path(path).resolve() for path in trial_dirs)
        self._selected = selected
        self._viewer_config = viewer_config

    @pyqtSlot()
    def run(self) -> None:
        if len(self._trial_dirs) > 1:
            results = []
            failures = []
            for trial_dir in self._trial_dirs:
                try:
                    results.append(export_trial_features(
                        trial_dir,
                        self._selected,
                        viewer_config=self._viewer_config,
                    ))
                except Exception as exc:
                    traceback.print_exc()
                    failures.append({"path": str(trial_dir), "error": str(exc)})
            self.finished.emit(BatchExportResult(tuple(results), tuple(failures)))
            return
        try:
            result = export_trial_features(
                self._trial_dirs[0],
                self._selected,
                viewer_config=self._viewer_config,
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(result)


class FeatureExportDialog(QDialog):
    def __init__(self, trial_dir, viewer_config=None, parent=None):
        super().__init__(parent)
        if isinstance(trial_dir, (str, Path)):
            trial_dir = [trial_dir]
        self.trial_dirs = tuple(Path(path).resolve() for path in trial_dir)
        if not self.trial_dirs:
            raise ValueError("请至少选择一个 Trial")
        self.trial_dir = self.trial_dirs[0]
        self.viewer_config = viewer_config
        self.result = None
        self._export_thread = None
        self._export_worker = None
        self._pending_result = None
        self._pending_error = None
        review = load_time_review(self.trial_dir) if len(self.trial_dirs) == 1 else None
        actions = load_action_segments(self.trial_dir) if review else []

        title = self.trial_dir.name if len(self.trial_dirs) == 1 else f"批量 {len(self.trial_dirs)} 个 Trial"
        self.setWindowTitle(f"指标与可视化导出 - {title}")
        self.setMinimumWidth(500)
        root = QVBoxLayout(self)
        if len(self.trial_dirs) > 1:
            scope_text = (
                f"{len(self.trial_dirs)} 个 Trial；分别采用动作段、审核时间窗或完整 Trial"
            )
        else:
            scope_text = (
                f"{len(actions)} 个已审核动作段" if actions else
                "1 个已审核时间窗口" if review else
                "完整 Trial（未做时间审查）"
            )
        root.addWidget(QLabel(f"计算范围: {scope_text}"))

        group = QGroupBox("选择要计算和导出的内容")
        group_layout = QVBoxLayout(group)
        self._raw = QCheckBox("原始信号示意图（PNG）")
        self._filtered = QCheckBox("滤波后信号示意图（PNG）")
        self._envelope = QCheckBox("RMS 包络图（PNG）")
        self._spectrum = QCheckBox("带通与陷波频谱对照图（PNG）")
        self._pli = QCheckBox("工频干扰柱状图（PNG）")
        self._mar = QCheckBox("运动伪迹柱状图（PNG）")
        self._mnf = QCheckBox("MNF 平均频率（CSV + PNG）")
        self._mdf = QCheckBox("MDF 中位频率（CSV + PNG）")
        for checkbox in (self._envelope, self._mnf, self._mdf):
            checkbox.setChecked(True)
        for checkbox in (
            self._raw, self._filtered, self._envelope, self._spectrum,
            self._pli, self._mar, self._mnf, self._mdf,
        ):
            group_layout.addWidget(checkbox)
        root.addWidget(group)

        self._output = QLabel(self._output_text())
        self._output.setWordWrap(True)
        self._output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._output)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("计算并导出")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._buttons = buttons

    def _output_text(self) -> str:
        if len(self.trial_dirs) == 1:
            return f"输出目录:\n{results_dir(self.trial_dir)}"
        return "输出目录: 每个 Trial 各自的 results 文件夹"

    def selected_metrics(self) -> set[str]:
        selected = set()
        if self._raw.isChecked():
            selected.add("raw")
        if self._filtered.isChecked():
            selected.add("filtered")
        if self._envelope.isChecked():
            selected.add("envelope")
        if self._spectrum.isChecked():
            selected.add("spectrum")
        if self._pli.isChecked():
            selected.add("pli")
        if self._mar.isChecked():
            selected.add("mar")
        if self._mnf.isChecked():
            selected.add("mnf")
        if self._mdf.isChecked():
            selected.add("mdf")
        return selected

    def _export(self) -> None:
        if self._export_thread is not None:
            return
        selected = self.selected_metrics()
        if not selected:
            QMessageBox.warning(self, "无法导出", "请至少选择一个指标")
            return

        self._set_exporting(True)
        self._pending_result = None
        self._pending_error = None
        thread = QThread(self)
        worker = _ExportWorker(self.trial_dirs, selected, self.viewer_config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._export_succeeded)
        worker.failed.connect(self._export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._export_thread_finished)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()

    def _set_exporting(self, exporting: bool) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(
            not exporting
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(
            not exporting
        )
        self._output.setText(
            "正在计算并生成图片，请稍候..."
            if exporting
            else self._output_text()
        )

    @pyqtSlot(object)
    def _export_succeeded(self, result) -> None:
        self._pending_result = result

    @pyqtSlot(str)
    def _export_failed(self, message: str) -> None:
        self._pending_error = message

    @pyqtSlot()
    def _export_thread_finished(self) -> None:
        self._export_thread = None
        self._export_worker = None
        if self._pending_error is not None:
            message = self._pending_error
            self._pending_error = None
            self._set_exporting(False)
            QMessageBox.critical(self, "导出失败", message)
            return
        if self._pending_result is not None:
            self.result = self._pending_result
            self._pending_result = None
            self.accept()

    def reject(self) -> None:
        if self._export_thread is not None:
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._export_thread is not None:
            event.ignore()
            return
        super().closeEvent(event)
