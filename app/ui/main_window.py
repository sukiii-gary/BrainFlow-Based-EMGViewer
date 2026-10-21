"""主窗口 —— 聚焦版: EMG 波形 + PSD 频谱 + 精简侧边栏。

左侧为时域波形和采集控制；右侧上半管理 Session，右侧下半显示频谱和热点图。
"""
from __future__ import annotations
import ipaddress
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# 添加 emg_python 到路径
_EMG_PYTHON = Path(__file__).resolve().parents[2] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

import numpy as np
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QStatusBar, QApplication,
    QAbstractItemView, QDockWidget, QHeaderView, QMessageBox,
    QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QCheckBox, QComboBox,
)
from PyQt6.QtCore import (
    QObject, QSettings, QThread, QTimer, Qt, QUrl, pyqtSignal, pyqtSlot,
)

from .time_view import TimeViewWidget
from .psd_view import PsdViewWidget
from .activation_map import ActivationMapWidget
from .compact_sidebar import CompactSidebar
from .time_review_dialog import TimeReviewDialog
from .action_review_dialog import ActionReviewDialog
from .feature_export_dialog import FeatureExportDialog
from .history_comparison_dialog import HistoryComparisonDialog
from .impedance_dialog import ImpedanceDialog
from action_review import load_action_segments
from time_review import load_time_review
from delsys_controller import DelsysController
from camera_controller import CameraController


# 信号质量快速判定阈值
FLATLINE_STD_UV = 0.5     # < 0.5 μV std → 电极断开
SATURATION_P2P_UV = 4000  # > 4mV p2p → 饱和


def _quick_quality(raw: np.ndarray) -> str:
    """单通道快速质量判定，不依赖 emg_python。"""
    if len(raw) < 16:
        return "OFF"
    std = float(np.std(raw))
    p2p = float(np.max(raw) - np.min(raw))
    if std < FLATLINE_STD_UV:
        return "OFF"
    if p2p > SATURATION_P2P_UV:
        return "SAT"
    return "OK"  # 更细的判定（NOISE/HIGH）留给 PLI 值体现


class _ConnectWorker(QObject):
    finished = pyqtSignal(bool)
    failed = pyqtSignal(str)

    def __init__(self, board):
        super().__init__()
        self._board = board

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(bool(self._board.connect()))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)


class _HardwareApplyWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, board):
        super().__init__()
        self._board = board

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(self._board.apply_and_verify_settings())
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)


class _SyncWorker(QObject):
    finished = pyqtSignal(object)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit({"result": self.operation()})
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit({"error": str(exc) or type(exc).__name__})


class MainWindow(QMainWindow):
    """EMG Viewer 主窗口（聚焦版 + PSD + 信号质量）。"""

    def __init__(self, board_manager=None, signal_chain=None,
                 metrics_engine=None, recorder=None, config=None,
                 camera_controller=None):
        super().__init__()
        self.board = board_manager
        self.chain = signal_chain
        self.metrics = metrics_engine
        self._packet_counter_missing_batches = 0
        self._packet_counter_missing_samples = 0
        self.recorder = recorder
        self.config = config or {}

        # 状态
        self._streaming = False
        self._recording = False
        self._start_time: float | None = None
        self.n_channels = getattr(self.config, 'n_channels', 8) or 8
        self._stream_data_confirmed = False
        self._settings = QSettings("YMS", "EMG Viewer Sync Camera")
        self._camera = camera_controller or CameraController()
        self._camera_error_reported = ""
        self._delsys = DelsysController()
        self._sync_thread = None
        self._sync_worker = None
        self._sync_result = None
        self._sync_callback = None
        self._connect_thread = None
        self._connect_worker = None
        self._connect_result = None
        self._connect_error = None
        self._connect_metadata = None
        self._hw_thread = None
        self._hw_worker = None
        self._hw_report = None
        self._hw_error = None

        # 定时器
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._on_poll_timer)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._on_display_timer)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._on_metrics_timer)
        self._data_verify_timer = QTimer(self)
        self._data_verify_timer.setSingleShot(True)
        self._data_verify_timer.timeout.connect(self._on_verify_timer)
        self._camera_preview_timer = QTimer(self)
        self._camera_preview_timer.setInterval(100)
        self._camera_preview_timer.timeout.connect(self._update_camera_preview)

        self._setup_ui()
        self._setup_timers()

        # 预设录制元数据（从配置文件）
        if self.config:
            self._sidebar.set_recording_metadata(
                group=getattr(self.config, 'record_group', ''),
                subject=getattr(self.config, 'record_subject', ''),
                task=getattr(self.config, 'record_task', ''),
            )

        # 显示滤波参数
        if self.chain:
            self._sidebar.set_filter_info(
                bandpass=self.chain.bandpass,
                notch_hz=self.chain.notch_hz,
                notch_bw=self.chain.notch_bw,
                harmonics=self.chain.notch_harmonics,
                envelope_ms=self.chain.envelope_ms,
            )

        # 硬件设置控件 → 仅存内存（应用按钮或采集前统一下发）
        if self.board:
            for combo in self._sidebar._gain_combos:
                combo.currentIndexChanged.connect(self._on_hw_setting_change)
            for checkbox in self._sidebar._bias_checkboxes + self._sidebar._srb2_checkboxes:
                checkbox.toggled.connect(self._on_hw_setting_change)
            if self._sidebar._btn_apply_hw:
                self._sidebar._btn_apply_hw.clicked.connect(self._on_apply_hw_settings)
            self._on_hw_setting_change()

        for checkbox in self._sidebar._ch_checkboxes:
            checkbox.toggled.connect(self._on_visible_channels_change)
        self._on_visible_channels_change()

    def _on_visible_channels_change(self, *_args):
        """立即按通道勾选状态重排时域和频谱子图。"""
        active = self._sidebar.get_active_channels()
        self._time_view.set_active_channels(active)
        self._psd_view.set_active_channels(active)

    def _impedance_available(self) -> bool:
        return bool(
            self.board
            and getattr(self.board, "board_id", None) in (0, 5)
            and self.board.is_connected
            and not self.board.is_streaming
            and not self._recording
            and not self._delsys.active
            and self._connect_thread is None
            and self._hw_thread is None
            and self._sync_thread is None
        )

    def _history_comparison_available(self) -> bool:
        """Historical analysis is read-only, but must not compete with acquisition."""
        return bool(
            self.recorder
            and not self._recording
            and not (self.board and self.board.is_streaming)
            and not self._delsys.active
            and self._connect_thread is None
            and self._hw_thread is None
            and self._sync_thread is None
        )

    def _on_hw_setting_change(self, *_args):
        """硬件设置变化时仅存储到 board_manager，不立即下发。"""
        if not self.board:
            return
        s = self._sidebar.get_hw_settings()
        if hasattr(self.board, 'set_channel_settings'):
            self.board.set_channel_settings(s["channels"])
        elif hasattr(self.board, 'set_hw_settings'):
            self.board.set_hw_settings(
                bias_on=s["bias_on"], srb2_on=s["srb2_on"], gain=s["gain"]
            )
        self._sidebar.mark_hw_pending()

    def _on_apply_hw_settings(self):
        """停止采集时显式下发硬件设置并读取 ADS1299 寄存器。"""
        if self._sync_thread is not None:
            return
        if not self.board or not self.board.is_connected:
            self._status_bar.showMessage("请先连接板卡")
            return
        if self.board.is_streaming:
            self._status_bar.showMessage("请先停止采集再修改硬件设置")
            return
        if not hasattr(self.board, 'apply_and_verify_settings'):
            self._status_bar.showMessage("当前板卡不支持硬件设置验证")
            return
        if self._hw_thread is not None:
            return

        self._sidebar.set_apply_hw_enabled(False)
        self._sidebar.set_hw_controls_enabled(False)
        self._btn_connect.setEnabled(False)
        self._btn_start.setEnabled(False)
        self._btn_impedance.setEnabled(False)
        self._status_bar.showMessage("正在下发设置并读取 ADS1299 寄存器...")
        self._start_hw_apply_worker()

    def _start_hw_apply_worker(self) -> None:
        self._hw_report = None
        self._hw_error = None
        thread = QThread(self)
        worker = _HardwareApplyWorker(self.board)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._store_hw_report)
        worker.failed.connect(self._store_hw_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finish_hw_apply_worker)
        self._hw_thread = thread
        self._hw_worker = worker
        thread.start()

    @pyqtSlot(object)
    def _store_hw_report(self, report) -> None:
        self._hw_report = report

    @pyqtSlot(str)
    def _store_hw_error(self, message: str) -> None:
        self._hw_error = message

    @pyqtSlot()
    def _finish_hw_apply_worker(self) -> None:
        report = self._hw_report
        error = self._hw_error
        self._hw_thread = None
        self._hw_worker = None
        self._hw_report = None
        self._hw_error = None
        connected = bool(self.board and self.board.is_connected)
        self._btn_connect.setEnabled(True)
        ready = bool(getattr(self.board, "impedance_restore_verified", True))
        self._btn_start.setEnabled(connected and not self.board.is_streaming and ready)
        self._btn_impedance.setEnabled(self._impedance_available())
        self._sidebar.set_hw_controls_enabled(True)
        self._sidebar.set_apply_hw_enabled(connected and not self.board.is_streaming)
        if error:
            self._status_bar.showMessage(
                f"硬件设置异常: {error}；详细堆栈见终端"
            )
            return
        if not isinstance(report, dict):
            self._status_bar.showMessage("硬件设置异常: 板卡未返回验证结果")
            return

        self._sidebar.update_hw_status(report, self._sidebar.get_hw_settings())
        state_text = {
            "verified": "硬件设置寄存器验证通过",
            "sent_unverified": "硬件设置已发送，但当前链路无法回读",
            "failed": "硬件设置验证失败",
            "not_applicable": "模拟板无需硬件设置",
        }.get(report.get("state"), "硬件设置状态未知")
        if report.get("state") == "sent_unverified" and report.get("diagnostic"):
            state_text += f"：{report['diagnostic']}"
        self._status_bar.showMessage(state_text)

    def _setup_ui(self):
        """构建 UI 布局。"""
        self.setWindowTitle("EMG Viewer UDP Sync + Camera + Impedance — OpenBCI + Delsys")
        self.resize(1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- 工具栏 ----
        toolbar = QHBoxLayout()

        self._btn_connect = QPushButton("连接")
        self._btn_start = QPushButton("开始")
        self._btn_stop = QPushButton("停止")
        self._btn_impedance = QPushButton("阻抗测试")

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_impedance.setEnabled(False)

        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_impedance.clicked.connect(self._on_impedance_test)

        self._label_wifi_ip = QLabel("Shield IP:")
        configured_ip = getattr(self.config, 'wifi_ip', '192.168.4.1')
        saved_ip = self._settings.value("connection/wifi_ip", "", type=str)
        explicit_ip = bool(getattr(self.config, 'wifi_ip_explicit', False))
        initial_ip = saved_ip if not explicit_ip and saved_ip else configured_ip
        self._input_wifi_ip = QLineEdit(initial_ip)
        self._input_wifi_ip.setFixedWidth(130)
        self._input_wifi_ip.setPlaceholderText("例如 192.168.1.100")
        self._input_wifi_ip.setToolTip("WiFi Station 模式下 WiFi Shield 的静态 IPv4 地址")
        self._label_wifi_rate = QLabel("1000 Hz")
        self._label_wifi_rate.setToolTip("Cyton WiFi 采样率固定为 1000 Hz")
        wifi_mode = bool(self.board and getattr(self.board, 'board_id', None) == 5)
        self._label_wifi_ip.setVisible(wifi_mode)
        self._input_wifi_ip.setVisible(wifi_mode)
        self._label_wifi_rate.setVisible(wifi_mode)

        toolbar.addWidget(self._label_wifi_ip)
        toolbar.addWidget(self._input_wifi_ip)
        toolbar.addWidget(self._label_wifi_rate)
        toolbar.addWidget(self._btn_connect)
        toolbar.addWidget(self._btn_start)
        toolbar.addWidget(self._btn_stop)
        toolbar.addWidget(self._btn_impedance)
        toolbar.addStretch()

        self._status_label = QLabel("未连接")
        self._status_label.setStyleSheet("font: 13px;")
        toolbar.addWidget(self._status_label)

        root.addLayout(toolbar)

        sync_bar = QHBoxLayout()
        self._sync_enabled = QCheckBox("启用 Delsys 同步")
        self._nano_port = QComboBox()
        self._nano_port.setEditable(True)
        self._nano_port.setMinimumWidth(110)
        self._nano_port.setToolTip("Nano 的 COM 口，连接会使部分 Nano 自动复位")
        self._nano_refresh = QPushButton("刷新串口")
        self._nano_connect = QPushButton("连接 Nano")
        self._sync_label = QLabel("同步关闭")
        self._sync_label.setWordWrap(True)
        self._sync_label.setToolTip("D2 → 单一 START/STOP 信号，D3 → LOW 返回，100 ms；连续触发至少间隔 2 秒")
        for widget in (self._sync_enabled, self._nano_port, self._nano_refresh,
                       self._nano_connect, self._sync_label):
            sync_bar.addWidget(widget)
        sync_bar.setStretch(4, 1)
        root.addLayout(sync_bar)
        self._nano_refresh.clicked.connect(self._refresh_nano_ports)
        self._nano_connect.clicked.connect(self._on_nano_connect)
        self._sync_enabled.toggled.connect(self._on_sync_toggle)
        self._refresh_nano_ports()

        camera_bar = QHBoxLayout()
        self._camera_enabled = QCheckBox("启用摄像机三设备同步")
        self._camera_source = QComboBox()
        self._camera_source.setEditable(True)
        self._camera_source.addItems([str(index) for index in range(4)])
        self._camera_source.setCurrentText(
            self._settings.value("camera/source", "0", type=str)
        )
        self._camera_source.setToolTip("Windows UVC 摄像机索引，通常内置摄像头为 0")
        self._camera_resolution = QComboBox()
        self._camera_resolution.addItems(["1920x1080", "1280x720", "640x480"])
        self._camera_resolution.setCurrentText(
            self._settings.value("camera/resolution", "1920x1080", type=str)
        )
        self._camera_fps = QComboBox()
        self._camera_fps.addItems(["30", "25", "20", "15"])
        self._camera_fps.setCurrentText(
            self._settings.value("camera/fps", "30", type=str)
        )
        self._camera_connect = QPushButton("连接摄像机")
        self._camera_label = QLabel("摄像机未连接")
        self._camera_label.setWordWrap(True)
        for widget in (self._camera_enabled, self._camera_source,
                       self._camera_resolution, self._camera_fps,
                       self._camera_connect, self._camera_label):
            camera_bar.addWidget(widget)
        camera_bar.setStretch(5, 1)
        root.addLayout(camera_bar)
        self._camera_preview = QLabel("无摄像机预览")
        self._camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_preview.setFixedSize(240, 135)
        self._camera_preview.setStyleSheet("border: 1px solid #777; background: #111; color: #bbb;")
        root.addWidget(self._camera_preview, 0, Qt.AlignmentFlag.AlignLeft)
        self._camera_connect.clicked.connect(self._on_camera_connect)
        self._camera_enabled.toggled.connect(self._on_camera_toggle)

        # ---- 主体: 波形 + 采集控制侧边栏 ----
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        display_sec = getattr(self.config, 'display_window_sec', 15.0) or 15.0
        self._time_view = TimeViewWidget(n_channels=self.n_channels, display_sec=display_sec)
        top_splitter.addWidget(self._time_view)

        self._sidebar = CompactSidebar(n_channels=self.n_channels)
        self._sidebar.setMinimumWidth(240)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sidebar_scroll.setMinimumWidth(250)
        sidebar_scroll.setMaximumWidth(290)
        sidebar_scroll.setWidget(self._sidebar)
        top_splitter.addWidget(sidebar_scroll)
        self._sidebar_scroll = sidebar_scroll

        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)

        # PSD 与热点图放入右侧 Session dock 的下半区。
        self._psd_view = PsdViewWidget(n_channels=self.n_channels)
        self._activation_map = ActivationMapWidget(n_channels=self.n_channels)
        analysis_tabs = QTabWidget()
        analysis_tabs.addTab(self._psd_view, "频谱")
        analysis_tabs.addTab(self._activation_map, "激活热点")
        analysis_tabs.setMinimumHeight(240)
        self._analysis_tabs = analysis_tabs

        root.addWidget(top_splitter, 1)

        # ---- 底部状态栏 ----
        self._status_bar = QStatusBar()
        self._status_bar.showMessage("就绪 — 请点击「连接」开始")
        self.setStatusBar(self._status_bar)
        self._setup_session_dock()

    def _setup_session_dock(self):
        """Create the current Session Trial browser."""
        dock = QDockWidget("当前 Session 数据", self)
        dock.setObjectName("currentSessionDock")
        dock.setMinimumWidth(360)
        dock.setMaximumWidth(460)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = QWidget(dock)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        root_row = QHBoxLayout()
        root_path = self.recorder.data_dir if self.recorder else None
        root_text = str(root_path) if root_path else "录制器未配置"
        self._recordings_path_label = QLabel(f"保存根目录:\n{root_text}")
        self._recordings_path_label.setWordWrap(True)
        self._recordings_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._btn_open_recordings = QPushButton("打开 recordings")
        self._btn_open_recordings.setEnabled(bool(root_path))
        self._btn_open_recordings.clicked.connect(self._on_open_recordings)
        root_row.addWidget(self._recordings_path_label, 1)
        root_row.addWidget(self._btn_open_recordings)
        layout.addLayout(root_row)

        path_row = QHBoxLayout()
        self._session_path_label = QLabel("当前 Session: 尚未建立")
        self._session_path_label.setWordWrap(True)
        self._session_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._btn_open_session = QPushButton("打开 Session")
        self._btn_new_session = QPushButton("新建 Session")
        self._btn_delete_session = QPushButton("删除 Session")
        self._btn_open_session.setEnabled(False)
        self._btn_new_session.setEnabled(False)
        self._btn_delete_session.setEnabled(False)
        self._btn_open_session.clicked.connect(self._on_open_session)
        self._btn_new_session.clicked.connect(self._on_new_session)
        self._btn_delete_session.clicked.connect(self._on_delete_session)
        path_row.addWidget(self._session_path_label, 1)
        path_row.addWidget(self._btn_open_session)
        path_row.addWidget(self._btn_new_session)
        path_row.addWidget(self._btn_delete_session)
        layout.addLayout(path_row)

        self._session_table = QTableWidget(0, 8)
        self._session_table.setHorizontalHeaderLabels(
            [
                "Trial", "任务", "开始时间", "时长", "样本数", "状态",
                "时间审查", "动作切分",
            ]
        )
        self._session_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._session_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._session_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self._session_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.itemDoubleClicked.connect(
            lambda _item: self._on_open_selected_trial()
        )
        self._session_table.itemSelectionChanged.connect(
            self._update_trial_action_buttons
        )
        layout.addWidget(self._session_table)

        review_actions = QHBoxLayout()
        self._btn_open_trial = QPushButton("打开 Trial")
        self._btn_review_time = QPushButton("时间切分审查")
        self._btn_review_action = QPushButton("动作切分审查")
        self._btn_export_features = QPushButton("导出指标")
        self._btn_compare_history = QPushButton("历史段对比")
        self._btn_delete_trial = QPushButton("删除 Trial")
        self._btn_open_trial.clicked.connect(self._on_open_selected_trial)
        self._btn_review_time.clicked.connect(self._on_review_selected_trial)
        self._btn_review_action.clicked.connect(self._on_review_action_segments)
        self._btn_export_features.clicked.connect(self._on_export_features)
        self._btn_compare_history.clicked.connect(self._on_compare_history)
        self._btn_delete_trial.clicked.connect(self._on_delete_selected_trial)
        review_actions.addWidget(self._btn_open_trial)
        review_actions.addWidget(self._btn_review_time)
        review_actions.addWidget(self._btn_review_action)
        layout.addLayout(review_actions)
        data_actions = QHBoxLayout()
        data_actions.addWidget(self._btn_export_features)
        data_actions.addWidget(self._btn_compare_history)
        data_actions.addWidget(self._btn_delete_trial)
        data_actions.addStretch()
        layout.addLayout(data_actions)

        panel.setMinimumHeight(240)
        right_splitter = QSplitter(Qt.Orientation.Vertical, dock)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(panel)
        right_splitter.addWidget(self._analysis_tabs)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([360, 360])

        dock.setWidget(right_splitter)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._session_dock = dock
        self._session_analysis_splitter = right_splitter
        self.resizeDocks([dock], [400], Qt.Orientation.Horizontal)
        dock.show()
        self._refresh_session_trials()

    def _setup_timers(self):
        poll_ms = int(getattr(self.config, "poll_interval_ms", 40) or 40)
        display_ms = int(getattr(self.config, "ui_refresh_ms", 100) or 100)
        self._poll_timer.setInterval(max(20, poll_ms))
        if self.n_channels >= 8:
            display_ms = max(200, display_ms)
        self._display_timer.setInterval(max(100, display_ms))
        self._metrics_timer.setInterval(1000)

    # ====== 按钮事件 ======

    @staticmethod
    def _click_event(name):
        return {"event": name, "unix_ns": time.time_ns(),
                "monotonic_ns": time.perf_counter_ns()}

    def _refresh_nano_ports(self):
        selected = self._nano_port.currentText()
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            self._nano_port.clear()
            self._nano_port.addItems([port.device for port in ports])
            if selected:
                self._nano_port.setCurrentText(selected)
            elif ports:
                usb = next((port for port in ports if port.vid is not None), ports[0])
                self._nano_port.setCurrentText(usb.device)
        except Exception as exc:
            self._sync_label.setText(f"串口枚举失败：{exc}")

    def _on_sync_toggle(self, enabled):
        if not enabled and hasattr(self, "_camera_enabled") and self._camera_enabled.isChecked():
            self._camera_enabled.setChecked(False)
        self._sync_label.setText(
            ("Nano 已连接；请先在 Delsys 中等待触发" if self._delsys.ready else "请连接 Nano")
            if enabled else "同步关闭"
        )

    def _on_camera_toggle(self, enabled):
        if enabled and not self._sync_enabled.isChecked():
            self._sync_enabled.setChecked(True)
        self._camera_label.setText(
            "摄像机已就绪" if self._camera.ready else
            "请先连接摄像机" if enabled else "摄像机同步关闭"
        )

    def _camera_parameters(self):
        source_text = self._camera_source.currentText().strip()
        source = int(source_text) if source_text.isdigit() else source_text
        width, height = (
            int(value) for value in self._camera_resolution.currentText().split("x", 1)
        )
        return source, width, height, float(self._camera_fps.currentText())

    def _on_camera_connect(self):
        if self._sync_thread is not None or (self.board and self.board.is_streaming):
            return
        disconnect = self._camera.ready
        if disconnect:
            operation = self._camera.disconnect
            self._camera_label.setText("正在断开摄像机…")
        else:
            try:
                source, width, height, fps = self._camera_parameters()
            except (TypeError, ValueError):
                self._camera_label.setText("摄像机参数无效")
                return
            operation = lambda: self._camera.connect(source, width, height, fps)
            self._camera_label.setText("正在连接摄像机…")

        def finished(outcome):
            if "error" in outcome:
                self._camera_label.setText(f"摄像机连接失败：{outcome['error']}")
                return
            if disconnect:
                self._camera_preview_timer.stop()
                self._camera_preview.clear()
                self._camera_preview.setText("无摄像机预览")
                self._camera_label.setText("摄像机已断开")
            else:
                self._settings.setValue("camera/source", self._camera_source.currentText())
                self._settings.setValue("camera/resolution", self._camera_resolution.currentText())
                self._settings.setValue("camera/fps", self._camera_fps.currentText())
                self._settings.sync()
                self._camera_preview_timer.start()
                self._camera_label.setText(
                    f"摄像机已连接：{width}x{height} @ {fps:g} FPS"
                )
            self._camera_connect.setText(
                "断开摄像机" if self._camera.ready else "连接摄像机"
            )

        self._run_sync_job(operation, finished)

    def _update_camera_preview(self):
        frame = self._camera.latest_frame
        status = self._camera.status()
        camera_error = status.get("error", "")
        if camera_error and camera_error != self._camera_error_reported:
            self._camera_error_reported = camera_error
            self._camera_label.setText(f"摄像机异常：{camera_error}")
            if self._delsys.active:
                self._delsys.note_event("camera_runtime_failed", error=camera_error)
        if frame is None:
            return
        height, width = frame.shape[:2]
        image = QImage(
            frame.data, width, height, int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        self._camera_preview.setPixmap(QPixmap.fromImage(image).scaled(
            self._camera_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _set_sync_busy(self, busy):
        streaming = bool(self.board and self.board.is_streaming)
        connected = bool(self.board and self.board.is_connected)
        self._btn_connect.setEnabled(not busy)
        ready = bool(getattr(self.board, "impedance_restore_verified", True))
        self._btn_start.setEnabled(not busy and connected and not streaming and ready)
        self._btn_stop.setEnabled(not busy and streaming)
        self._btn_impedance.setEnabled(not busy and self._impedance_available())
        if hasattr(self, "_btn_compare_history"):
            self._btn_compare_history.setEnabled(
                not busy and self._history_comparison_available()
            )
        if hasattr(self, "_btn_new_session"):
            session = self.recorder.session_dir if self.recorder else None
            self._btn_new_session.setEnabled(
                bool(not busy and connected and not streaming and self.recorder)
            )
            self._btn_delete_session.setEnabled(
                bool(not busy and not streaming and session and session.is_dir())
            )
            if busy:
                self._btn_open_trial.setEnabled(False)
                self._btn_review_time.setEnabled(False)
                self._btn_review_action.setEnabled(False)
                self._btn_export_features.setEnabled(False)
                self._btn_delete_trial.setEnabled(False)
            else:
                self._update_trial_action_buttons()
        self._sync_enabled.setEnabled(not busy and not streaming)
        self._nano_port.setEnabled(not busy and not streaming and not self._delsys.ready)
        self._nano_refresh.setEnabled(not busy and not streaming and not self._delsys.ready)
        self._nano_connect.setEnabled(not busy and not streaming)
        self._camera_enabled.setEnabled(not busy and not streaming)
        camera_settings_enabled = not busy and not streaming and not self._camera.ready
        self._camera_source.setEnabled(camera_settings_enabled)
        self._camera_resolution.setEnabled(camera_settings_enabled)
        self._camera_fps.setEnabled(camera_settings_enabled)
        self._camera_connect.setEnabled(not busy and not streaming)
        self._sidebar.set_apply_hw_enabled(not busy and connected and not streaming)
        self._sidebar.set_hw_controls_enabled(not busy and not streaming)

    def _on_impedance_test(self):
        if not self.board or not self.board.is_connected:
            self._status_bar.showMessage("请先连接 Cyton 板卡")
            return
        if getattr(self.board, "board_id", None) not in (0, 5):
            self._status_bar.showMessage("阻抗测试仅支持 Cyton 串口或 Cyton WiFi")
            return
        if not self._impedance_available():
            self._status_bar.showMessage("请先停止采集并等待当前硬件操作完成")
            return

        dialog = ImpedanceDialog(self.board, parent=self)
        dialog.exec()
        if hasattr(self.board, "get_last_readback"):
            self._sidebar.update_hw_status(
                self.board.get_last_readback(), self._sidebar.get_hw_settings()
            )
        restored = bool(
            getattr(self.board, "impedance_restore_verified", True)
        )
        self._set_sync_busy(False)
        if dialog.last_report is not None:
            self._status_bar.showMessage(
                "阻抗测试结束，原硬件设置已恢复并验证"
                if restored
                else "阻抗测试结束，但硬件恢复未验证；正式采集已锁定"
            )

    def _run_sync_job(self, operation, callback):
        if self._sync_thread is not None:
            return
        thread = QThread(self)
        worker = _SyncWorker(operation)
        worker.moveToThread(thread)
        self._sync_result = None
        self._sync_callback = callback
        thread.started.connect(worker.run)
        worker.finished.connect(self._store_sync_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finish_sync_job)
        self._sync_thread = thread
        self._sync_worker = worker
        self._set_sync_busy(True)
        thread.start()

    @pyqtSlot(object)
    def _store_sync_result(self, result):
        self._sync_result = result

    @pyqtSlot()
    def _finish_sync_job(self):
        result = self._sync_result or {"error": "同步任务未返回结果"}
        callback = self._sync_callback
        self._sync_thread = None
        self._sync_worker = None
        self._sync_result = None
        self._sync_callback = None
        try:
            callback(result)
        except Exception as exc:
            traceback.print_exc()
            self._sync_label.setText(f"同步流程异常：{exc}；请检查 Delsys 状态")
        finally:
            self._nano_connect.setText("断开 Nano" if self._delsys.ready else "连接 Nano")
            self._set_sync_busy(False)

    def _on_nano_connect(self):
        if self._sync_thread is not None or self._connect_thread is not None or self._hw_thread is not None:
            return
        if self.board and self.board.is_streaming:
            return
        port = self._nano_port.currentText().strip()
        if not self._delsys.ready and not port:
            self._sync_label.setText("请选择 Nano 串口")
            return
        disconnect = self._delsys.ready
        self._sync_label.setText("正在断开 Nano…" if disconnect else "正在连接 Nano，等待复位完成…")
        self._run_sync_job(
            self._delsys.disconnect if disconnect else lambda: self._delsys.connect(port),
            lambda outcome: self._sync_label.setText(
                f"Nano 连接失败：{outcome['error']}" if "error" in outcome else
                ("Nano 已断开" if disconnect else f"Nano {port} 已就绪；Delsys 请先等待触发")
            ),
        )

    def _on_connect(self):
        if self._sync_thread is not None:
            return
        if self._delsys.active:
            self._sync_label.setText("请先点击停止，完成当前同步 Trial 后再断开")
            return
        if not self.board:
            self._status_bar.showMessage("错误: board_manager 未注入")
            return
        if self._connect_thread is not None:
            return

        if self.board.is_connected:
            self._stop_all()
            self.board.disconnect()
            if self.recorder:
                self.recorder.end_session()
            self._activation_map.clear_display(reset_history=True)
            self._sidebar.set_apply_hw_enabled(False)
            self._btn_connect.setText("连接")
            self._btn_start.setEnabled(False)
            self._btn_impedance.setEnabled(False)
            self._input_wifi_ip.setEnabled(True)
            self._status_label.setText("未连接")
            self._status_label.setStyleSheet("font: 13px;")
            self._status_bar.showMessage("已断开")
        else:
            if getattr(self.board, 'board_id', None) == 5:
                try:
                    wifi_ip = str(ipaddress.IPv4Address(self._input_wifi_ip.text().strip()))
                    self.board.configure_wifi_address(wifi_ip)
                except (ValueError, RuntimeError) as e:
                    self._status_bar.showMessage(f"Shield IP 无效: {e}")
                    self._input_wifi_ip.setFocus()
                    return
                self._input_wifi_ip.setText(wifi_ip)
                if self.config:
                    self.config.wifi_ip = wifi_ip

            session_metadata = self._sidebar.get_recording_metadata()

            self._input_wifi_ip.setEnabled(False)
            self._btn_connect.setEnabled(False)
            self._status_label.setText("正在连接")
            self._status_label.setStyleSheet("font: 13px; color: #FF9800;")
            self._status_bar.showMessage("正在检查 Shield 并建立 1000Hz 数据链路...")
            self._start_connect_worker(session_metadata)

    def _start_connect_worker(self, session_metadata: dict) -> None:
        self._connect_result = None
        self._connect_error = None
        self._connect_metadata = session_metadata
        thread = QThread(self)
        worker = _ConnectWorker(self.board)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._store_connect_result)
        worker.failed.connect(self._store_connect_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finish_connect_worker)
        self._connect_thread = thread
        self._connect_worker = worker
        thread.start()

    @pyqtSlot(bool)
    def _store_connect_result(self, ok: bool) -> None:
        self._connect_result = ok

    @pyqtSlot(str)
    def _store_connect_error(self, message: str) -> None:
        self._connect_error = message

    @pyqtSlot()
    def _finish_connect_worker(self) -> None:
        ok = bool(self._connect_result)
        error = self._connect_error
        session_metadata = self._connect_metadata or {}
        self._connect_thread = None
        self._connect_worker = None
        self._connect_result = None
        self._connect_error = None
        self._connect_metadata = None
        self._btn_connect.setEnabled(True)

        if error:
            self._input_wifi_ip.setEnabled(True)
            self._status_label.setText("连接异常")
            self._status_label.setStyleSheet("font: 13px; color: #F44336;")
            self._status_bar.showMessage(f"连接异常: {error}；详细堆栈见终端")
            return
        if not ok:
            self._input_wifi_ip.setEnabled(True)
            self._status_label.setText("连接失败")
            self._status_label.setStyleSheet("font: 13px; color: #F44336;")
            connection = getattr(self.board, 'last_connection_status', {})
            detail = connection.get("detail", "") if isinstance(connection, dict) else ""
            self._status_bar.showMessage(detail or "连接失败，请查看终端日志")
            return

        session_error = ""
        if self.recorder and session_metadata.get("subject_id", "").strip():
            try:
                self.recorder.start_session(session_metadata["subject_id"])
                self._refresh_session_trials()
            except Exception as exc:
                traceback.print_exc()
                session_error = f"；Session 尚未建立: {exc}"
        actual_fs = self.board.sample_rate
        if self.chain:
            self.chain.update_fs(actual_fs)
        if self.metrics:
            self.metrics.sample_rate = actual_fs

        self._btn_connect.setText("断开")
        self._btn_start.setEnabled(True)
        self._btn_impedance.setEnabled(self._impedance_available())
        self._sidebar.set_apply_hw_enabled(True)
        self._status_label.setText(
            "已连接 (UDP 待收数验证)"
            if getattr(self.board, 'board_id', None) == 5 else "已连接"
        )
        self._status_label.setStyleSheet("font: 13px; color: #4CAF50;")
        if getattr(self.board, 'board_id', None) == 5:
            self._settings.setValue("connection/wifi_ip", self.board.wifi_ip)
            self._settings.sync()
        connection = getattr(self.board, 'last_connection_status', {})
        local_ip = connection.get("local_ip", "") if isinstance(connection, dict) else ""
        route_text = f", 本机 {local_ip}:{self.board.wifi_port}" if local_ip else ""
        self._status_bar.showMessage(
            f"已连接: {actual_fs}Hz, {self.board.n_channels} 通道{route_text}"
            + (" UDP" if getattr(self.board, 'board_id', None) == 5 else "")
            + (
                "；请填写被试 ID 后开始采集"
                if self.recorder and not session_metadata.get("subject_id", "").strip()
                else ""
            )
            + session_error
        )

    def _on_start(self):
        click = self._click_event("start_click")
        if self._sync_thread is not None or self._hw_thread is not None or self._connect_thread is not None:
            return
        if not self.board or not self.chain:
            return
        if self.board.is_streaming:
            return

        sidebar_meta = self._sidebar.get_recording_metadata()
        if self.recorder and not sidebar_meta.get("subject_id", "").strip():
            self._status_bar.showMessage("请先填写被试 ID，再开始采集")
            self._sidebar.focus_subject_input()
            return
        if self.recorder and not self.recorder.is_session_active:
            try:
                self.recorder.start_session(sidebar_meta["subject_id"])
                self._refresh_session_trials()
            except Exception as e:
                self._status_bar.showMessage(f"无法创建数据目录: {e}")
                return

        if self._sync_enabled.isChecked():
            if not self.recorder or not self._delsys.ready:
                self._sync_label.setText("同步启动需要已连接 Nano 和可用录制器")
                return
            if self._camera_enabled.isChecked() and not self._camera.ready:
                self._camera_label.setText("三设备同步需要先连接摄像机")
                return
            self._delsys.camera = self._camera if self._camera_enabled.isChecked() else None
            self._camera_error_reported = ""
            metadata = {"sampling_rate": self.board.sample_rate,
                        "n_channels": self.n_channels,
                        "start_time": datetime.fromtimestamp(click["unix_ns"] / 1e9).isoformat(),
                        **sidebar_meta}
            self._sync_label.setText(
                "正在准备并启动三套采集…" if self._delsys.camera else
                "正在准备录制并启动两套采集…"
            )
            self._run_sync_job(
                lambda: self._delsys.start(self.board, self.recorder, metadata, click),
                self._finish_sync_start,
            )
            return

        ok = self.board.start_stream()
        if ok:
            if self.recorder:
                metadata = {
                    "sampling_rate": self.board.sample_rate,
                    "n_channels": self.n_channels,
                    "start_time": datetime.now().isoformat(),
                }
                metadata.update(sidebar_meta)
                if not self.recorder.start_recording(metadata):
                    self.board.stop_stream()
                    self._status_bar.showMessage("Trial 数据文件创建失败，已停止采集")
                    return
                self._recording = True
                self._refresh_session_trials()

            # 更新硬件设置读回状态
            if hasattr(self.board, 'get_last_readback'):
                rb = self.board.get_last_readback()
                target = self._sidebar.get_hw_settings()
                self._sidebar.update_hw_status(rb, target)

            self._streaming = True
            self._stream_data_confirmed = False
            self._start_time = time.time()
            if self.chain:
                self.chain.clear()
            self._time_view.clear_data()
            self._psd_view.clear_data()
            self._activation_map.clear_display()
            if self.metrics:
                self.metrics.reset()
            self._packet_counter_missing_batches = 0
            self._packet_counter_missing_samples = 0

            self._poll_timer.start()
            self._display_timer.start()
            self._metrics_timer.start()
            self._data_verify_timer.start(2000)

            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._btn_impedance.setEnabled(False)
            self._sidebar.set_apply_hw_enabled(False)
            self._sidebar.set_hw_controls_enabled(False)
            self._status_label.setText("采集中 (等待数据验证)")
            self._status_label.setStyleSheet("font: 13px; color: #FF9800;")
            self._status_bar.showMessage(
                f"开始采集 @ {self.board.sample_rate}Hz"
            )
            self._set_sync_busy(False)
        else:
            if hasattr(self.board, 'get_last_readback'):
                self._sidebar.update_hw_status(
                    self.board.get_last_readback(), self._sidebar.get_hw_settings()
                )
            self._status_bar.showMessage("启动采集失败，请检查硬件设置状态和终端日志")

    def _on_stop(self):
        click = self._click_event("stop_click")
        if self._sync_thread is not None:
            return
        if self._delsys.active:
            for timer in (self._poll_timer, self._display_timer, self._metrics_timer, self._data_verify_timer):
                timer.stop()
            self._sync_label.setText("正在发送 Delsys 停止脉冲…")
            self._run_sync_job(lambda: self._delsys.stop_trigger(click), self._finish_sync_stop)
            return
        trial = self._stop_all()
        self._status_label.setText("已连接")
        self._status_label.setStyleSheet("font: 13px; color: #4CAF50;")
        if trial:
            if trial.get("write_error"):
                self._status_bar.showMessage(
                    f"Trial 写入失败: {trial['write_error']}"
                )
            else:
                self._status_bar.showMessage(f"Trial 已保存: {trial['txt_path']}")
        else:
            self._status_bar.showMessage("采集已停止")

    def _finish_sync_start(self, outcome):
        result = outcome.get("result") or {}
        running = bool(
            result.get("running") and self.board and self.board.is_streaming
        )
        if not running:
            self._recording = bool(self.recorder and self.recorder.is_recording)
            self._stop_all(reason="start_failed")
            self._sync_label.setText("启动失败：" + outcome.get("error", result.get("warning", "请检查设备")))
            return
        self._streaming = True
        self._recording = bool(self.recorder and self.recorder.is_recording)
        self._stream_data_confirmed = False
        sync_start_ns = result.get("sync_start_monotonic_ns")
        self._start_time = (
            time.time() - (time.perf_counter_ns() - sync_start_ns) / 1_000_000_000
            if sync_start_ns is not None else time.time()
        )
        self.chain.clear()
        self._time_view.clear_data()
        self._psd_view.clear_data()
        self._activation_map.clear_display()
        if self.metrics:
            self.metrics.reset()
        self._packet_counter_missing_batches = 0
        self._packet_counter_missing_samples = 0
        if result.get("report"):
            self._sidebar.update_hw_status(result["report"], self._sidebar.get_hw_settings())
        self._poll_timer.start()
        self._display_timer.start()
        self._metrics_timer.start()
        self._data_verify_timer.start(2000)
        self._refresh_session_trials()
        self._status_label.setText("采集中 (等待数据验证)")
        warning = outcome.get("error") or result.get("warning")
        self._sync_label.setText(
            f"触发未确认：{warning}；请检查 Delsys" if warning else
            "START 脉冲已执行；请确认 Delsys 正在采集"
        )

    def _finish_sync_stop(self, outcome):
        result = outcome.get("result") or {}
        trigger_warning = outcome.get("error") or result.get("warning", "")
        stop_monotonic_ns = result.get("sync_stop_monotonic_ns")
        stop_unix_ns = result.get("sync_stop_unix_ns")
        if (
            self.recorder and self.recorder.is_recording
            and stop_monotonic_ns is not None and stop_unix_ns is not None
        ):
            try:
                self.recorder.mark_acquisition_stop(stop_monotonic_ns, stop_unix_ns)
            except Exception as exc:
                boundary_warning = f"STOP 时间基准写入失败: {exc}"
                self._delsys.note_warning(boundary_warning, "stop_alignment_failed")
                trigger_warning = "；".join(
                    item for item in (trigger_warning, boundary_warning) if item
                )
        trial = self._stop_all(
            reason="user_stop", stop_boundary_unix_ns=stop_unix_ns
        )
        self._status_label.setText("已连接")
        finish_warning = self._delsys.warning
        if trigger_warning:
            sync_message = f"STOP 未确认：{trigger_warning}；请手动检查 Delsys"
        elif finish_warning:
            sync_message = f"STOP 脉冲已执行；{finish_warning}"
        else:
            sync_message = "STOP 脉冲已执行"
        if trial and trial.get("sync_report_path"):
            sync_message += "；同步报告 JSON 已生成"
        self._sync_label.setText(sync_message)
        if trial:
            self._status_bar.showMessage(
                f"Trial 写入失败：{trial['write_error']}" if trial.get("write_error") else
                (
                    f"Trial 已保存：{trial['txt_path']}；同步报告 JSON："
                    f"{trial['sync_report_path']}"
                    if trial.get("sync_report_path") else
                    f"Trial 已保存：{trial['txt_path']}"
                )
            )

    # ====== 当前 Session 数据 ======

    def _refresh_session_trials(self):
        if not hasattr(self, "_session_table"):
            return
        session_dir = self.recorder.session_dir if self.recorder else None
        session_exists = bool(session_dir and session_dir.is_dir())
        if session_exists:
            self._session_path_label.setText(
                f"当前 Session:\n{session_dir.resolve()}"
            )
            self._btn_open_session.setEnabled(True)
        else:
            self._session_path_label.setText("当前 Session: 尚未建立")
            self._btn_open_session.setEnabled(False)

        trials = self.recorder.list_trials() if self.recorder and session_exists else []
        current = self.recorder.current_trial_info if self.recorder else None
        if current:
            trials.append(current)
        self._session_table.setRowCount(len(trials))
        for row, trial in enumerate(trials):
            started = str(trial.get("start_time", "")).replace("T", " ")[:19]
            review = None
            actions = []
            if not trial.get("recording"):
                review = load_time_review(trial["path"])
                actions = load_action_segments(trial["path"])
            values = [
                f"Trial {int(trial['number']):03d}",
                str(trial.get("task", "") or "未命名"),
                started,
                f"{float(trial.get('duration_sec', 0.0)):.2f} s",
                str(int(trial.get("samples", 0))),
                (
                    "录制中" if trial.get("recording")
                    else "写入失败" if trial.get("write_error")
                    else "已保存"
                ),
                "已审查" if review else "未审查" if not trial.get("recording") else "--",
                f"{len(actions)} 段已审查" if actions else "未切分" if not trial.get("recording") else "--",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(trial["path"]))
                self._session_table.setItem(row, column, item)
        has_rows = bool(trials)
        self._btn_compare_history.setEnabled(self._history_comparison_available())
        session_busy = bool(
            self._recording or self._sync_thread is not None
            or self._connect_thread is not None or self._hw_thread is not None
        )
        board_connected = bool(self.board and self.board.is_connected)
        self._btn_new_session.setEnabled(
            bool(self.recorder and board_connected and not session_busy)
        )
        self._btn_delete_session.setEnabled(session_exists and not session_busy)
        if has_rows and self._session_table.currentRow() < 0:
            self._session_table.selectRow(0)
        self._update_trial_action_buttons()

    def _update_trial_action_buttons(self):
        paths = self._selected_trial_paths()
        completed = bool(
            paths and not self._recording
            and all(path.is_dir() for path in paths)
        )
        single_path = paths[0] if completed and len(paths) == 1 else None
        self._btn_open_trial.setEnabled(single_path is not None)
        self._btn_review_time.setEnabled(single_path is not None)
        self._btn_review_action.setEnabled(
            bool(single_path and load_time_review(single_path) is not None)
        )
        self._btn_export_features.setEnabled(completed)
        self._btn_delete_trial.setEnabled(completed)

    def _selected_trial_paths(self) -> list[Path]:
        selection = self._session_table.selectionModel()
        if selection is None:
            return []
        paths = []
        for index in sorted(selection.selectedRows(0), key=lambda item: item.row()):
            value = index.data(Qt.ItemDataRole.UserRole)
            path = Path(value) if value else None
            if path is not None and path not in paths:
                paths.append(path)
        return paths

    def _selected_trial_path(self) -> Path | None:
        paths = self._selected_trial_paths()
        return paths[0] if len(paths) == 1 else None

    def _open_directory(self, path: Path | None):
        if not path or not path.is_dir():
            self._status_bar.showMessage("数据目录不存在")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self._status_bar.showMessage(f"无法打开目录: {path}")

    def _on_open_session(self):
        self._open_directory(self.recorder.session_dir if self.recorder else None)

    def _session_operation_blocked(self) -> bool:
        return bool(
            self._recording or self._sync_thread is not None
            or self._connect_thread is not None or self._hw_thread is not None
        )

    def _on_new_session(self):
        if not self.recorder or self._session_operation_blocked():
            self._status_bar.showMessage("请先停止采集并等待当前操作完成")
            return
        if not self.board or not self.board.is_connected:
            self._status_bar.showMessage("请先连接 OpenBCI")
            return
        subject = self._sidebar.get_recording_metadata().get("subject_id", "").strip()
        if not subject:
            self._status_bar.showMessage("请先填写被试 ID")
            self._sidebar.focus_subject_input()
            return
        try:
            self.recorder.end_session()
            session = self.recorder.start_session(subject)
        except OSError as exc:
            self._status_bar.showMessage(f"无法新建 Session: {exc}")
            return
        self._refresh_session_trials()
        self._status_bar.showMessage(f"已新建 Session: {session}")

    def _on_delete_session(self):
        if not self.recorder or self._session_operation_blocked():
            self._status_bar.showMessage("请先停止采集并等待当前操作完成")
            return
        session = self.recorder.session_dir
        if session is None:
            self._status_bar.showMessage("当前没有可删除的 Session")
            return
        answer = QMessageBox.question(
            self,
            "永久删除 Session",
            f"确定永久删除当前 Session 及其中全部 Trial 吗？\n\n{session}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self.recorder.delete_current_session()
        except (OSError, ValueError, RuntimeError) as exc:
            self._status_bar.showMessage(f"无法删除 Session: {exc}")
            return
        self._refresh_session_trials()
        self._status_bar.showMessage(
            f"已删除 Session: {deleted}；可新建 Session 或直接开始采集"
        )

    def _on_open_recordings(self):
        self._open_directory(self.recorder.data_dir if self.recorder else None)

    def _on_open_selected_trial(self):
        self._open_directory(self._selected_trial_path())

    def _on_review_selected_trial(self):
        if self._recording:
            self._status_bar.showMessage("请先停止当前 Trial 再进行时间审查")
            return
        path = self._selected_trial_path()
        if path is None:
            self._status_bar.showMessage("请先选择一个 Trial")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            dialog = TimeReviewDialog(path, viewer_config=self.config, parent=self)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._status_bar.showMessage(f"无法打开时间审查: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if dialog.exec():
            self._refresh_session_trials()
            record = dialog.saved_record or {}
            self._status_bar.showMessage(
                "时间范围已审核: "
                f"{float(record.get('start_sec', 0.0)):.3f} - "
                f"{float(record.get('end_sec', 0.0)):.3f} s"
            )

    def _on_review_action_segments(self):
        if self._recording:
            self._status_bar.showMessage("请先停止当前 Trial 再进行动作切分")
            return
        path = self._selected_trial_path()
        if path is None:
            self._status_bar.showMessage("请先选择一个 Trial")
            return
        if load_time_review(path) is None:
            self._status_bar.showMessage("请先完成该 Trial 的时间切分审查")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            dialog = ActionReviewDialog(path, viewer_config=self.config, parent=self)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._status_bar.showMessage(f"无法打开动作切分: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if dialog.exec():
            self._refresh_session_trials()
            self._status_bar.showMessage(
                f"动作切分已审核: {len(dialog.saved_segments)} 段"
            )

    def _on_export_features(self):
        if self._recording:
            self._status_bar.showMessage("请先停止当前 Trial 再导出指标")
            return
        paths = self._selected_trial_paths()
        if not paths:
            self._status_bar.showMessage("请先选择至少一个 Trial")
            return
        try:
            dialog = FeatureExportDialog(paths, viewer_config=self.config, parent=self)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._status_bar.showMessage(f"无法打开指标导出: {exc}")
            return
        if dialog.exec() and dialog.result:
            if hasattr(dialog.result, "results"):
                success = len(dialog.result.results)
                failed = len(dialog.result.failures)
                self._status_bar.showMessage(
                    f"批量导出完成: 成功 {success}，失败 {failed}"
                )
                if failed:
                    details = "\n".join(
                        f"{item.get('path', '未知 Trial')}: "
                        f"{item.get('error', '未知错误')}"
                        for item in dialog.result.failures
                    )
                    QMessageBox.warning(
                        self,
                        "批量导出部分失败" if success else "批量导出失败",
                        f"成功 {success}，失败 {failed}\n\n{details}",
                    )
            else:
                self._status_bar.showMessage(
                    f"指标结果已导出: {dialog.result.output_dir}"
                )

    def _on_compare_history(self):
        """Open the read-only browser for segments recorded by this version."""
        if not self.recorder:
            self._status_bar.showMessage("录制器未配置，无法浏览历史数据")
            return
        if not self._history_comparison_available():
            self._status_bar.showMessage("请先停止采集并等待当前操作完成")
            return
        try:
            dialog = HistoryComparisonDialog(
                self.recorder.data_dir,
                viewer_config=self.config,
                parent=self,
            )
            dialog.exec()
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._status_bar.showMessage(f"无法打开历史段对比: {exc}")

    def _on_delete_selected_trial(self):
        if not self.recorder:
            return
        paths = self._selected_trial_paths()
        if not paths:
            self._status_bar.showMessage("请先选择至少一个 Trial")
            return
        if self._recording:
            self._status_bar.showMessage("请先停止当前 Trial 再删除数据")
            return
        answer = QMessageBox.question(
            self,
            "永久删除 Trial",
            f"确定永久删除选中的 {len(paths)} 个 Trial 及其中全部数据吗？\n\n"
            + "\n".join(str(path) for path in paths),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self.recorder.delete_trials(paths)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            self._status_bar.showMessage(str(e))
            return
        self._refresh_session_trials()
        self._status_bar.showMessage(
            f"已永久删除 {len(deleted)} 个 Trial" if deleted else "Trial 删除失败"
        )

    # ====== 定时器回调 ======

    def _run_timer_callback(self, name, callback, timer, stop_stream=False):
        """Keep exceptions in Qt timer slots from terminating the GUI process."""
        try:
            callback()
        except Exception as exc:
            traceback.print_exc()
            timer.stop()
            if stop_stream:
                try:
                    self._stop_all()
                except Exception:
                    traceback.print_exc()
            self._status_label.setText(f"{name}异常")
            self._status_label.setStyleSheet("font: 13px; color: #F44336;")
            self._status_bar.showMessage(f"{name}异常: {exc}；详细堆栈见终端")

    def _on_poll_timer(self):
        self._run_timer_callback(
            "数据轮询", self._on_poll, self._poll_timer, stop_stream=True
        )

    def _on_display_timer(self):
        self._run_timer_callback(
            "波形显示", self._on_update_display, self._display_timer
        )

    def _on_metrics_timer(self):
        self._run_timer_callback(
            "指标刷新", self._on_update_metrics, self._metrics_timer
        )

    def _on_verify_timer(self):
        self._run_timer_callback(
            "数据流验证", self._verify_data_stream, self._data_verify_timer
        )

    def _on_poll(self):
        """主数据轮询 (每 40ms)。"""
        if not self.board or not self.board.is_streaming or not self.chain:
            return

        data = self.board.get_new_data(n_samples=None)
        if data is None or data.size == 0:
            return

        source_timestamps = getattr(self.board, '_last_timestamps', None)
        sample_mask = self._sync_sample_mask(data, source_timestamps)
        record_data, record_timestamps = self._post_sync_samples(
            data, source_timestamps, sample_mask=sample_mask
        )

        if self._delsys.active:
            self._delsys.note_first_data(record_data, record_timestamps)
            if self._delsys.warning:
                self._sync_label.setText(self._delsys.warning)

        if (
            not self._stream_data_confirmed
            and getattr(self.board, 'data_stream_verified', False)
        ):
            self._stream_data_confirmed = True
            self._status_label.setText("采集中 (数据已验证)")
            self._status_label.setStyleSheet("font: 13px; color: #4CAF50;")
            self._status_bar.showMessage(
                f"数据流验证通过 @ {self.board.sample_rate}Hz"
            )

        # 计数器与 data 来自同一次 BrainFlow 读取，不会错位
        counter = getattr(self.board, '_last_counter', None)
        record_counter = self._filtered_counter(
            counter, data.shape[1], sample_mask, record_data.shape[1]
        )
        self.chain.push_samples(data)

        if self._recording and self.recorder:
            n_ch = min(record_data.shape[0], self.n_channels)
            ts = record_timestamps
            if ts is None or len(ts) != record_data.shape[1]:
                ts = (
                    time.time()
                    - (record_data.shape[1] - 1) / self.board.sample_rate
                    + np.arange(record_data.shape[1]) / self.board.sample_rate
                )
            self.recorder.append(record_data[:n_ch], ts, record_counter)

        if record_counter is not None and len(record_counter) > 0 and self.metrics:
            self.metrics.update(record_counter, record_timestamps)

    def _sync_sample_mask(self, data, timestamps, stop_boundary_unix_ns=None):
        boundary_ns = self._delsys.sync_start_unix_ns if self._delsys.active else None
        if boundary_ns is None:
            return None
        if timestamps is None or len(timestamps) != data.shape[1]:
            if boundary_ns is not None:
                self._delsys.note_warning(
                    "OpenBCI 时间戳缺失或数量不匹配，无法可靠执行 START/STOP 边界裁剪",
                    "openbci_boundary_filter_skipped",
                )
            return None
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        keep = timestamp_array >= boundary_ns / 1_000_000_000
        if stop_boundary_unix_ns is not None:
            keep &= timestamp_array <= stop_boundary_unix_ns / 1_000_000_000
        return keep

    def _post_sync_samples(self, data, timestamps, stop_boundary_unix_ns=None,
                           sample_mask=None):
        if sample_mask is None:
            sample_mask = self._sync_sample_mask(
                data, timestamps, stop_boundary_unix_ns
            )
        if sample_mask is None:
            return data, timestamps
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        keep = sample_mask
        return data[:, keep], timestamp_array[keep]

    def _filtered_counter(self, counter, source_count, sample_mask, result_count):
        if counter is None or len(counter) != source_count:
            if self._delsys.active:
                self._packet_counter_missing_batches += 1
                self._packet_counter_missing_samples += int(source_count)
                self._delsys.note_warning(
                    "OpenBCI 包计数器缺失或数量不匹配，本次包完整性诊断不可用",
                    "openbci_counter_alignment_failed",
                )
            return None
        values = np.asarray(counter, dtype=np.int32)
        if sample_mask is not None:
            values = values[sample_mask]
        if len(values) != result_count:
            return None
        return values

    def _on_update_display(self):
        """以 10Hz 左右刷新时域波形，避免被较重指标计算拖慢。"""
        if not self.chain or not self.board:
            return

        fs = self.board.sample_rate
        display_sec = self._sidebar.get_display_sec()
        n_display = int(display_sec * fs)
        self._time_view.fs = fs
        self._time_view.set_display_sec(display_sec)

        y_range = self._sidebar.get_y_range()
        if y_range is not None:
            self._time_view.set_y_range(-y_range, y_range)
        else:
            self._time_view.auto_range()

        active = self._sidebar.get_active_channels()
        self._time_view.set_active_channels(active)
        raw_list = [
            self.chain.get_filtered(ch, n_samples=n_display)
            if active[ch] else np.array([])
            for ch in range(self.n_channels)
        ]
        env_list = [
            self.chain.get_envelope(ch, n_samples=n_display)
            if active[ch] else np.array([])
            for ch in range(self.n_channels)
        ]
        self._time_view.update_display(raw_list, env_list)
        n_activation = max(16, int(0.25 * fs))
        activation_rms = []
        for channel in range(self.n_channels):
            signal = self.chain.get_filtered(channel, n_samples=n_activation)
            activation_rms.append(
                float(np.sqrt(np.mean(np.square(signal[-n_activation:]))))
                if len(signal) else float("nan")
            )
        self._activation_map.update_rms(activation_rms)

    def _on_update_metrics(self):
        """每秒更新信号质量、RMS、PSD 和 Session 状态。"""
        if not self.chain or not self.board:
            return

        fs = self.board.sample_rate
        n_quality = int(fs * 0.5)
        quality_statuses: list[str] = []
        for ch in range(self.n_channels):
            raw = self.chain.get_raw(ch, n_samples=n_quality)
            quality_statuses.append(_quick_quality(raw))
        self._sidebar.update_quality(quality_statuses)

        rms_values = [
            self.chain.get_rms(ch, window_sec=1.0)
            for ch in range(self.n_channels)
        ]
        baseline_rms = min(rms_values) if rms_values else 1.0
        pli_values: list[float] = []
        snr_values: list[float] = []
        for ch in range(self.n_channels):
            sq = self.chain.get_signal_quality(
                ch, baseline_rms=baseline_rms, n_samples=n_quality
            )
            if sq:
                pli_values.append(sq.get("PLI", 0.0))
                snr_values.append(sq.get("SNR_dB", float("nan")))
            else:
                pli_values.append(0.0)
                snr_values.append(float("nan"))
        self._sidebar.update_pli(pli_values)
        self._sidebar.update_quality(quality_statuses, snr_values)
        self._sidebar.update_rms(rms_values)

        if self.metrics:
            rates = self.metrics.get_rates()
            self._sidebar.update_loss(rates)
            self._status_bar.showMessage(f"丢包: {self.metrics.get_summary()}")
        if self._start_time is not None:
            self._sidebar.update_elapsed(time.time() - self._start_time)

        active = self._sidebar.get_active_channels()
        self._psd_view.set_active_channels(active)
        psd_data = [
            self.chain.get_psd(ch)
            if active[ch] else (np.array([]), np.array([]))
            for ch in range(self.n_channels)
        ]
        self._psd_view.update_display(psd_data)
        self._refresh_session_trials()

    # ====== 内部 ======

    def _verify_data_stream(self):
        """开始采集两秒后确认收到样本且 Cyton 包计数器正在推进。"""
        if not self._streaming or self._stream_data_confirmed:
            return
        if getattr(self.board, 'data_stream_verified', False):
            self._stream_data_confirmed = True
            self._status_label.setText("采集中 (数据已验证)")
            self._status_label.setStyleSheet("font: 13px; color: #4CAF50;")
            return
        self._status_label.setText("采集中 (未收到有效数据)")
        self._status_label.setStyleSheet("font: 13px; color: #F44336;")
        self._status_bar.showMessage(
            "未检测到推进的 Cyton 包计数器；请检查 Shield、Cyton、UDP 入站防火墙和网络"
        )

    def _stop_all(self, reason="aborted", stop_boundary_unix_ns=None) -> dict | None:
        self._poll_timer.stop()
        self._display_timer.stop()
        self._metrics_timer.stop()
        self._data_verify_timer.stop()

        trial = None
        synced = self._delsys.active
        pending_error = None
        if synced and self._delsys.camera_active:
            self._delsys.stop_camera()
        try:
            if (
                self._recording and self.recorder and self.board
                and self.board.is_streaming
            ):
                if synced:
                    self._delsys.note_event("openbci_stop_begin")
                    self.board.stop_stream()
                    self._delsys.note_event("openbci_stop_returned")
                    final_data = self.board.get_new_data(n_samples=None, allow_stopped=True)
                else:
                    final_data = self.board.get_new_data(n_samples=None)
                if final_data is not None and final_data.size > 0:
                    n_ch = min(final_data.shape[0], self.n_channels)
                    timestamps = getattr(self.board, '_last_timestamps', None)
                    counter = getattr(self.board, '_last_counter', None)
                    sample_mask = self._sync_sample_mask(
                        final_data, timestamps, stop_boundary_unix_ns
                    )
                    source_count = final_data.shape[1]
                    final_data, timestamps = self._post_sync_samples(
                        final_data, timestamps, stop_boundary_unix_ns,
                        sample_mask=sample_mask,
                    )
                    record_counter = self._filtered_counter(
                        counter, source_count, sample_mask, final_data.shape[1]
                    )
                    if timestamps is None or len(timestamps) != final_data.shape[1]:
                        timestamps = (
                            time.time()
                            - (final_data.shape[1] - 1) / self.board.sample_rate
                            + np.arange(final_data.shape[1]) / self.board.sample_rate
                        )
                    if synced:
                        self._delsys.note_first_data(final_data[:n_ch], timestamps)
                    if (
                        record_counter is not None and len(record_counter) > 0
                        and self.metrics
                    ):
                        self.metrics.update(record_counter, timestamps)
                    self.recorder.append(
                        final_data[:n_ch], timestamps, record_counter
                    )
        except Exception as exc:
            pending_error = exc
        finally:
            try:
                if self.board and self.board.is_streaming:
                    self.board.stop_stream()
            except Exception as exc:
                pending_error = pending_error or exc
            if synced:
                self._delsys.stop_camera()
            try:
                if self._recording and self.recorder:
                    trial = self.recorder.stop_recording()
                    if trial is not None and self.metrics:
                        diagnostics = self.metrics.get_diagnostics()
                        diagnostics["complete"] = (
                            self._packet_counter_missing_batches == 0
                        )
                        diagnostics["missing_counter_batch_count"] = (
                            self._packet_counter_missing_batches
                        )
                        diagnostics["missing_counter_sample_count"] = (
                            self._packet_counter_missing_samples
                        )
                        trial["packet_diagnostics"] = diagnostics
            except Exception as exc:
                pending_error = pending_error or exc
            finally:
                if synced:
                    self._delsys.note_event("recording_saved", trial=trial)
                    warning = self._delsys.finish(reason)
                    if trial is not None and self._delsys.last_report_path is not None:
                        trial["sync_report_path"] = str(self._delsys.last_report_path)
                    if warning:
                        self._sync_label.setText(warning)
                self._recording = False
                self._streaming = False
                self._start_time = None
                self._activation_map.clear_display()
                self._btn_start.setEnabled(True)
                self._btn_stop.setEnabled(False)
                self._sidebar.set_apply_hw_enabled(bool(self.board and self.board.is_connected))
                self._sidebar.set_hw_controls_enabled(True)
                self._refresh_session_trials()
                self._set_sync_busy(False)
        if pending_error is not None:
            raise pending_error
        return trial

    def closeEvent(self, event):
        if self._connect_thread is not None or self._hw_thread is not None or self._sync_thread is not None:
            self._status_bar.showMessage("硬件通信正在进行，请等待完成后再退出")
            event.ignore()
            return
        if self._delsys.active:
            self._sync_label.setText("请先点击停止并检查 Delsys 状态，再退出同步版")
            event.ignore()
            return
        self._stop_all()
        self._camera_preview_timer.stop()
        self._camera.disconnect()
        if self.board and self.board.is_connected:
            self.board.disconnect()
        if self.recorder:
            self.recorder.end_session()
        self._delsys.disconnect()
        event.accept()


# ====== 测试 ======
if __name__ == "__main__":
    app = QApplication(sys.argv)

    N_CH = 4  # 测试用通道数

    class FakeBoard:
        def __init__(self, n_ch=N_CH):
            self._t = 0.0
            self.is_connected = True
            self.is_streaming = True
            self.sample_rate = 250.0
            self.n_channels = n_ch
        def connect(self): return True
        def disconnect(self): pass
        def start_stream(self): return True
        def stop_stream(self): pass
        def get_new_data(self, n_samples=None):
            n = n_samples or 10
            t = np.arange(self._t, self._t + n / 250.0, 1 / 250.0)
            self._t += n / 250.0
            data = np.zeros((self.n_channels, len(t)))
            for ch in range(self.n_channels):
                freq = 30 + ch * 15
                data[ch] = 100 * np.sin(2 * np.pi * freq * t)
                data[ch] += 15 * np.random.randn(len(t))
            # 模拟 Ch3 有 50Hz 干扰
            if self.n_channels > 3:
                data[2] += 80 * np.sin(2 * np.pi * 50 * t)
            # 模拟最后一个通道信号很弱
            data[-1] *= 0.1
            return data.astype(np.float32)
        def get_counter_data(self, n_samples=250):
            return np.arange(n_samples, dtype=np.int32)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from signal_chain import SignalChain
    from metrics import PacketLossTracker

    board = FakeBoard()
    chain = SignalChain(fs=250.0, n_channels=N_CH)
    tracker = PacketLossTracker(sample_rate=250.0)

    window = MainWindow(
        board_manager=board,
        signal_chain=chain,
        metrics_engine=tracker,
        recorder=None,
    )
    window._btn_connect.click()
    window._btn_start.click()
    window.show()
    sys.exit(app.exec())
