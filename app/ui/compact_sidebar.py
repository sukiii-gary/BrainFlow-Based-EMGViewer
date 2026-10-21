"""精简侧边栏 —— 滤波参数 + 通道选择 + 录制 + 显示 + 质量 + RMS + 丢包 + 时间。"""
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox,
                              QCheckBox, QGroupBox, QGridLayout, QHBoxLayout,
                              QFormLayout, QPushButton)
from PyQt6.QtCore import Qt


class CompactSidebar(QWidget):
    """右侧窄面板。"""

    GREEN = "#4CAF50"
    YELLOW = "#FF9800"
    RED = "#F44336"
    GRAY = "#9E9E9E"

    QUALITY_COLORS = {
        "OK": "#4CAF50", "LOW": "#FF9800", "HIGH": "#FF9800",
        "NOISE": "#F44336", "SAT": "#F44336", "OFF": "#9E9E9E",
    }

    def __init__(self, parent=None, n_channels: int = 8):
        super().__init__(parent=parent)
        self.n_channels = n_channels

        # 通道选择
        self._ch_checkboxes: list[QCheckBox] = []

        # 滤波参数标签
        self._label_bandpass: QLabel | None = None
        self._label_notch: QLabel | None = None
        self._label_envelope: QLabel | None = None

        # 信号质量
        self._quality_labels: list[QLabel] = []
        self._rms_labels: list[QLabel] = []
        self._loss_label: QLabel | None = None
        self._pli_label: QLabel | None = None
        self._elapsed_label: QLabel | None = None

        # 录制
        self._input_group: QLineEdit | None = None
        self._input_subject: QLineEdit | None = None
        self._input_task: QLineEdit | None = None

        # 显示
        self._combo_time_window: QComboBox | None = None
        self._combo_y_range: QComboBox | None = None

        # 硬件设置 (Gain / BIAS / SRB2，逐通道)
        self._gain_combos: list[QComboBox] = []
        self._bias_checkboxes: list[QCheckBox] = []
        self._srb2_checkboxes: list[QCheckBox] = []
        self._master_bias: QCheckBox | None = None
        self._master_srb2: QCheckBox | None = None
        self._syncing_hw_master = False
        self._btn_apply_hw: QPushButton | None = None
        self._combo_gain: QComboBox | None = None
        self._cb_bias: QCheckBox | None = None
        self._cb_srb2: QCheckBox | None = None
        self._label_hw_status: QLabel | None = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ---- 滤波参数 ----
        filt_group = QGroupBox("滤波参数")
        filt_form = QFormLayout(filt_group)
        filt_form.setSpacing(1)
        filt_form.setContentsMargins(4, 6, 4, 4)

        self._label_bandpass = QLabel("--")
        self._label_bandpass.setStyleSheet("font: 10px 'Menlo';")
        filt_form.addRow(QLabel("带通:"), self._label_bandpass)

        self._label_notch = QLabel("--")
        self._label_notch.setStyleSheet("font: 10px 'Menlo';")
        filt_form.addRow(QLabel("陷波:"), self._label_notch)

        self._label_envelope = QLabel("--")
        self._label_envelope.setStyleSheet("font: 10px 'Menlo';")
        filt_form.addRow(QLabel("包络:"), self._label_envelope)

        layout.addWidget(filt_group)

        # ---- 硬件设置 ----
        hw_group = QGroupBox("硬件设置")
        hw_layout = QVBoxLayout(hw_group)
        hw_layout.setSpacing(3)
        hw_layout.setContentsMargins(4, 6, 4, 4)
        hw_grid = QGridLayout()
        hw_grid.setHorizontalSpacing(6)
        hw_grid.setVerticalSpacing(2)
        for col, text in enumerate(("通道", "增益", "BIAS", "SRB2")):
            header = QLabel(text)
            header.setStyleSheet("font: 9px; color: #616161;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hw_grid.addWidget(header, 0, col)

        all_label = QLabel("全部")
        all_label.setStyleSheet("font: 10px 'Menlo'; font-weight: bold;")
        self._master_bias = QCheckBox()
        self._master_bias.setTristate(True)
        self._master_bias.setCheckState(Qt.CheckState.Checked)
        self._master_bias.setToolTip("一键设置全部通道是否参与 BIAS 生成")
        self._master_srb2 = QCheckBox()
        self._master_srb2.setTristate(True)
        self._master_srb2.setCheckState(Qt.CheckState.Checked)
        self._master_srb2.setToolTip("一键设置全部通道是否连接 SRB2 参考")
        hw_grid.addWidget(all_label, 1, 0)
        hw_grid.addWidget(self._master_bias, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        hw_grid.addWidget(self._master_srb2, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)

        for ch in range(self.n_channels):
            channel_label = QLabel(f"Ch{ch + 1}")
            channel_label.setStyleSheet("font: 10px 'Menlo';")
            combo = QComboBox()
            combo.addItems(["1x", "2x", "4x", "6x", "8x", "12x", "24x"])
            combo.setCurrentIndex(6)
            combo.setMinimumWidth(55)
            combo.setStyleSheet("font: 10px; padding: 1px;")
            bias = QCheckBox()
            bias.setChecked(True)
            bias.setToolTip(f"Ch{ch + 1} 是否参与 BIAS 生成")
            srb2 = QCheckBox()
            srb2.setChecked(True)
            srb2.setToolTip(f"Ch{ch + 1} 是否连接 SRB2 参考")
            hw_grid.addWidget(channel_label, ch + 2, 0)
            hw_grid.addWidget(combo, ch + 2, 1)
            hw_grid.addWidget(bias, ch + 2, 2, alignment=Qt.AlignmentFlag.AlignCenter)
            hw_grid.addWidget(srb2, ch + 2, 3, alignment=Qt.AlignmentFlag.AlignCenter)
            self._gain_combos.append(combo)
            self._bias_checkboxes.append(bias)
            self._srb2_checkboxes.append(srb2)

        self._master_bias.clicked.connect(
            lambda checked: self._set_all_hw_checkboxes(self._bias_checkboxes, checked)
        )
        self._master_srb2.clicked.connect(
            lambda checked: self._set_all_hw_checkboxes(self._srb2_checkboxes, checked)
        )
        for checkbox in self._bias_checkboxes:
            checkbox.toggled.connect(
                lambda _checked: self._sync_hw_master_checkbox(
                    self._master_bias, self._bias_checkboxes
                )
            )
        for checkbox in self._srb2_checkboxes:
            checkbox.toggled.connect(
                lambda _checked: self._sync_hw_master_checkbox(
                    self._master_srb2, self._srb2_checkboxes
                )
            )

        # 保留旧字段别名，避免已有扩展代码立即失效。
        self._combo_gain = self._gain_combos[0] if self._gain_combos else None
        self._cb_bias = self._bias_checkboxes[0] if self._bias_checkboxes else None
        self._cb_srb2 = self._srb2_checkboxes[0] if self._srb2_checkboxes else None

        hw_layout.addLayout(hw_grid)
        self._btn_apply_hw = QPushButton("应用并验证")
        self._btn_apply_hw.setEnabled(False)
        self._btn_apply_hw.setToolTip("停止采集时向 Cyton 下发设置并读取 ADS1299 寄存器")
        hw_layout.addWidget(self._btn_apply_hw)

        self._label_hw_status = QLabel("待应用")
        self._label_hw_status.setWordWrap(True)
        self._label_hw_status.setStyleSheet(
            "font: 10px 'Menlo'; color: #9E9E9E; padding: 1px 2px;"
        )
        hw_layout.addWidget(self._label_hw_status)

        layout.addWidget(hw_group)

        # ---- 通道选择 ----
        ch_group = QGroupBox("波形/频谱 通道")
        ch_layout = QGridLayout(ch_group)
        ch_layout.setSpacing(4)
        ch_layout.setContentsMargins(4, 6, 4, 4)

        for ch in range(self.n_channels):
            cb = QCheckBox(f"Ch{ch+1}")
            cb.setChecked(True)
            cb.setStyleSheet("font: 10px; spacing: 2px;")
            ch_layout.addWidget(cb, ch // 2, ch % 2)
            self._ch_checkboxes.append(cb)

        layout.addWidget(ch_group)

        # ---- 显示控制 ----
        disp_group = QGroupBox("显示")
        disp_form = QFormLayout(disp_group)
        disp_form.setSpacing(2)
        disp_form.setContentsMargins(4, 6, 4, 4)

        self._combo_time_window = QComboBox()
        self._combo_time_window.addItems(["5s", "10s", "15s", "30s"])
        self._combo_time_window.setCurrentIndex(2)
        self._combo_time_window.setMinimumWidth(80)
        self._combo_time_window.setStyleSheet("font: 11px; padding: 2px;")
        disp_form.addRow(QLabel("时间窗:"), self._combo_time_window)

        self._combo_y_range = QComboBox()
        self._combo_y_range.addItems(["100 μV", "200 μV", "400 μV", "1000 μV", "Auto"])
        self._combo_y_range.setCurrentIndex(1)
        self._combo_y_range.setMinimumWidth(80)
        self._combo_y_range.setStyleSheet("font: 11px; padding: 2px;")
        disp_form.addRow(QLabel("Y轴:"), self._combo_y_range)

        layout.addWidget(disp_group)

        # ---- 录制设置 ----
        rec_group = QGroupBox("录制设置")
        rec_form = QFormLayout(rec_group)
        rec_form.setSpacing(2)
        rec_form.setContentsMargins(4, 6, 4, 4)

        self._input_group = QLineEdit()
        self._input_group.setStyleSheet("font: 11px; padding: 2px;")
        rec_form.addRow(QLabel("组别:"), self._input_group)

        self._input_subject = QLineEdit()
        self._input_subject.setStyleSheet("font: 11px; padding: 2px;")
        rec_form.addRow(QLabel("被试:"), self._input_subject)

        self._input_task = QLineEdit()
        self._input_task.setStyleSheet("font: 11px; padding: 2px;")
        rec_form.addRow(QLabel("任务:"), self._input_task)

        layout.addWidget(rec_group)

        # ---- 信号质量灯 ----
        qual_group = QGroupBox("信号质量")
        qual_layout = QGridLayout(qual_group)
        qual_layout.setSpacing(2)

        for ch in range(self.n_channels):
            row, col = ch // 2, ch % 2  # Ch1+Ch2同行, Ch3+Ch4同行
            label = QLabel(f"● Ch{ch+1}")
            label.setStyleSheet(
                f"font: 11px 'Menlo'; padding: 1px 3px; color: {self.GRAY};"
            )
            qual_layout.addWidget(label, row, col)
            self._quality_labels.append(label)

        layout.addWidget(qual_group)

        # ---- 工频干扰 ----
        pli_group = QGroupBox("50Hz 干扰")
        pli_inner = QHBoxLayout(pli_group)
        pli_inner.setContentsMargins(4, 4, 4, 4)
        self._pli_label = QLabel("--")
        self._pli_label.setStyleSheet("font: 12px 'Menlo'; padding: 2px 4px;")
        pli_inner.addWidget(self._pli_label)
        pli_inner.addStretch()
        layout.addWidget(pli_group)

        # ---- RMS 表 ----
        rms_group = QGroupBox("RMS (μV)")
        rms_layout = QGridLayout(rms_group)
        rms_layout.setSpacing(2)

        for ch in range(self.n_channels):
            row, col = ch // 2, ch % 2  # Ch1+Ch2同行, Ch3+Ch4同行
            label = QLabel(f"Ch{ch+1}:  ---")
            label.setStyleSheet("font: 11px 'Menlo'; padding: 1px 3px;")
            rms_layout.addWidget(label, row, col)
            self._rms_labels.append(label)

        layout.addWidget(rms_group)

        # ---- 丢包率 ----
        loss_group = QGroupBox("丢包率")
        loss_inner = QVBoxLayout(loss_group)
        self._loss_label = QLabel("--")
        self._loss_label.setStyleSheet("font: 12px 'Menlo'; padding: 2px 4px;")
        loss_inner.addWidget(self._loss_label)
        layout.addWidget(loss_group)

        # ---- 采集时间 ----
        time_group = QGroupBox("采集时间")
        time_inner = QVBoxLayout(time_group)
        self._elapsed_label = QLabel("00:00:00")
        self._elapsed_label.setStyleSheet("font: 16px 'Menlo'; padding: 4px;")
        self._elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_inner.addWidget(self._elapsed_label)
        layout.addWidget(time_group)

        layout.addStretch()

    # ====== 滤波参数 ======

    def set_filter_info(self, bandpass: tuple, notch_hz: float,
                        notch_bw: float, harmonics: int, envelope_ms: float):
        """设置滤波参数只读显示。"""
        if self._label_bandpass:
            self._label_bandpass.setText(f"{bandpass[0]:.0f} – {bandpass[1]:.0f} Hz")
        if self._label_notch:
            self._label_notch.setText(f"{notch_hz:.0f} Hz ±{notch_bw:.0f} ×{harmonics}")
        if self._label_envelope:
            self._label_envelope.setText(f"{envelope_ms:.0f} ms RMS")

    # ====== 硬件设置 ======

    def _set_all_hw_checkboxes(self, checkboxes: list[QCheckBox], checked: bool):
        self._syncing_hw_master = True
        try:
            for checkbox in checkboxes:
                checkbox.setChecked(checked)
        finally:
            self._syncing_hw_master = False
        master = self._master_bias if checkboxes is self._bias_checkboxes else self._master_srb2
        self._sync_hw_master_checkbox(master, checkboxes)

    def _sync_hw_master_checkbox(self, master: QCheckBox | None,
                                 checkboxes: list[QCheckBox]):
        if self._syncing_hw_master or not master or not checkboxes:
            return
        checked_count = sum(checkbox.isChecked() for checkbox in checkboxes)
        if checked_count == 0:
            state = Qt.CheckState.Unchecked
        elif checked_count == len(checkboxes):
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        master.blockSignals(True)
        master.setCheckState(state)
        master.blockSignals(False)

    def get_hw_settings(self) -> dict:
        """返回逐通道硬件设置，并保留首通道兼容字段。"""
        gain_map = {"1x": 0, "2x": 1, "4x": 2, "6x": 3,
                    "8x": 4, "12x": 5, "24x": 6}
        channels = []
        for index, (combo, bias, srb2) in enumerate(zip(
                self._gain_combos, self._bias_checkboxes, self._srb2_checkboxes)):
            channels.append({
                "channel": index + 1,
                "gain": gain_map.get(combo.currentText(), 6),
                "bias_on": bias.isChecked(),
                "srb2_on": srb2.isChecked(),
            })
        first = channels[0] if channels else {"gain": 6, "bias_on": True, "srb2_on": True}
        return {
            "channels": channels,
            "bias_on": first["bias_on"],
            "srb2_on": first["srb2_on"],
            "gain": first["gain"],
        }

    def update_hw_status(self, readback: dict, target: dict):
        """按待发送/已发送未验证/已验证/失败四态更新显示。"""
        if not self._label_hw_status:
            return

        state = readback.get("state", "pending")
        state_view = {
            "pending": (self.GRAY, "待应用"),
            "sent_unverified": (self.YELLOW, "已发送，未验证"),
            "verified": (self.GREEN, "寄存器已验证"),
            "failed": (self.RED, "硬件设置验证失败"),
            "not_applicable": (self.GRAY, "模拟板不适用"),
        }
        color, text = state_view.get(state, (self.GRAY, "状态未知"))
        channels = readback.get("channels", {})
        if state in ("verified", "failed") and channels:
            marks = []
            for target_item in target.get("channels", []):
                channel = target_item["channel"]
                actual = channels.get(channel) or channels.get(str(channel), {})
                mark = "OK" if actual.get("matches") else "FAIL"
                marks.append(f"Ch{channel}:{mark}")
            text += "  " + " ".join(marks)

        self._label_hw_status.setText(text)
        tooltip_parts = []
        source = readback.get("source")
        if source and source != "none":
            tooltip_parts.append(f"回读来源: {source}")
        diagnostic = readback.get("diagnostic", "")
        if diagnostic:
            tooltip_parts.append(diagnostic)
        raw = readback.get("raw", "")
        if raw and raw != diagnostic:
            tooltip_parts.append(raw)
        self._label_hw_status.setToolTip("\n\n".join(tooltip_parts)[:2000])
        self._label_hw_status.setStyleSheet(
            f"font: 10px 'Menlo'; color: {color}; padding: 1px 2px; font-weight: bold;"
        )

    def mark_hw_pending(self):
        self.update_hw_status({"state": "pending"}, self.get_hw_settings())

    def set_apply_hw_enabled(self, enabled: bool):
        if self._btn_apply_hw:
            self._btn_apply_hw.setEnabled(enabled)

    def set_hw_controls_enabled(self, enabled: bool):
        master_controls = [
            control for control in (self._master_bias, self._master_srb2)
            if control is not None
        ]
        for control in (
                self._gain_combos + self._bias_checkboxes
                + self._srb2_checkboxes + master_controls):
            control.setEnabled(enabled)

    # ====== 通道选择 ======

    def get_active_channels(self) -> list[bool]:
        """返回每通道是否选中。"""
        return [cb.isChecked() for cb in self._ch_checkboxes]

    # ====== 录制元数据 ======

    def get_recording_metadata(self) -> dict:
        return {
            "group": self._input_group.text().strip() if self._input_group else "",
            "subject_id": self._input_subject.text().strip() if self._input_subject else "",
            "task": self._input_task.text().strip() if self._input_task else "",
        }

    def set_recording_metadata(self, group: str = "", subject: str = "", task: str = ""):
        if self._input_group:
            self._input_group.setText(group)
        if self._input_subject:
            self._input_subject.setText(subject)
        if self._input_task:
            self._input_task.setText(task)

    def focus_subject_input(self):
        if self._input_subject:
            self._input_subject.setFocus()
            self._input_subject.selectAll()

    # ====== 显示设置 ======

    def get_display_sec(self) -> float:
        if self._combo_time_window:
            return float(self._combo_time_window.currentText().rstrip("s"))
        return 15.0

    def get_y_range(self) -> float | None:
        if self._combo_y_range:
            text = self._combo_y_range.currentText()
            if text == "Auto":
                return None
            return float(text.split()[0])
        return 200.0

    # ====== 更新方法 ======

    def update_quality(self, statuses: list[str], snr_values: list[float] | None = None):
        for ch, status in enumerate(statuses):
            if ch >= len(self._quality_labels):
                break
            color = self.QUALITY_COLORS.get(status, self.GRAY)
            if snr_values and ch < len(snr_values) and not np.isnan(snr_values[ch]):
                text = f"● Ch{ch+1} {snr_values[ch]:.0f}dB"
            else:
                text = f"● Ch{ch+1}"
            self._quality_labels[ch].setText(text)
            self._quality_labels[ch].setStyleSheet(
                f"font: 11px 'Menlo'; padding: 1px 3px; color: {color};"
            )

    def update_pli(self, pli_values: list[float]):
        max_pli = max(pli_values) * 100 if pli_values else 0.0
        if max_pli < 5:
            color = self.GREEN
        elif max_pli < 20:
            color = self.YELLOW
        else:
            color = self.RED
        text = f"{max_pli:.1f}%" if pli_values else "--"
        self._pli_label.setText(text)
        self._pli_label.setStyleSheet(
            f"font: 12px 'Menlo'; padding: 1px 4px; color: {color}; font-weight: bold;"
        )

    def update_rms(self, values: list[float]):
        for ch, val in enumerate(values):
            if ch >= len(self._rms_labels):
                break
            if val < 1.0:
                color = self.GRAY
            elif val < 50:
                color = self.GREEN
            elif val < 200:
                color = self.YELLOW
            else:
                color = self.RED
            self._rms_labels[ch].setText(f"Ch{ch+1}: {val:5.1f}")
            self._rms_labels[ch].setStyleSheet(
                f"font: 11px 'Menlo'; padding: 1px 3px; color: {color};"
            )

    def update_loss(self, rates: dict):
        parts = [f"{k}:{v:.1f}%" for k, v in rates.items()]
        text = "  ".join(parts)
        max_rate = max(rates.values()) if rates else 0.0
        color = self.GREEN if max_rate < 1 else self.YELLOW if max_rate < 5 else self.RED
        self._loss_label.setText(text)
        self._loss_label.setStyleSheet(
            f"font: 12px 'Menlo'; padding: 2px 4px; color: {color};"
        )

    def update_elapsed(self, seconds: float):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        self._elapsed_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
