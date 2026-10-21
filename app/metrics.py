"""丢包率 & 信号质量监测。

丢包检测原理:
  BrainFlow 数据 row[0] 是包计数器 (0-255 循环)。
  每次取新数据时比较相邻计数器差值，差值 >1 即丢包。
  在 3 个时间窗口 (1s/60s/600s) 内累计丢包率。
"""
from __future__ import annotations
import time
import numpy as np
from collections import deque


class PacketLossTracker:
    """三窗口丢包率追踪器。

    用法:
        tracker = PacketLossTracker(sample_rate=250, windows=(1, 60, 600))
        tracker.update(counter_array)  # counter_array: 1D int array of packet numbers
        rates = tracker.get_rates()    # {"1s": 0.0, "60s": 0.5, "600s": 0.2}
    """

    def __init__(self, sample_rate: float = 250.0,
                 windows_sec: tuple = (1, 60, 600)):
        self.sample_rate = sample_rate
        self.windows_sec = windows_sec

        # 每个窗口的 deque: (timestamp, lost_count)
        self._windows: dict[int, deque] = {
            w: deque() for w in windows_sec
        }
        # 各窗口累计
        self._total_expected: dict[int, int] = {w: 0 for w in windows_sec}
        self._total_lost: dict[int, int] = {w: 0 for w in windows_sec}

        self._prev_counter: int | None = None
        self._prev_timestamp: float | None = None
        self._start_time: float | None = None
        self._received = 0
        self._expected = 0
        self._lost = 0
        self._duplicate = 0
        self._out_of_order = 0
        self._invalid_counter = 0
        self._transition_count = 0
        self._first_counter: int | None = None
        self._last_counter: int | None = None
        self._max_timestamp_stall_sec = 0.0
        self._nonmonotonic_timestamp_count = 0

    def reset(self):
        """重置所有计数器（如重新开始采集时调用）。"""
        for w in self.windows_sec:
            self._windows[w].clear()
            self._total_expected[w] = 0
            self._total_lost[w] = 0
        self._prev_counter = None
        self._prev_timestamp = None
        self._start_time = None
        self._received = 0
        self._expected = 0
        self._lost = 0
        self._duplicate = 0
        self._out_of_order = 0
        self._invalid_counter = 0
        self._transition_count = 0
        self._first_counter = None
        self._last_counter = None
        self._max_timestamp_stall_sec = 0.0
        self._nonmonotonic_timestamp_count = 0

    def update(self, counter_array: np.ndarray,
               timestamps: np.ndarray | None = None):
        """输入一次取到的包计数器数组（BrainFlow data[0, :]）。

        自动计算批内及与前一次调用之间的连续性。包号按 8-bit 模计数：
        前进 2..127 计为缺失，0 计为重复，反向 128..255 计为倒序。
        由于包号只有 8 bit，一次连续缺失 >=128 个包无法与倒序区分。
        """
        if counter_array is None or len(counter_array) == 0:
            return

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        counters = np.asarray(counter_array).ravel()
        self._received += int(counters.size)
        expected_this_call = 0
        lost_this_call = 0

        for raw_counter in counters:
            try:
                numeric = float(raw_counter)
            except (TypeError, ValueError):
                numeric = float("nan")
            if (
                not np.isfinite(numeric)
                or numeric != round(numeric)
                or numeric < 0
                or numeric > 255
            ):
                self._invalid_counter += 1
                self._prev_counter = None
                continue

            counter = int(numeric)
            if self._first_counter is None:
                self._first_counter = counter
            self._last_counter = counter

            if self._prev_counter is None:
                expected_this_call += 1
            else:
                self._transition_count += 1
                delta = (counter - self._prev_counter) % 256
                if delta == 0:
                    self._duplicate += 1
                elif delta == 1:
                    expected_this_call += 1
                elif delta <= 127:
                    missing = delta - 1
                    self._lost += missing
                    lost_this_call += missing
                    expected_this_call += delta
                else:
                    self._out_of_order += 1
            # Re-anchor after every valid observation so one reordering event does
            # not turn all later packets into false gaps.
            self._prev_counter = counter

        self._expected += expected_this_call

        if timestamps is not None:
            timestamp_values = np.asarray(timestamps).ravel()
            if timestamp_values.size == counters.size:
                for raw_timestamp in timestamp_values:
                    try:
                        timestamp = float(raw_timestamp)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(timestamp):
                        continue
                    if self._prev_timestamp is not None:
                        delta_sec = timestamp - self._prev_timestamp
                        if delta_sec > 0:
                            self._max_timestamp_stall_sec = max(
                                self._max_timestamp_stall_sec, delta_sec
                            )
                        else:
                            self._nonmonotonic_timestamp_count += 1
                    self._prev_timestamp = timestamp

        # 累积到各窗口
        for w in self.windows_sec:
            self._windows[w].append((now, expected_this_call, lost_this_call))
            self._total_expected[w] += expected_this_call
            self._total_lost[w] += lost_this_call

            # 淘汰超过窗口期的旧记录
            cutoff = now - w
            while self._windows[w] and self._windows[w][0][0] < cutoff:
                t, exp, lost = self._windows[w].popleft()
                self._total_expected[w] -= exp
                self._total_lost[w] -= lost

    def get_rates(self) -> dict:
        """返回各窗口丢包率 (百分比)。key 格式: '1s', '60s', '600s'"""
        rates = {}
        for w in self.windows_sec:
            label = f"{w}s" if w < 60 else f"{w // 60}m" if w < 3600 else f"{w // 3600}h"
            # 用 label 显示更友好，但 key 保持为秒数便于内部使用
            label = f"{w}s"
            expected = max(self._total_expected[w], 1)
            rates[label] = round(self._total_lost[w] / expected * 100, 2)
        return rates

    def get_summary(self) -> str:
        """单行摘要，例如: '1s:0.0% 60s:0.2% 600s:0.1%'"""
        rates = self.get_rates()
        parts = [f"{w}:{v}%" for w, v in rates.items()]
        return "  ".join(parts)

    def get_diagnostics(self) -> dict:
        """返回自上次 reset 起、可直接写入 JSON 的包序列诊断。"""
        loss_rate = (
            self._lost / self._expected * 100.0 if self._expected > 0 else 0.0
        )
        return {
            "received": self._received,
            "expected": self._expected,
            "lost": self._lost,
            "duplicate": self._duplicate,
            "out_of_order": self._out_of_order,
            "invalid_counter": self._invalid_counter,
            "loss_rate_percent": round(loss_rate, 6),
            "first_counter": self._first_counter,
            "last_counter": self._last_counter,
            "transition_count": self._transition_count,
            "max_timestamp_stall_ms": round(
                self._max_timestamp_stall_sec * 1000.0, 6
            ),
            "nonmonotonic_timestamp_count": self._nonmonotonic_timestamp_count,
            "counter_modulus": 256,
            "counter_semantics": (
                "BrainFlow package_num_channel; assumed to increment once per "
                "returned sample"
            ),
            "classification": (
                "mod256 delta 1=continuous, 0=duplicate, 2..127=forward gap, "
                "128..255=out_of_order"
            ),
            "limitation": (
                "A forward loss of 128 or more consecutive packets cannot be "
                "distinguished from reordering with an 8-bit counter"
            ),
        }

    @property
    def elapsed_sec(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


class SignalQualityChecker:
    """信号质量快速检查 —— 检测断连、饱和、噪声过大。

    不导入 emg_python 的 signal_quality（那是离线评估，需要静息基线），
    这里用简单启发式做实时判断。
    """

    # 阈值常量
    SATURATION_UV = 4000.0     # > 4mV = ADS1299 接近饱和
    FLATLINE_UV = 0.5          # < 0.5μV 标准差 = 可能断开
    HIGH_NOISE_UV = 200.0      # > 200μV 标准差 = 可能噪声过大
    NOISE_RATIO = 0.8          # 50Hz 带功率 / 总功率 > 80% = 严重工频干扰

    @staticmethod
    def check(raw_data: np.ndarray, fs: float = 250.0,
              notch_hz: float = 50.0) -> list[dict]:
        """对 8 通道原始数据做快速质量评估。

        Args:
            raw_data: shape (n_channels, n_samples) 的原始数据（μV）
            fs: 采样率
            notch_hz: 工频

        Returns:
            list of dicts, 每通道: {"status": "OK"|"LOW"|"NOISE"|"SAT"|"OFF",
                                    "rms_uv": float, "detail": str}
        """
        n_channels = raw_data.shape[0]
        results = []

        for ch in range(n_channels):
            ch_data = raw_data[ch].astype(float)
            rms_val = float(np.sqrt(np.mean(ch_data ** 2)))
            std_val = float(np.std(ch_data))
            p2p_val = float(np.max(ch_data) - np.min(ch_data))

            # 判断逻辑
            if std_val < SignalQualityChecker.FLATLINE_UV:
                status, detail = "OFF", "无信号/电极断开"
            elif p2p_val > SignalQualityChecker.SATURATION_UV:
                status, detail = "SAT", f"饱和 p2p={p2p_val:.0f}μV"
            elif rms_val > SignalQualityChecker.HIGH_NOISE_UV:
                # 进一步检查是否是 50Hz 工频主导
                try:
                    from preprocessing.psd import compute_psd
                    f, p = compute_psd(ch_data, fs)
                    if len(f) > 0:
                        # 50Hz 附近 ±2Hz
                        f50_mask = (f >= notch_hz - 2) & (f <= notch_hz + 2)
                        f_total_mask = (f >= 10) & (f <= fs / 2)
                        p50 = np.trapz(p[f50_mask], f[f50_mask]) if f50_mask.sum() > 1 else 0
                        p_total = np.trapz(p[f_total_mask], f[f_total_mask]) if f_total_mask.sum() > 1 else 1e-12
                        noise_ratio = p50 / p_total if p_total > 0 else 0
                        if noise_ratio > SignalQualityChecker.NOISE_RATIO:
                            status, detail = "NOISE", f"50Hz 干扰 {noise_ratio:.0%}"
                        else:
                            status, detail = "HIGH", f"高噪声 RMS={rms_val:.0f}μV"
                    else:
                        status, detail = "HIGH", f"高噪声 RMS={rms_val:.0f}μV"
                except Exception:
                    status, detail = "HIGH", f"高噪声 RMS={rms_val:.0f}μV"
            else:
                status, detail = "OK", f"正常 RMS={rms_val:.1f}μV"

            results.append({
                "channel": ch,
                "status": status,
                "rms_uv": round(rms_val, 1),
                "std_uv": round(std_val, 1),
                "p2p_uv": round(p2p_val, 1),
                "detail": detail,
            })

        return results

    @staticmethod
    def status_color(status: str) -> str:
        """状态到颜色映射。"""
        return {
            "OK": "#4CAF50",    # 绿
            "LOW": "#FF9800",   # 橙
            "HIGH": "#FF9800",  # 橙
            "NOISE": "#F44336",  # 红
            "SAT": "#F44336",   # 红
            "OFF": "#9E9E9E",   # 灰
        }.get(status, "#9E9E9E")


# ====== 快速测试 ======

if __name__ == "__main__":
    print("=== PacketLossTracker 测试 ===\n")

    tracker = PacketLossTracker(sample_rate=250.0, windows_sec=(1, 60, 600))

    # 模拟 3 秒正常采集，250Hz
    for i in range(30):  # 30 次 poll, 每次约 10 个样本
        counter = np.arange(10) + i * 10
        if i == 15:  # 模拟丢 3 个包
            counter += 3
        tracker.update(counter)
        time.sleep(0.1)

    rates = tracker.get_rates()
    print(f"丢包率: {rates}")
    print(f"摘要: {tracker.get_summary()}")

    print("\n=== SignalQualityChecker 测试 ===\n")

    # 生成测试数据
    fs = 250
    t = np.arange(0, 2.0, 1 / fs)
    good_ch = 50 * np.sin(2 * np.pi * 80 * t) + 5 * np.random.randn(len(t))
    noise_ch = 300 * np.sin(2 * np.pi * 50 * t) + 20 * np.random.randn(len(t))
    flat_ch = np.zeros(len(t)) + 0.1 * np.random.randn(len(t))
    test_data = np.stack([good_ch, noise_ch, flat_ch, good_ch * 2,
                          np.ones(len(t)) * 5000, good_ch * 0.5],
                         axis=0)

    results = SignalQualityChecker.check(test_data, fs=fs)
    for r in results:
        color = SignalQualityChecker.status_color(r["status"])
        print(f"  Ch{r['channel']+1}: {r['status']:5s}  "
              f"RMS={r['rms_uv']:6.1f} μV  {r['detail']}")

    print("\n✅ Metrics 测试完成")
