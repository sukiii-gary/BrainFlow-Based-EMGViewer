"""实时信号处理管道 —— 复用 emg_python 的滤波/包络/PSD 代码。

设计原则:
  - 不修改 emg_python，只 import 使用
  - 维持环形缓冲，定期批量滤波（filtfilt 需要整段信号）
  - filtfilt 引入 ~1.5 秒群延迟，但信号质量监测场景完全可接受
  - 如果将来需要零延迟实时控制，可替换为 lfilter+zi 方案
"""
from __future__ import annotations
import sys
from pathlib import Path

# 添加 emg_python 到路径
_EMG_PYTHON = Path(__file__).resolve().parents[1] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

import numpy as np
from collections import deque

# 从现有 emg_python 直接导入（离线分析已验证通过）
from preprocessing.filters import apply_filters, notch_vals, notch_coeffs
from preprocessing.envelope import compute_envelope
from preprocessing.psd import compute_psd
from features.signal_quality import rms as rms_scalar


class SignalChain:
    """实时信号处理链。

    对每个通道维护独立的环形缓冲，每隔一定间隔批量滤波。
    直接复用 emg_python 的 apply_filters / compute_envelope / compute_psd。
    """

    def __init__(self, fs: float = 250.0, n_channels: int = 8,
                 buffer_sec: float = 60.0, bandpass: tuple = (20.0, 120.0),
                 notch_hz: float = 50.0, notch_bw: float = 8.0,
                 notch_harmonics: int = 3, envelope_ms: float = 100.0):
        """
        Args:
            fs: 采样率 (Hz)
            n_channels: 通道数
            buffer_sec: 环形缓冲长度 (秒)
            bandpass: (low, high) 带通截止频率
            notch_hz: 工频 (50/60 Hz)
            notch_bw: 陷波绝对带宽 (Hz)
            notch_harmonics: 谐波数
            envelope_ms: RMS 包络窗长 (ms)
        """
        self.fs = fs
        self.n_channels = n_channels
        self.buffer_len = int(fs * buffer_sec)  # 样本数
        self.display_len = int(fs * 5.0)         # 默认显示 5 秒

        # 每通道独立环形缓冲
        self._buffers = [deque(maxlen=self.buffer_len) for _ in range(n_channels)]

        # 滤波参数 (将用于构造临时 Config 对象传给 emg_python)
        # 保留原始带通，实际使用时按采样率钳制到 Nyquist 以内
        self._bandpass_raw = bandpass
        self._clamp_bandpass()
        self.notch_hz = notch_hz
        self.notch_bw = notch_bw
        self.notch_harmonics = notch_harmonics
        self.envelope_ms = envelope_ms

        # 缓存: key=(channel, version)，新数据到达后 version+1 自动失效
        self._version = 0
        self._cached_filtered: dict[tuple[int, int, int | None], np.ndarray] = {}
        self._cached_envelope: dict[tuple[int, int, int | None], np.ndarray] = {}
        self._cached_psd: dict[tuple[int, int, int | None], tuple[np.ndarray, np.ndarray]] = {}

        # 用于传给 emg_python 函数的简易配置对象
        self._make_filter_cfg()

    def _clamp_bandpass(self):
        """按当前采样率把带通上限钳制到 Nyquist 以内。"""
        nyquist = self.fs / 2.0
        lo = self._bandpass_raw[0]
        hi = min(self._bandpass_raw[1], nyquist - 1.0)
        self.bandpass = (lo, max(hi, lo + 1.0))

    def _make_filter_cfg(self):
        """构造 emg_python Config 对象 —— 与离线分析共享同一套配置。"""
        # 尝试导入真正的 emg_python Config
        # (emg_python 目录已在 sys.path 中，所以直接 import config 即可)
        try:
            from config import Config, Preprocess, Segmentation, Features, Channel
            self._cfg = Config(
                subject_id="live",
                sampling_rate=self.fs,
                preprocess=Preprocess(
                    notch_hz=self.notch_hz,
                    notch_bandwidth_hz=self.notch_bw,
                    notch_harmonics=self.notch_harmonics,
                    bandpass=self.bandpass,
                    smooth_method="rms",
                    smooth_window_ms=self.envelope_ms,
                ),
                segmentation=Segmentation(
                    use_preprocess_chain=False,
                    smooth_window_ms=self.envelope_ms,
                ),
                features=Features(),
                channel_map=[
                    Channel(i, f"Ch{i+1}", "test") for i in range(self.n_channels)
                ],
            )
        except ImportError:
            # fallback: emg_python 不可用时用简易 dataclass
            from dataclasses import dataclass

            @dataclass
            class _Preprocess:
                notch_hz: float; notch_bandwidth_hz: float
                notch_harmonics: int; bandpass: tuple

            @dataclass
            class _Segmentation:
                use_preprocess_chain: bool; smooth_window_ms: float

            @dataclass
            class _Cfg:
                sampling_rate: float; preprocess: _Preprocess
                segmentation: _Segmentation

            self._cfg = _Cfg(
                sampling_rate=self.fs,
                preprocess=_Preprocess(
                    notch_hz=self.notch_hz, notch_bandwidth_hz=self.notch_bw,
                    notch_harmonics=self.notch_harmonics, bandpass=self.bandpass,
                ),
                segmentation=_Segmentation(
                    use_preprocess_chain=False, smooth_window_ms=self.envelope_ms,
                ),
            )

    # ====== 数据输入 ======

    def push_samples(self, data: np.ndarray):
        """将新样本推入环形缓冲。过滤 NaN/Inf。data shape: (n_channels, n_new_samples)"""
        if data.size == 0:
            return

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        for ch in range(min(self.n_channels, data.shape[0])):
            self._buffers[ch].extend(data[ch].tolist())
        self._version += 1
        # 旧版本数组不再可复用；及时释放，避免实时采集时缓存无限增长。
        self._cached_filtered.clear()
        self._cached_envelope.clear()
        self._cached_psd.clear()

    # ====== 数据获取 ======

    def get_raw(self, channel: int, n_samples: int | None = None) -> np.ndarray:
        """原始信号（未滤波），最近 n_samples 个样本。"""
        buf = self._buffers[channel]
        if n_samples is None or n_samples > len(buf):
            return np.array(buf, dtype=float)
        data = list(buf)
        return np.array(data[-n_samples:], dtype=float)

    def get_filtered(self, channel: int, n_samples: int | None = None,
                     force_recompute: bool = False) -> np.ndarray:
        """滤波后信号。版本缓存，新数据到达后自动失效。"""
        key = (channel, self._version, n_samples)
        if not force_recompute:
            result = self._find_covering_window(
                self._cached_filtered, channel, n_samples
            )
            if result is not None:
                return result[-n_samples:] if n_samples is not None else result
        self._recompute_filtered(channel, n_samples=n_samples)
        result = self._cached_filtered.get(key, np.array([]))
        if n_samples is not None and len(result) > n_samples:
            return result[-n_samples:]
        return result

    def get_envelope(self, channel: int, n_samples: int | None = None,
                     force_recompute: bool = False) -> np.ndarray:
        """RMS 包络信号。版本缓存。"""
        key = (channel, self._version, n_samples)
        if not force_recompute:
            result = self._find_covering_window(
                self._cached_envelope, channel, n_samples
            )
            if result is not None:
                return result[-n_samples:] if n_samples is not None else result
        self._recompute_envelope(channel, n_samples=n_samples)
        result = self._cached_envelope.get(key, np.array([]))
        if n_samples is not None and len(result) > n_samples:
            return result[-n_samples:]
        return result

    def _find_covering_window(self, cache: dict, channel: int,
                              n_samples: int | None) -> np.ndarray | None:
        """Return a cached window that contains the requested trailing samples."""
        exact = cache.get((channel, self._version, n_samples))
        if exact is not None:
            return exact
        if n_samples is None:
            return None

        candidates = []
        for (cached_channel, version, cached_n), value in cache.items():
            if cached_channel != channel or version != self._version:
                continue
            if cached_n is None or cached_n >= n_samples:
                candidates.append((float("inf") if cached_n is None else cached_n, value))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def get_psd(self, channel: int, force_recompute: bool = False,
                n_samples: int | None = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD 估计。n_samples 限制用于 FFT 的数据量，默认 2000 (1000Hz/2s)。"""
        key = (channel, self._version, n_samples)
        if force_recompute or key not in self._cached_psd:
            filtered = self.get_filtered(channel, n_samples=n_samples)
            if len(filtered) >= 16:
                f, p = compute_psd(filtered, self.fs, normalize=False)
            else:
                f, p = np.array([]), np.array([])
            self._cached_psd[key] = (f, p)
        return self._cached_psd.get(key, (np.array([]), np.array([])))

    def get_rms(self, channel: int, window_sec: float = 1.0) -> float:
        """通道 RMS (μV)，基于最近 window_sec 秒的滤波数据。"""
        n = int(window_sec * self.fs)
        filtered = self.get_filtered(channel, n_samples=n)
        if len(filtered) == 0:
            return 0.0
        return rms_scalar(filtered[-n:])

    def get_features(self, channel: int,
                     band_hz: tuple = (20.0, 120.0)) -> dict:
        """实时特征集 —— 复用 emg_python features.calc_features。

        Returns dict with keys like RMS, MAV, IEMG, ZC, SSC, WL, MNF, MDF.
        """
        env = self.get_envelope(channel)
        filt = self.get_filtered(channel)
        if len(env) < 3 or len(filt) < 3:
            return {}
        try:
            from features.calc_features import calc_features
            return calc_features(env, filt, self.fs, band_power_hz=band_hz,
                                 zc_thr=5.0, ssc_thr=5.0)
        except ImportError:
            return {"RMS": self.get_rms(channel)}

    def get_signal_quality(self, channel: int,
                           baseline_rms: float | None = None,
                           n_samples: int | None = None) -> dict:
        """完整信号质量评估 —— 复用 emg_python features.signal_quality。

        Args:
            n_samples: 限制使用的样本数 (None=全部缓冲)。实时场景建议设 500-1000。
        """
        raw = self.get_raw(channel, n_samples=n_samples)
        filt = self.get_filtered(channel, n_samples=n_samples)
        if len(raw) < 16 or len(filt) < 16:
            return {}
        if baseline_rms is None:
            all_rms = [self.get_rms(ch, window_sec=1.0)
                       for ch in range(self.n_channels)]
            baseline_rms = min(all_rms) if all_rms else 1.0
        try:
            from features.signal_quality import signal_quality
            return signal_quality(raw, filt, baseline_rms, self.fs)
        except ImportError:
            return {}

    # ====== 内部计算 ======

    def _recompute_filtered(self, channel: int, n_samples: int | None = None):
        """对通道缓冲批量滤波。"""
        raw = self.get_raw(channel, n_samples=n_samples)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        key = (channel, self._version, n_samples)
        if len(raw) < 16:
            self._cached_filtered[key] = raw
            return
        self._cached_filtered[key] = apply_filters(raw, self._cfg, do_notch=True)

    def _recompute_envelope(self, channel: int, n_samples: int | None = None):
        """对滤波后数据计算包络。"""
        filtered = self.get_filtered(channel, n_samples=n_samples)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        key = (channel, self._version, n_samples)
        if len(filtered) < 16:
            self._cached_envelope[key] = filtered
            return
        self._cached_envelope[key] = compute_envelope(
            filtered, self._cfg, window_ms=self.envelope_ms
        )

    # ====== 维护 ======

    def clear(self):
        """清空所有缓冲和缓存。"""
        for buf in self._buffers:
            buf.clear()
        self._cached_filtered.clear()
        self._cached_envelope.clear()
        self._cached_psd.clear()
        self._version += 1

    def update_fs(self, fs: float, buffer_sec: float = 60.0):
        """连接硬件后更新实际采样率，重建缓冲。"""
        if fs == self.fs:
            return
        self.fs = fs
        self.buffer_len = int(fs * buffer_sec)
        self.display_len = int(fs * 5.0)
        from collections import deque
        self._buffers = [deque(maxlen=self.buffer_len) for _ in range(self.n_channels)]
        self._clamp_bandpass()  # 重新按新采样率钳制带通
        self.clear()
        self._make_filter_cfg()

    @property
    def is_dirty(self) -> bool:
        return self._dirty


# ====== 快速测试 ======

if __name__ == "__main__":
    print("=== SignalChain 测试 ===\n")

    chain = SignalChain(fs=250.0, n_channels=2, buffer_sec=5.0)

    # 生成测试信号: 50Hz 正弦 + 噪声
    t = np.arange(0, 3.0, 1/250)  # 3 秒
    test_signal = 100 * np.sin(2 * np.pi * 50 * t)  # 50Hz, 100 μV
    test_signal += 30 * np.sin(2 * np.pi * 80 * t)  # 80Hz, 30 μV
    test_signal += 10 * np.random.randn(len(t))      # 噪声

    data = np.stack([test_signal, test_signal * 0.5])  # 2 通道
    chain.push_samples(data)

    # 检查滤波效果
    for ch in range(2):
        raw = chain.get_raw(ch)
        filtered = chain.get_filtered(ch)
        env = chain.get_envelope(ch)
        freqs, psd = chain.get_psd(ch)
        rms_val = chain.get_rms(ch)

        print(f"Ch{ch+1}:")
        print(f"  Raw:      {len(raw)} samples, RMS={rms_scalar(raw):.1f} μV")
        print(f"  Filtered: {len(filtered)} samples, RMS={rms_scalar(filtered):.1f} μV")
        print(f"  Envelope: {len(env)} samples, mean={np.mean(env):.1f} μV")
        print(f"  PSD:      {len(freqs)} freq bins, max at {freqs[np.argmax(psd)]:.0f} Hz")
        print(f"  Live RMS: {rms_val:.1f} μV")

    # 验证 50Hz 被抑制
    if len(freqs) > 0:
        f50 = np.argmin(np.abs(freqs - 50))
        f80 = np.argmin(np.abs(freqs - 80))
        if f50 < len(psd) and f80 < len(psd):
            print(f"\n  50Hz 功率: {psd[f50]:.1f}")
            print(f"  80Hz 功率: {psd[f80]:.1f}")
            if psd[f50] < psd[f80] * 0.1:
                print("  ✅ 50Hz 工频被有效抑制")
            else:
                print("  ⚠️ 50Hz 抑制效果不佳，请检查滤波器配置")

    print("\n✅ SignalChain 测试完成")
