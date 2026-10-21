"""滤波链 —— 复刻 MATLAB apply_filters / notch_vals / notch_coeffs。

- 陷波:50/100/150Hz(受奈奎斯特与带通上限裁剪),每谐波保持 8Hz 绝对带宽(Q=f/bw);
  系数用 scipy.signal.iirnotch —— 与 MATLAB 自实现公式逐系数一致。
- 带通:butter(5, [20,450]) 5 阶巴特沃斯。
- 全程零相位 filtfilt(等效阶数翻倍、零相移)。
- 短段保护:len(x) <= 3*(max(len(a),len(b))-1) 时跳过该滤波器(与 filtfilt 最短长度要求一致)。
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def notch_vals(cfg) -> list[tuple[float, float]]:
    """生成 [(freq, Q), ...] 陷波表。对应 MATLAB emglib.notch_vals。"""
    pre = cfg.preprocess
    f0 = float(pre.notch_hz)
    bw = float(pre.notch_bandwidth_hz)
    if bw <= 0:
        bw = 8.0
    nh = int(pre.notch_harmonics)
    fs = float(cfg.sampling_rate)
    nyq = fs / 2.0
    hi = float(pre.bandpass[1])
    out = []
    for k in range(1, nh + 2):           # f0*(1..nh+1)
        f = f0 * k
        if f > 0 and f < min(nyq, hi):
            Q = max(1.0, f / bw)         # 绝对带宽恒为 bw Hz
            out.append((f, Q))
    return out


def notch_coeffs(f0: float, Q: float, fs: float):
    """二阶陷波系数。scipy.signal.iirnotch 与 MATLAB 自实现公式等价。"""
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return b, a


def _long_enough(n: int, b, a) -> bool:
    return n > 3 * (max(len(a), len(b)) - 1)


def apply_filters(x, cfg, do_notch: bool = True) -> np.ndarray:
    """notch(可选) + 带通,零相位 filtfilt。返回一维 float 数组。

    对应 MATLAB emglib.apply_filters(x, cfg, do_notch)。
    """
    y = np.asarray(x, dtype=float).ravel()
    fs = float(cfg.sampling_rate)
    n = y.size

    if do_notch:
        for f, Q in notch_vals(cfg):
            b, a = notch_coeffs(f, Q, fs)
            if _long_enough(n, b, a):
                y = signal.filtfilt(b, a, y)

    lo, hi = cfg.preprocess.bandpass
    b, a = signal.butter(5, [lo, hi], btype="band", fs=fs)
    if _long_enough(n, b, a):
        y = signal.filtfilt(b, a, y)
    return y
