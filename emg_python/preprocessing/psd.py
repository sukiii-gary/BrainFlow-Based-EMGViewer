"""Welch PSD —— 复刻 MATLAB compute_psd。

参数(贴近 EMGFlow emg_to_psd):
  N<16 → 返回空;
  nperseg = floor(N/2)(若<8 用 N);noverlap = floor(nperseg/2);nfft = 2*nperseg;
  window = hann(nperseg, sym=True)(与 MATLAB hann 默认对称窗一致);
  保留 f >= 2*fs/(N/2) 且 f <= fs/2;
  normalize=True 时 p/=max(p)。
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def compute_psd(x, fs: float, normalize: bool = False):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return np.array([]), np.array([])

    nperseg = n // 2
    if nperseg < 8:
        nperseg = n
    noverlap = nperseg // 2
    nfft = 2 * nperseg
    win = signal.windows.hann(nperseg, sym=True)

    f, p = signal.welch(x, fs=fs, window=win, nperseg=nperseg,
                        noverlap=noverlap, nfft=nfft, scaling="density")

    min_f = 2.0 * fs / (n / 2.0)
    keep = (f >= min_f) & (f <= fs / 2.0)
    f, p = f[keep], p[keep]

    if normalize and p.size and np.max(p) > 0:
        p = p / np.max(p)
    return f, p
