"""RMS 包络 —— 复刻 MATLAB compute_envelope。

链路(= EMGFlow: notch→bandpass→rectify→rms_smooth):
  1. use_preprocess_chain 时先 apply_filters(notch+带通),否则用原始;
  2. 整流 abs;
  3. 平方 → 居中滑动平均(movmean) → 开方 = RMS 平滑;
  4. 非有限值置 0。
窗长 win = max(1, round(window_ms * fs / 1000)) 样本。

注:MATLAB movmean 与 scipy uniform_filter1d 在偶数窗时的居中方式相差一个样本,
对包络影响可忽略;如需严格一致可自定义卷积核。
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
from .filters import apply_filters


def compute_envelope(x, cfg, window_ms: float | None = None) -> np.ndarray:
    if window_ms is None:
        window_ms = cfg.segmentation.smooth_window_ms
    fs = float(cfg.sampling_rate)

    if cfg.segmentation.use_preprocess_chain:
        sig = apply_filters(x, cfg, True)
    else:
        sig = np.asarray(x, dtype=float).ravel()

    sig = np.abs(sig)
    win = max(1, int(round(window_ms * fs / 1000.0)))
    mean_square = uniform_filter1d(sig ** 2, size=win, mode="nearest")
    env = np.sqrt(np.maximum(mean_square, 0.0))
    env[~np.isfinite(env)] = 0.0
    return env
