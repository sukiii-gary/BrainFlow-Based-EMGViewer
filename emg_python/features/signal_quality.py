"""Level-1 信号质量指标(复刻肱二头 q1lib/signal_quality)。

两路信号分工(不改处理逻辑):
  - 污染/伪迹类(P2P / MAR / PLI / PSD)用**原始**信号(去直流);
  - SNR / Active-Rest RMS / ActiveRMS / EMGBandPower 用**带通+陷波后干净 EMG**。
回答:这段 EMG 能不能用 / 噪声是否过大 / 电极接触是否稳定。
"""
from __future__ import annotations
import numpy as np
from preprocessing import compute_psd

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2.0 用 trapezoid


def band_power(f, p, lo, hi):
    """PSD 在 [lo,hi] 的梯形积分。"""
    if f.size == 0:
        return 0.0
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return 0.0
    return float(_trapz(p[m], f[m]))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def signal_quality(raw_act, filt_act, base_rms, fs, pli_hw=1.5):
    """raw_act=原始动作段(将去直流), filt_act=滤波后动作段, base_rms=该通道静息干净RMS。"""
    raw = np.asarray(raw_act, float)
    raw = raw - np.nanmean(raw)
    filt = np.asarray(filt_act, float)

    active_rms = rms(filt)
    p2p = float(np.nanmax(raw) - np.nanmin(raw))

    f_r, p_r = compute_psd(raw, fs)          # 原始 PSD → 伪迹/工频
    p_total = band_power(f_r, p_r, 0, fs / 2)
    mar = band_power(f_r, p_r, 0, 20) / (band_power(f_r, p_r, 20, 450) + 1e-12)
    pli = band_power(f_r, p_r, 50 - pli_hw, 50 + pli_hw) / (p_total + 1e-12)

    f_f, p_f = compute_psd(filt, fs)         # 干净 PSD → EMG 带功率
    emg_bp = band_power(f_f, p_f, 20, 450)

    snr_db = 20 * np.log10(active_rms / base_rms) if base_rms > 0 else np.nan
    ratio = active_rms / base_rms if base_rms > 0 else np.nan
    return {
        "BaselineRMS": base_rms, "ActiveRMS": active_rms,
        "SNR_dB": snr_db, "ActiveRestRatio": ratio,
        "P2P": p2p, "MAR": mar, "PLI": pli,
        "P_total": p_total, "EMGBandPower": emg_bp,
    }
